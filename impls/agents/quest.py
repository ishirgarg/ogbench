import glob
import pickle
from typing import Any, Optional, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


def _round_ste(z):
    return z + jax.lax.stop_gradient(jnp.round(z) - z)


def fsq_bound(z, levels, eps=1e-3):
    levels = levels.astype(z.dtype)
    half_l = (levels - 1.0) * (1.0 + eps) / 2.0
    offset = jnp.where(levels % 2 == 0, 0.5, 0.0)
    shift = jnp.arctanh(offset / half_l)
    return jnp.tanh(z + shift) * half_l - offset


def fsq_quantize(z, levels):
    quantized = _round_ste(fsq_bound(z, levels))
    half_width = (levels // 2).astype(z.dtype)
    return quantized / half_width


def _fsq_basis(levels):
    levels = jnp.asarray(levels)
    ones = jnp.ones((1,), dtype=jnp.int32)
    return jnp.concatenate([ones, jnp.cumprod(levels[:-1]).astype(jnp.int32)])


def fsq_codes_to_indices(codes, levels):
    half_width = (levels // 2)
    zhat = jnp.round(codes * half_width + half_width)
    basis = _fsq_basis(levels)
    return (zhat.astype(jnp.int32) * basis).sum(axis=-1)


def fsq_indices_to_codes(indices, levels):
    basis = _fsq_basis(levels)
    codes_non_centered = (indices[..., None] // basis) % levels
    half_width = (levels // 2).astype(jnp.float32)
    return (codes_non_centered.astype(jnp.float32) - half_width) / half_width


_FSQ_LEVELS_BY_POWER = {
    4: (5, 3),
    6: (8, 8),
    8: (8, 6, 5),
    9: (8, 8, 8),
    10: (8, 5, 5, 5),
    11: (8, 8, 6, 5),
    12: (7, 5, 5, 5, 5),
}

_FSQ_LEVELS_EXTRA = {
    15: (5, 3),
    50: (10, 5),
}


def get_fsq_level(codebook_size):
    if codebook_size in _FSQ_LEVELS_EXTRA:
        return _FSQ_LEVELS_EXTRA[codebook_size]
    power = int(np.log2(codebook_size))
    if power not in _FSQ_LEVELS_BY_POWER:
        raise ValueError(
            f'No FSQ level factorization registered for codebook_size={codebook_size}.'
        )
    return _FSQ_LEVELS_BY_POWER[power]


def top_k_sampling(logits, k, temperature, key):
    temp = jnp.maximum(temperature, 1e-8)
    scaled = logits / temp
    top_vals, top_idx = jax.lax.top_k(scaled, k)
    choice = jax.random.categorical(key, top_vals, axis=-1)
    sampled = jnp.take_along_axis(top_idx, choice[:, None], axis=-1)[:, 0]
    greedy = jnp.argmax(logits, axis=-1)
    return jnp.where(temperature == 0, greedy, sampled).astype(jnp.int32)


class MultiHeadAttention(nn.Module):
    dim: int
    num_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, q_in, kv_in, mask=None, deterministic=True):
        B, Tq, _ = q_in.shape
        Tk = kv_in.shape[1]
        h, hd = self.num_heads, self.dim // self.num_heads

        q = nn.Dense(self.dim, name='q_proj')(q_in).reshape(B, Tq, h, hd)
        k = nn.Dense(self.dim, name='k_proj')(kv_in).reshape(B, Tk, h, hd)
        v = nn.Dense(self.dim, name='v_proj')(kv_in).reshape(B, Tk, h, hd)

        attn = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(hd)
        if mask is not None:
            attn = jnp.where(mask, attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)
        out = jnp.einsum('bhqk,bkhd->bqhd', attn, v).reshape(B, Tq, self.dim)
        return nn.Dense(self.dim, name='out_proj')(out)


class TransformerBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: int = 4
    cross: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, context=None, self_mask=None, cross_mask=None, deterministic=True):
        h = nn.LayerNorm(epsilon=1e-5)(x)
        attn = MultiHeadAttention(self.dim, self.num_heads, self.dropout_rate, name='self_attn')(
            h, h, self_mask, deterministic=deterministic
        )
        x = x + nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)
        if self.cross:
            xq = nn.LayerNorm(epsilon=1e-5)(x)
            cattn = MultiHeadAttention(self.dim, self.num_heads, self.dropout_rate, name='cross_attn')(
                xq, context, cross_mask, deterministic=deterministic
            )
            x = x + nn.Dropout(self.dropout_rate)(cattn, deterministic=deterministic)
        y = nn.LayerNorm(epsilon=1e-5)(x)
        y = nn.Dense(self.dim * self.mlp_ratio)(y)
        y = nn.gelu(y, approximate=False)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(self.dim)(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        return x + y


def _causal_mask(length):
    return jnp.tril(jnp.ones((length, length), dtype=bool))


def sinusoidal_embedding(length, dim):
    pos = jnp.arange(length)[:, None]
    idx = jnp.arange(dim)[None, :]
    angle_rates = 1.0 / jnp.power(10000.0, (2 * (idx // 2)) / dim)
    angles = pos * angle_rates
    return jnp.where(idx % 2 == 0, jnp.sin(angles), jnp.cos(angles))


class CausalConv1d(nn.Module):
    features: int
    kernel_size: int
    stride: int

    @nn.compact
    def __call__(self, x):
        pad = self.kernel_size - 1
        x = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))
        return nn.Conv(
            self.features, (self.kernel_size,), strides=(self.stride,), padding='VALID'
        )(x)


class ActionEncoder(nn.Module):
    dim: int
    fsq_dim: int
    conv_kernels: Sequence[int]
    conv_strides: Sequence[int]
    num_layers: int
    num_heads: int
    attn_pdrop: float = 0.0

    @nn.compact
    def __call__(self, action_chunk, train=False):
        deterministic = not train
        x = nn.Dense(self.dim)(action_chunk)
        for k, s in zip(self.conv_kernels, self.conv_strides):
            x = CausalConv1d(self.dim, k, s)(x)
            x = nn.GroupNorm(num_groups=8, epsilon=1e-5)(x)
            x = x * jnp.tanh(jax.nn.softplus(x))
        n = x.shape[1]
        x = x + sinusoidal_embedding(n, self.dim)[None]
        mask = _causal_mask(n)[None, None]
        for _ in range(self.num_layers):
            x = TransformerBlock(self.dim, self.num_heads, dropout_rate=self.attn_pdrop)(
                x, self_mask=mask, deterministic=deterministic
            )
        return nn.Dense(self.fsq_dim)(x)


class ActionDecoder(nn.Module):
    dim: int
    horizon: int
    num_tokens: int
    action_dim: int
    num_layers: int
    num_heads: int
    attn_pdrop: float = 0.0

    @nn.compact
    def __call__(self, codes, train=False):
        deterministic = not train
        B = codes.shape[0]

        skill = nn.Dense(self.dim)(codes)

        queries = sinusoidal_embedding(self.horizon, self.dim)
        x = jnp.broadcast_to(queries[None], (B, self.horizon, self.dim))

        self_mask = _causal_mask(self.horizon)[None, None]
        cross_mask = None

        for _ in range(self.num_layers):
            x = TransformerBlock(self.dim, self.num_heads, cross=True, dropout_rate=self.attn_pdrop)(
                x, context=skill, self_mask=self_mask, cross_mask=cross_mask, deterministic=deterministic
            )
        return nn.Dense(self.action_dim)(x)


class SkillPrior(nn.Module):
    vocab_size: int
    num_tokens: int
    dim: int
    num_layers: int
    num_heads: int
    lowdim_embed_dim: int
    attn_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    goal_conditioned: bool = True
    gc_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, tokens, observations, goals, train=False):
        deterministic = not train

        tok_emb = nn.Embed(self.vocab_size + 1, self.dim, name='tok_emb')
        x = tok_emb(tokens)
        x = x + sinusoidal_embedding(self.num_tokens, self.dim)[None]

        obs_feat = self.gc_encoder(observations) if self.gc_encoder is not None else observations
        obs_tok = nn.Dense(self.dim, name='obs_proj')(
            nn.Dense(self.lowdim_embed_dim, name='obs_lowdim')(obs_feat)
        )
        context = [obs_tok]
        if self.goal_conditioned and goals is not None:
            goal_feat = self.gc_encoder(goals) if self.gc_encoder is not None else goals
            goal_tok = nn.Dense(self.dim, name='goal_proj')(goal_feat)
            context = [goal_tok, obs_tok]
        context = jnp.stack(context, axis=1)

        x = jnp.concatenate([context, x], axis=1)
        x = nn.Dropout(self.embd_pdrop)(x, deterministic=deterministic)

        length = x.shape[1]
        mask = _causal_mask(length)[None, None]
        for _ in range(self.num_layers):
            x = TransformerBlock(self.dim, self.num_heads, dropout_rate=self.attn_pdrop)(
                x, self_mask=mask, deterministic=deterministic
            )
        x = x[:, context.shape[1]:, :]
        x = nn.LayerNorm(epsilon=1e-5)(x)
        return nn.Dense(self.vocab_size, name='head')(x)


class QueSTAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @property
    def _levels(self):
        return jnp.asarray(self.config['fsq_levels'])

    def _get_chunk(self, batch):
        return batch['actions_seq']

    def reconstruction_loss(self, batch, grad_params, train=False, rng=None):
        chunk = self._get_chunk(batch)
        enc_kwargs = {}
        dec_kwargs = {}
        if train:
            ke, kd = jax.random.split(rng)
            enc_kwargs = {'rngs': {'dropout': ke}}
            dec_kwargs = {'rngs': {'dropout': kd}}
        z = self.network.select('encoder')(chunk, params=grad_params, train=train, **enc_kwargs)
        codes = fsq_quantize(z, self._levels)
        indices = fsq_codes_to_indices(jax.lax.stop_gradient(codes), self._levels)
        recon = self.network.select('decoder')(codes, params=grad_params, train=train, **dec_kwargs)
        l1 = jnp.abs(recon - chunk).mean()
        usage = jnp.unique(indices, size=indices.size, fill_value=-1)
        usage = (usage >= 0).sum() / self.config['vocab_size']
        return l1, {
            'recon_l1': l1,
            'code_usage': usage,
            'z_abs_mean': jnp.abs(z).mean(),
        }

    def prior_loss(self, batch, grad_params, train=False, rng=None):
        chunk = self._get_chunk(batch)
        z = self.network.select('encoder')(chunk, params=None, train=False)
        codes = fsq_quantize(z, self._levels)
        indices = fsq_codes_to_indices(jax.lax.stop_gradient(codes), self._levels)
        goals = batch.get('actor_goals') if self.config['goal_conditioned'] else None

        B = indices.shape[0]
        start = jnp.full((B, 1), self.config['vocab_size'], dtype=indices.dtype)
        x_in = jnp.concatenate([start, indices[:, :-1]], axis=1)

        prior_kwargs = {}
        if train:
            rng, r_drop = jax.random.split(rng)
            prior_kwargs = {'rngs': {'dropout': r_drop}}
        logits = self.network.select('prior')(
            x_in, batch['observations'], goals, params=grad_params, train=train, **prior_kwargs
        )
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, indices).mean()
        acc = (logits.argmax(axis=-1) == indices).mean()

        l1 = jnp.array(0.0)
        if self.config['l1_loss_scale'] > 0:
            dec_kwargs = {}
            if train:
                rng, r_sample, r_dec = jax.random.split(rng, 3)
                sampled = jax.random.categorical(r_sample, logits, axis=-1)
                dec_kwargs = {'rngs': {'dropout': r_dec}}
            else:
                sampled = jnp.argmax(logits, axis=-1)
            sampled = jax.lax.stop_gradient(sampled)
            recon = self.network.select('decoder')(
                fsq_indices_to_codes(sampled, self._levels), params=grad_params, train=train, **dec_kwargs
            )
            l1 = jnp.abs(recon - chunk).mean()

        total = ce + self.config['l1_loss_scale'] * l1
        return total, {
            'prior_ce': ce,
            'prior_token_acc': acc,
            'prior_perplexity': jnp.exp(ce),
            'prior_l1': l1,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        train = rng is not None
        if train:
            r_recon, r_prior = jax.random.split(rng)
        else:
            r_recon = r_prior = None
        recon_loss, recon_info = self.reconstruction_loss(batch, grad_params, train, r_recon)
        info.update({f'ae/{k}': v for k, v in recon_info.items()})

        prior_loss, prior_info = self.prior_loss(batch, grad_params, train, r_prior)
        info.update({f'prior/{k}': v for k, v in prior_info.items()})

        step = jnp.asarray(self.network.step, dtype=jnp.float32)
        stage = self.config['stage']
        if stage == 'ae':
            w_recon = jnp.array(1.0)
            w_prior = jnp.array(0.0)
        elif stage == 'prior':
            w_recon = jnp.array(0.0)
            w_prior = jnp.array(self.config['prior_weight'])
        elif self.config['joint_training']:
            w_recon = jnp.array(1.0)
            w_prior = jnp.array(self.config['prior_weight'])
        else:
            in_stage1 = (step - 1.0) < self.config['stage1_steps']
            w_recon = jnp.where(in_stage1, 1.0, 0.0)
            w_prior = jnp.where(in_stage1, 0.0, self.config['prior_weight'])
        info['ae/weight'] = w_recon
        info['prior/weight'] = w_prior

        total = w_recon * recon_loss + w_prior * prior_loss
        info['total_loss'] = total
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )
        return self.replace(network=new_network, rng=new_rng), info

    def _sample_tokens(self, observations, goals, rng, temperature):
        B = observations.shape[0]
        n = self.config['num_tokens']
        k = self.config['top_k']
        start = self.config['vocab_size']

        def body(i, carry):
            tokens, out, key = carry
            key, sub = jax.random.split(key)
            logits = self.network.select('prior')(tokens, observations, goals)
            s = top_k_sampling(logits[:, i, :], k, temperature, sub)
            out = out.at[:, i].set(s)
            tokens = jnp.where(
                i + 1 < n, tokens.at[:, jnp.clip(i + 1, 0, n - 1)].set(s), tokens
            )
            return tokens, out, key

        tokens0 = jnp.zeros((B, n), dtype=jnp.int32).at[:, 0].set(start)
        out0 = jnp.zeros((B, n), dtype=jnp.int32)
        _, out, _ = jax.lax.fori_loop(0, n, body, (tokens0, out0, rng))
        return out

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        if seed is None:
            seed = self.rng

        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        if single_obs:
            observations = observations[None, ...]
            if goals is not None:
                goals = goals[None, ...]

        cond_goals = goals if self.config['goal_conditioned'] else None
        tokens = self._sample_tokens(observations, cond_goals, seed, temperature)
        codes = fsq_indices_to_codes(tokens, self._levels)
        chunk = self.network.select('decoder')(codes)
        actions = jnp.clip(chunk[:, 0, :], -1, 1)

        if single_obs:
            actions = actions[0]
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        assert not config['discrete'], 'QueST targets continuous control (Sec. 4).'
        action_dim = ex_actions.shape[-1]
        T = config['horizon_length']
        F = config['downsample_factor']
        assert T % F == 0, 'horizon_length must be divisible by downsample_factor.'
        assert int(np.prod(config['conv_strides'])) == F, (
            'product(conv_strides) must equal downsample_factor so the encoder output length == num_tokens.'
        )
        assert config['sequence_length'] == T, (
            'sequence_length must equal horizon_length so actions_seq spans the chunk.'
        )
        stage = config['stage']
        assert stage in ('both', 'ae', 'prior'), "stage must be 'both', 'ae', or 'prior'."
        if stage == 'both':
            assert config['joint_training'] or config['total_steps'] > config['stage1_steps'], (
                'total_steps must exceed stage1_steps (and equal --train_steps) so the prior stage runs.'
            )
        num_tokens = T // F
        levels = tuple(config['fsq_levels'])
        if config.get('codebook_size') is not None:
            levels = get_fsq_level(config['codebook_size'])
        vocab_size = int(np.prod(levels))
        config = dict(config)
        config['fsq_levels'] = tuple(int(l) for l in levels)
        config['num_tokens'] = num_tokens
        config['vocab_size'] = vocab_size

        prior_encoder = None
        if config.get('encoder') is not None:
            enc = encoder_modules[config['encoder']]
            prior_encoder = GCEncoder(state_encoder=enc())

        encoder_def = ActionEncoder(
            dim=config['ae_dim'],
            fsq_dim=len(levels),
            conv_kernels=tuple(config['conv_kernels']),
            conv_strides=tuple(config['conv_strides']),
            num_layers=config['enc_layers'],
            num_heads=config['enc_heads'],
            attn_pdrop=config['attn_pdrop'],
        )
        decoder_def = ActionDecoder(
            dim=config['ae_dim'],
            horizon=T,
            num_tokens=num_tokens,
            action_dim=action_dim,
            num_layers=config['dec_layers'],
            num_heads=config['dec_heads'],
            attn_pdrop=config['attn_pdrop'],
        )
        prior_def = SkillPrior(
            vocab_size=vocab_size,
            num_tokens=num_tokens,
            dim=config['prior_dim'],
            num_layers=config['prior_layers'],
            num_heads=config['prior_heads'],
            lowdim_embed_dim=config['lowdim_embed_dim'],
            attn_pdrop=config['attn_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            goal_conditioned=config['goal_conditioned'],
            gc_encoder=prior_encoder,
        )

        ex_chunk = jnp.zeros((ex_observations.shape[0], T, action_dim))
        ex_codes = jnp.zeros((ex_observations.shape[0], num_tokens, len(levels)))
        ex_tokens = jnp.zeros((ex_observations.shape[0], num_tokens), dtype=jnp.int32)
        ex_goals = ex_observations if config['goal_conditioned'] else None

        network_def = ModuleDict(dict(
            encoder=encoder_def, decoder=decoder_def, prior=prior_def,
        ))
        network_params = network_def.init(
            init_rng,
            encoder=(ex_chunk,),
            decoder=(ex_codes,),
            prior=(ex_tokens, ex_observations, ex_goals),
        )['params']


        if stage == 'prior' and config.get('restore_ae_path') is not None:
            candidates = glob.glob(config['restore_ae_path'])
            assert len(candidates) == 1, f'restore_ae_path matched {len(candidates)} dirs: {candidates}'
            ckpt = candidates[0] + f"/params_{config['restore_ae_epoch']}.pkl"
            with open(ckpt, 'rb') as f:
                loaded = pickle.load(f)['agent']['network']['params']

            def _leaf_shapes(tree):
                return {'/'.join(str(getattr(k, 'key', k)) for k in p): tuple(v.shape)
                        for p, v in jax.tree_util.tree_leaves_with_path(tree)}

            network_params = flax.core.unfreeze(network_params)
            for mod in ('modules_encoder', 'modules_decoder'):
                restored = flax.serialization.from_state_dict(network_params[mod], loaded[mod])
                assert _leaf_shapes(network_params[mod]) == _leaf_shapes(restored), (
                    f'AE checkpoint architecture mismatch in {mod}: the stage="ae" run must use '
                    f'the same encoder/decoder config as this stage="prior" run.'
                )
                network_params[mod] = restored
        elif stage == 'prior':
            raise ValueError("stage='prior' requires restore_ae_path (the stage='ae' checkpoint).")

        lr = config['lr']
        alpha = (config['lr_eta_min'] / lr) if lr > 0 else 0.0
        finetune = config['l1_loss_scale'] > 0

        def _cos(n):
            return optax.cosine_decay_schedule(lr, max(int(n), 1), alpha)
        _zero = optax.constant_schedule(0.0)

        if stage == 'ae':
            enc_sched = dec_sched = _cos(config['total_steps'])
            prior_sched = _zero
        elif stage == 'prior':
            enc_sched = _zero
            prior_sched = _cos(config['total_steps'])
            dec_sched = _cos(config['total_steps']) if finetune else _zero
        elif config['joint_training']:
            enc_sched = dec_sched = prior_sched = _cos(config['total_steps'])
        else:
            s1 = max(int(config['stage1_steps']), 1)
            s2 = max(int(config['total_steps']) - s1, 1)
            enc_sched = optax.join_schedules([_cos(s1), _zero], boundaries=[s1])
            prior_sched = optax.join_schedules([_zero, _cos(s2)], boundaries=[s1])
            dec_sched = optax.join_schedules(
                [_cos(s1), _cos(s2) if finetune else _zero], boundaries=[s1]
            )

        def _param_labels(params):
            def label(path, leaf):
                name = '/'.join(str(getattr(k, 'key', k)) for k in path).lower()
                if 'prior' in name:
                    group = 'prior'
                elif 'decoder' in name:
                    group = 'decoder'
                else:
                    group = 'encoder'
                is_norm_or_embed = ('norm' in name) or ('embed' in name)
                decay = (leaf.ndim >= 2) and (not is_norm_or_embed)
                return f'{group}_{"decay" if decay else "nodecay"}'
            return jax.tree_util.tree_map_with_path(label, params)

        wd = config['weight_decay']

        def _adamw(schedule, weight_decay):
            return optax.adamw(learning_rate=schedule, b1=0.9, b2=0.999, weight_decay=weight_decay)

        tx = optax.chain(
            optax.clip_by_global_norm(config['grad_clip']),
            optax.multi_transform(
                {
                    'encoder_decay': _adamw(enc_sched, wd),
                    'encoder_nodecay': _adamw(enc_sched, 0.0),
                    'decoder_decay': _adamw(dec_sched, wd),
                    'decoder_nodecay': _adamw(dec_sched, 0.0),
                    'prior_decay': _adamw(prior_sched, wd),
                    'prior_nodecay': _adamw(prior_sched, 0.0),
                },
                _param_labels(network_params),
            ),
        )
        network = TrainState.create(network_def, network_params, tx=tx)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='quest',
        lr=1e-4,
        batch_size=128,
        weight_decay=1e-4,
        grad_clip=100.0,
        lr_eta_min=1e-5,
        total_steps=1000000,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
        horizon_length=32,
        downsample_factor=4,
        fsq_levels=(8, 5, 5, 5),
        codebook_size=ml_collections.config_dict.placeholder(int),
        ae_dim=256,
        conv_kernels=(5, 3, 3),
        conv_strides=(2, 2, 1),
        enc_layers=2,
        enc_heads=4,
        dec_layers=4,
        dec_heads=4,
        prior_dim=384,
        prior_layers=6,
        prior_heads=6,
        lowdim_embed_dim=128,
        top_k=5,
        prior_weight=1.0,
        l1_loss_scale=0.0,
        stage='both',
        stage1_steps=500000,
        joint_training=False,
        restore_ae_path=ml_collections.config_dict.placeholder(str),
        restore_ae_epoch=ml_collections.config_dict.placeholder(int),
        goal_conditioned=True,
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='SequenceDataset',
        sequence_length=32,
        value_p_curgoal=0.0,
        value_p_trajgoal=1.0,
        value_p_randomgoal=0.0,
        value_geom_sample=False,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=True,
        p_aug=0.0,
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))
