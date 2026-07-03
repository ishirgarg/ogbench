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
    half_l = (levels - 1.0) * (1.0 - eps) / 2.0
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


# FSQ per-channel level factorizations by effective codebook size (== product).
# The official get_fsq_level (skill_vae.py) values plus 15/50 for DDS-style
# codebook-size sweeps (15 = [5,3] matches the official smallest; 50 = [10,5],
# the exact product-50 factorization with all levels >= 3 — FSQ's tanh/arctanh
# bound is undefined for level 2, so [5,5,2] is not usable).
_FSQ_LEVELS = {
    15: (5, 3),
    50: (10, 5),
    64: (8, 8),
    240: (8, 6, 5),
    512: (8, 8, 8),
    1000: (8, 5, 5, 5),
    1920: (8, 8, 6, 5),
    4375: (7, 5, 5, 5, 5),
}


def get_fsq_level(codebook_size):
    if codebook_size not in _FSQ_LEVELS:
        raise ValueError(
            f'No FSQ level factorization registered for codebook_size={codebook_size}. '
            f'Known sizes: {sorted(_FSQ_LEVELS)}. Set fsq_levels directly for others.'
        )
    return _FSQ_LEVELS[codebook_size]


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
        h = nn.LayerNorm()(x)
        attn = MultiHeadAttention(self.dim, self.num_heads, self.dropout_rate, name='self_attn')(
            h, h, self_mask, deterministic=deterministic
        )
        x = x + nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)
        if self.cross:
            xq = nn.LayerNorm()(x)
            cattn = MultiHeadAttention(self.dim, self.num_heads, self.dropout_rate, name='cross_attn')(
                xq, context, cross_mask, deterministic=deterministic
            )
            x = x + nn.Dropout(self.dropout_rate)(cattn, deterministic=deterministic)
        y = nn.LayerNorm()(x)
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
            x = nn.GroupNorm(num_groups=8)(x)
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
    attn_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    gc_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations, goals, tokens, train=False):
        deterministic = not train
        B = tokens.shape[0]

        if self.gc_encoder is not None:
            cond_in = self.gc_encoder(observations, goals)
        elif goals is not None:
            cond_in = jnp.concatenate([observations, goals], axis=-1)
        else:
            cond_in = observations
        cond = MLP((self.dim, self.dim), activate_final=True)(cond_in)

        tok_embed = nn.Embed(self.vocab_size, self.dim, name='tok_embed')
        prev = tok_embed(tokens[:, : self.num_tokens - 1])
        x = jnp.concatenate([cond[:, None, :], prev], axis=1)
        x = x + sinusoidal_embedding(self.num_tokens, self.dim)[None]
        x = nn.Dropout(self.embd_pdrop)(x, deterministic=deterministic)

        mask = _causal_mask(self.num_tokens)[None, None]
        for _ in range(self.num_layers):
            x = TransformerBlock(self.dim, self.num_heads, dropout_rate=self.attn_pdrop)(
                x, self_mask=mask, deterministic=deterministic
            )
        x = nn.LayerNorm()(x)
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
        prior_kwargs = {}
        if train:
            prior_kwargs = {'rngs': {'dropout': rng}}
        logits = self.network.select('prior')(
            batch['observations'], goals, indices, params=grad_params, train=train, **prior_kwargs
        )
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, indices)
        ce = ce.mean()
        acc = (logits.argmax(axis=-1) == indices).mean()
        return ce, {
            'prior_ce': ce,
            'prior_token_acc': acc,
            'prior_perplexity': jnp.exp(ce),
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
        top_k = self.config['top_k']

        def sample_one(logits, key):
            kth = jax.lax.top_k(logits, top_k)[0][:, -1:]
            logits = jnp.where(logits < kth, jnp.finfo(logits.dtype).min, logits)
            greedy = logits.argmax(axis=-1)
            scaled = logits / jnp.maximum(temperature, 1e-8)
            sampled = jax.random.categorical(key, scaled, axis=-1)
            return jnp.where(temperature == 0, greedy, sampled)

        def body(i, carry):
            tokens, key = carry
            key, sub = jax.random.split(key)
            logits = self.network.select('prior')(observations, goals, tokens)
            next_tok = sample_one(logits[:, i, :], sub)
            tokens = tokens.at[:, i].set(next_tok)
            return tokens, key

        tokens0 = jnp.zeros((B, n), dtype=jnp.int32)
        tokens, _ = jax.lax.fori_loop(0, n, body, (tokens0, rng))
        return tokens

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
            prior_encoder = GCEncoder(concat_encoder=enc())

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
            attn_pdrop=config['attn_pdrop'],
            embd_pdrop=config['embd_pdrop'],
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
            prior=(ex_observations, ex_goals, ex_tokens),
        )['params']

        # Two-run stage split: for stage='prior', load the trained autoencoder
        # (encoder+decoder) params from a stage='ae' run's checkpoint and keep a
        # FRESH optimizer for the prior (fresh Adam bias-correction, fresh cosine)
        # — bit-exactly reproducing the official's separate stage-0/stage-1 jobs.
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

        def _cos(n):
            return optax.cosine_decay_schedule(lr, max(int(n), 1), alpha)
        _zero = optax.constant_schedule(0.0)

        if stage == 'ae':
            ae_sched, prior_sched = _cos(config['total_steps']), _zero
        elif stage == 'prior':
            ae_sched, prior_sched = _zero, _cos(config['total_steps'])
        elif config['joint_training']:
            ae_sched = prior_sched = _cos(config['total_steps'])
        else:
            s1 = max(int(config['stage1_steps']), 1)
            s2 = max(int(config['total_steps']) - s1, 1)
            ae_sched = optax.join_schedules([_cos(s1), _zero], boundaries=[s1])
            prior_sched = optax.join_schedules([_zero, _cos(s2)], boundaries=[s1])

        def _param_labels(params):
            def label(path, leaf):
                name = '/'.join(str(getattr(k, 'key', k)) for k in path).lower()
                group = 'prior' if 'prior' in name else 'ae'
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
                    'ae_decay': _adamw(ae_sched, wd),
                    'ae_nodecay': _adamw(ae_sched, 0.0),
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
        top_k=5,
        prior_weight=1.0,
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
