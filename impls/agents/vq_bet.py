
import glob
import pickle
from typing import Any, NamedTuple, Optional, Sequence

import distrax
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


def focal_loss(logits, targets, gamma):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_pt = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    pt = jnp.exp(log_pt)
    return jnp.mean(-((1.0 - pt) ** gamma) * log_pt)


class EncoderMLP(nn.Module):

    output_dim: int
    hidden_dim: int = 128
    layer_num: int = 1

    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.orthogonal()
        bias_init = nn.initializers.zeros
        x = nn.Dense(self.hidden_dim, kernel_init=kernel_init, bias_init=bias_init)(x)
        x = nn.relu(x)
        for _ in range(self.layer_num):
            x = nn.Dense(self.hidden_dim, kernel_init=kernel_init, bias_init=bias_init)(x)
            x = nn.relu(x)
        return nn.Dense(self.output_dim, kernel_init=kernel_init, bias_init=bias_init)(x)


def _codebook_uniform_init(codebook_size, latent_dim):
    bound = np.sqrt(6.0 / (codebook_size * latent_dim))

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


class VqVae(nn.Module):

    n_groups: int
    codebook_size: int
    latent_dim: int
    action_chunk_dim: int
    hidden_dim: int = 128
    ema_eps: float = 1e-5

    def setup(self):
        self.encoder = EncoderMLP(self.latent_dim, self.hidden_dim)
        self.decoder = EncoderMLP(self.action_chunk_dim, self.hidden_dim)

        cb_init = _codebook_uniform_init(self.codebook_size, self.latent_dim)
        self.cb_embed = self.param('cb_embed', cb_init, (self.n_groups, self.codebook_size, self.latent_dim))

        self.cb_embed_avg = self.param('cb_embed_avg', cb_init, (self.n_groups, self.codebook_size, self.latent_dim))
        self.cb_cluster_size = self.param(
            'cb_cluster_size', nn.initializers.zeros, (self.n_groups, self.codebook_size)
        )


    def encode(self, chunk_flat):
        return self.encoder(chunk_flat)

    def decode(self, latent):
        return self.decoder(latent)

    def _codebook(self):
        return jax.lax.stop_gradient(self.cb_embed)

    def _residual_quantize(self, z, straight_through):
        cb = self._codebook()
        residual = z
        quantized_out = jnp.zeros_like(z)
        indices, residual_inputs, onehots, commit_terms = [], [], [], []
        for i in range(self.n_groups):
            cb_i = cb[i]
            residual_inputs.append(residual)
            dist = jnp.sum((residual[:, None, :] - cb_i[None, :, :]) ** 2, axis=-1)
            idx = jnp.argmin(dist, axis=-1)
            onehot = jax.nn.one_hot(idx, self.codebook_size)
            q = cb_i[idx]


            commit_terms.append(jnp.mean((jax.lax.stop_gradient(q) - residual) ** 2))
            if straight_through:
                q_st = residual + jax.lax.stop_gradient(q - residual)
            else:
                q_st = q
            quantized_out = quantized_out + q_st
            residual = residual - jax.lax.stop_gradient(q)
            indices.append(idx)
            onehots.append(onehot)
        indices = jnp.stack(indices, axis=-1)
        residual_inputs = jnp.stack(residual_inputs, axis=1)
        onehots = jnp.stack(onehots, axis=1)
        commit_loss = jnp.sum(jnp.stack(commit_terms))
        return quantized_out, indices, commit_loss, residual_inputs, onehots

    def get_indices(self, chunk_flat):
        z = self.encode(chunk_flat)
        _, indices, _, _, _ = self._residual_quantize(z, straight_through=False)
        return indices

    def decode_indices(self, indices):
        cb = self._codebook()
        emb = jnp.zeros((indices.shape[0], self.latent_dim))
        for i in range(self.n_groups):
            emb = emb + cb[i][indices[:, i]]
        return self.decoder(emb)

    def __call__(self, chunk_flat):
        z = self.encode(chunk_flat)
        quantized_st, indices, commit_loss, residual_inputs, onehots = self._residual_quantize(
            z, straight_through=True
        )
        recon = self.decode(quantized_st)
        return recon, indices, commit_loss, residual_inputs, onehots


def _gpt_dense(features, n_layer, is_proj=False, use_bias=True, name=None):
    std = 0.02 / np.sqrt(2 * n_layer) if is_proj else 0.02
    return nn.Dense(
        features,
        use_bias=use_bias,
        kernel_init=nn.initializers.normal(stddev=std),
        bias_init=nn.initializers.zeros,
        name=name,
    )


class CausalSelfAttention(nn.Module):

    n_embd: int
    n_head: int
    n_layer: int
    dropout: float

    @nn.compact
    def __call__(self, x, deterministic=True):
        B, T, C = x.shape
        hs = C // self.n_head
        qkv = _gpt_dense(3 * self.n_embd, self.n_layer, name='c_attn')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def heads(z):
            return z.reshape(B, T, self.n_head, hs).transpose(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(hs))
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        att = jnp.where(mask[None, None], att, -jnp.inf)
        att = jax.nn.softmax(att, axis=-1)
        att = nn.Dropout(self.dropout)(att, deterministic=deterministic)
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = _gpt_dense(self.n_embd, self.n_layer, is_proj=True, name='c_proj')(y)
        return nn.Dropout(self.dropout)(y, deterministic=deterministic)


class GPTBlock(nn.Module):

    n_embd: int
    n_head: int
    n_layer: int
    dropout: float

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = x + CausalSelfAttention(self.n_embd, self.n_head, self.n_layer, self.dropout, name='attn')(
            nn.LayerNorm(epsilon=1e-5, name='ln_1')(x), deterministic=deterministic
        )
        h = nn.LayerNorm(epsilon=1e-5, name='ln_2')(x)
        h = _gpt_dense(4 * self.n_embd, self.n_layer, name='c_fc')(h)
        h = nn.gelu(h, approximate=True)
        h = _gpt_dense(self.n_embd, self.n_layer, is_proj=True, name='c_proj')(h)
        h = nn.Dropout(self.dropout)(h, deterministic=deterministic)
        return x + h


class GPT(nn.Module):

    input_dim: int
    output_dim: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float

    @nn.compact
    def __call__(self, x, deterministic=True):
        B, T, _ = x.shape
        tok = _gpt_dense(self.n_embd, self.n_layer, name='wte')(x)
        wpe = nn.Embed(
            self.block_size, self.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02), name='wpe',
        )
        pos = wpe(jnp.arange(T))[None]
        h = nn.Dropout(self.dropout)(tok + pos, deterministic=deterministic)
        for i in range(self.n_layer):
            h = GPTBlock(self.n_embd, self.n_head, self.n_layer, self.dropout, name=f'block_{i}')(
                h, deterministic=deterministic
            )
        h = nn.LayerNorm(epsilon=1e-5, name='ln_f')(h)
        return _gpt_dense(self.output_dim, self.n_layer, use_bias=False, name='lm_head')(h)


class BehaviorTransformer(nn.Module):

    obs_dim: int
    n_groups: int
    codebook_size: int
    action_chunk_dim: int
    gpt: GPT
    goal_conditioned: bool = True
    gc_encoder: Optional[nn.Module] = None

    def setup(self):
        C, G, WA = self.codebook_size, self.n_groups, self.action_chunk_dim
        self.bin_head = MLP((1024, 1024, G * C), activations=nn.relu, activate_final=False)
        self.offset_head = MLP((1024, 1024, G * C * WA), activations=nn.relu, activate_final=False)

    def _tokens(self, observations, goals):

        obs = self.gc_encoder(observations) if self.gc_encoder is not None else observations
        if self.goal_conditioned and goals is not None:
            g = self.gc_encoder(goals) if self.gc_encoder is not None else goals
            return jnp.stack([g, obs], axis=1), True
        return obs[:, None, :], False

    def logits_and_offsets(self, observations, goals, deterministic=True):
        x, has_goal = self._tokens(observations, goals)
        out = self.gpt(x, deterministic=deterministic)
        feat = out[:, 1, :] if has_goal else out[:, 0, :]
        B = feat.shape[0]
        C, G, WA = self.codebook_size, self.n_groups, self.action_chunk_dim
        logits = self.bin_head(feat).reshape(B, G, C)
        offsets = self.offset_head(feat).reshape(B, G, C, WA)
        return logits, offsets

    def __call__(self, observations, goals, deterministic=True):
        return self.logits_and_offsets(observations, goals, deterministic=deterministic)


class _GatedState(NamedTuple):
    count: jnp.ndarray
    inner: Any


def _gated(inner, active_fn):

    def init_fn(params):
        return _GatedState(count=jnp.asarray(1, jnp.int32), inner=inner.init(params))

    def update_fn(updates, state, params=None):
        new_updates, new_inner = inner.update(updates, state.inner, params)
        active = active_fn(state.count)
        updates_out = jax.tree_util.tree_map(lambda u: jnp.where(active, u, jnp.zeros_like(u)), new_updates)
        inner_out = jax.tree_util.tree_map(lambda n, o: jnp.where(active, n, o), new_inner, state.inner)
        return updates_out, _GatedState(count=state.count + 1, inner=inner_out)

    return optax.GradientTransformation(init_fn, update_fn)


def _adam_l2(lr, wd, b1=0.9, b2=0.999):
    return optax.chain(optax.add_decayed_weights(wd), optax.scale_by_adam(b1=b1, b2=b2), optax.scale(-lr))


def _param_label(path, leaf):
    name = '/'.join(str(getattr(k, 'key', k)) for k in path).lower()
    if 'cb_embed' in name or 'cb_cluster_size' in name:
        return 'frozen'
    if 'vqvae' in name:
        return 'vqvae'
    if 'offset_head' in name:
        return 'offset'
    if 'bin_head' in name:
        return 'bin'

    is_norm_or_embed = ('norm' in name) or ('embed' in name) or ('wpe' in name)
    decay = (leaf.ndim >= 2) and not is_norm_or_embed
    return f'gpt_{"decay" if decay else "nodecay"}'


def make_vqbet_optimizer(config, params):
    stage = config['stage']
    lr, wd = config['lr'], config['weight_decay']
    code_wd = config['code_head_weight_decay']
    b1, b2 = config['gpt_beta1'], config['gpt_beta2']
    vqvae_lr, vqvae_wd = config['vqvae_lr'], config['vqvae_weight_decay']
    zero = optax.set_to_zero()

    def adamw(weight_decay):
        return optax.adamw(learning_rate=lr, b1=b1, b2=b2, weight_decay=weight_decay)


    if stage == 'vqvae':
        transforms = {
            'frozen': zero,
            'vqvae': _adam_l2(vqvae_lr, vqvae_wd),
            'gpt_decay': zero, 'gpt_nodecay': zero, 'bin': zero, 'offset': zero,
        }
    elif stage == 'bet':
        s1 = config['bet_stage1_steps']
        code_active = lambda c: c < s1
        transforms = {
            'frozen': zero,
            'vqvae': zero,
            'gpt_decay': _gated(adamw(wd), code_active),
            'gpt_nodecay': _gated(adamw(0.0), code_active),
            'bin': _gated(adamw(code_wd), code_active),
            'offset': adamw(wd),
        }
    elif stage == 'both':
        pre = config['vqvae_pretrain_steps']
        s1 = config['bet_stage1_steps']
        vqvae_active = lambda c: c < pre
        code_active = lambda c: (c >= pre) & (c < pre + s1)
        offset_active = lambda c: c >= pre
        transforms = {
            'frozen': zero,
            'vqvae': _gated(_adam_l2(vqvae_lr, vqvae_wd), vqvae_active),
            'gpt_decay': _gated(adamw(wd), code_active),
            'gpt_nodecay': _gated(adamw(0.0), code_active),
            'bin': _gated(adamw(code_wd), code_active),
            'offset': _gated(adamw(wd), offset_active),
        }
    else:
        raise ValueError(f"stage must be 'vqvae', 'bet', or 'both'; got {stage!r}.")

    labels = jax.tree_util.tree_map_with_path(_param_label, params)
    return optax.multi_transform(transforms, labels)


class VQBeTAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()


    def _chunk(self, batch):
        seq = batch['actions_seq']
        flat = seq.reshape(seq.shape[0], -1)
        return flat, batch['seq_mask']

    def _masked_l1(self, target_flat, pred_flat, mask):
        B, W = mask.shape
        diff = jnp.abs(target_flat - pred_flat).reshape(B, W, -1)
        A = diff.shape[-1]
        return jnp.sum(diff * mask[:, :, None]) / (jnp.sum(mask) * A + 1e-8)

    def _vqvae_apply(self, method, *args, params=None):
        vqvae = self.network.model_def.modules['vqvae']
        p = self.network.params['modules_vqvae'] if params is None else params['modules_vqvae']
        if params is None:
            p = jax.lax.stop_gradient(p)
        return vqvae.apply({'params': p}, *args, method=method)


    def vqvae_loss(self, batch, grad_params):
        chunk, mask = self._chunk(batch)
        recon, indices, commit_loss, _, _ = self.network.select('vqvae')(chunk, params=grad_params)
        recon_l1 = self._masked_l1(chunk, recon, mask)
        loss = self.config['encoder_loss_multiplier'] * recon_l1 + commit_loss * 5.0
        metrics = {
            'vqvae_loss': loss,
            'encoder_loss': recon_l1,
            'vq_loss_state': commit_loss,
            'n_different_codes': jnp.sum(
                jnp.bincount(indices[:, 0], length=self.config['num_skills']) > 0
            ),
        }
        return loss, metrics

    def bet_loss(self, batch, grad_params, rng=None):
        chunk, mask = self._chunk(batch)
        observations = batch['observations']
        goals = batch['actor_goals'] if self.config['goal_conditioned'] else None
        train = rng is not None

        C, G = self.config['num_skills'], self.config['vqvae_groups']


        action_bins = self._vqvae_apply(VqVae.get_indices, chunk)
        action_bins = jax.lax.stop_gradient(action_bins)


        if train:
            r_drop, r_sample = jax.random.split(rng)
            logits, offsets = self.network.select('policy')(
                observations, goals, deterministic=False,
                params=grad_params, rngs={'dropout': r_drop},
            )
        else:
            logits, offsets = self.network.select('policy')(
                observations, goals, deterministic=True, params=grad_params,
            )


        if train:
            sampled = jax.random.categorical(r_sample, logits, axis=-1)
        else:
            sampled = jnp.argmax(logits, axis=-1)
        sampled = jax.lax.stop_gradient(sampled)


        onehot = jax.nn.one_hot(sampled, C)
        sampled_offsets = jnp.sum(offsets * onehot[..., None], axis=2).sum(axis=1)


        decoded = self._vqvae_apply(VqVae.decode_indices, sampled)
        decoded = jax.lax.stop_gradient(decoded)
        predicted = decoded + sampled_offsets

        offset_loss = self._masked_l1(chunk, predicted, mask)


        loss_primary = focal_loss(logits[:, 0, :], action_bins[:, 0], self.config['gamma'])
        loss_secondary = focal_loss(logits[:, 1, :], action_bins[:, 1], self.config['gamma'])
        cbet_loss = 5.0 * loss_primary + self.config['secondary_code_multiplier'] * loss_secondary

        loss = cbet_loss + self.config['offset_loss_multiplier'] * offset_loss

        pred_primary = jnp.argmax(logits[:, 0, :], axis=-1)
        pred_secondary = jnp.argmax(logits[:, 1, :], axis=-1)
        metrics = {
            'bet_loss': loss,
            'classification_loss': cbet_loss,
            'offset_loss': offset_loss,
            'focal_primary': loss_primary,
            'focal_secondary': loss_secondary,
            'equal_single_code_rate': jnp.mean((pred_primary == action_bins[:, 0]).astype(jnp.float32)),
            'equal_single_code_rate2': jnp.mean((pred_secondary == action_bins[:, 1]).astype(jnp.float32)),
        }
        return loss, metrics


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        stage = self.config['stage']
        if stage == 'vqvae':
            loss, m = self.vqvae_loss(batch, grad_params)
            info.update({f'vqvae/{k}': v for k, v in m.items()})
        elif stage == 'bet':
            loss, m = self.bet_loss(batch, grad_params, rng=rng)
            info.update({f'bet/{k}': v for k, v in m.items()})
        else:
            vq_loss, vq_m = self.vqvae_loss(batch, grad_params)
            info.update({f'vqvae/{k}': v for k, v in vq_m.items()})
            bt_loss, bt_m = self.bet_loss(batch, grad_params, rng=rng)
            info.update({f'bet/{k}': v for k, v in bt_m.items()})
            stage1 = (self.network.step < self.config['vqvae_pretrain_steps']).astype(jnp.float32)
            info['stage1_active'] = stage1
            loss = stage1 * vq_loss + (1.0 - stage1) * bt_loss
        info['total_loss'] = loss
        return loss, info

    def _ema_codebook_params(self, network, batch, run_ema):
        decay = self.config['vqvae_ema_decay']
        eps = self.config['vqvae_ema_eps']
        C = self.config['num_skills']

        chunk, _ = self._chunk(batch)
        _, indices, _, residual_inputs, onehots = network.select('vqvae')(chunk)
        residual_inputs = jax.lax.stop_gradient(residual_inputs)
        onehots = jax.lax.stop_gradient(onehots)

        bins = onehots.sum(0)
        embed_sum = jnp.einsum('bgc,bgd->gcd', onehots, residual_inputs)

        p = network.params['modules_vqvae']
        new_cluster = decay * p['cb_cluster_size'] + (1.0 - decay) * bins
        new_embed_avg = decay * p['cb_embed_avg'] + (1.0 - decay) * embed_sum

        n = new_cluster.sum(axis=-1, keepdims=True)
        smoothed = (new_cluster + eps) / (n + C * eps) * n
        new_embed = new_embed_avg / smoothed[..., None]

        new_cluster = jnp.where(run_ema, new_cluster, p['cb_cluster_size'])
        new_embed_avg = jnp.where(run_ema, new_embed_avg, p['cb_embed_avg'])
        new_embed = jnp.where(run_ema, new_embed, p['cb_embed'])
        return {'cb_cluster_size': new_cluster, 'cb_embed_avg': new_embed_avg, 'cb_embed': new_embed}

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        stage = self.config['stage']
        if stage == 'vqvae':
            run_ema = jnp.asarray(True)
        elif stage == 'bet':
            run_ema = jnp.asarray(False)
        else:
            run_ema = self.network.step < self.config['vqvae_pretrain_steps']

        new_cb = self._ema_codebook_params(self.network, batch, run_ema)
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p, rng=rng))
        p = new_network.params['modules_vqvae']
        new_network = new_network.replace(
            params={**new_network.params, 'modules_vqvae': {**p, **new_cb}}
        )
        return self.replace(network=new_network, rng=new_rng), info


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
        C, G = self.config['num_skills'], self.config['vqvae_groups']
        B = observations.shape[0]

        logits, offsets = self.network.select('policy')(observations, cond_goals, deterministic=True)

        greedy = jnp.argmax(logits, axis=-1)
        scaled = logits / jnp.maximum(temperature, 1e-8)
        sampled = jax.random.categorical(seed, scaled, axis=-1)
        codes = jnp.where(temperature == 0, greedy, sampled)

        onehot = jax.nn.one_hot(codes, C)
        sampled_offsets = jnp.sum(offsets * onehot[..., None], axis=2).sum(axis=1)
        decoded = self._vqvae_apply(VqVae.decode_indices, codes)
        chunk = (decoded + sampled_offsets).reshape(B, self.config['act_window_size'], -1)

        actions = jnp.clip(chunk[:, 0, :], -1, 1)
        if single_obs:
            actions = actions[0]
        return actions


    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        assert not config['discrete'], 'VQ-BeT operates on continuous actions only.'


        assert config['vqvae_groups'] == 2, (
            f"VQ-BeT's code head is specialized to vqvae_groups == 2 (primary + secondary); "
            f"got vqvae_groups={config['vqvae_groups']}."
        )
        assert config['sequence_length'] == config['act_window_size'], (
            f"SequenceDataset window (sequence_length={config['sequence_length']}) must equal the "
            f"action chunk length (act_window_size={config['act_window_size']})."
        )
        assert config['stage'] in ('vqvae', 'bet', 'both'), "stage must be 'vqvae', 'bet', or 'both'."

        action_dim = ex_actions.shape[-1]
        W = config['act_window_size']
        action_chunk_dim = W * action_dim
        obs_dim = ex_observations.shape[-1]
        B = ex_observations.shape[0]

        encoders = {}
        if config.get('encoder') is not None:
            enc = encoder_modules[config['encoder']]
            encoders['policy'] = GCEncoder(state_encoder=enc())

        vqvae_def = VqVae(
            n_groups=config['vqvae_groups'],
            codebook_size=config['num_skills'],
            latent_dim=config['n_latent_dims'],
            action_chunk_dim=action_chunk_dim,
            hidden_dim=config['vqvae_hidden_dim'],
            ema_eps=config['vqvae_ema_eps'],
        )
        gpt_def = GPT(
            input_dim=obs_dim,
            output_dim=config['gpt_output_dim'],
            block_size=config['gpt_block_size'],
            n_layer=config['gpt_n_layer'],
            n_head=config['gpt_n_head'],
            n_embd=config['gpt_n_embd'],
            dropout=config['gpt_dropout'],
        )
        policy_def = BehaviorTransformer(
            obs_dim=obs_dim,
            n_groups=config['vqvae_groups'],
            codebook_size=config['num_skills'],
            action_chunk_dim=action_chunk_dim,
            gpt=gpt_def,
            goal_conditioned=config['goal_conditioned'],
            gc_encoder=encoders.get('policy'),
        )

        ex_chunk = jnp.zeros((B, action_chunk_dim))
        ex_goals = ex_observations if config['goal_conditioned'] else None

        network_def = ModuleDict(dict(vqvae=vqvae_def, policy=policy_def))
        network_params = network_def.init(
            init_rng,
            vqvae=(ex_chunk,),
            policy=(ex_observations, ex_goals),
        )['params']

        network_params = flax.core.unfreeze(network_params)
        network_params['modules_vqvae']['cb_embed_avg'] = network_params['modules_vqvae']['cb_embed']

        if config['stage'] == 'bet' and config.get('restore_vqvae_path') is not None:
            candidates = glob.glob(config['restore_vqvae_path'])
            assert len(candidates) == 1, f'restore_vqvae_path matched {len(candidates)} dirs: {candidates}'
            ckpt = candidates[0] + f"/params_{config['restore_vqvae_epoch']}.pkl"
            with open(ckpt, 'rb') as f:
                loaded = pickle.load(f)['agent']['network']['params']

            def _leaf_shapes(tree):
                return {'/'.join(str(getattr(k, 'key', k)) for k in p): tuple(v.shape)
                        for p, v in jax.tree_util.tree_leaves_with_path(tree)}

            network_params = flax.core.unfreeze(network_params)
            restored = flax.serialization.from_state_dict(network_params['modules_vqvae'], loaded['modules_vqvae'])
            assert _leaf_shapes(network_params['modules_vqvae']) == _leaf_shapes(restored), (
                'VQ-VAE checkpoint architecture mismatch: the stage="vqvae" run must use the same '
                'tokenizer config as this stage="bet" run.'
            )
            network_params['modules_vqvae'] = restored
        elif config['stage'] == 'bet' and config.get('restore_vqvae_path') is None:
            raise ValueError("stage='bet' requires restore_vqvae_path (the stage='vqvae' checkpoint).")

        network = TrainState.create(
            network_def, network_params, tx=make_vqbet_optimizer(config, network_params)
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='vq_bet',

        lr=5.5e-5,
        weight_decay=2e-4,
        code_head_weight_decay=0.01,
        gpt_beta1=0.9,
        gpt_beta2=0.999,
        batch_size=1024,
        discrete=False,


        num_skills=10,
        vqvae_groups=2,
        n_latent_dims=512,
        vqvae_hidden_dim=128,
        encoder_loss_multiplier=0.033,
        vqvae_ema_decay=0.8,
        vqvae_ema_eps=1e-5,
        vqvae_lr=1e-3,
        vqvae_weight_decay=1e-4,
        act_window_size=1,


        gpt_block_size=110,
        gpt_n_layer=6,
        gpt_n_head=6,
        gpt_n_embd=120,
        gpt_output_dim=256,
        gpt_dropout=0.1,
        offset_loss_multiplier=0.1,
        secondary_code_multiplier=3.0,
        gamma=2.0,


        stage='both',
        vqvae_pretrain_steps=500000,
        bet_stage1_steps=250000,
        restore_vqvae_path=ml_collections.config_dict.placeholder(str),
        restore_vqvae_epoch=ml_collections.config_dict.placeholder(int),


        goal_conditioned=True,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='SequenceDataset',
        sequence_length=1,
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
