"""OPAL — offline VAE skill-controller agent.

================================================================================
WHAT THIS IS
--------------------------------------------------------------------------------
A faithful OGBench re-implementation of the OFFLINE VAE skill-pretraining stage
of SUPE ("Leveraging Skills from Unlabeled Prior Data for Efficient Online
Exploration", arXiv:2410.18076). SUPE's low-level skill controller is trained
with the OPAL VAE objective (Ajay et al. 2021, "OPAL: Offline Primitive
Discovery for Accelerating Offline Reinforcement Learning"). This file ports the
VAE *exactly* from the source implementation the SUPE repo ships:

    supe/supe/pretraining/opal.py            (the `VAE` module + `update_vae`)
    supe/configs/opal_config.py              (hyperparameters)
    supe/run_opal.py                         (the offline OPAL training driver)

Only the low-level skill controller (the VAE) is implemented here, per request.
The high-level components of SUPE/OPAL are intentionally EXCLUDED:
  * the IQL high-level skill policy (`OPAL.iql`, `update_iql`) — "reward labeling",
  * online exploration / RND reward bonuses.
These are the "exploration and reward labeling stuff" that we were asked to skip.

================================================================================
THE OPAL VAE (Sec. "VAE pre-training"; source `VAE` in opal.py)
--------------------------------------------------------------------------------
Given a length-c sub-trajectory chunk tau = (s_{1:c}, a_{1:c}):

  1. Posterior q(z | tau)     — a bidirectional-GRU sequence encoder over the
     per-step [MLP(s_i), a_i] tokens (`SeqEncoder`), projecting to
     (mean, log_std) of a diagonal Gaussian over the skill z in R^{skill_dim}.
     NOTE (matches source exactly): the posterior log_std is used RAW (unclipped)
     and its std is exp(0.5 * log_std).
  2. Prior p(z | s_1)         — a Gaussian MLP head (`GaussianModule`) conditioned
     ONLY on the first observation of the chunk. Its log_std IS clipped to
     [-20, 2] and its std is exp(log_std) (source `GaussianModule`).
  3. Decoder pi(a | s_i, z)   — a Gaussian MLP head (`GaussianModule`) that, given
     a SINGLE reparameterized skill z (shared across the whole chunk) and each
     per-step observation s_i, reconstructs the action a_i. This decoder IS the
     low-level skill-conditioned controller used to execute skills.

Loss (source `update_vae`, verbatim):
    recon_loss = -E[ log pi(a_{1:c} | s_{1:c}, z) ]          (z ~ q, reparameterized)
    kl_loss    =  E[ KL( q(z|tau) || p(z|s_1) ) ]
    total      =  recon_loss + kl_coef * kl_loss
There is no separate free-bits / beta term on the reconstruction; `beta_coef`
exists in the source config but is stored-and-unused by the VAE (it belongs to
the excluded high-level), so it plays no role here (kept in `get_config` only for
hyperparameter provenance).

================================================================================
ADAPTATIONS TO THE OGBench SETTING (documented, minimal)
--------------------------------------------------------------------------------
(1) Chunking / masking. The source `ChunkDataset` only ever samples chunks that
    lie ENTIRELY within one trajectory (its `_allowed_indx` filter), so its VAE
    recon is an unmasked mean over c real steps. OGBench's `SequenceDataset`
    instead clamps windows to the trajectory terminal and returns a per-step
    `seq_mask` (padded past-terminal steps repeat the terminal). To reproduce
    "average over real steps only", the recon loss here is a `seq_mask`-weighted
    mean; when a window has no padding this is identical to the source mean. The
    per-window KL needs no mask (one skill per window; the first state is always
    real). This is the same masking convention DDS uses in this repo.
(2) State-based only. SUPE's optional pixel/CNN path (`cnn=True`, D4PGEncoder,
    latent_dim) is out of scope — the target OGBench datasets are state-based
    (same envs as the QueST sweep). `encode_obs` is therefore the identity, and a
    visual encoder is asserted absent.
(3) Module partition. The single source `VAE` module is split into three OGBench
    `ModuleDict` sub-modules — `encoder` (`SeqEncoder`), `prior` and `decoder`
    (`GaussianModule`) — because `encode_obs` is the identity for state inputs,
    this partition performs the IDENTICAL computation with the IDENTICAL parameter
    set; only the init RNG stream differs (irrelevant to the method).
(4) Evaluation. Without the (excluded) high-level policy there is no goal-directed
    skill selector, so eval rolls out skills sampled from the LEARNED PRIOR
    p(z|s_1), committing each skill for `chunk_size` steps (the option execution
    horizon) and decoding a_i = pi(a|s_i, z). This measures the skill controller /
    prior, not goal-reaching; goal success is expected to be low BY DESIGN given
    that reward labeling / the high level were excluded.

Everything else (network widths, GRU encoder, Gaussian heads, skill_dim, kl_coef,
lr, chunk size, log-std clipping, reparameterization) matches the source.
================================================================================
"""

from functools import partial
from typing import Any, Callable, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


# ── Initialization (source `default_init`: xavier-uniform via fan_avg) ──────────


def default_init(scale: Optional[float] = 1.0):
    """variance_scaling(scale, 'fan_avg', 'uniform') — SUPE's `default_init`."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


# ── MLP (ported verbatim from supe/networks/mlp.py) ────────────────────────────


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    use_pnorm: bool = False
    default_init: nn.initializers.Initializer = nn.initializers.xavier_uniform

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: float = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=self.default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=self.default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)
        return x


# Match the source: OPAL binds MLP's init to `default_init` (xavier-uniform-ish).
MLP = partial(MLP, default_init=default_init)


# ── Bidirectional-GRU sequence encoder (ported verbatim from opal.py) ──────────


class SimpleGRU(nn.Module):
    hidden_size: int

    def setup(self):
        self.gru = nn.GRUCell(features=self.hidden_size)

    @partial(
        nn.transforms.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        return self.gru(carry, x)


class SimpleBiGRU(nn.Module):
    hidden_size: int

    def setup(self):
        self.forward_gru = SimpleGRU(self.hidden_size)
        self.backward_gru = SimpleGRU(self.hidden_size)

    def __call__(self, embedded_inputs):
        shape = embedded_inputs[:, 0].shape

        initial_state = self.forward_gru.gru.initialize_carry(jax.random.key(0), shape)
        _, forward_outputs = self.forward_gru(initial_state, embedded_inputs)

        reversed_inputs = embedded_inputs[:, ::-1, :]
        initial_state = self.backward_gru.gru.initialize_carry(jax.random.key(0), shape)
        _, backward_outputs = self.backward_gru(initial_state, reversed_inputs)
        backward_outputs = backward_outputs[:, ::-1, :]

        outputs = jnp.concatenate([forward_outputs, backward_outputs], -1)
        return outputs


class SeqEncoder(nn.Module):
    """Posterior q(z|tau) sequence encoder (source `SeqEncoder`).

    Per-step obs MLP -> concat action -> `num_recur_layers` BiGRUs -> concat over
    time -> linear projection to `output_dim` (= 2 * skill_dim, i.e. mean||log_std).
    """

    num_recur_layers: int = 2
    output_dim: int = 2
    recur_output: str = "concat"
    hidden_size: int = 256

    def setup(self) -> None:
        self.obs_mlp = MLP([self.hidden_size, self.hidden_size], activate_final=True)
        self.recurs = [
            SimpleBiGRU(self.hidden_size) for _ in range(self.num_recur_layers)
        ]
        self.projection = MLP([self.output_dim], activate_final=False)

    def __call__(
        self,
        seq_observations: jnp.ndarray,
        seq_actions: jnp.ndarray,
    ):
        B, C, D = seq_observations.shape
        observations = jnp.reshape(seq_observations, (B * C, D))
        outputs = jnp.reshape(self.obs_mlp(observations), (B, C, -1))
        outputs = jnp.concatenate([outputs, seq_actions], axis=-1)

        for recur in self.recurs:
            outputs = recur(outputs)
        if self.recur_output == "concat":
            outputs = jnp.reshape(outputs, (B, -1))
        else:
            outputs = outputs[:, -1]
        outputs = self.projection(outputs)

        return outputs


# ── Gaussian MLP head (ported verbatim from opal.py `GaussianModule`) ──────────


class GaussianModule(nn.Module):
    """Diagonal-Gaussian MLP head (source `GaussianModule`).

    Used for BOTH the prior p(z|s_1) (output_dim = skill_dim) and the decoder /
    low-level controller pi(a|s,z) (output_dim = action_dim). log_std is clipped
    to [log_std_min, log_std_max] and the std is exp(log_std) * temperature.
    """

    hidden_dims: Sequence[int]
    output_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
    ) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(inputs)

        means = nn.Dense(
            self.output_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)
        log_stds = nn.Dense(
            self.output_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )

        return distribution


# ── Agent ──────────────────────────────────────────────────────────────────────


class OPALAgent(flax.struct.PyTreeNode):
    """OPAL offline VAE skill-controller (SUPE's low-level pretraining).

    ModuleDict layout:
        encoder   posterior q(z|tau) BiGRU sequence encoder  (`SeqEncoder`)
        prior     prior p(z|s_1) Gaussian MLP head           (`GaussianModule`)
        decoder   low-level controller pi(a|s,z) Gaussian MLP (`GaussianModule`)
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── VAE forward + loss (source `VAE.__call__` + `update_vae`) ──────────────

    def vae_loss(self, batch, grad_params, rng):
        """OPAL VAE loss: recon(-log pi) + kl_coef * KL(q || p) (source update_vae)."""
        obs_seq = batch["observations_seq"]  # [B, C, obs_dim]
        act_seq = batch["actions_seq"]       # [B, C, act_dim]
        seq_mask = batch["seq_mask"]         # [B, C]  (1 real, 0 padded)
        B, C = seq_mask.shape
        skill_dim = self.config["skill_dim"]

        # Posterior q(z|tau): encoder -> (mean, log_std). log_std RAW (source),
        # std = exp(0.5 * log_std).
        enc_out = self.network.select("encoder")(obs_seq, act_seq, params=grad_params)
        means = enc_out[..., :skill_dim]
        log_stds = enc_out[..., skill_dim:]
        stds = jnp.exp(0.5 * log_stds)
        posteriors = distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)

        # Prior p(z|s_1): conditioned on the first observation of the chunk.
        priors = self.network.select("prior")(obs_seq[:, 0], params=grad_params)

        # Reparameterized skill (shared across the whole chunk), then per-step decode.
        zs = means + stds * jax.random.normal(rng, means.shape)          # [B, skill_dim]
        zs = jnp.broadcast_to(zs[:, None, :], (B, C, skill_dim))         # repeat over chunk
        szs = jnp.concatenate([obs_seq, zs], axis=-1)                    # [B, C, obs+skill]
        recon_action_dists = self.network.select("decoder")(szs, params=grad_params)

        # Masked recon mean over real steps (== source unmasked mean when no padding).
        recon_logprob = recon_action_dists.log_prob(act_seq)            # [B, C]
        denom = jnp.maximum(seq_mask.sum(), 1.0)
        recon_loss = -(recon_logprob * seq_mask).sum() / denom

        kl_loss = posteriors.kl_divergence(priors).mean()

        total_loss = recon_loss + self.config["kl_coef"] * kl_loss

        return total_loss, {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
            "prior_mean": priors.loc.mean(),
            "prior_std": priors.scale_diag.mean(),
            "posterior_mean": posteriors.loc.mean(),
            "posterior_std": posteriors.scale_diag.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        loss, info = self.vae_loss(batch, grad_params, rng)
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )
        return self.replace(network=new_network, rng=new_rng), info

    # ── Evaluation: roll out skills sampled from the learned prior p(z|s_1) ─────
    #
    # No high-level policy (excluded by request), so the prior is the skill source.
    # Each skill is committed for `chunk_size` steps (option horizon) and the
    # decoder produces a_i = pi(a | s_i, z). Not goal-directed by construction.

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Stateless: draw z ~ p(z|s) and decode a ~ pi(a|s,z). `goals` unused."""
        if seed is None:
            seed = self.rng
        single_obs = observations.ndim == 1
        obs = observations[None] if single_obs else observations

        skill_seed, action_seed = jax.random.split(seed)
        prior = self.network.select("prior")(obs, temperature)
        skills = prior.sample(seed=skill_seed)
        szs = jnp.concatenate([obs, skills], axis=-1)
        actions = self.network.select("decoder")(szs, temperature).sample(seed=action_seed)
        actions = jnp.clip(actions, -1.0, 1.0)

        if single_obs:
            actions = actions[0]
        return actions

    def init_eval_state(self):
        """Per-episode option state: committed skill + step counter."""
        return {
            "skill": jnp.zeros((self.config["skill_dim"],)),
            "count": jnp.zeros((), jnp.int32),
        }

    @jax.jit
    def sample_actions_with_state(
        self, observations, goals=None, agent_state=None, seed=None, temperature=1.0
    ):
        """Option-style: draw a fresh z ~ p(z|s) every `chunk_size` steps, hold it,
        and decode a_i = pi(a|s_i, z). Returns (action, new_state). `goals` unused."""
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()

        single_obs = observations.ndim == 1
        obs = observations[None] if single_obs else observations

        skill_seed, action_seed = jax.random.split(seed)
        chunk_size = int(self.config["chunk_size"])
        reselect = (agent_state["count"] % chunk_size) == 0

        prior = self.network.select("prior")(obs, temperature)
        sampled = prior.sample(seed=skill_seed)                         # [B, skill_dim]
        committed = jnp.broadcast_to(agent_state["skill"], sampled.shape)
        skills = jnp.where(reselect, sampled, committed)               # hold for chunk_size

        szs = jnp.concatenate([obs, skills], axis=-1)
        actions = self.network.select("decoder")(szs, temperature).sample(seed=action_seed)
        actions = jnp.clip(actions, -1.0, 1.0)

        if single_obs:
            actions = actions[0]
            new_skill = skills[0]
        else:
            new_skill = skills
        new_state = {"skill": new_skill, "count": agent_state["count"] + 1}
        return actions, new_state

    # ── Constructor ────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        assert not config["discrete"], "OPAL's Gaussian decoder targets continuous control."
        assert config.get("encoder") is None, (
            "OPAL here is state-based only (SUPE's pixel/CNN path is out of scope)."
        )
        assert int(config["sequence_length"]) == int(config["chunk_size"]), (
            "sequence_length (SequenceDataset window) must equal chunk_size (skill horizon)."
        )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        config = dict(config)
        skill_dim = config["skill_dim"]
        action_dim = ex_actions.shape[-1]

        B = ex_observations.shape[0]
        C = int(config["chunk_size"])

        # Example inputs for init.
        ex_obs_seq = jnp.broadcast_to(
            ex_observations[:, None], (B, C) + ex_observations.shape[1:]
        )
        ex_act_seq = jnp.broadcast_to(
            ex_actions[:, None], (B, C) + ex_actions.shape[1:]
        )
        ex_obs0 = ex_observations                       # [B, obs_dim]
        ex_skills = jnp.zeros((B, skill_dim))
        ex_sz = jnp.concatenate([ex_obs0, ex_skills], axis=-1)

        encoder_def = SeqEncoder(
            num_recur_layers=2,
            output_dim=skill_dim * 2,
            recur_output="concat",
            hidden_size=config["vae_encoder_hidden_size"],
        )
        prior_def = GaussianModule(
            hidden_dims=tuple(config["vae_hidden_dims"]), output_dim=skill_dim
        )
        decoder_def = GaussianModule(
            hidden_dims=tuple(config["vae_hidden_dims"]), output_dim=action_dim
        )

        network_def = ModuleDict(
            dict(encoder=encoder_def, prior=prior_def, decoder=decoder_def)
        )
        network_params = network_def.init(
            init_rng,
            encoder=(ex_obs_seq, ex_act_seq),
            prior=(ex_obs0,),
            decoder=(ex_sz,),
        )["params"]

        network_tx = optax.adam(learning_rate=config["lr"])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config (source configs/opal_config.py + run_opal.py OGBench overrides) ──────


def get_config():
    return ml_collections.ConfigDict(
        dict(
            # ── Agent ───────────────────────────────────────────────────────
            agent_name="opal",
            lr=3e-4,                     # source opal_config.lr
            batch_size=256,              # source run_opal.py batch_size
            skill_dim=8,                 # source opal_config.skill_dim (latent z dim)
            kl_coef=0.1,                 # source opal_config.kl_coef (antmaze/antsoccer/pointmaze)
            beta_coef=0.25,              # source opal_config.beta_coef (STORED-BUT-UNUSED by the VAE)
            chunk_size=4,                # source run_opal.py horizon_length (skill / option horizon c)
            # ── VAE networks ─────────────────────────────────────────────────
            # OGBench setting from run_opal.py: `if is_ogbench:` overrides the base
            # D4RL widths (256 / (256,256)) with the wider OGBench widths below.
            vae_hidden_dims=(512, 512, 512),  # prior + decoder MLP widths (OGBench override)
            vae_encoder_hidden_size=512,      # BiGRU / obs-MLP width (OGBench override)
            # ── Misc ────────────────────────────────────────────────────────
            discrete=False,              # continuous control only (Gaussian decoder)
            encoder=ml_collections.config_dict.placeholder(str),  # visual encoder (unsupported; keep None)
            # ── Dataset: SequenceDataset feeds length-c windows ─────────────
            # (observations_seq/actions_seq/seq_mask). sequence_length == chunk_size.
            dataset_class="SequenceDataset",
            sequence_length=4,
            discount=0.99,               # source opal_config.discount (GCDataset geom sampling; VAE-irrelevant)
            # Goal/reward sampler knobs are unused by the VAE loss but required by
            # GCDataset; kept as harmless defaults.
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
        )
    )
