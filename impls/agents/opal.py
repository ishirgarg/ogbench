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
Setting `latent_type="discrete"` instead runs the paper's Appendix F offline-DADS
path: sub-trajectories are clustered into `num_skills` primitives by EM on
p(tau,z) = p_omega(z) prod_t p_phi(s_t|s_{t-1},z), then pi(a|s,z) is BC-trained on
the frozen posterior's labels (EM for `cluster_steps`, BC for the rest).
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


# ── Categorical mixture prior p_omega(z) (Appendix F) ──────────────────────────


class CategoricalPrior(nn.Module):
    """Unconditional mixture prior over k skills. Returns unnormalized logits."""

    num_skills: int

    @nn.compact
    def __call__(self):
        return self.param("logits", nn.initializers.zeros, (self.num_skills,))


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

    # ── Discrete path: Appendix F clustering + BC ──────────────────────────────

    def _log_p_tau_given_z(self, obs_seq, seq_mask, grad_params):
        """log p_phi(tau|z) = sum_{t=1..c-1} log p_phi(s_t|s_{t-1}, z), for every z.

        Returns [k, B]. The constant log p(s_0) is dropped (no phi dependence).
        """
        B, C, D = obs_seq.shape
        K = self.config["num_skills"]

        prev = obs_seq[:, :-1]
        deltas = obs_seq[:, 1:] - prev
        step_mask = seq_mask[:, 1:]

        def log_p_for_skill(z_onehot):
            zs = jnp.broadcast_to(z_onehot, (B, C - 1, K))
            dist = self.network.select("traj_model")(
                jnp.concatenate([prev, zs], axis=-1), params=grad_params
            )
            return (dist.log_prob(deltas) * step_mask).sum(axis=-1)

        return jax.vmap(log_p_for_skill)(jnp.eye(K))

    def discrete_loss(self, batch, grad_params, rng):
        """EM on p(tau,z) for `cluster_steps`, then BC on the frozen posterior."""
        obs_seq = batch["observations_seq"]
        act_seq = batch["actions_seq"]
        seq_mask = batch["seq_mask"]
        B, C = seq_mask.shape
        K = self.config["num_skills"]

        # E-step: p(z|tau) by Bayes rule, held fixed via stop_gradient.
        log_prior = jax.nn.log_softmax(
            self.network.select("skill_prior")(params=grad_params)
        )
        log_p_tau = self._log_p_tau_given_z(obs_seq, seq_mask, grad_params)
        log_joint = log_prior[:, None] + log_p_tau
        log_evidence = jax.scipy.special.logsumexp(log_joint, axis=0)
        log_resp = log_joint - log_evidence[None]
        resp = jax.lax.stop_gradient(jnp.exp(log_resp))

        # M-step, normalized per real step to match the VAE path's loss scale.
        denom = jnp.maximum(seq_mask.sum(), 1.0)
        em_loss = -(resp * log_joint).sum(axis=0).sum() / denom

        # BC on one z per window sampled from the posterior.
        label_rng, _ = jax.random.split(rng)
        z_idx = jax.random.categorical(
            label_rng, jax.lax.stop_gradient(log_resp).T, axis=-1
        )
        zs = jnp.broadcast_to(jnp.eye(K)[z_idx][:, None, :], (B, C, K))
        szs = jnp.concatenate([obs_seq, zs], axis=-1)
        bc_logprob = self.network.select("decoder")(szs, params=grad_params).log_prob(act_seq)
        bc_loss = -(bc_logprob * seq_mask).sum() / denom

        in_bc = (self.network.step >= self.config["cluster_steps"]).astype(jnp.float32)
        total_loss = (1.0 - in_bc) * em_loss + in_bc * bc_loss

        prior_probs = jnp.exp(log_prior)
        prior_entropy = -(prior_probs * log_prior).sum()
        post_entropy = -(resp * log_resp).sum(axis=0).mean()

        return total_loss, {
            "total_loss": total_loss,
            "em_loss": em_loss,
            "bc_loss": bc_loss,
            "in_bc_stage": in_bc,
            "log_evidence": log_evidence.mean(),
            "mutual_information": prior_entropy - post_entropy,
            "prior_entropy": prior_entropy,
            "posterior_entropy": post_entropy,
            "prior_min_prob": prior_probs.min(),
            "prior_max_prob": prior_probs.max(),
            "num_active_skills": (resp.mean(axis=1) > 1e-3).sum().astype(jnp.float32),
            "bc_log_prob": (bc_logprob * seq_mask).sum() / denom,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        if self.config["latent_type"] == "discrete":
            return self.discrete_loss(batch, grad_params, rng)
        loss, info = self.vae_loss(batch, grad_params, rng)
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )

        if self.config["latent_type"] == "discrete":
            # Revert the clustering params in the BC stage: their gradient is zero
            # there, but leftover Adam momentum would keep moving them.
            in_bc = self.network.step >= self.config["cluster_steps"]
            frozen = {
                key: jax.tree_util.tree_map(
                    lambda old, new: jnp.where(in_bc, old, new),
                    self.network.params[key],
                    new_network.params[key],
                )
                for key in ("modules_traj_model", "modules_skill_prior")
            }
            new_network = new_network.replace(params={**new_network.params, **frozen})

        return self.replace(network=new_network, rng=new_rng), info

    # ── Evaluation: roll out skills sampled from the learned prior ──────────────
    #
    # No high-level policy (excluded by request), so the prior is the skill source:
    # p(z|s_1) on the continuous path, p_omega(z) on the discrete one. Each skill
    # is committed for `chunk_size` steps (option horizon) and the decoder produces
    # a_i = pi(a | s_i, z). Not goal-directed by construction.

    def _sample_prior_skills(self, obs, seed, temperature):
        """Draw one skill per batch row from the learned prior. [B, skill_width]."""
        if self.config["latent_type"] == "discrete":
            K = self.config["num_skills"]
            logits = jax.nn.log_softmax(self.network.select("skill_prior")())
            idx = jax.random.categorical(
                seed, jnp.broadcast_to(logits, (obs.shape[0], K)), axis=-1
            )
            return jnp.eye(K)[idx]
        return self.network.select("prior")(obs, temperature).sample(seed=seed)

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Stateless: draw z from the prior and decode a ~ pi(a|s,z). `goals` unused."""
        if seed is None:
            seed = self.rng
        single_obs = observations.ndim == 1
        obs = observations[None] if single_obs else observations

        skill_seed, action_seed = jax.random.split(seed)
        skills = self._sample_prior_skills(obs, skill_seed, temperature)
        szs = jnp.concatenate([obs, skills], axis=-1)
        actions = self.network.select("decoder")(szs, temperature).sample(seed=action_seed)
        actions = jnp.clip(actions, -1.0, 1.0)

        if single_obs:
            actions = actions[0]
        return actions

    def _skill_width(self):
        """Width of the skill vector fed to the decoder."""
        if self.config["latent_type"] == "discrete":
            return int(self.config["num_skills"])
        return int(self.config["skill_dim"])

    def init_eval_state(self):
        """Per-episode option state: committed skill + step counter."""
        return {
            "skill": jnp.zeros((self._skill_width(),)),
            "count": jnp.zeros((), jnp.int32),
        }

    @jax.jit
    def sample_actions_with_state(
        self, observations, goals=None, agent_state=None, seed=None, temperature=1.0
    ):
        """Option-style: draw a fresh z from the prior every `chunk_size` steps, hold
        it, and decode a_i = pi(a|s_i, z). Returns (action, new_state). `goals` unused."""
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()

        single_obs = observations.ndim == 1
        obs = observations[None] if single_obs else observations

        skill_seed, action_seed = jax.random.split(seed)
        chunk_size = int(self.config["chunk_size"])
        reselect = (agent_state["count"] % chunk_size) == 0

        sampled = self._sample_prior_skills(obs, skill_seed, temperature)  # [B, skill_width]
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
        assert config["latent_type"] in ("continuous", "discrete"), (
            f'latent_type must be "continuous" or "discrete"; got {config["latent_type"]!r}.'
        )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        config = dict(config)
        skill_dim = config["skill_dim"]
        action_dim = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]

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

        if config["latent_type"] == "discrete":
            assert int(config["cluster_steps"]) > 0, (
                "cluster_steps must be > 0: the BC stage needs a trained p(z|tau)."
            )
            K = int(config["num_skills"])

            traj_model_def = GaussianModule(
                hidden_dims=tuple(config["vae_hidden_dims"]),
                output_dim=obs_dim,
                log_std_min=config["traj_log_std_min"],
            )
            skill_prior_def = CategoricalPrior(num_skills=K)
            decoder_def = GaussianModule(
                hidden_dims=tuple(config["vae_hidden_dims"]), output_dim=action_dim
            )

            ex_onehot = jnp.zeros((B, K))
            ex_traj_in = jnp.concatenate([ex_obs0, ex_onehot], axis=-1)
            network_def = ModuleDict(
                dict(
                    traj_model=traj_model_def,
                    skill_prior=skill_prior_def,
                    decoder=decoder_def,
                )
            )
            network_params = network_def.init(
                init_rng,
                traj_model=(ex_traj_in,),
                skill_prior=(),
                decoder=(jnp.concatenate([ex_obs0, ex_onehot], axis=-1),),
            )["params"]
            network_tx = optax.adam(learning_rate=config["lr"])
            network = TrainState.create(network_def, network_params, tx=network_tx)
            return cls(rng, network=network, config=flax.core.FrozenDict(**config))

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
            latent_type="continuous",    # "continuous" (VAE) | "discrete" (Appendix F)
            skill_dim=8,                 # source opal_config.skill_dim (latent z dim)
            kl_coef=0.1,                 # source opal_config.kl_coef (antmaze/antsoccer/pointmaze)
            beta_coef=0.25,              # source opal_config.beta_coef (STORED-BUT-UNUSED by the VAE)
            chunk_size=10,               # skill / option horizon c
            # ── VAE networks ─────────────────────────────────────────────────
            # OGBench setting from run_opal.py: `if is_ogbench:` overrides the base
            # D4RL widths (256 / (256,256)) with the wider OGBench widths below.
            vae_hidden_dims=(512, 512, 512),  # prior + decoder MLP widths (OGBench override)
            vae_encoder_hidden_size=512,      # BiGRU / obs-MLP width (OGBench override)
            # ── Discrete path (latent_type="discrete"; Appendix F) ───────────
            num_skills=10,               # k (Appendix F.1 tried 5/10/20, chose 10)
            cluster_steps=500_000,       # EM stage length; BC for the remaining steps
            traj_log_std_min=-5.0,       # floor for p_phi; guards mixture collapse
            # ── Misc ────────────────────────────────────────────────────────
            discrete=False,              # continuous control only (Gaussian decoder)
            encoder=ml_collections.config_dict.placeholder(str),  # visual encoder (unsupported; keep None)
            # ── Dataset: SequenceDataset feeds length-c windows ─────────────
            # (observations_seq/actions_seq/seq_mask). sequence_length == chunk_size.
            dataset_class="SequenceDataset",
            sequence_length=10,          # must equal chunk_size (asserted in create)
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
