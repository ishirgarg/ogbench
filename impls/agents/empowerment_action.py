"""
Offline empowerment agent (Myers 2025), **without skills**.

This variant learns:
  - Q(s⁺ | s, a):  action-conditioned discounted future-state occupancy measure
  - V(s⁺ | s):     marginal occupancy measure under the (current) policy

and trains the policy to maximise the information gain:

    I(S⁺; A | s) = E_{a~π(s), s⁺~Q(·|s,a)} [ log Q(s⁺|s,a) - log V(s⁺|s) ].

Implementation notes:
  - We keep log-space modelling with `clipped_linexp_loss` for stability.
  - `separate_qv=False` (default): V is derived from Q via the policy action.
  - `separate_qv=True`: independent Q and V networks with separate targets.
"""

from typing import Any, Optional, Sequence
import copy

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, MLP, TransformedWithMode


# ── Numerics helpers ──────────────────────────────────────────────────────────


def clipped_linexp_loss(target, pred, gamma, t=-6, min_q_value=-8):
    """Clipped linexp loss in log space; regresses e^pred to gamma * e^target."""
    target, t = jax.lax.stop_gradient(target), jax.lax.stop_gradient(t)
    target, pred = jnp.clip(target, min_q_value, 0), jnp.clip(pred, min_q_value, 0)
    loss = jnp.where(
        pred > t,
        jnp.exp(jnp.log(gamma) + target - pred) + pred,
        pred + jnp.exp(jnp.log(gamma) + target - t) * (1 - pred + t),
    )
    return loss.mean()


class EmpowermentQNetwork(nn.Module):
    """Q(s⁺ | s, a): φ(s, a) · ψ(s⁺) bilinear structure."""

    hidden_dims: Sequence[int]
    action_dim: int
    latent_dim: int = 128
    layer_norm: bool = True
    discrete: bool = False
    gc_encoder: Optional[nn.Module] = None

    def setup(self):
        self.phi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.psi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def phi(self, observations, actions):
        obs = self.gc_encoder(observations, None) if self.gc_encoder else observations
        acts = jnp.eye(self.action_dim)[actions] if self.discrete else actions
        return self.phi_net(jnp.concatenate([obs, acts], axis=-1))

    def psi(self, future_states):
        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states
        return self.psi_net(future)

    def __call__(self, observations, actions, future_states):
        phi_emb = self.phi(observations, actions)
        psi_emb = self.psi(future_states)
        diff = phi_emb - psi_emb
        l2_sq = jnp.sum(diff**2, axis=-1)
        return -l2_sq / self.latent_dim


class EmpowermentVNetwork(nn.Module):
    """V(s⁺ | s): φ(s) · ψ(s⁺) — independent of actions.

    Only instantiated when separate_qv=True; otherwise V is derived from Q.
    """

    hidden_dims: Sequence[int]
    action_dim: int  # unused; kept for symmetric construction
    latent_dim: int = 128
    layer_norm: bool = True
    discrete: bool = False  # unused; kept for symmetric construction
    gc_encoder: Optional[nn.Module] = None

    def setup(self):
        self.phi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.psi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def phi(self, observations):
        obs = self.gc_encoder(observations, None) if self.gc_encoder else observations
        return self.phi_net(obs)

    def psi(self, future_states):
        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states
        return self.psi_net(future)

    def __call__(self, observations, future_states):
        phi_emb = self.phi(observations)
        psi_emb = self.psi(future_states)
        diff = phi_emb - psi_emb
        l2_sq = jnp.sum(diff**2, axis=-1)
        return -l2_sq / self.latent_dim


class EmpowermentActor(GCActor):
    """Continuous actor π(a|s) (no goals, no skills)."""

    def __call__(self, observations, goals=None, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, None)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        # Always learn variance from state
        log_stds = self.log_std_net(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        dist = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            dist = TransformedWithMode(dist, distrax.Block(distrax.Tanh(), ndims=1))
        return dist


class EmpowermentDiscreteActor(GCDiscreteActor):
    """Discrete actor π(a|s) (no goals, no skills)."""

    def __call__(self, observations, goals=None, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, None)
        else:
            inputs = observations
        logits = self.logit_net(self.actor_net(inputs))
        return distrax.Categorical(logits=logits / temperature)


# ── Agent ─────────────────────────────────────────────────────────────────────


class EmpowermentActionAgent(flax.struct.PyTreeNode):
    """Offline empowerment agent maximising I(S⁺;A|s) (no skills)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _extract_future(self, states):
        obs_indices = self.config.get("obs_indices", None)
        if obs_indices is not None:
            return states[..., jnp.array(obs_indices)]
        return states

    def _policy_actions(self, observations, params):
        dist = self.network.select("policy")(observations, params=params)
        if self.config["discrete"]:
            return dist.probs.argmax(axis=-1)
        return dist.mode()

    def _sample_policy_actions(self, rng, observations, params):
        """Sample actions from π(a|s) (SAC-style stochastic policy)."""
        # Use temperature=1.0; entropy_alpha controls exploration via SAC entropy regularization
        dist = self.network.select("policy")(observations, params=params, temperature=1.0)
        actions = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions)
        return actions, log_prob

    def _logits_from_embeddings(self, phi_emb, psi_emb, latent_dim):
        diff = phi_emb - psi_emb
        l2_sq = jnp.sum(diff**2, axis=-1)
        return -l2_sq / latent_dim

    def _v_phi(self, observations, *, use_target: bool, policy_params=None):
        if self.config["separate_qv"]:
            key = "target_v" if use_target else "v"
            net = self.network.model_def.modules[key]
            vars_ = jax.lax.stop_gradient({"params": self.network.params[f"modules_{key}"]})
            return net.apply(vars_, observations, method=net.phi)

        key = "target_q" if use_target else "q"
        net = self.network.model_def.modules[key]
        vars_ = jax.lax.stop_gradient({"params": self.network.params[f"modules_{key}"]})
        actions = self._policy_actions(observations, params=policy_params)
        return net.apply(vars_, observations, actions, method=net.phi)

    def _q_phi(self, observations, actions, *, use_target: bool):
        key = "target_q" if use_target else "q"
        net = self.network.model_def.modules[key]
        vars_ = jax.lax.stop_gradient({"params": self.network.params[f"modules_{key}"]})
        return net.apply(vars_, observations, actions, method=net.phi)

    def compute_q_logits(self, observations, actions, future_states, params=None):
        future_extracted = self._extract_future(future_states)
        return self.network.select("q")(observations, actions, future_extracted, params=params)

    def compute_q_logits_target(self, observations, actions, future_states):
        future_extracted = self._extract_future(future_states)
        return self.network.select("target_q")(observations, actions, future_extracted, params=None)

    def compute_v_logits(self, observations, future_states, params=None, policy_params=None):
        future_extracted = self._extract_future(future_states)
        if self.config["separate_qv"]:
            return self.network.select("v")(observations, future_extracted, params=params)
        actions = self._policy_actions(observations, params=policy_params)
        return self.network.select("q")(observations, actions, future_extracted, params=params)

    def compute_v_logits_target(self, observations, future_states, policy_params=None):
        future_extracted = self._extract_future(future_states)
        if self.config["separate_qv"]:
            return self.network.select("target_v")(observations, future_extracted, params=None)
        actions = self._policy_actions(observations, params=policy_params)
        return self.network.select("target_q")(observations, actions, future_extracted, params=None)

    def empowerment(self, observations, rng):
        N = self.config["num_splus_samples"]
        M = self.config["num_a_samples"]
        d = self.config["value_latent_dim"]
        action_rng, noise_rng = jax.random.split(rng)

        # Sample M actions for each observation
        action_seeds = jax.random.split(action_rng, M)
        actions_m, _ = jax.vmap(
            lambda r: self._sample_policy_actions(r, observations, params=None)
        )(action_seeds)  # [M, batch, act_dim]

        # Compute φ_Q for each action sample: [M, batch, d]
        phi_q_m = jax.vmap(lambda a: self._q_phi(observations, a, use_target=False))(actions_m)
        # φ_V shared across actions: [batch, d] → [M, batch, d]
        phi_v = self._v_phi(observations, use_target=False, policy_params=None)
        phi_v_m = jnp.repeat(phi_v[None, ...], M, axis=0)

        # For each action sample, Monte Carlo over ψ
        def ig_for_action(phi_q_single, phi_v_single, seed):
            noise = jax.random.normal(seed, (N, *phi_q_single.shape))
            psi = phi_q_single[None] + noise * jnp.sqrt(d / 2.0)  # [N, batch, d]
            log_q = self._logits_from_embeddings(phi_q_single[None], psi, d)  # [N, batch]
            log_v = self._logits_from_embeddings(phi_v_single[None], psi, d)  # [N, batch]
            return (log_q - log_v).mean(axis=0)  # [batch]

        noise_seeds = jax.random.split(noise_rng, M)
        ig_m = jax.vmap(ig_for_action)(phi_q_m, phi_v_m, noise_seeds)  # [M, batch]
        return ig_m.mean(axis=0)

    def q_loss(self, batch, grad_params):
        future = batch["value_goals"]
        log_v = self.compute_v_logits_target(batch["next_observations"], future)
        log_q = self.compute_q_logits(batch["observations"], batch["actions"], future, params=grad_params)
        loss = clipped_linexp_loss(target=log_v, pred=log_q, gamma=1.0, min_q_value=self.config["min_q_value"])
        return loss, {"q_loss": loss, "q_log_mean": log_q.mean(), "v_log_mean": log_v.mean()}

    def v_loss(self, batch, grad_params):
        future = batch["value_goals"]
        actions_next = self._policy_actions(batch["observations"], params=None)
        log_q_next = self.compute_q_logits_target(batch["observations"], actions_next, future)
        log_v = self.compute_v_logits(batch["observations"], future, params=grad_params, policy_params=None)

        loss_future = clipped_linexp_loss(
            target=log_q_next,
            pred=log_v,
            gamma=self.config["discount"],
            min_q_value=self.config["min_q_value"],
        )

        loss = loss_future
        metrics = {"v_loss": loss, "v_loss_future": loss_future, "v_log_mean": log_v.mean()}

        if self.config.get("use_self_v_loss", True):
            log_v_current = self.compute_v_logits(
                batch["observations"], batch["observations"], params=grad_params, policy_params=None
            )
            loss_current = clipped_linexp_loss(
                target=jnp.log(1.0 - self.config["discount"]),
                pred=log_v_current,
                gamma=1.0,
                min_q_value=self.config["min_q_value"],
            )
            loss = loss_future + loss_current
            metrics["v_loss"] = loss
            metrics["v_loss_current"] = loss_current
            metrics["v_log_current_mean"] = log_v_current.mean()

        return loss, metrics

    def bc_loss(self, batch, grad_params):
        dist = self.network.select("policy")(batch["observations"], params=grad_params)
        log_prob = dist.log_prob(batch["actions"])
        loss = -log_prob.mean()
        return loss, {"bc_loss": loss, "bc_log_prob_mean": log_prob.mean()}

    def policy_loss(self, batch, grad_params):
        N = self.config["num_splus_samples"]
        M = self.config["num_a_samples"]
        d = self.config["value_latent_dim"]
        action_rng, noise_rng = jax.random.split(self.rng)

        # Sample M actions from π(a|s); gradients flow through actions and log_prob
        action_seeds = jax.random.split(action_rng, M)
        actions_m, log_prob_m = jax.vmap(
            lambda r: self._sample_policy_actions(r, batch["observations"], params=grad_params)
        )(action_seeds)  # actions_m: [M, batch, act_dim]; log_prob_m: [M, batch]

        # φ_Q for each action sample with target Q params: [M, batch, d]
        phi_q_m = jax.vmap(
            lambda a: self._q_phi(batch["observations"], a, use_target=True)
        )(actions_m)
        # φ_V shared across action samples (target): [batch, d] -> [M, batch, d]
        phi_v = self._v_phi(batch["observations"], use_target=True, policy_params=None)
        phi_v_m = jnp.repeat(phi_v[None, ...], M, axis=0)

        # For each action sample, Monte Carlo over ψ to estimate IG
        def ig_for_action(phi_q_single, phi_v_single, seed):
            noise = jax.random.normal(seed, (N, *phi_q_single.shape))
            psi = phi_q_single[None] + noise * jnp.sqrt(d / 2.0)
            log_q = self._logits_from_embeddings(phi_q_single[None], psi, d)
            log_v = self._logits_from_embeddings(phi_v_single[None], psi, d)
            return (log_q - log_v).mean(axis=0)  # [batch]

        noise_seeds = jax.random.split(noise_rng, M)
        ig_m = jax.vmap(ig_for_action)(phi_q_m, phi_v_m, noise_seeds)  # [M, batch]
        ig = ig_m.mean(axis=0)  # [batch]

        alpha = self.config.get("entropy_alpha", 0.1)
        loss = (alpha * log_prob_m.mean(axis=0) - ig).mean()
        return loss, {
            "policy_loss": loss,
            "ig_mean": ig.mean(),
            "log_pi_mean": log_prob_m.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        _, emp_rng = jax.random.split(rng)
        info = {}

        q_loss, q_info = self.q_loss(batch, grad_params)
        info.update({f"q/{k}": v for k, v in q_info.items()})

        v_loss, v_info = self.v_loss(batch, grad_params)
        info.update({f"v/{k}": v for k, v in v_info.items()})

        pi_loss, pi_info = self.policy_loss(batch, grad_params)
        info.update({f"policy/{k}": v for k, v in pi_info.items()})

        bc_loss, bc_info = self.bc_loss(batch, grad_params)
        info.update({f"bc/{k}": v for k, v in bc_info.items()})

        emp = jax.lax.stop_gradient(self.empowerment(batch["observations"], rng=emp_rng))
        info["empowerment/mean"] = emp.mean()
        info["empowerment/min"] = emp.min()
        info["empowerment/max"] = emp.max()

        alpha = self.config.get("bc_alpha", 0.0)
        total = q_loss + v_loss + pi_loss + alpha * bc_loss
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p, rng=rng))

        new_target_q = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            new_network.params["modules_q"],
            new_network.params["modules_target_q"],
        )
        new_params = {**new_network.params, "modules_target_q": new_target_q}

        if self.config["separate_qv"]:
            new_target_v = jax.tree_util.tree_map(
                lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
                new_network.params["modules_v"],
                new_network.params["modules_target_v"],
            )
            new_params["modules_target_v"] = new_target_v

        new_network = new_network.replace(params=new_params)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions. Accepts goals for API compatibility; goals are ignored."""
        if seed is None:
            seed = self.rng

        single_obs = observations.ndim == 1
        if single_obs:
            observations = observations[None, :]
        if goals is not None and goals.ndim == 1:
            goals = goals[None, :]

        dist = self.network.select("policy")(observations, params=None, temperature=temperature)
        actions = dist.sample(seed=seed)
        return actions[0] if single_obs else actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.max() + 1 if config["discrete"] else ex_actions.shape[-1]
        hidden_dims = config.get("hidden_dims", (512, 512))
        separate_qv = config.get("separate_qv", False)

        encoders = {}
        if config.get("encoder") is not None:
            enc = encoder_modules[config["encoder"]]
            keys = ("q", "v", "policy") if separate_qv else ("q", "policy")
            encoders = {k: GCEncoder(concat_encoder=enc()) for k in keys}

        value_kwargs = dict(
            hidden_dims=config.get("value_hidden_dims", None) or hidden_dims,
            action_dim=action_dim,
            latent_dim=config.get("value_latent_dim", 128),
            layer_norm=config.get("layer_norm", True),
            discrete=config["discrete"],
        )

        q_def = EmpowermentQNetwork(**value_kwargs, gc_encoder=encoders.get("q"))
        target_q_def = copy.deepcopy(q_def)

        actor_cls = EmpowermentDiscreteActor if config["discrete"] else EmpowermentActor
        actor_kwargs = dict(
            hidden_dims=config.get("actor_hidden_dims", (512, 512, 512)),
            action_dim=action_dim,
            gc_encoder=encoders.get("policy"),
        )
        if not config["discrete"]:
            # Always learn variance from state
            actor_kwargs.update(
                state_dependent_std=True,
                const_std=False,
            )
        policy_def = actor_cls(**actor_kwargs)

        obs_indices = config.get("obs_indices", None)
        ex_future = ex_observations[:, jnp.array(obs_indices)] if obs_indices is not None else ex_observations

        if separate_qv:
            v_def = EmpowermentVNetwork(**value_kwargs, gc_encoder=encoders.get("v"))
            target_v_def = copy.deepcopy(v_def)
            network_def = ModuleDict(
                dict(q=q_def, v=v_def, target_q=target_q_def, target_v=target_v_def, policy=policy_def)
            )
            params = network_def.init(
                init_rng,
                q=(ex_observations, ex_actions, ex_future),
                v=(ex_observations, ex_future),
                target_q=(ex_observations, ex_actions, ex_future),
                target_v=(ex_observations, ex_future),
                policy=(ex_observations,),
            )["params"]
        else:
            network_def = ModuleDict(dict(q=q_def, target_q=target_q_def, policy=policy_def))
            params = network_def.init(
                init_rng,
                q=(ex_observations, ex_actions, ex_future),
                target_q=(ex_observations, ex_actions, ex_future),
                policy=(ex_observations,),
            )["params"]

        tx = optax.adam(learning_rate=config.get("lr", 3e-4))
        network = TrainState.create(model_def=network_def, params=params, tx=tx)
        return cls(rng=rng, network=network, config=config)


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="empowerment_action",
            lr=3e-4,
            batch_size=1024,
            hidden_dims=(512, 512, 512),
            value_hidden_dims=ml_collections.config_dict.placeholder(tuple),
            value_latent_dim=128,
            actor_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            num_splus_samples=16,
            num_a_samples=8,
            obs_indices=ml_collections.config_dict.placeholder(tuple),
            bc_alpha=0.0,
            separate_qv=False,
            min_q_value=-8,
            use_self_v_loss=True,
            entropy_alpha=0.1,  # SAC-style entropy regularization coefficient
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            dataset_class="GCDataset",
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )

