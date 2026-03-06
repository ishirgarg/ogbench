from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, MLP, TransformedWithMode, default_init


# ── Network modules ──────────────────────────────────────────────────────────


class EmpowermentValueNetwork(nn.Module):
    """MLP that computes log Q^z(s^+ | s, a) or log V^z(s^+ | s) from all 4 inputs."""

    hidden_dims: Sequence[int]
    action_dim: int
    num_skills: int
    layer_norm: bool = True
    discrete: bool = False
    gc_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations, actions, skills, future_states):
        """Compute log Q^z(s^+ | s, a) or log V^z(s^+ | s).
        
        Args:
            observations: Current state s
            actions: Action a (or π(s, z) for V)
            skills: Skill one-hot z
            future_states: Future state s^+ (already extracted if obs_indices is set)
        
        Returns:
            Scalar log value (always negative due to negative softplus)
        """
        obs = self.gc_encoder(observations, None) if self.gc_encoder else observations
        acts = jnp.eye(self.action_dim)[actions] if self.discrete else actions
        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states
        inputs = jnp.concatenate([obs, acts, skills, future], axis=-1)
        
        # Build hidden layers
        x = MLP(self.hidden_dims, activate_final=False, layer_norm=self.layer_norm)(inputs)
        
        logits = nn.Dense(
            1,
            kernel_init=default_init(scale=0.1),  # Smaller scale for final layer
            bias_init=nn.initializers.constant(4.0)  # Bias to center around 0.5
        )(x)
        # Apply negative softplus to ensure output is always negative
        return -jax.nn.softplus(logits[..., 0])


class SkillConditionedActor(GCActor):
    """Continuous actor conditioned on a skill one-hot rather than a goal."""

    def __call__(self, observations, skills, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = jnp.concatenate([self.gc_encoder(observations, None), skills], axis=-1)
        else:
            inputs = jnp.concatenate([observations, skills], axis=-1)
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        elif self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_stds
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        dist = distrax.MultivariateNormalDiag(loc=means,
                                              scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            dist = TransformedWithMode(dist, distrax.Block(distrax.Tanh(), ndims=1))
        return dist


class SkillConditionedDiscreteActor(GCDiscreteActor):
    """Discrete actor conditioned on a skill one-hot rather than a goal."""

    def __call__(self, observations, skills, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = jnp.concatenate([self.gc_encoder(observations, None), skills], axis=-1)
        else:
            inputs = jnp.concatenate([observations, skills], axis=-1)
        logits = self.logit_net(self.actor_net(inputs))
        return distrax.Categorical(logits=logits / temperature)


# ── Agent ────────────────────────────────────────────────────────────────────


class EmpowermentAgent(flax.struct.PyTreeNode):
    """Offline empowerment agent (Myers 2025).

    Learns a value network modeling the discounted future-state occupancy measure
    and a skill-conditioned policy π(a | s, z) that maximises empowerment.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_future(self, states):
        """Slice states to the goal-relevant subspace fed into ψ(s^+)."""
        obs_indices = self.config.get('obs_indices', None)
        if obs_indices is not None:
            return states[..., jnp.array(obs_indices)]
        return states

    def _sample_skills(self, rng, batch_size):
        """Sample a batch of skill indices and their one-hot encodings."""
        K = self.config['num_skills']
        skills = jax.random.randint(rng, (batch_size,), 0, K)
        return skills, jnp.eye(K)[skills]

    def _policy_actions(self, observations, skills_onehot, params):
        """Query the policy and return deterministic actions (mode/argmax)."""
        dist = self.network.select('policy')(observations, skills_onehot, params=params)
        if self.config['discrete']:
            return dist.probs.argmax(axis=-1)
        return dist.mode()

    # ── Core value computations ────────────────────────────────────────────

    def compute_q_logits(self, observations, actions, skills_onehot,
                         future_states, params=None):
        """log Q^z(s^+ | s, a) = MLP(s, a, z, s^+) with negative softplus."""
        future_extracted = self._extract_future(future_states)
        return self.network.select('value')(observations, actions, skills_onehot,
                                            future_extracted, params=params)

    def compute_v_logits(self, observations, skills_onehot, future_states, params=None, policy_params=None):
        """log V^z(s^+ | s) = MLP(s, π(s,z), z, s^+) with negative softplus.

        Wraps compute_q_logits with stop-gradient policy actions.
        """
        policy_actions = self._policy_actions(observations, skills_onehot, params=policy_params)
        return self.compute_q_logits(observations, policy_actions, skills_onehot,
                                     future_states, params=params)

    # ── Losses ────────────────────────────────────────────────────────────────

    def q_loss(self, batch, grad_params, skills_onehot):
        """L_Q = -V^z(s^+ | s) ⊳ log Q^z(s^+ | s, a)  (eq. 15)."""
        future = batch['value_goals']  # s^+ ~ Unif(S), sampled geometrically by GCDataset

        log_v = self.compute_v_logits(batch['next_observations'], skills_onehot, future)
        log_q = self.compute_q_logits(batch['observations'], batch['actions'],
                                      skills_onehot, future, params=grad_params)
        v = jnp.exp(log_v)
        _ = jax.debug.print("q_loss - log_v: mean={mean}, min={min}, max={max}", 
                           mean=log_v.mean(), min=log_v.min(), max=log_v.max(), ordered=True)
        _ = jax.debug.print("q_loss - v: mean={mean}, min={min}, max={max}", 
                           mean=v.mean(), min=v.min(), max=v.max(), ordered=True)
        loss = -(jax.lax.stop_gradient(v) * log_q + jax.lax.stop_gradient(1 - v) * jnp.log(1 - jnp.exp(log_q))).mean()
        return loss, {'q_loss': loss, 'q_log_mean': log_q.mean(), 'v_mean': v.mean()}

    def v_loss(self, batch, grad_params, skills_onehot):
        """L_V from eqs. 16-17: Bellman backup for the occupancy V."""
        future = batch['value_goals']  # s^+ ~ Unif(S), sampled geometrically by GCDataset

        actions_next = self._policy_actions(batch['observations'], skills_onehot, params=None)
        
        # Eq. 16: -γ Q^z(s^+ | s, π(s, z)) ⊳ log V^z(s^+ | s)
        log_q_next = self.compute_q_logits(batch['observations'], actions_next,
                                           skills_onehot, future)
        log_v = self.compute_v_logits(batch['observations'], skills_onehot,
                                      future, params=grad_params)
        v = jnp.exp(log_v)
        _ = jax.debug.print("v_loss - log_v: mean={mean}, min={min}, max={max}", 
                           mean=log_v.mean(), min=log_v.min(), max=log_v.max(), ordered=True)
        _ = jax.debug.print("v_loss - v: mean={mean}, min={min}, max={max}", 
                           mean=v.mean(), min=v.min(), max=v.max(), ordered=True)
        loss_1 = - (jax.lax.stop_gradient(self.config['discount'] * jnp.exp(log_q_next)) * log_v + jax.lax.stop_gradient(1 - self.config['discount'] * jnp.exp(log_q_next)) * jnp.log(1 - jnp.exp(log_v))).mean()

        # Eq. 17: -[(1-γ) + γ Q^z(s | s, π(s, z))] ⊳ log V^z(s | s)
        log_q_self = self.compute_q_logits(batch['observations'], actions_next,
                                           skills_onehot, batch['observations'])
        log_v_self = self.compute_v_logits(batch['observations'], skills_onehot,
                                           batch['observations'], params=grad_params)
        v_self = jnp.exp(log_v_self)
        _ = jax.debug.print("v_loss - log_v_self: mean={mean}, min={min}, max={max}", 
                           mean=log_v_self.mean(), min=log_v_self.min(), max=log_v_self.max(), ordered=True)
        _ = jax.debug.print("v_loss - v_self: mean={mean}, min={min}, max={max}", 
                           mean=v_self.mean(), min=v_self.min(), max=v_self.max(), ordered=True)
        target_self = (1 - self.config['discount']) + \
                      self.config['discount'] * jax.lax.stop_gradient(jnp.exp(log_q_self))
        loss_2 = -(jax.lax.stop_gradient(target_self) * log_v_self + jax.lax.stop_gradient(1 - target_self) * jnp.log(1 - jnp.exp(log_v_self))).mean()

        loss = loss_1 + loss_2
        return loss, {'v_loss': loss, 'v_loss_1': loss_1, 'v_loss_2': loss_2,
                      'v_log_mean': log_v.mean()}

    def policy_loss(self, batch, grad_params, skills, skills_onehot):
        """L_π = M log M - (1/K) Q^z log Q^z  (eq. 18)."""
        batch_size = batch['observations'].shape[0]
        future = batch['value_goals']  # s^+ ~ Unif(S), sampled geometrically by GCDataset

        # Q^z(s^+ | s, π(s, z)) — gradients only through the policy (φ/ψ are fixed per eq. 12)
        policy_actions = self._policy_actions(batch['observations'], skills_onehot,
                                              params=grad_params)
        log_q_pi = self.compute_q_logits(batch['observations'], policy_actions,
                                         skills_onehot, future, params=None)
        q_pi = jnp.exp(log_q_pi)
        # Debug: log q_pi values (force evaluation by using in a way that can't be optimized away)
        _ = jax.debug.print("log_q_pi: mean={mean}, min={min}, max={max}", 
                           mean=log_q_pi.mean(), min=log_q_pi.min(), max=log_q_pi.max(),
                           ordered=True)
        _ = jax.debug.print("q_pi: mean={mean}, min={min}, max={max}", 
                           mean=q_pi.mean(), min=q_pi.min(), max=q_pi.max(),
                           ordered=True)

        # V^{z'}(s^+ | s) for all K skills — stop-gradient, computed in parallel
        all_skills_onehot = jnp.eye(self.config['num_skills'])   # [K, K]

        def v_for_skill(skill_onehot):
            s_batch = jnp.repeat(skill_onehot[None, :], batch_size, axis=0)
            return jnp.exp(self.compute_v_logits(batch['observations'], s_batch, future, policy_params=grad_params))

        v_all = jax.vmap(v_for_skill)(all_skills_onehot)         # [K, batch]

        # M(s^+ | s, a, z) = (1/K) [Q^z + Σ_{z'≠z} V^{z'}]  (eq. 5)
        v_others = v_all.sum(axis=0) - v_all[skills, jnp.arange(batch_size)]
        m = (q_pi + v_others) / self.config['num_skills']

        m_log_m = m * jnp.log(m)
        q_log_q = (1.0 / self.config['num_skills']) * q_pi * jnp.log(q_pi)
        loss = -(m_log_m - q_log_q).mean()
        return loss, {'policy_loss': loss, 'm_mean': m.mean(), 'q_pi_mean': q_pi.mean()}

    # ── Training ──────────────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        # Sample skills once for all three losses
        batch_size = batch['observations'].shape[0]
        skills, skills_onehot = self._sample_skills(rng, batch_size)
        info = {}

        q_loss, q_info = self.q_loss(batch, grad_params, skills_onehot)
        info.update({f'q/{k}': v for k, v in q_info.items()})

        v_loss, v_info = self.v_loss(batch, grad_params, skills_onehot)
        info.update({f'v/{k}': v for k, v in v_info.items()})

        pi_loss, pi_info = self.policy_loss(batch, grad_params, skills, skills_onehot)
        info.update({f'policy/{k}': v for k, v in pi_info.items()})

        total = q_loss + v_loss + pi_loss
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )
        return self.replace(network=new_network, rng=new_rng), info

    # ── Evaluation ────────────────────────────────────────────────────────────

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions. Goals are mapped deterministically to skills."""
        if seed is None:
            seed = self.rng
        
        # Add batch dimension if missing (for single observation evaluation)
        single_obs = observations.ndim == 1
        if single_obs:
            observations = observations[None, :]
        if goals is not None and goals.ndim == 1:
            goals = goals[None, :]
        
        batch_size = observations.shape[0]

        if goals is not None:
            # Hash the goal-relevant subspace → deterministic skill index (per sample)
            goal_future = self._extract_future(goals)
            # Flatten each goal sample and sum to get a hash
            goal_flat = goal_future.reshape(batch_size, -1).astype(jnp.int32)
            goal_hash = jnp.sum(goal_flat, axis=-1)
            skills = (jnp.abs(goal_hash) % self.config['num_skills']).astype(jnp.int32)
        else:
            skills = jax.random.randint(seed, (batch_size,), 0, self.config['num_skills'])

        skills_onehot = jnp.eye(self.config['num_skills'])[skills]
        dist = self.network.select('policy')(observations, skills_onehot, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        
        # Remove batch dimension if it was added
        if single_obs:
            actions = actions[0]
        return actions

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.max() + 1 if config['discrete'] else ex_actions.shape[-1]
        num_skills  = config.get('num_skills', 10)
        hidden_dims = config.get('hidden_dims', (512, 512))

        # Optional visual encoders
        encoders = {}
        if config.get('encoder') is not None:
            enc = encoder_modules[config['encoder']]
            encoders = {k: GCEncoder(concat_encoder=enc()) for k in ('value', 'policy')}

        value_def = EmpowermentValueNetwork(
            hidden_dims=config.get('value_hidden_dims', None) or hidden_dims,
            action_dim=action_dim,
            num_skills=num_skills,
            layer_norm=config.get('layer_norm', True),
            discrete=config['discrete'],
            gc_encoder=encoders.get('value'),
        )

        actor_cls  = SkillConditionedDiscreteActor if config['discrete'] else SkillConditionedActor
        actor_kwargs = dict(
            hidden_dims=config.get('actor_hidden_dims', (512, 512, 512)),
            action_dim=action_dim,
            gc_encoder=encoders.get('policy'),
        )
        if not config['discrete']:
            actor_kwargs.update(state_dependent_std=False,
                                const_std=config.get('const_std', True))
        policy_def = actor_cls(**actor_kwargs)

        # Example inputs
        batch_size = ex_observations.shape[0]
        ex_skills  = jnp.eye(num_skills)[jnp.arange(batch_size) % num_skills]
        obs_indices = config.get('obs_indices', None)
        ex_future  = (ex_observations[:, jnp.array(obs_indices)]
                      if obs_indices is not None else ex_observations)

        network_def = ModuleDict(dict(value=value_def, policy=policy_def))
        network_params = network_def.init(init_rng,
                                          value=(ex_observations, ex_actions, ex_skills, ex_future),
                                          policy=(ex_observations, ex_skills))['params']
        network = TrainState.create(network_def, network_params,
                                    tx=optax.adam(config.get('lr', 3e-4)))

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ────────────────────────────────────────────────────────────────────


def get_config():
    return ml_collections.ConfigDict(dict(
        # Agent
        agent_name='empowerment',
        lr=3e-4,
        batch_size=1024,
        hidden_dims=(512, 512, 512),        # default for value network (overridable)
        value_hidden_dims=ml_collections.config_dict.placeholder(tuple),
        actor_hidden_dims=(512, 512, 512),
        layer_norm=True,
        discount=0.99,
        num_skills=5,                 # K: number of skills
        obs_indices=ml_collections.config_dict.placeholder(tuple),  # e.g. (0,1) for x,y
        discrete=False,
        const_std=True,
        encoder=ml_collections.config_dict.placeholder(str),
        # Dataset
        dataset_class='GCDataset',
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
    ))
