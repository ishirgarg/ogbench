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
from utils.networks import GCActor, GCDiscreteActor, MLP, TransformedWithMode, default_init, GCBilinearValue


# ── Network modules ──────────────────────────────────────────────────────────


class EmpowermentBilinearNetwork(nn.Module):
    """Bilinear discriminator h_θ(s,a,z,s^+) = φ(s,a,z)^T ψ(s^+) / √d.
    
    This represents the log-ratio of occupancy Q^z relative to the buffer distribution.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    action_dim: int
    num_skills: int
    layer_norm: bool = True
    discrete: bool = False
    gc_encoder: Optional[nn.Module] = None

    def setup(self):
        # φ(s, a, z) encoder
        phi_input_dim = self.action_dim if self.discrete else self.action_dim
        # We'll concatenate obs, action, and skill in the phi MLP
        self.phi_mlp = MLP((*self.hidden_dims, self.latent_dim), 
                          activate_final=False, layer_norm=self.layer_norm)
        # ψ(s^+) encoder  
        self.psi_mlp = MLP((*self.hidden_dims, self.latent_dim),
                          activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions, skills, future_states, params=None):
        """Compute h_θ(s,a,z,s^+) = φ(s,a,z)^T ψ(s^+) / √d.
        
        Args:
            observations: Current state s
            actions: Action a
            skills: Skill one-hot z
            future_states: Future state s^+ (already extracted if obs_indices is set)
            params: Optional parameters (for stop-gradient)
        
        Returns:
            Scalar logit value
        """
        # Encode observations
        obs = self.gc_encoder(observations, None) if self.gc_encoder else observations
        
        # Encode actions
        acts = jnp.eye(self.action_dim)[actions] if self.discrete else actions
        
        # Concatenate [obs, action, skill] for phi
        phi_input = jnp.concatenate([obs, acts, skills], axis=-1)
        phi = self.phi_mlp(phi_input)
        
        # Encode future states for psi
        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states
        psi = self.psi_mlp(future)
        
        # Bilinear form: φ^T ψ / √d
        h = (phi * psi).sum(axis=-1) / jnp.sqrt(self.latent_dim)
        
        return h


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

    def compute_h_theta(self, observations, actions, skills_onehot,
                        future_states, params=None):
        """h_θ(s,a,z,s^+) = φ(s,a,z)^T ψ(s^+) / √d (bilinear discriminator)."""
        future_extracted = self._extract_future(future_states)
        return self.network.select('value')(observations, actions, skills_onehot,
                                            future_extracted, params=params)
    
    def compute_q_logits(self, observations, actions, skills_onehot,
                         future_states, params=None):
        """Alias for compute_h_theta for compatibility."""
        return self.compute_h_theta(observations, actions, skills_onehot,
                                    future_states, params=params)
    
    def compute_v_logits(self, observations, skills_onehot, future_states, 
                         params=None, policy_params=None):
        """V^z(s^+ | s) = Q^z(s^+ | s, π(s,z)) = h_θ(s, π(s,z), z, s^+).
        
        Wraps compute_h_theta with policy-sampled actions.
        """
        policy_actions = self._policy_actions(observations, skills_onehot, params=policy_params)
        return self.compute_h_theta(observations, policy_actions, skills_onehot,
                                   future_states, params=params)

    # ── Losses ────────────────────────────────────────────────────────────────

    def q_loss(self, batch, grad_params, rng):
        """InfoNCE loss for Q: L_Q = -E[log exp(h_θ(s,a,z,s^+)) / (exp(h_θ(s,a,z,s^+)) + Σ exp(h_θ(s,a,z,s^-)))].
        
        Positive sampling:
        - s^+ = value_goals (already discounted future samples from GCDataset)
        
        Negative sampling:
        - Sample N random states from the batch
        """
        batch_size = batch['observations'].shape[0]
        rng, skill_rng, neg_rng = jax.random.split(rng, 3)
        _, skills_onehot = self._sample_skills(skill_rng, batch_size)
        
        # Use value_goals as positive future states (already discounted samples)
        future_pos = batch['value_goals']
        
        # Sample negative future states (random states from batch)
        num_negatives = self.config.get('num_negatives', batch_size)
        neg_idxs = jax.random.randint(neg_rng, (batch_size, num_negatives), 0, batch_size)
        neg_futures = batch['observations'][neg_idxs]  # [batch_size, num_negatives, obs_dim]
        
        # Compute h_θ for positive pairs
        h_pos = self.compute_h_theta(batch['observations'], batch['actions'],
                                     skills_onehot, future_pos, params=grad_params)
        
        # Compute h_θ for negative pairs
        # Expand for broadcasting: [batch_size, 1, ...]
        obs_expanded = jnp.expand_dims(batch['observations'], 1)
        acts_expanded = jnp.expand_dims(batch['actions'], 1)
        skills_expanded = jnp.expand_dims(skills_onehot, 1)
        
        # Reshape for batch processing: [batch_size * num_negatives, ...]
        obs_flat = jnp.reshape(obs_expanded, (batch_size * num_negatives, -1))
        acts_flat = jnp.reshape(acts_expanded, (batch_size * num_negatives, -1))
        skills_flat = jnp.reshape(skills_expanded, (batch_size * num_negatives, -1))
        neg_futures_flat = jnp.reshape(neg_futures, (batch_size * num_negatives, -1))
        
        h_neg_flat = self.compute_h_theta(obs_flat, acts_flat, skills_flat,
                                          neg_futures_flat, params=grad_params)
        h_neg = jnp.reshape(h_neg_flat, (batch_size, num_negatives))
        
        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + log(exp(pos) + sum(exp(neg)))
        h_neg_sum = jax.scipy.special.logsumexp(h_neg, axis=1)  # log(sum(exp(neg)))
        h_pos_neg = jax.scipy.special.logsumexp(
            jnp.stack([h_pos, h_neg_sum], axis=1), axis=1
        )  # log(exp(pos) + sum(exp(neg)))
        loss = -(h_pos - h_pos_neg).mean()
        
        q_pos = jnp.exp(h_pos)
        _ = jax.debug.print("q_loss - h_pos: mean={mean}, min={min}, max={max}", 
                           mean=h_pos.mean(), min=h_pos.min(), max=h_pos.max(), ordered=True)
        _ = jax.debug.print("q_loss - q_pos: mean={mean}, min={min}, max={max}", 
                           mean=q_pos.mean(), min=q_pos.min(), max=q_pos.max(), ordered=True)
        
        return loss, {'q_loss': loss, 'h_pos_mean': h_pos.mean(), 'q_pos_mean': q_pos.mean()}


    def policy_loss(self, batch, grad_params, rng):
        """L_π = E_{s^+ ~ D} [M(s^+ | s, π(s,z), z) log M - (1/K) Q^z(s^+ | s, π(s,z)) log Q^z].
        
        Since V^z(s^+ | s) = Q^z(s^+ | s, π(s,z)), we only need Q.
        M(s^+ | s, a, z) = (1/K) Σ_{z'} Q^{z'}(s^+ | s, π(s,z'))
        """
        batch_size = batch['observations'].shape[0]
        skills, skills_onehot = self._sample_skills(rng, batch_size)
        future = batch['value_goals']  # s^+ ~ Unif(S), sampled geometrically by GCDataset

        # Q^z(s^+ | s, π(s, z)) — gradients only through the policy (φ/ψ are fixed)
        policy_actions = self._policy_actions(batch['observations'], skills_onehot,
                                              params=grad_params)
        h_pi = self.compute_h_theta(batch['observations'], policy_actions,
                                    skills_onehot, future, params=None)
        q_pi = jnp.exp(h_pi)
        
        _ = jax.debug.print("policy_loss - h_pi: mean={mean}, min={min}, max={max}", 
                           mean=h_pi.mean(), min=h_pi.min(), max=h_pi.max(), ordered=True)
        _ = jax.debug.print("policy_loss - q_pi: mean={mean}, min={min}, max={max}", 
                           mean=q_pi.mean(), min=q_pi.min(), max=q_pi.max(), ordered=True)

        # V^{z'}(s^+ | s) = Q^{z'}(s^+ | s, π(s, z')) for all K skills — stop-gradient
        all_skills_onehot = jnp.eye(self.config['num_skills'])   # [K, K]

        def v_for_skill(skill_onehot):
            s_batch = jnp.repeat(skill_onehot[None, :], batch_size, axis=0)
            h = self.compute_v_logits(batch['observations'], s_batch, future, 
                                     params=None, policy_params=grad_params)
            return jnp.exp(h)

        v_all = jax.vmap(v_for_skill)(all_skills_onehot)         # [K, batch]

        # M(s^+ | s, a, z) = (1/K) [Q^z + Σ_{z'≠z} V^{z'}]  (eq. 5)
        # v_all[skills, jnp.arange(batch_size)] selects V^z for each sample
        v_z = v_all[skills, jnp.arange(batch_size)]  # [batch]
        v_others = v_all.sum(axis=0) - v_z  # Sum of all V^{z'} minus V^z
        m = (q_pi + v_others) / self.config['num_skills']

        m_log_m = m * jnp.log(m)
        q_log_q = (1.0 / self.config['num_skills']) * q_pi * jnp.log(q_pi)
        loss = -(m_log_m - q_log_q).mean()
        return loss, {'policy_loss': loss, 'm_mean': m.mean(), 'q_pi_mean': q_pi.mean()}

    # ── Training ──────────────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        rng, q_rng, pi_rng = jax.random.split(rng, 3)
        info = {}

        q_loss, q_info = self.q_loss(batch, grad_params, q_rng)
        info.update({f'q/{k}': v for k, v in q_info.items()})

        pi_loss, pi_info = self.policy_loss(batch, grad_params, pi_rng)
        info.update({f'policy/{k}': v for k, v in pi_info.items()})

        total = q_loss + pi_loss
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        
        # Compute gradients
        grads, info = jax.grad(lambda p: self.total_loss(batch, p, rng=rng), has_aux=True)(self.network.params)
        
        # Compute gradient norms for each module
        def compute_grad_norm(grad_tree):
            """Compute L2 norm of gradients for a parameter tree."""
            if not grad_tree:
                return 0.0
            grad_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_tree)], axis=0)
            return jnp.linalg.norm(grad_flat)
        
        # Extract gradients for each module and compute norms
        value_grads = grads.get('value', {})
        policy_grads = grads.get('policy', {})
        info['grad/value_norm'] = compute_grad_norm(value_grads)
        info['grad/policy_norm'] = compute_grad_norm(policy_grads)
        
        # Apply gradients
        new_network = self.network.apply_gradients(grads=grads)
        
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

        latent_dim = config.get('latent_dim', 256)
        value_def = EmpowermentBilinearNetwork(
            hidden_dims=config.get('value_hidden_dims', None) or hidden_dims,
            latent_dim=latent_dim,
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
        latent_dim=256,               # d: latent dimension for bilinear form
        num_negatives=256,            # N: number of negative samples for InfoNCE
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
