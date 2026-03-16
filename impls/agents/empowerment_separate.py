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
from utils.networks import GCActor, GCDiscreteActor, MLP, TransformedWithMode, default_init


from jax.scipy.special import logsumexp


def log1mexp(x):
    """Stable computation of log(1 - exp(x)) for x <= 0."""
    log2 = -0.6931471805599453
    return jnp.where(
        x < log2,
        jnp.log1p(-jnp.exp(x)),
        jnp.log(-jnp.expm1(x))
    )

# ── Network modules ──────────────────────────────────────────────────────────


class EmpowermentQNetwork(nn.Module):
    """Q network: Q^z(s^+ | s, a) - takes (s, a, z, s^+)."""
    hidden_dims: Sequence[int]
    action_dim: int
    num_skills: int
    latent_dim: int = 128
    layer_norm: bool = True
    discrete: bool = False
    gc_encoder: Optional[nn.Module] = None

    def setup(self):

        self.phi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm
        )

        self.psi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def phi(self, observations, actions, skills):

        obs = self.gc_encoder(observations, None) if self.gc_encoder else observations
        acts = jnp.eye(self.action_dim)[actions] if self.discrete else actions

        saz_inputs = jnp.concatenate([obs, acts, skills], axis=-1)

        return self.phi_net(saz_inputs)

    def psi(self, future_states):

        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states

        return self.psi_net(future)

    def __call__(self, observations, actions, skills, future_states):

        phi_embedding = self.phi(observations, actions, skills)
        psi_embedding = self.psi(future_states)

        # Compute log Q with proper Gaussian normalization
        # For psi ~ N(phi, (d/2) * I): log p(psi | phi) = -0.5 * d * log(π * d) - ||phi - psi||^2 / d
        diff = phi_embedding - psi_embedding
        l2_squared = jnp.sum(diff ** 2, axis=-1)
        # Normalization constant for N(phi, (d/2) * I): -0.5 * d * log(π * d)
        normalization = -0.5 * self.latent_dim * jnp.log(jnp.pi * self.latent_dim)
        return normalization - l2_squared / self.latent_dim


class EmpowermentVNetwork(nn.Module):
    """V network: V^z(s^+ | s) - takes (s, z, s^+)."""
    hidden_dims: Sequence[int]
    action_dim: int
    num_skills: int
    latent_dim: int = 128
    layer_norm: bool = True
    discrete: bool = False
    gc_encoder: Optional[nn.Module] = None

    def setup(self):

        self.phi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm
        )

        self.psi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def phi(self, observations, skills):

        obs = self.gc_encoder(observations, None) if self.gc_encoder else observations

        sz_inputs = jnp.concatenate([obs, skills], axis=-1)

        return self.phi_net(sz_inputs)

    def psi(self, future_states):

        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states

        return self.psi_net(future)

    def __call__(self, observations, skills, future_states):

        phi_embedding = self.phi(observations, skills)
        psi_embedding = self.psi(future_states)

        # Compute log V with proper Gaussian normalization
        # For psi ~ N(phi, (d/2) * I): log p(psi | phi) = -0.5 * d * log(π * d) - ||phi - psi||^2 / d
        diff = phi_embedding - psi_embedding
        l2_squared = jnp.sum(diff ** 2, axis=-1)
        # Normalization constant for N(phi, (d/2) * I): -0.5 * d * log(π * d)
        normalization = -0.5 * self.latent_dim * jnp.log(jnp.pi * self.latent_dim)
        return normalization - l2_squared / self.latent_dim


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

    def compute_logits_from_embedding(self, phi_embedding, psi_embedding, latent_dim):
        """Compute log Q^z(s^+ | s, a) from phi and psi embeddings.
        
        For psi ~ N(phi, (d/2) * I), the log probability density is:
        log p(psi | phi) = -0.5 * d * log(π * d) - ||phi - psi||^2 / d
        
        Args:
            phi_embedding: [..., d] phi embedding (s, a, z)
            psi_embedding: [..., d] psi embedding (s^+)
            latent_dim: int, latent dimension d
            
        Returns:
            log_q: [...,] log Q value with proper Gaussian normalization
        """
        diff = phi_embedding - psi_embedding
        l2_squared = jnp.sum(diff ** 2, axis=-1)
        # Normalization constant for N(phi, (d/2) * I): -0.5 * d * log(π * d)
        normalization = -0.5 * latent_dim * jnp.log(jnp.pi * latent_dim)
        return normalization - l2_squared / latent_dim

    # ── Core value computations ────────────────────────────────────────────

    def empowerment(self, observations, rng):
        batch_size = observations.shape[0]
        K = self.config['num_skills']
        num_samples = self.config['num_splus_samples']

        rng, sample_rng = jax.random.split(rng)
        skill_rngs = jax.random.split(sample_rng, K)

        skills_onehot = jnp.eye(K)

        v_def = self.network.model_def.modules['v']
        v_variables = {'params': self.network.params['modules_v']}

        def phi_for_skill(z_onehot):
            z_batch = jnp.repeat(z_onehot[None, :], batch_size, axis=0)
            return v_def.apply(v_variables, observations, z_batch, method=v_def.phi)

        phi_all = jax.vmap(phi_for_skill)(skills_onehot)  # [K, batch, d]

        def empowerment_for_skill(phi_z, skill_rng):

            noise = jax.random.normal(
                skill_rng,
                (num_samples, *phi_z.shape)
            )

            psi_samples = phi_z[None] + noise * jnp.sqrt(self.config['value_latent_dim'] / 2.0)

            def contribution(psi_splus):

                log_v = self.compute_logits_from_embedding(
                    phi_z,
                    psi_splus,
                    self.config['value_latent_dim']
                )  # [batch]

                log_v_all = self.compute_logits_from_embedding(
                    phi_all,
                    psi_splus,
                    self.config['value_latent_dim']
                )  # [K, batch]

                log_denom = logsumexp(log_v_all, axis=0) - jnp.log(K)

                return log_v - log_denom

            contributions = jax.vmap(contribution)(psi_samples)  # [N, batch]

            return contributions.mean(axis=0)  # average over samples

        emp_per_skill = jax.vmap(
            empowerment_for_skill,
            in_axes=(0, 0)
        )(phi_all, skill_rngs)  # [K, batch]

        empowerment = emp_per_skill.mean(axis=0)  # average over skills

        return empowerment
       

    def compute_q_logits(self, observations, actions, skills_onehot,
                         future_states, params=None):
        """log Q^z(s^+ | s, a) using Q network."""
        future_extracted = self._extract_future(future_states)
        return self.network.select('q')(observations, actions, skills_onehot,
                                        future_extracted, params=params)

    def compute_v_logits(self, observations, skills_onehot, future_states, params=None, policy_params=None):
        """log V^z(s^+ | s) using V network directly (not derived from Q)."""
        future_extracted = self._extract_future(future_states)
        return self.network.select('v')(observations, skills_onehot,
                                        future_extracted, params=params)
    
    def compute_q_logits_target(self, observations, actions, skills_onehot, future_states):
        """log Q^z(s^+ | s, a) using target Q network for stability."""
        future_extracted = self._extract_future(future_states)
        return self.network.select('target_q')(observations, actions, skills_onehot,
                                               future_extracted, params=None)
    
    def compute_v_logits_target(self, observations, skills_onehot, future_states, policy_params=None):
        """log V^z(s^+ | s) using target V network for stability."""
        future_extracted = self._extract_future(future_states)
        return self.network.select('target_v')(observations, skills_onehot,
                                               future_extracted, params=None)

    # ── Losses ────────────────────────────────────────────────────────────────

    def q_loss(self, batch, grad_params, skills_onehot):
        """L_Q = -V^z(s^+ | s) ⊳ log Q^z(s^+ | s, a)  (eq. 15)."""
        future = batch['value_goals']  # s^+ ~ Unif(S), sampled geometrically by GCDataset

        # Use target network for V from next_observations for stability
        log_v = self.compute_v_logits_target(batch['next_observations'], skills_onehot, future)
        log_q = self.compute_q_logits(batch['observations'], batch['actions'],
                                      skills_onehot, future, params=grad_params)
        v = jnp.exp(log_v)
      
        # Numerically stable: log(1 - exp(x)) = log1p(-exp(x)) for x < 0
        # Add epsilon to ensure 1 - exp(log_q) > 0
        log_one_minus_exp_q = log1mexp(log_q)
        
        loss = -(jax.lax.stop_gradient(v) * log_q + jax.lax.stop_gradient(1 - v) * log_one_minus_exp_q).mean()
        return loss, {'q_loss': loss, 'q_log_mean': log_q.mean(), 'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min()}

    def v_loss(self, batch, grad_params, skills_onehot):
        """L_V from eqs. 16-17: Bellman backup for the occupancy V."""
        future = batch['value_goals']  # s^+ ~ Unif(S), sampled geometrically by GCDataset

        actions_next = self._policy_actions(batch['observations'], skills_onehot, params=None)
        
        # Eq. 16: -γ Q^z(s^+ | s, π(s, z)) ⊳ log V^z(s^+ | s) (for future != current)
        # Use target network for Q in Bellman backup for stability
        log_q_next = self.compute_q_logits_target(batch['observations'], actions_next,
                                                  skills_onehot, future)
        log_v = self.compute_v_logits(batch['observations'], skills_onehot,
                                      future, params=grad_params)
        v = jnp.exp(log_v)
       
        # Compute loss_1 per sample (for when future != current, i.e., masks == 1.0)
        # Numerically stable computation
        discount_exp_q = self.config['discount'] * jnp.exp(log_q_next)
        loss_per_sample = - (jax.lax.stop_gradient(discount_exp_q) * log_v + jax.lax.stop_gradient(1 - discount_exp_q) * log1mexp(log_v))
        
        loss = loss_per_sample.mean()
        return loss, {'v_loss': loss, 'v_log_mean': log_v.mean(), 'v_max': v.max(), 'v_min': v.min()}

    def bc_loss(self, batch, grad_params, skills_onehot):
        """Behavioral cloning loss: -log π(a_expert | s, z)."""
        dist = self.network.select('policy')(batch['observations'], skills_onehot, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        loss = -log_prob.mean()
        return loss, {'bc_loss': loss, 'bc_log_prob_mean': log_prob.mean(), 
                     'bc_log_prob_max': log_prob.max(), 'bc_log_prob_min': log_prob.min()}

    def policy_loss(self, batch, grad_params, skills, skills_onehot):
        batch_size = batch['observations'].shape[0]
        K = self.config['num_skills']
        N = self.config['num_splus_samples']
        d = self.config['value_latent_dim']

        rng, sample_rng = jax.random.split(self.rng)

        # ----- sample s+ from V^z(s+ | s) -----

        v_def = self.network.model_def.modules['v']
        target_v_vars = jax.lax.stop_gradient(
            {'params': self.network.params['modules_target_v']}
        )

        phi_z = v_def.apply(
            target_v_vars,
            batch['observations'],
            skills_onehot,
            method=v_def.phi,
        )  # [batch, d]

        psi = phi_z[None] + jax.random.normal(
            sample_rng, (N, *phi_z.shape)
        ) * jnp.sqrt(d / 2.0)

        # ----- compute Q^z(s+ | s, π(s,z)) -----

        q_def = self.network.model_def.modules['q']
        target_q_vars = jax.lax.stop_gradient(
            {'params': self.network.params['modules_target_q']}
        )

        policy_actions = self._policy_actions(
            batch['observations'], skills_onehot, params=grad_params
        )

        phi_z_q = q_def.apply(
            target_q_vars,
            batch['observations'],
            policy_actions,
            skills_onehot,
            method=q_def.phi,
        )  # [batch, d]

        log_q = self.compute_logits_from_embedding(
            phi_z_q[None], psi, d
        )  # [N, batch]

        # ----- compute V^{z'}(s+ | s) for all skills -----

        all_skills_onehot = jnp.eye(K)

        def phi_for_skill(z_onehot):
            z_batch = jnp.repeat(z_onehot[None, :], batch_size, axis=0)
            return v_def.apply(
                target_v_vars,
                batch['observations'],
                z_batch,
                method=v_def.phi,
            )

        phi_all = jax.vmap(phi_for_skill)(all_skills_onehot)  # [K, batch, d]

        log_v_all = self.compute_logits_from_embedding(
            phi_all[None], psi[:, None], d
        )  # [N, K, batch]

        # ----- convert to densities -----

        q = jnp.exp(log_q)           # [N, batch]
        v_all = jnp.exp(log_v_all)   # [N, K, batch]

        # select V^z
        v_z = v_all[:, skills, jnp.arange(batch_size)]  # [N, batch]

        # mixture denominator
        m_bar = (q + (v_all.sum(axis=1) - v_z)) / K
        log_m_bar = jnp.log(m_bar)

        # ----- importance weights -----

        inv_v_z = 1.0 / v_z

        # ----- compute E_delta -----

        # Q^z term
        q_term = (q * inv_v_z) * (log_q - log_m_bar)

        # V^{z'} terms
        v_weighted = v_all * inv_v_z[:, None, :]
        v_terms = v_weighted * (log_v_all - log_m_bar[:, None])

        # remove z skill contribution
        v_terms = v_terms.at[:, skills, jnp.arange(batch_size)].set(0.0)

        v_term_sum = jnp.sum(v_terms, axis=1)  # [N, batch]

        e_delta = ((q_term + v_term_sum) / K).sum(axis=0)  # [batch]

        loss = -e_delta.mean()

        return loss, {
            'policy_loss': loss,
            'e_delta_mean': e_delta.mean(),
            'e_delta_max': e_delta.max(),
            'e_delta_min': e_delta.min(),
        }

    # ── Training ──────────────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        skills_rng, empowerment_rng = jax.random.split(rng, 2)
        # Sample skills once for all losses
        batch_size = batch['observations'].shape[0]
        skills, skills_onehot = self._sample_skills(skills_rng, batch_size)
        info = {}

        q_loss, q_info = self.q_loss(batch, grad_params, skills_onehot)
        info.update({f'q/{k}': v for k, v in q_info.items()})

        v_loss, v_info = self.v_loss(batch, grad_params, skills_onehot)
        info.update({f'v/{k}': v for k, v in v_info.items()})

        pi_loss, pi_info = self.policy_loss(batch, grad_params, skills, skills_onehot)
        info.update({f'policy/{k}': v for k, v in pi_info.items()})

        bc_loss, bc_info = self.bc_loss(batch, grad_params, skills_onehot)
        info.update({f'bc/{k}': v for k, v in bc_info.items()})

        # Compute empowerment for the batch (with stop_gradient to avoid affecting gradients)
        empowerment_vals = jax.lax.stop_gradient(self.empowerment(batch['observations'], rng=empowerment_rng))
        info['empowerment/mean'] = empowerment_vals.mean()
        info['empowerment/min'] = empowerment_vals.min()
        info['empowerment/max'] = empowerment_vals.max()

        alpha = self.config.get('bc_alpha', 0.0)
        total = q_loss + v_loss + pi_loss + alpha * bc_loss
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )
        # Update target networks using soft update
        new_target_q_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            new_network.params[f'modules_q'],
            new_network.params[f'modules_target_q'],
        )
        new_target_v_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            new_network.params[f'modules_v'],
            new_network.params[f'modules_target_v'],
        )
        new_params = {
            **new_network.params,
            f'modules_target_q': new_target_q_params,
            f'modules_target_v': new_target_v_params,
        }
        new_network = new_network.replace(params=new_params)
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
            encoders = {k: GCEncoder(concat_encoder=enc()) for k in ('q', 'v', 'policy')}

        # Create separate Q and V networks
        q_def = EmpowermentQNetwork(
            hidden_dims=config.get('value_hidden_dims', None) or hidden_dims,
            action_dim=action_dim,
            num_skills=num_skills,
            latent_dim=config.get('value_latent_dim', 128),
            layer_norm=config.get('layer_norm', True),
            discrete=config['discrete'],
            gc_encoder=encoders.get('q'),
        )

        v_def = EmpowermentVNetwork(
            hidden_dims=config.get('value_hidden_dims', None) or hidden_dims,
            action_dim=action_dim,
            num_skills=num_skills,
            latent_dim=config.get('value_latent_dim', 128),
            layer_norm=config.get('layer_norm', True),
            discrete=config['discrete'],
            gc_encoder=encoders.get('v'),
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

        # Create target networks (copies of Q and V networks)
        target_q_def = copy.deepcopy(q_def)
        target_v_def = copy.deepcopy(v_def)
        
        network_def = ModuleDict(dict(
            q=q_def,
            v=v_def,
            target_q=target_q_def,
            target_v=target_v_def,
            policy=policy_def
        ))
        network_params = network_def.init(init_rng,
                                          q=(ex_observations, ex_actions, ex_skills, ex_future),
                                          v=(ex_observations, ex_skills, ex_future),
                                          target_q=(ex_observations, ex_actions, ex_skills, ex_future),
                                          target_v=(ex_observations, ex_skills, ex_future),
                                          policy=(ex_observations, ex_skills))['params']
        # Initialize target networks with same params as main networks
        network_params['modules_target_q'] = network_params['modules_q']
        network_params['modules_target_v'] = network_params['modules_v']
        network = TrainState.create(network_def, network_params,
                                    tx=optax.adam(config.get('lr', 3e-4)))

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ────────────────────────────────────────────────────────────────────


def get_config():
    return ml_collections.ConfigDict(dict(
        # Agent
        agent_name='empowerment_separate',
        lr=3e-4,
        batch_size=1024,
        hidden_dims=(512, 512, 512),        # default for value network (overridable)
        value_hidden_dims=ml_collections.config_dict.placeholder(tuple),
        value_latent_dim=128,                # Latent dimension for L2 distance embeddings
        actor_hidden_dims=(512, 512, 512),
        layer_norm=True,
        discount=0.99,
        tau=0.005,                   # Soft update coefficient for target network
        num_skills=5,                 # K: number of skills
        num_splus_samples=8,
        obs_indices=ml_collections.config_dict.placeholder(tuple),  # e.g. (0,1) for x,y
        bc_alpha=0.0,                 # Weight for behavioral cloning loss
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
