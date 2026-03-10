import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCValue


class DDPGBCAgent(flax.struct.PyTreeNode):
    """DDPG + Behavioral Cloning agent (no goal conditioning)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params):
        """Compute the DDPG critic loss (TD learning)."""
        # Get next actions from target policy (deterministic for DDPG)
        next_dist = self.network.select('actor')(batch['next_observations'], goals=None)
        next_actions = jnp.clip(next_dist.mode(), -1, 1)
        
        # Compute target Q values using target critic
        next_q = self.network.select('target_critic')(batch['next_observations'], goals=None, actions=next_actions)
        if self.config['ensemble']:
            # Use min Q for ensemble (conservative estimate)
            next_q = jnp.min(next_q, axis=0) if self.config['min_q'] else jnp.mean(next_q, axis=0)
        
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        
        # Compute current Q values
        q = self.network.select('critic')(batch['observations'], goals=None, actions=batch['actions'], params=grad_params)
        if self.config['ensemble']:
            # For ensemble, compute loss for each Q function
            if q.ndim > 1:
                critic_loss = jnp.mean((q - target_q[None, :]) ** 2)
            else:
                critic_loss = jnp.mean((q - target_q) ** 2)
        else:
            critic_loss = jnp.mean((q - target_q) ** 2)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean() if not self.config['ensemble'] or q.ndim == 1 else q.mean(),
            'q_max': q.max() if not self.config['ensemble'] or q.ndim == 1 else q.max(),
            'q_min': q.min() if not self.config['ensemble'] or q.ndim == 1 else q.min(),
            'target_q_mean': target_q.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the DDPG+BC actor loss: -Q(s, π(s)) + α * BC_loss."""
        assert not self.config['discrete'], "DDPG+BC only supports continuous actions"
        
        # Get policy actions (DDPG uses deterministic policy)
        dist = self.network.select('actor')(batch['observations'], goals=None, params=grad_params)
        q_actions = jnp.clip(dist.mode(), -1, 1)
        
        # Compute Q values for policy actions
        q = self.network.select('critic')(batch['observations'], goals=None, actions=q_actions)
        if self.config['ensemble']:
            q = jnp.min(q, axis=0) if self.config['min_q'] else jnp.mean(q, axis=0)
        
        # DDPG loss: maximize Q (minimize -Q)
        # Normalize Q values by the absolute mean to make the loss scale invariant
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        
        # BC loss: match expert actions
        log_prob = dist.log_prob(batch['actions'])
        bc_loss = -(self.config['alpha'] * log_prob).mean()
        
        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob_mean': log_prob.mean(),
            'bc_log_prob_max': log_prob.max(),
            'bc_log_prob_min': log_prob.min(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network):
        """Update the target critic network using soft update."""
        tau = self.config['tau']
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            network.params['modules_critic'],
            network.params['modules_target_critic'],
        )
        new_params = network.params.copy()
        new_params['modules_target_critic'] = new_target_params
        return network.replace(params=new_params)

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self.target_update(new_network)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor (goals are ignored for pure DDPG+BC).
        
        DDPG uses a deterministic policy, so we use dist.mode() instead of dist.sample().
        """
        dist = self.network.select('actor')(observations, goals=None, temperature=temperature)
        actions = dist.mode()  # DDPG is deterministic - use mode, not sample
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        assert not config.get('discrete', False), "DDPG+BC only supports continuous actions"
        action_dim = ex_actions.shape[-1]

        # Define encoder (for pure DDPG+BC, we use state_encoder since we don't have goals)
        encoders = dict()
        if config.get('encoder') is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = GCEncoder(state_encoder=encoder_module())
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())

        # Define critic network (Q(s, a))
        critic_def = GCValue(
            hidden_dims=config.get('value_hidden_dims', config.get('critic_hidden_dims', (512, 512, 512))),
            layer_norm=config.get('layer_norm', True),
            ensemble=config.get('ensemble', True),
            gc_encoder=encoders.get('critic'),
        )

        # Define actor network (π(s))
        # DDPG uses deterministic policy with constant std (no learned std)
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=True,  # Always use constant std for DDPG
            gc_encoder=encoders.get('actor'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, None, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, None, ex_actions)),
            actor=(actor_def, (ex_observations, None)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialize target critic with same params as critic
        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='ddpgbc',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Critic network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network soft update rate.
            ensemble=True,  # Whether to use ensemble for critic.
            min_q=True,  # Whether to use min Q (True) or mean Q (False) for ensemble.
            # const_std is always True for DDPG (hardcoded in create method)
            alpha=0.1,  # BC coefficient in DDPG+BC loss.
            discrete=False,  # Whether the action space is discrete (must be False for DDPG+BC).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters (still uses GCDataset but ignores goals).
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            actor_p_curgoal=0.0,  # Unused (pure DDPG+BC ignores goals).
            actor_p_trajgoal=1.0,  # Unused (pure DDPG+BC ignores goals).
            actor_p_randomgoal=0.0,  # Unused (pure DDPG+BC ignores goals).
            actor_geom_sample=False,  # Unused (pure DDPG+BC ignores goals).
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
