"""Online contrastive RL (CRL) baseline: the flat goal-conditioned agent, trained from env interaction.

This is the OGBench-side counterpart of JaxGCRL's flat `crl` agent
(`jaxgcrl/agents/crl/crl.py`), built from this repo's primitives:

  * Critic: `GCBilinearValue` (ensemble of two), Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d).
    Trained with the *same* contrastive loss as the offline `agents/crl.py`
    (in-batch dot-product logits + sigmoid binary cross-entropy); positives are
    future observations of the same trajectory drawn at sample time by
    `utils/online_buffer.TrajectoryReplayBuffer` with P(offset = j) proportional
    to discount^j over the remaining rows (JaxGCRL's `flatten_batch` distribution).
  * Actor: `GCActor` with a tanh-squashed, state-dependent-std Gaussian (as in
    `agents/sac.py`) trained with JaxGCRL's SAC-style objective
        E[ alpha * log pi(a | s, g) - Q(s, a, g) ],   Q = critic head 0 (JaxGCRL uses one critic),
    using the reparameterised sample, with the same future observation as the goal.
  * alpha: a `LogParam` auto-tuned toward target entropy -0.5 * action_dim
    (JaxGCRL's `target_entropy = -0.5 * action_size`).

There is no reward, no Bellman target and no target network: the critic is purely
contrastive, faithful to CRL. Rollouts, the replay buffer and the update schedule
live in `main_online.py` / `utils/online_rollout.py`; this file is only the learner
plus `sample_actions`, in the same shape as every other agent in `agents/`.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCBilinearValue, LogParam


class OnlineCRLAgent(flax.struct.PyTreeNode):
    """Flat online CRL agent (contrastive critic + entropy-regularised actor)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── Losses ────────────────────────────────────────────────────────────────

    def contrastive_loss(self, batch, grad_params):
        """In-batch contrastive critic loss; identical in form to `agents/crl.py`."""
        batch_size = batch['observations'].shape[0]

        v, phi, psi = self.network.select('critic')(
            batch['observations'],
            batch['value_goals'],
            actions=batch['actions'],
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        v = jnp.exp(v)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """SAC-style actor + alpha losses (JaxGCRL `update_actor_and_alpha`), goal = the sampled future state."""
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)

        # Critic at its stored params (no `params=grad_params`): the actor loss has zero
        # gradient w.r.t. the critic, so the gradient reaches the actor only through the
        # reparameterised action, as in JaxGCRL. (All losses share one Adam step here;
        # their parameter dependences are disjoint, so this equals separate updates.)
        # JaxGCRL's actor reads a single contrastive critic: use head 0 of the ensemble
        # (both heads are still trained by the contrastive loss, as in agents/crl.py).
        qs = self.network.select('critic')(batch['observations'], batch['actor_goals'], actions=actions)
        q = qs[0]

        alpha = self.network.select('alpha')()
        actor_loss = (alpha * log_probs - q).mean()

        # Entropy temperature: alpha * (H - H_target), H from the stop-gradient sample.
        alpha_param = self.network.select('alpha')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha_param * (entropy - self.config['target_entropy'])).mean()

        total_loss = actor_loss + alpha_loss
        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': entropy,
            'target_entropy': self.config['target_entropy'],
            'q_pi_mean': q.mean(),
            'std': dist._distribution.stddev().mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    # ── Acting ────────────────────────────────────────────────────────────────

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor (temperature=0 gives the tanh(mean) action)."""
        if seed is None:
            seed = self.rng
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    # ── Constructor ───────────────────────────────────────────────────────────

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
            ex_observations: Example batch of observations (also used as example goals).
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if config['discrete']:
            raise NotImplementedError('online_crl supports continuous action spaces only.')

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define networks.
        critic_def = GCBilinearValue(
            hidden_dims=tuple(config['value_hidden_dims']),
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
            state_encoder=encoders.get('critic_state'),
            goal_encoder=encoders.get('critic_goal'),
        )
        actor_def = GCActor(
            hidden_dims=tuple(config['actor_hidden_dims']),
            action_dim=action_dim,
            log_std_min=-5,
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
            gc_encoder=encoders.get('actor'),
        )
        alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        # Future-goal sampling discount for the replay buffer (per env-step row).
        stored_config['goal_discount'] = float(config['discount'])

        return cls(rng, network=network, config=flax.core.FrozenDict(**stored_config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='online_crl',  # Agent name.
            rollout_type='flat',  # Experience collector (see utils/online_rollout.py).
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor (future-goal sampling: P(offset=j) ~ discount^j).
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None -> -mult * dim(A)).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy (JaxGCRL: 0.5).
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            # Online schedule (consumed by main_online.py).
            unroll_length=50,  # Env steps collected between update rounds (JaxGCRL unroll_length).
            utd_ratio=1,  # Gradient steps per env step; each round runs unroll_length * utd_ratio updates.
            min_replay_size=1000,  # Transitions collected before the first update.
            replay_size=1000000,  # Replay buffer capacity in transitions.
            offline_ratio=0.5,  # RLPD (--offline_dataset): fraction of every batch drawn from the offline buffer.
            # Observation pipeline.
            discrete=False,  # Whether the action space is discrete (unsupported here).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None for state-based).
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
