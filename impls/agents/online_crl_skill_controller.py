"""Online CRL (contrastive) high-level skill controller over a frozen OGBench skill policy.

OGBench-side port of JaxGCRL's `crl_skill_controller` (`GoExploreSimple`
``agent_type="crl_skill"``): freeze a pretrained skill-conditioned policy
pi(a | s, z) and learn, online, a goal-conditioned high-level controller
pi_hi(z | s, g) over discrete skills on a Semi-MDP with fixed k-step temporal
commitment. Each SMDP transition is (s_t, z, s_{t+k}); the learner never reads the
macro reward -- the critic is purely contrastive.

Built from this repo's primitives (no JaxGCRL networks):

  * Contrastive critic Q(s, z, g): `GCDiscreteBilinearCritic` (ensemble of two)
    over (observation, one_hot(z)) and the goal observation, trained with the
    same in-batch sigmoid-BCE loss as `agents/crl.py`. Positives are future
    observations of the same episode, drawn at sample time by
    `TrajectoryReplayBuffer` with P(offset = j) proportional to (discount^k)^j over the remaining macro-rows
    (JaxGCRL: gamma measured in env steps, so gamma_macro = gamma^k per row).
  * Categorical actor pi_hi(z | s, g): `GCDiscreteActor`. Actor loss is the exact
    soft discrete objective from JaxGCRL,
        J = E_s sum_z pi(z | s, g) * (alpha * log pi(z | s, g) - Q(s, z, g)),
    with Q evaluated for *every* skill (critic head 0, no gradient).
  * alpha: `LogParam` auto-tuned with alpha * (H(pi) - H_target),
    H_target = min(target_entropy_multiplier * action_dim, target_entropy_cap_frac * log(num_skills)).
    The uncapped term is the same formula and multiplier as `online_crl`'s continuous target entropy,
    so the two agents are directly comparable; the cap keeps it below the categorical distribution's
    maximum possible entropy log(num_skills), since an unreachable target sends alpha to infinity and
    collapses the (temperature=0) eval policy onto one fixed skill regardless of state.

Goals are full goal observations (OGBench convention): the behaviour policy sees
`info['goal']`; training goals are relabelled future observations.

Low-level execution goes through the frozen agent's own `skill_set()` /
`sample_actions_with_skill()` hooks (the contract `eval_skill_policy.py` uses), so
any checkpoint family exposing them works: `empowerment_skill` (one-hot skills)
and `dds` (VQ codebook skills). The frozen agent is a plain pytree field, so
`save_agent` writes a full copy of it into every controller checkpoint.

Eval follows the repo's `init_eval_state` / `sample_actions_with_state` contract:
the argmax skill (at temperature 0) is held for `skill_commitment_k` env steps.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.dds import DDSAgent
from agents.empowerment_skill import EmpowermentAgent
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCDiscreteActor, GCDiscreteBilinearCritic, LogParam
from utils.skill_checkpoint import load_frozen_skill_agent

SKILL_AGENT_CLASSES = dict(empowerment_skill=EmpowermentAgent, dds=DDSAgent)


class OnlineCRLSkillControllerAgent(flax.struct.PyTreeNode):
    """Contrastive high-level controller pi_hi(z | s, g) over a frozen skill policy.

    Fields:
        rng: PRNG key (seeds sampling when no seed is supplied).
        network: `TrainState` over a `ModuleDict` with `actor` (GCDiscreteActor over
            the K skills), `critic` (GCDiscreteBilinearCritic) and `alpha` (LogParam).
        skill_agent: The frozen pretrained skill agent (never updated).
        config: Static configuration dictionary.
    """

    rng: Any
    network: Any
    skill_agent: Any
    config: Any = nonpytree_field()

    # ── Losses ────────────────────────────────────────────────────────────────

    def contrastive_loss(self, batch, grad_params):
        """In-batch contrastive critic loss over (s, one_hot(z)) vs. future goals; form of `agents/crl.py`."""
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
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

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

    def _all_skill_values(self, observations, goals):
        """Q(s, z, g) for every skill z: (B, K), from critic head 0, no gradient.

        JaxGCRL's actor reads a single contrastive critic, so only ensemble head 0 is
        used here (both heads are still trained by the contrastive loss, as in
        agents/crl.py). phi(s, one_hot(z)) is evaluated for all K skills; psi(g) once per row.
        """
        num_skills = self.config['num_skills']
        batch_size = observations.shape[0]
        obs_b = jnp.broadcast_to(observations[:, None, ...], (batch_size, num_skills, *observations.shape[1:]))
        skills_b = jnp.broadcast_to(jnp.arange(num_skills)[None, :], (batch_size, num_skills))
        goals_b = jnp.broadcast_to(goals[:, None, ...], (batch_size, num_skills, *goals.shape[1:]))
        _, phi, psi = self.network.select('critic')(obs_b, goals_b, actions=skills_b, info=True)
        # phi: (E, B, K, d); psi: (E, B, K, d) (goal rows repeated along K).
        qs = (phi * psi[..., :1, :]).sum(-1) / jnp.sqrt(phi.shape[-1])  # (E, B, K)
        return jax.lax.stop_gradient(qs[0])

    def actor_loss(self, batch, grad_params, rng):
        """Exact soft categorical actor loss + alpha loss (JaxGCRL `crl_controller_update`)."""
        del rng  # The discrete objective enumerates every skill; nothing is sampled.
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_pi = jax.nn.log_softmax(dist.logits, axis=-1)  # (B, K)
        pi = jnp.exp(log_pi)
        entropy = -jnp.sum(pi * log_pi, axis=-1)  # (B,)  H(pi)

        q = self._all_skill_values(batch['observations'], batch['actor_goals'])  # (B, K)

        alpha = self.network.select('alpha')()
        # J = E_s sum_z pi(z | s, g) * (alpha * log pi(z | s, g) - Q(s, z, g))
        actor_loss = jnp.sum(pi * (alpha * log_pi - q), axis=-1).mean()

        alpha_param = self.network.select('alpha')(params=grad_params)
        entropy_sg = jax.lax.stop_gradient(entropy).mean()
        alpha_loss = alpha_param * (entropy_sg - self.config['target_entropy'])

        total_loss = actor_loss + alpha_loss
        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': entropy_sg,
            'target_entropy': self.config['target_entropy'],
            'q_pi_mean': jnp.sum(pi * q, axis=-1).mean(),
            'q_max_skill_mean': q.max(axis=-1).mean(),
            'pi_max_mean': jnp.mean(jnp.max(pi, axis=-1)),
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

    # ── Acting: high level ────────────────────────────────────────────────────

    def _single_obs(self, observations):
        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        return observations.ndim == single_obs_ndim

    @jax.jit
    def sample_skills(self, observations, goals, seed=None, temperature=1.0):
        """z ~ pi_hi(. | s, g) (temperature=0 -> argmax). Accepts a single obs or a batch."""
        if seed is None:
            seed = self.rng
        if goals is None:
            raise ValueError('online_crl_skill_controller needs a goal: pi_hi(z | s, g) is goal-conditioned.')
        single = self._single_obs(observations)
        obs_b = observations[None, ...] if single else observations
        goals_b = goals[None, ...] if single else goals
        dist = self.network.select('actor')(obs_b, goals_b, temperature=temperature)
        skills = dist.sample(seed=seed).astype(jnp.int32)
        return skills[0] if single else skills

    # ── Acting: low level (frozen skill policy) ───────────────────────────────

    def _low_temperature(self):
        low_temperature = float(self.config['low_temperature'])
        if self.skill_agent.config['discrete']:
            # A categorical at temperature exactly 0 divides logits by 0; the floor
            # keeps it a (near-)argmax without the NaN.
            low_temperature = max(low_temperature, 1e-6)
        return low_temperature

    @jax.jit
    def low_level_actions(self, observations, skills, seed=None):
        """a ~ pi(. | s, z) from the frozen skill policy for a single observation and skill index."""
        if seed is None:
            seed = self.rng
        skill_vectors = self.skill_agent.skill_set()  # (K, skill_width): one-hots or codebook rows
        skill_vector = skill_vectors[jnp.asarray(skills, dtype=jnp.int32)]
        return self.skill_agent.sample_actions_with_skill(
            observations, skill_vector, seed=seed, temperature=self._low_temperature()
        )

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Stateless hierarchical action: reselect the skill every step (k=1 behaviour)."""
        if seed is None:
            seed = self.rng
        high_seed, low_seed = jax.random.split(seed)
        skill = self.sample_skills(observations, goals, seed=high_seed, temperature=temperature)
        return self.low_level_actions(observations, skill, seed=low_seed)

    # ── k-step skill commitment at eval (contract used by utils/evaluation.py) ─

    def init_eval_state(self):
        """Per-episode state: the committed skill and the step counter."""
        return {'skill': jnp.zeros((), jnp.int32), 'count': jnp.zeros((), jnp.int32)}

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None, temperature=1.0):
        """`sample_actions` with the skill held for `skill_commitment_k` env steps (single-obs eval)."""
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()
        high_seed, low_seed = jax.random.split(seed)

        k = int(self.config['skill_commitment_k'])
        reselect = (agent_state['count'] % k) == 0
        sampled = self.sample_skills(observations, goals, seed=high_seed, temperature=temperature)
        skill = jnp.where(reselect, sampled, agent_state['skill']).astype(jnp.int32)
        actions = self.low_level_actions(observations, skill, seed=low_seed)
        new_state = {'skill': skill, 'count': agent_state['count'] + 1}
        return actions, new_state

    # ── Offline window labelling (RLPD; see utils/rlpd.py) ───────────────────

    @jax.jit
    def chunk_skill_logliks(self, observations, actions):
        """Per-step log pi(a | s, z) of the frozen `empowerment_skill` policy for every skill: [B, K].

        The `skill_bc_relabel_controller` labeller; `SequenceDataset.relabel_chunk_skills`
        turns these into window sums with a prefix sum and takes the argmax per window.
        """
        num_skills = int(self.config['num_skills'])
        batch_size = jax.tree_util.tree_leaves(observations)[0].shape[0]
        eye = jnp.eye(num_skills)

        if self.skill_agent.config['discrete']:
            targets = actions
        else:
            # A tanh-squashed actor's log_prob is +inf at |a| == 1 (OGBench actions do hit +-1).
            targets = jnp.clip(actions, -1.0 + 1e-6, 1.0 - 1e-6)

        def loglik_for_skill(skill):
            skills_onehot = jnp.broadcast_to(eye[skill], (batch_size, num_skills))
            dist = self.skill_agent.network.select('policy')(observations, skills_onehot)
            return dist.log_prob(targets)  # [B]

        return jax.lax.map(loglik_for_skill, jnp.arange(num_skills)).T  # [B, K]

    @jax.jit
    def label_chunk_skills(self, observations_seq, actions_seq, seq_mask, seed):
        """One codebook index per window from the frozen DDS encoder: int32 [B] (the `dds_controller` labeller)."""
        del seed
        return self.skill_agent._assign_skill(observations_seq, actions_seq, seq_mask).astype(jnp.int32)

    def label_offline_windows(self, seq_dataset, seed=0):
        """Label every window [t, t + k) of an offline `SequenceDataset` with the frozen agent's labeller.

        Dispatches on the skill family: `empowerment_skill` uses the BC log-likelihood
        argmax, `dds` the encoder + codebook assignment. Returns `(labels [size] int32,
        stats)`.
        """
        k = int(self.config['skill_commitment_k'])
        assert int(seq_dataset.config['sequence_length']) == k, (
            f'offline windows must be skill_commitment_k={k} long, got {seq_dataset.config["sequence_length"]}.'
        )
        family = self.config['skill_agent_name']
        if family == 'empowerment_skill':
            stats = seq_dataset.relabel_chunk_skills(self)
        elif family == 'dds':
            sequence_length = int(self.skill_agent.config['sequence_length'])
            if sequence_length != k:
                raise ValueError(
                    f'The DDS encoder scores windows of exactly sequence_length={sequence_length} steps, so '
                    f'offline windows cannot be labelled with skill_commitment_k={k}; use k={sequence_length} '
                    f'or a checkpoint trained with sequence_length={k}.'
                )
            stats = seq_dataset.relabel_chunk_skills_from_windows(
                self, seed=seed, num_skills=int(self.config['num_skills'])
            )
        else:
            raise ValueError(f'No offline window labeller for skill agent {family!r}.')
        return np.asarray(seq_dataset.chunk_skills, dtype=np.int32), stats

    # ── Skill-conditioned evaluation hooks (see eval_skill_policy.py) ─────────

    def skill_set(self, seed=None, num_skills=None, observations=None):
        return self.skill_agent.skill_set(seed=seed, num_skills=num_skills, observations=observations)

    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        del temperature  # Reproduce the frozen policy's own execution at low_temperature.
        return self.skill_agent.sample_actions_with_skill(
            observations, skills, seed=seed, temperature=self._low_temperature()
        )

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations (also the example goals).
            ex_actions: Example batch of *low-level* actions (shapes the frozen skill agent).
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        if int(config['skill_commitment_k']) < 1:
            raise ValueError(f"skill_commitment_k must be >= 1, got {config['skill_commitment_k']}.")

        # ── Frozen low-level skill policy ────────────────────────────────────
        skill_agent, resolved = load_frozen_skill_agent(
            seed, ex_observations, ex_actions, config, SKILL_AGENT_CLASSES, caller='online_crl_skill_controller'
        )
        num_skills = resolved['num_skills']
        if resolved['agent_name'] == 'dds':
            seq_len = int(resolved['skill_config'].get('sequence_length', 0))
            if seq_len and int(config['skill_commitment_k']) != seq_len:
                print(
                    f'[online_crl_skill_controller] WARNING: skill_commitment_k={config["skill_commitment_k"]} '
                    f'differs from the DDS checkpoint\'s sequence_length={seq_len} (the horizon its skills were '
                    f'trained for; dds_controller defaults to it).'
                )

        # ── Trainable controller: actor + contrastive critic + alpha ─────────
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        ex_goals = ex_observations
        ex_skills = np.zeros((ex_observations.shape[0],), dtype=np.int32)

        actor_def = GCDiscreteActor(
            hidden_dims=tuple(config['actor_hidden_dims']),
            action_dim=num_skills,
            gc_encoder=encoders.get('actor'),
        )
        critic_def = GCDiscreteBilinearCritic(
            hidden_dims=tuple(config['value_hidden_dims']),
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
            state_encoder=encoders.get('critic_state'),
            goal_encoder=encoders.get('critic_goal'),
            action_dim=num_skills,
        )
        alpha_def = LogParam()

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_skills)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))

        # Resolved values for the agent's own use (main_online.py serialises FLAGS before
        # create, so these do not reach flags.json unless passed explicitly).
        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        stored_config['num_skills'] = num_skills
        stored_config['skill_restore_epoch'] = resolved['restore_epoch']
        stored_config['skill_checkpoint_path'] = resolved['ckpt_path']
        stored_config['skill_agent_name'] = resolved['agent_name']
        # Same formula as online_crl's target entropy (-> `target_entropy_multiplier`), evaluated on
        # the low-level action_dim, so the two agents are directly comparable -- but a categorical
        # over num_skills can never exceed log(num_skills), so clamp to a margin below that ceiling.
        # An unreachable target sends alpha to infinity, which erases the Q-learning signal from the
        # actor loss and collapses the (deterministic, temperature=0) eval policy onto one fixed skill
        # regardless of state (verified 2026-09-02: alpha reached ~1e17 by 1M steps on a K=50
        # checkpoint with the unclamped multiplier*action_dim=4.0 > log(50)=3.912).
        action_dim = ex_actions.shape[-1]
        uncapped_target_entropy = float(config['target_entropy_multiplier']) * float(action_dim)
        max_entropy = float(np.log(num_skills))
        target_entropy_cap = float(config['target_entropy_cap_frac']) * max_entropy
        target_entropy = min(uncapped_target_entropy, target_entropy_cap)
        if uncapped_target_entropy > target_entropy_cap:
            print(
                f'[online_crl_skill_controller] target_entropy clamped: target_entropy_multiplier * '
                f'action_dim = {uncapped_target_entropy:.3f} exceeds target_entropy_cap_frac * '
                f'log(num_skills) = {target_entropy_cap:.3f} (num_skills={num_skills}); using '
                f'{target_entropy:.3f}.'
            )
        stored_config['target_entropy'] = target_entropy
        # Future-goal sampling discount per macro-row: gamma^k (gamma measured in env steps).
        stored_config['goal_discount'] = float(config['discount']) ** int(config['skill_commitment_k'])

        return cls(rng, network=network, skill_agent=skill_agent, config=flax.core.FrozenDict(**stored_config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='online_crl_skill_controller',  # Agent name.
            rollout_type='macro',  # Experience collector (see utils/online_rollout.py).
            lr=3e-4,  # Learning rate (actor, critic and alpha).
            batch_size=1024,  # Batch size (macro-transitions).
            actor_hidden_dims=(512, 512, 512),  # Controller actor hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Contrastive critic hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount per env step; macro-row goal sampling uses discount ** skill_commitment_k.
            # Frozen skill policy.
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),  # Skill-agent run dir (flags.json + params_*.pkl).
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),  # Pretrained epoch (None -> latest).
            num_skills=ml_collections.config_dict.placeholder(int),  # Read from the checkpoint; asserted if set.
            low_temperature=0.0,  # Temperature of the frozen low-level policy (0 -> mode).
            # SMDP.
            skill_commitment_k=20,  # Fixed temporal commitment: env steps per high-level decision.
            gamma_low=1.0,  # Intra-macro-step reward discount (bookkeeping only; the learner ignores rewards).
            target_entropy_multiplier=0.5,  # H_target = multiplier * action_dim (same formula as online_crl).
            target_entropy_cap_frac=0.9,  # H_target is also clamped to <= cap_frac * log(num_skills) (reachable).
            # Online schedule (consumed by main_online.py); units are macro-steps.
            unroll_length=50,  # Macro-steps collected between update rounds.
            utd_ratio=1,  # Gradient steps per macro-step; each round runs unroll_length * utd_ratio updates.
            min_replay_size=1000,  # Macro-transitions collected before the first update.
            replay_size=50000,  # Replay buffer capacity in macro-transitions.
            offline_ratio=0.5,  # RLPD (--offline_dataset): fraction of every batch drawn from the offline buffer.
            # Observation pipeline (must match the skill checkpoint).
            discrete=False,  # Whether the low-level action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None for state-based).
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
