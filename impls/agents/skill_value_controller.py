"""
Goal-conditioned skill controller distilled from a frozen empowerment value function.

Given a policy π(a | s, z) and a value function V(s, z, g) pretrained with
`empowerment_skill`, this agent learns the high-level controller

    π_hi(z | s, g)  ≈  argmax_z V(s, z, g)

by plain supervised classification. Every gradient step draws a batch of (s, g)
pairs from the offline dataset — with `goal_key='actor_goals'` and
`actor_p_randomgoal=1`, so g is an unrelated state drawn uniformly from the whole
dataset rather than a future state of s's own trajectory — labels each pair with
the value-greedy skill

    z*(s, g) = argmax_z V(s, z, g)

evaluated by the *frozen* pretrained agent, and fits a categorical actor over the
K skills with cross-entropy. Nothing in the pretrained agent is trained here; the
only learned parameters are the controller's.

The point is not to save compute — the value head is bilinear, so
`eval_skill_value_policy.py`'s online selector is already cheap. It is to turn
that selector into an actual parametric policy: something with π_hi(z | s, g)
weights you can inspect, fine-tune with RL, or compare against, and a way to
measure how much of the selector survives amortisation. At eval time
z ~ π_hi(· | s, g), then a = π(s, z) from the frozen low-level policy; with
`skill_horizon > 1` the chosen skill is held for that many env steps before
reselecting, which is how the online selector is scored (default 10 there).

Caveat on eval flags: `--eval_temperature` reaches the *controller* here, not the
low-level actor. At 0 (the default) the skill is the argmax of π_hi — the same
greedy form as the selector, up to distillation error; above 0 the skill is
sampled, which is a different policy. The low-level actor always runs at
`low_temperature` (0 by default) regardless. (`--eval_gaussian` still perturbs the
emitted action, as it does for any agent.)
"""

import glob
import json
import os
import re
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.empowerment_skill import EmpowermentAgent
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent
from utils.networks import GCDiscreteActor


def _latest_epoch(run_dir):
    """Return the largest epoch among params_*.pkl checkpoints in `run_dir`."""
    epochs = [
        int(m.group(1))
        for path in glob.glob(os.path.join(run_dir, 'params_*.pkl'))
        if (m := re.search(r'params_(\d+)\.pkl$', os.path.basename(path)))
    ]
    if not epochs:
        raise FileNotFoundError(f'No params_*.pkl checkpoint found in {run_dir}')
    return max(epochs)


class SkillValueControllerAgent(flax.struct.PyTreeNode):
    """High-level skill controller distilled from a frozen empowerment value function.

    Fields:
        rng: PRNG key (seeds action sampling when no seed is supplied).
        network: `TrainState` over a `ModuleDict` holding the single trainable
            module, `controller`, a `GCDiscreteActor` over the K skill indices.
        skill_agent: The frozen pretrained `empowerment_skill` agent. Its value
            head supplies the labels; its policy π(a | s, z) is executed at
            eval time. It is never updated. It is a plain pytree field, so
            `save_agent` writes a full copy of it (params + optimiser state) into
            every controller checkpoint, and a later `restore_agent` reloads it
            from there rather than from `skill_checkpoint_path`.
        config: Static configuration dictionary.
    """

    rng: Any
    network: Any
    skill_agent: Any
    config: Any = nonpytree_field()

    # ── Goal representation ───────────────────────────────────────────────────

    def _goal_input(self, goals):
        """The slice of `g` the controller conditions on.

        V(s, z, g) depends on g only through ψ(`_extract_future(g)`), so those are
        the only goal dimensions that can move the label. Feeding the controller
        the same slice keeps it from conditioning on dimensions that provably
        cannot matter. With `obs_indices=None` (all state envs so far) this is the
        identity, so it only bites on envs that set `obs_indices`.

        `goals` is required: a goal-conditioned controller has nothing to condition
        on without one. The `goals=None` default on the samplers exists only to match
        the shared actor signature.
        """
        if goals is None:
            raise ValueError(
                'skill_value_controller needs a goal: pi_hi(z | s, g) is goal-conditioned. '
                'Pass `goals` (OGBench supplies it as info["goal"]).'
            )
        return self.skill_agent._extract_future(goals)

    # ── Losses ────────────────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Cross-entropy of π_hi(z | s, g) against the value-greedy skill.

        `label_mode='argmax'` regresses onto the hard label z*(s, g). With
        `label_mode='softmax'` the target is softmax(V(s, ·, g) / label_temperature)
        instead, which keeps the near-ties between skills rather than collapsing
        them; at label_temperature=1 that target is the posterior p(z | s⁺ = g, s)
        implied by the learned V under a uniform skill prior, because V is the log of
        a Gaussian kernel whose normaliser is the same for every z — so its argmax is
        still z*(s, g), just reached from a lower-variance signal.

        The loss is deterministic; `rng` is accepted only so the signature matches
        the `loss_fn` that `update` builds (`main.py`'s validation call omits it).
        """
        del rng
        observations = batch['observations']
        goals = self._goal_input(batch[self.config['goal_key']])

        # The frozen agent's params are constants here, not part of the tree
        # `apply_loss_fn` differentiates, so no gradient can reach them. It is handed
        # the *raw* goal on purpose: `skill_values` applies `_extract_future` itself.
        values = self.skill_agent.skill_values(observations, batch[self.config['goal_key']])
        labels = jnp.argmax(values, axis=-1)                                    # [B]

        dist = self.network.select('controller')(observations, goals, params=grad_params)
        log_probs = jax.nn.log_softmax(dist.logits)                             # [B, K]

        if self.config['label_mode'] == 'argmax':
            loss = -jnp.take_along_axis(log_probs, labels[:, None], axis=-1).squeeze(-1).mean()
        elif self.config['label_mode'] == 'softmax':
            targets = jax.nn.softmax(values / self.config['label_temperature'], axis=-1)
            loss = -jnp.sum(targets * log_probs, axis=-1).mean()
        else:
            raise ValueError(
                f"label_mode must be 'argmax' or 'softmax', got {self.config['label_mode']!r}"
            )

        # Diagnostics. `value_regret` is the quantity that actually matters: how much
        # value the controller's own greedy pick gives up, in log-V nats (not return
        # or success), against the label it is fitting. Accuracy can be poor while
        # regret is ~0 when skills are near-tied, which they often are at K=50.
        preds = jnp.argmax(log_probs, axis=-1)                                  # [B]
        best_values = jnp.max(values, axis=-1)
        pred_values = jnp.take_along_axis(values, preds[:, None], axis=-1).squeeze(-1)
        probs = jnp.exp(log_probs)
        K = self.config['num_skills']

        controller_info = {
            'controller_loss': loss,
            'accuracy': (preds == labels).mean(),
            'value_regret': (best_values - pred_values).mean(),
            'label_value': best_values.mean(),
            'pred_value': pred_values.mean(),
            'entropy': -jnp.sum(probs * log_probs, axis=-1).mean(),
            'max_prob': probs.max(axis=-1).mean(),
            # Fraction of the skill set the labels / predictions actually cover in
            # this batch: a controller that collapses onto one skill shows up here.
            'label_skill_coverage': (jnp.bincount(labels, length=K) > 0).mean(),
            'pred_skill_coverage': (jnp.bincount(preds, length=K) > 0).mean(),
        }
        info = {f'controller/{k}': v for k, v in controller_info.items()}
        # Duplicates controller/controller_loss; kept because several agents here
        # (empowerment_skill, skill_dt, quest, vq_bet) report a `total_loss`, so
        # cross-agent wandb panels line up.
        info['total_loss'] = loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """One gradient step on the controller. The pretrained agent is untouched."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _low_level_actions(self, observations, skills_onehot, seed):
        """a ~ π(· | s, z) from the frozen low-level policy."""
        # `SkillConditionedDiscreteActor` divides logits by the temperature with no
        # clamp (unlike `GCDiscreteActor`), so a literal 0 would give all-NaN logits
        # and an action index of -1 on every step. Floor it there only: the continuous
        # actor scales by the temperature, where an exact 0 is the intended mode.
        temperature = self.config['low_temperature']
        if self.skill_agent.config['discrete']:
            temperature = max(float(temperature), 1e-6)
        dist = self.skill_agent.network.select('policy')(
            observations, skills_onehot, temperature=temperature
        )
        actions = dist.sample(seed=seed)
        if not self.skill_agent.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Hierarchical action: z ~ π_hi(· | s, g), then a ~ π(· | s, z).

        The stateless per-step path. This agent always exposes the committed pair
        below, and `evaluate()` dispatches on their presence, so it never lands
        here; this is for other callers. At `skill_horizon == 1` the two agree.
        """
        if seed is None:
            seed = self.rng
        high_seed, low_seed = jax.random.split(seed)

        # A single (unbatched) state-based obs is 1D; a single visual obs is 3D (HWC).
        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        if single_obs:
            observations = observations[None, ...]
            goals = goals[None, ...] if goals is not None else None

        dist = self.network.select('controller')(
            observations, self._goal_input(goals), temperature=temperature
        )
        skills = dist.sample(seed=high_seed)                     # [B]
        skills_onehot = jnp.eye(self.config['num_skills'])[skills]
        actions = self._low_level_actions(observations, skills_onehot, low_seed)

        if single_obs:
            actions = actions[0]
        return actions

    # ── H-step skill commitment at eval ───────────────────────────────────────
    # `evaluate()` picks up this pair automatically and threads the state through the
    # rollout. With skill_horizon == 1 it reselects every step and is exactly
    # `sample_actions`; larger values match how `eval_skill_value_policy.py` runs the
    # greedy selector (default there: reselect every 10 steps).

    def init_eval_state(self):
        """Per-episode state: the committed skill and the step counter."""
        return {'skill': jnp.zeros((), jnp.int32), 'count': jnp.zeros((), jnp.int32)}

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None,
                                  temperature=1.0):
        """`sample_actions` with the skill held fixed for `skill_horizon` steps.

        Returns `(action, new_state)`; the eval harness threads `new_state` back in.
        """
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()
        high_seed, low_seed = jax.random.split(seed)

        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        obs_b = observations[None, ...] if single_obs else observations
        goals_b = goals[None, ...] if (single_obs and goals is not None) else goals

        H = int(self.config['skill_horizon'])
        reselect = (agent_state['count'] % H) == 0

        dist = self.network.select('controller')(
            obs_b, self._goal_input(goals_b), temperature=temperature
        )
        sampled = dist.sample(seed=high_seed)                                  # [B]
        committed = jnp.broadcast_to(agent_state['skill'], sampled.shape)
        skills = jnp.where(reselect, sampled, committed)

        skills_onehot = jnp.eye(self.config['num_skills'])[skills]
        actions = self._low_level_actions(obs_b, skills_onehot, low_seed)

        if single_obs:
            actions = actions[0]
            new_skill = skills[0]
        else:
            new_skill = skills
        new_state = {'skill': new_skill.astype(jnp.int32), 'count': agent_state['count'] + 1}
        return actions, new_state

    # ── Skill-conditioned evaluation hooks (see eval_skill_policy.py) ─────────
    #
    # Delegated to the frozen agent, so `eval_skill_policy.py`'s skill sweep measures
    # the pretrained low-level policy and bypasses the controller entirely — which is
    # what that sweep is for. Deliberately *not* exposing `skill_values`: the third
    # hook would make `eval_skill_value_policy.py` accept a run of this agent and then
    # score the frozen online selector, reporting numbers that look like the
    # controller's but never touch it. Run that script on the pretrained run dir.

    def skill_set(self, seed=None, num_skills=None, observations=None):
        return self.skill_agent.skill_set(seed=seed, num_skills=num_skills, observations=observations)

    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        # `temperature` is deliberately dropped for `low_temperature`: this hook exists
        # to reproduce the pretrained policy's own execution, which is what the frozen
        # agent's `sample_actions_with_skill` does at `low_temperature`. Same choice as
        # `agents/skill_match.py`. The floor is the one `_low_level_actions` explains.
        del temperature
        low_temperature = self.config['low_temperature']
        if self.skill_agent.config['discrete']:
            low_temperature = max(float(low_temperature), 1e-6)
        return self.skill_agent.sample_actions_with_skill(
            observations, skills, seed=seed, temperature=low_temperature
        )

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of low-level env actions (used only to build
                the frozen pretrained agent).
            config: Configuration dictionary. Must contain `skill_checkpoint_path`
                pointing at an `empowerment_skill` run directory.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # ── Load the frozen pretrained empowerment_skill agent. ───────────────
        ckpt_path = config['skill_checkpoint_path']
        if ckpt_path is None:
            raise ValueError(
                'skill_value_controller requires '
                '--agent.skill_checkpoint_path=<empowerment_skill run dir>.'
            )
        ckpt_path = ckpt_path.rstrip('/')
        flags_path = os.path.join(ckpt_path, 'flags.json')
        if not os.path.exists(flags_path):
            raise FileNotFoundError(f'flags.json not found in {ckpt_path}')
        with open(flags_path) as f:
            emp_flags = json.load(f)
        emp_config = emp_flags['agent']
        if emp_config.get('agent_name') != 'empowerment_skill':
            raise ValueError(
                f"Expected an empowerment_skill checkpoint, got agent_name="
                f"{emp_config.get('agent_name')!r} in {flags_path}"
            )

        # The controller's own observation pipeline has to match the pretrained one:
        # `sample_actions*` reads *this* config's `encoder` to decide whether an
        # unbatched obs is 1D or 3D, then hands the obs to the *pretrained* network;
        # `frame_stack` sets the observation width that network was built for; and
        # main.py fills the example actions from *this* config's `discrete`, which is
        # what EmpowermentAgent.create infers the pretrained action_dim from.
        for key in ('encoder', 'frame_stack', 'discrete'):
            if config[key] != emp_config.get(key):
                expected = emp_config.get(key)
                fix = (f'omit --agent.{key}' if expected is None
                       else f'pass --agent.{key}={expected!r}')
                raise ValueError(
                    f'{key}={config[key]!r} does not match the pretrained checkpoint\'s '
                    f'{key}={expected!r} ({flags_path}); {fix}.'
                )

        if int(config['skill_horizon']) < 1:
            raise ValueError(
                f"skill_horizon must be at least 1, got {config['skill_horizon']}."
            )
        if config['label_mode'] not in ('argmax', 'softmax'):
            raise ValueError(
                f"label_mode must be 'argmax' or 'softmax', got {config['label_mode']!r}."
            )
        if config['label_mode'] == 'softmax' and not config['label_temperature'] > 0:
            # softmax(V / 0) is NaN, and the loss would go NaN silently while the
            # argmax-based diagnostics kept printing finite numbers.
            raise ValueError(
                f"label_temperature must be > 0, got {config['label_temperature']}."
            )

        num_skills = int(emp_config['num_skills'])
        if config['num_skills'] is not None and int(config['num_skills']) != num_skills:
            raise ValueError(
                f"num_skills={config['num_skills']} disagrees with the pretrained "
                f'checkpoint\'s num_skills={num_skills} ({flags_path}); omit the flag, '
                f'it is read from the checkpoint.'
            )

        # `env_name` is a main.py flag the agent never sees, so a mismatch (distilling
        # a value function fit on a different dataset) cannot be caught here. Print what
        # the checkpoint was trained on so the log makes it checkable.
        print(
            f'[skill_value_controller] distilling {ckpt_path}\n'
            f'[skill_value_controller]   pretrained env_name={emp_flags.get("env_name")!r}, '
            f'num_skills={emp_config["num_skills"]} -- --env_name must match.'
        )

        skill_agent = EmpowermentAgent.create(seed, ex_observations, ex_actions, emp_config)
        restore_epoch = config['skill_restore_epoch']
        if restore_epoch is None:
            restore_epoch = _latest_epoch(ckpt_path)
        skill_agent = restore_agent(skill_agent, ckpt_path, restore_epoch)

        # ── Build the trainable controller π_hi(z | s, g). ────────────────────
        gc_encoder = None
        if config['encoder'] is not None:
            gc_encoder = GCEncoder(concat_encoder=encoder_modules[config['encoder']]())

        controller_def = GCDiscreteActor(
            hidden_dims=tuple(config['actor_hidden_dims']),
            action_dim=num_skills,
            gc_encoder=gc_encoder,
        )
        network_def = ModuleDict(dict(controller=controller_def))
        # The goal side is the `_extract_future` slice (see `_goal_input`), so the
        # example goal has to be sliced too or the input width would be wrong on an
        # `obs_indices` env.
        ex_goals = skill_agent._extract_future(ex_observations)
        network_params = network_def.init(
            init_rng, controller=(ex_observations, ex_goals)
        )['params']
        network = TrainState.create(
            network_def, network_params, tx=optax.adam(learning_rate=config['lr'])
        )

        # Resolved from the checkpoint for the agent's own use. Note this does NOT
        # reach the run's flags.json: main.py serialises FLAGS before calling create,
        # so flags.json keeps `num_skills: null` / `skill_restore_epoch: null`. Pass
        # --agent.skill_restore_epoch explicitly if you need the run to record which
        # pretrained epoch it distilled (the restore_agent print above also logs it).
        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        stored_config['num_skills'] = num_skills
        stored_config['skill_restore_epoch'] = restore_epoch
        stored_config['skill_checkpoint_path'] = ckpt_path

        return cls(
            rng,
            network=network,
            skill_agent=skill_agent,
            config=flax.core.FrozenDict(**stored_config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='skill_value_controller',  # Agent name.
            # Path to a pretrained empowerment_skill run directory (holds flags.json
            # and params_*.pkl). Required.
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),
            # Epoch of the checkpoint to restore (None -> latest params_*.pkl).
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),
            # Number of skills. Auto-filled from the checkpoint's flags.json (the
            # run's own flags.json still records null -- see the note in `create`).
            num_skills=ml_collections.config_dict.placeholder(int),
            # Which goal key of the sampled batch the controller conditions on and is
            # labelled with. 'actor_goals' is drawn with the actor_p_* probabilities
            # below (random goals by default).
            goal_key='actor_goals',
            # 'argmax':  cross-entropy onto z*(s, g) = argmax_z V(s, z, g).
            # 'softmax': cross-entropy onto softmax(V(s, ., g) / label_temperature).
            #   At label_temperature=1 that target is the exact posterior
            #   p(z | s+ = g, s) under a uniform skill prior (V is a log Gaussian
            #   kernel with a z-independent normaliser), and its argmax is the same
            #   z*. Untested: no tracked run has used it yet. Plausibly a
            #   lower-variance target when skills are near-tied (they are, at K=50),
            #   but 'argmax' is the literal objective, so it stays the default.
            label_mode='argmax',
            label_temperature=1.0,  # Only used when label_mode='softmax'; must be > 0.
            # Temperature for the frozen low-level policy at execution time
            # (0 -> deterministic).
            low_temperature=0.0,
            # Env steps a chosen skill is held for before reselecting (1 -> every step).
            skill_horizon=1,
            # Controller hyperparameters.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Controller network hidden dimensions.
            # `discrete` refers to the *environment* action space (continuous here);
            # the skill the controller emits is discrete by construction. Read by
            # `evaluate()`, main.py, and `create`'s compatibility check -- it must
            # match the pretrained checkpoint's.
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
            # Dataset hyperparameters (goal-conditioned).
            dataset_class='GCDataset',  # Dataset class name.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            # The value_* block is unused while goal_key='actor_goals'; it is kept
            # matching the actor_* block so switching goal_key changes nothing else.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            # This is the requested (s, g) distribution: state x uniformly random state.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.0,  # Probability of using a future state in the same trajectory.
            actor_p_randomgoal=1.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
