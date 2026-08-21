from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    # Additive hook: agents that commit a skill/option for several steps (e.g. DDS) expose
    # `init_eval_state`/`sample_actions_with_state`, which thread a small per-episode state
    # (the committed skill + step counter) through the eval loop. Agents without the hook keep
    # the original stateless per-step `sample_actions` path unchanged.
    use_eval_state = hasattr(agent, 'init_eval_state') and hasattr(agent, 'sample_actions_with_state')
    if use_eval_state:
        actor_fn = supply_rng(agent.sample_actions_with_state, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    else:
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        agent_state = agent.init_eval_state() if use_eval_state else None
        while not done:
            if use_eval_state:
                action, agent_state = actor_fn(
                    observations=observation, goals=goal, temperature=eval_temperature, agent_state=agent_state
                )
            else:
                action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def evaluate_skill(
    agent,
    env,
    skill,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    eval_temperature=0,
    eval_gaussian=None,
    seed=None,
):
    """Evaluate a skill-conditioned agent under one *fixed* skill on one task.

    Unlike `evaluate`, the skill is pinned for the whole episode and the goal is
    never fed to the policy: these agents are goal-agnostic, and the goal enters
    only through the env's task_id (which sets the init state and the success
    criterion). Requires the agent to implement `sample_actions_with_skill`.

    Args:
        agent: Skill-conditioned agent.
        env: Environment.
        skill: Fixed skill vector, shape [skill_width].
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.
        seed: Seed for the action-sampling RNG (None draws one at random).

    Returns:
        The episode statistics, averaged over episodes (`success` among them).
    """
    if not hasattr(agent, 'sample_actions_with_skill'):
        raise TypeError(
            f'{type(agent).__name__} does not expose `sample_actions_with_skill`, so it has no '
            f'skill-conditioned policy to roll out under a fixed skill.'
        )
    rng_seed = np.random.randint(0, 2**32) if seed is None else seed
    actor_fn = supply_rng(agent.sample_actions_with_skill, rng=jax.random.PRNGKey(rng_seed))
    skill = jnp.asarray(skill)

    stats = defaultdict(list)
    for _ in trange(num_eval_episodes, leave=False):
        observation, info = env.reset(options=dict(task_id=task_id, render_goal=False))
        done = False
        while not done:
            action = np.array(actor_fn(observations=observation, skills=skill, temperature=eval_temperature))
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        add_to(stats, flatten(info))

    return {k: np.mean(v) for k, v in stats.items()}


def evaluate_value_selected_skill(
    agent,
    env,
    skills,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    skill_horizon=10,
    skill_ids=None,
    eval_temperature=0,
    eval_gaussian=None,
    seed=None,
):
    """Evaluate a skill-conditioned agent under a value-greedy high-level selector.

    Every `skill_horizon` env steps the skill is reselected greedily under the
    agent's own learned value function,

        z* = argmax_z V(s, z, g),

    and pi(. | ., z*) is then executed for the next `skill_horizon` steps. Unlike
    `evaluate_skill` (which pins one skill for the whole episode and never shows
    the policy the goal), the goal enters through the selector, so this measures
    a single deployable hierarchical policy rather than a best-case skill oracle.

    Args:
        agent: Skill-conditioned agent implementing `skill_values` and
            `sample_actions_with_skill`.
        env: Environment.
        skills: The full candidate skill set, shape [K, skill_width], in the same
            order as the values `agent.skill_values` returns.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        skill_horizon: Number of env steps a selected skill is held for.
        skill_ids: Optional subset of `skills` indices the selector may choose
            from (default: all of them).
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.
        seed: Seed for the action-sampling RNG, and for the env (None draws one at
            random). Note that `evaluate_skill` does not seed the env, so its episodes
            are not the same draws as these.

    Returns:
        The episode statistics averaged over episodes (`success` among them), plus
        `skill_selection_counts`: how often each candidate skill was selected,
        indexed by position in `skill_ids` (not by absolute skill id). The counts
        are per *decision*, so longer episodes contribute proportionally more.
    """
    for hook in ('skill_values', 'sample_actions_with_skill'):
        if not hasattr(agent, hook):
            raise TypeError(
                f'{type(agent).__name__} does not expose `{hook}`, so its skills cannot be '
                f'selected by value. See agents/empowerment_skill.py for the reference hooks.'
            )
    if skill_horizon < 1:
        raise ValueError(f'skill_horizon must be at least 1, got {skill_horizon}.')
    if num_eval_episodes < 1:
        raise ValueError(f'num_eval_episodes must be at least 1, got {num_eval_episodes}.')

    rng_seed = np.random.randint(0, 2**32) if seed is None else seed
    actor_fn = supply_rng(agent.sample_actions_with_skill, rng=jax.random.PRNGKey(rng_seed))

    num_skills = len(skills)
    candidates = np.arange(num_skills) if skill_ids is None else np.asarray(skill_ids, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError('skill_ids is empty; the selector needs at least one candidate skill.')
    # Explicit range check: negative ids would otherwise wrap to the end of the set.
    if candidates.min() < 0 or candidates.max() >= num_skills:
        raise ValueError(f'skill_ids must all be in [0, {num_skills}), got {candidates.tolist()}.')
    candidate_skills = jnp.asarray(skills)[candidates]

    # Reproducibility takes three separate seedings: the stabilizing action samples
    # here, the qpos/qvel draw via `reset(seed=...)` on the first episode below, and
    # ogbench's initial-position noise, which comes from the *global* numpy RNG and so
    # must be seeded by the caller (eval_skill_value_policy.py does).
    env.action_space.seed(int(rng_seed) % (2**31))

    # Checked once up front rather than per step / per episode.
    probe_obs, probe_info = env.reset(options=dict(task_id=task_id, render_goal=False))
    if probe_info.get('goal') is None:
        raise ValueError(
            f'env.reset(task_id={task_id}) returned no `goal` in info; the value-based skill '
            f'selector needs the goal observation to score skills.'
        )
    num_values = np.asarray(
        agent.skill_values(observations=jnp.asarray(probe_obs), goals=jnp.asarray(probe_info['goal']))
    ).shape[-1]
    if num_values != num_skills:
        raise ValueError(
            f'agent.skill_values returned {num_values} values but {num_skills} skills were passed '
            f'in; they must be the same set, in the same order.'
        )
    stats = defaultdict(list)
    selection_counts = np.zeros(len(candidates), dtype=np.int64)
    for episode in trange(num_eval_episodes, leave=False):
        reset_kwargs = dict(options=dict(task_id=task_id, render_goal=False))
        if episode == 0:
            # Seeds the env's `np_random` (the qpos/qvel draw); later resets continue
            # that stream, so seeding the first one fixes the whole sequence.
            reset_kwargs['seed'] = int(rng_seed) % (2**31)
        observation, info = env.reset(**reset_kwargs)
        goal = jnp.asarray(info['goal'])

        done = False
        step = 0
        switches = 0
        zi = None
        while not done:
            if step % skill_horizon == 0:
                # V for every candidate skill at the current state, restricted to the
                # allowed subset. Deterministic: no rng enters the value computation.
                values = np.asarray(agent.skill_values(observations=jnp.asarray(observation), goals=goal))
                new_zi = int(values[candidates].argmax())
                switches += int(zi is not None and new_zi != zi)
                zi = new_zi
                selection_counts[zi] += 1

            action = np.array(
                actor_fn(observations=observation, skills=candidate_skills[zi], temperature=eval_temperature)
            )
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        # The colliding keys would come from the *step* info that gets merged into
        # `stats`, so this can only be checked once an episode has actually run.
        info_stats = flatten(info)
        if episode == 0:
            for key in ('episode_length', 'skill_switches'):
                if key in info_stats:
                    raise ValueError(
                        f'env info already reports `{key}`; it would collide with the selector stat.'
                    )
        add_to(stats, info_stats)
        stats['episode_length'].append(step)
        stats['skill_switches'].append(switches)

    out = {k: np.mean(v) for k, v in stats.items()}
    out['skill_selection_counts'] = selection_counts.tolist()
    return out
