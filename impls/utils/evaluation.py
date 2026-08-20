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
