"""Evaluation for online runs: random-task episodes plus skill-usage diagnostics.

Mirrors JaxGCRL's evaluator rather than OGBench's per-task loop: each episode
resets the env with a *random* task (`env.reset()` without `task_id`), the agent
acts deterministically (`eval_temperature=0`; a skill controller takes the argmax
skill and holds it for `skill_commitment_k` steps through the
`init_eval_state` / `sample_actions_with_state` hooks), and metrics are averaged
over episodes: `success` (the env's final-step `info['success']`), `episode_return`,
`episode_length`. For skill controllers the per-episode skill sequence is kept so
the caller can log the skill-usage histogram / entropy and the skill-colored
trajectory plot (ports of the JaxGCRL diagnostics).
"""

from collections import defaultdict

import jax
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

from utils.evaluation import supply_rng  # noqa: E402
from utils.online_env import env_agent_xy, env_goal_xy  # noqa: E402


def evaluate_online(
    agent,
    env,
    config,
    num_eval_episodes=20,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0.0,
):
    """Roll out `num_eval_episodes + num_video_episodes` random-task episodes.

    Returns:
        stats: dict of per-episode metrics averaged over the non-video episodes.
        trajs: list of per-episode dicts with `xy` (T, 2) or None, `skills` (T,) or
            None, `goal_xy`, `success`.
        renders: list of (T, H, W, C) uint8 videos for the video episodes.
    """
    use_eval_state = hasattr(agent, 'init_eval_state') and hasattr(agent, 'sample_actions_with_state')
    if use_eval_state:
        actor_fn = supply_rng(agent.sample_actions_with_state, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    else:
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    stats = defaultdict(list)
    trajs = []
    renders = []
    for i in range(num_eval_episodes + num_video_episodes):
        should_render = i >= num_eval_episodes
        observation, info = env.reset(options=dict(render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        goal_xy = env_goal_xy(env)
        agent_state = agent.init_eval_state() if use_eval_state else None

        done = False
        step = 0
        render = []
        xy_list, skill_list = [], []
        ep_return = 0.0
        while not done:
            xy = env_agent_xy(env)
            if xy is not None:
                xy_list.append(xy)
            if use_eval_state:
                action, agent_state = actor_fn(
                    observations=observation, goals=goal, temperature=eval_temperature, agent_state=agent_state
                )
                if 'skill' in agent_state:
                    skill_list.append(int(np.asarray(agent_state['skill'])))
            else:
                action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            ep_return += float(reward)

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)
            observation = next_observation

        xy = env_agent_xy(env)
        if xy is not None:
            xy_list.append(xy)
        success = float(info.get('success', 0.0))  # final-step success == "reached the goal at any step" under terminate_at_goal
        trajs.append(
            dict(
                xy=np.stack(xy_list) if xy_list else None,
                skills=np.asarray(skill_list, dtype=np.int64) if skill_list else None,
                goal_xy=goal_xy,
                success=success,
            )
        )
        if not should_render:
            stats['success'].append(success)
            stats['episode_return'].append(ep_return)
            stats['episode_length'].append(step)
        else:
            renders.append(np.array(render))

    stats = {k: float(np.mean(v)) for k, v in stats.items()}
    return stats, trajs, renders


def skill_usage_stats(skill_sequences, skill_commitment_k, num_skills):
    """Skill-choice distribution over the reselection steps (every `k`-th step), as in JaxGCRL.

    Returns `(stats, counts)` where `stats` has `skill_entropy`, `skill_max_frac`,
    `skill_active_count` and `counts` is the per-skill histogram.
    """
    chosen = np.concatenate([np.asarray(s)[0::skill_commitment_k] for s in skill_sequences if s is not None])
    counts = np.bincount(chosen.astype(int), minlength=num_skills).astype(np.float64)
    fracs = counts / max(counts.sum(), 1.0)
    nz = fracs > 0
    stats = dict(
        skill_entropy=float(-(fracs[nz] * np.log(fracs[nz])).sum()),
        skill_max_frac=float(fracs.max()),
        skill_active_count=float((counts > 0).sum()),
    )
    return stats, counts


def plot_skill_colored_trajectory(xy, skills, num_skills, goal_xy=None, title=''):
    """One 2D trajectory with each segment colored by the active skill (JaxGCRL port).

    `xy`: (T, 2) agent positions; `skills`: (T,) skill index in effect at each
    step (the last position has no outgoing segment). Returns a matplotlib Figure.
    """
    xy = np.asarray(xy, dtype=np.float32)
    skills = np.asarray(skills).astype(int)
    n_seg = min(len(xy) - 1, len(skills))
    cmap = plt.get_cmap('tab20' if num_skills <= 20 else 'gist_ncar', num_skills)
    norm = BoundaryNorm(np.arange(-0.5, num_skills + 0.5, 1.0), num_skills)

    fig, ax = plt.subplots(figsize=(6, 6))
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)[:n_seg]
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(skills[:n_seg])
    lc.set_linewidth(2.0)
    ax.add_collection(lc)

    ax.scatter(xy[0, 0], xy[0, 1], c='black', marker='o', s=60, zorder=5, label='start')
    ax.scatter(xy[-1, 0], xy[-1, 1], c='black', marker='s', s=60, zorder=5, label='end')
    pts = [xy]
    if goal_xy is not None:
        goal_xy = np.asarray(goal_xy, dtype=np.float32)
        ax.scatter(goal_xy[0], goal_xy[1], c='red', marker='*', s=220, zorder=6, label='goal')
        pts.append(goal_xy[None])
    allpts = np.concatenate(pts, axis=0)
    lo, hi = allpts.min(axis=0) - 2.0, allpts.max(axis=0) + 2.0
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.arange(num_skills), fraction=0.046, pad=0.04
    )
    cbar.set_label('skill')
    fig.tight_layout()
    return fig
