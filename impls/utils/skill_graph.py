"""Offline subgoal-graph planning on top of a skill-conditioned occupancy value.

`empowerment_skill` learns `V^z(s+ | s)`, the discounted state-occupancy density of
pi(.|.,z). Two properties decide how it can be used:

1.  It is a log-hitting-time metric: if the skill reaches `g` at step T and dwells,
    V ~ (1 - gamma) gamma^T rho, so `-log V` is additive along a path and shortest
    path is the right way to compose it.
2.  Its horizon is 1/(1 - gamma) ~ 100 steps at gamma = 0.99. Measured on
    antmaze-medium the cost separates pairs up to ~3 maze cells and is flat beyond,
    while benchmark goals sit 500-700 steps away, so `argmax_z V(s, z, g)` against a
    task goal is querying the value function outside its informative range.

The planner therefore only ever asks "how good is this skill for reaching a state
~1 cell away", and recovers the long horizon by Dijkstra over a graph of dataset
states. Maze topology comes from the graph rather than the network, which matters:
the learned cost tracks straight-line displacement (corr 0.42) better than geodesic
distance (corr 0.35) and will propose an edge through a wall if allowed to span one.

Beyond the `skill_set` / `skill_values` / `sample_actions_with_skill` hooks that
`eval_skill_value_policy.py` requires, planning needs `value_goal_embeddings` and
`skill_values_from_goal_embeddings` (see `agents/empowerment_skill.py`). Splitting
the goal side out keeps replanning cheap: the graph's states are embedded once at
build time, and each replan costs one phi evaluation.
"""
import heapq

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

REQUIRED_HOOKS = ('value_goal_embeddings', 'skill_values_from_goal_embeddings')


def missing_hooks(agent_class):
    """Hooks from the planning contract that `agent_class` does not implement."""
    return [h for h in REQUIRED_HOOKS if not hasattr(agent_class, h)]


class SkillValueCalibrator:
    """Puts every skill's log-occupancy on a common measure before they are compared.

    `V^z(. | s)` is pinned in scale only at `s+ = s`, by the self-loss; the fitted
    normaliser drifts elsewhere and by different amounts per skill (1.2 nats across
    skills on antmaze-medium, against a ~1.5 nat real signal). A raw `argmax_z` then
    mostly reports which skill has the largest partition -- a quantity independent of
    both `s` and `g` -- so the selector returns a constant and never switches.

    Subtracting each skill's empirical log-partition over a fixed reference set `R`
    drawn from the offline data fixes that:

        Vhat(s, z, g) = f(s, z, g) - logsumexp_{g' in R} f(s, z, g') + log|R|.

    This estimates `E_{g ~ p_data}[V^z(g|s)]`, not `int V^z dg`, so it does not
    converge to 0 for a perfectly normalised V. That is fine for ranking skills at a
    fixed state, since it is the same functional of `V^z` for every `z`, but it does
    also remove the legitimate advantage of a skill steering into a data-dense region.
    `calibrate=False` keeps the trade measurable.
    """

    def __init__(self, agent, reference_observations, chunk_size=256):
        self.agent = agent
        self.chunk_size = int(chunk_size)
        self.reference = jnp.asarray(np.asarray(reference_observations))
        self._ref_psi = agent.value_goal_embeddings(self.reference)
        self._log_n_ref = float(np.log(self.reference.shape[0]))

    def log_partition(self, observations):
        """log E_{g ~ R}[V^z(g | s)] for each (observation, skill)  ->  [B, K]."""
        obs = jnp.asarray(np.asarray(observations))
        out = []
        for i in range(0, obs.shape[0], self.chunk_size):
            logits = self.agent.skill_values_from_goal_embeddings(
                obs[i:i + self.chunk_size], self._ref_psi
            )  # [b, |R|, K]
            out.append(logsumexp(logits, axis=1) - self._log_n_ref)
        return jnp.concatenate(out, axis=0)


class SkillGraphPlanner:
    """Shortest-path subgoal planning over dataset states, scored by the skill value.

    Build time:
      1. sample `num_nodes` states from the offline dataset as graph nodes;
      2. score every ordered pair, `c(i, j) = -max_z Vhat(s_i, z, s_j)`;
      3. keep an edge i -> j only if it is within the `max_degree` cheapest
         out-edges of i, mutual (i is also among j's `max_degree` cheapest,
         tested *before* thresholding), and below a cost threshold set at
         `edge_quantile` of the pairwise cost distribution.

    Step 3 needs both halves. The threshold confines the graph to the range where the
    metric is trustworthy (at the 2% quantile 97% of admitted edges span <= 2 maze
    cells, versus 43% at the median), but it only bounds the false-positive *rate*,
    and shortest path actively seeks the exceptions -- hence mutuality.

    Query time, once per `skill_horizon` steps:
      * `D[j]`, the shortest-path cost from node `j` to the task goal, is computed
        once per goal by Dijkstra on the reversed graph and cached;
      * the entry node is `argmin_j w(s, j) + D[j]` over the `entry_degree` nearest
        nodes *that have a finite path to the goal*, further restricted to those
        within the edge threshold when any qualifies;
      * the waypoint is the node `subgoal_hops - 1` edges past the entry node,
        stopping early if the next hop is the goal;
      * the executed skill is `argmax_z Vhat(s, z, waypoint)`, or a sample from
        `softmax(Vhat / temperature)` when `temperature > 0`.
    """

    def __init__(
        self,
        agent,
        dataset_observations,
        *,
        num_nodes=750,
        num_reference=512,
        edge_quantile=0.02,
        max_degree=64,
        mutual_edges=True,
        hop_cost=1.0,
        subgoal_hops=2,
        goal_degree=16,
        entry_degree=5,
        stall_patience=2,
        progress_margin=0.1,
        temperature=0.0,
        calibrate=True,
        chunk_size=256,
        seed=0,
    ):
        missing = missing_hooks(type(agent))
        if missing:
            raise ValueError(
                f'agent {type(agent).__name__} cannot be used for graph planning: it is missing '
                f'{", ".join(chr(96) + h + chr(96) for h in missing)}. See utils/skill_graph.py.'
            )
        if num_nodes < 2:
            raise ValueError(f'num_nodes must be >= 2, got {num_nodes}.')
        if calibrate and num_reference < 1:
            raise ValueError(f'num_reference must be >= 1 when calibrating, got {num_reference}.')
        if not 0.0 < edge_quantile <= 1.0:
            raise ValueError(f'edge_quantile must be in (0, 1], got {edge_quantile}.')
        if max_degree < 1:
            raise ValueError(f'max_degree must be >= 1, got {max_degree}.')
        if subgoal_hops < 1:
            raise ValueError(f'subgoal_hops must be >= 1, got {subgoal_hops}.')
        if goal_degree < 1:
            raise ValueError(f'goal_degree must be >= 1, got {goal_degree}.')
        if entry_degree < 1:
            raise ValueError(f'entry_degree must be >= 1, got {entry_degree}.')
        if stall_patience < 0:
            raise ValueError(f'stall_patience must be >= 0, got {stall_patience}.')
        if progress_margin < 0:
            raise ValueError(f'progress_margin must be >= 0, got {progress_margin}.')
        if temperature < 0:
            raise ValueError(f'temperature must be >= 0, got {temperature}.')
        if hop_cost < 0:
            # Dijkstra never revisits a settled node, so negative weights corrupt it.
            raise ValueError(f'hop_cost must be >= 0, got {hop_cost}.')

        self.agent = agent
        self.hop_cost = float(hop_cost)
        self.mutual_edges = bool(mutual_edges)
        self.edge_quantile = float(edge_quantile)
        self.max_degree = int(max_degree)
        self.subgoal_hops = int(subgoal_hops)
        self.goal_degree = int(goal_degree)
        self.entry_degree = int(entry_degree)
        self.stall_patience = int(stall_patience)
        self.progress_margin = float(progress_margin)
        # Recorded even when calibration is off: it still shifts the node draw, since
        # `rng.choice` is not prefix-stable in `num_reference`.
        self.num_reference = int(num_reference)
        self.temperature = float(temperature)
        self.calibrate = bool(calibrate)
        self.chunk_size = int(chunk_size)
        self._rng = np.random.default_rng(seed)

        obs = np.asarray(dataset_observations)
        rng = np.random.default_rng(seed)
        if num_nodes + num_reference > len(obs):
            raise ValueError(
                f'dataset has {len(obs)} states, too few for {num_nodes} nodes + '
                f'{num_reference} reference states.'
            )
        # Disjoint draws: a node that also anchors the partition estimate would have its
        # own log-partition pulled up by its near-zero self-cost.
        picked = rng.choice(len(obs), num_nodes + num_reference, replace=False)
        self.nodes = obs[picked[:num_nodes]]
        self.num_nodes = int(num_nodes)

        self.calibrator = (
            SkillValueCalibrator(agent, obs[picked[num_nodes:]], chunk_size=chunk_size)
            if calibrate else None
        )
        self._node_psi = agent.value_goal_embeddings(jnp.asarray(self.nodes))
        self._node_log_z = self._log_partition(self.nodes)

        cost = self._cost_to(self.nodes, self._node_psi, self._node_log_z)  # [N, N]
        np.fill_diagonal(cost, np.inf)  # a self-edge is a free way to make no progress
        self.threshold = float(np.quantile(cost[np.isfinite(cost)], edge_quantile))

        # Every edge weight is measured against the cheapest edge leaving the *same*
        # row, which is what makes the weights additive. `cost(i, j)` carries a
        # calibration term `logZ(s_i)` indexed by the source, so a raw sum along
        # i -> j -> k accumulates one such term per node visited: it would rank routes
        # partly by how little each waypoint's occupancy overlaps the data, not by
        # travel time. Subtracting the row minimum removes exactly that, keeps the
        # within-row differences that carry reachability, and makes the weights
        # non-negative as Dijkstra requires.
        self._row_floor = cost.min(axis=1)  # [N]

        # Mutual filtering: i -> j survives only if i is also among j's cheapest.
        # Thresholding sets the false-positive rate but does not stop shortest path from
        # seeking out the exceptions -- a spurious 6-cell edge replaces three real hops
        # while saving two `hop_cost` charges, so it wins every competition it enters
        # (33% of paths through a plain kNN graph contained a 4+ cell hop). Mutuality
        # removes those structurally, since the error is asymmetric, taking that to 0%
        # at the cost of mean degree 16 -> 6.6 and pairwise reachability 0.98 -> 0.77.
        #
        # Mutuality is tested on the un-thresholded sets, then the threshold is applied
        # in *both* directions. Testing membership on already-thresholded sets is much
        # stricter than intended (reachability 0.13 instead of 0.77), and thresholding
        # one direction only readmits the per-source offset the row normalisation exists
        # to remove, leaving sinks the reverse Dijkstra routes into.
        neighbours = [set(int(j) for j in np.argsort(cost[i])[:max_degree]) for i in range(self.num_nodes)]
        self.edges = []
        for i in range(self.num_nodes):
            keep = [
                j for j in neighbours[i]
                if cost[i, j] <= self.threshold
                and (not self.mutual_edges
                     or (i in neighbours[j] and cost[j, i] <= self.threshold))
            ]
            self.edges.append([(j, float(self._weight(cost[i, j], self._row_floor[i]))) for j in keep])
        self.num_edges = sum(len(e) for e in self.edges)

        self._reverse = [[] for _ in range(self.num_nodes)]
        for i, out in enumerate(self.edges):
            for j, w in out:
                self._reverse[j].append((i, w))

        self._goal_cache = None
        self._last_cost = np.inf   # best cost-to-go seen since the last re-baseline
        self._stalled = 0          # consecutive decisions without progress
        self._last_skill = -1
        self._last_branch = None   # which target the previous decision priced

    # ── value plumbing ────────────────────────────────────────────────────────

    def _log_partition(self, observations):
        if self.calibrator is None:
            return np.zeros((len(observations), 1), dtype=np.float32)
        return np.asarray(self.calibrator.log_partition(observations))

    def _cost_to(self, observations, goal_psi, log_z):
        """`-max_z Vhat(s, z, g)` for every (observation, goal) pair  ->  [B, G]."""
        obs = jnp.asarray(np.asarray(observations))
        log_z = jnp.asarray(log_z)[:, None, :]  # [B, 1, K], broadcasts over goals
        out = []
        for i in range(0, obs.shape[0], self.chunk_size):
            logits = self.agent.skill_values_from_goal_embeddings(
                obs[i:i + self.chunk_size], goal_psi
            )  # [b, G, K]
            out.append(jnp.max(logits - log_z[i:i + self.chunk_size], axis=-1))
        cost = -np.asarray(jnp.concatenate(out, axis=0))
        # Every cost comes through here, and NaN is invisible downstream: it would be
        # dropped from the edge quantile, make nodes unreachable, and silently reduce
        # the planner to the greedy selector.
        if not np.isfinite(cost).all():
            raise FloatingPointError(
                f'{np.count_nonzero(~np.isfinite(cost))} of {cost.size} skill values are not '
                f'finite; the value network or its log-partition produced NaN/inf.'
            )
        return cost

    def _weight(self, cost, floor):
        """Edge weight: cost relative to the cheapest edge out of the same source."""
        return np.maximum(np.asarray(cost, dtype=np.float64) - floor, 0.0) + self.hop_cost

    # ── planning ──────────────────────────────────────────────────────────────

    def _plan_to_goal(self, goal):
        """Dijkstra on the reversed graph: `D[j]` = shortest cost from node j to goal."""
        goal = np.asarray(goal)
        goal_psi = self.agent.value_goal_embeddings(jnp.asarray(goal)[None])
        node_to_goal = self._cost_to(self.nodes, goal_psi, self._node_log_z)[:, 0]  # [N]

        # The goal is attached by k-nearest, not by the edge threshold: it is one
        # specific state, and its cost to every node can sit above a threshold
        # calibrated on node-to-node pairs (on task 1 that left 0 of 750 nodes able to
        # reach it). The last hop may be long, but it is re-decided at every replan.
        dist = np.full(self.num_nodes, np.inf)
        succ = np.full(self.num_nodes, -1, dtype=np.int64)  # -1 == step to the goal itself
        queue = []
        for j in np.argsort(node_to_goal)[:self.goal_degree]:
            j = int(j)
            dist[j] = float(self._weight(node_to_goal[j], self._row_floor[j]))
            heapq.heappush(queue, (dist[j], j))
        while queue:
            d, j = heapq.heappop(queue)
            if d > dist[j]:
                continue
            for i, w in self._reverse[j]:
                if d + w < dist[i]:
                    dist[i] = d + w
                    succ[i] = j
                    heapq.heappush(queue, (dist[i], i))
        return dict(goal=goal, goal_psi=goal_psi, dist=dist, succ=succ,
                    num_reachable=int(np.isfinite(dist).sum()))

    def reset(self):
        """Clear per-episode controller state. Call at every `env.reset()`.

        The stall detector cannot infer an episode boundary on its own, since every
        episode of a task shares one goal. Left unreset, `_last_cost` carries the
        near-goal value from the previous episode into the start of the next one and
        the detector trips spuriously.
        """
        self._last_cost, self._stalled, self._last_skill = np.inf, 0, -1
        self._last_branch = None

    def plan(self, goal):
        """Compute and cache the shortest-path costs to `goal`. Returns the plan dict."""
        goal = np.asarray(goal)
        if self._goal_cache is None or not np.array_equal(self._goal_cache['goal'], goal):
            self._goal_cache = self._plan_to_goal(goal)
            self.reset()
        return self._goal_cache

    def select(self, observation, goal):
        """Choose the skill to execute for the next chunk.

        Returns `(skill_index, info)`, where `info` carries `waypoint_node` (-1 when
        the goal itself is targeted), `direct` (the goal was in range and taken as the
        endgame), `fallback` (the goal was targeted only because no graph node reaches
        it), `reachable`, and `in_range` (entry used a node within the range the cost
        was validated on).
        """
        plan = self.plan(goal)
        obs = np.asarray(observation)[None]
        log_z = self._log_partition(obs)

        to_nodes = self._cost_to(obs, self._node_psi, log_z)[0]      # [N]
        to_goal = float(self._cost_to(obs, plan['goal_psi'], log_z)[0, 0])

        # Entry is restricted to the `entry_degree` nearest-scoring nodes, filtered by
        # reachability *first*. Both halves matter. An unrestricted `argmin_j w(s, j) +
        # D[j]` is an adversarial search against the metric's own error rate: it
        # specifically rewards a node that *looks* near the agent while *being* near the
        # goal, which is what a false positive on the first term is. Traces showed the
        # ant handed a waypoint 16 units away, adjacent to the goal, with the range
        # check reporting 1.00. And ranking by proximity before reachability is a trap:
        # mutual filtering cuts reachability to ~0.77, so the nearest few nodes often
        # have no path to the goal, and the plan was dropped for the greedy fallback on
        # 91% of decisions.
        entry_floor = float(to_nodes.min())
        usable = np.flatnonzero(np.isfinite(plan['dist']))
        reachable = usable.size > 0
        if reachable:
            k = min(self.entry_degree, usable.size)
            nearest = usable[np.argpartition(to_nodes[usable], k - 1)[:k]]
            in_range = bool((to_nodes[nearest] <= self.threshold).any())
            candidates = nearest[to_nodes[nearest] <= self.threshold] if in_range else nearest
            entry = self._weight(to_nodes[candidates], entry_floor)
            total = entry + plan['dist'][candidates]
            pick = int(np.argmin(total))
            best = int(candidates[pick])
            route_cost = float(total[pick])
        else:
            in_range, best, route_cost = False, -1, np.inf

        if not reachable:
            # Nothing to compose, so fall back to the value-greedy selector. With
            # `goal_degree` nodes attached unconditionally this is a guard, not an
            # expected branch, and it is reported separately from `direct`.
            waypoint_psi, waypoint_node, direct, fallback = plan['goal_psi'], -1, False, True
            progress_cost = to_goal
        elif to_goal <= self.threshold and float(self._weight(to_goal, entry_floor)) <= route_cost:
            # The goal is in range and no worse than the best graph route: the endgame
            # should not be forced through a waypoint.
            waypoint_psi, waypoint_node, direct, fallback = plan['goal_psi'], -1, True, False
            progress_cost = to_goal
        else:
            node = best
            for _ in range(self.subgoal_hops - 1):
                nxt = int(plan['succ'][node])
                if nxt < 0:  # the next hop is the goal; stop here rather than overshoot
                    break
                node = nxt
            waypoint_psi = self._node_psi[node:node + 1]
            waypoint_node, direct, fallback = node, False, False
            progress_cost = float(to_nodes[best] + plan['dist'][best])

        values = np.asarray(
            self.agent.skill_values_from_goal_embeddings(jnp.asarray(obs), waypoint_psi)
        )[0, 0] - log_z[0]

        # Progress is measured in the planner's own cost-to-go. `_last_cost` is the best
        # cost since the last re-baseline, so "no progress" means failing to beat that
        # best -- stricter than beating the previous decision, deliberately, since an
        # agent oscillating between two states makes progress on alternate steps under
        # the looser test. After `stall_patience` such decisions, mask the last skill.
        #
        # `progress_cost` is the *raw* cost to whatever is being targeted, not the
        # row-normalised route total: that total is a step function, since its entry
        # term is pinned at `hop_cost` whenever entry is the nearest node, so it only
        # moves when the agent crosses into another node's cell.
        #
        # This is the deadlock a memoryless argmax cannot escape: in a deterministic
        # environment a skill that fails to move the agent leaves the state, and so the
        # argmax, unchanged. Traces show the ant frozen for 900 steps on one skill while
        # other skills would have moved it 4+ units from that same state.
        total_cost = progress_cost
        # The branch is chosen by a threshold gate, not by whichever route is cheaper,
        # so crossing that boundary steps the signal by an unbounded amount (0.72 nats
        # for a 0.15-unit move, against a 0.1 margin). Comparisons are only meaningful
        # within a branch, so a change of branch re-baselines.
        branch = 'direct' if (direct or fallback) else 'graph'
        if branch != self._last_branch:
            self._last_cost, self._stalled = np.inf, 0
        self._last_branch = branch
        if self.stall_patience:
            # A margin, not an epsilon: `progress_cost` is continuous in the state, so
            # jitter would read as progress under a 1e-6 test and the detector would
            # never fire.
            self._stalled = (
                0 if total_cost < self._last_cost - self.progress_margin else self._stalled + 1
            )
            self._last_cost = min(total_cost, self._last_cost)
            if (self._stalled >= self.stall_patience and len(values) > 1
                    and 0 <= self._last_skill < len(values)):
                values = values.copy()
                values[self._last_skill] = -np.inf
                self._stalled = 0
                self._last_cost = np.inf

        skill = self._choose(values)
        self._last_skill = skill
        return skill, dict(
            waypoint_node=waypoint_node, direct=direct, fallback=fallback,
            reachable=reachable, in_range=in_range,
        )

    def _choose(self, values):
        """Take the argmax of `Vhat`, or sample `softmax(Vhat / temperature)` if asked.

        Sampling is an alternative escape from the argmax deadlock, but it is off by
        default because it loses end to end: temperature 0 / 0.5 / 1.0 scored
        0.100 / 0.040 / 0.020 on antmaze-medium-navigate.
        """
        if self.temperature <= 0:
            return int(values.argmax())
        logits = (values - values.max()) / self.temperature
        p = np.exp(logits)
        return int(self._rng.choice(len(p), p=p / p.sum()))

    def summary(self):
        """Graph statistics and the construction knobs needed to reproduce them."""
        return dict(
            num_nodes=self.num_nodes,
            num_reference=self.num_reference,
            num_edges=self.num_edges,
            edge_quantile=self.edge_quantile,
            max_degree=self.max_degree,
            mean_degree=self.num_edges / self.num_nodes,
            edge_threshold=self.threshold,
            hop_cost=self.hop_cost,
            mutual_edges=self.mutual_edges,
            mean_row_floor=float(self._row_floor.mean()),
            subgoal_hops=self.subgoal_hops,
            goal_degree=self.goal_degree,
            entry_degree=self.entry_degree,
            stall_patience=self.stall_patience,
            progress_margin=self.progress_margin,
            temperature=self.temperature,
            calibrated=self.calibrate,
        )
