"""
Skill-Restricted IQL (SR-IQL): a fully offline goal-conditioned high level over frozen skills.

Given K frozen skill policies pi_z(a | s) (an `empowerment_skill` checkpoint, used as a
function only -- never executed in the environment during training) and an offline dataset
D = {(s, a, s')} with no skill labels, learn a selector z*(s, g).

The observation that organises the method: because the skills are fixed functions, "offline
RL over skills" is the same problem as "offline goal-conditioned RL at the low level with the
policy class restricted to {mu_z}". So out-of-distribution is defined in ACTION space, where
the data lives: skill z is in-distribution at s iff its action mu_z(s) is covered by the
behaviour policy. Every learned quantity below is fit only at data (s, a):

    w_z(s, a)   = exp(-||a - mu_z(s)||^2 / 2 sigma_k^2)          skill kernel, all K at once
    L_Q         = E_D (r + gamma m V(s', g) - Q(s, a, g))^2         gciql critic, data actions only
    L_Qz        = E_D sum_z w_z (Qbar(s, a, g) - Qz(s, g)[z])^2     per-skill value by weighted regression
                                                                    (minimiser = E_beta[w_z Q] / E_beta[w_z])
    L_C         = E_D sum_z (w_z - C(s)[z])^2                       coverage C(s)[z] -> E_beta[w_z]
    L_V         = E_D sum_z c(z|s) l2^tau(Qz(s, g)[z] - V(s, g))    IQL expectile over COVERED skills,
                                                                    c(z|s) = C(s)[z] / sum_z' C(s)[z']
    z*(s, g)    = argmax_z [ Qz(s, g)[z] + alpha log c(z|s) ]        exact argmax, no actor

Compared with `skill_bc_relabel_controller` (same data, same IQL critic, same skills) the
difference is what evidence links a skill to a transition: one data step weighted by action
similarity and bootstrapped, instead of a 10-step window hard-labelled by BC likelihood and
credited with the window's endpoint. See impls/docs/grounded_option_qi_proposal.md (v3).

The kernel width is set from the data unless given: sigma_k^2 = E_D[min_z ||a - mu_z(s)||^2],
so "one sigma" is the typical distance from a behaviour action to the nearest skill.
`kernel='hard'` is the parameter-free limit w_z = 1[z = argmin_z ||a - mu_z(s)||].

Eval: `init_eval_state` / `sample_actions_with_state` hold z for `skill_horizon` steps
(1 = re-pick every step, the choice consistent with what V assumes). `selector='qmu'`
swaps the in-sample Qz for the off-support Q(s, mu_z(s), g) -- the E0 ablation.
"""

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.empowerment_skill import EmpowermentAgent
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCValue
from utils.skill_checkpoint import load_frozen_skill_agent


class SkillHead(nn.Module):
    """f(s[, g]) -> R^K: one output per skill."""

    hidden_dims: Any
    num_skills: int
    layer_norm: bool = True
    gc_encoder: nn.Module = None

    def setup(self):
        self.net = MLP((*self.hidden_dims, self.num_skills), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals=None):
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        return self.net(jnp.concatenate(inputs, axis=-1))


class SkillRestrictedIQLAgent(flax.struct.PyTreeNode):
    """SR-IQL agent.

    Fields:
        rng: PRNG key.
        network: `TrainState` over `critic`, `target_critic`, `value`, `qz`, `cov`.
        skill_agent: the frozen `empowerment_skill` agent (policy used as a function only).
            A plain pytree field, so checkpoints of this agent carry a full copy of it.
        kernel_sigma: jnp scalar, the kernel width actually used (set by `prepare_datasets`
            from the data unless `config['kernel_sigma']` is given).
        config: static configuration.
    """

    rng: Any
    network: Any
    skill_agent: Any
    kernel_sigma: Any
    config: Any = nonpytree_field()

    # -- skill kernel --------------------------------------------------------------------

    def skill_means(self, observations):
        """mu_z(s) for every skill: [B, K, A] (clipped like executed actions)."""
        K = int(self.config['num_skills'])
        B = observations.shape[0]
        eye = jnp.eye(K)

        def one(k):
            onehot = jnp.broadcast_to(eye[k], (B, K))
            dist = self.skill_agent.network.select('policy')(observations, onehot)
            return jnp.clip(dist.mode(), -1.0, 1.0)

        return jnp.moveaxis(jax.lax.map(one, jnp.arange(K)), 0, 1)  # [B, K, A]

    def skill_dist2(self, observations, actions):
        """||a - mu_z(s)||^2 for every skill: [B, K]."""
        means = self.skill_means(observations)
        return jnp.sum((actions[:, None, :] - means) ** 2, axis=-1)

    def skill_weights(self, observations, actions):
        """w_z(s, a) in (0, 1] for every skill: [B, K]."""
        dist2 = self.skill_dist2(observations, actions)
        if self.config['kernel'] == 'hard':
            return jnp.eye(int(self.config['num_skills']))[jnp.argmin(dist2, axis=-1)], dist2
        sigma2 = jnp.maximum(self.kernel_sigma, 1e-6) ** 2
        return jnp.exp(-dist2 / (2.0 * sigma2)), dist2

    # -- coverage -------------------------------------------------------------------------

    def coverage(self, observations, params=None):
        """C(s)[z] in (0, 1): [B, K]."""
        logits = self.network.select('cov')(observations, params=params)
        return jax.nn.sigmoid(logits)

    def log_coverage_dist(self, observations):
        """log c(z|s), c(z|s) = C(s)[z] / sum_z' C(s)[z']: [B, K]."""
        c = jnp.maximum(self.coverage(observations), 1e-6)
        return jnp.log(c) - jnp.log(c.sum(axis=-1, keepdims=True))

    # -- losses ---------------------------------------------------------------------------

    @staticmethod
    def expectile_loss(diff, expectile):
        weight = jnp.where(diff >= 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def critic_loss(self, batch, grad_params):
        """gciql critic loss, at data actions."""
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        return loss, {'critic_loss': loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min()}

    def skill_losses(self, batch, grad_params, w):
        """L_Qz (weighted regression of per-skill values) and L_C (coverage)."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'])
        q_t = jnp.minimum(q1, q2)  # [B]
        qz = self.network.select('qz')(batch['observations'], batch['value_goals'], params=grad_params)  # [B, K]
        sq = (q_t[:, None] - qz) ** 2
        qz_loss = (w * sq).sum(axis=-1).mean() / (w.sum(axis=-1).mean() + 1e-6)

        c = self.coverage(batch['observations'], params=grad_params)
        c_loss = ((w - c) ** 2).mean()
        return qz_loss, c_loss, {
            'qz_loss': qz_loss,
            'cov_loss': c_loss,
            'qz_mean': qz.mean(),
            'cov_mean': c.mean(),
        }

    def value_loss(self, batch, grad_params):
        """IQL expectile of V onto the per-skill values, weighted by the coverage distribution."""
        qz = jax.lax.stop_gradient(self.network.select('qz')(batch['observations'], batch['value_goals']))
        log_c = jax.lax.stop_gradient(self.log_coverage_dist(batch['observations']))
        c = jnp.exp(log_c)  # rows sum to 1
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        diff = qz - v[:, None]
        loss = (c * self.expectile_loss(diff, self.config['expectile'])).sum(axis=-1).mean()
        ent = -(c * log_c).sum(axis=-1).mean()
        return loss, {
            'value_loss': loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'cov_entropy': ent,
            'cov_max_prob': c.max(axis=-1).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        w, dist2 = self.skill_weights(batch['observations'], batch['actions'])
        w = jax.lax.stop_gradient(w)

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        qz_loss, c_loss, skill_info = self.skill_losses(batch, grad_params, w)
        value_loss, value_info = self.value_loss(batch, grad_params)

        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        for k, v in skill_info.items():
            info[f'skill/{k}'] = v
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        ess = (w.sum(axis=-1) ** 2) / ((w ** 2).sum(axis=-1) + 1e-8)
        info['kernel/ess'] = ess.mean()
        info['kernel/w_mean'] = w.mean()
        info['kernel/w_max'] = w.max(axis=-1).mean()
        info['kernel/min_dist2'] = dist2.min(axis=-1).mean()
        info['kernel/sigma'] = self.kernel_sigma

        # Selector agreement: in-sample Qz vs off-support Q(s, mu_z(s), g).
        scores_qz = self._scores(batch['observations'], batch['value_goals'], 'qz')
        scores_qmu = self._scores(batch['observations'], batch['value_goals'], 'qmu')
        info['selector/agree_qz_qmu'] = (scores_qz.argmax(-1) == scores_qmu.argmax(-1)).mean()
        info['selector/qz_pick_entropy'] = _pick_entropy(scores_qz.argmax(-1), int(self.config['num_skills']))

        loss = critic_loss + qz_loss + c_loss + value_loss
        info['total_loss'] = loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    # -- data-estimated kernel width ------------------------------------------------------

    def prepare_datasets(self, datasets, num_samples=20000, chunk=2048):
        """Set `kernel_sigma` from the data unless the config fixes it. Returns the new agent."""
        if self.config['kernel_sigma'] is not None:
            sigma = float(self.config['kernel_sigma'])
            print(f'[skill_restricted_iql] kernel_sigma={sigma:.4f} (from config)')
            return self.replace(kernel_sigma=jnp.asarray(sigma, jnp.float32))

        dataset = datasets[0]
        n = min(int(num_samples), int(dataset.size))
        idxs = np.random.RandomState(0).choice(int(dataset.size), size=n, replace=False)
        dist2_fn = jax.jit(self.skill_dist2)
        mins, ess_at = [], []
        for start in range(0, n, chunk):
            sub_idxs = idxs[start:start + chunk]
            sub = dataset.sample(len(sub_idxs), idxs=sub_idxs)
            d2 = np.asarray(dist2_fn(jnp.asarray(sub['observations']), jnp.asarray(sub['actions'])))
            mins.append(d2.min(axis=-1))
        mins = np.concatenate(mins)
        sigma2 = float(mins.mean())
        sigma = float(np.sqrt(max(sigma2, 1e-8)))
        print(
            f'[skill_restricted_iql] kernel_sigma={sigma:.4f} from {n} transitions '
            f'(E[min_z ||a-mu_z||^2]={sigma2:.4f}, median={np.median(mins):.4f}, '
            f'kernel={self.config["kernel"]})'
        )
        return self.replace(kernel_sigma=jnp.asarray(sigma, jnp.float32))

    # -- selection ------------------------------------------------------------------------

    def _scores(self, observations, goals, selector):
        """Per-skill scores [B, K] for the given selector, including the coverage term."""
        if selector == 'qz':
            base = self.network.select('qz')(observations, goals)
        elif selector == 'qmu':
            means = self.skill_means(observations)  # [B, K, A]
            K = means.shape[1]

            def one(k):
                q1, q2 = self.network.select('critic')(observations, goals, means[:, k])
                return jnp.minimum(q1, q2)

            base = jax.lax.map(one, jnp.arange(K)).T  # [B, K]
        else:
            raise ValueError(f'unknown selector {selector!r}')
        alpha = float(self.config['coverage_alpha'])
        if alpha != 0.0:
            base = base + alpha * self.log_coverage_dist(observations)
        return base

    def _low_level_actions(self, observations, skills_onehot, seed):
        temperature = self.config['low_temperature']
        if self.skill_agent.config['discrete']:
            temperature = max(float(temperature), 1e-6)
        dist = self.skill_agent.network.select('policy')(observations, skills_onehot, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.skill_agent.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    def _single(self, observations):
        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        return observations.ndim == single_obs_ndim

    @jax.jit
    def skill_values(self, observations, goals):
        """Selector scores aligned with `skill_set()`: [K] or [B, K] (eval_skill_value_policy hook)."""
        single = self._single(observations)
        obs_b = observations[None, ...] if single else observations
        goals_b = goals[None, ...] if single else goals
        scores = self._scores(obs_b, goals_b, self.config['selector'])
        return scores[0] if single else scores

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        actions, _ = self.sample_actions_with_state(observations, goals, None, seed, temperature)
        return actions

    def init_eval_state(self):
        return {'skill': jnp.zeros((), jnp.int32), 'count': jnp.zeros((), jnp.int32)}

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None, temperature=1.0):
        """Greedy skill (held for `skill_horizon` steps), then a = mu_z(s) from the frozen skill."""
        if goals is None:
            raise ValueError('skill_restricted_iql is goal-conditioned; pass `goals`.')
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()
        high_seed, low_seed = jax.random.split(seed)

        single = self._single(observations)
        obs_b = observations[None, ...] if single else observations
        goals_b = goals[None, ...] if single else goals

        scores = self._scores(obs_b, goals_b, self.config['selector'])  # [B, K]
        # `temperature` may be traced under jit: compute both and select.
        temp = jnp.asarray(temperature, jnp.float32)
        stochastic = jax.random.categorical(high_seed, scores / jnp.maximum(temp, 1e-6), axis=-1)
        greedy = jnp.argmax(scores, axis=-1)
        sampled = jnp.where(temp > 0, stochastic, greedy)
        H = int(self.config['skill_horizon'])
        reselect = (agent_state['count'] % H) == 0
        committed = jnp.broadcast_to(agent_state['skill'], sampled.shape)
        skills = jnp.where(reselect, sampled, committed)

        skills_onehot = jnp.eye(int(self.config['num_skills']))[skills]
        actions = self._low_level_actions(obs_b, skills_onehot, low_seed)
        if single:
            actions = actions[0]
            new_skill = skills[0]
        else:
            new_skill = skills
        return actions, {'skill': new_skill.astype(jnp.int32), 'count': agent_state['count'] + 1}

    # -- skill-conditioned hooks (eval_skill_policy.py) ----------------------------------

    def skill_set(self, seed=None, num_skills=None, observations=None):
        return self.skill_agent.skill_set(seed=seed, num_skills=num_skills, observations=observations)

    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        del temperature
        low_temperature = self.config['low_temperature']
        if self.skill_agent.config['discrete']:
            low_temperature = max(float(low_temperature), 1e-6)
        return self.skill_agent.sample_actions_with_skill(observations, skills, seed=seed, temperature=low_temperature)

    # -- constructor ----------------------------------------------------------------------

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        skill_agent, resolved = load_frozen_skill_agent(
            seed, ex_observations, ex_actions, config,
            {'empowerment_skill': EmpowermentAgent}, caller='skill_restricted_iql',
        )
        if skill_agent.config['discrete']:
            raise ValueError('skill_restricted_iql assumes continuous actions (the kernel is an L2 distance).')
        if config['kernel'] not in ('gauss', 'hard'):
            raise ValueError(f"kernel must be 'gauss' or 'hard', got {config['kernel']!r}")
        if config['selector'] not in ('qz', 'qmu'):
            raise ValueError(f"selector must be 'qz' or 'qmu', got {config['selector']!r}")
        if int(config['skill_horizon']) < 1:
            raise ValueError('skill_horizon must be >= 1')
        num_skills = resolved['num_skills']

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            for name in ('value', 'critic', 'qz', 'cov'):
                encoders[name] = GCEncoder(concat_encoder=encoder_module())

        ex_goals = ex_observations
        value_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'],
                            ensemble=False, gc_encoder=encoders.get('value'))
        critic_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'],
                             ensemble=True, gc_encoder=encoders.get('critic'))
        qz_def = SkillHead(hidden_dims=config['value_hidden_dims'], num_skills=num_skills,
                           layer_norm=config['layer_norm'], gc_encoder=encoders.get('qz'))
        cov_def = SkillHead(hidden_dims=config['value_hidden_dims'], num_skills=num_skills,
                            layer_norm=config['layer_norm'], gc_encoder=encoders.get('cov'))

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            qz=(qz_def, (ex_observations, ex_goals)),
            cov=(cov_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))
        network.params['modules_target_critic'] = network.params['modules_critic']

        stored = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        stored['num_skills'] = num_skills
        stored['skill_restore_epoch'] = resolved['restore_epoch']
        stored['skill_checkpoint_path'] = resolved['ckpt_path']
        sigma0 = 1.0 if config['kernel_sigma'] is None else float(config['kernel_sigma'])
        return cls(rng, network=network, skill_agent=skill_agent,
                   kernel_sigma=jnp.asarray(sigma0, jnp.float32), config=flax.core.FrozenDict(**stored))


def _pick_entropy(picks, num_skills):
    counts = jnp.bincount(picks, length=num_skills).astype(jnp.float32)
    p = counts / jnp.maximum(counts.sum(), 1.0)
    return -(jnp.where(p > 0, p * jnp.log(jnp.maximum(p, 1e-12)), 0.0)).sum()


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='skill_restricted_iql',
            # Frozen skills.
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),  # empowerment_skill run dir (required).
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),  # default: latest params_*.pkl.
            num_skills=ml_collections.config_dict.placeholder(int),  # read from the checkpoint.
            low_temperature=0.0,  # frozen skill actor temperature at eval (0 = mean action).
            # SR-IQL.
            lr=3e-4,
            batch_size=1024,
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,  # target critic Polyak rate.
            expectile=0.9,  # IQL expectile over covered skills.
            kernel='gauss',  # 'gauss' (width kernel_sigma) or 'hard' (nearest skill, parameter-free).
            kernel_sigma=ml_collections.config_dict.placeholder(float),  # None -> sqrt(E_D[min_z ||a-mu_z||^2]).
            coverage_alpha=1.0,  # weight of log c(z|s) in the selector.
            selector='qz',  # 'qz' (in-sample per-skill value) or 'qmu' (off-support Q(s, mu_z(s), g)).
            skill_horizon=1,  # env steps a chosen skill is held at eval.
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            # Dataset (gciql defaults).
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
        )
    )
    return config
