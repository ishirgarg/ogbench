# Empowerment Agent — Backup Equations

## Notation

| Symbol | Meaning |
|---|---|
| $s, s', s^+$ | Current state, next state, future state |
| $a$ | Action |
| $z \in \{1,\ldots,K\}$ | Discrete skill index; $\mathbf{z}$ is its one-hot encoding |
| $Q^z_\theta(s^+ \mid s, a)$ | Online Q network |
| $Q^z_{\bar\theta}(s^+ \mid s, a)$ | Target Q network |
| $V^z_\theta(s^+ \mid s)$ | Online V network |
| $V^z_{\bar\theta}(s^+ \mid s)$ | Target V network |
| $\pi_\theta(s, \mathbf{z})$ | Deterministic policy (mode / argmax) |
| $\gamma$ | Discount factor |
| $\operatorname{sg}[\cdot]$ | Stop-gradient |

---

## 1. Q Backup

### Mode A — `separate_qv = True`

Q is regressed onto the **target V** at the next state:

$$
Q^z_\theta(s^+ \mid s, a) \;\leftarrow\; \operatorname{sg}\!\left[V^z_{\bar\theta}(s^+ \mid s')\right]
$$

### Mode B — `separate_qv = False`

No separate V network exists; Q carries both Bellman targets.

**Future-state term** — Q at $(s, a)$ is backed up by Q at the next state under the frozen policy:

$$
Q^z_\theta(s^+ \mid s, a) \;\leftarrow\; \operatorname{sg}\!\left[\gamma \cdot Q^z_{\bar\theta}\!\left(s^+ \mid s',\, \pi_{\bar\theta}(s', \mathbf{z})\right)\right]
$$

**Current-state term** — self-consistency: the value of the current state under itself must equal the immediate occupancy plus its discounted continuation:

$$
Q^z_\theta(s' \mid s, a) \;\leftarrow\; \operatorname{sg}\!\left[(1 - \gamma) + \gamma \cdot Q^z_{\bar\theta}\!\left(s' \mid s',\, \pi_{\bar\theta}(s', \mathbf{z})\right)\right]
$$

---

## 2. V Backup

*(Only active when `separate_qv = True`; otherwise $\mathcal{L}_V = 0$.)*

Let $a^* = \operatorname{sg}[\pi_\theta(s, \mathbf{z})]$ (policy is frozen in the V loss).

**Future-state term** — V is backed up by the **target Q** evaluated at the frozen policy action:

$$
V^z_\theta(s^+ \mid s) \;\leftarrow\; \operatorname{sg}\!\left[\gamma \cdot Q^z_{\bar\theta}(s^+ \mid s,\, a^*)\right]
$$

**Current-state term** (`use_self_v_loss = True`) — same self-consistency constraint as in Mode B above:

$$
V^z_\theta(s \mid s) \;\leftarrow\; \operatorname{sg}\!\left[(1 - \gamma) + \gamma \cdot Q^z_{\bar\theta}(s \mid s,\, a^*)\right]
$$

---

## 3. Policy Backup

The policy maximises empowerment — the mutual information between the skill $Z$ and the future state $S^+$:

$$
\mathcal{I}(Z;\, S^+ \mid s) \;=\; \mathbb{E}_{z,\, s^+}\!\left[\log \frac{V^z(s^+ \mid s)}{\dfrac{1}{K}\displaystyle\sum_{z'} V^{z'}(s^+ \mid s)}\right]
$$

The gradient is estimated by sampling $N$ future-state embeddings $\psi^{(n)}$ from the V-network's implied distribution and decomposing the gradient into two terms.

**Q-term** — gradient flows through $Q^z$ via the policy action $\pi_\theta(s, \mathbf{z})$:

$$
\Delta_Q^{(n)} \;=\; \frac{\operatorname{sg}[Q^z(\psi^{(n)} \mid s, \pi_\theta)]}{{\operatorname{sg}[V^z(\psi^{(n)} \mid s)]}} \cdot \left(\log Q^z\!\left(\psi^{(n)} \mid s, \pi_\theta\right) - \log \bar{m}^{(n)}\right)
$$

where the mixture denominator is:

$$
\bar{m}^{(n)} \;=\; \frac{1}{K}\!\left(Q^z\!\left(\psi^{(n)} \mid s, \pi_\theta\right) + \operatorname{sg}\!\left[\sum_{z' \ne z} V^{z'}\!\left(\psi^{(n)} \mid s\right)\right]\right)
$$

**V-others term** — accounts for how the $z' \ne z$ skills shift the denominator; all V network parameters are stop-gradiented, but the term still receives gradient through $\log\bar{m}^{(n)}$:

$$
\Delta_{V}^{(n)} \;=\; \operatorname{sg}\!\left[\sum_{z'} \frac{V^{z'}(\psi^{(n)})}{V^z(\psi^{(n)})}\right]\!\cdot\!\left(\operatorname{sg}\!\left[\log V^{z'}(\psi^{(n)})\right] - \log\bar{m}^{(n)}\right) - \left(\operatorname{sg}\!\left[\log V^z(\psi^{(n)})\right] - \log\bar{m}^{(n)}\right)
$$

**Final policy loss**, averaged over $N$ samples and batch:

$$
\mathcal{L}_\pi \;=\; -\frac{1}{BK} \sum_{b=1}^{B}\sum_{n=1}^{N} \left(\Delta_Q^{(n)} + \Delta_{V}^{(n)}\right)
$$