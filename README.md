# Regret-Parameterized Bellman RL

This repo explores a reparameterization of the action-value function that separates **value propagation** from **action regret**. Instead of learning `Q(s,a)` directly, it models:

```
Q(s,a) = f(s) - g(s,a),    g(s,a) >= 0
```

Where:
- `f(s)` is the value envelope (equal to `V(s)`),
- `g(s,a)` is the nonnegative regret (energy) of action `a`,
- optimal actions satisfy `g(s,a*) = 0`.

Because `V(s) = max_a Q(s,a) = f(s)`, the value function is represented explicitly and **no action maximization appears in the Bellman target**.

## Bellman Training

Given a transition `(s, a, r, s')`, the target is:

```
y = r + gamma * f(s')
```

and the Bellman loss is:

```
L = ( f(s) - g(s,a) - y )^2
```

This is equivalent to standard Q-learning under the change of variables `Q = f - g`, but factorizes learning into:
- value propagation through `f`,
- action discrimination through `g`.

## Discrete Action Improvement

For finite action spaces, policy improvement is exact:

```
pi(s) = argmin_a g(s,a)
```

So the optimal action can be found by enumeration with **no actor network** and **no gradient-based search**. In the discrete case, the algorithm reduces to critic-only value iteration in energy coordinates; the challenge is approximating the Bellman operator.

## How This Repo Implements It

The implementation uses two networks:
- a value network `f(s)` that outputs a scalar,
- a regret network `g(s,a)` that outputs per-action regrets, constrained to be nonnegative via a positivity transform (e.g. `softplus`).

Action selection is:
- greedy `argmin_a g(s,a)` for evaluation,
- or a softmax over `f(s) - g(s,a)` for exploration, with optional learned temperature.


# TODOs
* implement optional target network
* debug PER (pass use_per also to replay buffer constructor, cant use both samplers on one instance anyways)
* add importance weights and beta to PER sample return and in the agent-computation
* add config param for architecture choice: 2 networks or 2 heads on one network?
* config param to push regret down for best found action? i mean training the regret function to output 0 for best found action
