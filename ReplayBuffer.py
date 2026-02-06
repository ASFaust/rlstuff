import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape,
        action_shape=(),
        device="cpu",
        dtype_obs=np.float32,
        dtype_action=np.int64,
    ):
        self.capacity = capacity
        self.device = device

        # ---- normalize shapes ----
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        if isinstance(action_shape, int):
            action_shape = (action_shape,)

        # storage
        self.obs = np.zeros((capacity, *obs_shape), dtype=dtype_obs)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=dtype_obs)
        self.actions = np.zeros((capacity, *action_shape), dtype=dtype_action)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.errors = np.zeros((capacity, 1), dtype=np.float32)

        # fairness tracking
        self.use_count = np.zeros(capacity, dtype=np.int32)
        self.min_use = 0

        self.ptr = 0
        self.size = 0


    # --------------------------------------------------
    # insert transition
    # --------------------------------------------------

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        # new data is fresh → zero usage
        self.use_count[self.ptr] = 0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # fresh samples exist → reset min layer
        self.min_use = 0

        self.errors[self.ptr] = 0.0

    def update_errors(self, indices, errors):
        self.errors[indices] = errors.reshape(-1, 1)

    #simple uniform sampling, for ablation
    def sample(self, batch_size: int):
        assert self.size > 0, "Cannot sample from empty buffer"

        choice = np.random.choice(self.size, batch_size, replace=False)

        batch = dict(
            obs=torch.as_tensor(self.obs[choice], device=self.device),
            actions=torch.as_tensor(self.actions[choice], device=self.device),
            rewards=torch.as_tensor(self.rewards[choice], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[choice], device=self.device),
            dones=torch.as_tensor(self.dones[choice], device=self.device),
        )

        return batch

    # --------------------------------------------------
    # fairness sampling
    # --------------------------------------------------

    def sample_fair(self, batch_size: int):
        assert self.size >= batch_size, "Not enough samples in buffer"

        # fairness priority:
        # first key = use_count (lower is better)
        # second key = random tie breaker
        noise = np.random.rand(self.size)
        priority = self.use_count[:self.size] + 1e-6 * noise

        # pick smallest priorities
        choice = np.argsort(priority)[:batch_size]

        # update usage
        self.use_count[choice] += 1

        batch = dict(
            obs=torch.as_tensor(self.obs[choice], device=self.device),
            actions=torch.as_tensor(self.actions[choice], device=self.device),
            rewards=torch.as_tensor(self.rewards[choice], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[choice], device=self.device),
            dones=torch.as_tensor(self.dones[choice], device=self.device),
        )

        return batch





    def sample_prioritized(
            self, batch_size: int, ratio=10.0, eps=1e-8
        ):
        raise NotImplementedError("Prioritized sampling is broken, do not use --- IGNORE ---")
        assert self.size > 0

        N = self.size
        choice = []

        # -------------------------
        # 1. unseen samples first
        # -------------------------
        unused = np.nonzero(self.use_count[:N] == 0)[0]
        take_new = min(len(unused), batch_size)

        if take_new > 0:
            new_samples = np.random.choice(unused, take_new, replace=False)
            choice.extend(new_samples.tolist())

        remaining = batch_size - len(choice)

        # -------------------------
        # 2. histogram transport
        # -------------------------
        if remaining > 0:

            usage = self.use_count[:N].astype(np.float64)
            errors = np.abs(self.errors[:N].flatten()).astype(np.float64)

            usage += eps
            errors += eps

            u = usage / usage.sum()
            e = errors / errors.sum()

            d = np.clip(e - u, 0.0, None)

            # fallback
            if d.sum() == 0:
                d = np.ones_like(d)

            # ratio constraint vs uniform
            uniform = 1.0 / N
            min_w = uniform / ratio
            max_w = uniform * ratio

            d = np.clip(d, min_w, max_w)
            d /= d.sum()

            extra = np.random.choice(N, remaining, replace=False, p=d)
            choice.extend(extra.tolist())

        choice = np.array(choice, dtype=np.int64)

        self.use_count[choice] += 1

        batch = dict(
            obs=torch.as_tensor(self.obs[choice], device=self.device),
            actions=torch.as_tensor(self.actions[choice], device=self.device),
            rewards=torch.as_tensor(self.rewards[choice], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[choice], device=self.device),
            dones=torch.as_tensor(self.dones[choice], device=self.device),
            indices=choice,
        )

        return batch

    # --------------------------------------------------

    def __len__(self):
        return self.size
