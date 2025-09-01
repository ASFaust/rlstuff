import numpy as np
import torch
from torch.utils.data import Dataset

# --- your two datasets exactly as provided (unchanged) ------------------------

class RewardDataset(Dataset):
    def __init__(self, S, R, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

        all_states = []
        all_rewards = []
        for s, r in zip(S, R):
            all_states.append(torch.as_tensor(s, dtype=dtype, device=device))
            all_rewards.append(torch.as_tensor(r, dtype=torch.float32, device=device))
        self.states = torch.cat(all_states, dim=0)   # [N, *obs_shape]
        self.rewards = torch.cat(all_rewards, dim=0) # [N]

        self.pos_idx = (self.rewards > 0).nonzero(as_tuple=True)[0]
        self.zero_idx = (self.rewards == 0).nonzero(as_tuple=True)[0]
        self.neg_idx = (self.rewards < 0).nonzero(as_tuple=True)[0]

        self.has_pos = len(self.pos_idx) > 0
        self.has_zero = len(self.zero_idx) > 0
        self.has_neg = len(self.neg_idx) > 0

        cats = []
        if self.has_pos: cats.append("pos")
        if self.has_zero: cats.append("zero")
        if self.has_neg: cats.append("neg")

        if cats == ["pos", "zero"]:
            self.cat_probs = {"pos": 0.5, "zero": 0.5}
        elif cats == ["pos", "neg"]:
            self.cat_probs = {"pos": 0.5, "neg": 0.5}
        elif len(cats) == 3:
            self.cat_probs = {"pos": 1/3, "zero": 1/3, "neg": 1/3}
        elif len(cats) == 1:
            self.cat_probs = {cats[0]: 1.0}
        else:
            raise ValueError("No rewards found in dataset")

        self.pos_weights = None
        if self.has_pos:
            pos_vals = self.rewards[self.pos_idx].abs().cpu().numpy()
            self.pos_weights = pos_vals / pos_vals.sum()

        self.neg_weights = None
        if self.has_neg:
            neg_vals = self.rewards[self.neg_idx].abs().cpu().numpy()
            self.neg_weights = neg_vals / neg_vals.sum()

        self.length = len(self.states)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        cat = np.random.choice(list(self.cat_probs.keys()),
                               p=list(self.cat_probs.values()))
        if cat == "pos":
            if self.pos_weights is not None:
                choice = np.random.choice(len(self.pos_idx), p=self.pos_weights)
            else:
                choice = np.random.randint(len(self.pos_idx))
            i = self.pos_idx[choice]
        elif cat == "neg":
            if self.neg_weights is not None:
                choice = np.random.choice(len(self.neg_idx), p=self.neg_weights)
            else:
                choice = np.random.randint(len(self.neg_idx))
            i = self.neg_idx[choice]
        elif cat == "zero":
            choice = np.random.randint(len(self.zero_idx))
            i = self.zero_idx[choice]
        else:
            raise RuntimeError("Invalid category chosen")

        s = self.states[i]
        r = self.rewards[i]
        return {"task": "reward", "state": s, "reward": r}


class ReachabilityDataset(Dataset):
    def __init__(self, S, A, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

        self.S = [torch.as_tensor(s, dtype=dtype, device=device) for s in S]
        self.A = [torch.as_tensor(a, dtype=torch.long, device=device) for a in A]
        self.n_episodes = len(self.S)
        self.lengths = [len(s) for s in self.S]
        self.index = [(ep, t) for ep, L in enumerate(self.lengths) for t in range(L - 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep, step = self.index[idx]

        s = self.S[ep][step]
        next_s = self.S[ep][step + 1]
        a = self.A[ep][step]

        if np.random.rand() < 0.5:
            future_step = np.random.randint(step + 1, self.lengths[ep])
            s_cmp = self.S[ep][future_step]
            label = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        else:
            choose_past = (step > 0) and (np.random.rand() < 0.5 or self.n_episodes == 1)
            if choose_past:
                past_step = np.random.randint(0, step)
                s_cmp = self.S[ep][past_step]
            else:
                other_ep = np.random.randint(0, max(1, self.n_episodes - 1))
                if self.n_episodes > 1 and other_ep >= ep:
                    other_ep += 1
                other_step = np.random.randint(0, self.lengths[other_ep])
                s_cmp = self.S[other_ep][other_step]
            label = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return {
            "task": "reach",
            "state": s,
            "next_state": next_s,
            "action": a,
            "compare_state": s_cmp,
            "label": label,
        }

class ZippedDataset(Dataset):
    """
    Zip two datasets together.

    mode = "truncate"  -> length = min(len(ds1), len(ds2)), stop when the first ends.
    mode = "cycle"     -> length = max(len(ds1), len(ds2)), index the shorter modulo its length.

    Each __getitem__ returns a tuple: (item_from_ds1, item_from_ds2).
    Optionally apply a `transform` to that tuple before returning.
    """
    def __init__(self, ds1: Dataset, ds2: Dataset, mode: str = "truncate", transform=None):
        assert mode in ("truncate", "cycle")
        self.ds1 = ds1
        self.ds2 = ds2
        self.mode = mode
        self.transform = transform

        if mode == "truncate":
            self.length = min(len(ds1), len(ds2))
        else:  # cycle
            self.length = max(len(ds1), len(ds2))

        if self.length == 0:
            raise ValueError("At least one dataset is empty (nothing to zip).")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == "truncate":
            i1 = idx
            i2 = idx
        else:  # cycle the shorter
            i1 = idx % len(self.ds1)
            i2 = idx % len(self.ds2)

        item1 = self.ds1[i1]
        item2 = self.ds2[i2]

        pair = (item1, item2)
        if self.transform is not None:
            pair = self.transform(pair)
        return pair
