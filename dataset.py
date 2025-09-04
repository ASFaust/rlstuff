import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, S, A, R, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

        # S: list of arrays/tensors of states per episode, shape [T_i, ...]
        # A: list of arrays/tensors of actions per episode, shape [T_i], aligned with S
        self.S = [torch.as_tensor(s, dtype=dtype, device=device) for s in S]
        self.A = [torch.as_tensor(a, dtype=torch.long, device=device) for a in A]
        self.R = [torch.as_tensor(r, dtype=dtype, device=device) for r in R]
        self.n_episodes = len(self.S)
        self.lengths = [len(s) for s in self.S]

        # need at least 3 steps ahead (t, n, m), so t ∈ [0, L-3]
        self.index = [(ep, t) for ep, L in enumerate(self.lengths) for t in range(L - 3)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep, t = self.index[idx]
        S_ep, A_ep = self.S[ep], self.A[ep]
        R_ep = self.R[ep]
        L = self.lengths[ep]

        # base state
        s1 = S_ep[t]
        a1 = A_ep[t]  # action from s1 -> s2 if needed
        a2 = A_ep[t + 1]
        s2 = S_ep[t + 1]
        s3 = S_ep[t + 2]
        # we need sx from 50% chance the same episode but before, or 50% chance from another episode
        # so first we look wether t allows to sample from the same episode before
        if t > 0 and np.random.rand() < 0.5:
            sx = S_ep[np.random.randint(0, t)]
        else:
            # sample from another episode
            other_ep = np.random.randint(0, self.n_episodes)
            while other_ep == ep:
                other_ep = np.random.randint(0, self.n_episodes)
            sx = self.S[other_ep][np.random.randint(0, self.lengths[other_ep])]

        # sx is a state for which we want to classify that it is not within our reachability set

        # sample n, m with t < n < m < L
        n = np.random.randint(t + 1, L-1)
        m = n
        while m == n:
            m = np.random.randint(t + 1, L)
        if n > m:
            n, m = m, n

        sn = S_ep[n]
        sm = S_ep[m]

        # step distances
        d1 = n - t       # steps from s1 to sn
        d2 = m - n       # steps from sn to sm

        # reward at time t
        r1 = R_ep[t]


        return {
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "sx": sx, #sx is used to classify wether a goal state is reachable or not
            "sn": sn,
            "sm": sm,
            "a1": a1,
            "a2": a2,
            "d1" : d1,
            "d2" : d2,
            "r1" : r1,
        }
