import numpy as np


class FlyHash:
    def __init__(self, input_dim, expansion=20, sample_frac=0.1, wta_frac=0.05, seed=42):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.m = input_dim * expansion
        k = max(1, int(input_dim * sample_frac))
        self.proj = np.zeros((self.m, input_dim), dtype=np.float32)
        for i in range(self.m):
            self.proj[i, rng.choice(input_dim, k, replace=False)] = 1.0
        self.top = max(1, int(self.m * wta_frac))

    def hash(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = x - x.mean(axis=-1, keepdims=True)      # center the input
        act = x @ self.proj.T                        # expand
        out = np.zeros(act.shape, dtype=bool)
        idx = np.argpartition(-act, self.top - 1, axis=-1)[..., : self.top]
        np.put_along_axis(out, idx, True, axis=-1)   # winner-take-all
        return out
