import numpy as np
from .flyhash import FlyHash


class FlyIndex:
    def __init__(self, input_dim, **kw):
        self.fh = FlyHash(input_dim, **kw)
        self.codes = np.empty((0, self.fh.m), dtype=np.uint8)
        self.ids = []

    def add(self, ids, vectors):
        codes = self.fh.hash(np.asarray(vectors)).astype(np.uint8)
        self.codes = np.vstack([self.codes, codes])
        self.ids.extend(ids)

    def search(self, vector, k=5):
        q = self.fh.hash(np.asarray(vector)).astype(np.uint8)
        scores = self.codes.astype(np.int32) @ q.astype(np.int32)
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], int(scores[i])) for i in order]
