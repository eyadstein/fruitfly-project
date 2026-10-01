import time
import numpy as np
from fruitfly.index import FlyIndex

rng = np.random.default_rng(0)
N, D, Q, K = 2000, 128, 50, 10
data = rng.normal(size=(N, D)).astype(np.float32)
queries = data[rng.choice(N, Q, replace=False)] + rng.normal(scale=0.5, size=(Q, D)).astype(np.float32)
ids = [str(i) for i in range(N)]

unit = data / np.linalg.norm(data, axis=1, keepdims=True)
truth = []
for q in queries:
    sims = unit @ (q / np.linalg.norm(q))
    truth.append(set(str(i) for i in np.argsort(-sims)[:K]))

print(f"{'expansion':>9} {'wta':>6} {'recall@10':>10} {'ms/query':>9}")
for expansion in (5, 10, 20, 40):
    for wta in (0.02, 0.05, 0.10):
        idx = FlyIndex(D, expansion=expansion, wta_frac=wta)
        idx.add(ids, data)
        t = time.perf_counter()
        hits = [len({r[0] for r in idx.search(q, K)} & truth[i]) / K for i, q in enumerate(queries)]
        ms = (time.perf_counter() - t) / Q * 1000
        print(f"{expansion:>9} {wta:>6} {np.mean(hits):>10.2f} {ms:>9.1f}")
