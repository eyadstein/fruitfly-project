import numpy as np
from fruitfly.index import FlyIndex
from fruitfly.foa import foa

rng = np.random.default_rng(0)
N, D, Q, K = 1000, 128, 40, 10
data = rng.normal(size=(N, D)).astype(np.float32)
queries = data[rng.choice(N, Q, replace=False)] + rng.normal(scale=0.5, size=(Q, D)).astype(np.float32)
ids = [str(i) for i in range(N)]

unit = data / np.linalg.norm(data, axis=1, keepdims=True)
truth = [set(str(i) for i in np.argsort(-(unit @ (q / np.linalg.norm(q))))[:K]) for q in queries]

def recall(expansion, wta):
    idx = FlyIndex(D, expansion=int(round(expansion)), wta_frac=float(wta))
    idx.add(ids, data)
    return np.mean([len({r[0] for r in idx.search(q, K)} & truth[i]) / K for i, q in enumerate(queries)])

bounds = [(2, 40), (0.01, 0.15)]
objective = lambda p: -recall(p[0], p[1])

x, val, hist = foa(objective, bounds, n_flies=8, iters=15, step_frac=0.3)
print(f"FOA    : expansion={round(x[0])} wta={x[1]:.3f} recall@10={-val:.3f}")

r = np.random.default_rng(1)
best = max(((recall(*p), p) for p in (r.uniform([2, 0.01], [40, 0.15]) for _ in range(120))), key=lambda t: t[0])
print(f"Random : expansion={round(best[1][0])} wta={best[1][1]:.3f} recall@10={best[0]:.3f}")
