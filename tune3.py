import time
import numpy as np
from fruitfly.index import FlyIndex
from fruitfly.foa import foa

rng = np.random.default_rng(0)
N, D, Q, K = 1000, 128, 40, 10
data = rng.normal(size=(N, D)).astype(np.float32)
ids = [str(i) for i in range(N)]
unit = data / np.linalg.norm(data, axis=1, keepdims=True)

def make_queries():
    q = data[rng.choice(N, Q, replace=False)] + rng.normal(scale=0.5, size=(Q, D)).astype(np.float32)
    t = [set(str(i) for i in np.argsort(-(unit @ (x / np.linalg.norm(x))))[:K]) for x in q]
    return q, t

tune_q, tune_t = make_queries()
test_q, test_t = make_queries()

def evaluate(expansion, wta, sf, qs, ts):
    idx = FlyIndex(D, expansion=int(round(expansion)), wta_frac=float(wta), sample_frac=float(sf))
    idx.add(ids, data)
    t0 = time.perf_counter()
    r = np.mean([len({x[0] for x in idx.search(q, K)} & ts[i]) / K for i, q in enumerate(qs)])
    return r, (time.perf_counter() - t0) / len(qs) * 1000

# wta capped at 0.10 to keep the code fly-like sparse
bounds = [(2, 80), (0.01, 0.10), (0.02, 0.30)]
lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
objective = lambda p: -evaluate(p[0], p[1], p[2], tune_q, tune_t)[0]

rec, ms = evaluate(20, 0.05, 0.1, test_q, test_t)
print(f"Default (exp=20, wta=0.05, sf=0.10): held-out recall {rec:.3f} | {ms:.1f} ms/query")

for name in ("FOA", "Random"):
    scores, speeds, cfgs = [], [], []
    for seed in range(3):
        if name == "FOA":
            x, _, _ = foa(objective, bounds, n_flies=8, iters=12, step_frac=0.3, seed=seed)
        else:
            r = np.random.default_rng(100 + seed)
            x = min((r.uniform(lo, hi) for _ in range(96)), key=objective)
        rec, ms = evaluate(x[0], x[1], x[2], test_q, test_t)
        scores.append(rec); speeds.append(ms)
        cfgs.append((int(round(x[0])), round(float(x[1]), 3), round(float(x[2]), 3)))
    print(f"{name:6}: held-out recall {np.mean(scores):.3f} +/- {np.std(scores):.3f} | {np.mean(speeds):.1f} ms/query | (exp, wta, sf) {cfgs}")
