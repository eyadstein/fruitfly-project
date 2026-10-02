import numpy as np

K, N, Q, SEEDS = 10, 1000, 40, 5
WTAS = (0.02, 0.05, 0.10)
KINDS = ("gaussian", "correlated")

z = np.load("connectome/pn_kc_matrix.npz")
B = (z["W"] > 0).astype(np.float32)
nk, D = B.shape
deg = B.sum(axis=1).astype(int)


def random_like(rng):
    R = np.zeros_like(B)
    for i, d in enumerate(deg):
        R[i, rng.choice(D, d, replace=False)] = 1
    return R


def shuffled(rng, swaps=10):
    ks, ps = (a.tolist() for a in np.nonzero(B))
    edges = set(zip(ks, ps))
    E = len(ks)
    for _ in range(swaps * E):
        i, j = rng.integers(E, size=2)
        k1, p1, k2, p2 = ks[i], ps[i], ks[j], ps[j]
        if k1 == k2 or p1 == p2 or (k1, p2) in edges or (k2, p1) in edges:
            continue
        edges.remove((k1, p1)); edges.remove((k2, p2))
        edges.add((k1, p2)); edges.add((k2, p1))
        ps[i], ps[j] = p2, p1
    S = np.zeros_like(B)
    for k, p in edges:
        S[k, p] = 1
    return S


def make(kind, rng):
    if kind == "gaussian":
        return rng.normal(size=(N, D)).astype(np.float32)
    Z = rng.normal(size=(N, 8))
    A = rng.normal(size=(8, D))
    return np.maximum(Z @ A + 0.5 * rng.normal(size=(N, D)), 0).astype(np.float32)


def code(M, X, top):
    X = X - X.mean(axis=1, keepdims=True)
    act = X @ M.T
    idx = np.argpartition(-act, top - 1, axis=1)[:, :top]
    out = np.zeros(act.shape, dtype=np.int32)
    np.put_along_axis(out, idx, 1, axis=1)
    return out


names = ("real", "random in-degree", "shuffled both")
res = {k: {w: {n: [] for n in names} for w in WTAS} for k in KINDS}
for kind in KINDS:
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        data = make(kind, rng)
        qs = data[rng.choice(N, Q, replace=False)] + rng.normal(scale=0.5 * data.std(), size=(Q, D)).astype(np.float32)
        Dn = data / np.linalg.norm(data, axis=1, keepdims=True)
        Qn = qs / np.linalg.norm(qs, axis=1, keepdims=True)
        truth = np.argsort(-(Qn @ Dn.T), axis=1)[:, :K]
        mats = {"real": B, "random in-degree": random_like(rng), "shuffled both": shuffled(rng)}
        for wta in WTAS:
            top = max(1, int(nk * wta))
            for n, M in mats.items():
                cd, cq = code(M, data, top), code(M, qs, top)
                found = np.argsort(-(cq @ cd.T), axis=1)[:, :K]
                res[kind][wta][n].append(np.mean([len(set(found[i]) & set(truth[i])) / K for i in range(Q)]))

for kind in KINDS:
    print(f"\n[{kind} data] recall@{K}, {SEEDS} seeds")
    print(f"{'wta':>5}" + "".join(f"{n:>23}" for n in names))
    for wta in WTAS:
        print(f"{wta:>5}" + "".join(f"{np.mean(res[kind][wta][n]):>13.3f} +/-{np.std(res[kind][wta][n]):.3f}" for n in names))
