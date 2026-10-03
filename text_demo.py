import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from fruitfly.flyhash import FlyHash

K, N, Q, SEEDS = 10, 3000, 150, [0, 1, 2]

z = np.load("connectome/pn_kc_matrix.npz")
B = (z["W"] > 0).astype(np.float32)
nk, D = B.shape
deg = B.sum(axis=1).astype(int)
print(f"real circuit: {D} PNs -> {nk} KCs")

print("loading and vectorising 20 Newsgroups (first run downloads ~14 MB)...")
data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
X = TfidfVectorizer(stop_words="english", min_df=3, max_features=20000, sublinear_tf=True).fit_transform(data.data)
keep = np.asarray(X.getnnz(axis=1)).ravel() > 0
X, y = X[keep], data.target[keep]
V = TruncatedSVD(D, random_state=0).fit_transform(X).astype(np.float32)
V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
print(f"{len(V)} documents, {D}-d vectors, {len(set(y))} newsgroups")


def code(M, A, top):
    A = A - A.mean(axis=1, keepdims=True)
    act = A @ M.T
    idx = np.argpartition(-act, top - 1, axis=1)[:, :top]
    out = np.zeros(act.shape, dtype=np.float32)
    np.put_along_axis(out, idx, 1.0, axis=1)
    return out


def random_like(rng):
    R = np.zeros_like(B)
    for i, d in enumerate(deg):
        R[i, rng.choice(D, d, replace=False)] = 1
    return R


def shuffled(rng, swaps=10):
    ks, ps = (a.tolist() for a in np.nonzero(B))
    edges, E = set(zip(ks, ps)), len(ks)
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


def topk(scores):
    return np.argpartition(-scores, K, axis=1)[:, :K]


names = ["exact cosine", "FlyHash default (exp20 wta.05 sf.10)", "FlyHash tuned (exp75 wta.10 sf.28)"]
for w in (0.05, 0.10):
    names += [f"real wiring, wta {w}", f"shuffled wiring, wta {w}", f"random in-degree, wta {w}"]
res = {n: {"recall": [], "prec": []} for n in names}

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(V))
    ii, qi = perm[:N], perm[N:N + Q]
    Iv, Qv = V[ii], V[qi]
    truth = topk(Qv @ Iv.T)
    mats = {"real": B, "shuffled": shuffled(rng), "random in-degree": random_like(rng)}
    runs = {"exact cosine": None}
    for name, kw in (("FlyHash default (exp20 wta.05 sf.10)", dict(expansion=20, wta_frac=0.05, sample_frac=0.10)),
                     ("FlyHash tuned (exp75 wta.10 sf.28)", dict(expansion=75, wta_frac=0.10, sample_frac=0.28))):
        fh = FlyHash(D, seed=seed, **kw)
        runs[name] = (fh.hash(Iv).astype(np.float32), fh.hash(Qv).astype(np.float32))
    for w in (0.05, 0.10):
        top = int(nk * w)
        for label, key in (("real wiring", "real"), ("shuffled wiring", "shuffled"), ("random in-degree", "random in-degree")):
            M = mats[key]
            runs[f"{label}, wta {w}"] = (code(M, Iv, top), code(M, Qv, top))
    for name, pair in runs.items():
        found = truth if pair is None else topk(pair[1] @ pair[0].T)
        rec = np.mean([len(set(found[i]) & set(truth[i])) / K for i in range(Q)])
        prec = np.mean([(y[ii][found[i]] == y[qi][i]).mean() for i in range(Q)])
        res[name]["recall"].append(rec)
        res[name]["prec"].append(prec)

print(f"\n{N} indexed documents, {Q} query documents per seed, {len(SEEDS)} seeds, k={K}")
print(f"{'method':<40}{'recall vs exact cosine':>26}{'same-newsgroup precision':>28}")
for n in names:
    r, p = res[n]["recall"], res[n]["prec"]
    print(f"{n:<40}{np.mean(r):>14.3f} +/-{np.std(r):.3f}{np.mean(p):>16.3f} +/-{np.std(p):.3f}")
