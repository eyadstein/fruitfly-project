import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp

ETA, TOP, G, SEEDS = 0.15, 0.05, 0.9, range(10)
N_LEARN, N_REVERSE = 10, 20


def read(p, **kw):
    return pd.read_csv(glob.glob(f"connectome/data/{p}*")[0], **kw)


W = sp.load_npz("connectome/brain_W.npz").tocsr()
conn = read("conn", usecols=["pre_root_id", "post_root_id", "nt_type"])
conn = conn[conn["nt_type"].astype(str).str.upper().isin(["ACH", "GABA", "GLUT"])]
ids = pd.Index(np.unique(np.concatenate([conn["pre_root_id"].values, conn["post_root_id"].values])))
assert len(ids) == W.shape[0], "neuron order does not match the cached brain model"
N = W.shape[0]
cls, ct = read("class"), read("*cell_type")


def ix(root):
    a = ids.get_indexer(root)
    return a[a >= 0]


KC = ix(cls.loc[cls["class"] == "Kenyon_Cell", "root_id"])
MB = ix(cls.loc[cls["class"] == "MBON", "root_id"])
orn = ct[ct["primary_type"].astype(str).str.startswith("ORN_")]
orn_idx = {t: ix(g["root_id"]) for t, g in orn.groupby("primary_type")}
types = sorted(orn_idx)
M0 = np.abs(W[MB][:, KC].toarray()).astype(np.float32)
print(f"{len(types)} ORN types, {len(KC)} KCs, {len(MB)} MBONs; "
      f"KC->MBON connections: {int((M0 > 0).sum()):,}, MBONs with KC input: {int((M0.sum(axis=1) > 0).sum())}")


def kc_code(odour_types):
    I = np.zeros(N, np.float32)
    for t in odour_types:
        I[orn_idx[t]] = 1.0
    r = np.zeros(N, np.float32)
    for _ in range(60):
        r = np.maximum(G * (W @ r) + I, 0.0)
    a = r[KC]
    k = int(TOP * len(KC))
    idx = np.argpartition(-a, k)[:k]
    return idx[a[idx] > 0]


def shuffled(M, rng, swaps=5):
    ks, ps = np.nonzero(M)
    w = M[ks, ps].copy()
    ks, ps = ks.tolist(), ps.tolist()
    edges, E = set(zip(ks, ps)), len(ks)
    for _ in range(swaps * E):
        i, j = rng.integers(E, size=2)
        k1, p1, k2, p2 = ks[i], ps[i], ks[j], ps[j]
        if k1 == k2 or p1 == p2 or (k1, p2) in edges or (k2, p1) in edges:
            continue
        edges.remove((k1, p1)); edges.remove((k2, p2))
        edges.add((k1, p2)); edges.add((k2, p1))
        ps[i], ps[j] = p2, p1
    S = np.zeros_like(M)
    S[ks, ps] = w
    return S


def pi(w, S, app, av):
    m = w[:, S].sum(axis=1)
    a, b = m[app].sum(), m[av].sum()
    return (a - b) / (a + b + 1e-12)


def pref(w, SA, SB, app, av):
    return pi(w, SA, app, av) - pi(w, SB, app, av)      # > 0 means she prefers A over B


res = {"real wiring": [], "shuffled wiring": []}
overlaps, sizes = [], []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(types))
    SA = kc_code([types[i] for i in perm[:8]])
    SB = kc_code([types[i] for i in perm[8:16]])
    overlaps.append(len(set(SA) & set(SB)) / max(len(SA), 1))
    sizes.append((len(SA), len(SB)))
    split = rng.permutation(len(MB))
    app, av = split[: len(MB) // 2], split[len(MB) // 2:]
    for name, M in (("real wiring", M0), ("shuffled wiring", shuffled(M0, rng))):
        w = M.copy()
        row = [pref(w, SA, SB, app, av)]
        for _ in range(N_LEARN):
            w[np.ix_(av, SA)] *= 1 - ETA       # A paired with reward: weaken avoidance synapses
            w[np.ix_(app, SB)] *= 1 - ETA      # B paired with punishment: weaken approach synapses
        row.append(pref(w, SA, SB, app, av))
        for _ in range(N_REVERSE):
            w[np.ix_(app, SA)] *= 1 - ETA      # now A is punished
            w[np.ix_(av, SB)] *= 1 - ETA       # and B is rewarded
        row.append(pref(w, SA, SB, app, av))
        res[name].append(row)

print(f"\nKC code per odour: ~{int(np.mean([s[0] for s in sizes]))} active KCs, "
      f"overlap between the two odours {np.mean(overlaps):.1%}")
print(f"preference for A over B (+1 = always A), {len(list(SEEDS))} random odour pairs / MBON valence splits")
print(f"{'wiring':<18}{'before':>16}{'after '+str(N_LEARN)+' trials':>20}{'after reversal':>18}")
for name, rows in res.items():
    a = np.array(rows)
    print(f"{name:<18}" + "".join(f"{a[:, j].mean():>11.3f} +/-{a[:, j].std():.3f}" for j in range(3)))
