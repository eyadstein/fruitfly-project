import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp

ETA, RECOV, TOP, G = 0.15, 0.1, 0.05, 0.9
N_LEARN, N_REVERSE, SEEDS = 10, 20, range(10)
AVOID = ["MBON01", "MBON02", "MBON03", "MBON04", "MBON05"]
APPROACH = ["MBON09", "MBON11", "MBON12"]


def read(p, **kw):
    return pd.read_csv(glob.glob(f"connectome/data/{p}*")[0], **kw)


W = sp.load_npz("connectome/brain_W.npz").tocsr()
conn = read("conn", usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"])
kept = conn[conn["nt_type"].astype(str).str.upper().isin(["ACH", "GABA", "GLUT"])]
ids = pd.Index(np.unique(np.concatenate([kept["pre_root_id"].values, kept["post_root_id"].values])))
assert len(ids) == W.shape[0], "neuron order does not match the cached brain model"
N = W.shape[0]
cls, ct = read("class"), read("*cell_type")


def ix(root):
    a = ids.get_indexer(root)
    return a[a >= 0]


ptype = ct.set_index("root_id")["primary_type"].astype(str)
mb = ptype[ptype.str.startswith("MBON")]
print("MBON types in the table:", mb.value_counts().sort_index().to_dict())
roots = {t: mb.index[mb == t].values for t in AVOID + APPROACH}
print("avoid types:   ", {t: len(roots[t]) for t in AVOID})
print("approach types:", {t: len(roots[t]) for t in APPROACH})

rt = {r: t for t, v in roots.items() for r in v}
sub = conn[conn["pre_root_id"].isin(list(rt))].copy()
sub["type"] = sub["pre_root_id"].map(rt)
share = sub.groupby(["type", "nt_type"])["syn_count"].sum().unstack(fill_value=0)
share = share.div(share.sum(axis=1), axis=0).round(2)
print("\npredicted output transmitter of the chosen types (share of synapses):")
print(share.to_string())

KC = ix(cls.loc[cls["class"] == "Kenyon_Cell", "root_id"])
MBALL = ix(cls.loc[cls["class"] == "MBON", "root_id"])
rows = lambda names: ix(np.concatenate([roots[t] for t in names]))
APP = np.abs(W[rows(APPROACH)][:, KC].toarray()).astype(np.float32)
AV = np.abs(W[rows(AVOID)][:, KC].toarray()).astype(np.float32)
assert APP.shape[0] > 0 and AV.shape[0] > 0, "no MBONs found for one group; paste the MBON types line"
print(f"\nMBONs used: {AV.shape[0]} avoid, {APP.shape[0]} approach; "
      f"KC->MBON connections: {int((AV > 0).sum()):,} avoid, {int((APP > 0).sum()):,} approach")
MALL = np.abs(W[MBALL][:, KC].toarray()).astype(np.float32)

orn = ct[ct["primary_type"].astype(str).str.startswith("ORN_")]
orn_idx = {t: ix(g["root_id"]) for t, g in orn.groupby("primary_type")}
types = sorted(orn_idx)


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


def shuffled(M, rng, swaps=10):
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


def run(app_m, av_m, SA, SB):
    wa, wv = app_m.copy(), av_m.copy()

    def pi(S):
        a, b = wa[:, S].sum(), wv[:, S].sum()
        return (a - b) / (a + b + 1e-12)

    pref = lambda: pi(SA) - pi(SB)
    out = [pref()]
    for _ in range(N_LEARN):
        wv[:, SA] *= 1 - ETA
        wa[:, SB] *= 1 - ETA
    out.append(pref())
    for _ in range(N_REVERSE):
        wa += RECOV * (app_m - wa)
        wv += RECOV * (av_m - wv)
        wa[:, SA] *= 1 - ETA
        wv[:, SB] *= 1 - ETA
    out.append(pref())
    return out


res = {"faithful valence, real wiring": [], "faithful valence, shuffled wiring": [],
       "random 50/50 split of all MBONs (earlier method)": []}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(types))
    SA = kc_code([types[i] for i in perm[:8]])
    SB = kc_code([types[i] for i in perm[8:16]])
    res["faithful valence, real wiring"].append(run(APP, AV, SA, SB))
    S = shuffled(np.vstack([APP, AV]), rng)
    res["faithful valence, shuffled wiring"].append(run(S[: len(APP)], S[len(APP):], SA, SB))
    sp_ = rng.permutation(len(MBALL))
    h = len(MBALL) // 2
    res["random 50/50 split of all MBONs (earlier method)"].append(run(MALL[sp_[:h]], MALL[sp_[h:]], SA, SB))

print(f"\npreference for A over B (-2 to +2), {len(list(SEEDS))} random odour pairs")
print(f"{'condition':<52}{'before':>16}{'after 10 trials':>20}{'after reversal':>18}")
for name, rows_ in res.items():
    a = np.array(rows_)
    print(f"{name:<52}" + "".join(f"{a[:, j].mean():>11.3f} +/-{a[:, j].std():.3f}" for j in range(3)))
