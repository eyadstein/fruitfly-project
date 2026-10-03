import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp


def read(p):
    return pd.read_csv(glob.glob(f"connectome/data/{p}*")[0])


conn, cls = read("conn"), read("class")
print("nt_type values:")
print(conn["nt_type"].value_counts().to_string())

SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0}      # other transmitters ignored in this first version
conn["s"] = conn["nt_type"].astype(str).str.upper().map(SIGN)
conn = conn.dropna(subset=["s"])
conn["w"] = conn["syn_count"] * conn["s"]
e = conn.groupby(["pre_root_id", "post_root_id"], as_index=False)["w"].sum()

ids = pd.Index(np.unique(np.concatenate([e["pre_root_id"].values, e["post_root_id"].values])))
N = len(ids)
pre, post = ids.get_indexer(e["pre_root_id"]), ids.get_indexer(e["post_root_id"])
W = sp.csr_matrix((e["w"].values.astype(np.float32), (post, pre)), shape=(N, N))
absrow = np.asarray(abs(W).sum(axis=1)).ravel()
W = (sp.diags(1.0 / np.maximum(absrow, 1e-9)) @ W).tocsr().astype(np.float32)
print(f"\nbrain model: {N:,} neurons, {W.nnz:,} signed connections")


def sel(mask, side):
    idx = ids.get_indexer(cls.loc[mask & (cls["side"] == side), "root_id"])
    return idx[idx >= 0]


olf, dn = cls["class"] == "olfactory", cls["super_class"] == "descending"
OL, OR, DL, DR = sel(olf, "left"), sel(olf, "right"), sel(dn, "left"), sel(dn, "right")
print(f"olfactory neurons: {len(OL)} left, {len(OR)} right | descending neurons: {len(DL)} left, {len(DR)} right")


def run(left, right, G=0.9, iters=60):
    I = np.zeros(N, np.float32)
    I[OL], I[OR] = left, right
    r = np.zeros(N, np.float32)
    for _ in range(iters):
        r = np.maximum(G * (W @ r) + I, 0.0)
    return r


print("\nsteady-state descending output (signal = (L - R) / (L + R)):")
print(f"{'odour':<14}{'DN left':>12}{'DN right':>12}{'signal':>10}{'active DNs':>12}")
for name, (a, b) in {"on the left": (1.0, 0.5), "on the right": (0.5, 1.0), "both equal": (1.0, 1.0)}.items():
    r = run(a, b)
    dl, dr = r[DL].sum(), r[DR].sum()
    active = int((r[np.concatenate([DL, DR])] > 1e-6).sum())
    print(f"{name:<14}{dl:>12.4g}{dr:>12.4g}{(dl - dr) / (dl + dr + 1e-12):>10.3f}{active:>12}")
