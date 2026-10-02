import glob
import numpy as np
import pandas as pd

K, N, Q, SEEDS = 10, 1000, 40, 3
WTAS = (0.02, 0.05, 0.10)


def read(p):
    return pd.read_csv(glob.glob(f"connectome/data/{p}*")[0])


conn = read("conn")
cls = read("class")
conn = conn.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()

pn = cls[cls["class"] == "ALPN"]
kc = cls[cls["class"] == "Kenyon_Cell"]
side = kc["side"].value_counts().idxmax()
print("KC sides:", kc["side"].value_counts().to_dict(), "-> using", side)
pn_ids = set(pn.loc[pn["side"] == side, "root_id"])
kc_ids = set(kc.loc[kc["side"] == side, "root_id"])
e = conn[conn["pre_root_id"].isin(pn_ids) & conn["post_root_id"].isin(kc_ids)]
if e.empty:
    raise SystemExit("no PN->KC connections found, paste the lines printed above")

pn_list = sorted(e["pre_root_id"].unique())
kc_list = sorted(e["post_root_id"].unique())
pi = {n: i for i, n in enumerate(pn_list)}
ki = {n: i for i, n in enumerate(kc_list)}
W = np.zeros((len(kc_list), len(pn_list)), dtype=np.float32)
for a, b, w in zip(e["pre_root_id"], e["post_root_id"], e["syn_count"]):
    W[ki[b], pi[a]] = w

B = (W > 0).astype(np.float32)
deg = B.sum(axis=1).astype(int)
print(f"{len(pn_list)} PNs -> {len(kc_list)} KCs, {int(B.sum()):,} connections")
print(f"inputs per KC: mean {deg.mean():.1f}, median {int(np.median(deg))}, max {deg.max()}")
print(f"KCs per PN: mean {B.sum(axis=0).mean():.1f}")
np.savez("connectome/pn_kc_matrix.npz", W=W, pn=np.array(pn_list), kc=np.array(kc_list))


def random_like(rng):
    R = np.zeros_like(B)
    for i, d in enumerate(deg):
        R[i, rng.choice(B.shape[1], d, replace=False)] = 1
    return R


def code(M, X, top):
    X = X - X.mean(axis=1, keepdims=True)
    act = X @ M.T
    idx = np.argpartition(-act, top - 1, axis=1)[:, :top]
    out = np.zeros(act.shape, dtype=np.int32)
    np.put_along_axis(out, idx, 1, axis=1)
    return out


D = B.shape[1]
results = {w: {"real binary": [], "real weighted": [], "random matched": []} for w in WTAS}
for seed in range(SEEDS):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(N, D)).astype(np.float32)
    qs = data[rng.choice(N, Q, replace=False)] + rng.normal(scale=0.5, size=(Q, D)).astype(np.float32)
    Dn = data / np.linalg.norm(data, axis=1, keepdims=True)
    Qn = qs / np.linalg.norm(qs, axis=1, keepdims=True)
    truth = np.argsort(-(Qn @ Dn.T), axis=1)[:, :K]
    mats = {"real binary": B, "real weighted": W, "random matched": random_like(rng)}
    for wta in WTAS:
        top = max(1, int(B.shape[0] * wta))
        for name, M in mats.items():
            cd, cq = code(M, data, top), code(M, qs, top)
            found = np.argsort(-(cq @ cd.T), axis=1)[:, :K]
            r = np.mean([len(set(found[i]) & set(truth[i])) / K for i in range(Q)])
            results[wta][name].append(r)

print(f"\nrecall@{K} vs exact cosine ({N} random {D}-d vectors, {SEEDS} seeds)")
print(f"{'wta':>5} {'real binary':>18} {'real weighted':>18} {'random matched':>18}")
for wta in WTAS:
    row = "".join(f"{np.mean(v):>11.3f} +/-{np.std(v):.3f}" for v in results[wta].values())
    print(f"{wta:>5} {row}")
