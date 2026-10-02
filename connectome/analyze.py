import sys, glob
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_SYN = 5
SAMPLE_NODES = 60


def load(pattern, required=True):
    files = glob.glob(f"connectome/data/{pattern}*")
    if not files:
        if required:
            sys.exit(f"No file matching connectome/data/{pattern}*")
        return None
    f = files[0]
    print("loading", f)
    return pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)


def pick(df, options, what):
    for o in options:
        if o in df.columns:
            return o
    for o in options:
        for c in df.columns:
            if c.startswith(o):
                return c
    sys.exit(f"Could not find the {what} column. Tried {options}. Columns found: {list(df.columns)}")


conn = load("conn")
print("connections columns:", list(conn.columns))
pre = pick(conn, ["pre_root_id", "pre_pt_root_id", "pre"], "pre-synaptic")
post = pick(conn, ["post_root_id", "post_pt_root_id", "post"], "post-synaptic")
syn = pick(conn, ["syn_count", "weight", "synapses"], "synapse count")

conn = conn.groupby([pre, post], as_index=False)[syn].sum().rename(columns={syn: "weight"})
conn = conn[conn["weight"] >= MIN_SYN]
print(f"{len(conn):,} connections kept (>= {MIN_SYN} synapses)")

G = nx.from_pandas_edgelist(conn, pre, post, edge_attr="weight", create_using=nx.DiGraph)

labels = {}
cls = load("class", required=False)
if cls is not None:
    print("classification columns:", list(cls.columns))
    idc = next((c for c in ["root_id", "pt_root_id", "id"] if c in cls.columns), None)
    lab = next((c for c in ["super_class", "class", "cell_class", "cell_type"] if c in cls.columns), None)
    if idc and lab:
        labels = dict(zip(cls[idc], cls[lab].fillna("?")))
        counts = pd.Series([labels.get(n, "?") for n in G.nodes]).value_counts().head(10)
        print(f"\nTop neuron groups in the graph (by '{lab}'):\n{counts.to_string()}")


def name(n):
    return f"{n} [{labels.get(n, '?')}]"


print(f"\nNodes: {G.number_of_nodes():,}   Edges: {G.number_of_edges():,}")
indeg, outdeg = dict(G.in_degree()), dict(G.out_degree())

print("\nTop 5 by out-degree (broadcasters):")
for n in sorted(outdeg, key=outdeg.get, reverse=True)[:5]:
    print(f"  {name(n)}  out={outdeg[n]}")

print("\nTop 5 by in-degree (integrators):")
for n in sorted(indeg, key=indeg.get, reverse=True)[:5]:
    print(f"  {name(n)}  in={indeg[n]}")

print("\nComputing PageRank...")
pr = nx.pagerank(G, weight="weight")
ranked = sorted(pr, key=pr.get, reverse=True)
print("Top 10 hubs by PageRank:")
for n in ranked[:10]:
    print(f"  {name(n)}  pr={pr[n]:.2e}")

src = max(outdeg, key=outdeg.get)
dst = max((n for n in indeg if n != src), key=indeg.get)
try:
    path = nx.shortest_path(G, src, dst)
    print(f"\nShortest path, {len(path) - 1} hops:")
    print("  " + "\n  -> ".join(name(n) for n in path))
except nx.NetworkXNoPath:
    print("\nNo directed path between the top broadcaster and the top integrator.")

H = G.subgraph(ranked[:SAMPLE_NODES])
pos = nx.spring_layout(H, seed=1, k=0.6)
top_pr = max(pr[n] for n in H)
sizes = [150 + 2500 * pr[n] / top_pr for n in H]
plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(H, pos, alpha=0.2, arrowsize=6, width=0.5)
nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color="tab:orange", alpha=0.85)
plt.title(f"Fruit fly connectome: top {SAMPLE_NODES} hubs by PageRank")
plt.axis("off")
plt.savefig("connectome/hubs.png", dpi=150, bbox_inches="tight")
print("\nSaved connectome/hubs.png")
