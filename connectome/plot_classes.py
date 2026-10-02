import sys, glob
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOP_PLOT = 80
TOP_COMP = 500


def read(pattern):
    f = glob.glob(f"connectome/data/{pattern}*")
    if not f:
        sys.exit(f"missing connectome/data/{pattern}*")
    return pd.read_csv(f[0])


conn = read("conn")
conn = (conn.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"]
            .sum().rename(columns={"syn_count": "weight"}))
cls = read("class")
sup = dict(zip(cls["root_id"], cls["super_class"].fillna("unknown")))
klass = dict(zip(cls["root_id"], cls["class"].fillna("")))
sub = dict(zip(cls["root_id"], cls["sub_class"].fillna("")))

G = nx.from_pandas_edgelist(conn, "pre_root_id", "post_root_id", edge_attr="weight", create_using=nx.DiGraph)
print(f"{G.number_of_nodes():,} neurons, {G.number_of_edges():,} connections")

print("Computing PageRank...")
pr = nx.pagerank(G, weight="weight")
ranked = sorted(pr, key=pr.get, reverse=True)

print("\nTop 15 hubs:")
print(f"{'id':>20}  {'super_class':<18}{'class':<22}{'sub_class':<18}{'in':>5}{'out':>6}")
for n in ranked[:15]:
    print(f"{n:>20}  {sup.get(n, '?'):<18}{str(klass.get(n, '')):<22}{str(sub.get(n, '')):<18}{G.in_degree(n):>5}{G.out_degree(n):>6}")

all_share = pd.Series([sup.get(n, "unknown") for n in G]).value_counts(normalize=True)
top_share = pd.Series([sup.get(n, "unknown") for n in ranked[:TOP_COMP]]).value_counts(normalize=True)
comp = pd.DataFrame({"all neurons": all_share, f"top {TOP_COMP} hubs": top_share}).fillna(0)
comp = comp.sort_values("all neurons", ascending=False)
print("\nClass share (%): whole brain vs top hubs")
print((comp * 100).round(1).to_string())

ax = comp.plot(kind="bar", figsize=(10, 5))
ax.set_ylabel("share of neurons")
ax.set_title("Which neuron classes are over-represented among hubs?")
plt.tight_layout()
plt.savefig("connectome/composition.png", dpi=150)
plt.close()

H = G.subgraph(ranked[:TOP_PLOT])
pos = nx.spring_layout(H, k=1.2, seed=1, weight=None)
classes = sorted({sup.get(n, "unknown") for n in H})
cmap = {c: plt.cm.tab10(i % 10) for i, c in enumerate(classes)}
top_pr = max(pr[n] for n in H)
maxw = max(d["weight"] for _, _, d in H.edges(data=True))

plt.figure(figsize=(12, 9))
nx.draw_networkx_edges(H, pos, alpha=0.25, arrowsize=5,
                       width=[0.2 + 2.5 * H[u][v]["weight"] / maxw for u, v in H.edges])
nx.draw_networkx_nodes(H, pos, node_size=[100 + 1500 * pr[n] / top_pr for n in H],
                       node_color=[cmap[sup.get(n, "unknown")] for n in H], alpha=0.9)
nx.draw_networkx_labels(H, pos, labels={n: f"{sup.get(n, '?')}\n..{str(n)[-4:]}" for n in ranked[:8]}, font_size=7)
for c in classes:
    plt.scatter([], [], color=cmap[c], label=c)
plt.legend(title="super_class", loc="lower left")
plt.title(f"Top {TOP_PLOT} connectome hubs by PageRank, colored by neuron class")
plt.axis("off")
plt.savefig("connectome/hubs_classes.png", dpi=150, bbox_inches="tight")
print("\nSaved connectome/composition.png and connectome/hubs_classes.png")
