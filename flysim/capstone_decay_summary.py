import glob
import json
from pathlib import Path


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        out[int(Path(f).stem.split("seed")[1].split("_")[0])] = json.load(open(f))
    return out


old = load("results/capstone_seed*_learn.json")
ctl = load("results/capstone_seed*_control.json")
new = load("results/capstone_decay_seed*_learn.json")
seeds = sorted(set(new) & set(old) & set(ctl))
if not seeds:
    raise SystemExit("no time-decay run has finished yet")
t_flip, t_total = old[seeds[0]]["t_flip"], old[seeds[0]]["t_total"]
late = t_flip + (t_total - t_flip) / 2


def count(d, lo, hi):
    blk = [x[1] for x in d["visits"] if lo <= x[0] < hi]
    return blk.count("A"), blk.count("B")


def win(d):
    return count(d, 0, t_flip), count(d, t_flip, t_total + 1), count(d, late, t_total + 1)


def pct(k, n):
    return f"{k / n:.0%}" if n else "-"


rows = {"no-decay learning": old, "time-decay learning": new, "control": ctl}
print(f"seeds complete: {seeds}; flip at {t_flip:.0f} s; late phase {late:.0f}-{t_total:.0f} s")
print(f"\n{'seed':<6}{'condition':<21}{'phase 1 A/B':>12}{'at A':>6}{'phase 2 A/B':>13}{'at B':>6}{'late A/B':>10}{'at B':>6}  end valence")
tot = {n: [0] * 6 for n in rows}
for sd in seeds:
    for n, d in rows.items():
        (a1, b1), (a2, b2), (a3, b3) = win(d[sd])
        v = d[sd]["visits"]
        end = f"A {v[-1][2]:+.2f}, B {v[-1][3]:+.2f}" if (n != "control" and v) else ""
        print(f"{sd:<6}{n:<21}{f'{a1}/{b1}':>12}{pct(a1, a1 + b1):>6}{f'{a2}/{b2}':>13}{pct(b2, a2 + b2):>6}{f'{a3}/{b3}':>10}{pct(b3, a3 + b3):>6}  {end}")
        for j, x in enumerate((a1, b1, a2, b2, a3, b3)):
            tot[n][j] += x
print("\npooled over seeds")
for n, t in tot.items():
    print(f"{n:<21} phase 1 at A {pct(t[0], t[0] + t[1])} ({t[0]}/{t[1]}), phase 2 at B {pct(t[3], t[2] + t[3])} ({t[2]}/{t[3]}), "
          f"late at B {pct(t[5], t[4] + t[5])} ({t[4]}/{t[5]})")

late_ok = []
for sd in seeds:
    a3, b3 = win(new[sd])[2]
    late_ok.append(a3 + b3 >= 5 and b3 / (a3 + b3) >= 0.6)
t = tot["time-decay learning"]
p1 = t[0] / (t[0] + t[1]) if t[0] + t[1] else 0
print("\nCRITERION (fixed in advance) for the time-decay learning runs")
print(f"late-phase share at B >= 60% with >= 5 visits: {sum(late_ok)} of {len(seeds)} seeds {late_ok}")
print(f"pooled phase-1 share at A >= 65%: {p1:.0%} -> {'yes' if p1 >= 0.65 else 'no'}")
if len(seeds) < 5:
    print(f"verdict: INCOMPLETE ({len(seeds)} of 5 seeds)")
else:
    print("verdict:", "MET" if all(late_ok) and p1 >= 0.65 else "NOT MET")
