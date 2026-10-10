import json
from pathlib import Path

SEEDS = [5, 6, 7, 8, 9]


def load(tag):
    out = {}
    for sd in SEEDS:
        f = Path(f"results/capstone_seed{sd}_{tag}.json")
        if f.exists():
            out[sd] = json.load(open(f))
    return out


learn, ctl = load("learn"), load("control")
seeds = sorted(set(learn) & set(ctl))
if not seeds:
    raise SystemExit("no held-out seed has both a learning and a control run yet")
d0 = learn[seeds[0]]
t_flip, t_total = d0["t_flip"], d0["t_total"]
late = t_flip + (t_total - t_flip) / 2


def count(d, lo, hi):
    blk = [x[1] for x in d["visits"] if lo <= x[0] < hi]
    return blk.count("A"), blk.count("B")


def share(d, lo, hi, good):
    a, b = count(d, lo, hi)
    n = a + b
    return ((a if good == "A" else b) / n if n else None), a, b


def fmt(sh, a, b):
    return f"{a}/{b} " + (f"{sh:.0%}" if sh is not None else "-")


windows = {"phase 1 (share at A)": (0, t_flip, "A"),
           "phase 2 (share at B)": (t_flip, t_total + 1, "B"),
           "late phase 2 (share at B, not in the criterion)": (late, t_total + 1, "B")}
wins, pooled = {}, {}
print(f"held-out seeds complete (both conditions): {seeds}; flip at {t_flip:.0f} s of {t_total:.0f} s")
for name, (lo, hi, good) in windows.items():
    print(f"\n{name}")
    print(f"{'seed':<6}{'learning':>14}{'control':>14}{'ahead?':>9}")
    w, tl, tc = 0, [0, 0], [0, 0]
    for sd in seeds:
        sl, al, bl = share(learn[sd], lo, hi, good)
        sc, ac, bc = share(ctl[sd], lo, hi, good)
        ok = sl is not None and sc is not None and sl > sc
        w += ok
        print(f"{sd:<6}{fmt(sl, al, bl):>14}{fmt(sc, ac, bc):>14}{'yes' if ok else 'no':>9}")
        tl[0] += al; tl[1] += bl; tc[0] += ac; tc[1] += bc
    kl, kc_ = (tl[0], tc[0]) if good == "A" else (tl[1], tc[1])
    pl = kl / sum(tl) if sum(tl) else None
    pc = kc_ / sum(tc) if sum(tc) else None
    wins[name], pooled[name] = w, (pl, pc)
    print(f"pooled: learning {fmt(pl, *tl)}, control {fmt(pc, *tc)}; learning ahead in {w} of {len(seeds)} seeds")

print("\nboxes and valence")
for sd in seeds:
    v = learn[sd]["visits"]
    print(f"seed {sd}: learning {len(v)} visits, control {len(ctl[sd]['visits'])}; outside the box {learn[sd]['outside']:.1%} / "
          f"{ctl[sd]['outside']:.1%}, furthest {learn[sd]['furthest']:.1f} / {ctl[sd]['furthest']:.1f} mm; "
          f"end valence A {v[-1][2]:+.2f}, B {v[-1][3]:+.2f}" if v else f"seed {sd}: no learning visits")

print("\nCRITERION (fixed in advance)")
keys = list(windows)[:2]
pool_ok = all(pooled[k][0] is not None and pooled[k][1] is not None and pooled[k][0] > pooled[k][1] for k in keys)
maj = all(wins[k] >= 3 for k in keys)
strict = all(wins[k] >= 4 for k in keys)
for k in keys:
    print(f"{k}: pooled learning {pooled[k][0]:.0%} vs control {pooled[k][1]:.0%}; learning ahead in {wins[k]} of {len(seeds)} seeds")
if len(seeds) < 5:
    print(f"verdict: INCOMPLETE ({len(seeds)} of 5 seeds)")
else:
    print("verdict:", "MET" if pool_ok and maj else "NOT MET", f"(stricter reading, 4 of 5 in both phases: {'holds' if pool_ok and strict else 'does not hold'})")
