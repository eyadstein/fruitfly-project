import glob
import json

data = {"learn": {}, "control": {}}
for f in sorted(glob.glob("results/capstone_seed*_*.json")):
    name = f.replace("\\", "/").split("/")[-1][:-5]
    _, sd, tag = name.split("_")
    data[tag][int(sd[4:])] = json.load(open(f))
seeds = sorted(set(data["learn"]) & set(data["control"]))
d0 = data["learn"][seeds[0]]
t_flip, t_total = d0["t_flip"], d0["t_total"]
late = t_flip + (t_total - t_flip) / 2


def count(v, lo, hi):
    blk = [x[1] for x in v if lo <= x[0] < hi]
    return blk.count("A"), blk.count("B")


def cell(k, n):
    return f"{k}/{n} " + (f"{k / n:.0%}" if n else "-")


windows = [("last 20 s before the flip", t_flip - 20, t_flip, "A"),
           ("first 20 s after the flip", t_flip, t_flip + 20, "B"),
           ("late phase 2", late, t_total + 1, "B")]
for title, lo, hi, good in windows:
    print(f"\n{title} ({lo:.0f}-{min(hi, t_total):.0f} s), visits at the rewarded odour {good}")
    print(f"{'seed':<6}{'learning':>14}{'control':>14}{'difference':>12}")
    pos = 0
    for sd in seeds:
        cells, shares = [], []
        for tag in ("learn", "control"):
            a, b = count(data[tag][sd]["visits"], lo, hi)
            k, n = (a, a + b) if good == "A" else (b, a + b)
            cells.append(cell(k, n))
            shares.append(k / n if n else float("nan"))
        diff = shares[0] - shares[1]
        pos += diff > 0
        print(f"{sd:<6}{cells[0]:>14}{cells[1]:>14}{diff:>+12.0%}")
    print(f"seeds where learning > control: {pos} of {len(seeds)}")
