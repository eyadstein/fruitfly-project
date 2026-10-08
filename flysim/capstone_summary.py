import glob
import json

files = sorted(glob.glob("results/capstone_seed*_*.json"))
data = {"learn": {}, "control": {}}
for f in files:
    name = f.replace("\\", "/").split("/")[-1][:-5]
    _, sd, tag = name.split("_")
    data[tag][int(sd[4:])] = json.load(open(f))
seeds = sorted(set(data["learn"]) & set(data["control"]))
if not seeds:
    raise SystemExit("no seed has both a learning and a control run yet")
first = data["learn"][seeds[0]]
t_flip, t_total = first["t_flip"], first["t_total"]


def count(v, lo, hi):
    blk = [x[1] for x in v if lo <= x[0] < hi]
    return blk.count("A"), blk.count("B")


def share(a, b, good):
    n = a + b
    return f"{(a if good == 'A' else b) / n:.0%}" if n else "-"


print(f"seeds complete (both conditions): {seeds}; flip at {t_flip:.0f} s of {t_total:.0f} s")
print("\nper seed: visits to A / B, and share at the rewarded odour")
print(f"{'seed':<6}{'condition':<10}{'phase 1 A/B':>13}{'share':>7}{'phase 2 A/B':>13}{'share':>7}{'outside':>9}{'furthest':>10}")
for sd in seeds:
    for tag in ("learn", "control"):
        d = data[tag][sd]
        a1, b1 = count(d["visits"], 0, t_flip)
        a2, b2 = count(d["visits"], t_flip, t_total + 1)
        print(f"{sd:<6}{tag:<10}{f'{a1}/{b1}':>13}{share(a1, b1, 'A'):>7}{f'{a2}/{b2}':>13}{share(a2, b2, 'B'):>7}"
              f"{d['outside']:>9.1%}{d['furthest']:>9.1f}")

print("\npooled over seeds, by 20 s bin (visits A/B and share at the rewarded odour)")
print(f"{'time (s)':<11}{'rewarded':>9}{'learning':>13}{'share':>7}{'control':>13}{'share':>7}")
for lo in range(0, int(t_total), 20):
    good = "A" if lo < t_flip else "B"
    row = f"{lo:>3}-{lo + 20:<7}{good:>9}"
    for tag in ("learn", "control"):
        a = b = 0
        for sd in seeds:
            x, y = count(data[tag][sd]["visits"], lo, lo + 20)
            a, b = a + x, b + y
        row += f"{f'{a}/{b}':>13}{share(a, b, good):>7}"
    print(row)

print("\npooled phases")
late = t_flip + (t_total - t_flip) / 2
for name, lo, hi, good in (("phase 1", 0, t_flip, "A"), ("phase 2", t_flip, t_total + 1, "B"),
                           ("late phase 2", late, t_total + 1, "B")):
    for tag in ("learn", "control"):
        a = b = 0
        for sd in seeds:
            x, y = count(data[tag][sd]["visits"], lo, hi)
            a, b = a + x, b + y
        print(f"{name:<13}{tag:<9} visits A {a:>3}, B {b:>3}; share at {good}: {share(a, b, good)}")

print("\nlearned valence in the learning runs (after the last visit before the flip / at the end)")
for sd in seeds:
    v = data["learn"][sd]["visits"]
    pre = [x for x in v if x[0] < t_flip]
    head = f"before flip A {pre[-1][2]:+.2f}, B {pre[-1][3]:+.2f}; " if pre else "no visit before the flip; "
    print(f"seed {sd}: {head}end A {v[-1][2]:+.2f}, B {v[-1][3]:+.2f}")
