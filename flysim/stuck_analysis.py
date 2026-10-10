import json
from pathlib import Path

print("after the flip, no-decay learning runs (seeds 0-9)")
print(f"{'seed':<6}{'visits A/B':>11}{'1st B visit':>12}{'A visits before':>16}{'B>A at':>9}{'B share after':>15}  sequence of visits after the flip")
for sd in range(10):
    f = Path(f"results/capstone_seed{sd}_learn.json")
    if not f.exists():
        continue
    d = json.load(open(f))
    tf = d["t_flip"]
    v = [x for x in d["visits"] if x[0] >= tf]
    a, b = sum(x[1] == "A" for x in v), sum(x[1] == "B" for x in v)
    firstB = next((x[0] - tf for x in v if x[1] == "B"), None)
    aBefore = sum(x[1] == "A" for x in v if firstB is not None and x[0] - tf < firstB) if firstB is not None else a
    cross = next((x[0] - tf for x in v if x[3] > x[2]), None)
    after = [x for x in v if cross is not None and x[0] - tf >= cross]
    bshare = f"{sum(x[1] == 'B' for x in after)}/{len(after)}" if after else "-"
    seq = "".join(x[1] for x in v)
    print(f"{sd:<6}{f'{a}/{b}':>11}{(f'{firstB:.0f} s' if firstB is not None else 'never'):>12}{aBefore:>16}"
          f"{(f'{cross:.0f} s' if cross is not None else 'never'):>9}{bshare:>15}  {seq}")
