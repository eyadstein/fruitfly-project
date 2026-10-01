import numpy as np


def foa(f, bounds, n_flies=30, iters=100, step_frac=0.2, decay=0.97, seed=0):
    """Minimize f over bounds=[(lo, hi), ...]. Returns (best_x, best_value, history)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.array(bounds, dtype=float).T
    swarm = rng.uniform(lo, hi)
    best_x, best_val = swarm.copy(), f(swarm)
    step = (hi - lo) * step_frac
    history = [best_val]

    for _ in range(iters):
        flies = swarm + rng.uniform(-1, 1, (n_flies, len(lo))) * step
        flies = np.clip(flies, lo, hi)
        vals = np.array([f(p) for p in flies])
        i = vals.argmin()
        if vals[i] < best_val:
            best_val, best_x = vals[i], flies[i].copy()
            swarm = best_x                           # swarm flies to the best smell
        step *= decay
        history.append(best_val)
    return best_x, best_val, history
