"""Steering signal through the real PN to KC wiring, with sparse KC coding."""
from pathlib import Path
import numpy as np
from odour_steer import antenna_positions, concentration

_d = np.load(Path(__file__).resolve().parents[1] / "connectome" / "pn_kc_matrix.npz", allow_pickle=True)
W = _d["W"].astype(np.float64)

ODOUR = np.random.default_rng(0).random(W.shape[1])
C_REF = 0.2
THETA = np.percentile(W @ (C_REF * ODOUR), 50)
K = int(0.05 * W.shape[0])
EPS = 1e-9


def kc_total(conc):
    """Total activity of the K most driven KCs above threshold."""
    drive = W @ (conc * ODOUR) - THETA
    top = np.partition(drive, -K)[-K:]
    return np.maximum(top, 0.0).sum()


def circuit_error(pos, heading, source):
    """Signed value between minus one and one. Positive means more KC drive on the left."""
    left_pt, right_pt = antenna_positions(pos, heading)
    t_left = kc_total(concentration(left_pt, source))
    t_right = kc_total(concentration(right_pt, source))
    return (t_left - t_right) / (t_left + t_right + EPS)


if __name__ == "__main__":
    print("source on left :", round(circuit_error([0, 0], 0.0, [5, 5]), 4))
    print("source on right:", round(circuit_error([0, 0], 0.0, [5, -5]), 4))
    print("source ahead   :", round(circuit_error([0, 0], 0.0, [8, 0]), 4))
