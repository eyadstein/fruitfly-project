"""Odour signal for steering. Replacement for the geometric angle error.

Use odour_error(...) where steer.py computes err, then keep the rest of the
drive logic unchanged. Positions are 2D arrays in mm, heading is in radians.
"""
import numpy as np

SOURCE_STRENGTH = 1.0
DECAY_LENGTH = 6.0
ANTENNA_FORWARD = 0.5
ANTENNA_SPREAD = 0.4
EPS = 1e-9


def concentration(point, source):
    d = np.linalg.norm(np.asarray(point) - np.asarray(source))
    return SOURCE_STRENGTH / (1.0 + (d / DECAY_LENGTH) ** 2)


def antenna_positions(pos, heading):
    fwd = np.array([np.cos(heading), np.sin(heading)])
    left_dir = np.array([-np.sin(heading), np.cos(heading)])
    base = np.asarray(pos) + ANTENNA_FORWARD * fwd
    return base + ANTENNA_SPREAD * left_dir, base - ANTENNA_SPREAD * left_dir


def odour_error(pos, heading, source):
    """Signed value between minus one and one. Positive means more odour on the left."""
    left_pt, right_pt = antenna_positions(pos, heading)
    c_left = concentration(left_pt, source)
    c_right = concentration(right_pt, source)
    return (c_left - c_right) / (c_left + c_right + EPS)


if __name__ == "__main__":
    print("source on left :", round(odour_error([0, 0], 0.0, [5, 5]), 4))
    print("source on right:", round(odour_error([0, 0], 0.0, [5, -5]), 4))
    print("source ahead   :", round(odour_error([0, 0], 0.0, [8, 0]), 4))
