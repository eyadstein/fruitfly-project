import numpy as np
from fruitfly.foa import foa


def test_foa_minimizes_sphere():
    _, val, _ = foa(lambda p: float(np.sum(p**2)), [(-10, 10)] * 3, iters=200)
    assert val < 0.5
