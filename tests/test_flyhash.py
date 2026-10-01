import numpy as np
from fruitfly.index import FlyIndex


def test_nearest_neighbor_is_itself():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(200, 128))
    idx = FlyIndex(128)
    idx.add([str(i) for i in range(200)], data)
    assert idx.search(data[17], k=1)[0][0] == "17"


def test_noisy_query_still_finds_original():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(200, 128))
    idx = FlyIndex(128)
    idx.add([str(i) for i in range(200)], data)
    noisy = data[42] + rng.normal(scale=0.1, size=128)
    assert idx.search(noisy, k=3)[0][0] == "42"
