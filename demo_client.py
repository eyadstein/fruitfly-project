import json, urllib.request
import numpy as np

def post(path, payload):
    req = urllib.request.Request(
        "http://127.0.0.1:8000" + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req))

rng = np.random.default_rng(5)
data = rng.normal(size=(100, 128))
print(post("/index", {"ids": [f"doc{i}" for i in range(100)], "vectors": data.tolist()}))

query = data[7] + rng.normal(scale=0.1, size=128)
print(post("/search", {"vector": query.tolist(), "k": 3}))
