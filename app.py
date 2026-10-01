import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fruitfly.index import FlyIndex
from fruitfly.foa import foa

app = FastAPI(title="Fruit Fly Toolkit")
DIM = 128
index = FlyIndex(DIM)


class AddReq(BaseModel):
    ids: list[str]
    vectors: list[list[float]]


class SearchReq(BaseModel):
    vector: list[float]
    k: int = 5


@app.post("/index")
def add(req: AddReq):
    index.add(req.ids, req.vectors)
    return {"total": len(index.ids)}


@app.post("/search")
def search(req: SearchReq):
    return {"results": index.search(req.vector, req.k)}


@app.get("/optimize/sphere")
def optimize(dim: int = 5, iters: int = 100):
    x, val, hist = foa(lambda p: float(np.sum(p**2)), [(-10, 10)] * dim, iters=iters)
    return {"best_x": x.tolist(), "best_value": val, "history": hist[::10]}
