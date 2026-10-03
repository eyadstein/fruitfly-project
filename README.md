# Fruit Fly Toolkit

A small project that borrows two ideas from the fruit fly, tests them with measurements, and then connects them to the FlyWire connectome and a simulated fly body.

- **FlyHash:** the fly's olfactory circuit (sparse random expansion plus winner-take-all) used as a similarity-search index.
- **FOA:** the Fruit Fly Optimization Algorithm, used to tune FlyHash.
- **Connectome:** analysis of the FlyWire FAFB v783 brain, and a PN to Kenyon cell circuit built from the real wiring.
- **NeuroMechFly:** a simulated fly (FlyGym 2.1.0) that walks and steers toward an odour.

## Main findings

1. Tuned FlyHash matches exact cosine search on retrieval quality for real text (0.395 vs 0.399 same-newsgroup precision), at 0.83 recall of the exact neighbours.
2. FOA and random search tie on this tuning problem.
3. The four biggest mushroom-body hubs in the connectome are APL and DPM neurons.
4. The specific PN to Kenyon cell wiring never beat a degree-preserving shuffle in any test (recall, text retrieval, steering, odour discrimination). What mattered was sparse expansion with winner-take-all, and how evenly the PNs are used.
5. In steering, a sparse KC code with a readout discriminates two odours that a plain concentration signal cannot.

## Repository layout

| Path | What it does |
|---|---|
| `fruitfly/`, `app.py`, `tests/` | FlyHash index, FOA, FastAPI service (`/index`, `/search`, `/optimize/sphere`), pytest |
| `Dockerfile`, `.github/workflows/ci.yml` | Container and CI that runs the tests |
| `benchmark.py`, `tune.py`, `tune2.py`, `tune3.py` | Recall benchmark and FOA vs random tuning |
| `text_demo.py` | FlyHash on 20 Newsgroups text |
| `connectome/` | Graph analysis, hub plots, real-wiring FlyHash benchmarks |
| `flysim/` | NeuroMechFly demos: standing, walking, steering, controller comparisons |
| `figures/` | Result figures |

## Setup

- Core toolkit: Python with numpy, fastapi, uvicorn, pytest (`pip install -r requirements.txt`). Run `python -m pytest -q` and `uvicorn app:app --reload`.
- Connectome scripts: also need pandas, networkx, matplotlib, pyarrow, scipy. Download the FAFB v783 **Connections (Filtered)**, **Classification / Hierarchical Annotations** and **Consolidated cell types** tables from codex.flywire.ai into `connectome/data/` (not committed).
- Simulation scripts: FlyGym supports Python 3.9 to 3.12, so use a separate environment on Python 3.12 with `pip install flygym tqdm`. `text_demo.py` also needs scikit-learn.

## Results

### FlyHash tuning (1,000 random 128-d vectors, held-out queries, 3 seeds)

| Setting | recall@10 | ms/query |
|---|---|---|
| Default (expansion 20, wta 0.05, sample 0.10) | 0.345 | 4.3 |
| Random search, wta capped at 0.10 | 0.650 +/- 0.026 | 13.4 |
| FOA, wta capped at 0.10 | 0.667 +/- 0.028 | 15.2 |

- Tuning nearly doubles recall. FOA and random search are tied.
- Best settings sit at the search bounds, so recall is limited by code size, at about 3.5x the query time.
- Without the sparsity cap, both methods reach about 0.758 with wta near 0.30, so keeping the code fly-sparse costs about 0.09 recall.

### FlyHash on real text (20 Newsgroups, TF-IDF + SVD to 142 dimensions)

3,000 indexed documents, 150 queries per seed, 3 seeds, k=10. Precision is the fraction of retrieved documents from the query's own newsgroup (chance 0.05).

| Method | Recall vs exact cosine | Same-newsgroup precision |
|---|---|---|
| Exact cosine | 1.000 | 0.399 |
| FlyHash default | 0.521 | 0.316 |
| FlyHash tuned (exp 75, wta 0.10, sf 0.28) | 0.826 | 0.395 |
| Real PN to KC wiring, wta 0.05 / 0.10 | 0.320 / 0.328 | 0.260 / 0.277 |
| Shuffled wiring, wta 0.05 / 0.10 | 0.333 / 0.335 | 0.270 / 0.277 |
| Random in-degree, wta 0.05 / 0.10 | 0.390 / 0.542 | 0.304 / 0.354 |

- The tuned code uses 10,650 cells per document (about 1.3 KB as bits against 568 bytes for the raw vector), so there is no storage saving at these settings.
- Real wiring ties a degree-preserving shuffle. Evenly spread random wiring is better, increasingly so at higher sparsity.

### Connectome (FlyWire FAFB v783)

- 138,584 neurons and 3,732,460 connections (at least 5 synapses). Optic neurons are 56% of the brain but about 13% of the top 500 hubs by PageRank. Descending neurons are about 1% of the brain and 21% of those hubs.
- PageRank rewards neurons that receive from many well-connected neurons, so sensory neurons (sources) rank low by construction.
- The four biggest mushroom-body hubs are two APL and two DPM neurons (checked against the cell-type table).
- Real circuit (right hemisphere, connections of at least 5 synapses): 142 PNs, 2,376 Kenyon cells, 10,702 connections, about 4.5 inputs per Kenyon cell. The top 10 PNs hold 28.8% of the connections (even spread would be 7.0%).
- recall@10 against exact cosine on random vectors, 5 seeds: real wiring and a degree-preserving shuffle are tied in every setting; random wiring with matched in-degree is better by about 0.05 to 0.12 at wta 0.05 to 0.10.

### Simulated fly

A FlyGym walking demo reproduces the official tutorial (27.3 mm in 2 s against 27.34 mm). Steering uses the hybrid turning controller driven by a left/right antenna signal.

Single odour, 6 target angles x 3 seeds, target 12.8 mm away, arrival is within 3 mm:

| Controller | Arrived | Median time | Mean closest |
|---|---|---|---|
| Plain odour (gain 3.8) | 10/18 | 1.04 s | 3.5 mm |
| Circuit, real wiring (gain 3.0) | 12/18 | 1.06 s | 3.1 mm |
| Circuit, shuffled wiring (gain 3.0) | 12/18 | 1.07 s | 3.2 mm |

Two odours (A rewarded at 12.8 mm, B distractor at 8.0 mm, closer and on the opposite side; nearly uncorrelated PN patterns, about 93 KCs active each), 4 angles x 3 seeds:

| Controller | Reached A | Reached B | Neither | Median time to A |
|---|---|---|---|---|
| Plain concentration | 0 | 3 | 9 | - |
| Circuit, real wiring + readout | 11 | 0 | 1 | 1.04 s |
| Circuit, shuffled wiring + readout | 12 | 0 | 0 | 1.06 s |

![Fly paths on the two-odour task](figures/two_odour_paths.png)

## Caveats

- Synthetic and classic TF-IDF inputs, not real odour data or neural embeddings.
- The connection table is filtered at 5 synapses, and each PN is treated as an independent input dimension.
- The odour discrimination readout is set directly from the two odours (no learning), with one odour pair and 12 trials per controller.
- Sample sizes are small, so differences of one or two trials are within noise.
