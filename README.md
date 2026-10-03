# Fruit Fly Toolkit

FlyHash similarity search + Fruit Fly Optimization Algorithm, served with FastAPI.

## Run
    python -m pytest -q
    uvicorn app:app --reload

## Results

Recall@10 against exact cosine search (1,000 random 128-d vectors, noisy held-out queries, 3 seeds):

| Setting | recall@10 | ms/query |
|---|---|---|
| Default (expansion 20, wta 0.05, sample 0.10) | 0.345 | 4.3 |
| Random search, 96 evaluations | 0.650 +/- 0.026 | 13.4 |
| FOA, 8 flies x 12 iterations | 0.667 +/- 0.028 | 15.2 |

- Tuning nearly doubles recall over the defaults.
- FOA and random search are statistically tied on this 3-parameter problem.
- Best settings sit at the search bounds (expansion ~75-80, wta 0.10, sample_frac ~0.28-0.30), so recall is limited by code size, at the cost of ~3.5x query time.
- Relaxing sparsity (wta ~0.30) reached 0.758, so the sparse cap costs about 0.09 recall.
- Caveat: random Gaussian data is a hard, unrealistic test; real embeddings are the next step.

## Connectome experiment (FlyWire FAFB v783)

Built a PN -> Kenyon cell projection from the real connectome (142 PNs, 2,376 KCs, 10,702 connections, right hemisphere, connections >= 5 synapses) and compared it with random wiring as the FlyHash expansion layer.

- The four biggest mushroom-body hubs are APL and DPM neurons (checked against the cell-type table), the feedback neurons associated with keeping the KC code sparse.
- recall@10 vs exact cosine (5 seeds): real wiring and a degree-preserving shuffle are tied in every setting; random wiring with matched KC in-degree is better by ~0.05-0.12 at wta 0.05-0.10.
- Interpretation: the specific PN->KC pairing does not matter on this data; the uneven PN fan-out costs recall. Real data and recall-vs-discrimination tasks are untested.
- Caveats: synthetic inputs, each PN treated as an independent dimension, filtered connection table.

## Steering experiment (NeuroMechFly 2.1.0, hybrid turning controller)

A simulated fly walks toward an odour source, steering from a left/right antenna signal. Controllers compared: plain odour difference, and a sparse PN->Kenyon-cell circuit built from the FlyWire wiring (threshold + top-5% winner-take-all as an APL stand-in), with real and degree-preserving shuffled wiring.

6 target angles x 3 seeds per controller, target 12.8 mm away, arrival = within 3 mm:

| Controller | Arrived | Median time | Mean closest |
|---|---|---|---|
| Plain odour (gain 3.8) | 10/18 | 1.04 s | 3.5 mm |
| Circuit, real wiring (gain 3.0) | 12/18 | 1.06 s | 3.1 mm |
| Circuit, shuffled wiring (gain 3.0) | 12/18 | 1.07 s | 3.2 mm |

- Real and shuffled wiring perform identically; the circuit acts as a gain/nonlinearity stage, not a wiring-specific one.
- With a single odour every channel scales with the same concentration, so PN identity cannot matter here.
- The plain-vs-circuit gap (2 trials of 18) is within chance and confounded by different gains.
- Untested: two odours with different PN patterns, where a KC code could discriminate and a plain concentration signal cannot.

## Two-odour discrimination (steering toward A, away from a closer distractor B)

Odour A (rewarded, 12.8 mm) and odour B (distractor, 8.0 mm, closer, opposite side) with nearly uncorrelated PN patterns (r = -0.04, ~93 KCs active each). The circuit uses a readout set from the two odours' KC codes (+1 on A's cells, -1 on B's). 4 angles x 3 seeds per controller:

| Controller | Reached A | Reached B | Neither | Median time to A |
|---|---|---|---|---|
| Plain concentration | 0 | 3 | 9 | - |
| Circuit, real wiring + readout | 11 | 0 | 1 | 1.04 s |
| Circuit, shuffled wiring + readout | 12 | 0 | 0 | 1.06 s |

- A sparse KC code with a readout discriminates odours that a plain concentration signal cannot.
- Real and shuffled wiring are indistinguishable (11 vs 12 of 12): the discrimination comes from sparse expansion and winner-take-all, not the specific PN->KC pairing. This matches the FlyHash recall benchmark and the single-odour steering result.
- Caveats: the readout is set directly from the two odours (no learning), one odour pair, 12 trials per controller.
