# FairGRUNT-T

Code for the paper **"FairGRUNT-T: Temporal Representation Disentanglement for Fair
Fraud Detection on Dynamic Financial Graphs"** (A. M. Khaise, J. Ravindran, S. S. Muni,
Digital University Kerala).

FairGRUNT-T adapts FairGRUNT-style representation disentanglement (Santos et al., ASONAM 2025)
to the timestamped DGraphFin fraud graph. The embedding of each node is split into a
task part and a cohort part. The model trains with a weighted fraud loss, a cohort loss,
a decorrelation penalty between the two parts and a soft statistical-parity penalty.

## Repository layout

```
config.py           all hyperparameters and the three experiment settings
data_loader.py      DGraphFin loading, fraud-seeded BFS sample, recency adjacency,
                    feature scaling, 60/20/20 split, temporal cohort
model.py            residual GCN, MLP encoder, fraud head and cohort head
train.py            training loop for one (alpha, beta) setting
metrics.py          SP / EO gaps, AUC, validation-F1 threshold search
run_experiments.py  entry point: runs the three settings and saves results/
report.py           prints the paper's results table from results/
make_figures.py     draws every figure in the paper into figures/
results/            the saved results reported in the paper
data/DGraphFin/     put dgraphfin.npz here (not included, see below)
```

## Setup

```bash
pip install -r requirements.txt
```

A GPU is not needed. Training runs on CPU.

**Data.** DGraphFin is distributed by Finvolution under its own licence, so it is not
included here. Register and download `dgraphfin.npz` from <https://dgraph.xinye.com>
and place it at `data/DGraphFin/dgraphfin.npz`. The file holds 3,700,550 nodes,
4,300,999 edges, 17 features, 11 edge types and edge timestamps (see
`data/DGraphFin/DATASET_README.md`).

## Running

```bash
python run_experiments.py   # trains the three settings, writes results/*.json
python report.py            # results table (ACC, AUC, F1, precision, recall + CI, BA, MCC, SP, EO)
python make_figures.py      # all paper figures -> figures/
```

`report.py` and `make_figures.py` work from the saved files in `results/`, so you can
check the paper's numbers and figures without retraining.

Other options:

```bash
python run_experiments.py --alpha 0.01 --beta 1.0   # one custom setting
python run_experiments.py --epochs 100              # shorter run
python run_experiments.py --data path/to/dgraphfin.npz --out my_results
```

## Where the parameters are

Every setting lives in `config.py`, grouped by stage:

| Group | Parameter | Value | Meaning |
|---|---|---|---|
| `DataConfig` | `seed` | 42 | NumPy seed for BFS seeds and the split |
| | `n_fraud_seeds` | 500 | number of fraud nodes the BFS starts from |
| | `max_visited` | 50,000 | BFS stops after visiting this many nodes |
| | `train_frac`, `val_frac` | 0.6, 0.2 | random split (test = remaining 0.2) |
| `ModelConfig` | `gcn_hidden`, `emb_dim` | 128, 64 | GCN widths d → 128 → 128 → 64 |
| | `cohort_dim` | 16 | size of z_s, so z_c has 48 dimensions |
| | `task_head`, `cohort_head` | (64, 32), (32, 16) | hidden widths of the two heads |
| | `gcn_dropout`, `mlp_dropout` | 0.3, 0.2 | dropout rates |
| `TrainConfig` | `alpha` | 0 / 0.01 / 0.05 | weight of the decorrelation loss |
| | `beta` | 0 / 1 / 5 | weight of the soft SP penalty |
| | `lr`, `weight_decay` | 1e-3, 5e-4 | Adam |
| | `epochs`, `patience` | 500, 80 | epoch limit and early stopping |
| | `max_fraud_weight` | 15 | cap on the fraud class weight |
| | `thresholds` | 0.10 to 0.88, step 0.02 | validation-F1 threshold grid |

The three settings compared in the paper are listed in `EXPERIMENTS` at the bottom of
`config.py`.

## How the pipeline works

1. **Sampling** (`data_loader.py`). A breadth-first search starts from 500 random fraud
   nodes and walks the full graph, including background nodes, until 50,000 nodes have
   been visited. Only labelled nodes (normal/fraud) and the edges between them are kept.
   This gives 17,098 nodes, 10,012 edges and 862 fraud cases.
2. **Graph.** Each edge gets a fixed recency weight `a_e = 0.5 + 0.5 * t_e / t_max`. The
   matrix is symmetrised by taking the maximum, then self-loops are added and it is
   degree-normalised. It has no trainable parameters.
3. **Cohort.** DGraphFin has no demographic attribute. Each node gets the sum of its
   edge timestamps divided by its weighted degree, and is split into early (0) and
   late (1) at the median. SP and EO are measured between these two proxy groups.
4. **Model** (`model.py`). A three-layer residual GCN feeds an MLP encoder that outputs
   a 64-dimensional embedding `h`. The fraud head sees `z_c = h[16:]` and the cohort head
   sees `z_s = h[:16]`.
5. **Training** (`train.py`). Each epoch makes three updates: the fraud head on a
   detached `z_c`, the cohort head on a detached `z_s`, then the encoders on
   `(1 - alpha)(L_cls + L_sens) + alpha * L_dis + beta * SP_soft`. Every 10 epochs the
   validation AUC selects the checkpoint and drives the LR scheduler and early stopping.
6. **Evaluation.** The best checkpoint is restored. The decision threshold is the one
   with the highest validation F1, and test metrics are computed at that threshold.
   Test metrics logged during training are for monitoring only.

## Expected results

All three settings select τ = 0.58. Test set: 3,421 nodes, 168 fraud.

| Setting | α | β | AUC | F1 | Recall | SP gap | EO gap |
|---|---|---|---|---|---|---|---|
| Temporal backbone | 0 | 0 | 0.686 | 0.157 | 0.339 | 0.098 | 0.057 |
| Moderate | 0.01 | 1 | 0.676 | 0.151 | 0.536 | 0.112 | 0.029 |
| Strong | 0.05 | 5 | 0.651 | 0.135 | 0.268 | 0.134 | 0.225 |

Each setting is a single run with seed 42. With the package versions in
`requirements.txt`, rerunning on CPU reproduces these numbers. Other PyTorch versions or
hardware may give slightly different values, because sparse operations and BatchNorm are
not bit-identical across builds.

## Limitations

These are discussed in the paper. The sample is fraud-seeded, the split is random rather
than chronological, the SP penalty uses all nodes (transductive), the cohort is a
timestamp proxy rather than a protected attribute, and each setting was trained once.

## Acknowledgement

The disentanglement design follows FairGRUNT by F. Santos, P.-N. Tan and A.-H. Esfahanian
("Mitigating bias for unseen demographic groups in graph neural networks", ASONAM 2025).
The dataset is DGraph (Huang et al., NeurIPS 2022 Datasets and Benchmarks).
