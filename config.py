"""All settings used in the paper, kept in one place.

Change values here (or override alpha/beta/epochs from the command line in
run_experiments.py). The defaults reproduce the reported results.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    path: str = "data/DGraphFin/dgraphfin.npz"
    seed: int = 42                  # NumPy seed: fraud seeds for BFS and the random split
    n_fraud_seeds: int = 500        # BFS starts from this many random fraud nodes
    max_visited: int = 50_000       # BFS stops expanding after visiting this many nodes
    max_extra_edges: int = 500_000  # safety cap for the second edge scan (never reached)
    train_frac: float = 0.6         # random 60/20/20 split, not stratified
    val_frac: float = 0.2


@dataclass
class ModelConfig:
    gcn_hidden: int = 128           # residual GCN: d -> 128 -> 128 -> 64
    emb_dim: int = 64               # size of the embedding h_v
    cohort_dim: int = 16            # z_s, first 16 dims of h_v (cohort subspace)
    mlp_hidden: int = 128           # MLP encoder: 64 -> 128 -> 64
    task_head: tuple = (64, 32)     # fraud predictor: 48 -> 64 -> 32 -> 2
    cohort_head: tuple = (32, 16)   # cohort predictor: 16 -> 32 -> 16 -> 2
    gcn_dropout: float = 0.3
    mlp_dropout: float = 0.2


@dataclass
class TrainConfig:
    alpha: float = 0.0              # weight of the decorrelation loss L_dis
    beta: float = 0.0               # weight of the soft statistical-parity penalty
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 5e-4
    seed: int = 42                  # torch seed: weight init and dropout
    max_fraud_weight: float = 15.0  # w1 = min(N0/N1, 15) in the weighted cross-entropy
    grad_clip: float = 1.0
    eval_every: int = 10            # validation check (checkpoint, scheduler, early stop)
    patience: int = 80              # early stopping, counted in epochs
    lr_factor: float = 0.5          # ReduceLROnPlateau settings
    lr_patience: int = 30           # counted in validation checks, not epochs
    min_lr: float = 1e-5
    thresholds: tuple = (0.10, 0.90, 0.02)  # validation-F1 threshold grid: start, stop, step


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# The three settings compared in the paper.
EXPERIMENTS = [
    {"name": "Temporal backbone", "alpha": 0.0, "beta": 0.0},
    {"name": "Moderate", "alpha": 0.01, "beta": 1.0},
    {"name": "Strong", "alpha": 0.05, "beta": 5.0},
]
