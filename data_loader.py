"""Load DGraphFin, take a fraud-seeded subgraph and build the model inputs.

Steps (Section V-A of the paper):
  1. BFS from 500 random fraud nodes over the full graph (incl. background nodes)
     until 50,000 nodes have been visited.
  2. Keep only labelled nodes (class 0 = normal, 1 = fraud) and the edges between them.
  3. Build a fixed recency-weighted adjacency  a_e = 0.5 + 0.5 * t_e / t_max,
     symmetrise with max(A, A^T), add self-loops and normalise.
  4. Min-max scale the 17 node features.
  5. Random 60/20/20 split.
  6. Build the binary early/late temporal cohort used as the group attribute.
"""
import time
from collections import deque

import numpy as np
import scipy.sparse as sp
import torch


def _neighbour_lists(edge_index, n_nodes):
    """CSR-style neighbour lists over the undirected view of the graph.

    For every node the neighbours are listed in edge order, so the BFS visits
    nodes in exactly the same order as a plain dict-of-lists implementation.
    """
    src, dst = edge_index[:, 0], edge_index[:, 1]
    edge_ids = np.arange(len(src))
    node = np.concatenate([src, dst])
    neighbour = np.concatenate([dst, src])
    eid = np.concatenate([edge_ids, edge_ids])
    side = np.repeat([0, 1], len(src))
    order = np.lexsort((side, eid, node))
    indptr = np.concatenate([[0], np.cumsum(np.bincount(node, minlength=n_nodes))])
    return indptr, neighbour[order], eid[order]


def _bfs_sample(y, edge_index, cfg):
    """Fraud-seeded BFS. Returns visited node ids and the edge ids touched."""
    fraud = np.where(y == 1)[0]
    seeds = np.random.choice(fraud, size=min(cfg.n_fraud_seeds, len(fraud)), replace=False)
    indptr, neighbours, edge_ids = _neighbour_lists(edge_index, len(y))

    visited = set(seeds.tolist())
    queue = deque(seeds.tolist())
    touched_edges = set()
    while len(visited) < cfg.max_visited and queue:
        u = queue.popleft()
        for v, e in zip(neighbours[indptr[u]:indptr[u + 1]].tolist(),
                        edge_ids[indptr[u]:indptr[u + 1]].tolist()):
            touched_edges.add(e)
            if v not in visited and len(visited) < cfg.max_visited:
                visited.add(v)
                queue.append(v)

    # If the BFS runs out of nodes early, top up with random labelled nodes.
    if len(visited) < cfg.max_visited:
        labelled = np.where((y == 0) | (y == 1))[0]
        pool = np.setdiff1d(labelled, list(visited))
        extra = np.random.choice(pool, size=min(cfg.max_visited - len(visited), len(pool)), replace=False)
        visited.update(extra.tolist())
    return visited, touched_edges


def _normalised_adjacency(edges, weights, n):
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    # Symmetrise: A_uv = max(B_uv, B_vu)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    with_self = sp.coo_matrix(adj + sp.eye(n))
    deg = np.array(with_self.sum(1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d = sp.diags(d_inv_sqrt)
    norm = d.dot(with_self).dot(d).tocoo()
    indices = torch.from_numpy(np.vstack((norm.row, norm.col)).astype(np.int64))
    values = torch.from_numpy(norm.data.astype(np.float32))
    return adj, torch.sparse_coo_tensor(indices, values, torch.Size([n, n]), check_invariants=True)


def _temporal_cohort(edges, timestamps, adj, n):
    """Early (0) / late (1) cohort: sum of incident timestamps divided by the
    weighted degree, split at the median of the positive values."""
    score = np.zeros(n, dtype=np.float32)
    for (s, d), t in zip(edges, timestamps):
        score[s] += t
        score[d] += t
    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1
    score /= degree
    median = np.median(score[score > 0]) if np.any(score > 0) else 0
    return (score > median).astype(np.int64)


def load_dgraphfin(cfg):
    t0 = time.time()
    print(f"Loading {cfg.path} ...")
    raw = np.load(cfg.path)
    x, y = raw["x"].astype(np.float32), raw["y"]
    edge_index, edge_type, edge_time = raw["edge_index"], raw["edge_type"], raw["edge_timestamp"]

    np.random.seed(cfg.seed)
    visited, touched_edges = _bfs_sample(y, edge_index, cfg)

    # Keep labelled nodes only (background classes 2 and 3 are dropped).
    nodes = np.sort(list(visited))
    nodes = nodes[np.isin(y[nodes], [0, 1])][:cfg.max_visited]
    n = len(nodes)
    new_id = np.full(len(y), -1, dtype=np.int64)
    new_id[nodes] = np.arange(n)

    # Edges with both endpoints kept: first those met during BFS, then any others.
    kept = lambda ids: ids[(new_id[edge_index[ids, 0]] >= 0) & (new_id[edge_index[ids, 1]] >= 0)]
    bfs_ids = kept(np.fromiter(touched_edges, dtype=np.int64, count=len(touched_edges)))
    in_bfs = np.zeros(len(edge_index), dtype=bool)
    in_bfs[list(touched_edges)] = True
    other_ids = kept(np.where(~in_bfs)[0])[:max(0, cfg.max_extra_edges - len(bfs_ids))]
    eids = np.concatenate([bfs_ids, other_ids])

    edges = new_id[edge_index[eids]]
    etype, etime = edge_type[eids], edge_time[eids]

    weights = 0.5 + 0.5 * etime.astype(np.float32) / etime.max()
    adj_raw, adj = _normalised_adjacency(edges, weights, n)

    feats = torch.FloatTensor(x[nodes])
    fmin, fmax = feats.min(dim=0)[0], feats.max(dim=0)[0]
    span = fmax - fmin
    span[span == 0] = 1.0
    feats = (feats - fmin) / span
    labels = torch.LongTensor(y[nodes])

    perm = np.random.permutation(n)
    n_train, n_val = int(cfg.train_frac * n), int(cfg.val_frac * n)
    masks = []
    for part in (perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]):
        m = torch.zeros(n, dtype=torch.bool)
        m[part] = True
        masks.append(m)

    cohort = torch.LongTensor(_temporal_cohort(edges, etime, adj_raw, n))

    hist, bins = np.histogram(etime, bins=20)
    types, type_counts = np.unique(etype, return_counts=True)
    stats = {
        "n_nodes": int(n), "n_edges": int(len(edges)),
        "n_fraud": int((labels == 1).sum()), "n_normal": int((labels == 0).sum()),
        "timestamp_range": [int(etime.min()), int(etime.max())],
        "n_edge_types": int(len(types)),
        "edge_type_dist": {int(k): int(v) for k, v in zip(types, type_counts)},
        "timestamp_dist": hist.tolist(), "timestamp_bins": bins.tolist(),
    }
    print(f"  {n:,} labelled nodes, {len(edges):,} edges, {stats['n_fraud']:,} fraud "
          f"| split {int(masks[0].sum())}/{int(masks[1].sum())}/{int(masks[2].sum())} "
          f"| cohorts {int((cohort == 0).sum())}/{int((cohort == 1).sum())} | {time.time() - t0:.0f}s")
    return {"adj": adj, "features": feats, "labels": labels, "cohort": cohort,
            "train": masks[0], "val": masks[1], "test": masks[2], "stats": stats}
