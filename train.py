"""Training loop for one (alpha, beta) setting of FairGRUNT-T.

Each epoch has three updates (Algorithm 1 in the paper):
  1. fraud head on detached z_c (weighted cross-entropy, training nodes)
  2. cohort head on detached z_s (cross-entropy, training nodes)
  3. GCN + MLP encoder on
       L = (1 - alpha) * (L_cls + L_sens) + alpha * L_dis + beta * SP_soft
     where L_dis is the largest |Pearson correlation| between a z_s and a z_c
     coordinate (training nodes) and SP_soft is the gap in mean fraud score
     between the two cohorts (computed over all nodes, i.e. transductively).

Every `eval_every` epochs the validation AUC drives checkpointing, the LR
schedulers and early stopping. Test metrics are only logged for monitoring.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from metrics import best_f1_threshold, fairness_gaps, safe_auc
from model import FairGRUNTT


def _adam(params, cfg):
    return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _plateau(opt, cfg):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)


def _test_metrics(probs, pred, data):
    m = data["test"]
    y, s = data["labels"][m].numpy(), data["cohort"][m].numpy()
    p, yhat = probs[m].numpy(), pred[m]
    sp, eo = fairness_gaps(yhat, y, s)
    return {"acc": accuracy_score(y, yhat), "auc": safe_auc(y, p),
            "f1": f1_score(y, yhat, zero_division=0), "sp": sp, "eo": eo}


def train(data, model_cfg, cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    x, adj, y, s = data["features"], data["adj"], data["labels"], data["cohort"]
    tr, va = data["train"], data["val"]
    model = FairGRUNTT(x.shape[1], model_cfg)
    k = model.cohort_dim

    opt_gcn, opt_enc = _adam(model.gcn.parameters(), cfg), _adam(model.encoder.parameters(), cfg)
    opt_task, opt_cohort = _adam(model.task_head.parameters(), cfg), _adam(model.cohort_head.parameters(), cfg)
    schedulers = [_plateau(o, cfg) for o in (opt_gcn, opt_enc, opt_task)]  # cohort head: no scheduler

    # Fraud weight from training labels only: w1 = min(N0 / N1, cap)
    w1 = min(((y[tr] == 0).sum().float() / (y[tr] == 1).sum().float()).item(), cfg.max_fraud_weight)
    task_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, w1]))
    cohort_loss = nn.CrossEntropyLoss()
    print(f"\n--- alpha={cfg.alpha}, beta={cfg.beta} | fraud weight {w1:.2f} ---")

    history = {key: [] for key in ("epoch", "loss", "acc", "auc", "sp", "eo", "f1")}
    best_auc, best_state, waited = 0.0, None, 0

    for epoch in range(cfg.epochs):
        model.train()
        for opt in (opt_gcn, opt_enc, opt_task, opt_cohort):
            opt.zero_grad()

        # 1-2. Update the two heads on detached embeddings.
        z_s, z_c, _ = model.embed(x, adj)
        task_loss(model.task_head(z_c.detach())[tr], y[tr]).backward()
        nn.utils.clip_grad_norm_(model.task_head.parameters(), cfg.grad_clip)
        opt_task.step()
        cohort_loss(model.cohort_head(z_s.detach())[tr], s[tr]).backward()
        nn.utils.clip_grad_norm_(model.cohort_head.parameters(), cfg.grad_clip)
        opt_cohort.step()

        # 3. Update the encoders with the full objective.
        opt_gcn.zero_grad()
        opt_enc.zero_grad()
        z_s, z_c, h = model.embed(x, adj)
        logits = model.task_head(z_c)
        l_cls = task_loss(logits[tr], y[tr])
        l_sens = cohort_loss(model.cohort_head(z_s)[tr], s[tr])
        l_dis = torch.abs(torch.corrcoef(h[tr].T))[:k, k:].max()
        fraud_score = F.softmax(logits, dim=1)[:, 1]
        sp_soft = torch.abs(fraud_score[s == 0].mean() - fraud_score[s == 1].mean())
        loss = (1 - cfg.alpha) * (l_cls + l_sens) + cfg.alpha * l_dis + cfg.beta * sp_soft
        loss.backward()
        nn.utils.clip_grad_norm_(model.gcn.parameters(), cfg.grad_clip)
        nn.utils.clip_grad_norm_(model.encoder.parameters(), cfg.grad_clip)
        opt_gcn.step()
        opt_enc.step()

        if epoch % cfg.eval_every and epoch != cfg.epochs - 1:
            continue

        # Validation check
        model.eval()
        probs = model.fraud_probability(x, adj)
        val_auc = safe_auc(y[va].numpy(), probs[va].numpy())
        for sch in schedulers:
            sch.step(val_auc)
        if val_auc > best_auc:
            best_auc, waited = val_auc, 0
            best_state = {n: t.clone() for n, t in model.state_dict().items()}
        else:
            waited += cfg.eval_every

        # Monitoring only: test metrics at the argmax decision (ties go to "normal")
        logged = _test_metrics(probs, (probs > 0.5).long().numpy(), data)
        history["epoch"].append(epoch)
        history["loss"].append(float(loss.item()))
        for key in ("acc", "auc", "sp", "eo", "f1"):
            history[key].append(float(logged[key]))
        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} | val AUC {val_auc:.4f} "
                  f"| test AUC {logged['auc']:.4f} | lr {opt_gcn.param_groups[0]['lr']:.1e}")

        if waited >= cfg.patience:
            print(f"  early stop at epoch {epoch} (best val AUC {best_auc:.4f})")
            break

    # Restore the best checkpoint, pick the threshold on validation F1, evaluate on test.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probs = model.fraud_probability(x, adj)
    tau = best_f1_threshold(y[va].numpy(), probs[va].numpy(), cfg.thresholds)
    pred = (probs >= tau).long().numpy()
    final = _test_metrics(probs, pred, data)
    cm = confusion_matrix(y[data["test"]].numpy(), pred[data["test"].numpy()])

    result = {"alpha": cfg.alpha, "beta": cfg.beta, **{k_: float(v) for k_, v in final.items()},
              "threshold": float(tau), "confusion_matrix": cm.tolist(), "history": history}
    print(f"  test: ACC {final['acc']:.4f} AUC {final['auc']:.4f} F1 {final['f1']:.4f} "
          f"SP {final['sp']:.4f} EO {final['eo']:.4f} | tau {tau:.2f} | CM {cm.tolist()}")
    return result
