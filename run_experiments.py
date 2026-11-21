"""Run the three FairGRUNT-T settings from the paper and save the results.

    python run_experiments.py                      # all three settings (paper)
    python run_experiments.py --alpha 0.01 --beta 1  # a single custom setting

Outputs (in results/):
    dgraphfin_results.json  test metrics, threshold, confusion matrix, logged history
    dgraphfin_stats.json    statistics of the sampled graph (used for the figures)
"""
import argparse
import json
import os
from dataclasses import replace

from config import EXPERIMENTS, Config
from data_loader import load_dgraphfin
from train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, help="decorrelation weight (single run)")
    parser.add_argument("--beta", type=float, help="parity-penalty weight (single run)")
    parser.add_argument("--epochs", type=int, help="override the epoch limit")
    parser.add_argument("--data", help="path to dgraphfin.npz")
    parser.add_argument("--out", default="results", help="output folder")
    args = parser.parse_args()

    cfg = Config()
    if args.data:
        cfg.data = replace(cfg.data, path=args.data)
    if args.epochs:
        cfg.train = replace(cfg.train, epochs=args.epochs)

    if args.alpha is not None or args.beta is not None:
        settings = [{"name": "custom", "alpha": args.alpha or 0.0, "beta": args.beta or 0.0}]
    else:
        settings = EXPERIMENTS

    data = load_dgraphfin(cfg.data)
    results = [train(data, cfg.model, replace(cfg.train, alpha=e["alpha"], beta=e["beta"]))
               for e in settings]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "dgraphfin_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "dgraphfin_stats.json"), "w") as f:
        json.dump(data["stats"], f)

    print(f"\n{'setting':<20}{'ACC':>8}{'AUC':>8}{'F1':>8}{'SP':>8}{'EO':>8}{'tau':>7}")
    for e, r in zip(settings, results):
        print(f"{e['name']:<20}{r['acc']:>8.3f}{r['auc']:>8.3f}{r['f1']:>8.3f}"
              f"{r['sp']:>8.3f}{r['eo']:>8.3f}{r['threshold']:>7.2f}")
    print(f"\nSaved to {args.out}/. Run `python report.py` for the full paper table.")


if __name__ == "__main__":
    main()
