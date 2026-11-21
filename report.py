"""Print Tables IV and V of the paper from results/dgraphfin_results.json.

Precision, recall (with a 95% Wilson interval), balanced accuracy and MCC are
derived from the saved test confusion matrices; the all-normal and all-fraud
rows are constant-classifier references.
"""
import json
import math
import sys

from config import EXPERIMENTS


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def derived(tn, fp, fn, tp):
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn)
    ba = (rec + tn / (tn + fp)) / 2
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return prec, rec, ba, mcc


def main(path="results/dgraphfin_results.json"):
    results = json.load(open(path))
    print(f"{'Model':<18}{'ACC':>7}{'AUC':>7}{'F1':>7}{'Prec':>7}  {'Recall [95% CI]':<22}"
          f"{'BA':>7}{'MCC':>7}{'SP':>7}{'EO':>7}")
    for e, r in zip(EXPERIMENTS, results):
        (tn, fp), (fn, tp) = r["confusion_matrix"]
        prec, rec, ba, mcc = derived(tn, fp, fn, tp)
        lo, hi = wilson(tp, tp + fn)
        print(f"{e['name']:<18}{r['acc']:>7.3f}{r['auc']:>7.3f}{r['f1']:>7.3f}{prec:>7.3f}  "
              f"{rec:.3f} [{lo:.3f}, {hi:.3f}]   {ba:>7.3f}{mcc:>7.3f}{r['sp']:>7.3f}{r['eo']:>7.3f}")

    (tn, fp), (fn, tp) = results[0]["confusion_matrix"]
    n_pos, n_neg = tp + fn, tn + fp
    n = n_pos + n_neg
    print(f"{'All-normal':<18}{n_neg / n:>7.3f}{0.5:>7.3f}{0.0:>7.3f}{'---':>7}  {0.0:<22.3f}"
          f"{0.5:>7.3f}{0.0:>7.3f}{0:>7}{0:>7}")
    f1_all = 2 * (n_pos / n) / (n_pos / n + 1)
    print(f"{'All-fraud':<18}{n_pos / n:>7.3f}{0.5:>7.3f}{f1_all:>7.3f}{n_pos / n:>7.3f}  {1.0:<22.3f}"
          f"{0.5:>7.3f}{0.0:>7.3f}{0:>7}{0:>7}")

    print(f"\nTest set: {n:,} nodes, {n_pos} fraud.  Confusion matrices (TN FP FN TP):")
    for e, r in zip(EXPERIMENTS, results):
        (tn, fp), (fn, tp) = r["confusion_matrix"]
        print(f"  {e['name']:<18}{tn:>6}{fp:>6}{fn:>6}{tp:>6}")


if __name__ == "__main__":
    main(*sys.argv[1:])
