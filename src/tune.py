"""Hyper-parameter sweep on the VALIDATION split (the test split is never used here) for the models whose
behaviour hinges on one regularisation knob:
  Logistic (L1) / (L2)  C          inverse regularisation strength (small C = strong penalty)
  GDA (Quadratic)       reg_param  covariance shrinkage (0 = plain QDA, 1 = per-class diagonal covariance)
  SVM (RBF)             C          margin penalty (trained on the same kind of subset as src.train)
Scores: validation PR-AUC (threshold-free, the right ranking metric at ~1:67) and the best validation F1.
Each model's value is CHOSEN on validation only; the chosen and the current values are then scored once on test.
  python -m src.tune                      -> outputs/tuning.csv, tuning_test.csv, tuning.png
  python -m src.tune --data freddie --folder data/freddie --macro data/macro.csv
"""
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import load_freddie, make_synthetic, split_by_loan, to_matrix
from .models import logistic, scores

# (model, parameter, values to try, value used in src/models.py, estimator factory(value, seed))
GRID = [
    ("Logistic (L1)", "C", [0.001, 0.01, 0.1, 1, 10], 1,
     lambda v, s: logistic("l1", solver="saga", C=v, max_iter=300, random_state=s)),
    ("Logistic (L2)", "C", [0.001, 0.01, 0.1, 1, 10], 1, lambda v, s: logistic("l2", C=v, max_iter=500)),
    ("GDA (Quadratic)", "reg_param", [0, 0.01, 0.1, 0.3, 0.6, 1.0], 0.1,
     lambda v, s: QuadraticDiscriminantAnalysis(reg_param=v)),
    ("SVM (RBF)", "C", [0.1, 1, 10], 1, lambda v, s: SVC(kernel="rbf", C=v, class_weight="balanced")),
]


def best_f1(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return float(np.max(2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "freddie"], default="synthetic")
    ap.add_argument("--folder", default="data/freddie")
    ap.add_argument("--macro", default=None)
    ap.add_argument("--loans", type=int, default=3000)
    ap.add_argument("--svm-sample", type=int, default=8000)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)

    df = make_synthetic(a.loans) if a.data == "synthetic" else load_freddie(a.folder, a.macro, a.loans)
    tr, va, te = split_by_loan(df, a.seed)  # same split as src.train
    Xtr, ytr = to_matrix(tr), tr.prepaid
    Xva, yva = to_matrix(va, list(Xtr.columns)), va.prepaid
    Xte, yte = to_matrix(te, list(Xtr.columns)), te.prepaid
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    p_idx = rng.choice(pos, min(2000, len(pos)), replace=False)
    svm_idx = np.concatenate([p_idx, rng.choice(neg, min(max(a.svm_sample - len(p_idx), 1), len(neg)), replace=False)])

    rows, fitted = [], {}
    for name, param, values, default, make in GRID:
        X, y = (Xtr.iloc[svm_idx], ytr.iloc[svm_idx]) if name.startswith("SVM") else (Xtr, ytr)
        for v in values:
            t0 = time.time()
            with warnings.catch_warnings():  # extreme settings (C=10 with saga, reg_param=0) warn by design
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.filterwarnings("ignore", message="The covariance matrix of class")
                m = make_pipeline(StandardScaler(), make(v, a.seed)).fit(X, y)
            fitted[name, v] = m
            s = scores(m, Xva)
            rows.append({"model": name, "param": param, "value": v, "is_default": v == default,
                         "val_pr_auc": average_precision_score(yva, s), "val_f1": best_f1(yva, s),
                         "fit_sec": round(time.time() - t0, 1)})
            r = rows[-1]
            print(f"{name:16s} {param}={v:<6g} PR-AUC={r['val_pr_auc']:.4f} F1={r['val_f1']:.4f}"
                  f"{'  (current default)' if r['is_default'] else ''}")
    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/tuning.csv", index=False)

    fig, axes = plt.subplots(1, len(GRID), figsize=(13, 3.4))
    for axi, (name, param, values, default, _) in zip(axes, GRID):
        g = res[res.model == name]
        x = np.arange(len(g))
        axi.plot(x, g.val_pr_auc, "o-", color="#ed8936", label="PR-AUC")
        axi.plot(x, g.val_f1, "s-", color="#2b6cb0", label="best F1")
        axi.axvline(values.index(default), color="grey", ls="--", lw=.8)
        axi.set_xticks(x, [str(v) for v in values], fontsize=8)
        axi.set_title(name, fontsize=10); axi.set_xlabel(param + "  (dashed = current)", fontsize=8)
        axi.grid(axis="y", color="#e2e8f0", lw=.6)
    axes[0].set_ylabel("validation score"); axes[0].legend(fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(f"{a.out}/tuning.png", dpi=150); plt.close()

    # one-time test check: the value chosen on validation vs the value currently used in src/models.py
    test = []
    for name, g in res.groupby("model", sort=False):
        b, d = g.loc[g.val_pr_auc.idxmax()], g[g.is_default].iloc[0]
        for kind, row in (("chosen on validation", b), ("current (src/models.py)", d)):
            test.append({"model": name, "setting": kind, "param": row.param, "value": row.value,
                         "val_pr_auc": row.val_pr_auc,
                         "test_pr_auc": average_precision_score(yte, scores(fitted[name, row.value], Xte))})
    test = pd.DataFrame(test)
    test.to_csv(f"{a.out}/tuning_test.csv", index=False)
    print("\nValue chosen on validation vs current value, each scored once on the test split (PR-AUC):")
    for name, g in test.groupby("model", sort=False):
        b, d = g.iloc[0], g.iloc[1]
        print(f"  {name:16s} chosen {b.param}={b.value:<6g} test {b.test_pr_auc:.4f}  |  "
              f"current {d.param}={d.value:<6g} test {d.test_pr_auc:.4f}  (diff {b.test_pr_auc - d.test_pr_auc:+.4f})")


if __name__ == "__main__":
    main()
