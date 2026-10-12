"""Train + evaluate all models. Usage:
  python -m src.train --data synthetic
  python -m src.train --data freddie --folder data/freddie [--macro data/macro.csv]
"""
import argparse
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler

from .data import load_freddie, make_synthetic, split_by_loan, to_matrix
from .models import build_models, is_kernel_svm, oversample_positives, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "freddie"], default="synthetic")
    ap.add_argument("--folder", default="data/freddie")
    ap.add_argument("--macro", default=None)
    ap.add_argument("--loans", type=int, default=3000, help="synthetic loans / loans per Freddie file")
    ap.add_argument("--svm-sample", type=int, default=8000, help="training rows for kernel SVMs")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.RandomState(a.seed)

    df = make_synthetic(a.loans) if a.data == "synthetic" else load_freddie(a.folder, a.macro, a.loans)
    tr, va, te = split_by_loan(df, a.seed)
    Xtr, ytr = to_matrix(tr), tr.prepaid
    cols = list(Xtr.columns)
    Xva, yva = to_matrix(va, cols), va.prepaid
    Xte, yte = to_matrix(te, cols), te.prepaid
    info = {"data": a.data, "rows": len(df), "loans": int(df.loan_id.nunique()), "features": len(cols),
            "prepay_rate": float(df.prepaid.mean()), "imbalance": f"1:{(1 - df.prepaid.mean()) / df.prepaid.mean():.0f}",
            "train": len(tr), "val": len(va), "test": len(te), "test_positives": int(yte.sum())}
    print(info)

    # PCA visualisation (balanced sample)
    pos = Xtr[ytr == 1].sample(min(500, int(ytr.sum())), random_state=a.seed)
    neg = Xtr[ytr == 0].sample(len(pos), random_state=a.seed)
    P = PCA(2).fit_transform(StandardScaler().fit_transform(pd.concat([pos, neg])))
    plt.figure(figsize=(5, 4))
    plt.scatter(P[len(pos):, 0], P[len(pos):, 1], s=6, c="red", alpha=.5, label="No prepayment")
    plt.scatter(P[:len(pos), 0], P[:len(pos), 1], s=6, c="blue", alpha=.5, label="Prepayment")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA (balanced sample)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{a.out}/pca.png", dpi=150); plt.close()

    results, curves, cms, fitted = [], {}, {}, {}
    for name, model in build_models(a.seed).items():
        t0 = time.time()
        if is_kernel_svm(name):  # kernel matrix is O(n^2): train on a subset (all-positive-capped)
            p_idx = rng.choice(np.where(ytr == 1)[0], min(2000, int(ytr.sum())), replace=False)
            neg = np.where(ytr == 0)[0]
            n_idx = rng.choice(neg, min(max(a.svm_sample - len(p_idx), 1), len(neg)), replace=False)
            idx = np.concatenate([p_idx, n_idx])
            model.fit(Xtr.iloc[idx], ytr.iloc[idx])
        elif name == "Neural Network":
            model.fit(*oversample_positives(Xtr, ytr, 10, a.seed))
        else:
            model.fit(Xtr, ytr)
        # decision threshold tuned on the VALIDATION split (max F1) - 0.5 is meaningless at ~1:50 imbalance
        sv = scores(model, Xva)
        pv, rv, tv = precision_recall_curve(yva, sv)
        f1v = 2 * pv[:-1] * rv[:-1] / np.maximum(pv[:-1] + rv[:-1], 1e-12)
        thr = float(tv[int(np.argmax(f1v))])
        s = scores(model, Xte)
        pred = (s >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        r = {"model": name, "accuracy": (tp + tn) / len(yte), "precision": precision_score(yte, pred, zero_division=0),
             "recall_prepay": recall_score(yte, pred), "specificity": tn / (tn + fp), "f1": f1_score(yte, pred),
             "roc_auc": roc_auc_score(yte, s), "pr_auc": average_precision_score(yte, s),
             "threshold": thr, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "fit_sec": round(time.time() - t0, 1)}
        results.append(r); cms[name] = [[tn, fp], [fn, tp]]; fitted[name] = model
        curves[name] = (roc_curve(yte, s), precision_recall_curve(yte, s))
        print(f"{name:22s} acc={r['accuracy']:.4f} rec={r['recall_prepay']:.3f} prec={r['precision']:.3f} "
              f"f1={r['f1']:.3f} auc={r['roc_auc']:.3f} pr={r['pr_auc']:.3f} ({r['fit_sec']}s)")

    res = pd.DataFrame(results)
    res.to_csv(f"{a.out}/results.csv", index=False)
    with open(f"{a.out}/results.json", "w") as f:
        json.dump({"info": info, "results": results}, f, indent=2)

    # figures: confusion matrices, ROC / PR, L1 feature importance
    fig, ax = plt.subplots(3, 3, figsize=(9, 8))
    for axi, (n, cm) in zip(ax.ravel(), cms.items()):
        shade = np.log1p(cm)
        axi.imshow(shade, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):  # white text on dark cells, black on light ones
            axi.text(j, i, f"{v}", ha="center", va="center", fontsize=9,
                     color="white" if shade[i][j] > (shade.min() + shade.max()) / 2 else "black")
        axi.set_title(n, fontsize=9); axi.set_xticks([0, 1]); axi.set_yticks([0, 1])
        axi.set_xlabel("Predicted"); axi.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(f"{a.out}/confusion_matrices.png", dpi=150); plt.close()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for n, ((fpr, tpr, _), (pr, rc, _)) in curves.items():
        a1.plot(fpr, tpr, label=n, lw=1); a2.plot(rc, pr, label=n, lw=1)
    a1.plot([0, 1], [0, 1], "k--", lw=.5); a1.set_xlabel("FPR"); a1.set_ylabel("TPR"); a1.set_title("ROC")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_title("Precision-Recall"); a2.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(f"{a.out}/roc_pr.png", dpi=150); plt.close()

    coef = fitted["Logistic (L1)"][-1].coef_[0]
    imp = pd.Series(np.abs(coef), index=cols).sort_values().tail(10)
    plt.figure(figsize=(6, 4)); imp.plot.barh(color="#2b6cb0"); plt.title("Top 10 features (|coef|, L1 logistic, standardised)")
    plt.tight_layout(); plt.savefig(f"{a.out}/feature_importance.png", dpi=150); plt.close()
    imp.sort_values(ascending=False).to_csv(f"{a.out}/feature_importance.csv", header=["abs_coef"])

    # F1 / PR-AUC bar chart (an image, so it renders the same in PowerPoint, Keynote and Google Slides)
    x, wd = np.arange(len(res)), 0.38
    fig, axb = plt.subplots(figsize=(9.2, 3.9))
    for k, (col, lab, colr) in enumerate([("f1", "F1", "#2b6cb0"), ("pr_auc", "PR-AUC", "#ed8936")]):
        bars = axb.bar(x + (k - 0.5) * wd, res[col], wd, label=lab, color=colr)
        axb.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    axb.set_xticks(x, [m.replace(" (", "\n(") for m in res.model], fontsize=8)
    axb.set_ylabel("Score"); axb.set_ylim(0, res[["f1", "pr_auc"]].max().max() * 1.2)
    axb.spines[["top", "right"]].set_visible(False); axb.grid(axis="y", color="#e2e8f0", lw=.6); axb.set_axisbelow(True)
    axb.legend(frameon=False, ncol=2, loc="upper right")
    plt.tight_layout(); plt.savefig(f"{a.out}/metrics_bar.png", dpi=200); plt.close()

    # save best model by PR-AUC (right metric under heavy imbalance)
    best = res.sort_values("pr_auc", ascending=False).iloc[0].model

    # large, readable confusion matrix of the best model (for the slides)
    cm = np.array(cms[best]); shade = np.log1p(cm)
    fig, axc = plt.subplots(figsize=(4.2, 3.6))
    axc.imshow(shade, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axc.text(j, i, f"{['TN', 'FP', 'FN', 'TP'][2 * i + j]}\n{v:,}", ha="center", va="center", fontsize=12,
                 color="white" if shade[i][j] > (shade.min() + shade.max()) / 2 else "black")
    axc.set_xticks([0, 1], ["No prepay", "Prepaid"]); axc.set_yticks([0, 1], ["No prepay", "Prepaid"])
    axc.set_xlabel("Predicted"); axc.set_ylabel("Actual"); axc.set_title(f"{best} - test set", fontsize=11)
    plt.tight_layout(); plt.savefig(f"{a.out}/best_confusion.png", dpi=200); plt.close()
    # keep a few held-out TEST loan-months so the demo scores real unseen rows of the same dataset
    demo_rows = pd.concat([te[te.prepaid == 1].sample(min(100, int(yte.sum())), random_state=a.seed),
                           te[te.prepaid == 0].sample(min(100, int((yte == 0).sum())), random_state=a.seed)])
    joblib.dump({"model": fitted[best], "columns": cols, "name": best, "data": a.data, "demo_rows": demo_rows,
                 "threshold": float(res.set_index("model").loc[best, "threshold"])}, f"{a.out}/model.joblib")
    print("Best by PR-AUC:", best)


if __name__ == "__main__":
    main()
