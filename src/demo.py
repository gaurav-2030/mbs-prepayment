"""Live demo: score loan-months with the saved best model.
  python -m src.demo --random 6            # 6 held-out TEST loan-months (half prepaid) from the training dataset
  python -m src.demo --loan examples/loan.json
  python -m src.demo --loan '{"credit_score":760,"orig_interest_rate":6.5,"mtg_rate":4.5,"current_upb":90000,"loan_age":30}'
--loan takes a JSON file path or an inline JSON dict; unspecified fields fall back to typical values."""
import argparse
import json
import os
import joblib
import pandas as pd
from .data import make_synthetic, to_matrix

TYPICAL = {"credit_score": 745, "dti": 34, "ltv": 76, "cltv": 77, "orig_upb": 200000, "orig_interest_rate": 5.2,
           "loan_age": 24, "current_upb": 180000, "months_to_maturity": 336, "current_interest_rate": 5.2,
           "mtg_rate": 5.2, "hpi_change": 4, "unemployment": 6, "occupancy_status": "P", "loan_purpose": "P", "channel": "R"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/model.joblib")
    ap.add_argument("--loan", default=None, help="JSON file path or inline JSON dict of loan features")
    ap.add_argument("--random", type=int, default=0)
    a = ap.parse_args()
    b = joblib.load(a.model)
    m, cols = b["model"], b["columns"]
    print(f"Model: {b['name']} trained on {b.get('data', 'synthetic')} data (decision threshold {b['threshold']:.4f})")
    if a.loan:
        if os.path.exists(a.loan):
            with open(a.loan) as f:
                loan = json.load(f)
        else:
            loan = json.loads(a.loan)
        d = {**TYPICAL, **loan}
        d["rate_incentive"] = d["orig_interest_rate"] - d["mtg_rate"]
        df = pd.DataFrame([d])
        truth = None
    else:
        # held-out test rows saved by src.train; older model files fall back to fresh synthetic loans
        pool = b["demo_rows"] if "demo_rows" in b else make_synthetic(300, seed=7)
        k = a.random or 6
        pos, neg = pool[pool.prepaid == 1], pool[pool.prepaid == 0]
        df = pd.concat([pos.sample(min(k // 2, len(pos)), random_state=1),
                        neg.sample(min(k - k // 2, len(neg)), random_state=1)])
        truth = df.prepaid.values
    X = to_matrix(df, cols)
    s = m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)
    pred = s >= b["threshold"]
    for i in range(len(df)):
        t = "" if truth is None else f" | actual: {'PREPAID' if truth[i] else 'no prepay'}"
        print(f"loan-month {i + 1}: score={s[i]:.4f} -> {'PREPAYMENT' if pred[i] else 'no prepayment'}{t}")


if __name__ == "__main__":
    main()
