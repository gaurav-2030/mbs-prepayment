"""Data loading: real Freddie Mac Single-Family Loan-Level data, or a synthetic stand-in.

Target: prepaid = 1 in the loan's final record when it terminated via voluntary
prepayment (zero balance code '01'), else 0.
"""
import glob
import os
import numpy as np
import pandas as pd

ORIG_COLS = ["credit_score", "first_pmt_date", "first_time_homebuyer", "maturity_date", "msa", "mi_pct",
             "num_units", "occupancy_status", "cltv", "dti", "orig_upb", "ltv", "orig_interest_rate",
             "channel", "ppm_flag", "amortization_type", "property_state", "property_type", "postal_code",
             "loan_seq_no", "loan_purpose", "orig_loan_term", "num_borrowers", "seller_name",
             "servicer_name", "super_conforming_flag", "pre_relief_refi_loan_seq_no", "program_indicator",
             "relief_refi_indicator", "property_valuation_method", "interest_only_indicator",
             "mi_cancellation_indicator"]  # 32 columns (Freddie Mac SF LLD user guide)
PERF_COLS = ["loan_seq_no", "period", "current_upb", "delinquency", "loan_age", "months_to_maturity",
             "defect_date", "mod_flag", "zero_balance_code", "zero_balance_date", "current_interest_rate",
             "current_deferred_upb", "ddlpi", "mi_recoveries", "net_sale_proceeds", "non_mi_recoveries",
             "expenses", "legal_costs", "maintenance_costs", "taxes_insurance", "misc_expenses",
             "actual_loss", "mod_cost", "step_mod_flag", "deferred_payment_mod", "eltv",
             "zero_balance_removal_upb", "delinquent_accrued_interest", "delinquency_due_to_disaster",
             "borrower_assistance_status", "current_month_mod_cost", "interest_bearing_upb"]  # 32 columns
NUM = ["credit_score", "dti", "ltv", "cltv", "orig_upb", "orig_interest_rate", "loan_age", "current_upb",
       "months_to_maturity", "current_interest_rate", "mtg_rate", "rate_incentive", "hpi_change", "unemployment"]
CAT = ["occupancy_status", "loan_purpose", "channel"]


def load_freddie(folder, macro_csv=None, n_loans_per_file=5000, seed=0):
    """Load sample_orig_*.txt + sample_svcg_*.txt (pipe-delimited, no header) from `folder`.
    Only 30-year (360-month) loans are kept: zero balance code 01 means "prepaid OR matured", and shorter
    loans in the sample (15/20-year) can reach maturity, which would be mislabelled as prepayment.
    macro_csv (optional): columns period (YYYYMM), mtg_rate, hpi_change, unemployment."""
    rng = np.random.RandomState(seed)
    frames = []
    for of in sorted(glob.glob(os.path.join(folder, "*orig_*.txt"))):
        sf = of.replace("orig_", "svcg_")
        if not os.path.exists(sf):
            continue
        # index_col=False: if a file has more/fewer columns than names, never shift columns into the index
        o = pd.read_csv(of, sep="|", header=None, names=ORIG_COLS, index_col=False, low_memory=False)
        o = o[o.orig_loan_term == 360]
        ids = rng.choice(np.asarray(o.loan_seq_no.unique()), min(n_loans_per_file, o.loan_seq_no.nunique()), replace=False)
        o = o[o.loan_seq_no.isin(ids)]
        p = pd.read_csv(sf, sep="|", header=None, names=PERF_COLS, index_col=False, low_memory=False)
        p = p[p.loan_seq_no.isin(ids)]
        frames.append(p.merge(o, on="loan_seq_no", how="inner"))
    if not frames:
        raise FileNotFoundError(f"No Freddie Mac sample_orig_*/sample_svcg_* files in {folder}")
    df = pd.concat(frames, ignore_index=True).sort_values(["loan_seq_no", "period"], ignore_index=True)
    # current_upb is 0 in the payoff month, which leaks the label: use the balance at the START of the month
    df["current_upb"] = df.groupby("loan_seq_no").current_upb.shift(1).fillna(df.orig_upb)
    df["prepaid"] = (df.zero_balance_code.astype(str).str.replace(".0", "", regex=False).str.zfill(2) == "01").astype(int)
    df["credit_score"] = df.credit_score.replace(9999, np.nan)
    for c in ["dti", "ltv", "cltv"]:
        df[c] = df[c].replace(999, np.nan)
    if macro_csv:
        df = df.merge(pd.read_csv(macro_csv), on="period", how="left")
    for c in ["mtg_rate", "hpi_change", "unemployment"]:
        if c not in df:
            df[c] = np.nan
    df["rate_incentive"] = df.orig_interest_rate - df.mtg_rate
    df["loan_id"] = df.loan_seq_no
    return _finalise(df)


def make_synthetic(n_loans=3000, seed=42):
    """Synthetic loan-month panel with a prepayment hazard (~1:67 class imbalance with the defaults)."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_loans):
        cs = int(np.clip(rng.normal(745, 50), 580, 830))
        dti = float(np.clip(rng.normal(34, 9), 5, 60))
        ltv = float(np.clip(rng.normal(76, 15), 20, 97))
        upb0 = float(np.clip(rng.lognormal(12.1, 0.5), 40000, 800000))
        rate0 = float(np.clip(rng.normal(5.2, 1.0), 2.5, 8))
        term = 360
        occ = rng.choice(["P", "S", "I"], p=[0.88, 0.04, 0.08])
        purp = rng.choice(["P", "C", "N"], p=[0.4, 0.2, 0.4])
        chan = rng.choice(["R", "B", "C"], p=[0.5, 0.2, 0.3])
        mrate = float(np.clip(rate0 - rng.normal(0, 0.3), 2.5, 8))
        bal, age = upb0, 0
        while age < term:
            age += 1
            mrate = float(np.clip(mrate + rng.normal(-0.01, 0.12), 2.5, 8))
            hpi, unemp = rng.normal(4, 3), np.clip(rng.normal(6, 2), 3, 12)
            incentive = rate0 - mrate
            z = (-4.9 + 2.0 * incentive + 0.0012 * (cs - 740) - 0.5 * (bal < 80000)
                 + 0.35 * min(age, 36) / 36 - 0.012 * (ltv - 76) - 0.02 * (dti - 34)
                 + (0.25 if purp == "N" else 0) - (0.4 if occ == "I" else 0) + 0.02 * hpi - 0.05 * unemp)
            prepay = rng.rand() < 1 / (1 + np.exp(-z))
            rows.append((i, cs, dti, ltv, ltv + rng.uniform(0, 3), upb0, rate0, age, bal, term - age, rate0,
                         mrate, incentive, hpi, unemp, occ, purp, chan, int(prepay)))
            if prepay:
                break
            r = rate0 / 1200
            pay = upb0 * r / (1 - (1 + r) ** -term)
            bal = max(bal * (1 + r) - pay, 0)
    cols = ["loan_id", "credit_score", "dti", "ltv", "cltv", "orig_upb", "orig_interest_rate", "loan_age",
            "current_upb", "months_to_maturity", "current_interest_rate", "mtg_rate", "rate_incentive",
            "hpi_change", "unemployment", "occupancy_status", "loan_purpose", "channel", "prepaid"]
    return _finalise(pd.DataFrame(rows, columns=cols))


def _finalise(df):
    df = df[["loan_id", "prepaid"] + NUM + CAT].copy()
    for c in NUM:
        df[c] = 0.0 if df[c].isna().all() else df[c].fillna(df[c].median())
    for c in CAT:
        df[c] = df[c].fillna("U").astype(str)
    return df


def to_matrix(df, columns=None):
    X = pd.get_dummies(df[NUM + CAT], columns=CAT, dtype=float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def split_by_loan(df, seed=0, frac=(0.6, 0.2, 0.2)):
    """60/20/20 split grouped by loan so one loan never appears in two splits (no leakage)."""
    rng = np.random.RandomState(seed)
    ids = np.asarray(df.loan_id.unique())
    rng.shuffle(ids)
    a, b = int(frac[0] * len(ids)), int((frac[0] + frac[1]) * len(ids))
    part = {**{i: 0 for i in ids[:a]}, **{i: 1 for i in ids[a:b]}, **{i: 2 for i in ids[b:]}}
    g = df.loan_id.map(part)
    return df[g == 0], df[g == 1], df[g == 2]
