"""Data-pipeline guarantees: Freddie Mac parsing, labels, leakage fix, 30-year filter, loan-grouped split.
Run from the project root:  python -m pytest
"""
import numpy as np
import pandas as pd
import pytest

from src.data import NUM, ORIG_COLS, PERF_COLS, load_freddie, make_synthetic, split_by_loan, to_matrix


def orig_row(loan, term=360, credit=750, rate=6.5, dti=35):
    r = dict.fromkeys(ORIG_COLS, "")
    r.update(credit_score=credit, first_pmt_date=201902, occupancy_status="P", cltv=80, dti=dti, orig_upb=200000,
             ltv=80, orig_interest_rate=rate, channel="R", loan_seq_no=loan, loan_purpose="P", orig_loan_term=term)
    return "|".join(str(r[c]) for c in ORIG_COLS)


def perf_rows(loan, upbs, zb_code=""):
    """One row per month (201901, 201902, ...). If zb_code is set, the last month terminates the loan and,
    as in the real files, reports Current Actual UPB = 0."""
    rows = []
    for m, upb in enumerate(upbs, 1):
        last = m == len(upbs)
        r = dict.fromkeys(PERF_COLS, "")
        r.update(loan_seq_no=loan, period=201900 + m, current_upb=0 if last and zb_code else upb, delinquency=0,
                 loan_age=m, months_to_maturity=360 - m, zero_balance_code=zb_code if last else "",
                 current_interest_rate=6.5)
        rows.append("|".join(str(r[c]) for c in PERF_COLS))
    return rows


@pytest.fixture
def freddie_dir(tmp_path):
    """Pipe-delimited, header-less sample files in Freddie Mac's 32 + 32 column layout."""
    upb = [200000, 199000, 198000, 197000, 196000]
    orig = [orig_row("PREPAID"), orig_row("ACTIVE"), orig_row("SHORTSALE"), orig_row("FIFTEENYR", term=180),
            orig_row("NOSCORE", credit=9999, dti=999)]
    perf = (perf_rows("PREPAID", upb[:4], "01") + perf_rows("ACTIVE", upb) + perf_rows("SHORTSALE", upb[:3], "03")
            + perf_rows("FIFTEENYR", upb[:3], "01") + perf_rows("NOSCORE", upb[:2]))
    (tmp_path / "sample_orig_2019.txt").write_text("\n".join(orig) + "\n")
    (tmp_path / "sample_svcg_2019.txt").write_text("\n".join(perf) + "\n")
    return tmp_path


def test_layout_has_32_columns_per_file():
    assert len(ORIG_COLS) == 32 and len(PERF_COLS) == 32


def test_columns_parse_without_shifting(freddie_dir):
    df = load_freddie(freddie_dir)
    assert (df.orig_interest_rate == 6.5).all()
    assert df.credit_score.between(300, 850).all()
    assert df.loan_age.between(1, 5).all()
    assert set(df.channel) == {"R"}


def test_new_trailing_column_does_not_shift_fields(freddie_dir):
    # Freddie Mac appends fields over time (31 -> 32 origination columns); an unknown extra column must be ignored,
    # not pushed into the index (which would shift every field one place to the right)
    for name in ("sample_orig_2019.txt", "sample_svcg_2019.txt"):
        f = freddie_dir / name
        f.write_text("".join(line + "|NEWFIELD\n" for line in f.read_text().splitlines()))
    with pytest.warns(pd.errors.ParserWarning):
        df = load_freddie(freddie_dir)
    assert (df.orig_interest_rate == 6.5).all() and df.credit_score.between(300, 850).all()
    assert df.prepaid.sum() == 1


def test_only_30_year_loans_are_kept(freddie_dir):
    # zero balance code 01 also means "matured", so shorter loans could be mislabelled as prepayments
    assert set(load_freddie(freddie_dir).loan_id) == {"PREPAID", "ACTIVE", "SHORTSALE", "NOSCORE"}


def test_label_is_one_only_in_the_voluntary_payoff_month(freddie_dir):
    df = load_freddie(freddie_dir)
    pos = df[df.prepaid == 1]
    assert len(pos) == 1
    assert pos.loan_id.iloc[0] == "PREPAID" and pos.loan_age.iloc[0] == 4
    assert df.loc[df.loan_id == "SHORTSALE", "prepaid"].eq(0).all()  # code 03 is not a prepayment


def test_payoff_month_uses_start_of_month_balance(freddie_dir):
    df = load_freddie(freddie_dir).set_index(["loan_id", "loan_age"])
    assert df.loc[("PREPAID", 4), "current_upb"] == 198000  # previous month's balance, not the leaked 0
    assert (df.current_upb > 0).all()
    assert df.loc[("ACTIVE", 1), "current_upb"] == 200000  # first month falls back to the original balance


def test_missing_sentinels_are_imputed(freddie_dir):
    df = load_freddie(freddie_dir)
    assert not df[NUM].isna().any().any()
    nos = df[df.loan_id == "NOSCORE"]
    assert (nos.credit_score == 750).all()  # 9999 = not available -> NaN -> median
    assert (nos.dti == 35).all()  # 999 = not available -> NaN -> median


def test_macro_csv_joins_by_month(freddie_dir, tmp_path):
    macro = tmp_path / "macro.csv"
    pd.DataFrame({"period": range(201901, 201906), "mtg_rate": 4.0, "hpi_change": 5.0,
                  "unemployment": 4.0}).to_csv(macro, index=False)
    df = load_freddie(freddie_dir, macro_csv=macro)
    assert np.allclose(df.rate_incentive, 6.5 - 4.0)


def test_without_macro_rate_incentive_is_constant(freddie_dir):
    df = load_freddie(freddie_dir)
    assert (df.rate_incentive == 0).all() and (df.mtg_rate == 0).all()


def test_split_by_loan_partitions_loans():
    df = make_synthetic(200, seed=1)
    tr, va, te = split_by_loan(df, seed=0)
    a, b, c = set(tr.loan_id), set(va.loan_id), set(te.loan_id)
    assert not (a & b or a & c or b & c)  # no loan appears in two splits
    assert a | b | c == set(df.loan_id)
    assert len(tr) + len(va) + len(te) == len(df)
    assert (len(a), len(b), len(c)) == (120, 40, 40)


def test_split_is_reproducible():
    df = make_synthetic(100, seed=1)
    assert split_by_loan(df, seed=3)[0].loan_id.unique().tolist() == split_by_loan(df, seed=3)[0].loan_id.unique().tolist()


def test_synthetic_loans_end_at_prepayment():
    df = make_synthetic(300, seed=2)
    assert 0.005 < df.prepaid.mean() < 0.05  # heavy class imbalance, as in the real data
    per_loan = df.groupby("loan_id").prepaid
    assert per_loan.sum().max() == 1  # at most one prepayment per loan
    last = df.groupby("loan_id").tail(1)
    assert df[df.prepaid == 1].index.isin(last.index).all()  # and it is the loan's final month


def test_to_matrix_aligns_unseen_categories():
    df = make_synthetic(50, seed=0)
    cols = list(to_matrix(df).columns)
    new = df.head(3).copy()
    new["channel"] = "X"  # category never seen in training
    X = to_matrix(new, cols)
    assert list(X.columns) == cols
    assert (X.filter(like="channel_") == 0).all().all()
