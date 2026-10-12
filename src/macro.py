"""Build the monthly macro-economic features used by the Freddie Mac loader (--macro), from public FRED series:
  MORTGAGE30US   Freddie Mac PMMS 30-year fixed mortgage rate (weekly -> monthly mean)  -> mtg_rate
  HPIPONM226S    FHFA purchase-only house price index (monthly, SA) -> 12-month % change -> hpi_change
  UNRATE         BLS civilian unemployment rate (monthly)                                -> unemployment
Usage:
  python -m src.macro                          # writes data/macro.csv (periods from 1999-01)
  python -m src.macro --out data/macro.csv --start 200501 --cache data/fred
"""
import argparse
import io
import os
import ssl
import urllib.request
import certifi
import pandas as pd

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
SERIES = {"mtg_rate": "MORTGAGE30US", "hpi": "HPIPONM226S", "unemployment": "UNRATE"}
# certifi's CA bundle: python.org builds on macOS ship without root certificates, so HTTPS fails otherwise
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def fetch(series_id, cache_dir=None):
    """One FRED series as a monthly mean, indexed by period (YYYYMM int, the format of Freddie Mac's reporting period).
    With cache_dir, a saved <series_id>.csv is reused (offline runs) and fresh downloads are saved there."""
    path = os.path.join(cache_dir, f"{series_id}.csv") if cache_dir else None
    if path and os.path.exists(path):
        with open(path) as f:
            text = f.read()
    else:
        with urllib.request.urlopen(FRED.format(series_id), timeout=60, context=SSL_CONTEXT) as r:
            text = r.read().decode()
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
    s = pd.read_csv(io.StringIO(text), parse_dates=[0], index_col=0, na_values=".").iloc[:, 0]
    return s.groupby(s.index.year * 100 + s.index.month).mean()


def build(start=199901, cache_dir=None):
    """DataFrame with columns period, mtg_rate, hpi_change (% vs 12 months earlier), unemployment."""
    m = pd.DataFrame({k: fetch(v, cache_dir) for k, v in SERIES.items()}).sort_index()
    m["hpi_change"] = m.hpi.pct_change(12, fill_method=None) * 100
    m = m.loc[m.index >= start, ["mtg_rate", "hpi_change", "unemployment"]].dropna(how="all")
    m.index.name = "period"
    return m.round(3).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/macro.csv")
    ap.add_argument("--start", type=int, default=199901, help="first period, YYYYMM")
    ap.add_argument("--cache", default=None, help="folder to save/reuse the raw FRED CSVs")
    a = ap.parse_args()
    m = build(a.start, a.cache)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    m.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(m)} months {m.period.iloc[0]}-{m.period.iloc[-1]}, "
          f"missing values: {m.isna().sum().to_dict()}")


if __name__ == "__main__":
    main()
