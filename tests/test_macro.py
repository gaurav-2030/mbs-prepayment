"""Macro-feature builder, tested offline from cached FRED-style CSVs (no network needed)."""
import numpy as np
import pandas as pd
import pytest

from src.macro import SERIES, build


@pytest.fixture
def fred_cache(tmp_path):
    months = pd.date_range("2019-01-01", "2020-03-01", freq="MS")
    weeks = pd.date_range("2019-01-04", "2020-03-27", freq="W-FRI")
    pd.DataFrame({"observation_date": weeks, "MORTGAGE30US": np.linspace(4.5, 3.5, len(weeks))}).to_csv(
        tmp_path / "MORTGAGE30US.csv", index=False)
    pd.DataFrame({"observation_date": months, "HPIPONM226S": 100 + np.arange(len(months))}).to_csv(
        tmp_path / "HPIPONM226S.csv", index=False)
    unrate = ["3.5"] * (len(months) - 1) + ["."]  # FRED marks missing values with "."
    pd.DataFrame({"observation_date": months, "UNRATE": unrate}).to_csv(tmp_path / "UNRATE.csv", index=False)
    return tmp_path


def test_series_ids():
    assert SERIES == {"mtg_rate": "MORTGAGE30US", "hpi": "HPIPONM226S", "unemployment": "UNRATE"}


def test_output_columns_and_period_format(fred_cache):
    m = build(start=201901, cache_dir=fred_cache)
    assert list(m.columns) == ["period", "mtg_rate", "hpi_change", "unemployment"]
    assert m.period.tolist() == [*range(201901, 201913), 202001, 202002, 202003]


def test_weekly_rate_is_averaged_per_month(fred_cache):
    weekly = pd.read_csv(fred_cache / "MORTGAGE30US.csv", parse_dates=[0])
    jan = weekly[weekly.observation_date.dt.strftime("%Y%m") == "201901"].MORTGAGE30US.mean()
    m = build(start=201901, cache_dir=fred_cache).set_index("period")
    assert m.loc[201901, "mtg_rate"] == pytest.approx(jan, abs=1e-3)


def test_hpi_change_is_twelve_month_percent_change(fred_cache):
    m = build(start=201901, cache_dir=fred_cache).set_index("period")
    assert np.isnan(m.loc[201912, "hpi_change"])  # needs a year of history
    assert m.loc[202001, "hpi_change"] == pytest.approx((112 / 100 - 1) * 100, abs=1e-3)


def test_missing_values_stay_missing(fred_cache):
    m = build(start=201901, cache_dir=fred_cache).set_index("period")
    assert np.isnan(m.loc[202003, "unemployment"]) and m.loc[202002, "unemployment"] == 3.5


def test_start_filter(fred_cache):
    assert build(start=202001, cache_dir=fred_cache).period.min() == 202001
