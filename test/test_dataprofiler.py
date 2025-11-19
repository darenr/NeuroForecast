import pytest
import pandas as pd
import numpy as np
import sys
import os
from rich import print

# Add parent directory to path to import neuro_forecast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neuro_forecast import DataProfiler


@pytest.fixture
def sample_df():
    # Create a sample dataframe
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    # Create a clear positive trend, starting from 1 to avoid initial zero
    values = np.linspace(1, 10, 100)
    return pd.DataFrame({"date": dates, "value": values})


def test_analyze_structure(sample_df):
    result = DataProfiler.analyze(sample_df, "date", "value")

    print("\n[bold blue]Structure Test Profile:[/bold blue]")
    print(result)

    expected_keys = [
        "n_samples",
        "start_date",
        "end_date",
        "missing_values",
        "zero_values",
        "trend",
        "seasonality",
        "volatility",
        "regressors_available",
        "data_sample_head",
        "data_sample_tail",
    ]

    for key in expected_keys:
        assert key in result


def test_analyze_values(sample_df):
    result = DataProfiler.analyze(sample_df, "date", "value")

    assert result["n_samples"] == 100
    assert result["missing_values"] == 0
    # Trend should be positive given linspace(0, 10, 100)
    assert result["trend"] == "positive"


def test_missing_values(sample_df):
    df_missing = sample_df.copy()
    df_missing.loc[0, "value"] = np.nan
    result = DataProfiler.analyze(df_missing, "date", "value")
    assert result["missing_values"] == 1


def test_zero_values(sample_df):
    df_zeros = sample_df.copy()
    df_zeros.loc[1, "value"] = 0
    result = DataProfiler.analyze(df_zeros, "date", "value")
    assert result["zero_values"] == 1


def test_analyze_retail_sales():
    # Path to the retail sales file (which is actually a CSV despite .py extension)
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "retail_sales_multivariate.py")
    )

    # Read the CSV data
    df = pd.read_csv(file_path)

    # Analyze
    result = DataProfiler.analyze(
        df, "date", "revenue", regressors=["marketing_spend", "local_event"]
    )

    print("\n[bold green]Retail Sales Profile:[/bold green]")
    print(result)

    # Assertions
    assert result["n_samples"] > 0
    assert result["missing_values"] == 0
    assert "marketing_spend" in result["regressors_available"]
    assert "local_event" in result["regressors_available"]
