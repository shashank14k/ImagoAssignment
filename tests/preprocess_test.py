import pytest
import pandas as pd
import numpy as np
from src.preprocess.preprocess import DataScaler, get_numerical_cols, Preprocessor


def test_get_numerical_cols():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["x", "y", "z"]})
    result = get_numerical_cols(df, ignore_cols=["C"])
    assert "C" not in result.columns
    assert "A" in result.columns and "B" in result.columns


@pytest.fixture
def sample_csv(tmp_path):
    file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, np.nan, 4.0],
        "feature2": [10.0, 20.0, 30.0, 40.0],
        "target": [0, 1, 0, 1],
    })
    df.to_csv(file, index=False)
    return str(file)


def test_preprocessor_init(sample_csv):
    preprocessor = Preprocessor(sample_csv, target_col="target")
    assert isinstance(preprocessor.df, pd.DataFrame)
    assert "target" in preprocessor.df.columns


def test_fill_null(sample_csv):
    preprocessor = Preprocessor(sample_csv, target_col="target")
    preprocessor.fill_null(method="mean")
    assert not preprocessor.df["feature1"].isna().any()


def test_treat_outliers(sample_csv):
    preprocessor = Preprocessor(sample_csv, target_col="target")

    # Compute original IQR bounds before treatment
    col = "feature1"
    q1 = preprocessor.df[col].quantile(0.2)
    q3 = preprocessor.df[col].quantile(0.8)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Treat outliers
    preprocessor.treat_outliers()

    # Check that no values are outside the IQR bounds
    assert not (preprocessor.df[col] < lower_bound).any()
    assert not (preprocessor.df[col] > upper_bound).any()


def test_get_feature_list_and_scalers(sample_csv):
    preprocessor = Preprocessor(sample_csv, target_col="target")
    top_features, scalers = preprocessor.get_feature_list_and_scalers()
    assert isinstance(top_features, list)
    assert isinstance(scalers, DataScaler)
