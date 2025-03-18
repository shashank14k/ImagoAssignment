import os
import pandas as pd
import numpy as np
import argparse
from dataclasses import dataclass
from typing import Literal, Tuple, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class DataScaler:
    feature_scaler: StandardScaler
    target_scaler: StandardScaler

    def to_dict(self):
        return {
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
        }


def get_numerical_cols(df: pd.DataFrame, ignore_cols: List) -> pd.DataFrame:
    return df.drop(columns=ignore_cols)


class Preprocessor:
    def __init__(self, csv_file: str, target_col: str, ignore_columns: List[str] = None):
        assert os.path.exists(csv_file), f"File {csv_file} does not exist."
        self.df = pd.read_csv(csv_file)
        if ignore_columns is None:
            ignore_columns = []
        ignore_columns += [col for col in self.df.columns if self.df[col].dtype == "object"]
        self.feature_cols = list(set(self.df.columns) - set(ignore_columns) - {target_col})
        self.ignore_cols = ignore_columns
        self.target_col = target_col

    def fill_null(self, method: Literal["median", "mean", None] = None):
        if method == "median":
            self.df[self.feature_cols] = self.df[self.feature_cols].fillna(self.df[self.feature_cols].median())
            self.df[self.target_col] = self.df[self.target_col].fillna(self.df[self.target_col].median())
        elif method == "mean":
            self.df[self.feature_cols] = self.df[self.feature_cols].fillna(self.df[self.feature_cols].mean())
            self.df[self.target_col] = self.df[self.target_col].fillna(self.df[self.target_col].mean())

    def treat_outliers(self, iqr_factor: float = 1.5, low_q: float = 0.2, high_q: float = 0.8,
                       method: Literal["clip", "mean"] = "clip", include_target: bool = True):
        assert 0 < low_q < high_q < 1, "Invalid range for low and high quantiles."
        cols = self.feature_cols.copy()
        if include_target:
            cols.append(self.target_col)
        for col in cols:
            if self.df[col].dtype == "object":
                continue
            q1, q3 = self.df[col].quantile([low_q, high_q])
            iqr = q3 - q1
            lower_bound = q1 - iqr_factor * iqr
            upper_bound = q3 + iqr_factor * iqr
            if method == "clip":
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
            elif method == "mean":
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                self.df.loc[outlier_mask, col] = self.df[col].mean()

    def get_feature_list_and_scalers(self, n_components: float = 0.8) -> Tuple[List[str], DataScaler]:
        if self.df.isna().sum().sum() > 0:
            self.fill_null("median")
        ignore_cols = list(set(self.ignore_cols) & set(self.df.columns))
        numerical_data = get_numerical_cols(self.df, ignore_cols)
        feature_scaler = StandardScaler()
        scaled_data = feature_scaler.fit_transform(numerical_data)
        target_data = self.df[self.target_col].values.reshape(-1, 1)
        target_scaler = StandardScaler()
        target_scaler.fit_transform(target_data)
        n_components = int(n_components * len(self.feature_cols))
        pca = PCA(n_components=n_components)
        pca.fit(scaled_data)
        feature_importance = np.abs(pca.components_).sum(axis=0)
        ranked_features = np.argsort(feature_importance)[::-1]
        top_features = [self.feature_cols[i] for i in ranked_features[:n_components]]
        feature_scaler.fit(self.df[top_features])
        scalers = DataScaler(feature_scaler, target_scaler)
        return top_features, scalers



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--target_col", type=str, required=True, help="Name of the target column.")
    parser.add_argument("--ignore_columns", nargs="*", default=[], help="List of columns to ignore.")
    parser.add_argument("--null_method", type=str, choices=["median", "mean", "None"], default="None",
                        help="Method for handling missing values.")
    parser.add_argument("--iqr_factor", type=float, default=1.5, help="IQR multiplier for outlier detection.")
    parser.add_argument("--low_q", type=float, default=0.2, help="Lower quantile for outlier detection.")
    parser.add_argument("--high_q", type=float, default=0.8, help="Upper quantile for outlier detection.")
    parser.add_argument("--outlier_method", type=str, choices=["clip", "mean"], default="clip",
                        help="Method for handling outliers.")
    parser.add_argument("--include_target", action="store_true", help="Include target column in outlier treatment.")
    parser.add_argument("--n_components", type=float, default=0.2, help="Fraction of PCA components to retain.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save transformed CSV file.")
    args = parser.parse_args()

    preprocessor = Preprocessor(args.csv_file, args.target_col, args.ignore_columns)
    preprocessor.fill_null(args.null_method)
    preprocessor.treat_outliers(args.iqr_factor, args.low_q, args.high_q, args.outlier_method, args.include_target)
    feature_list, _ = preprocessor.get_feature_list_and_scalers(args.n_components)
    feature_list += [args.target_col]
    transformed_df = preprocessor.df[feature_list]
    transformed_df.to_csv(args.output_file, index=False)
