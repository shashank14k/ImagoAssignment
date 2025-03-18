import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def split_df(input_file: str, train_output: str, test_output: str, test_size: float = 0.2):
    df = pd.read_csv(input_file)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"Train set saved to {train_output}")
    print(f"Test set saved to {test_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a CSV file into train and test sets.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("train_output", type=str, help="Path to save the training set.")
    parser.add_argument("test_output", type=str, help="Path to save the test set.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test set (default: 0.2).")

    args = parser.parse_args()

    split_df(args.input_file, args.train_output, args.test_output, args.test_size)