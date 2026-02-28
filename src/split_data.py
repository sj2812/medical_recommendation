"""
Split interactions 80-20 by time, per user: each user's first 80% of interactions (by time)
go to train, last 20% to test. Saves under data/train_data_per_user_80_20/ and
data/test_data_per_user_80_20/.

Usage (from project root):
  python src/split_data.py [path_to_interactions_csv]

Defaults to data/ds_test_case_user_answers.csv (expects user_id, question_id, created_at).
For synthetic data (student_id, timestamp), use: python src/split_data.py data/student_interactions.csv
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INTERACTIONS = DATA_DIR / "ds_test_case_user_answers.csv"
TRAIN_DIR = DATA_DIR / "train_data"
TEST_DIR = DATA_DIR / "test_data"
TRAIN_FRAC = 0.8


def split_per_user_by_time(
    interactions_df: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "created_at",
    train_frac: float = TRAIN_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user, sort by time and assign first train_frac to train, rest to test.
    Returns (train_df, test_df).
    """
    # Normalize time column for sorting
    if time_col not in interactions_df.columns and "timestamp" in interactions_df.columns:
        time_col = "timestamp"
    interactions_df = interactions_df.sort_values([user_col, time_col]).reset_index(drop=True)

    train_parts = []
    test_parts = []
    for _, group in interactions_df.groupby(user_col, sort=False):
        n = len(group)
        n_train = max(1, int(n * train_frac))
        n_test = n - n_train
        if n_test < 1:
            continue
        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def main():
    interactions_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INTERACTIONS
    if not interactions_path.is_absolute():
        interactions_path = PROJECT_ROOT / interactions_path
    if not interactions_path.exists():
        raise FileNotFoundError(f"Interactions file not found: {interactions_path}")

    df = pd.read_csv(interactions_path)
    # Detect schema
    if "user_id" in df.columns and "created_at" in df.columns:
        user_col, time_col = "user_id", "created_at"
    elif "student_id" in df.columns and "timestamp" in df.columns:
        user_col, time_col = "student_id", "timestamp"
    else:
        raise ValueError("Expected columns user_id + created_at or student_id + timestamp")

    df[time_col] = pd.to_datetime(df[time_col])
    train_df, test_df = split_per_user_by_time(df, user_col=user_col, time_col=time_col, train_frac=TRAIN_FRAC)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    train_path = TRAIN_DIR / "train_data_per_user_80_20.csv"
    test_path = TEST_DIR / "test_data_per_user_80_20.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_df)} rows -> {train_path}")
    print(f"Test:  {len(test_df)} rows -> {test_path}")
    print(f"Users in train: {train_df[user_col].nunique()}, in test: {test_df[user_col].nunique()}")


if __name__ == "__main__":
    main()
