"""
Data loading and vocab/term lookups for the Streamlit app and KG-policy training.
Run from project root so paths resolve correctly.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QBANK_PATH = DATA_DIR / "ds_test_case_qbank_questions.csv"
TRAIN_PATH = DATA_DIR / "train_data" / "train_data_per_user_80_20.csv"
TEST_PATH = DATA_DIR / "test_data" / "test_data_per_user_80_20.csv"
EMB_PATH = DATA_DIR / "kg_question_embeddings.npy"

MAX_HISTORY = 50


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train_df, test_df, qbank with normalized types."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    qbank = pd.read_csv(QBANK_PATH)
    for df, col in [(train_df, "user_id"), (test_df, "user_id"), (train_df, "question_id"), (test_df, "question_id"), (qbank, "question_id")]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    train_df["created_at"] = pd.to_datetime(train_df["created_at"], utc=True)
    test_df["created_at"] = pd.to_datetime(test_df["created_at"], utc=True)
    return train_df, test_df, qbank


def build_vocabs_and_terms(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    qbank: pd.DataFrame,
) -> dict[str, Any]:
    """Build qid2idx, uid2idx, question_to_terms, term_to_questions_train, valid_q_indices, etc."""
    all_qbank_qids = sorted(qbank["question_id"].dropna().unique().tolist())
    qid2idx = {q: i + 1 for i, q in enumerate(all_qbank_qids)}
    idx2qid = {i: q for q, i in qid2idx.items()}
    num_questions = len(all_qbank_qids) + 1

    all_user_ids = sorted(set(train_df["user_id"]) | set(test_df["user_id"]))
    uid2idx = {u: i for i, u in enumerate(all_user_ids)}
    num_users = len(uid2idx)

    question_to_terms: dict[str, set[tuple[str, str]]] = defaultdict(set)
    term_to_questions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for _, row in qbank.dropna(subset=["question_id", "category", "term"]).iterrows():
        q = str(row["question_id"])
        key = (str(row["category"]).strip(), str(row["term"]).strip())
        question_to_terms[q].add(key)
        term_to_questions[key].add(q)

    Q_TRAIN = set(train_df["question_id"].unique())
    term_to_questions_train = {k: (v & Q_TRAIN) for k, v in term_to_questions.items() if len(v & Q_TRAIN) > 0}
    valid_q_indices = np.array([qid2idx[q] for q in sorted(Q_TRAIN)], dtype=np.int64)

    return {
        "qid2idx": qid2idx,
        "idx2qid": idx2qid,
        "num_questions": num_questions,
        "num_users": num_users,
        "uid2idx": uid2idx,
        "question_to_terms": dict(question_to_terms),
        "term_to_questions_train": term_to_questions_train,
        "valid_q_indices": valid_q_indices,
        "Q_TRAIN": Q_TRAIN,
    }


def build_temporal_samples(
    train_df: pd.DataFrame,
    qid2idx: dict[str, int],
    uid2idx: dict[str, int],
    max_history: int = MAX_HISTORY,
) -> list[dict]:
    """Build list of training samples (history -> next question, correctness)."""
    train_df = train_df.sort_values(["user_id", "created_at"]).reset_index(drop=True)
    samples = []
    for user_id, group in train_df.groupby("user_id"):
        user_idx = uid2idx.get(user_id)
        if user_idx is None:
            continue
        rows = group.reset_index(drop=True)
        for i in range(len(rows) - 1):
            past = rows.iloc[: i + 1]
            if len(past) > max_history:
                past = past.iloc[-max_history:]
            t0 = past["created_at"].iloc[0]
            history_q = np.array([qid2idx.get(q, 0) for q in past["question_id"]], dtype=np.int64)
            history_correct = past["is_correct"].astype(np.float32).values
            history_dt = (past["created_at"] - t0).dt.total_seconds().values.astype(np.float32)
            next_row = rows.iloc[i + 1]
            target_q = qid2idx.get(next_row["question_id"], 0)
            if target_q == 0:
                continue
            target_correct = float(next_row["is_correct"])
            samples.append({
                "user_idx": user_idx,
                "history_q": history_q,
                "history_correct": history_correct,
                "history_dt": history_dt,
                "target_q": target_q,
                "target_correct": target_correct,
            })
    return samples


if TORCH_AVAILABLE:

    class TemporalRecDataset(Dataset):
        def __init__(self, samples: list[dict], num_questions: int, max_history: int):
            self.samples = samples
            self.num_questions = num_questions
            self.max_history = max_history

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict:
            s = self.samples[idx]
            L = len(s["history_q"])
            start = self.max_history - L
            hq = np.zeros(self.max_history, dtype=np.int64)
            hc = np.zeros(self.max_history, dtype=np.float32)
            hdt = np.zeros(self.max_history, dtype=np.float32)
            hq[start:] = s["history_q"]
            hc[start:] = s["history_correct"]
            hdt[start:] = s["history_dt"]
            return {
                "user_idx": s["user_idx"],
                "history_q": torch.from_numpy(hq),
                "history_correct": torch.from_numpy(hc),
                "history_dt": torch.from_numpy(hdt),
                "target_q": s["target_q"],
                "target_correct": s["target_correct"],
            }


def load_app_data() -> dict[str, Any]:
    """Load all data and build vocabs, term lookups, and training dataset. Cache result in caller."""
    train_df, test_df, qbank = load_dataframes()
    vocabs = build_vocabs_and_terms(train_df, test_df, qbank)
    samples = build_temporal_samples(
        train_df,
        vocabs["qid2idx"],
        vocabs["uid2idx"],
        MAX_HISTORY,
    )
    out = {
        "train_df": train_df,
        "test_df": test_df,
        "qbank": qbank,
        "emb_path": EMB_PATH,
        "max_history": MAX_HISTORY,
        **vocabs,
    }
    if TORCH_AVAILABLE and samples:
        out["train_dataset"] = TemporalRecDataset(
            samples,
            vocabs["num_questions"],
            MAX_HISTORY,
        )
    else:
        out["train_dataset"] = None
    return out
