# src/kg_export_embeddings.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "qbankuser")

WRITE_PROP = os.environ.get("GDS_WRITE_PROP", "embedding")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_NPY = DATA_DIR / "kg_question_embeddings.npy"
OUT_CSV = DATA_DIR / "kg_qid_order.csv"

# you can build qid2idx from qbank question_ids (recommended)
QBANK_CSV = DATA_DIR / "ds_test_case_qbank_questions.csv"


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def main():
    qbank = pd.read_csv(QBANK_CSV)
    all_qids = sorted(set(qbank["question_id"].dropna().astype(str).tolist()))

    # reserve 0 for padding
    qid2idx = {q: i + 1 for i, q in enumerate(all_qids)}
    num_questions = len(all_qids) + 1

    driver = get_driver()
    driver.verify_connectivity()

    # Fetch embeddings from Neo4j
    rows = []
    with driver.session(database=NEO4J_DATABASE) as session:
        # stream results
        res = session.run(
            f"""
            MATCH (q:Question)
            WHERE q.{WRITE_PROP} IS NOT NULL
            RETURN q.question_id AS question_id, q.{WRITE_PROP} AS emb
            """
        )
        for r in res:
            rows.append((str(r["question_id"]), r["emb"]))

    driver.close()

    if not rows:
        raise RuntimeError(f"No embeddings found on Question nodes. Did you run kg_embed_gds.py? writeProperty={WRITE_PROP}")

    # infer dim
    dim = len(rows[0][1])
    mat = np.zeros((num_questions, dim), dtype=np.float32)  # 0 is padding

    missing = 0
    for qid, emb in tqdm(rows, desc="Filling matrix"):
        idx = qid2idx.get(qid)
        if idx is None:
            continue
        mat[idx] = np.array(emb, dtype=np.float32)

    # check if any qids missing embeddings
    have = set(q for q, _ in rows)
    missing_qids = [q for q in all_qids if q not in have]
    missing = len(missing_qids)
    print(f"Total qids: {len(all_qids)}, embeddings present: {len(have)}, missing embeddings: {missing}")

    np.save(OUT_NPY, mat)
    pd.DataFrame({"question_id": all_qids, "idx": [qid2idx[q] for q in all_qids]}).to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_NPY)
    print("Saved:", OUT_CSV)
    print("Embedding matrix shape:", mat.shape)


if __name__ == "__main__":
    main()
