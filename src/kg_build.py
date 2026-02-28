"""
Build the knowledge graph in Neo4j:
- Full question bank (all questions): from ds_test_case_qbank_questions.csv.
- User–question interactions: only from train split (train_data_per_user_80_20).

Run from project root: python src/kg_build.py

Requires Neo4j running. Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE (default: neo4j) if needed.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# In Docker/non-TTY, tqdm may not show updates; mininterval forces periodic flush
def _tqdm(*args, **kwargs):
    kwargs.setdefault("mininterval", 1.0)
    kwargs.setdefault("file", sys.stderr)
    return tqdm(*args, **kwargs)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QBANK_CSV = DATA_DIR / "ds_test_case_qbank_questions.csv"
# Train split only (no test data in KG)
TRAIN_PATHS = [
    DATA_DIR / "train_data_per_user_80_20" / "train_interactions.csv",
    DATA_DIR / "train_data" / "train_data_per_user_80_20.csv",
]


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _train_path() -> Path:
    for p in TRAIN_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"Train data not found. Tried: {TRAIN_PATHS}")


def clear_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def create_constraints_and_indexes(tx):
    tx.run("CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.question_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT difficulty_level IF NOT EXISTS FOR (d:Difficulty) REQUIRE d.level IS UNIQUE")
    tx.run("CREATE CONSTRAINT term_name IF NOT EXISTS FOR (t:Term) REQUIRE t.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
    # ParentTerm for hierarchy (optional)
    tx.run("CREATE CONSTRAINT parent_term_name IF NOT EXISTS FOR (p:ParentTerm) REQUIRE p.name IS UNIQUE")


def load_qbank(path: Path) -> pd.DataFrame:
    """Load full qbank; one row per (question_id, category, term)."""
    df = pd.read_csv(path)
    # Drop rows with missing key fields
    df = df.dropna(subset=["question_id", "category", "term"])
    return df


def load_train_interactions(path: Path) -> pd.DataFrame:
    """Load train split only (user_id, question_id, is_correct, earned_points, total_points, created_at)."""
    df = pd.read_csv(path)
    return df


def build_qbank_graph(driver, qbank: pd.DataFrame, clear_first: bool = True):
    """Build Question, Category, Term, Difficulty, ParentTerm nodes and relationships from full qbank."""
    with driver.session(database=NEO4J_DATABASE) as session:
        if clear_first:
            session.execute_write(clear_graph)
        session.execute_write(create_constraints_and_indexes)

    # Unique entities
    categories = qbank["category"].dropna().unique().tolist()
    terms = qbank["term"].dropna().unique().tolist()
    difficulties = qbank["difficulty"].dropna().unique().tolist()
    parent_terms = qbank["parent_term"].dropna().replace("", pd.NA).dropna().unique().tolist()

    with driver.session(database=NEO4J_DATABASE) as session:
        for c in _tqdm(categories, desc="Categories"):
            session.execute_write(_merge_category, c)
        for d in _tqdm(difficulties, desc="Difficulties"):
            session.execute_write(_merge_difficulty, str(d).strip())
        for t in _tqdm(terms, desc="Terms"):
            session.execute_write(_merge_term, t)
        for p in _tqdm(parent_terms, desc="Parent terms"):
            session.execute_write(_merge_parent_term, str(p).strip())

    # One Question node per question_id (use first row for type, points, difficulty)
    q_meta = qbank.drop_duplicates(subset="question_id", keep="first")
    with driver.session(database=NEO4J_DATABASE) as session:
        for _, row in _tqdm(q_meta.iterrows(), total=len(q_meta), desc="Questions"):
            session.execute_write(
                _merge_question,
                row["question_id"],
                str(row.get("type", "")),
                int(row.get("points", 0)) if pd.notna(row.get("points")) else 0,
                str(row["difficulty"]).strip() if pd.notna(row["difficulty"]) else "Medium",
            )

    # Relationships: each row in qbank gives Question->Category, Question->Term, Term->Category; optionally Term->ParentTerm
    # Batch with UNWIND to avoid 4 round-trips per row (~58k rows = 232k round-trips otherwise).
    BATCH_SIZE = 2000

    def _batch_link_qbank(tx, batch_rows):
        rows = [r for r in batch_rows]
        if not rows:
            return
        qc = [{"qid": r["qid"], "cat": r["cat"]} for r in rows]
        qt = [{"qid": r["qid"], "term": r["term"]} for r in rows]
        tc = [{"term": r["term"], "cat": r["cat"]} for r in rows]
        tp = [{"term": r["term"], "parent": r["parent"]} for r in rows if r.get("parent")]
        tx.run(
            """
            UNWIND $rows AS r
            MATCH (q:Question {question_id: r.qid})
            MATCH (c:Category {name: r.cat})
            MERGE (q)-[:BELONGS_TO_CATEGORY]->(c)
            """,
            rows=qc,
        )
        tx.run(
            """
            UNWIND $rows AS r
            MATCH (q:Question {question_id: r.qid})
            MATCH (t:Term {name: r.term})
            MERGE (q)-[:COVERS_TERM]->(t)
            """,
            rows=qt,
        )
        tx.run(
            """
            UNWIND $rows AS r
            MATCH (t:Term {name: r.term})
            MATCH (c:Category {name: r.cat})
            MERGE (t)-[:PART_OF_CATEGORY]->(c)
            """,
            rows=tc,
        )
        if tp:
            tx.run(
                """
                UNWIND $rows AS r
                MATCH (t:Term {name: r.term})
                MERGE (p:ParentTerm {name: r.parent})
                MERGE (t)-[:PART_OF]->(p)
                """,
                rows=tp,
            )

    with driver.session(database=NEO4J_DATABASE) as session:
        batch = []
        n_rows = len(qbank)
        pbar = _tqdm(total=n_rows, desc="Qbank tags (Category/Term links)", unit="row")
        for _, row in qbank.iterrows():
            batch.append({
                "qid": row["question_id"],
                "cat": str(row["category"]).strip(),
                "term": str(row["term"]).strip(),
                "parent": str(row["parent_term"]).strip() if pd.notna(row["parent_term"]) and str(row["parent_term"]).strip() else None,
            })
            if len(batch) >= BATCH_SIZE:
                session.execute_write(_batch_link_qbank, batch)
                pbar.update(len(batch))
                batch = []
        if batch:
            session.execute_write(_batch_link_qbank, batch)
            pbar.update(len(batch))
        pbar.close()

    return len(q_meta)


def _merge_category(tx, name: str):
    tx.run("MERGE (c:Category {name: $name})", name=name)


def _merge_difficulty(tx, level: str):
    tx.run("MERGE (d:Difficulty {level: $level})", level=level)


def _merge_term(tx, name: str):
    tx.run("MERGE (t:Term {name: $name})", name=name)


def _merge_parent_term(tx, name: str):
    tx.run("MERGE (p:ParentTerm {name: $name})", name=name)


def _merge_question(tx, question_id: str, qtype: str, points: int, difficulty: str):
    tx.run(
        "MERGE (q:Question {question_id: $question_id}) SET q.type = $type, q.points = $points",
        question_id=question_id,
        type=qtype,
        points=points,
    )
    tx.run(
        """
        MATCH (q:Question {question_id: $question_id})
        MATCH (d:Difficulty {level: $level})
        MERGE (q)-[:HAS_DIFFICULTY]->(d)
        """,
        question_id=question_id,
        level=difficulty,
    )


def _link_question_category(tx, question_id: str, category: str):
    tx.run(
        """
        MATCH (q:Question {question_id: $question_id})
        MATCH (c:Category {name: $category})
        MERGE (q)-[:BELONGS_TO_CATEGORY]->(c)
        """,
        question_id=question_id,
        category=category,
    )


def _link_question_term(tx, question_id: str, term: str):
    tx.run(
        """
        MATCH (q:Question {question_id: $question_id})
        MATCH (t:Term {name: $term})
        MERGE (q)-[:COVERS_TERM]->(t)
        """,
        question_id=question_id,
        term=term,
    )


def _link_question_difficulty(tx, question_id: str, level: str):
    tx.run(
        """
        MATCH (q:Question {question_id: $question_id})
        MATCH (d:Difficulty {level: $level})
        MERGE (q)-[:HAS_DIFFICULTY]->(d)
        """,
        question_id=question_id,
        level=level,
    )


def _link_term_category(tx, term: str, category: str):
    tx.run(
        """
        MATCH (t:Term {name: $term})
        MATCH (c:Category {name: $category})
        MERGE (t)-[:PART_OF_CATEGORY]->(c)
        """,
        term=term,
        category=category,
    )


def _link_term_parent(tx, term: str, parent_name: str):
    tx.run(
        """
        MATCH (t:Term {name: $term})
        MERGE (p:ParentTerm {name: $parent_name})
        MERGE (t)-[:PART_OF]->(p)
        """,
        term=term,
        parent_name=parent_name,
    )


def build_user_train_edges(driver, train_df: pd.DataFrame):
    """Create User nodes and (User)-[:ANSWERED]->(Question) from train data only."""
    with driver.session(database=NEO4J_DATABASE) as session:
        for _, row in _tqdm(train_df.iterrows(), total=len(train_df), desc="Train (User-ANSWERED->Question)"):
            uid = row["user_id"]
            qid = row["question_id"]
            is_correct = int(row["is_correct"]) if "is_correct" in row else (1 if row.get("earned_points", 0) == row.get("total_points", 1) else 0)
            earned = int(row.get("earned_points", 0))
            total = int(row.get("total_points", 1))
            if total == 0:
                total = 1
            session.execute_write(
                _merge_user_answered,
                uid,
                qid,
                is_correct,
                earned,
                total,
            )
    return len(train_df)


def _merge_user_answered(tx, user_id: str, question_id: str, is_correct: int, earned_points: int, total_points: int):
    tx.run("MERGE (u:User {user_id: $user_id})", user_id=user_id)
    tx.run(
        """
        MATCH (u:User {user_id: $user_id})
        MATCH (q:Question {question_id: $question_id})
        MERGE (u)-[r:ANSWERED]->(q)
        SET r.is_correct = $is_correct, r.earned_points = $earned_points, r.total_points = $total_points
        """,
        user_id=user_id,
        question_id=question_id,
        is_correct=is_correct,
        earned_points=earned_points,
        total_points=total_points,
    )


def main():
    if not QBANK_CSV.exists():
        raise FileNotFoundError(f"Qbank not found: {QBANK_CSV}")

    train_path = _train_path()
    driver = get_driver()
    try:
        driver.verify_connectivity()
    except Exception as e:
        print("Cannot connect to Neo4j. Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD if needed.")
        raise

    qbank = load_qbank(QBANK_CSV)
    n_questions = build_qbank_graph(driver, qbank, clear_first=True)
    print(f"KG: loaded full qbank -> {n_questions} questions.")

    train_df = load_train_interactions(train_path)
    n_edges = build_user_train_edges(driver, train_df)
    print(f"KG: loaded train interactions only -> {n_edges} (User)-[:ANSWERED]->(Question) edges.")
    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
