# Medical Students Learning Assistant

A recommendation engine for a medical education platform that suggests the **next best question(s)** for a student based on their learning history and learning objectives (e.g. passing a board exam). The system combines a **knowledge graph** (Neo4j) with a **temporal neural model** and is exposed via a Streamlit app.

---

## Features

- **Streamlit app**: Developer mode (EDA notebook viewer, train model, evaluate) and User mode.
- **Knowledge graph**: Neo4j graph of questions, terms, categories, and user–question interactions (train split only).
- **Hybrid recommender**: Temporal LSTM over interaction history + KG-derived question embeddings (e.g. FastRP); training optimises a learning-oriented reward (weak-term coverage, challenge band, spacing) with optional click imitation.
- **Evaluation**: Rolling per-step metrics (TermHit@K, ChallengeInBand@K, RepeatFocus@K, coverage, repeat rate) with macro and per-user reporting in the UI.
- **Docker**: Optional `docker-compose` setup with Neo4j + Streamlit app and data volume.

---

## Data

- **Question metadata** (`data/ds_test_case_qbank_questions.csv`): `question_id`, `type`, `points`, `difficulty`, `category`, `term`, `parent_term`.
- **User answers** (`data/ds_test_case_user_answers.csv`): `user_id`, `question_id`, `is_correct`, `earned_points`, `total_points`, `created_at`.
- **Splits**: 80/20 per user by time → `data/train_data/train_data_per_user_80_20.csv` and `data/test_data/test_data_per_user_80_20.csv` (see `src/split_data.py`).

See `data/README.text` for column details.

---

## Recommender system logic

The system recommends the next question by (1) representing the student’s **state** from their history, (2) using the **knowledge graph** to define mastery, weak terms, and candidate questions, and (3) **scoring** candidates with a neural model and (at training time) a **reward** that favours learning.

### 1. Knowledge graph (Neo4j)

- **Nodes**: `Question`, `Category`, `Term`, `Difficulty`, `ParentTerm`, `User`.
- **Edges**:  
  - `Question` → `Category` / `Term` / `Difficulty`; `Term` → `Category` / `ParentTerm`.  
  - `User` → `Question` via `ANSWERED` with `is_correct`, `earned_points`, `total_points` (from **train** data only).
- **Purpose**: Connects questions to competencies (terms). Used to compute **term-level mastery**, **weak terms**, and **candidate sets** (questions that cover weak terms). Optional: GDS FastRP for question embeddings written back to Neo4j and exported to `data/kg_question_embeddings.npy`.

### 2. Student state and “weak terms”

- **History**: For each user, interactions are ordered by `created_at` and truncated to a fixed window (e.g. last 50).
- **Term mastery**: For each term, mastery = (correct + α) / (attempts + α + β). Low mastery ⇒ **weak term**.
- **Weak terms**: The top‑N terms with lowest mastery form the set of concepts the student should practice. The recommender prioritises questions that cover these terms.

### 3. Temporal scoring model (`HybridTemporalRecommender`)

- **Inputs**: Sequence of (question index, correctness, time delta) over the history window.
- **Question embedding**: From KG (e.g. FastRP) or learned; one vector per question, padding index 0.
- **User state**: Each step is (question_embed, is_correct, normalised time_delta) → linear → LSTM; final hidden state = **user state** at that time.
- **Scoring**: For a candidate set of question indices, `score(user_state, candidate_embeddings)` = MLP(concat(user_state, question_embed)) → one score per candidate. Optional **correctness head**: same concat → probability of answering the next question correctly (used for “challenge in band” and reward).
- So: **state** = f(history); **score(user, question)** = g(state, question_embed).

### 4. KG‑policy training (`train_kg_policy.py`)

Training has two phases.

**Phase 1 – Calibrate correctness head**  
- Train only the correctness head (rest frozen) to predict `is_correct` for the next question from (user_state, next_question_embed).  
- Ensures p(correct) is meaningful for reward and evaluation; then this head is usually **frozen** in phase 2.

**Phase 2 – Listwise policy with learning reward**  
- For each batch of (history, next_question):
  - Encode history → user_state.
  - **Candidate set**: From **weak terms** (low mastery), sample questions that cover those terms (plus random exploration). Candidates are in “index” space (train question set).
  - Score all candidates with the model; get p(correct) for candidates from the (frozen) correctness head.
  - **Reward** per candidate (conceptually “how good is this question for learning?”):
    - **Need**: prefer low mastery on the question’s terms (1 − mastery).
    - **Objective**: IDF‑weighted need (emphasise rarer/important terms).
    - **Spacing**: bonus for repeating weak terms after a gap (Gaussian bump in time since last seen).
    - **Challenge**: prefer p(correct) near a target (e.g. 0.7) so items are “in the learning zone”.
    - **Repeat penalty**: discourage repeats unless they target weak terms with spacing.
  - **Loss**: Listwise cross‑entropy: turn rewards into a target distribution (softmax over candidates), and minimise CE between that target and the model’s score distribution. Optionally add a **click loss** (rank the actual next question above negatives) to keep global ranking sensible.

So the **logic** is: **recommend questions that (a) cover the student’s weak terms, (b) are appropriately challenging, and (c) benefit from spacing**, by training the scorer to align with this reward.

### 5. Evaluation (`evaluate.py`)

- **Rolling**: For each test user, initialise history from train; for each test step, compute current weak terms and user state, score all (or a fixed) candidate set, take top‑K.
- **Metrics**:
  - **TermHit@K**: At least one of the top‑K recommendations covers a current weak term.
  - **ChallengeInBand@K**: Fraction of top‑K with p(correct) in a band (e.g. [0.6, 0.8]).
  - **RepeatFocus@K**: Among recommended questions that are repeats, fraction that focus on weak terms.
  - **RepeatRate (target)**: How often the actual next question was already seen.
  - **UniqueWeakTermsCovered**: Count of weak terms touched by the top‑K.
- Results are aggregated **per user** and **macro** (averaged across users) and shown in the Streamlit UI.

---

## Project structure

```
├── app.py                 # Streamlit app (Developer: EDA, Train, Evaluate; User)
├── src/
│   ├── app_data.py        # Load data, build vocabs, term lookups, temporal dataset
│   ├── model.py           # HybridTemporalRecommender (LSTM + score pair + correct head)
│   ├── train_kg_policy.py # KG-policy training (calibrate + listwise reward)
│   ├── evaluate.py        # Rolling evaluation and learning metrics
│   ├── kg_build.py        # Build Neo4j graph (qbank + train interactions)
│   ├── kg_embed_gds.py    # Neo4j GDS FastRP embeddings (requires GDS plugin)
│   ├── kg_export_embeddings.py  # Export question embeddings to .npy + qid order CSV
│   └── split_data.py      # 80/20 per-user-by-time split
├── data/                  # CSVs (qbank, answers, train/test splits, kg_question_embeddings.npy)
├── notebooks/             # EDA, hybrid recommender, recommenders with KG
├── Dockerfile
├── docker-compose.yml     # Neo4j + Streamlit app
└── docker-entrypoint.sh  # Wait for Neo4j, optional KG build, then CMD (Streamlit)
```

---

## Setup

**Local**

```bash
cd medical_recommendation
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Neo4j (optional)**  
Run Neo4j (e.g. Docker or Neo4j Desktop). Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` if needed. For question embeddings, install the GDS plugin and run:

```bash
python -m src.kg_embed_gds
python -m src.kg_export_embeddings
```

This produces `data/kg_question_embeddings.npy` used by the temporal model.

---

## Running

**Streamlit app (from project root)**

```bash
streamlit run app.py
```

Then open http://localhost:8501. Use **Developer** → **EDA** to view the EDA notebook; **Train model** to run KG-policy training (loads data and trains); **Evaluate** to run learning metrics and see them in the UI.

**Docker**

```bash
docker compose up --build
```

- Streamlit: http://localhost:8501  
- Neo4j Browser: http://localhost:17474 (Bolt: localhost:17687)  
Data is mounted from `./data`; the entrypoint waits for Neo4j and optionally runs `kg_build` if train/qbank data exist.

---

## Scripts (reference)

| Script | Purpose |
|--------|--------|
| `python -m src.split_data` | Build 80/20 per-user train/test splits . |
| `python -m src.kg_build` | Build Neo4j graph from qbank + train interactions. |
| `python -m src.kg_embed_gds` | Run GDS FastRP on the graph, write embeddings to Question nodes (requires GDS). |
| `python -m src.kg_export_embeddings` | Export Question embeddings to `data/kg_question_embeddings.npy` and qid order CSV. |

Training and evaluation are triggered from the Streamlit app (Developer → Train model / Evaluate) and use `src/app_data`, `src/train_kg_policy`, and `src/evaluate`.


## Time log:

EDA -> 60 min
Problem understanding and approach definition -> 60 min
Implementation with iteration -> 240 min
Streamlit + dockerization -> 60 min
Testing -> 60+ min

