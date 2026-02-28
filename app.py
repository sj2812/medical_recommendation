"""
Medical Students Learning Assistant – Streamlit app.
Run from project root: streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Medical Students Learning Assistant", layout="wide")
st.title("Medical Students Learning Assistant")

# Session state
if "mode" not in st.session_state:
    st.session_state["mode"] = None
if "show_eda" not in st.session_state:
    st.session_state["show_eda"] = False
if "app_data" not in st.session_state:
    st.session_state["app_data"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "device" not in st.session_state:
    st.session_state["device"] = None

# Mode selection
col1, col2 = st.columns(2)
with col1:
    developer = st.button("Developer", use_container_width=True)
with col2:
    user_btn = st.button("User", use_container_width=True)

if developer:
    st.session_state["mode"] = "developer"
    st.session_state["show_eda"] = False
if user_btn:
    st.session_state["mode"] = "user"
    st.session_state["show_eda"] = False

# Developer mode: dropdown (EDA) and optional Train / Evaluate
if st.session_state["mode"] == "developer":
    st.subheader("Developer")
    dev_option = st.selectbox(
        "Choose an option",
        options=["", "EDA", "Train model", "Evaluate"],
        format_func=lambda x: "Select..." if x == "" else x,
        key="dev_select",
    )
    st.session_state["show_eda"] = dev_option == "EDA"

    if st.session_state["show_eda"]:
        notebook_path = Path(__file__).resolve().parent / "notebooks" / "01_eda_data.ipynb"
        if not notebook_path.exists():
            st.error(f"Notebook not found: {notebook_path}")
        else:
            with open(notebook_path, encoding="utf-8") as f:
                nb = json.load(f)
            for cell in nb.get("cells", []):
                src = "".join(cell.get("source", []))
                if not src.strip():
                    continue
                if cell.get("cell_type") == "markdown":
                    st.markdown(src)
                else:
                    st.code(src, language="python")
                    for out in cell.get("outputs", []):
                        if out.get("output_type") == "stream" and "text" in out:
                            text = "".join(out["text"])
                            if text.strip():
                                st.text(text)
                        elif out.get("output_type") == "execute_result" and "data" in out:
                            data = out["data"]
                            if "text/plain" in data:
                                st.text("".join(data["text/plain"]))
                            if "text/html" in data:
                                st.markdown("".join(data["text/html"]), unsafe_allow_html=True)

    if dev_option == "Train model":
        if st.button("Run training", key="run_train"):
            with st.spinner("Loading data..."):
                from src.app_data import load_app_data
                data = load_app_data()
                st.session_state["app_data"] = data
            with st.spinner("Training (calibrate + KG-policy)..."):
                from src.train_kg_policy import run_kg_policy_training
                try:
                    model, device = run_kg_policy_training(
                        data,
                        calibrate_epochs=3,
                        policy_epochs=5,
                        batch_size=32,
                    )
                    st.session_state["model"] = model
                    st.session_state["device"] = device
                    st.success("Training completed.")
                except Exception as e:
                    st.error(str(e))
        if st.session_state.get("model") is not None:
            st.info("Model is loaded. Use Evaluate to run learning metrics.")

    if dev_option == "Evaluate":
        if st.session_state.get("model") is None or st.session_state.get("app_data") is None:
            st.warning("Load data and train a model first (choose Train model and run).")
        else:
            if st.button("Run evaluation", key="run_eval"):
                with st.spinner("Evaluating..."):
                    try:
                        from src.evaluate import eval_learning_metrics_rolling
                        import pandas as pd
                        data = st.session_state["app_data"]
                        model = st.session_state["model"]
                        device = st.session_state["device"]
                        per_user, macro = eval_learning_metrics_rolling(
                            model=model,
                            device=device,
                            train_df=data["train_df"],
                            test_df=data["test_df"],
                            qid2idx=data["qid2idx"],
                            question_to_terms=data["question_to_terms"],
                            valid_q_indices=data["valid_q_indices"],
                            max_history=data["max_history"],
                            top_k=10,
                            weak_top_n=30,
                            p_band=(0.6, 0.8),
                            score_batch_size=1024,
                        )
                        st.success("Evaluation completed.")
                        if not macro:
                            st.info("No evaluable users for learning metrics.")
                        else:
                            st.subheader("Macro metrics (across users)")
                            top_k = 10
                            c1, c2, c3, c4, c5 = st.columns(5)
                            with c1:
                                st.metric("Avg steps", f"{macro['avg_steps']:.1f}")
                            with c2:
                                st.metric(f"TermHit@{top_k}", f"{macro['TermHit@K']:.4f}")
                            with c3:
                                cb = macro.get("ChallengeInBand@K")
                                st.metric(f"ChallengeInBand@{top_k}", f"{cb:.4f}" if cb is not None else "NA")
                            with c4:
                                rf = macro.get("RepeatFocus@K")
                                st.metric(f"RepeatFocus@{top_k}", f"{rf:.4f}" if rf is not None else "NA")
                            with c5:
                                rr = macro.get("RepeatRate_target")
                                st.metric("RepeatRate(target)", f"{rr:.4f}" if rr is not None else "NA")
                            st.metric("UniqueWeakTermsCovered (avg)", f"{macro['UniqueWeakTermsCovered']:.1f}")
                            with st.expander("Per-user metrics"):
                                rows = []
                                for uid, m in per_user.items():
                                    def _fmt(v):
                                        return f"{v:.4f}" if v is not None and isinstance(v, (int, float)) else "NA"
                                    rows.append({
                                        "user_id": uid[:12] + "…" if len(uid) > 12 else uid,
                                        "steps": m["steps"],
                                        f"TermHit@{top_k}": f"{m['TermHit@K']:.4f}",
                                        f"ChallengeInBand@{top_k}": _fmt(m.get("ChallengeInBand@K")),
                                        f"RepeatFocus@{top_k}": _fmt(m.get("RepeatFocus@K")),
                                        "RepeatRate(target)": _fmt(m.get("RepeatRate_target")),
                                        "WeakTermsCovered": m["UniqueWeakTermsCovered"],
                                    })
                                if rows:
                                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(str(e))

# User mode: example inference simulation
if st.session_state["mode"] == "user":
    st.subheader("User")
    st.caption("Simulate a recommendation for an example student using their past activity.")
    if st.button("Run example inference", key="user_inference"):
        if st.session_state.get("model") is None or st.session_state.get("app_data") is None:
            st.warning("No trained model or data in session. Go to **Developer** → **Train model** first, then come back here.")
        else:
            with st.spinner("Loading data and running inference..."):
                try:
                    import numpy as np
                    from src.app_data import load_app_data
                    from src.evaluate import recommend_for_user
                    data = st.session_state.get("app_data") or load_app_data()
                    if st.session_state.get("app_data") is None:
                        st.session_state["app_data"] = data
                    model = st.session_state["model"]
                    device = st.session_state["device"]
                    train_df = data["train_df"].sort_values(["user_id", "created_at"])
                    test_df = data["test_df"]
                    qid2idx, idx2qid = data["qid2idx"], data["idx2qid"]
                    max_history = data["max_history"]
                    valid_q = data["valid_q_indices"]
                    question_to_terms = data["question_to_terms"]
                    # Pick an example user: one who has both train and test interactions
                    train_users = set(train_df["user_id"].unique())
                    test_users = set(test_df["user_id"].unique())
                    example_users = list(train_users & test_users)
                    if not example_users:
                        st.error("No user has both train and test data. Cannot run example.")
                    else:
                        example_user = example_users[0]
                        train_u = train_df[train_df["user_id"] == example_user]
                        if len(train_u) < 2:
                            st.error("Example user has too few train interactions.")
                        else:
                            hist_qids = train_u["question_id"].astype(str).tolist()
                            hist_q_idx = [qid2idx.get(q, 0) for q in hist_qids]
                            hist_correct = train_u["is_correct"].astype(np.float32).values.tolist()
                            t0 = train_u["created_at"].iloc[0]
                            hist_dt = ((train_u["created_at"] - t0).dt.total_seconds().values.astype(np.float32)).tolist()
                            out = recommend_for_user(
                                model, device,
                                hist_q_idx, hist_correct, hist_dt,
                                qid2idx, idx2qid, question_to_terms,
                                valid_q, max_history=max_history, top_k=10, weak_top_n=15,
                            )
                            if out is None:
                                st.error("Could not compute user state (empty or invalid history).")
                            else:
                                st.success("Recommendation ready.")
                                st.markdown("#### Example student")
                                st.write(f"**User ID:** `{example_user[:20]}...` (from train/test set)")
                                st.write(f"**History length:** {out['history_len']} attempted questions")
                                st.markdown("#### Weak terms (concepts to practice)")
                                weak = out["weak_terms_sample"]
                                if weak:
                                    for t in weak[:5]:
                                        cat, term = str(t[0])[:40], str(t[1])[:50]
                                        st.write(f"• **{cat}** — {term}")
                                else:
                                    st.write("_None inferred from history._")
                                st.markdown("#### Top 10 recommended questions")
                                import pandas as pd
                                rows = []
                                for r, (qid, sc) in enumerate(zip(out["top_k_qids"], out["top_k_scores"]), 1):
                                    row = {"Rank": r, "Question ID": qid[:16] + "…" if len(qid) > 16 else qid, "Score": f"{sc:.4f}"}
                                    if out.get("top_k_p_correct") is not None:
                                        row["p(correct)"] = f"{out['top_k_p_correct'][r-1]:.3f}"
                                    rows.append(row)
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                                with st.expander("Raw recommendation payload"):
                                    st.json({k: v for k, v in out.items() if k != "top_k_scores" or (isinstance(v, list) and len(v) <= 15)})
                except Exception as e:
                    st.error(str(e))
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("Click **Run example inference** to get top-10 recommended questions for an example student (uses the trained model and their history).")
