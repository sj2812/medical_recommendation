from collections import defaultdict

import numpy as np
import torch


def _term_mastery_from_history(hist_qids, hist_correct, question_to_terms, alpha=1.0, beta=1.0):
    """Compute per-term mastery from (qid, correct) history. mastery = (correct + alpha) / (attempts + alpha + beta)."""
    attempts = defaultdict(float)
    corrects = defaultdict(float)
    for qid, c in zip(hist_qids, hist_correct):
        for t in question_to_terms.get(str(qid), set()):
            attempts[t] += 1.0
            corrects[t] += float(c)
    mastery = {t: (corrects[t] + alpha) / (attempts[t] + alpha + beta) for t in attempts}
    return mastery, attempts


def _weak_terms(mastery, top_n=30):
    """Return set of term keys with lowest mastery (weakest terms)."""
    if not mastery:
        return set()
    sorted_terms = sorted(mastery.items(), key=lambda x: x[1])[:top_n]
    return set(t for t, _ in sorted_terms)


def _pad_state_from_idx_history(model, device, hist_q_idx, hist_correct, hist_dt, max_history):
    """Pad history to max_history, run model.forward_user_state, return (1, H) tensor on device or None if empty."""
    if not hist_q_idx or not hist_correct or not hist_dt:
        return None
    L = min(len(hist_q_idx), max_history)
    start = max_history - L
    hq = np.zeros(max_history, dtype=np.int64)
    hc = np.zeros(max_history, dtype=np.float32)
    hdt = np.zeros(max_history, dtype=np.float32)
    hq[start:] = np.array(hist_q_idx[-L:], dtype=np.int64)
    hc[start:] = np.array(hist_correct[-L:], dtype=np.float32)
    hdt[start:] = np.array(hist_dt[-L:], dtype=np.float32)
    hq_t = torch.from_numpy(hq).unsqueeze(0).to(device)
    hc_t = torch.from_numpy(hc).unsqueeze(0).to(device)
    hdt_t = torch.from_numpy(hdt).unsqueeze(0).to(device)
    with torch.no_grad():
        state = model.forward_user_state(hq_t, hc_t, hdt_t)
    return state


def _score_candidates(model, device, user_state, cand, batch_size=1024):
    """Score all candidates; cand is np (N,). Returns np (N,) scores."""
    cand = np.asarray(cand, dtype=np.int64)
    scores_list = []
    for i in range(0, len(cand), batch_size):
        batch = cand[i : i + batch_size]
        q_t = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            s = model.score_pair(user_state, q_t.unsqueeze(0))
        scores_list.append(s.squeeze(0).cpu().numpy())
    return np.concatenate(scores_list)


def _predict_p_correct(model, device, user_state, top_idx):
    """Return p(correct) for each question index in top_idx; (K,) numpy or None if no correct_head."""
    if getattr(model, "correct_head", None) is None:
        return None
    top_idx = np.asarray(top_idx, dtype=np.int64)
    q_t = torch.from_numpy(top_idx).to(device)
    with torch.no_grad():
        u = user_state.expand(q_t.shape[0], -1)
        q_emb = model.q_embed(q_t)
        logits = model.correct_head(torch.cat([u, q_emb], dim=-1)).squeeze(-1)
        p = torch.sigmoid(logits).cpu().numpy()
    return p


def recommend_for_user(
    model,
    device,
    hist_q_idx,
    hist_correct,
    hist_dt,
    qid2idx,
    idx2qid,
    question_to_terms,
    valid_q_indices,
    max_history=50,
    top_k=10,
    weak_top_n=15,
    score_batch_size=1024,
):
    """
    Run inference for one user: encode history, score candidates, return top-K recommendations.
    hist_* are lists (from the user's train history). valid_q_indices is the candidate set (e.g. train question indices).
    Returns dict with top_k_qids, top_k_scores, top_k_p_correct, weak_terms_sample, history_len.
    """
    model.eval()
    valid_q_indices = np.asarray(valid_q_indices, dtype=np.int64)
    user_state = _pad_state_from_idx_history(model, device, hist_q_idx, hist_correct, hist_dt, max_history)
    if user_state is None:
        return None
    scores = _score_candidates(model, device, user_state, valid_q_indices, batch_size=score_batch_size)
    order = np.argsort(-scores)
    top_indices = valid_q_indices[order[:top_k]]
    top_scores = scores[order[:top_k]]
    p_correct = _predict_p_correct(model, device, user_state, top_indices)
    hist_qids = [idx2qid.get(int(i), "") for i in hist_q_idx if int(i) in idx2qid]
    mastery, _ = _term_mastery_from_history(hist_qids, hist_correct, question_to_terms)
    weak = _weak_terms(mastery, top_n=weak_top_n)
    weak_sample = list(weak)[:10]
    return {
        "history_len": len(hist_q_idx),
        "weak_terms_sample": weak_sample,
        "top_k_indices": top_indices.tolist(),
        "top_k_qids": [idx2qid.get(int(i), "") for i in top_indices],
        "top_k_scores": top_scores.tolist(),
        "top_k_p_correct": p_correct.tolist() if p_correct is not None else None,
    }


def eval_learning_metrics_rolling(
    model,
    device,
    train_df,
    test_df,
    qid2idx,
    question_to_terms,
    valid_q_indices,               
    max_history=50,
    top_k=10,
    weak_top_n=30,
    p_band=(0.6, 0.8),
    score_batch_size=1024,
    evaluate_steps_per_user=None,  
    alpha=1.0,
    beta=1.0,
):
    """
    Rolling per-test-step evaluation of learning-aligned metrics.

    Metrics per step:
      - TermHit@K: top-K includes at least one question covering current weak terms
      - ChallengeInBand@K: fraction of top-K with p(correct) in [lo,hi] (if correct_head exists)
      - RepeatFocus@K: among top-K questions that are repeats, fraction that focus on weak terms

    Also reports:
      - RepeatRate in test (how often the actual target is already seen)
      - WeakTermCoverageCount: how many unique weak terms were touched by recommended questions (proxy coverage)
    """
    model.eval()

    train_sorted = train_df.sort_values(["user_id", "created_at"])
    test_sorted  = test_df.sort_values(["user_id", "created_at"])
    valid_q_indices = np.asarray(valid_q_indices, dtype=np.int64)

    per_user = {}

    for user_id in test_sorted["user_id"].unique():
        train_u = train_sorted[train_sorted["user_id"] == user_id]
        test_u  = test_sorted[test_sorted["user_id"] == user_id]
        if len(train_u) < 2 or len(test_u) < 1:
            continue

        # history init from train
        hist_qids = train_u["question_id"].astype(str).tolist()
        hist_q_idx = [qid2idx.get(q, 0) for q in hist_qids]
        hist_correct = train_u["is_correct"].astype(np.float32).values.tolist()
        t0 = train_u["created_at"].iloc[0]
        hist_dt = ((train_u["created_at"] - t0).dt.total_seconds().values.astype(np.float32)).tolist()

        seen_idx = set([qi for qi in hist_q_idx if qi != 0])
        seen_qids = set(hist_qids)

        # accumulators
        term_hit = []
        challenge_in_band = []
        repeat_focus = []  # only defined if there are repeats in topK
        repeat_target_flags = []  # whether the actual next question in test was a repeat
        weak_terms_covered = set()

        steps_done = 0

        for _, row in test_u.iterrows():
            if evaluate_steps_per_user is not None and steps_done >= evaluate_steps_per_user:
                break

            target_qid = str(row["question_id"])
            target_idx = qid2idx.get(target_qid, 0)
            if target_idx == 0:
                continue

            # current mastery/weak terms from history (qid strings)
            mastery, _ = _term_mastery_from_history(hist_qids, hist_correct, question_to_terms, alpha=alpha, beta=beta)
            weak = _weak_terms(mastery, top_n=weak_top_n)

            # user state
            user_state = _pad_state_from_idx_history(model, device, hist_q_idx, hist_correct, hist_dt, max_history)
            if user_state is None:
                break

            # candidate set for recommendation: train-known indices
            cand = valid_q_indices

            # score + rank
            scores = _score_candidates(model, device, user_state, cand, batch_size=score_batch_size)
            order = np.argsort(-scores)
            top_idx = cand[order[:top_k]]  # q indices
            # map to qids for term lookup
            # (idx2qid not passed; reconstruct inverse map on the fly using qid2idx is expensive)
            # Instead, we store qid strings in history and for term hit we check question_to_terms by qid.
            # We'll build a local idx->qid once per function call:
            steps_done += 1

            # actual target repeat?
            repeat_target_flags.append(1.0 if target_idx in seen_idx else 0.0)

            # TermHit@K: does any recommended question touch weak terms?
            hit = 0
            # Need idx->qid mapping: build once globally outside loop for speed
            # We'll lazily create it on first use and cache.
            if "_idx2qid_cache" not in eval_learning_metrics_rolling.__dict__:
                inv = {v: k for k, v in qid2idx.items()}
                eval_learning_metrics_rolling.__dict__["_idx2qid_cache"] = inv
            idx2qid_cache = eval_learning_metrics_rolling.__dict__["_idx2qid_cache"]

            top_qids = [idx2qid_cache.get(int(qi), None) for qi in top_idx]
            top_qids = [q for q in top_qids if q is not None]

            for q in top_qids:
                if weak and (question_to_terms.get(q, set()) & weak):
                    hit = 1
                    break
            term_hit.append(float(hit))

            # record weak terms covered by recommendations (coverage proxy)
            for q in top_qids:
                weak_terms_covered |= (question_to_terms.get(q, set()) & weak)

            # ChallengeInBand@K: fraction of top-K with p(correct) in band
            p = _predict_p_correct(model, device, user_state, top_idx)
            if p is not None:
                lo, hi = p_band
                challenge_in_band.append(float(np.mean((p >= lo) & (p <= hi))))

            # RepeatFocus@K: among recommended repeats, what fraction focuses on weak terms?
            rep_mask = [1 if qid2idx.get(q, 0) in seen_idx else 0 for q in top_qids]
            rep_qids = [q for q, m in zip(top_qids, rep_mask) if m == 1]
            if len(rep_qids) > 0:
                good = 0
                for q in rep_qids:
                    if weak and (question_to_terms.get(q, set()) & weak):
                        good += 1
                repeat_focus.append(float(good / len(rep_qids)))

            # consume this test event into history
            dt_val = float((row["created_at"] - t0).total_seconds())
            hist_qids.append(target_qid)
            hist_q_idx.append(target_idx)
            hist_correct.append(float(row["is_correct"]))
            hist_dt.append(dt_val)
            seen_idx.add(target_idx)
            seen_qids.add(target_qid)

        if len(term_hit) == 0:
            continue

        per_user[user_id] = {
            "steps": len(term_hit),
            "TermHit@K": float(np.mean(term_hit)),
            "ChallengeInBand@K": float(np.mean(challenge_in_band)) if len(challenge_in_band) else None,
            "RepeatFocus@K": float(np.mean(repeat_focus)) if len(repeat_focus) else None,
            "RepeatRate_target": float(np.mean(repeat_target_flags)) if len(repeat_target_flags) else None,
            "UniqueWeakTermsCovered": int(len(weak_terms_covered)),
        }

    # ---- build macro and return ----
    if not per_user:
        print("No evaluable users for learning metrics.")
        return per_user, {}

    print(f"Learning-metrics rolling evaluation (K={top_k}, weak_top_n={weak_top_n}, p_band={p_band}):")
    for u, m in per_user.items():
        _cb, _rf, _rr = m.get("ChallengeInBand@K"), m.get("RepeatFocus@K"), m.get("RepeatRate_target")
        print(
            f"  user={u} | steps={m['steps']}"
            f" | TermHit@{top_k}={m['TermHit@K']:.4f}"
            f" | ChallengeInBand@{top_k}={'NA' if _cb is None else f'{_cb:.4f}'}"
            f" | RepeatFocus@{top_k}={'NA' if _rf is None else f'{_rf:.4f}'}"
            f" | RepeatRate(target)={'NA' if _rr is None else f'{_rr:.4f}'}"
            f" | WeakTermsCovered={m['UniqueWeakTermsCovered']}"
        )

    vals = list(per_user.values())
    macro = {
        "TermHit@K": float(np.mean([v["TermHit@K"] for v in vals])),
        "ChallengeInBand@K": None,
        "RepeatFocus@K": None,
        "RepeatRate_target": None,
        "UniqueWeakTermsCovered": float(np.mean([v["UniqueWeakTermsCovered"] for v in vals])),
        "avg_steps": float(np.mean([v["steps"] for v in vals])),
    }
    if any(v["ChallengeInBand@K"] is not None for v in vals):
        macro["ChallengeInBand@K"] = float(np.mean([v["ChallengeInBand@K"] for v in vals if v["ChallengeInBand@K"] is not None]))
    if any(v["RepeatFocus@K"] is not None for v in vals):
        macro["RepeatFocus@K"] = float(np.mean([v["RepeatFocus@K"] for v in vals if v["RepeatFocus@K"] is not None]))
    if any(v["RepeatRate_target"] is not None for v in vals):
        macro["RepeatRate_target"] = float(np.mean([v["RepeatRate_target"] for v in vals if v["RepeatRate_target"] is not None]))

    print("\nMacro (across users):")
    print(f"  avg_steps={macro['avg_steps']:.1f}")
    print(f"  TermHit@{top_k}={macro['TermHit@K']:.4f}")
    print(f"  ChallengeInBand@{top_k}={'NA' if (_mcb := macro.get('ChallengeInBand@K')) is None else f'{_mcb:.4f}'}")
    print(f"  RepeatFocus@{top_k}={'NA' if (_mrf := macro.get('RepeatFocus@K')) is None else f'{_mrf:.4f}'}")
    print(f"  RepeatRate(target)={'NA' if (_mrr := macro.get('RepeatRate_target')) is None else f'{_mrr:.4f}'}")
    print(f"  UniqueWeakTermsCovered(avg)={macro['UniqueWeakTermsCovered']:.1f}")

    return per_user, macro
