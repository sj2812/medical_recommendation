"""
KG-policy training: calibrate correctness head, then listwise reward + optional click loss.
Uses data and vocabs from app_data.load_app_data().
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import HybridTemporalRecommender


def run_kg_policy_training(
    data: dict[str, Any],
    *,
    calibrate_epochs: int = 10,
    policy_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weak_top_n: int = 100,
    max_candidates: int = 512,
    per_term_cap: int = 20,
    explore_random: int = 256,
    use_click_loss: bool = True,
    click_weight: float = 0.2,
    num_neg_click: int = 64,
    freeze_correct_head: bool = True,
    freeze_q_embed: bool = False,
    grad_clip: float = 1.0,
    p_target: float = 0.7,
    temperature: float = 1.0,
) -> tuple[HybridTemporalRecommender, torch.device]:
    """
    Load KG embeddings, build model, calibrate correctness head, then run KG-policy training.
    Returns (model, device) for use in evaluation or saving.
    """
    train_df = data["train_df"]
    qid2idx = data["qid2idx"]
    num_questions = data["num_questions"]
    num_users = data["num_users"]
    question_to_terms = data["question_to_terms"]
    term_to_questions_train = data["term_to_questions_train"]
    valid_q_indices = data["valid_q_indices"]
    train_dataset = data["train_dataset"]
    emb_path = data["emb_path"]
    max_history = data["max_history"]

    if train_dataset is None:
        raise ValueError("train_dataset is None; ensure app_data built it (PyTorch + samples).")

    emb = np.load(emb_path)
    EMBED_DIM = emb.shape[1]
    pretrained = torch.tensor(emb, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridTemporalRecommender(
        num_questions=num_questions,
        num_users=num_users,
        embed_dim=EMBED_DIM,
        hidden_dim=128,
        predict_correctness=True,
        pretrained_q_embeddings=pretrained,
        freeze_q_embeddings=False,
    ).to(device)

    # KG lookups in index space
    qidx_to_terms = {}
    for qid, terms in question_to_terms.items():
        qi = qid2idx.get(str(qid), 0)
        if qi != 0 and terms:
            qidx_to_terms[qi] = list(terms)

    term_to_qidx_pool = {}
    for term_key, qids in term_to_questions_train.items():
        qidxs = [qid2idx.get(str(q), 0) for q in qids]
        qidxs = [x for x in qidxs if x != 0]
        if qidxs:
            term_to_qidx_pool[term_key] = np.array(sorted(set(qidxs)), dtype=np.int64)

    valid_q_indices_np = np.asarray(valid_q_indices, dtype=np.int64)
    N_trainQ = len(valid_q_indices_np)
    term_counts = {t: len(pool) for t, pool in term_to_qidx_pool.items()}
    term_idf = {t: float(np.log1p(N_trainQ / (1.0 + c))) for t, c in term_counts.items()}

    def compute_term_mastery(q_hist_idx, corr_hist, alpha=1.0, beta=1.0):
        attempts = defaultdict(float)
        corrects = defaultdict(float)
        for qi, c in zip(q_hist_idx, corr_hist):
            for t in qidx_to_terms.get(int(qi), []):
                attempts[t] += 1.0
                corrects[t] += float(c)
        mastery = {t: (corrects[t] + alpha) / (a + alpha + beta) for t, a in attempts.items()}
        return mastery, attempts

    def weak_terms_from_mastery(mastery, top_n=100):
        if not mastery:
            return []
        return [t for t, _ in sorted(mastery.items(), key=lambda x: x[1])[:top_n]]

    def build_last_seen_term_time(q_hist_idx, dt_hist):
        last_seen = {}
        for qi, dt in zip(q_hist_idx, dt_hist):
            for t in qidx_to_terms.get(int(qi), []):
                last_seen[t] = float(dt)
        return last_seen

    def spacing_bump(dt_since, peak=24 * 3600, width=24 * 3600):
        x = (dt_since - peak) / (width + 1e-6)
        return math.exp(-0.5 * x * x)

    def sample_candidates_kg(
        weak_terms,
        seen_set,
        max_candidates=512,
        per_term_cap=20,
        explore_random=256,
        allow_repeats=True,
    ):
        cand = set()
        for t in weak_terms:
            pool = term_to_qidx_pool.get(t)
            if pool is None or len(pool) == 0:
                continue
            take = min(per_term_cap, len(pool))
            chosen = np.random.choice(pool, size=take, replace=False)
            for qi in chosen:
                if allow_repeats or (qi not in seen_set):
                    cand.add(int(qi))
            if len(cand) >= (max_candidates - explore_random):
                break
        if explore_random > 0:
            base = valid_q_indices_np if allow_repeats else valid_q_indices_np[~np.isin(valid_q_indices_np, np.fromiter(seen_set, dtype=np.int64))]
            if len(base) == 0:
                base = valid_q_indices_np
            take = min(explore_random, len(base))
            for qi in np.random.choice(base, size=take, replace=False):
                cand.add(int(qi))
        if len(cand) == 0:
            take = min(max_candidates, len(valid_q_indices_np))
            cand = set(np.random.choice(valid_q_indices_np, size=take, replace=False).tolist())
        cand = np.array(list(cand), dtype=np.int64)
        if len(cand) > max_candidates:
            cand = np.random.choice(cand, size=max_candidates, replace=False)
        return cand

    def compute_rewards(
        candidates,
        mastery,
        weak_set,
        last_seen_term_time,
        now_t,
        seen_set,
        p_correct=None,
        p_target=0.7,
        mastery_default=0.5,
        mastery_repeat_threshold=0.6,
        w_need=1.0,
        w_obj=0.6,
        w_spacing=0.5,
        w_challenge=2.0,
        repeat_penalty=0.25,
    ):
        N = len(candidates)
        r = np.zeros(N, dtype=np.float32)
        for i, qi in enumerate(candidates):
            terms = qidx_to_terms.get(int(qi), [])
            if not terms:
                continue
            need_vals = []
            obj_num, obj_den = 0.0, 0.0
            for t in terms:
                m = mastery.get(t, mastery_default)
                need_vals.append(1.0 - m)
                w = term_idf.get(t, 1.0)
                obj_num += w * (1.0 - m)
                obj_den += w
            need = float(np.mean(need_vals)) if need_vals else 0.0
            obj = float(obj_num / (obj_den + 1e-6))
            spacing = 0.0
            cnt = 0
            for t in terms:
                m = mastery.get(t, mastery_default)
                if t in last_seen_term_time and m < mastery_repeat_threshold:
                    dt_since = max(0.0, now_t - last_seen_term_time[t])
                    spacing += spacing_bump(dt_since)
                    cnt += 1
            if cnt > 0:
                spacing /= cnt
            is_repeat = (int(qi) in seen_set)
            rep_pen = repeat_penalty if is_repeat else 0.0
            if is_repeat and (set(terms) & weak_set) and spacing > 0.2:
                rep_pen *= 0.2
            challenge = 0.0
            if p_correct is not None:
                challenge = -abs(float(p_correct[i]) - p_target)
            r[i] = (w_need * need + w_obj * obj + w_spacing * spacing + w_challenge * challenge - rep_pen)
        return r

    def listwise_reward_ce_stable(scores, rewards, temperature=1.0, reward_temp=1.0, reward_clip=2.0):
        r = rewards.clamp(min=-reward_clip, max=reward_clip)
        target = torch.softmax((r / reward_temp).float(), dim=1)
        logp = torch.log_softmax((scores / temperature).float(), dim=1)
        return -(target * logp).sum(dim=1).mean()

    def sampled_softmax_click_loss(model, user_state, target_q, neg_q):
        q_stack = torch.cat([target_q.unsqueeze(1), neg_q], dim=1)
        logits = model.score_pair(user_state, q_stack)
        return nn.CrossEntropyLoss()(logits, torch.zeros(target_q.size(0), dtype=torch.long, device=target_q.device))

    def finetune_correct_head(model, device, train_dataset, epochs=5, lr=1e-3):
        for p in model.parameters():
            p.requires_grad = False
        for p in model.correct_head.parameters():
            p.requires_grad = True
        opt = torch.optim.Adam(model.correct_head.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()
        loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        model.train()
        for ep in range(1, epochs + 1):
            tot, n = 0.0, 0
            for batch in loader:
                hq = batch["history_q"].to(device)
                hc = batch["history_correct"].to(device)
                hdt = batch["history_dt"].to(device)
                tq = batch["target_q"].to(device)
                y = batch["target_correct"].to(device).float()
                with torch.no_grad():
                    ustate = model.forward_user_state(hq, hc, hdt)
                q_emb = model.q_embed(tq)
                logits = model.correct_head(torch.cat([ustate, q_emb], dim=-1)).squeeze(-1)
                loss = bce(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * y.size(0)
                n += y.size(0)
            print(f"[CorrectHead] epoch {ep}/{epochs} loss={tot/n:.4f}")
        for p in model.parameters():
            p.requires_grad = True

    # Calibrate correctness head
    finetune_correct_head(model, device, train_dataset, epochs=calibrate_epochs, lr=lr)

    if freeze_correct_head and getattr(model, "correct_head", None) is not None:
        for p in model.correct_head.parameters():
            p.requires_grad = False
    if freeze_q_embed:
        for p in model.q_embed.parameters():
            p.requires_grad = False

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    valid_q_t = torch.from_numpy(valid_q_indices_np.astype(np.int64)).to(device)

    def sample_click_negs(target_q):
        B = target_q.size(0)
        N = valid_q_t.numel()
        idx = torch.randint(0, N, (B, num_neg_click), device=device)
        neg = valid_q_t[idx]
        mask = neg.eq(target_q.unsqueeze(1))
        if mask.any():
            repl = valid_q_t[torch.randint(0, N, (mask.sum().item(),), device=device)]
            neg[mask] = repl
        return neg

    for ep in range(1, policy_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in tqdm(loader, desc=f"KG-policy {ep}/{policy_epochs}"):
            history_q = batch["history_q"].to(device)
            history_correct = batch["history_correct"].to(device)
            history_dt = batch["history_dt"].to(device)
            target_q = batch["target_q"].to(device)
            B, T = history_q.shape
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                user_state = model.forward_user_state(history_q, history_correct, history_dt)
            hq_cpu = history_q.detach().cpu().numpy()
            hc_cpu = history_correct.detach().cpu().numpy()
            hdt_cpu = history_dt.detach().cpu().numpy()
            cand_list = []
            meta_list = []
            for b in range(B):
                mask = (hq_cpu[b] != 0)
                q_hist_idx = hq_cpu[b][mask].astype(np.int64).tolist()
                corr_hist = hc_cpu[b][mask].astype(np.float32).tolist()
                dt_hist = hdt_cpu[b][mask].astype(np.float32).tolist()
                recent_window = corr_hist[-50:] if len(corr_hist) >= 50 else corr_hist
                recent_acc = float(np.mean(recent_window)) if corr_hist else 0.5
                p_target_dynamic = float(np.clip(recent_acc + 0.05, 0.55, 0.75))
                seen_set = set(q_hist_idx)
                mastery, _ = compute_term_mastery(q_hist_idx, corr_hist)
                weak = weak_terms_from_mastery(mastery, top_n=weak_top_n)
                weak_set = set(weak)
                last_seen = build_last_seen_term_time(q_hist_idx, dt_hist)
                now_t = float(dt_hist[-1]) if dt_hist else 0.0
                cand = sample_candidates_kg(
                    weak_terms=weak,
                    seen_set=seen_set,
                    max_candidates=max_candidates,
                    per_term_cap=per_term_cap,
                    explore_random=explore_random,
                    allow_repeats=True,
                )
                cand_list.append(cand)
                meta_list.append((mastery, weak_set, last_seen, now_t, seen_set))
            maxN = max(len(c) for c in cand_list)
            cand_mat = np.zeros((B, maxN), dtype=np.int64)
            mask_mat = np.zeros((B, maxN), dtype=np.float32)
            for b in range(B):
                c = cand_list[b]
                cand_mat[b, : len(c)] = c
                mask_mat[b, : len(c)] = 1.0
            cand_t = torch.from_numpy(cand_mat).to(device)
            mask_t = torch.from_numpy(mask_mat).to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                scores = model.score_pair(user_state, cand_t)
                scores = scores.masked_fill(mask_t == 0, torch.finfo(scores.dtype).min)
            p_corr = None
            if getattr(model, "correct_head", None) is not None:
                with torch.no_grad():
                    q_emb = model.q_embed(cand_t)
                    u_exp = user_state.unsqueeze(1).expand(-1, maxN, -1)
                    logits = model.correct_head(torch.cat([u_exp, q_emb], dim=-1)).squeeze(-1)
                    p_corr = torch.sigmoid(logits).detach().cpu().numpy()
            reward_mat = np.zeros((B, maxN), dtype=np.float32)
            for b in range(B):
                valid_len = int(mask_mat[b].sum())
                if valid_len == 0:
                    continue
                mastery, weak_set, last_seen, now_t, seen_set = meta_list[b]
                p_b = p_corr[b, :valid_len] if p_corr is not None else None
                reward_mat[b, :valid_len] = compute_rewards(
                    candidates=cand_mat[b, :valid_len],
                    mastery=mastery,
                    weak_set=weak_set,
                    last_seen_term_time=last_seen,
                    now_t=now_t,
                    seen_set=seen_set,
                    p_correct=p_b,
                    p_target=p_target_dynamic,
                )
            rewards_t = torch.from_numpy(reward_mat).to(device)
            rewards_t = rewards_t.masked_fill(mask_t == 0, 0.0)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss_policy = listwise_reward_ce_stable(scores, rewards_t, temperature=temperature, reward_temp=1.0, reward_clip=2.0)
                loss = loss_policy
                if use_click_loss and click_weight > 0:
                    neg_q = sample_click_negs(target_q)
                    loss_click = sampled_softmax_click_loss(model, user_state, target_q, neg_q)
                    loss = loss + click_weight * loss_click
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item()) * B
            n += B
        print(f"[KG-Policy] epoch {ep:02d}/{policy_epochs} loss={tot/max(1,n):.4f}")

    if freeze_q_embed:
        for p in model.q_embed.parameters():
            p.requires_grad = True
    if freeze_correct_head and getattr(model, "correct_head", None) is not None:
        for p in model.correct_head.parameters():
            p.requires_grad = True

    print("KG-policy training completed.")
    return model, device
