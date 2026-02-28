import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm

class HybridTemporalRecommender(nn.Module):
    """LSTM over (q_embed, correct, dt) -> user state; score(user_state, q_embed)."""

    def __init__(
        self,
        num_questions,
        num_users,
        embed_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2,
        predict_correctness=True,
        pretrained_q_embeddings=None,   # torch tensor (num_questions, embed_dim)
        freeze_q_embeddings=False,
    ):
        super().__init__()
        self.num_questions = num_questions
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.predict_correctness = predict_correctness

        if pretrained_q_embeddings is not None:
            assert tuple(pretrained_q_embeddings.shape) == (num_questions, embed_dim), (
                f"Expected {(num_questions, embed_dim)}, got {tuple(pretrained_q_embeddings.shape)}"
            )
            self.q_embed = nn.Embedding.from_pretrained(
                pretrained_q_embeddings,
                freeze=freeze_q_embeddings,
                padding_idx=0,
            )
        else:
            self.q_embed = nn.Embedding(num_questions, embed_dim, padding_idx=0)
            with torch.no_grad():
                self.q_embed.weight[0].zero_()

        self.input_proj = nn.Linear(embed_dim + 2, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.correct_head = None
        if predict_correctness:
            self.correct_head = nn.Sequential(
                nn.Linear(hidden_dim + embed_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward_user_state(self, history_q, history_correct, history_dt):
        q_emb = self.q_embed(history_q)  # (B,T,E)
        dt_norm = torch.log1p(history_dt.clamp(min=0).float())
        dt_norm = (dt_norm - dt_norm.mean()) / (dt_norm.std() + 1e-6)
        x = torch.cat([q_emb, history_correct.unsqueeze(-1), dt_norm.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        _, (h, _) = self.lstm(x)
        return h[-1]  # (B,H)

    def score_pair(self, user_state, question_indices):
        q_emb = self.q_embed(question_indices)  # (B,K,E) or (K,E)
        if q_emb.dim() == 2:
            q_emb = q_emb.unsqueeze(0).expand(user_state.size(0), -1, -1)
        u = user_state.unsqueeze(1).expand(-1, q_emb.size(1), -1)
        pair = torch.cat([u, q_emb], dim=-1)
        return self.score_mlp(pair).squeeze(-1)

    def forward(self, history_q, history_correct, history_dt, target_q, neg_q, return_correct_logits=False):
        user_state = self.forward_user_state(history_q, history_correct, history_dt)
        q_stack = torch.cat([target_q.unsqueeze(1), neg_q], dim=1)  # (B,1+N)
        logits = self.score_pair(user_state, q_stack)

        out = {"logits": logits}
        if return_correct_logits and self.correct_head is not None:
            target_emb = self.q_embed(target_q)
            out["correct_logits"] = self.correct_head(torch.cat([user_state, target_emb], dim=-1)).squeeze(-1)
        return out
