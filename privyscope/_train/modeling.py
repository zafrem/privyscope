"""Self-contained BIOES token classifier for user fine-tuning (SRS FR-2.6).

Mirrors the privyscope-core training model but depends only on the public engine
plus torch/transformers (the ``[train]`` extra). Reuses the shared transition
scheme from ``_core.transitions`` so decoding stays identical to inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .._core.transitions import CATEGORIES, NEG_INF, category_grid


@dataclass
class TrainConfig:
    base_model: str
    labels: List[str]
    focal_gamma: float = 2.0
    transition_loss_weight: float = 0.1
    dropout: float = 0.1


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, ignore_index: int = -100) -> torch.Tensor:
    valid = targets != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0
    logits, targets = logits[valid], targets[valid]
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = logp_t.exp()
    return (-((1.0 - pt) ** gamma) * logp_t).mean()


class BioesModel(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.base_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(cfg.labels))
        self.transition_biases = nn.Parameter(torch.zeros(len(CATEGORIES)))
        grid = category_grid(cfg.labels)
        cat = torch.tensor([[-1 if c is None else c for c in row] for row in grid], dtype=torch.long)
        self.register_buffer("cat_grid", cat, persistent=False)

    def transition_matrix(self) -> torch.Tensor:
        legal = self.cat_grid >= 0
        biases = self.transition_biases[self.cat_grid.clamp(min=0)]
        return torch.where(legal, biases, biases.new_full((), NEG_INF))

    def emissions(self, input_ids, attention_mask):
        seq = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(seq))

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.emissions(input_ids, attention_mask)
        loss = None
        if labels is not None:
            B, T, L = logits.shape
            loss = focal_loss(logits.reshape(B * T, L), labels.reshape(B * T), self.cfg.focal_gamma)
            if self.cfg.transition_loss_weight > 0:
                Tm = self.transition_matrix()
                prev, cur = labels[:, :-1], labels[:, 1:]
                valid = (prev != -100) & (cur != -100)
                if valid.sum() > 0:
                    scores = logits[:, 1:, :][valid] + Tm[prev[valid]]
                    loss = loss + self.cfg.transition_loss_weight * F.cross_entropy(scores, cur[valid])
        return {"loss": loss, "logits": logits}
