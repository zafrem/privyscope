"""User fine-tuning runner (SRS FR-2.6).

Reads JSONL (optionally remapping labels via a custom label space), fine-tunes a
BIOES model with Focal Loss, and writes the FR-2.6 artifacts:
``config.json``, ``model.safetensors``, ``finetune_summary.json``, ``USAGE.txt``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .._core.bioes import Span, spans_to_bioes
from .._eval.dataset import read_eval_jsonl
from .._model.config import DEFAULT_BASE_MODEL, load_label_space, remap_label
from .modeling import BioesModel, TrainConfig

USAGE_TEMPLATE = """\
# privyscope fine-tuned checkpoint

Produced by `privyscope train`. To use this checkpoint, export it to ONNX with the
privyscope-core `export_onnx.py` pipeline, then load via:

    from privyscope import Privyscope
    engine = Privyscope.from_pretrained(cache_dir="{out}")

Labels: {labels}
Base model: {base}
"""


class _Dataset(Dataset):
    def __init__(self, path, tokenizer, label2id, label_map, max_length=256):
        self.items: List[Dict[str, torch.Tensor]] = []
        for text, spans in read_eval_jsonl(path):
            mapped: List[Span] = []
            for s in spans:
                tgt = remap_label(s.label, label_map)
                if tgt is not None:
                    mapped.append(Span(tgt, s.start, s.end))
            enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
            offsets = enc["offset_mapping"]
            tags = spans_to_bioes(mapped, offsets)
            labels = [-100 if (a == 0 and b == 0) else label2id[t] for t, (a, b) in zip(tags, offsets)]
            self.items.append({
                "input_ids": torch.tensor(enc["input_ids"]),
                "attention_mask": torch.tensor(enc["attention_mask"]),
                "labels": torch.tensor(labels),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _collate(batch, pad_id):
    maxlen = max(x["input_ids"].size(0) for x in batch)
    pad = lambda k, v: torch.stack([  # noqa: E731
        torch.cat([x[k], x[k].new_full((maxlen - x[k].size(0),), v)]) for x in batch
    ])
    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100)}


def run_train(args) -> dict:
    base = args.base_model or DEFAULT_BASE_MODEL
    labels, label2id, id2label, label_map = load_label_space(args.label_space_json)
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )

    model = BioesModel(TrainConfig(base_model=base, labels=labels)).to(device)
    train_ds = _Dataset(args.dataset, tokenizer, label2id, label_map)
    pad_id = tokenizer.pad_token_id or 0
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: _collate(b, pad_id))

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(total * 0.06), total)

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            if step % 50 == 0:
                print(f"[train] epoch {epoch} step {step}/{len(loader)} loss {out['loss'].item():.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(out_dir / "model.safetensors"))
    tokenizer.save_pretrained(out_dir)
    (out_dir / "config.json").write_text(json.dumps({
        "model_type": "privyscope-bioes", "base_model": base, "labels": labels,
        "id2label": id2label, "label2id": label2id,
        "transition_categories": ["outside", "begin", "single", "inside", "end", "exit"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "base_model": base, "labels": labels, "epochs": args.epochs,
        "train_examples": len(train_ds), "custom_label_space": bool(args.label_space_json),
        "output_dir": str(out_dir),
    }
    (out_dir / "finetune_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "USAGE.txt").write_text(USAGE_TEMPLATE.format(out=out_dir, labels=labels, base=base), encoding="utf-8")
    return summary
