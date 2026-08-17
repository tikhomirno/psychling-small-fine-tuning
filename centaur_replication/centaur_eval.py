#!/usr/bin/env python3
"""Evaluation matching Centaur's own two real evaluation methodologies -- not one
invented in parallel. See FINETUNING_PLAN_CENTAUR_REPLICATION.md for the full
verification trail.

1. `eval_loss` (compute_eval_loss): matches generalization/generalization.py exactly
   -- `trainer.evaluate()` on the held-out set, using the SAME
   DataCollatorForCompletionOnlyLM as training. This is the metric Centaur's own
   generalization test reports.

2. Summed per-item log-likelihood (compute_summed_log_likelihoods): matches
   test_adapter_full_log_likelihoods.py -- a single teacher-forced forward pass per
   sequence, cross-entropy SUMMED (never averaged/length-normalized) over each
   bracketed <<...>> span's own tokens. This is a different, complementary metric --
   generalization.py's eval_loss is a per-sequence AVERAGE over all masked tokens
   (standard HF Trainer convention); this one is a per-ITEM (per-trial) sum, letting
   individual trials be compared/aggregated per study rather than only per session.

Both are reported per held-out study/paradigm and aggregated -- neither replaces the
other, matching the plan's explicit "two real Centaur metrics, both included" design.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def compute_eval_loss(model, tokenizer, held_out_dataset, collator, batch_size: int = 1):
    """Matches generalization.py's `result = trainer.evaluate()` exactly -- builds a
    throwaway Trainer with the same collator used in training, evaluates once."""
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir="/tmp/centaur_eval_scratch",
        per_device_eval_batch_size=batch_size,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, eval_dataset=held_out_dataset,
                       data_collator=collator, tokenizer=tokenizer)
    result = trainer.evaluate()
    return result["eval_loss"]


def _labels_from_collator(collator, tokenizer, text: str):
    """Runs one record through the real training collator to get its (input_ids,
    labels) pair -- the exact same masking used in training, not a re-derived
    approximation."""
    encoded = tokenizer(text, truncation=True, max_length=dl.MAX_SEQ_TOKENS,
                         return_tensors=None)
    batch = collator([{"input_ids": encoded["input_ids"]}])
    return batch["input_ids"][0], batch["labels"][0]


def _summed_nll_per_item(input_ids: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor):
    """Splits one sequence's per-token cross-entropy into per-ITEM (per bracketed
    <<...>> span) sums -- matching test_adapter_full_log_likelihoods.py's
    `item_loss += ce[i]` accumulation over each item's own tokens, never divided by
    token count. Contiguous runs of non-masked (!= -100) label positions are treated
    as one item each."""
    shift_logits = logits[:-1, :]
    shift_labels = labels[1:]
    ce = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, reduction="none", ignore_index=-100)

    items, current = [], 0.0
    in_item = False
    for pos in range(shift_labels.shape[0]):
        if shift_labels[pos].item() != -100:
            current += ce[pos].item()
            in_item = True
        elif in_item:
            items.append(current)
            current, in_item = 0.0, False
    if in_item:
        items.append(current)
    return items


def compute_summed_log_likelihoods(model, tokenizer, held_out_dataset, collator, device="cpu"):
    """Returns a list of dicts: {experiment, participant_id, item_nlls: [...]}, one
    entry per record in held_out_dataset, each carrying the summed NLL for every
    bracketed item (trial) in that record's sequence."""
    model.eval()
    out = []
    with torch.no_grad():
        for record in held_out_dataset:
            input_ids, labels = _labels_from_collator(collator, tokenizer, record["text"])
            input_ids_t = torch.tensor([input_ids]).to(device)
            logits = model(input_ids_t).logits[0].to("cpu")
            item_nlls = _summed_nll_per_item(
                torch.tensor(input_ids), torch.tensor(labels), logits)
            out.append({
                "experiment": record.get("experiment"),
                "participant_id": record.get("participant_id"),
                "item_nlls": item_nlls,
                "n_items": len(item_nlls),
                "mean_item_nll": sum(item_nlls) / len(item_nlls) if item_nlls else None,
            })
    return out


def summarize_per_study(per_record_results: list[dict]) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    by_experiment: dict[str, list[float]] = {}
    for r in per_record_results:
        by_experiment.setdefault(r["experiment"], []).extend(r["item_nlls"])
    for exp, nlls in by_experiment.items():
        rows.append({
            "experiment": exp,
            "n_items": len(nlls),
            "mean_summed_nll_per_item": sum(nlls) / len(nlls) if nlls else None,
            "total_nll": sum(nlls),
        })
    return pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)


def run_evaluation(model, tokenizer, collator, held_out_studies=None,
                    held_out_paradigm=None, run_label: str = "run", device="cpu"):
    """Convenience entry point: loads the held-out set, computes both metrics,
    writes results to centaur_replication/results/<run_label>_*.csv."""
    held_out_dataset = dl.build_held_out_dataset(
        held_out_studies=held_out_studies, held_out_paradigm=held_out_paradigm)

    eval_loss = compute_eval_loss(model, tokenizer, held_out_dataset, collator)
    per_record = compute_summed_log_likelihoods(
        model, tokenizer, held_out_dataset, collator, device=device)
    per_study = summarize_per_study(per_record)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_study.to_csv(RESULTS_DIR / f"{run_label}_per_study.csv", index=False)
    with (RESULTS_DIR / f"{run_label}_eval_loss.txt").open("w") as fh:
        fh.write(f"{eval_loss}\n")

    print(f"eval_loss (generalization.py-style): {eval_loss}")
    print(per_study.to_string(index=False))
    return eval_loss, per_study
