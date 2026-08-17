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
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def compute_eval_loss(model, tokenizer, held_out_dataset, collator, batch_size: int = 1,
                       max_length: int = None, device: str = "cpu"):
    """Reports the same NUMBER generalization.py's `trainer.evaluate()` reports
    (mean cross-entropy over all masked/non-ignored tokens in the held-out set,
    using the same collator as training) -- computed with a direct manual loop
    rather than an actual `Trainer` object.

    CHANGED THIS SESSION: originally built a throwaway `Trainer` and called
    `.evaluate()`, matching generalization.py's own code shape literally. In
    practice, on this machine, `Trainer.evaluate()` hung indefinitely (confirmed
    across two separate attempts, including after setting
    `dataloader_num_workers=0` to rule out a suspected Mac/MPS DataLoader-forking
    deadlock -- that fix didn't resolve it, so the real cause is inside `Trainer`'s
    other setup/bookkeeping, not identified further). A manual loop reports the
    identical statistic (mean masked-token cross-entropy) without any Trainer
    machinery, and turned out simpler and more transparent besides -- kept this way
    rather than reverting once CUDA/Unsloth is available, since there's no
    Trainer-specific behavior actually being relied on here.

    max_length defaults to dl.MAX_SEQ_TOKENS (32768) to match the real recipe on
    CUDA -- DO NOT use that default for local Mac/MPS testing, pass an explicit
    small cap (see centaur_replication/hour_test.py's SAFE_MAX_LEN and its safety
    warning) or risk the same crash a full-length local forward pass caused before.
    """
    max_length = max_length or dl.MAX_SEQ_TOKENS
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for record in held_out_dataset:
            enc = tokenizer(record["text"], truncation=True, max_length=max_length)
            batch = collator([{"input_ids": enc["input_ids"]}])
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            n_real = (shift_labels != -100).sum().item()
            if n_real == 0:
                continue
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="sum", ignore_index=-100)
            total_loss += loss.item()
            total_tokens += n_real
    return total_loss / total_tokens if total_tokens else None


def _labels_from_collator(collator, tokenizer, text: str, max_length: int = None):
    """Runs one record through the real training collator to get its (input_ids,
    labels) pair -- the exact same masking used in training, not a re-derived
    approximation.

    max_length defaults to dl.MAX_SEQ_TOKENS (32768, correct for the real recipe on
    CUDA) -- pass an explicit small cap for local Mac/MPS testing. See
    centaur_replication/hour_test.py's module docstring: an uncapped forward pass at
    real sequence lengths crashed this machine once already."""
    max_length = max_length or dl.MAX_SEQ_TOKENS
    encoded = tokenizer(text, truncation=True, max_length=max_length,
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


def compute_summed_log_likelihoods(model, tokenizer, held_out_dataset, collator, device="cpu",
                                    max_length: int = None):
    """Returns a list of dicts: {experiment, participant_id, item_nlls: [...]}, one
    entry per record in held_out_dataset, each carrying the summed NLL for every
    bracketed item (trial) in that record's sequence.

    max_length defaults to dl.MAX_SEQ_TOKENS (32768) -- override with a small safe
    cap for local testing (see _labels_from_collator's docstring)."""
    model.eval()
    out = []
    with torch.no_grad():
        for record in held_out_dataset:
            input_ids, labels = _labels_from_collator(collator, tokenizer, record["text"], max_length)
            # The real trl.DataCollatorForCompletionOnlyLM returns torch.Tensor
            # rows (unlike a hand-rolled collator returning plain lists) -- accept
            # either without double-wrapping a tensor inside torch.tensor([...]).
            input_ids_ct = input_ids if torch.is_tensor(input_ids) else torch.tensor(input_ids)
            labels_ct = labels if torch.is_tensor(labels) else torch.tensor(labels)
            input_ids_t = input_ids_ct.unsqueeze(0).to(device)
            logits = model(input_ids_t).logits[0].to("cpu")
            item_nlls = _summed_nll_per_item(input_ids_ct, labels_ct, logits)
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


MASTER_RESULTS_CSV = RESULTS_DIR / "all_runs_generalization.csv"
MASTER_COLUMNS = ["run_label", "model_name", "held_out_type", "held_out_name", "eval_loss",
                   "mean_summed_nll_per_item", "sem_nll_per_item", "n_items", "n_sequences", "timestamp"]
# Canonical model_name values for the 3-way comparison -- use these exactly so
# compute_delta_log_likelihood (below) can find matching rows without string-parsing
# run_label. Nothing enforces this at the type level; it's a convention.
MODEL_BASE, MODEL_MINITAUR, MODEL_OURS = "base", "minitaur", "ours"


def _append_master_row(row: dict):
    """Appends one row to the persistent master results table -- incremental,
    matching this repo's own established dynamic-save convention (write after every
    result, not just at the end), so a 72-run sweep that gets interrupted partway
    still has every completed run's generalization score safely on disk."""
    import csv
    from datetime import datetime, timezone

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {**row, "timestamp": datetime.now(timezone.utc).isoformat()}
    is_new = not MASTER_RESULTS_CSV.exists()
    with MASTER_RESULTS_CSV.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MASTER_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in MASTER_COLUMNS})


def already_evaluated(run_label: str) -> bool:
    """Resume-skip check: has this run_label already been recorded in the master
    table? Lets a sweep script skip finished runs instead of re-evaluating them."""
    if not MASTER_RESULTS_CSV.exists():
        return False
    import pandas as pd
    existing = pd.read_csv(MASTER_RESULTS_CSV)
    return run_label in set(existing["run_label"])


def _save_item_nlls(run_label: str, per_record: list[dict]) -> "pd.DataFrame":
    """Flattens per_record's nested item_nlls into one row per response
    (experiment, participant_id, item_index, nll) and writes it to
    <run_label>_item_nlls.csv. This is what makes delta computation possible
    across separately-run evaluations (different models, different times,
    different checkpoints) -- two runs' items are paired by
    (experiment, participant_id, item_index), not by re-running both models in the
    same process. item_index is the response's position within its own sequence
    (0-indexed) -- stable as long as both runs score the same held-out source data,
    which build_held_out_dataset guarantees (same JSONL files, same order)."""
    import pandas as pd

    rows = []
    for r in per_record:
        for i, nll in enumerate(r["item_nlls"]):
            rows.append({"experiment": r["experiment"], "participant_id": r["participant_id"],
                         "item_index": i, "nll": nll})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{run_label}_item_nlls.csv", index=False)
    return df


def run_evaluation(model, tokenizer, collator, held_out_studies=None,
                    held_out_paradigm=None, run_label: str = "run",
                    model_name: str = MODEL_OURS, device="cpu", max_length: int = None,
                    max_participants_per_study: int = None):
    """Convenience entry point: loads the held-out set, computes both metrics,
    writes per-run detail to centaur_replication/results/<run_label>_*.csv
    (including the full per-item NLL list, needed for paired delta computation
    later), AND appends one summary row to results/all_runs_generalization.csv --
    call this once per leave-one-out run; the master CSV is what accumulates
    across the full sweep.

    `model_name` should be one of MODEL_BASE/MODEL_MINITAUR/MODEL_OURS for the 3-way
    comparison (FINETUNING_PLAN_CENTAUR_REPLICATION.md §7) -- e.g. run this 3x per
    held-out study (base Llama, zero-shot Minitaur, your fine-tuned model), same
    held_out_studies/held_out_paradigm each time, different model + run_label.
    compute_delta_log_likelihood below depends on model_name being set correctly,
    not on run_label naming conventions.

    `max_length` defaults to dl.MAX_SEQ_TOKENS (32768) -- correct for the real
    recipe on CUDA, but pass an explicit small cap (e.g. 1024) for any local Mac/MPS
    run. See hour_test.py's module docstring for why this default is dangerous
    off-CUDA.

    `max_participants_per_study` is None for every real evaluation -- pass an int
    (e.g. 1) only for a tiny cluster smoke test (see cluster_smoke_test.py), so the
    held-out set stays as small as the training pool it's paired with.
    """
    held_out_dataset = dl.build_held_out_dataset(
        held_out_studies=held_out_studies, held_out_paradigm=held_out_paradigm,
        max_participants_per_study=max_participants_per_study)

    eval_loss = compute_eval_loss(model, tokenizer, held_out_dataset, collator,
                                   max_length=max_length, device=device)
    per_record = compute_summed_log_likelihoods(
        model, tokenizer, held_out_dataset, collator, device=device, max_length=max_length)
    per_study = summarize_per_study(per_record)
    item_df = _save_item_nlls(run_label, per_record)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_study.to_csv(RESULTS_DIR / f"{run_label}_per_study.csv", index=False)
    with (RESULTS_DIR / f"{run_label}_eval_loss.txt").open("w") as fh:
        fh.write(f"{eval_loss}\n")

    all_nlls = item_df["nll"].tolist()
    mean_nll = sum(all_nlls) / len(all_nlls) if all_nlls else None
    sem_nll = (pd_series_sem(all_nlls) if all_nlls else None)
    _append_master_row({
        "run_label": run_label,
        "model_name": model_name,
        "held_out_type": "paradigm" if held_out_paradigm else "study",
        "held_out_name": held_out_paradigm or ",".join(held_out_studies or []),
        "eval_loss": eval_loss,
        "mean_summed_nll_per_item": mean_nll,
        "sem_nll_per_item": sem_nll,
        "n_items": len(all_nlls),
        "n_sequences": len(per_record),
    })

    print(f"eval_loss (generalization.py-style): {eval_loss}")
    print(per_study.to_string(index=False))
    print(f"appended summary row to {MASTER_RESULTS_CSV}, "
          f"per-item detail to {run_label}_item_nlls.csv")
    return eval_loss, per_study


def pd_series_sem(values: list[float]) -> float:
    """Standard error of the mean -- std / sqrt(n), ddof=1. Tiny helper so
    run_evaluation doesn't need a full pandas Series just for this."""
    import statistics

    n = len(values)
    if n < 2:
        return 0.0
    return statistics.stdev(values) / (n ** 0.5)


def compute_delta_log_likelihood(held_out_name: str, model_a: str = MODEL_MINITAUR,
                                  model_b: str = MODEL_OURS,
                                  run_label_a: str = None, run_label_b: str = None) -> "pd.DataFrame":
    """Delta log-likelihood between two models, computed the way Centaur's own
    paper reports it (figure caption, quoted): "Difference in log-likelihood of
    [model] and [baseline]... for each experiment. A value of zero corresponds to
    the goodness-of-fit of the [baseline]... Log-likelihoods are averaged over
    responses... Error bars correspond to the standard error of the mean."

    Concretely, per experiment:
    1. Load both runs' per-item NLLs (<run_label>_item_nlls.csv).
    2. PAIR them on (experiment, participant_id, item_index) -- the same human
       response, scored by both models. This is what makes the SEM statistically
       valid: it's the SEM of the per-response PAIRED difference, not two
       independently-computed SEMs naively combined (which would overstate the
       error if the two models find the same items hard/easy, which they will,
       since it's the same human data).
    3. delta_per_response = model_a's NLL - model_b's NLL for that response (so
       delta = LL(model_b) - LL(model_a) in log-likelihood terms -- positive means
       model_b fits human responses better than model_a, matching the paper's
       "value above zero indicates improved goodness-of-fit" convention).
    4. Average delta_per_response within each experiment, with SEM across those
       per-response deltas -- exactly "log-likelihoods averaged over responses,
       error bars = SEM" from the quoted caption.

    `held_out_name` is REQUIRED, not optional -- this is deliberate. A fair
    generalization comparison only exists on the ONE held-out set a given "ours"
    checkpoint never trained on; across a full leave-one-out sweep there will be one
    ours_loo_X run per held-out study/paradigm X, each with its own run_label.
    Silently grabbing "whichever ours run is most recent" (an earlier version of
    this function did that) would happily pair the wrong checkpoint with the wrong
    held-out set and produce a meaningless number. Passing held_out_name forces the
    lookup to the specific (model_name, held_out_name) pair in the master table.

    run_label_a/run_label_b override the lookup entirely, for full manual control.
    """
    import pandas as pd

    def _resolve_label(model_name, explicit):
        if explicit:
            return explicit
        if not MASTER_RESULTS_CSV.exists():
            raise FileNotFoundError(f"{MASTER_RESULTS_CSV} doesn't exist yet")
        master = pd.read_csv(MASTER_RESULTS_CSV)
        rows = master[(master["model_name"] == model_name) & (master["held_out_name"] == held_out_name)]
        if rows.empty:
            raise ValueError(
                f"no run found for model_name={model_name!r}, held_out_name={held_out_name!r} "
                f"in {MASTER_RESULTS_CSV} -- run_evaluation must be called for this exact "
                f"(model, held-out set) pair first")
        if len(rows) > 1:
            print(f"note: {len(rows)} runs found for model_name={model_name!r}, "
                  f"held_out_name={held_out_name!r} -- using the most recent by timestamp")
        return rows.sort_values("timestamp").iloc[-1]["run_label"]

    label_a = _resolve_label(model_a, run_label_a)
    label_b = _resolve_label(model_b, run_label_b)

    path_a = RESULTS_DIR / f"{label_a}_item_nlls.csv"
    path_b = RESULTS_DIR / f"{label_b}_item_nlls.csv"
    for p in (path_a, path_b):
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} not found -- run_evaluation must be called (and complete) "
                f"for both models before computing a delta")

    items_a = pd.read_csv(path_a).rename(columns={"nll": f"nll_{model_a}"})
    items_b = pd.read_csv(path_b).rename(columns={"nll": f"nll_{model_b}"})
    paired = items_a.merge(items_b, on=["experiment", "participant_id", "item_index"], how="inner")
    dropped_a = len(items_a) - len(paired)
    dropped_b = len(items_b) - len(paired)
    if dropped_a or dropped_b:
        print(f"warning: {dropped_a} items from {model_a} and {dropped_b} items from "
              f"{model_b} had no match in the other run and were excluded from the "
              f"paired comparison (held-out sets/response counts weren't identical)")

    # delta uses NLLs directly: LL = -NLL, so LL(b) - LL(a) = NLL(a) - NLL(b).
    paired["delta_log_likelihood"] = paired[f"nll_{model_a}"] - paired[f"nll_{model_b}"]

    rows = []
    for exp, group in paired.groupby("experiment"):
        deltas = group["delta_log_likelihood"].tolist()
        rows.append({
            "experiment": exp,
            "n_responses": len(deltas),
            f"mean_nll_{model_a}": group[f"nll_{model_a}"].mean(),
            f"sem_nll_{model_a}": pd_series_sem(group[f"nll_{model_a}"].tolist()),
            f"mean_nll_{model_b}": group[f"nll_{model_b}"].mean(),
            f"sem_nll_{model_b}": pd_series_sem(group[f"nll_{model_b}"].tolist()),
            "mean_delta_log_likelihood": sum(deltas) / len(deltas),
            "sem_delta_log_likelihood": pd_series_sem(deltas),
        })
    per_experiment = pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)

    all_deltas = paired["delta_log_likelihood"].tolist()
    overall = {
        "n_responses": len(all_deltas),
        f"mean_nll_{model_a}": paired[f"nll_{model_a}"].mean(),
        f"mean_nll_{model_b}": paired[f"nll_{model_b}"].mean(),
        "mean_delta_log_likelihood": sum(all_deltas) / len(all_deltas) if all_deltas else None,
        "sem_delta_log_likelihood": pd_series_sem(all_deltas) if all_deltas else None,
    }

    # Persist everything -- both models' per-response likelihoods (nll_<model_a>,
    # nll_<model_b> columns) AND their difference (delta_log_likelihood), plus the
    # per-experiment aggregate with both models' means/SEMs and the delta's SEM.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # held_out_name is included in the filename -- without it, every held-out set's
    # comparison in a full sweep would overwrite the same 3 files.
    safe_held_out = re.sub(r"[^A-Za-z0-9_.-]+", "_", held_out_name)
    pair_tag = f"{model_a}_vs_{model_b}__{safe_held_out}"
    per_response_path = RESULTS_DIR / f"delta_{pair_tag}_per_response.csv"
    per_experiment_path = RESULTS_DIR / f"delta_{pair_tag}_per_experiment.csv"
    overall_path = RESULTS_DIR / f"delta_{pair_tag}_overall.csv"

    paired.to_csv(per_response_path, index=False)
    per_experiment.to_csv(per_experiment_path, index=False)
    pd.DataFrame([overall]).to_csv(overall_path, index=False)

    print(f"delta_log_likelihood = LL({model_b}) - LL({model_a}); zero = {model_a}'s "
          f"goodness-of-fit, positive = {model_b} fits human responses better")
    print(f"\nper-experiment:\n{per_experiment.to_string(index=False)}")
    print(f"\noverall (n={overall['n_responses']} responses): "
          f"mean={overall['mean_delta_log_likelihood']:.4f}, "
          f"SEM={overall['sem_delta_log_likelihood']:.4f}")
    print(f"\nsaved:\n  {per_response_path}  (both models' NLLs + delta, per response)"
          f"\n  {per_experiment_path}  (both models' means/SEMs + delta, per experiment)"
          f"\n  {overall_path}  (single-row overall summary)")

    return per_experiment, overall


def compute_all_deltas(model_a: str = MODEL_MINITAUR, model_b: str = MODEL_OURS) -> "pd.DataFrame":
    """Sweep-wide convenience: finds every held_out_name that has BOTH model_a and
    model_b evaluated in the master table, calls compute_delta_log_likelihood for
    each one (writing that held-out set's own 3 files, per the held_out_name-scoped
    naming above), and returns one combined DataFrame -- one row per held-out
    study/paradigm, with each row's own mean + SEM delta. This is the actual
    generalization-comparison table for your full leave-one-out sweep: skips any
    held-out set where one of the two models hasn't been evaluated yet, so it's
    safe to call partway through a long-running sweep.
    """
    import pandas as pd

    if not MASTER_RESULTS_CSV.exists():
        raise FileNotFoundError(f"{MASTER_RESULTS_CSV} doesn't exist yet")
    master = pd.read_csv(MASTER_RESULTS_CSV)

    names_a = set(master[master["model_name"] == model_a]["held_out_name"])
    names_b = set(master[master["model_name"] == model_b]["held_out_name"])
    ready = sorted(names_a & names_b)
    missing_b = sorted(names_a - names_b)
    if missing_b:
        print(f"skipping {len(missing_b)} held-out set(s) with {model_a} but no "
              f"{model_b} evaluation yet: {missing_b}")

    rows = []
    for held_out_name in ready:
        per_experiment, overall = compute_delta_log_likelihood(
            held_out_name=held_out_name, model_a=model_a, model_b=model_b)
        rows.append({"held_out_name": held_out_name, **overall})

    summary = pd.DataFrame(rows).sort_values("held_out_name").reset_index(drop=True)
    summary_path = RESULTS_DIR / f"all_deltas_{model_a}_vs_{model_b}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{len(summary)} held-out sets compared; combined summary saved to {summary_path}")
    return summary
