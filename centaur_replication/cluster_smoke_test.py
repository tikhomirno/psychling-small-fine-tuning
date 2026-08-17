#!/usr/bin/env python3
"""Cluster pre-flight smoke test -- MECHANICS ONLY, numbers are meaningless.

Runs the REAL leave-one-out pipeline end to end (real Unsloth, real CUDA, real
centaur_finetune.py as a subprocess, real centaur_eval.run_evaluation) on an
artificially tiny subsample: at most 1 real participant per study, one small
held-out study, 1 epoch. This is NOT a correctness/quality check of the model or
the science -- it is a "does the mechanism run without crashing on this cluster,
in these Python/CUDA versions, with this config schema" gate, meant to run BEFORE
committing to the real ~hours-per-run, many-study sweep.

Nothing here should ever be compared to real results in
results/all_runs_generalization.csv -- every artifact this script produces is
prefixed `smoke_` specifically so it can never be confused with, or silently
pollute, real runs. Safe to re-run any number of times; every artifact is
overwritten, not appended to.

REQUIRES CUDA (Unsloth depends on Triton, no Apple Silicon/MPS support) -- run on
the cluster, not locally. See CLUSTER_SETUP.md section 4 for how to invoke this
from a Posit Workbench Jupyter session.

Exit code 0 = PASS, 1 = FAIL (with the failing stage and exception printed).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl
import centaur_eval

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
SMOKE_OUTPUT_DIR = RUNS_DIR / "smoke_test"
SMOKE_CONFIG_PATH = RUNS_DIR / "smoke_test_config.json"
# Smallest non-CHUNK_CANDIDATES study (40 participants, measured directly this
# session) -- excluding CHUNK_CANDIDATES from the held-out choice keeps the
# held-out load path simple; chunking still gets exercised on the training side
# automatically, since only the held-out study is excluded from the pool.
DEFAULT_HELD_OUT_STUDY = "hilton2021_comprehension"


class SmokeTestFailure(Exception):
    pass


def gpu_sanity_check():
    import torch

    if not torch.cuda.is_available():
        raise SmokeTestFailure(
            "torch.cuda.is_available() is False -- this session's kernel isn't "
            "attached to a GPU node, or the resource profile chosen at session "
            "launch has no GPU. Unsloth will not run without CUDA. See "
            "CLUSTER_SETUP.md section 0/1.")
    print(f"GPU OK: {torch.cuda.get_device_name(0)}")

    try:
        import unsloth  # noqa: F401
    except Exception as e:
        raise SmokeTestFailure(f"`import unsloth` failed: {e!r}") from e
    print("unsloth import OK")


def build_smoke_config(held_out_study: str) -> Path:
    with open(ROOT / "centaur_config.json") as f:
        config = json.load(f)

    config["held_out_study"] = held_out_study
    config["held_out_paradigm"] = None
    config["max_participants_per_study"] = 1
    config["num_train_epochs"] = 1
    # Lowered from the real config's 32/100 -- a ~1-participant-per-study pool
    # (well under 30 records) needs a much shorter schedule to reliably produce at
    # least one optimizer step and one checkpoint quickly. The goal here is
    # finishing fast and proving the mechanism, not reproducing real training
    # dynamics.
    config["gradient_accumulation_steps"] = 1
    config["save_steps"] = 1
    config["eval_steps"] = 999999  # unchanged: still functionally disabled
    config["output_dir"] = str(SMOKE_OUTPUT_DIR)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SMOKE_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    return SMOKE_CONFIG_PATH, config["model_name_or_path"]


def run_finetune(config_path: Path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "centaur_finetune.py"), str(config_path)],
        cwd=str(ROOT.parent),
    )
    if result.returncode != 0:
        raise SmokeTestFailure(
            f"centaur_finetune.py exited with code {result.returncode} -- see the "
            f"subprocess output above for the actual error.")


def assert_checkpoint_exists(output_dir: Path):
    if not output_dir.is_dir():
        raise SmokeTestFailure(f"expected output dir {output_dir} does not exist")
    # peft's standard adapter save layout (confirmed against this repo's own
    # CLUSTER_SETUP.md reload instructions, which load output_dir via
    # PeftModel.from_pretrained -- that call requires exactly these two files).
    required = ["adapter_model.safetensors", "adapter_config.json"]
    missing = [f for f in required if not (output_dir / f).is_file()]
    if missing:
        found = sorted(p.name for p in output_dir.iterdir())
        raise SmokeTestFailure(
            f"checkpoint at {output_dir} is missing {missing} -- found: {found}")
    print(f"checkpoint OK: {output_dir} contains {required}")


def run_smoke_eval(output_dir: Path, base_model_name: str, held_out_study: str):
    from unsloth import FastLanguageModel
    from trl import DataCollatorForCompletionOnlyLM
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=dl.MAX_SEQ_TOKENS,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(output_dir))

    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id, instruction_template=r_id, tokenizer=tokenizer)

    eval_loss, per_study = centaur_eval.run_evaluation(
        model, tokenizer, collator,
        held_out_studies=[held_out_study],
        run_label="smoke_test",
        model_name="smoke",
        device="cuda",
        max_length=dl.MAX_SEQ_TOKENS,
        max_participants_per_study=1,
    )
    return eval_loss, per_study


def assert_finite(eval_loss, per_study):
    if eval_loss is None or not math.isfinite(eval_loss):
        raise SmokeTestFailure(f"eval_loss is not finite: {eval_loss!r}")
    for col in per_study.select_dtypes(include="number").columns:
        bad = per_study[~per_study[col].apply(
            lambda v: v is not None and math.isfinite(v))]
        if not bad.empty:
            raise SmokeTestFailure(
                f"non-finite values in per_study column {col!r}:\n{bad}")
    print("all metrics finite: OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out-study", default=DEFAULT_HELD_OUT_STUDY,
                         help=f"default: {DEFAULT_HELD_OUT_STUDY}")
    args = parser.parse_args()

    t0 = time.time()
    stage = "gpu_sanity_check"
    try:
        gpu_sanity_check()

        stage = "build_smoke_config"
        config_path, base_model_name = build_smoke_config(args.held_out_study)
        print(f"smoke config written to {config_path} "
              f"(held_out_study={args.held_out_study}, max_participants_per_study=1)")

        stage = "run_finetune"
        run_finetune(config_path)

        stage = "assert_checkpoint_exists"
        assert_checkpoint_exists(SMOKE_OUTPUT_DIR)

        stage = "run_smoke_eval"
        eval_loss, per_study = run_smoke_eval(SMOKE_OUTPUT_DIR, base_model_name, args.held_out_study)

        stage = "assert_finite"
        assert_finite(eval_loss, per_study)

    except Exception as e:
        print(f"\n==== SMOKE TEST: FAIL (stage: {stage}) ====")
        print(f"{type(e).__name__}: {e}")
        return 1

    print(f"\n==== SMOKE TEST: PASS ({time.time() - t0:.1f}s) ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
