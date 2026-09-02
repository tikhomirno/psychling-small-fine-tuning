#!/usr/bin/env python3
"""Full leave-one-PARADIGM-out sweep across N GPUs (default all 4, override
with --gpu-ids e.g. "1,2,3" if one GPU is already busy with something else):
PARADIGMS split round-robin into len(gpu_ids) groups, one group per GPU. All
GPU queues run IN PARALLEL with each other; within one GPU's queue, its
paradigms run strictly one after another (they share that GPU).

Fork of run_full_sweep.py's leave-one-study-out sweep, kept as a fully
separate file rather than a shared/parameterized script -- run_full_sweep.py
stays untouched, so the existing, working study-based sweep is never at risk
from this change.

PARADIGMS is computed dynamically from data_loader.paradigm_lookup(), not
hardcoded: every paradigm with >=2 member studies (per PARADIGM_AUDIT.md,
currently 5: comprehension, lexical decision, norms/ratings, self-paced
reading, word association). A single-study paradigm is excluded -- leaving it
out would be identical to leave-one-study-out, not a real generalization
test -- and this list can never drift out of sync with
experiment_metadata.csv the way a copied-in literal list could.

Deliberately NOT torchrun/DDP -- same reasoning as run_full_sweep.py (4
independent single-GPU jobs, one per CUDA_VISIBLE_DEVICES).

For every paradigm, after fine-tuning: evaluates the freshly fine-tuned model
("ours"), the real published Minitaur (zero-shot), and the untrained base
Llama ("base") against that same held-out paradigm -- via
eval_one_model_paradigm.py -- then computes all three pairwise delta
log-likelihoods (ours-vs-minitaur, base-vs-ours, base-vs-minitaur).

Run/eval labels use a loo_paradigm_/base_paradigm_/minitaur_paradigm_ prefix
-- distinct from run_full_sweep.py's loo_/base_/minitaur_ labels, so the two
sweeps' rows can never collide in the shared
results/all_runs_generalization.csv even though both write to the same file.
Sweep logs go to runs/sweep_paradigm_gpu*.log (not sweep_gpu*.log), so both
sweeps can run concurrently on the same GPU ids without interleaving logs.

Resumable: skips training a paradigm whose checkpoint (adapter_model.safetensors)
already exists, and skips any (model_label, held_out_paradigm) eval already
present in results/all_runs_generalization.csv (centaur_eval.already_evaluated)
-- safe to kill and rerun, picks up where it left off.

Usage:
    python run_full_sweep_paradigm.py                        # launches workers on all 4 GPUs, exits
    python run_full_sweep_paradigm.py --gpu-ids 1,2,3         # launches workers on GPUs 1,2,3 only
    python run_full_sweep_paradigm.py --check-held-out        # fast, GPU-free correctness check
    python run_full_sweep_paradigm.py --audit                 # fast, GPU-free per-paradigm status report
    python run_full_sweep_paradigm.py --worker --gpu 2 --paradigms p1,p2,...   # (internal)

Check progress: tail -f runs/sweep_paradigm_gpu*.log

Before running the real sweep, the same two pre-flight checks
run_full_sweep.py's docstring recommends are worth doing here too:
1. `--check-held-out` -- fast, GPU-free data-correctness check: for every
   paradigm, confirms its training pool genuinely excludes EVERY study
   belonging to that paradigm and its held-out set is genuinely present and
   non-empty.
2. A real, small mechanics check on one GPU with one paradigm before
   committing to the full sweep -- run the worker directly with a single
   --paradigms entry.

`--audit` reads only on-disk checkpoints and the master results table, same
trustworthy-over-logs pattern as run_full_sweep.py --audit.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl
import centaur_eval

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
NUM_GPUS = 4
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
MINITAUR_MODEL = "marcelbinz/Llama-3.1-Minitaur-8B"

_paradigm_counts = Counter(dl.paradigm_lookup().values())
PARADIGMS = sorted(p for p, c in _paradigm_counts.items() if c >= 2)


def _log(log_path: Path, msg: str):
    with log_path.open("a") as fh:
        fh.write(msg + "\n")


def _run_subprocess(cmd: list[str], gpu: int, log_path: Path) -> bool:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    _log(log_path, f"$ {' '.join(cmd)}")
    with log_path.open("a") as fh:
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return result.returncode == 0


def _run_name(paradigm: str, output_suffix: str) -> str:
    """Base name used for this paradigm's output_dir/adapter_dir/config path/eval
    run_labels -- 'loo_paradigm_{slug}' by default, or with output_suffix appended
    for a second, distinctly-named variant that must never collide with an
    existing run's files. Reuses data_loader._paradigm_slug (already used for the
    subsampled_by_paradigm/ cache filenames) so a paradigm name with a slash or
    space (e.g. "norms/ratings") becomes a safe directory/file name, and so the
    slugging logic can't drift out of sync between the two files."""
    return f"loo_paradigm_{dl._paradigm_slug(paradigm)}{output_suffix}"


def _checkpoint_exists(paradigm: str, output_suffix: str = "") -> bool:
    return (RUNS_DIR / _run_name(paradigm, output_suffix) / "adapter_model.safetensors").is_file()


def _build_config(paradigm: str, output_suffix: str = "") -> Path:
    name = _run_name(paradigm, output_suffix)
    with open(ROOT / "centaur_config.json") as f:
        config = json.load(f)
    config["held_out_study"] = None
    config["held_out_paradigm"] = paradigm
    config["output_dir"] = str(RUNS_DIR / name)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def _run_eval(paradigm: str, model_name: str, model_label: str, adapter_dir: str | None,
              run_label: str, gpu: int, log_path: Path) -> bool:
    if centaur_eval.already_evaluated(run_label):
        _log(log_path, f"  eval '{run_label}' already in results, skipping")
        return True
    cmd = [sys.executable, str(ROOT / "eval_one_model_paradigm.py"),
           "--model-name", model_name, "--held-out-paradigm", paradigm,
           "--run-label", run_label, "--model-label", model_label]
    if adapter_dir:
        cmd += ["--adapter-dir", adapter_dir]
    ok = _run_subprocess(cmd, gpu, log_path)
    if not ok:
        _log(log_path, f"  EVAL FAILED: {run_label}")
    return ok


def check_held_out_correctness(paradigms: list[str]) -> bool:
    """Fast, GPU-free correctness check: for every paradigm, confirms
    build_training_dataset's training pool genuinely excludes ALL of that
    paradigm's member studies (not just one) and build_held_out_dataset's
    held-out set is genuinely present and non-empty for it. No model loading,
    no CUDA/Unsloth needed. Prints a PASS/FAIL line per paradigm and returns
    whether every paradigm passed."""
    all_ok = True
    for paradigm in paradigms:
        member_studies = set(dl.studies_in_paradigm(paradigm))
        train_ds, _ = dl.build_training_dataset(held_out_paradigm=paradigm, inner_split=False)
        leaked = set(train_ds["experiment"]) & member_studies
        held = dl.build_held_out_dataset(held_out_paradigm=paradigm)
        ok = not leaked and len(held) > 0
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {paradigm} ({sorted(member_studies)}): "
              f"train_pool={len(train_ds)} sequences (leaked={leaked or 'none'}), "
              f"held_out={len(held)} sequences")
    return all_ok


def audit_sweep(paradigms: list[str]) -> bool:
    """GPU-free audit: for each paradigm, cross-references three independent
    sources of truth -- (1) whether a checkpoint exists on disk, (2) whether
    base/minitaur/ours eval rows exist in the master results table, (3)
    whether a delta comparison was computed -- and reports a clear per-paradigm
    status. Mirrors run_full_sweep.py's audit_sweep exactly, keyed by paradigm
    name instead of study name. Returns whether every paradigm is COMPLETE."""
    import pandas as pd

    results = (pd.read_csv(centaur_eval.MASTER_RESULTS_CSV)
               if centaur_eval.MASTER_RESULTS_CSV.exists() else pd.DataFrame())

    rows = []
    for paradigm in paradigms:
        has_checkpoint = _checkpoint_exists(paradigm)
        evaluated = {}
        for label, model_name in [("ours", centaur_eval.MODEL_OURS),
                                   ("base", centaur_eval.MODEL_BASE),
                                   ("minitaur", centaur_eval.MODEL_MINITAUR)]:
            evaluated[label] = not results.empty and bool((
                (results["model_name"] == model_name) & (results["held_out_name"] == paradigm)
            ).any())
        delta_path = centaur_eval.RESULTS_DIR / f"delta_minitaur_vs_ours__{dl._paradigm_slug(paradigm)}_overall.csv"
        rows.append({"paradigm": paradigm, "checkpoint": has_checkpoint, "eval": evaluated,
                     "delta_computed": delta_path.is_file()})

    counts = {}
    for r in rows:
        vals = list(r["eval"].values())
        if r["checkpoint"] and all(vals) and r["delta_computed"]:
            status = "COMPLETE"
        elif r["checkpoint"] and not any(vals):
            status = "TRAINED_NOT_EVALUATED"
        elif r["checkpoint"]:
            status = "PARTIALLY_EVALUATED"
        else:
            status = "NOT_TRAINED"
        counts[status] = counts.get(status, 0) + 1
        detail = ", ".join(f"{k}={'Y' if v else 'n'}" for k, v in r["eval"].items())
        print(f"  [{status:<22}] {r['paradigm']:<25} ckpt={'Y' if r['checkpoint'] else 'n'} "
              f"{detail} delta={'Y' if r['delta_computed'] else 'n'}")

    print(f"\n--- summary ({len(paradigms)} paradigms) ---")
    for status in ["COMPLETE", "TRAINED_NOT_EVALUATED", "PARTIALLY_EVALUATED", "NOT_TRAINED"]:
        print(f"  {status}: {counts.get(status, 0)}")
    return counts.get("COMPLETE", 0) == len(paradigms)


def run_worker(gpu: int, paradigms: list[str], output_suffix: str = ""):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNS_DIR / f"sweep_paradigm_gpu{gpu}.log"
    _log(log_path, f"=== worker for GPU {gpu}: {len(paradigms)} paradigms: {paradigms} "
                    f"(output_suffix={output_suffix!r}) ===")

    for paradigm in paradigms:
        _log(log_path, f"\n--- {paradigm} ---")
        name = _run_name(paradigm, output_suffix)
        adapter_dir = str(RUNS_DIR / name)

        if _checkpoint_exists(paradigm, output_suffix):
            _log(log_path, "  checkpoint already exists, skipping training")
        else:
            config_path = _build_config(paradigm, output_suffix)
            ok = _run_subprocess(
                [sys.executable, str(ROOT / "centaur_finetune.py"), str(config_path)],
                gpu, log_path)
            if not ok:
                _log(log_path, f"  TRAINING FAILED for {paradigm}, skipping its evals")
                continue

        # eval run_labels also carry the suffix -- otherwise a second, corrected
        # variant of an already-evaluated paradigm would be silently skipped by
        # already_evaluated() thinking the original run's eval already covers it.
        _run_eval(paradigm, BASE_MODEL, centaur_eval.MODEL_OURS, adapter_dir,
                  name, gpu, log_path)
        _run_eval(paradigm, BASE_MODEL, centaur_eval.MODEL_BASE, None,
                  f"base_paradigm_{dl._paradigm_slug(paradigm)}{output_suffix}", gpu, log_path)
        _run_eval(paradigm, MINITAUR_MODEL, centaur_eval.MODEL_MINITAUR, None,
                  f"minitaur_paradigm_{dl._paradigm_slug(paradigm)}{output_suffix}", gpu, log_path)

        for model_a, model_b in [
            (centaur_eval.MODEL_MINITAUR, centaur_eval.MODEL_OURS),
            (centaur_eval.MODEL_BASE, centaur_eval.MODEL_OURS),
            (centaur_eval.MODEL_BASE, centaur_eval.MODEL_MINITAUR),
        ]:
            try:
                centaur_eval.compute_delta_log_likelihood(
                    held_out_name=paradigm, model_a=model_a, model_b=model_b)
            except Exception as e:
                _log(log_path, f"  delta {model_a} vs {model_b} for {paradigm} failed: {e}")

    _log(log_path, f"\n=== worker for GPU {gpu} done ===")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="internal: run one GPU's queue")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--paradigms", default=None, help="comma-separated")
    parser.add_argument("--gpu-ids", default=None,
                         help="comma-separated physical GPU ids to use, e.g. '1,2,3' if GPU 0 "
                              "is already busy with something else. Default: all 4 (0,1,2,3).")
    parser.add_argument("--exclude-paradigms", default=None,
                         help="comma-separated paradigms to skip entirely")
    parser.add_argument("--output-suffix", default="",
                         help="appended to every run's output_dir/adapter_dir/config path/eval "
                              "run_label (e.g. '_subsampled') -- use when running a second, "
                              "distinctly-named variant of a paradigm that must not collide with "
                              "an existing run's files")
    parser.add_argument("--check-held-out", action="store_true",
                         help="fast, GPU-free correctness check across --paradigms (default: all "
                              f"{len(PARADIGMS)}) -- confirms each paradigm's pool excludes every "
                              "member study and its held-out set is present, then exits without "
                              "launching anything")
    parser.add_argument("--audit", action="store_true",
                         help="fast, GPU-free per-paradigm status report across --paradigms "
                              "(default: all) -- cross-references checkpoints on disk against eval "
                              "rows in the master results table and delta CSVs. Reports COMPLETE / "
                              "TRAINED_NOT_EVALUATED / PARTIALLY_EVALUATED / NOT_TRAINED per paradigm.")
    args = parser.parse_args()

    paradigms_arg = args.paradigms.split(",") if args.paradigms else PARADIGMS

    if args.check_held_out:
        print(f"checking held-out correctness for {len(paradigms_arg)} paradigms...")
        ok = check_held_out_correctness(paradigms_arg)
        print(f"\n{'ALL PASS' if ok else 'SOME FAILED -- see above'}")
        sys.exit(0 if ok else 1)

    if args.audit:
        print(f"auditing {len(paradigms_arg)} paradigms...")
        ok = audit_sweep(paradigms_arg)
        sys.exit(0 if ok else 1)

    if args.worker:
        run_worker(args.gpu, paradigms_arg, output_suffix=args.output_suffix)
        return

    exclude = set(args.exclude_paradigms.split(",")) if args.exclude_paradigms else set()
    pool = [p for p in PARADIGMS if p not in exclude]
    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else list(range(NUM_GPUS))
    groups = [pool[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    print(f"splitting {len(pool)} paradigms ({len(exclude)} excluded: {exclude or 'none'}) "
          f"across {len(gpu_ids)} GPUs {gpu_ids}:")
    for gpu, paradigms in zip(gpu_ids, groups):
        print(f"  GPU {gpu}: {len(paradigms)} paradigms: {paradigms}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu, paradigms in zip(gpu_ids, groups):
        log_path = RUNS_DIR / f"sweep_paradigm_gpu{gpu}.log"
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker", "--gpu", str(gpu), "--paradigms", ",".join(paradigms)]
        if args.output_suffix:
            cmd += ["--output-suffix", args.output_suffix]
        proc = subprocess.Popen(cmd)
        procs.append(proc)
        print(f"launched GPU {gpu} worker, pid {proc.pid}, log: {log_path}")

    print(f"\nall {len(gpu_ids)} GPU workers launched as background processes.")
    print("this launcher process can exit safely -- the workers are independent")
    print("subprocess.Popen children, not tied to this process's lifetime once")
    print("detached (run this whole script itself via nohup/disown if launching")
    print("from an interactive shell you intend to close).")
    print(f"\nCheck progress: tail -f {RUNS_DIR}/sweep_paradigm_gpu*.log")


if __name__ == "__main__":
    main()
