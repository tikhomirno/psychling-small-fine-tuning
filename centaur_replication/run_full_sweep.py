#!/usr/bin/env python3
"""Full leave-one-out sweep across N GPUs (default all 4, override with
--gpu-ids e.g. "1,2,3" if one GPU is already busy with something else):
data_loader.STUDIES split round-robin into len(gpu_ids) groups, one group per
GPU. All GPU queues run IN PARALLEL with each other; within one GPU's queue,
its studies run strictly one after another (they share that GPU).

Deliberately NOT torchrun/DDP -- this is 4 independent single-GPU jobs, one per
CUDA_VISIBLE_DEVICES, matching the pattern already verified working this session
(and avoiding the Unsloth+torchrun multi-process hang debugged earlier, which
never occurred with this simpler single-GPU-per-process approach).

For every study, after fine-tuning: evaluates the freshly fine-tuned model
("ours"), the real published Minitaur (zero-shot), and the untrained base
Llama ("base") against that same held-out study -- via eval_one_model.py, the
exact pattern already verified working earlier this session -- then computes
all three pairwise delta log-likelihoods (ours-vs-minitaur, base-vs-ours,
base-vs-minitaur).

Resumable: skips training a study whose checkpoint (adapter_model.safetensors)
already exists, and skips any (model_label, held_out_study) eval already present
in results/all_runs_generalization.csv (centaur_eval.already_evaluated) -- safe
to kill and rerun, picks up where it left off rather than redoing finished work.

Usage:
    python run_full_sweep.py                        # launches workers on all 4 GPUs, exits
    python run_full_sweep.py --gpu-ids 1,2,3         # launches workers on GPUs 1,2,3 only
    python run_full_sweep.py --check-held-out        # fast, GPU-free correctness check (see below)
    python run_full_sweep.py --worker --gpu 2 --studies s1,s2,...   # (internal)

Check progress: tail -f runs/sweep_gpu*.log

Before running the real sweep, two independent pre-flight checks are worth
doing (see CLUSTER_SETUP.md section 7 for the full walkthrough):
1. `--check-held-out` -- a fast, GPU-free data-correctness check (seconds, not
   minutes) that confirms, for every study, its training pool genuinely
   excludes it and its held-out set is genuinely present and non-empty. Doesn't
   load any model -- pure data_loader-level verification.
2. A real, small mechanics check on one GPU with 1-2 studies (needs a real
   GPU/Unsloth, takes as long as one real training run) -- run the worker
   directly with a tiny --studies list before committing to the full sweep.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl
import centaur_eval

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
NUM_GPUS = 4
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
MINITAUR_MODEL = "marcelbinz/Llama-3.1-Minitaur-8B"


def _log(log_path: Path, msg: str):
    with log_path.open("a") as fh:
        fh.write(msg + "\n")


def _run_subprocess(cmd: list[str], gpu: int, log_path: Path) -> bool:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    _log(log_path, f"$ {' '.join(cmd)}")
    with log_path.open("a") as fh:
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return result.returncode == 0


def _run_name(study: str, output_suffix: str) -> str:
    """Base name used for this study's output_dir/adapter_dir/config path/eval
    run_labels -- 'loo_{study}' by default, or 'loo_{study}{output_suffix}' when
    running a second, distinctly-named variant of a study that must never
    collide with an existing run's files (e.g. a corrected-pool comparison
    against a study that's already training elsewhere with the old pool)."""
    return f"loo_{study}{output_suffix}"


def _checkpoint_exists(study: str, output_suffix: str = "") -> bool:
    return (RUNS_DIR / _run_name(study, output_suffix) / "adapter_model.safetensors").is_file()


def _build_config(study: str, output_suffix: str = "") -> Path:
    name = _run_name(study, output_suffix)
    with open(ROOT / "centaur_config.json") as f:
        config = json.load(f)
    config["held_out_study"] = study
    config["held_out_paradigm"] = None
    config["output_dir"] = str(RUNS_DIR / name)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def _run_eval(study: str, model_name: str, model_label: str, adapter_dir: str | None,
              run_label: str, gpu: int, log_path: Path) -> bool:
    if centaur_eval.already_evaluated(run_label):
        _log(log_path, f"  eval '{run_label}' already in results, skipping")
        return True
    cmd = [sys.executable, str(ROOT / "eval_one_model.py"),
           "--model-name", model_name, "--held-out-study", study,
           "--run-label", run_label, "--model-label", model_label]
    if adapter_dir:
        cmd += ["--adapter-dir", adapter_dir]
    ok = _run_subprocess(cmd, gpu, log_path)
    if not ok:
        _log(log_path, f"  EVAL FAILED: {run_label}")
    return ok


def check_held_out_correctness(studies: list[str]) -> bool:
    """Fast, GPU-free correctness check: for every study, confirms
    build_training_dataset's training pool genuinely excludes it (no leakage)
    and build_held_out_dataset's held-out set is genuinely present and
    non-empty for it. No model loading, no CUDA/Unsloth needed -- pure
    data_loader-level verification, runs in seconds even across the whole
    28-study corpus. Prints a PASS/FAIL line per study and returns whether
    every study passed."""
    all_ok = True
    for study in studies:
        train_ds, _ = dl.build_training_dataset(held_out_studies=[study], inner_split=False)
        leaked = set(train_ds["experiment"]) & {study}
        held = dl.build_held_out_dataset(held_out_studies=[study])
        ok = not leaked and len(held) > 0
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {study}: train_pool={len(train_ds)} sequences "
              f"(leaked={leaked or 'none'}), held_out={len(held)} sequences")
    return all_ok


def run_worker(gpu: int, studies: list[str], output_suffix: str = ""):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNS_DIR / f"sweep_gpu{gpu}.log"
    _log(log_path, f"=== worker for GPU {gpu}: {len(studies)} studies: {studies} "
                    f"(output_suffix={output_suffix!r}) ===")

    for study in studies:
        _log(log_path, f"\n--- {study} ---")
        name = _run_name(study, output_suffix)
        adapter_dir = str(RUNS_DIR / name)

        if _checkpoint_exists(study, output_suffix):
            _log(log_path, "  checkpoint already exists, skipping training")
        else:
            config_path = _build_config(study, output_suffix)
            ok = _run_subprocess(
                [sys.executable, str(ROOT / "centaur_finetune.py"), str(config_path)],
                gpu, log_path)
            if not ok:
                _log(log_path, f"  TRAINING FAILED for {study}, skipping its evals")
                continue

        # eval run_labels also carry the suffix -- otherwise a second, corrected
        # variant of an already-evaluated study would be silently skipped by
        # already_evaluated() thinking the original run's eval already covers it.
        _run_eval(study, BASE_MODEL, centaur_eval.MODEL_OURS, adapter_dir,
                  name, gpu, log_path)
        _run_eval(study, BASE_MODEL, centaur_eval.MODEL_BASE, None,
                  f"base_{study}{output_suffix}", gpu, log_path)
        _run_eval(study, MINITAUR_MODEL, centaur_eval.MODEL_MINITAUR, None,
                  f"minitaur_{study}{output_suffix}", gpu, log_path)

        for model_a, model_b in [
            (centaur_eval.MODEL_MINITAUR, centaur_eval.MODEL_OURS),
            (centaur_eval.MODEL_BASE, centaur_eval.MODEL_OURS),
            (centaur_eval.MODEL_BASE, centaur_eval.MODEL_MINITAUR),
        ]:
            try:
                centaur_eval.compute_delta_log_likelihood(
                    held_out_name=study, model_a=model_a, model_b=model_b)
            except Exception as e:
                _log(log_path, f"  delta {model_a} vs {model_b} for {study} failed: {e}")

    _log(log_path, f"\n=== worker for GPU {gpu} done ===")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="internal: run one GPU's queue")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--studies", default=None, help="comma-separated")
    parser.add_argument("--gpu-ids", default=None,
                         help="comma-separated physical GPU ids to use, e.g. '1,2,3' if GPU 0 "
                              "is already busy with something else. Default: all 4 (0,1,2,3).")
    parser.add_argument("--exclude-studies", default=None,
                         help="comma-separated studies to skip entirely -- e.g. a study that's "
                              "already training elsewhere under the old pool, so this sweep "
                              "doesn't also try to write to its output_dir")
    parser.add_argument("--output-suffix", default="",
                         help="appended to every run's output_dir/adapter_dir/config path/eval "
                              "run_label (e.g. '_subsampled') -- use when running a second, "
                              "distinctly-named variant of a study that must not collide with "
                              "an existing run's files")
    parser.add_argument("--check-held-out", action="store_true",
                         help="fast, GPU-free correctness check across --studies (default: all "
                              "28) -- confirms each study's pool excludes it and its held-out "
                              "set is present, then exits without launching anything")
    args = parser.parse_args()

    studies_arg = args.studies.split(",") if args.studies else dl.STUDIES

    if args.check_held_out:
        print(f"checking held-out correctness for {len(studies_arg)} studies...")
        ok = check_held_out_correctness(studies_arg)
        print(f"\n{'ALL PASS' if ok else 'SOME FAILED -- see above'}")
        sys.exit(0 if ok else 1)

    if args.worker:
        run_worker(args.gpu, studies_arg, output_suffix=args.output_suffix)
        return

    exclude = set(args.exclude_studies.split(",")) if args.exclude_studies else set()
    pool = [s for s in dl.STUDIES if s not in exclude]
    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else list(range(NUM_GPUS))
    groups = [pool[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    print(f"splitting {len(pool)} studies ({len(exclude)} excluded: {exclude or 'none'}) "
          f"across {len(gpu_ids)} GPUs {gpu_ids}:")
    for gpu, studies in zip(gpu_ids, groups):
        print(f"  GPU {gpu}: {len(studies)} studies: {studies}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu, studies in zip(gpu_ids, groups):
        log_path = RUNS_DIR / f"sweep_gpu{gpu}.log"
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker", "--gpu", str(gpu), "--studies", ",".join(studies)]
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
    print(f"\nCheck progress: tail -f {RUNS_DIR}/sweep_gpu*.log")


if __name__ == "__main__":
    main()
