# Running centaur_finetune.py from a JupyterLab/Notebook-only cluster

Companion to `FINETUNING_PLAN_CENTAUR_REPLICATION.md` and the cluster-transfer
walkthrough from this session's chat. Everything below is meant to be pasted into
notebook cells, in order — no terminal required, though a JupyterLab terminal (File
→ New → Terminal) works too if available.

## 0. Launch a Posit Workbench session

1. Open your Posit Workbench homepage and click **New Session**.
2. Choose **Jupyter** (or **JupyterLab**) as the session type — not RStudio/VS Code.
3. Pick a **GPU-enabled resource profile/cluster** from the launch dialog's dropdown.
   The exact name varies by institution (look for anything mentioning "GPU", a CUDA
   partition, or a queue name your admin has told you has GPUs attached). If you're
   not sure which profile has a GPU, check with your cluster admin *before* launching
   — picking a CPU-only profile means step 1 below (and `cluster_smoke_test.py`'s own
   first check) will fail immediately.
4. Wait for the session to start, then open it. You'll land in a JupyterLab (or
   classic Notebook) interface in your browser.
5. **Terminal access varies by install.** Some Workbench deployments expose a
   JupyterLab terminal (File → New → Terminal, or the Launcher tab's "Terminal"
   tile); locked-down installs don't. Every command in this document is written to
   work purely through notebook cells (`!shell command` or `subprocess.run(...)`),
   so a terminal is convenient if you have one but never required.
6. **Get the code onto the cluster.** Two options:
   - **Preferred**: if the session has outbound network access, git-clone the
     `cluster_bundle` repo directly from a notebook cell. The repo is **public**,
     so no credentials are needed:
     ```python
     !git clone https://github.com/tikhomirno/psychling-small-fine-tuning.git cluster_bundle
     %cd cluster_bundle
     ```
     If `cluster_bundle` is already cloned here from an earlier session, pull the
     update instead of re-cloning:
     ```python
     %cd cluster_bundle
     !git pull origin main
     ```
   - **Fallback**: if outbound git access isn't available, use Workbench's file
     browser to upload `cluster_bundle/` — drag-and-drop onto the file browser pane,
     or use its upload-arrow icon. For a zipped bundle, upload the zip and unzip it
     from a notebook cell (`!unzip cluster_bundle.zip`).

All paths in the rest of this document are relative to wherever `cluster_bundle/`
(or its contents) ended up — cd into that directory first if you haven't already.

## 1. Sanity-check the GPU first

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
!nvidia-smi
```
If `torch.cuda.is_available()` is `False`, stop here and check the notebook's kernel
is actually attached to a GPU node — Unsloth will not run without CUDA (see
`FINETUNING_PLAN_CENTAUR_REPLICATION.md` §10). `cluster_smoke_test.py` (section 4
below) re-runs this exact check programmatically as its own first gate, so if this
cell passes, that script's first stage will too.

## 2. Install dependencies

```python
!pip install -q -r centaur_replication/requirements.txt
```
Restart the kernel after this (Unsloth's installer sometimes needs a fresh process
to pick up its CUDA-specific build correctly) — Kernel → Restart Kernel, then
continue from the next cell.

## 3. Authenticate with HuggingFace (for the gated Llama-3.1 download)

```python
from huggingface_hub import login
from getpass import getpass
login(token=getpass("HF token: "))
```

## 4. Pre-flight smoke test (run this before any real training)

Before committing to a real leave-one-out run (~hours) or a full sweep, run
`cluster_smoke_test.py`. It exercises the exact same real pipeline —
`centaur_finetune.py`, real Unsloth, real CUDA, real `centaur_eval.py` — but on an
artificially tiny subsample: at most 1 real participant per study, 1 epoch. It
finishes in minutes, not hours, and proves the mechanism actually runs in this
environment (right dependency versions, disk/permissions OK, config schema
matches) before you spend real GPU time finding that out the hard way. The numbers
it produces are meaningless — see its own module docstring — only PASS/FAIL matters.

```python
!python centaur_replication/cluster_smoke_test.py
```

It prints `==== SMOKE TEST: PASS ====` (exit code 0) or
`==== SMOKE TEST: FAIL (stage: <name>) ====` with the exception (exit code 1). **A
FAIL here means stop and diagnose before running the real sweep** — the class of
problem this catches (dependency mismatch, wrong resource profile, disk/permissions
issue, a broken config field) is exactly what's cheap to find now and expensive to
find hours into a real run.

Safe to re-run any number of times — every artifact it writes is `smoke_`-prefixed
and overwritten each time, never touching a real run's output directory or adding
rows to `results/all_runs_generalization.csv` that could be confused with real
results.

## 5. Build the subsampled corpus (run once before any real training)

The pooled training corpus is heavily imbalanced by trial count -- one study
alone (`lynott2020lancaster`) contributes over 10.6M individual response
targets, while the smallest contributes under 3,000. `data_loader.py` now caps
every study at 100k trials by default (leave-one-study-out) and every paradigm
at 500k trials with equal-per-study sampling (leave-one-paradigm-out, a
separate flow selected by setting `held_out_paradigm` instead of
`held_out_study` in your config -- same script, same invocation, no other
changes needed).

This subsampling happens automatically either way, but running the
materialization script once up front builds real, inspectable files instead of
silently recomputing the subsample live on first use:

```python
!python centaur_replication/build_subsampled_corpus.py
```

This writes `centaur_replication/data/subsampled_by_study/<study>.jsonl` (28
files), `subsampled_by_paradigm/<paradigm>.jsonl` (15 files), and a
`subsampled_corpus_summary.csv` showing the before/after trial count for every
study and paradigm -- worth a quick look to confirm the imbalance fix landed
where you expect. Safe to re-run any time; fully deterministic (fixed seed),
every output gets overwritten.

## 6. One leave-one-out run

Only proceed here after section 4's smoke test has printed PASS.

Copy the base config, set which study to hold out and where checkpoints go, write it
to disk, then invoke the script exactly the way `cluster_train.sh` invokes
`finetune.py` — as a subprocess, not an in-process import, so it behaves identically
to a real cluster job:

```python
import json, subprocess

with open("centaur_replication/centaur_config.json") as f:
    config = json.load(f)

config["held_out_study"] = "balota2007_LDT"   # or set held_out_paradigm instead
config["output_dir"] = "centaur_replication/runs/loo_balota2007_LDT"

run_config_path = "centaur_replication/runs/loo_balota2007_LDT_config.json"
import os
os.makedirs("centaur_replication/runs", exist_ok=True)
with open(run_config_path, "w") as f:
    json.dump(config, f, indent=2)

subprocess.run(["python", "centaur_replication/centaur_finetune.py", run_config_path], check=True)
```

`!python centaur_replication/centaur_finetune.py {run_config_path}` works equally
well as a `!`-prefixed shell cell if you'd rather see live output stream directly
instead of via `subprocess.run`.

**Long-running**: this is ~6,600 optimizer steps at 5 epochs for a typical
leave-one-out pool (`FINETUNING_PLAN_CENTAUR_REPLICATION.md` §5) — likely hours, not
minutes. `--save_steps 100` checkpoints every 100 steps regardless, and
`centaur_finetune.py` auto-resumes from the latest checkpoint in `output_dir` if
one already exists there (no manual edit needed) -- so if the kernel dies, the
session times out, or you kill it yourself, just rerun the exact same command
(or the same `run_full_sweep.py` invocation) and it picks up where it left off
instead of retraining from scratch.

## 7. Many leave-one-out runs, in parallel across all 4 GPUs

For running the full sweep (all 28 studies -> 28 fine-tuned models), use
`run_full_sweep.py` rather than looping sequentially in a notebook: it splits
`data_loader.STUDIES` round-robin into 4 groups (7 studies each, for the current
28-study corpus) and runs each group on its own dedicated GPU, all 4 GPUs
working in parallel -- no need to hand-launch one job per GPU yourself.

This is deliberately **not** `torchrun`/DDP (4 independent single-GPU jobs
instead) -- a genuine multi-GPU DDP attempt with Unsloth hung reliably in
testing this session (`accelerate launch --multi_gpu`/`torchrun` both stalled
indefinitely mid-startup); 4 independent single-GPU jobs is simpler, more
robust, and gets the same aggregate 4x throughput for a sweep of many
independent runs, which is what this actually is. Each individual job still
uses the subprocess approach from step 6 internally (clean process, clean GPU
memory per run -- no stale optimizer state or stacked LoRA adapters).

For every study, after fine-tuning it also evaluates the freshly fine-tuned
model, the real published Minitaur (zero-shot), and the untrained base Llama --
all three against that same held-out study -- and computes all three pairwise
delta log-likelihoods, exactly the comparisons you'd otherwise run by hand.

```python
!python centaur_replication/run_full_sweep.py
```

This launches all 4 GPU workers as background processes and returns
immediately -- it does not block the notebook. Check progress with:

```python
!tail -n 30 centaur_replication/runs/sweep_gpu0.log centaur_replication/runs/sweep_gpu1.log \
             centaur_replication/runs/sweep_gpu2.log centaur_replication/runs/sweep_gpu3.log
```

**Resumable**: if a worker (or the whole session) dies partway through, just
re-run the same command -- it skips any study whose checkpoint already exists
and any evaluation already recorded in
`results/all_runs_generalization.csv`, so nothing gets redone.

To run only a subset of studies instead of the full 28, or to change how many
GPUs are used, edit `NUM_GPUS`/`BASE_MODEL`/`MINITAUR_MODEL` at the top of
`run_full_sweep.py`, or invoke one GPU's queue directly for a manual subset:

```python
!python centaur_replication/run_full_sweep.py --worker --gpu 0 \
    --studies balota2007_LDT,marson2026_eplep,kyroelaeinen2022_valence
```

## 8. Evaluation, after a run finishes

**Note on reloading a saved checkpoint — not independently verified this session,
flagging rather than asserting**: `trainer.save_model()` saves the LoRA *adapter*
only (small — matches Centaur's own `-adapter` HF repos, not their merged model
repos), not a standalone full model. Centaur's own repo has a separate `merge.py`
specifically to combine an adapter with its base model, which suggests reloading an
adapter checkpoint for further use is a two-step process (load the base model, then
apply the adapter), not a one-call `FastLanguageModel.from_pretrained(adapter_dir)`.
The standard `peft` pattern is:
```python
import sys
sys.path.insert(0, "centaur_replication")
from unsloth import FastLanguageModel
from trl import DataCollatorForCompletionOnlyLM
from peft import PeftModel
import centaur_eval

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # the BASE model, not the run's output_dir
    max_seq_length=32768, dtype=None, load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "centaur_replication/runs/loo_balota2007_LDT")
l_id = tokenizer(" <<").input_ids[1:]
r_id = tokenizer(">>").input_ids[1:]
collator = DataCollatorForCompletionOnlyLM(response_template=l_id, instruction_template=r_id, tokenizer=tokenizer)

eval_loss, per_study = centaur_eval.run_evaluation(
    model, tokenizer, collator,
    held_out_studies=["balota2007_LDT"],
    run_label="loo_balota2007_LDT",
    device="cuda",
)
```
Results land in `centaur_replication/results/loo_balota2007_LDT_*.csv`.

## 9. If disk space on the cluster is tight

Each run's checkpoint directory is small (LoRA adapters only, tens of MB — not full
model weights), but `--save_steps 100` across ~6,600 steps means ~66 checkpoints per
run unless you thin them. Add `save_total_limit` to the config if needed:
```python
config["save_total_limit"] = 3  # keep only the 3 most recent checkpoints
```
(Not in Centaur's own recipe — `cluster_train.sh` doesn't set it — but harmless to
add locally if storage is the binding constraint; doesn't change the training math.)
