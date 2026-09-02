#!/usr/bin/env python3
"""Loads one model (optionally with a LoRA adapter on top) and evaluates it
against one held-out PARADIGM via centaur_eval.run_evaluation -- a fork of
eval_one_model.py for run_full_sweep_paradigm.py, since data_loader/centaur_eval
already support held_out_paradigm= end-to-end. Kept as a separate file rather
than adding a flag to eval_one_model.py so that script (used by the existing,
working leave-one-study-out sweep) stays completely untouched.

REQUIRES CUDA + Unsloth -- run on the cluster only, one GPU per invocation
(set CUDA_VISIBLE_DEVICES before launching, don't rely on device_map).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", required=True,
                    help="base model repo, e.g. unsloth/Meta-Llama-3.1-8B-bnb-4bit or marcelbinz/Llama-3.1-Minitaur-8B")
    p.add_argument("--adapter-dir", default=None,
                    help="LoRA adapter dir/repo to apply on top of --model-name, if any (omit for base/Minitaur)")
    p.add_argument("--held-out-paradigm", required=True)
    p.add_argument("--run-label", required=True)
    p.add_argument("--model-label", required=True,
                    help="stored as model_name in the master results table -- use "
                         "centaur_eval.MODEL_BASE/MODEL_MINITAUR/MODEL_OURS for the "
                         "3-way comparison so compute_delta_log_likelihood can find it")
    p.add_argument("--max-length", type=int, default=dl.MAX_SEQ_TOKENS)
    args = p.parse_args()

    from unsloth import FastLanguageModel
    from trl import DataCollatorForCompletionOnlyLM
    import centaur_eval

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name, max_seq_length=args.max_length, dtype=None, load_in_4bit=True,
    )
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id, instruction_template=r_id, tokenizer=tokenizer)

    centaur_eval.run_evaluation(
        model, tokenizer, collator,
        held_out_paradigm=args.held_out_paradigm,
        run_label=args.run_label,
        model_name=args.model_label,
        device="cuda",
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
