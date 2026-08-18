#!/usr/bin/env python3
"""Centaur-exact fine-tuning, adapted for leave-one-out runs on our own data.

Structurally mirrors marcelbinz/Llama-3.1-Centaur-70B/finetune.py as closely as
possible -- same dataclass shape, same FastLanguageModel/UnslothTrainer calls, same
collator construction -- so results are genuinely comparable to Centaur's own. See
FINETUNING_PLAN_CENTAUR_REPLICATION.md for the full verification trail behind every
choice here (each one traced to the real source, not assumed).

REQUIRES CUDA (Unsloth depends on Triton, no Apple Silicon/MPS support) -- run on
ARC/DAIS, not locally. For a portable local pipeline sanity check, see
smoke_test.py, which is explicitly NOT this recipe (no Unsloth, no embedding-LR
mechanism, no real quantization -- a pipeline check only).

Example invocation -- copy centaur_config.json per run, override held_out_study/
output_dir, then:
    python centaur_finetune.py loo_balota2007_LDT_config.json

centaur_config.json's values match scripts/cluster_train.sh exactly (verified this
session), with two deliberate exceptions: model_name_or_path is our own
unsloth/Meta-Llama-3.1-8B-bnb-4bit (matching Minitaur, per this session's earlier
decision), not cluster_train.sh's literal 70B path; and num_train_epochs is set to
1 (matching the paper's own methods-text description), not the real script's
literal 5 -- see FINETUNING_PLAN_CENTAUR_REPLICATION.md for that discrepancy.
5 epochs is what the real code does and is the more literal replication, but for
a 28-study sweep the wall-clock cost multiplies accordingly (measured this
session: ~5x), so 1 epoch is the deliberate default here for tractability -- pass
num_train_epochs=5 explicitly in a run's own config if literal recipe fidelity
matters more than sweep turnaround time for a specific run. A plain CLI-flags
invocation also works (HfArgumentParser), but skips the config file's guarantee
that every value matches this session's chosen recipe rather than falling back
to a generic HF default.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "e.g. unsloth/Meta-Llama-3.1-8B-bnb-4bit"})
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0)


@dataclass
class DataTrainingArguments:
    dataset_text_field: str = field(default="text")
    max_seq_length: Optional[int] = field(default=dl.MAX_SEQ_TOKENS)


@dataclass
class LeaveOneOutArguments:
    """Not part of Centaur's own finetune.py -- our leave-one-out addition. A run
    is invoked with exactly one of these set (or neither, for the full-pool
    reference model), matching cluster_train.sh's one-script-many-invocations
    pattern rather than a different script per condition."""
    held_out_study: Optional[str] = field(default=None)
    held_out_paradigm: Optional[str] = field(default=None)
    max_participants_per_study: Optional[int] = field(
        default=None,
        metadata={"help": "Cap real participants per study to this many -- for a "
                           "tiny cluster smoke test only (see cluster_smoke_test.py). "
                           "None (default) means every real run uses the full dataset."})
    disable_subsampling: bool = field(
        default=False,
        metadata={"help": "Skip the 100k-trial-per-study / 500k-trial-per-paradigm caps "
                           "entirely and pool the OLD, fully unbounded data instead. False "
                           "(the default) for every normal run -- set True only for a "
                           "deliberate full-data-vs-subsampled comparison run."})


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,
                                LeaveOneOutArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, loo_args, training_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, loo_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    if loo_args.held_out_study and loo_args.held_out_paradigm:
        raise ValueError("set at most one of --held_out_study / --held_out_paradigm")

    held_out_studies = [loo_args.held_out_study] if loo_args.held_out_study else None
    # Pooling/parsing the full corpus is expensive (raw JSONL + on-the-fly zip
    # extraction across ~27 studies) and data_loader.build_training_dataset now
    # caches its result to disk -- but under a multi-GPU `torchrun` launch, every
    # DDP rank runs this exact main() independently, so without this barrier all N
    # ranks would race to build (and redundantly pay for) that same expensive pool
    # simultaneously. main_process_first() makes rank 0 build+cache it alone while
    # the other ranks wait, then they all hit the now-populated cache -- a no-op
    # single pass-through when not running under torchrun at all.
    from accelerate import PartialState
    with PartialState().main_process_first():
        train_dataset, eval_datasets = dl.build_training_dataset(
            held_out_studies=held_out_studies,
            held_out_paradigm=loo_args.held_out_paradigm,
            seed=training_args.seed,
            max_participants_per_study=loo_args.max_participants_per_study,
            disable_subsampling=loo_args.disable_subsampling,
        )

    # Everything below mirrors finetune.py's own structure line-for-line where
    # possible -- imported here, not at module top, so this file can still be
    # imported (e.g. by tests) on machines without CUDA/Unsloth installed.
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
    from trl import DataCollatorForCompletionOnlyLM

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=data_args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_args.seed,
        use_rslora=True,
        loftq_config=None,
    )

    # response_template/instruction_template as token-ID sequences with the first
    # token sliced off -- verified this is exactly how finetune.py builds them, not
    # a plain-string match.
    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id, instruction_template=r_id, tokenizer=tokenizer)

    # Recent transformers hard-validates eval_strategy != "no" against having a
    # real eval_dataset (Trainer._validate_args). eval_datasets is None whenever
    # every pooled study has <=1 real participant after leakage_safe_split's own
    # guard -- true for every real run's inner-eval split in practice (studies
    # always have many participants), but exactly what happens under
    # max_participants_per_study=1 (cluster_smoke_test.py). Falling back to "no"
    # only in that case, never silently changing behavior for a real run.
    effective_eval_strategy = training_args.eval_strategy if eval_datasets is not None else "no"

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
        dataset_num_proc=8,
        data_collator=collator,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            warmup_steps=training_args.warmup_steps,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            embedding_learning_rate=training_args.learning_rate / 10,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            log_level=training_args.log_level,
            logging_strategy=training_args.logging_strategy,
            logging_steps=training_args.logging_steps,
            # Centaur's own cluster_train.sh flag is --evaluation_strategy; current
            # transformers renamed this TrainingArguments field to eval_strategy --
            # same value, name changed upstream since their script was written.
            eval_strategy=effective_eval_strategy,
            eval_steps=training_args.eval_steps,
            save_strategy=training_args.save_strategy,
            save_steps=training_args.save_steps,
            optim=training_args.optim,
            weight_decay=training_args.weight_decay,
            lr_scheduler_type=training_args.lr_scheduler_type,
            seed=training_args.seed,
            output_dir=training_args.output_dir,
        ),
    )
    # Auto-resume if this output_dir already has a periodic checkpoint (e.g. the
    # job was killed/died partway through) -- resume_from_checkpoint=True makes
    # HF Trainer find and load the latest checkpoint-N/ in output_dir itself, no
    # path needed. A fresh output_dir (no checkpoints yet) trains from scratch as
    # before. This was previously hardcoded to None (never resume) with a manual
    # "edit this line to True" note in CLUSTER_SETUP.md -- automated instead,
    # since restarting a killed multi-hour run from scratch wastes real GPU time
    # for no reason when a perfectly good checkpoint already exists on disk.
    output_dir = Path(training_args.output_dir)
    has_checkpoint = output_dir.is_dir() and any(output_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=has_checkpoint if has_checkpoint else None)
    trainer.save_model()


if __name__ == "__main__":
    main()
