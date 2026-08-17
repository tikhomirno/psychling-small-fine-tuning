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
session), with one deliberate exception: model_name_or_path is our own
unsloth/Meta-Llama-3.1-8B-bnb-4bit (matching Minitaur, per this session's earlier
decision), not cluster_train.sh's literal 70B path. num_train_epochs defaults to 5,
matching the actual training script, not the "one epoch" description in the paper's
methods text -- see FINETUNING_PLAN_CENTAUR_REPLICATION.md for that discrepancy and
why the code is treated as authoritative here. A plain CLI-flags invocation also
works (HfArgumentParser), but skips the config file's guarantee that every value
matches Centaur's real recipe rather than falling back to a generic HF default.
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
    train_dataset, eval_datasets = dl.build_training_dataset(
        held_out_studies=held_out_studies,
        held_out_paradigm=loo_args.held_out_paradigm,
        seed=training_args.seed,
        max_participants_per_study=loo_args.max_participants_per_study,
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
            eval_strategy=training_args.eval_strategy,
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
    trainer.train(resume_from_checkpoint=None)
    trainer.save_model()


if __name__ == "__main__":
    main()
