#!/usr/bin/env python3
"""Tests whether the model reproduces the classic lexical-decision word-
frequency effect (higher-frequency real words -> lower NLL / more confident
correct response), for both the un-fine-tuned base model and our fine-tuned
one. Run ONCE on the cluster -- everything it needs is already there except
norms_cache/freqwords_en_50k.txt (add to package_for_cluster.sh before
pushing) and the 5 LDT studies' base_<study>/loo_<study> item_nlls.csv (must
already exist, i.e. the study sweep must have finished evaluating these 5).

Uses data_loader.build_held_out_dataset directly (not a hand-rolled JSONL
reader) so the extracted stimulus words are read from EXACTLY the same
held-out set (same subsampling cache, same order) that centaur_eval actually
scored -- a hand-rolled reader risked silently diverging from that if the
materialized subsampled_by_study/ cache was in play. Each LDT study has its
own prompt wording (verified against real prompt text this session, not
assumed identical across studies just because the paradigm is the same), so
each gets its own stimulus-extraction regex below.

Output is a small, aggregated CSV (results/frequency_effect_summary.csv,
frequency-binned per study/model) -- deliberately not the raw per-item join,
which would be as large as the item_nlls.csv files themselves and defeat the
point of computing this on the cluster in the first place.

Usage: python compute_frequency_join.py
"""
import math
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FREQ_PATH = Path(__file__).resolve().parent.parent / "norms_cache" / "freqwords_en_50k.txt"

# Verified against real prompt text, full corpus (2026-09) -- each LDT study
# phrases its trials differently even though the paradigm is the same.
# Captures the STIMULUS word (not the response key) immediately before
# " You press <<". `(.+?)` (not `[^']+`) for the single-quoted patterns --
# real stimuli can contain an apostrophe themselves (e.g. "generap's",
# "Christ's"), which broke a `[^']+` version of this pattern on 36/3256
# balota2007_LDT records before catching it against real data; `.+?` lets
# the anchor after the closing quote (`'. You press <<`) resolve the
# ambiguity via backtracking instead.
STUDY_STIMULUS_PATTERNS = {
    "balota2007_LDT": re.compile(r"You see '(.+?)'\.\s*You press <<"),
    "guenther2020LDT": re.compile(r'The word "([^"]+)" appears on the screen\.\s*You press <<'),
    "keuleers2011_britishlexiconproject": re.compile(r"The string is '(.+?)'\.\s*You press <<"),
    "marson2026_eplep": re.compile(r"You see (.+?)\.\s*You press <<"),
    # priming design: the scored item is the TARGET, not the cue
    "hutchison2013_semantic": re.compile(r"Target: '(.+?)'\.\s*You press <<"),
}
# hutchison2013_semantic's prompts_fixed.jsonl mixes TWO genuinely different
# sub-tasks across different records -- 512 records are pure lexical-decision
# ("... You press <<key>>"), 257 are pure naming ("Naming time: <<435>> ms."),
# NEVER mixed within one session (verified full-corpus, this session). The
# same class of issue PARADIGM_AUDIT.md already caught and fixed for
# frank2013_reading's mixed self-paced-reading/eye-tracking data -- not yet
# fixed here. Harmless for THIS script specifically: a pure-naming record
# yields zero regex matches, contributes zero stimulus rows, and the inner
# merge below silently excludes it rather than misaligning item_index --
# but it means hutchison2013_semantic's naming-task participants are
# currently being pooled into the "lexical decision" paradigm bucket
# elsewhere (the sweep, the theory-diagnostics notebook's entropy numbers),
# which is a separate, real finding worth a dedicated fix later, not
# addressed by this script.


def load_freqwords() -> dict[str, int]:
    freq = {}
    with FREQ_PATH.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                word, count = parts
                freq[word.lower()] = int(count)
    return freq


def extract_stimuli(study: str, text: str) -> list[str]:
    return STUDY_STIMULUS_PATTERNS[study].findall(text)


def main():
    if not FREQ_PATH.is_file():
        raise FileNotFoundError(f"{FREQ_PATH} not found -- add norms_cache/ to package_for_cluster.sh first")
    freq = load_freqwords()
    print(f"loaded {len(freq)} frequency entries from {FREQ_PATH}")

    stim_rows = []
    for study in STUDY_STIMULUS_PATTERNS:
        held_out = dl.build_held_out_dataset(held_out_studies=[study])
        n_records = 0
        for record in held_out:
            n_records += 1
            stimuli = extract_stimuli(study, record["text"])
            # Left-to-right order matches centaur_eval._summed_nll_per_item's own
            # item_index assignment (contiguous non-masked label runs, in text
            # order) -- no manual truncation handling needed: an oversized
            # session's tail may get dropped by tokenizer truncation before
            # scoring, but the merge below is inner-join, so any stimulus with
            # no matching (experiment, participant_id, item_index) in the real
            # item_nlls.csv just silently doesn't contribute, exactly as it should.
            for item_index, word in enumerate(stimuli):
                stim_rows.append({
                    "experiment": study,
                    "participant_id": str(record["participant_id"]),
                    "item_index": item_index,
                    "stimulus": word,
                    "freq_count": freq.get(word.lower()),
                })
        print(f"  {study}: {n_records} held-out records, "
              f"{sum(1 for r in stim_rows if r['experiment'] == study)} stimuli extracted")

    stim_df = pd.DataFrame(stim_rows)
    matched_frac = stim_df["freq_count"].notna().mean()
    print(f"\n{len(stim_df)} stimulus rows total, {matched_frac:.1%} matched a frequency entry "
          f"(the rest are nonwords or real words outside the top 50k -- expected, not an error)")

    out_frames = []
    for study in STUDY_STIMULUS_PATTERNS:
        for model, run_label in [("base", f"base_{study}"), ("ours", f"loo_{study}")]:
            nll_path = RESULTS_DIR / f"{run_label}_item_nlls.csv"
            if not nll_path.is_file():
                print(f"WARNING: {nll_path} not found -- has this study's sweep eval finished? skipping")
                continue
            nll_df = pd.read_csv(nll_path)
            nll_df["participant_id"] = nll_df["participant_id"].astype(str)
            merged = stim_df[stim_df.experiment == study].merge(
                nll_df, on=["experiment", "participant_id", "item_index"], how="inner")
            merged = merged.dropna(subset=["freq_count"])
            if merged.empty:
                print(f"WARNING: {study}/{model}: zero rows survived the join -- check the "
                      f"stimulus-extraction regex against this study's real prompt text")
                continue
            merged["log_freq"] = merged["freq_count"].apply(lambda c: math.log10(c + 1))
            merged["freq_bin"] = pd.cut(merged["log_freq"], bins=10)
            summary = (merged.groupby("freq_bin", observed=True)
                       .agg(mean_log_freq=("log_freq", "mean"), mean_nll=("nll", "mean"),
                            n_items=("nll", "size"))
                       .reset_index())
            summary["study"] = study
            summary["model"] = model
            out_frames.append(summary)

    out_df = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "frequency_effect_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(out_df)} rows) -- small, paste-back-able")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
