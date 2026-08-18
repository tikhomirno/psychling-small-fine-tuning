#!/usr/bin/env python3
"""Materializes the corpus-balance fix as real, inspectable files.

Builds and saves both the per-study (100k trials/study) and per-paradigm (500k
trials/paradigm, equal per-study with shortfall redistribution) subsampled
corpora, plus a summary CSV showing the before/after imbalance fix. Run this
once (or whenever the underlying prompts_fixed.jsonl files change) --
data_loader.py's load_all_records/build_paradigm_balanced_records automatically
use these materialized files when present, instead of re-parsing+re-subsampling
raw JSONL on every call.

Safe to re-run any number of times -- fully deterministic given the fixed seed,
every output is overwritten, nothing here mutates the source prompts files.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl

SEED = 100


def _summary_row(unit_type: str, name: str, raw: list[dict], kept: list[dict]) -> dict:
    return {
        "unit_type": unit_type,
        "name": name,
        "original_trials": sum(dl._count_trials(r["text"]) for r in raw),
        "kept_trials": sum(dl._count_trials(r["text"]) for r in kept),
        "original_records": len(raw),
        "kept_records": len(kept),
    }


def main():
    dl.SUBSAMPLED_BY_STUDY_DIR.mkdir(parents=True, exist_ok=True)
    dl.SUBSAMPLED_BY_PARADIGM_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print(f"--- per-study subsampling (target: {dl.TARGET_TRIALS_PER_STUDY:,} trials/study) ---")
    chunked_by_study = {}
    for study in dl.STUDIES:
        raw = dl._load_chunked_study(study)
        chunked_by_study[study] = raw
        kept = dl._subsample_to_trial_count(raw, dl.TARGET_TRIALS_PER_STUDY, seed=SEED)
        dl._write_jsonl(dl.SUBSAMPLED_BY_STUDY_DIR / f"{study}.jsonl", kept)
        row = _summary_row("study", study, raw, kept)
        summary_rows.append(row)
        print(f"  {study}: {row['original_trials']:>10,} -> {row['kept_trials']:>9,} trials "
              f"({row['original_records']:,} -> {row['kept_records']:,} records)")

    print(f"\n--- per-paradigm subsampling (target: {dl.TARGET_TRIALS_PER_PARADIGM:,} trials/paradigm) ---")
    lookup = dl.paradigm_lookup()
    by_paradigm: dict[str, list[str]] = {}
    for study in dl.STUDIES:
        by_paradigm.setdefault(lookup[study], []).append(study)

    for paradigm, studies in sorted(by_paradigm.items()):
        records_by_study = {s: chunked_by_study[s] for s in studies}
        kept = dl._subsample_paradigm_equal(records_by_study, dl.TARGET_TRIALS_PER_PARADIGM, seed=SEED)
        slug = dl._paradigm_slug(paradigm)
        dl._write_jsonl(dl.SUBSAMPLED_BY_PARADIGM_DIR / f"{slug}.jsonl", kept)
        raw = [r for recs in records_by_study.values() for r in recs]
        row = _summary_row("paradigm", paradigm, raw, kept)
        summary_rows.append(row)
        print(f"  {paradigm} ({len(studies)} studies): {row['original_trials']:>10,} -> "
              f"{row['kept_trials']:>9,} trials ({row['original_records']:,} -> {row['kept_records']:,} records)")

    summary_path = dl.DATA_CACHE / "subsampled_corpus_summary.csv"
    dl.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["unit_type", "name", "original_trials", "kept_trials",
                                                 "original_records", "kept_records"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nsummary written to {summary_path}")
    print(f"per-study files: {dl.SUBSAMPLED_BY_STUDY_DIR} ({len(dl.STUDIES)} files)")
    print(f"per-paradigm files: {dl.SUBSAMPLED_BY_PARADIGM_DIR} ({len(by_paradigm)} files)")


if __name__ == "__main__":
    main()
