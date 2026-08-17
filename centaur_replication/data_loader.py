"""Data loading for the Centaur-replication leave-one-out fine-tuning runs.

Mirrors how Centaur's own generalization.py loads held-out data
(`load_dataset('json', data_files={'test': [task_name]})`) and how Psych-101 itself
is structured (one row per participant-session, columns `text`/`experiment`/
`participant`) -- see PROMPT_CLEANING_FIXES.md and FINETUNING_PLAN_CENTAUR_REPLICATION.md
for the full verification trail behind every design choice here.

No new data preprocessing happens here beyond what PROMPT_CLEANING_FIXES.md already
did (prompts_fixed.jsonl) -- this module only pools, filters, splits, and chunks
oversized sequences, matching Centaur's own "preprocessing already happened upstream"
pattern (Psych-101 was pre-built; finetune.py itself does zero .map()/preprocessing).
"""
import io
import json
import math
import zipfile
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = Path(__file__).resolve().parent / "data"

CHARS_PER_TOKEN = 4  # this repo's established rule-of-thumb (stats_plots.ipynb)
MAX_SEQ_TOKENS = 32768  # Centaur's literal cap, DataTrainingArguments.max_seq_length

# The 28 English, no-image studies established in DATA_OVERVIEW_LEAVE_ONE_OUT.md §5.
STUDIES = [
    "Dymarska2025_associations", "Leivada2020_manipulativeDiscourse", "balota2007_LDT",
    "balota2007_naming", "brysbaert2014_Concreteness", "devardaetal2024_cloze",
    "devardaetal2024_rating", "frank2013_reading", "futrell2021_corpus",
    "guenther2020LDT", "guenther2020TS", "guenther2022relational", "guenther2023ViSpa",
    "guenther2023associations_individual", "guenther2023grammaticality",
    "guenther2024comprehension", "guenther2024substitutions", "hilton2021_comprehension",
    "hutchison2013_semantic", "keuleers2011_britishlexiconproject",
    "kyroelaeinen2022_valence", "lynott2020lancaster", "marson2026_eplep",
    "pexman2016_calgary", "pissani2026_metaphor", "schiekiera2026_pwi_en",
    "stella2026_formamentis_data", "tsaregorodtseva2026_mousetracking",
]

# Studies whose real sequences can exceed MAX_SEQ_TOKENS after the session-collapsing
# fix in PROMPT_CLEANING_FIXES.md -- confirmed by direct measurement, not assumed.
# Every other study's max sequence is comfortably under the cap.
CHUNK_CANDIDATES = {"futrell2021_corpus", "marson2026_eplep"}


def _read_records(study: str) -> list[dict]:
    """Reads one study's records: prompts_fixed.jsonl if the fix exists, else the
    original prompts.jsonl -- each tried plain first, then zipped (the cluster
    bundle from package_for_cluster.sh re-zips prompts_fixed.jsonl too, since the
    unzipped file can be 100+MB -- see that script). Every record gets a plain
    `experiment` value equal to the study name, so downstream filtering/grouping is
    uniform even though the raw `experiment` field sometimes carries a sub-task
    suffix (e.g. hutchison2013_semantic's "hutchison2013_semantic/lexical_decision")."""
    study_dir = ROOT / study
    for plain_name, zip_name in (("prompts_fixed.jsonl", "prompts_fixed.jsonl.zip"),
                                  ("prompts.jsonl", "prompts.jsonl.zip")):
        path = study_dir / plain_name
        if path.is_file():
            with path.open(encoding="utf-8") as fh:
                records = [json.loads(l) for l in fh if l.strip()]
            break
        zpath = study_dir / zip_name
        if zpath.is_file():
            with zipfile.ZipFile(zpath) as zf:
                name = next((n for n in zf.namelist() if n.endswith(".jsonl")), None)
                if name is None:
                    continue
                with zf.open(name) as fh:
                    records = [json.loads(l) for l in io.TextIOWrapper(fh, encoding="utf-8")]
            break
    else:
        return []

    for r in records:
        r["experiment"] = study
        if "participant_id" not in r:
            # a few generators use "participant" instead -- normalize (see
            # PROMPT_CLEANING_FIXES.md; already fixed at the source for 3 studies,
            # this is a defensive fallback for anything not yet re-run)
            r["participant_id"] = r.get("participant", r.get("participant_id"))
    return records


def _chunk_record(record: dict) -> list[dict]:
    """Splits one oversized record into as few sequential chunks as needed, only at
    trial-line boundaries (never mid-trial). The instruction header (everything
    before the first line containing '<<') is repeated at the top of every chunk so
    each chunk is independently well-formed."""
    text = record["text"]
    n_tokens = len(text) / CHARS_PER_TOKEN
    if n_tokens <= MAX_SEQ_TOKENS:
        return [record]

    first_bracket = text.find("<<")
    header_end = text.rfind("\n\n", 0, first_bracket)
    header_end = 0 if header_end == -1 else header_end
    header = text[:header_end].strip()
    body_lines = [l for l in text[header_end:].split("\n") if l.strip()]

    max_body_chars = int(MAX_SEQ_TOKENS * CHARS_PER_TOKEN) - len(header) - 2
    chunks, current = [], []
    current_len = 0
    for line in body_lines:
        if current and current_len + len(line) + 1 > max_body_chars:
            chunks.append(current)
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append(current)

    out = []
    for i, lines in enumerate(chunks):
        chunk_record = dict(record)
        chunk_record["text"] = header + "\n\n" + "\n".join(lines)
        chunk_record["participant_id"] = f"{record['participant_id']}_chunk{i}"
        chunk_record["_source_participant_id"] = record["participant_id"]
        out.append(chunk_record)
    return out


def load_all_records(studies: list[str] | None = None) -> list[dict]:
    """Loads every in-scope study's records, applying trial-boundary chunking to the
    two studies confirmed to need it (FINETUNING_PLAN_CENTAUR_REPLICATION.md §4)."""
    studies = studies or STUDIES
    all_records = []
    for study in studies:
        records = _read_records(study)
        if study in CHUNK_CANDIDATES:
            chunked = []
            for r in records:
                chunked.extend(_chunk_record(r))
            records = chunked
        all_records.extend(records)
    return all_records


def paradigm_lookup() -> dict[str, str]:
    """study -> paradigm, from experiment_metadata.csv (curated task labels),
    restricted to the folders in STUDIES."""
    meta = pd.read_csv(ROOT / "experiment_metadata.csv")
    task_per_folder = meta.groupby("experiment")["task"].first()
    return {s: task_per_folder.get(s, "unknown") for s in STUDIES}


def studies_in_paradigm(paradigm: str) -> list[str]:
    lookup = paradigm_lookup()
    return [s for s, p in lookup.items() if p == paradigm]


def leakage_safe_split(records: list[dict], test_frac: float = 0.10, seed: int = 100):
    """Centaur's own 90/10 train/test split, per experiment -- but grouped by REAL
    participant first (see FINETUNING_PLAN_CENTAUR_REPLICATION.md, "multi-record
    participants" section). A participant with multiple records (balota's
    session x batch structure, or a chunked oversized sequence) never has some
    records in train and others in test -- the split happens on unique
    (experiment, participant_id) groups, and every record for a chosen participant
    follows their group.

    Note: chunked records already carry a `_source_participant_id` distinct from
    their (per-chunk) `participant_id`, so grouping uses the real source identity.
    """
    import random

    rng = random.Random(seed)
    by_experiment: dict[str, list[dict]] = {}
    for r in records:
        by_experiment.setdefault(r["experiment"], []).append(r)

    train, test = [], []
    for exp, exp_records in by_experiment.items():
        groups: dict[str, list[dict]] = {}
        for r in exp_records:
            key = r.get("_source_participant_id", r["participant_id"])
            groups.setdefault(key, []).append(r)
        participant_keys = sorted(groups.keys())
        rng.shuffle(participant_keys)
        n_test = max(1, round(len(participant_keys) * test_frac)) if len(participant_keys) > 1 else 0
        test_keys = set(participant_keys[:n_test])
        for key, recs in groups.items():
            (test if key in test_keys else train).extend(recs)
    return train, test


def build_training_dataset(held_out_studies: list[str] | None = None,
                            held_out_paradigm: str | None = None,
                            inner_split: bool = True, seed: int = 100):
    """Builds the pooled training pool for one leave-one-out run.

    Returns (train_dataset, inner_eval_dataset) -- the inner_eval_dataset is
    Centaur's own 90/10 in-training holdout (functionally inert at
    --eval_steps 999999, but included for parity; see the plan's "where the 90/10
    split comes from" note).
    """
    held_out = set(held_out_studies or [])
    if held_out_paradigm:
        held_out |= set(studies_in_paradigm(held_out_paradigm))

    pool_studies = [s for s in STUDIES if s not in held_out]
    records = load_all_records(pool_studies)

    if inner_split:
        train_records, test_records = leakage_safe_split(records, seed=seed)
    else:
        train_records, test_records = records, []

    def _row(r):
        return {"text": r.get("text"), "experiment": r.get("experiment"),
                "participant_id": str(r.get("participant_id"))}

    train_ds = Dataset.from_list([_row(r) for r in train_records]).shuffle(seed=seed)
    eval_ds = Dataset.from_list([_row(r) for r in test_records]) if test_records else None
    return train_ds, eval_ds


def _ensure_plain_jsonl(study: str) -> str:
    """Returns a real, on-disk plain-JSONL path for a study -- prompts_fixed.jsonl or
    prompts.jsonl if either exists directly (plain); otherwise extracts whichever
    zip variant exists (prompts_fixed.jsonl.zip or prompts.jsonl.zip -- the cluster
    bundle from package_for_cluster.sh always zips) into DATA_CACHE once (cached,
    not re-extracted every call), since load_dataset('json', ...) needs a real file
    path, not a zip member."""
    study_dir = ROOT / study
    for plain_name, zip_name in (("prompts_fixed.jsonl", "prompts_fixed.jsonl.zip"),
                                  ("prompts.jsonl", "prompts.jsonl.zip")):
        plain = study_dir / plain_name
        if plain.is_file():
            return str(plain)
        zpath = study_dir / zip_name
        if zpath.is_file():
            cached = DATA_CACHE / f"{study}.{plain_name}"
            if cached.is_file():
                return str(cached)
            DATA_CACHE.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
                with zf.open(name) as src, cached.open("wb") as dst:
                    dst.write(src.read())
            return str(cached)
    raise FileNotFoundError(f"no prompts file found for {study}")


def build_held_out_dataset(held_out_studies: list[str] | None = None,
                            held_out_paradigm: str | None = None):
    """Loads the held-out generalization test set, mirroring generalization.py's own
    `load_dataset('json', data_files={'test': [task_name]})` pattern as literally as
    possible -- real file paths, not records re-serialized from Python."""
    held_out = list(held_out_studies or [])
    if held_out_paradigm:
        held_out = studies_in_paradigm(held_out_paradigm)

    # Note: unlike our normalized prompts_fixed.jsonl, an unfixed study's original
    # `experiment` field may carry a raw sub-task-suffixed value (e.g.
    # "balota2007_LDT_exp1") rather than the plain study name. Harmless here --
    # callers identify the held-out set by `held_out` itself, not by parsing this
    # column -- but worth knowing if inspecting this dataset's `experiment` column
    # directly.
    paths = [_ensure_plain_jsonl(study) for study in held_out]
    return load_dataset("json", data_files={"test": paths})["test"]
