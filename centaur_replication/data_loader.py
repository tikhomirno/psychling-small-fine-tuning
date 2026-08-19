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
import hashlib
import io
import json
import math
import os
import zipfile
from pathlib import Path

import pandas as pd
from datasets import Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = Path(__file__).resolve().parent / "data"

CHARS_PER_TOKEN = 4  # this repo's established rule-of-thumb (stats_plots.ipynb)
MAX_SEQ_TOKENS = 32768  # Centaur's literal cap, DataTrainingArguments.max_seq_length

# The 27 English, no-image studies established in DATA_OVERVIEW_LEAVE_ONE_OUT.md §5,
# minus Leivada2020_manipulativeDiscourse -- removed this session after a full
# non-Latin-script scan confirmed it's the only study with substantial non-English
# (Greek) stimuli text, which was also causing a real collator/tokenizer-boundary
# crash during evaluation (KeyError: 'nll' from zero extracted response items).
STUDIES = [
    "Dymarska2025_associations", "balota2007_LDT",
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


def _cap_participants(records: list[dict], max_participants_per_study: int,
                       seed: int = 100) -> list[dict]:
    """Keeps at most `max_participants_per_study` real participants per study --
    for a cluster pre-flight smoke test (cluster_smoke_test.py), never for a real
    run. Groups by (experiment, _source_participant_id or participant_id), the same
    key leakage_safe_split already groups by, so a chunked CHUNK_CANDIDATES study
    never loses only some of one participant's chunks -- "1 participant" always
    means one complete real participant, including every chunk of an oversized
    session. Deterministic given `seed` (sorted keys, then shuffled, then take the
    first N) so repeated smoke-test runs sample the same participants."""
    import random

    rng = random.Random(seed)
    by_experiment: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        key = r.get("_source_participant_id", r["participant_id"])
        by_experiment.setdefault(r["experiment"], {}).setdefault(key, []).append(r)

    out = []
    for exp, groups in by_experiment.items():
        keys = sorted(groups.keys())
        rng.shuffle(keys)
        for key in keys[:max_participants_per_study]:
            out.extend(groups[key])
    return out


def _count_trials(text: str) -> int:
    return text.count("<<")


def _subsample_to_trial_count(records: list[dict], target_trials: int,
                               seed: int = 100) -> list[dict]:
    """Randomly (seeded) subsamples records down to ~target_trials worth of
    <<...>> targets -- per your explicit instruction, each record (one JSON line,
    including individual chunks of an oversized session) is treated as its own
    independent unit here, NOT grouped back to the real source participant the
    way leakage_safe_split/_cap_participants do elsewhere in this file. This is a
    corpus-volume concern, not a leakage-safety one, so a study with pre-existing
    multi-row-per-participant "batch" structure (balota2007_LDT,
    balota2007_naming) gets each row sampled independently.

    Returns every record unchanged if the true total is already <= target_trials
    (never pads up to reach the target). Deterministic given `seed`."""
    import random

    total = sum(_count_trials(r["text"]) for r in records)
    if total <= target_trials:
        return records
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    out, cum = [], 0
    for i in order:
        out.append(records[i])
        cum += _count_trials(records[i]["text"])
        if cum >= target_trials:
            break
    return out


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# Bump this manually if the *shape* of the subsampling logic changes (e.g. the
# redistribution algorithm) without either TARGET_TRIALS_* constant changing --
# included in build_training_dataset's cache key so a stale on-disk cache from
# an older subsampling policy can never be silently reused for a new one.
SUBSAMPLING_VERSION = 1

TARGET_TRIALS_PER_STUDY = 100_000
SUBSAMPLED_BY_STUDY_DIR = DATA_CACHE / "subsampled_by_study"


def load_all_records(studies: list[str] | None = None,
                      max_participants_per_study: int | None = None,
                      seed: int = 100,
                      target_trials_per_study: int | None = TARGET_TRIALS_PER_STUDY) -> list[dict]:
    """Loads every in-scope study's records, applying trial-boundary chunking to the
    two studies confirmed to need it (FINETUNING_PLAN_CENTAUR_REPLICATION.md §4),
    then subsamples each study down to `target_trials_per_study` individual
    <<...>> targets (the new default corpus-balance fix -- was previously
    unbounded; real per-study counts range from 2,920 to 10.6M trials).

    If `build_subsampled_corpus.py` has already been run, the DEFAULT
    (target_trials_per_study == TARGET_TRIALS_PER_STUDY) path loads directly from
    its materialized `subsampled_by_study/<study>.jsonl` output instead of
    re-parsing+re-subsampling raw JSONL every call -- guarantees the training run
    uses exactly the same subsample you can inspect on disk. Falls back to live
    computation if that file doesn't exist yet (e.g. a fresh clone), or whenever a
    caller passes a different target_trials_per_study explicitly (e.g. None, used
    internally by build_paradigm_balanced_records to get the raw, unsubsampled
    pool before applying its own per-paradigm rule).

    `max_participants_per_study` is None for every real run (unrelated to the
    trial-count subsampling above) -- pass an int only for a tiny cluster smoke
    test. Applied AFTER trial subsampling: chunking is what produces
    `_source_participant_id`, so participant-capping before it would have nothing
    correct to group on."""
    studies = studies or STUDIES
    all_records = []
    for study in studies:
        cached_path = SUBSAMPLED_BY_STUDY_DIR / f"{study}.jsonl"
        if target_trials_per_study == TARGET_TRIALS_PER_STUDY and cached_path.is_file():
            records = _read_jsonl(cached_path)
        else:
            records = _read_records(study)
            if study in CHUNK_CANDIDATES:
                chunked = []
                for r in records:
                    chunked.extend(_chunk_record(r))
                records = chunked
            if target_trials_per_study is not None:
                records = _subsample_to_trial_count(records, target_trials_per_study, seed=seed)
        all_records.extend(records)
    if max_participants_per_study is not None:
        all_records = _cap_participants(all_records, max_participants_per_study, seed=seed)
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


TARGET_TRIALS_PER_PARADIGM = 500_000
SUBSAMPLED_BY_PARADIGM_DIR = DATA_CACHE / "subsampled_by_paradigm"


def _paradigm_slug(paradigm: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", paradigm)


def _subsample_paradigm_equal(records_by_study: dict[str, list[dict]],
                               target_trials: int, seed: int = 100) -> list[dict]:
    """Equal-per-study subsampling within one paradigm, with shortfall
    redistribution: sorts studies ascending by available trial count, assigns
    each study min(its own total, the current equal share of the still-unassigned
    target across still-unassigned studies) -- so a study too small to meet its
    equal share contributes everything it has, and the leftover target gets
    spread across the remaining (larger) studies instead of going unused.

    If the paradigm's true total is already <= target_trials, returns everything
    unchanged (no subsampling at all) -- matches the per-study rule's "never pad
    up" behavior."""
    totals = {s: sum(_count_trials(r["text"]) for r in recs)
              for s, recs in records_by_study.items()}
    if sum(totals.values()) <= target_trials:
        return [r for recs in records_by_study.values() for r in recs]

    remaining_studies = sorted(totals, key=lambda s: totals[s])
    remaining_target = target_trials
    allocation = {}
    for i, s in enumerate(remaining_studies):
        share = remaining_target / (len(remaining_studies) - i)
        take = min(totals[s], share)
        allocation[s] = take
        remaining_target -= take

    out = []
    for s, recs in records_by_study.items():
        out.extend(_subsample_to_trial_count(recs, int(allocation[s]), seed=seed))
    return out


def _load_chunked_study(study: str) -> list[dict]:
    """Raw records for one study, chunked but NOT trial-subsampled -- the input
    _subsample_paradigm_equal/build_subsampled_corpus.py need, distinct from
    load_all_records's default (which trial-subsamples per study first)."""
    records = _read_records(study)
    if study in CHUNK_CANDIDATES:
        records = [c for r in records for c in _chunk_record(r)]
    return records


def build_paradigm_balanced_records(studies: list[str], seed: int = 100,
                                     target_trials_per_paradigm: int = TARGET_TRIALS_PER_PARADIGM
                                     ) -> list[dict]:
    """Builds the training pool for leave-one-paradigm-out: pools `studies`
    (already excluding the held-out paradigm's own studies), groups by paradigm,
    applies _subsample_paradigm_equal to each paradigm group independently, then
    concatenates. This is a separate, mutually exclusive pooling strategy from
    load_all_records's default per-study 100k cap -- selected by which
    leave-one-out mode is active in build_training_dataset.

    If build_subsampled_corpus.py has already been run, loads directly from its
    materialized subsampled_by_paradigm/<slug>.jsonl output per paradigm instead
    of re-deriving live -- falls back to live computation otherwise."""
    lookup = paradigm_lookup()
    by_paradigm: dict[str, list[str]] = {}
    for s in studies:
        by_paradigm.setdefault(lookup.get(s, "unknown"), []).append(s)

    out = []
    for paradigm, paradigm_studies in by_paradigm.items():
        cached_path = SUBSAMPLED_BY_PARADIGM_DIR / f"{_paradigm_slug(paradigm)}.jsonl"
        if cached_path.is_file():
            # cached_path holds the FULL paradigm's balanced subsample (built from
            # every study in that paradigm); when this paradigm's studies are only
            # a subset of `studies` (shouldn't normally happen -- a paradigm's
            # studies are excluded as a whole -- but guard against it anyway),
            # filter down to just the requested studies.
            cached = _read_jsonl(cached_path)
            out.extend(r for r in cached if r["experiment"] in paradigm_studies)
        else:
            records_by_study = {s: _load_chunked_study(s) for s in paradigm_studies}
            out.extend(_subsample_paradigm_equal(records_by_study, target_trials_per_paradigm, seed=seed))
    return out


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


def _dataset_cache_key(*parts) -> str:
    """Deterministic short key from run-shape parameters (held-out selection,
    seed, cap, etc.) -- used to name a persistent on-disk cache dir under
    DATA_CACHE, so an identical call (same params) loads instantly instead of
    re-pooling/re-parsing raw JSONL from scratch. Different parameter
    combinations get different keys, so nothing stale gets reused by accident."""
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _load_cached_dataset(cache_dir: Path) -> "Dataset | None":
    return Dataset.load_from_disk(str(cache_dir)) if cache_dir.is_dir() else None


def _save_cached_dataset(ds: "Dataset", cache_dir: Path):
    """Builds to a temp dir then atomically renames into place -- if two processes
    race to build the same (uncached) dataset simultaneously (e.g. two DDP ranks
    that reach this call before either's cache write completes), the loser's
    rename just overwrites the winner's with an equivalent dataset rather than
    corrupting a partially-written cache dir."""
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    tmp_dir = DATA_CACHE / f".tmp_{cache_dir.name}_{os.getpid()}"
    ds.save_to_disk(str(tmp_dir))
    os.replace(tmp_dir, cache_dir)


def build_training_dataset(held_out_studies: list[str] | None = None,
                            held_out_paradigm: str | None = None,
                            inner_split: bool = True, seed: int = 100,
                            max_participants_per_study: int | None = None,
                            disable_subsampling: bool = False):
    """Builds the pooled training pool for one leave-one-out run.

    Returns (train_dataset, inner_eval_dataset) -- the inner_eval_dataset is
    Centaur's own 90/10 in-training holdout (functionally inert at
    --eval_steps 999999, but included for parity; see the plan's "where the 90/10
    split comes from" note).

    `max_participants_per_study` is None for every real run -- pass an int (e.g. 1)
    only to build a tiny cluster smoke-test pool (see cluster_smoke_test.py). Note:
    with a 1-participant cap, leakage_safe_split's own `len(participant_keys) > 1`
    guard means every study contributes 0 records to the inner eval split, so
    inner_eval_dataset comes back None -- harmless (real runs already treat this
    split as functionally inert at --eval_steps 999999), not a bug.

    `disable_subsampling` is False for every real run (the 100k-trial-per-study /
    500k-trial-per-paradigm caps apply as normal, the intended default) -- pass
    True only to deliberately reproduce the OLD, pre-subsampling, fully unbounded
    pool, e.g. for a direct full-data-vs-subsampled comparison run. Threaded into
    the cache key (below) so a disabled-subsampling run and a normal capped run
    for the same held-out study never collide in the on-disk dataset cache.

    Cached to disk under DATA_CACHE, keyed by every parameter that affects the
    result -- pooling/parsing the full ~41k-sequence corpus is expensive (raw
    JSONL parsing + on-the-fly zip extraction for ~27 studies), and under a
    multi-GPU torchrun launch every DDP rank calls this independently; wrap this
    call in accelerate's `PartialState().main_process_first()` (see
    centaur_finetune.py) so only rank 0 pays that cost and the other ranks hit
    this cache instead of redoing it in parallel.
    """
    # TARGET_TRIALS_PER_STUDY/TARGET_TRIALS_PER_PARADIGM are included so a cache
    # built under a different subsampling policy (e.g. before this trial-count
    # cap existed at all, or after tuning the target) can never be silently
    # reused -- confirmed this session: a stale pre-subsampling cache for
    # hilton2021_comprehension got reused after the cap was added, since the key
    # didn't depend on it, producing a real training run on the wrong
    # (unsubsampled, 41,061-sequence) pool. Bump SUBSAMPLING_VERSION manually if
    # the *shape* of the subsampling logic itself ever changes without changing
    # either constant (e.g. redistribution algorithm tweaks). STUDIES itself is
    # also included (sorted, so order doesn't matter) so that removing/adding a
    # study (e.g. Leivada2020_manipulativeDiscourse, dropped this session for
    # non-English content) automatically invalidates any cache built under the
    # old study list, instead of repeating the exact same class of staleness bug
    # a second time.
    key = _dataset_cache_key("train", sorted(held_out_studies or []), held_out_paradigm,
                              inner_split, seed, max_participants_per_study,
                              TARGET_TRIALS_PER_STUDY, TARGET_TRIALS_PER_PARADIGM,
                              SUBSAMPLING_VERSION, disable_subsampling, sorted(STUDIES))
    train_cache = DATA_CACHE / f"train_{key}"
    eval_cache = DATA_CACHE / f"eval_{key}"
    cached_train = _load_cached_dataset(train_cache)
    if cached_train is not None:
        return cached_train, _load_cached_dataset(eval_cache)

    if held_out_paradigm:
        held_out = set(studies_in_paradigm(held_out_paradigm))
        pool_studies = [s for s in STUDIES if s not in held_out]
        if disable_subsampling:
            # No per-paradigm-balanced equivalent of "disabled" -- fall back to
            # plain unbounded per-study loading (still excludes the held-out
            # paradigm's studies, just skips both the 100k/study and 500k/paradigm
            # caps entirely).
            records = load_all_records(pool_studies, seed=seed, target_trials_per_study=None)
        else:
            # Leave-one-PARADIGM-out: pool capped per-paradigm (500k trials,
            # equal per-study within each remaining paradigm), NOT the per-study
            # 100k cap -- a separate, mutually exclusive pooling strategy from
            # the study-out case.
            records = build_paradigm_balanced_records(pool_studies, seed=seed)
    else:
        held_out = set(held_out_studies or [])
        pool_studies = [s for s in STUDIES if s not in held_out]
        # Leave-one-STUDY-out (and the no-held-out full-pool reference run):
        # per-study 100k-trial cap applies by default (load_all_records's own
        # default), plus max_participants_per_study for the smoke test if set --
        # unless disable_subsampling explicitly asks for the old, fully unbounded
        # pool instead (target_trials_per_study=None turns the cap off entirely).
        records = load_all_records(
            pool_studies, max_participants_per_study=max_participants_per_study, seed=seed,
            target_trials_per_study=None if disable_subsampling else TARGET_TRIALS_PER_STUDY)

    if inner_split:
        train_records, test_records = leakage_safe_split(records, seed=seed)
    else:
        train_records, test_records = records, []

    def _row(r):
        return {"text": r.get("text"), "experiment": r.get("experiment"),
                "participant_id": str(r.get("participant_id"))}

    train_ds = Dataset.from_list([_row(r) for r in train_records]).shuffle(seed=seed)
    eval_ds = Dataset.from_list([_row(r) for r in test_records]) if test_records else None

    _save_cached_dataset(train_ds, train_cache)
    if eval_ds is not None:
        _save_cached_dataset(eval_ds, eval_cache)
    return train_ds, eval_ds


def _ensure_plain_jsonl(study: str) -> str:
    """Returns a real, on-disk plain-JSONL path for a study -- prompts_fixed.jsonl or
    prompts.jsonl if either exists directly (plain); otherwise extracts whichever
    zip variant exists (prompts_fixed.jsonl.zip or prompts.jsonl.zip -- the cluster
    bundle from package_for_cluster.sh always zips) into DATA_CACHE once (cached,
    not re-extracted every call), since build_held_out_dataset needs a real file
    path to read line-by-line, not a zip member."""
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
                            held_out_paradigm: str | None = None,
                            max_participants_per_study: int | None = None):
    """Loads the held-out generalization test set from real, on-disk file paths
    (via _ensure_plain_jsonl) -- not records re-serialized from Python -- keeping
    the spirit of generalization.py's real-file-based loading.

    Reads each study's raw JSONL with plain Python `json.loads` (not
    `datasets.load_dataset('json', ...)`, despite that being generalization.py's
    own literal call): real per-study raw files carry many extra columns beyond
    text/experiment/participant_id (rt, age, gender, batch_no, etc.), and two
    separate real failures confirmed this session that pyarrow's strict JSON
    schema inference chokes on them -- (1) a single study's own file with a
    column whose type varies row-to-row (balota2007_LDT's `gender`: string in
    some rows, numeric in others), and (2) incompatible schemas across different
    studies' files when held_out_paradigm spans more than one study. Plain
    Python dict parsing doesn't care about cross-row/cross-file type
    consistency in columns we don't even use, so extracting only the 3 needed
    columns this way sidesteps both failure modes at the root rather than
    patching each one as it's discovered.

    `max_participants_per_study` is None for every real run -- pass an int (e.g. 1)
    only for a tiny cluster smoke test (see cluster_smoke_test.py). Applied as a
    post-hoc filter (first N sorted unique participant_id values per study),
    deliberately NOT by rebuilding this function on top of
    load_all_records/_cap_participants -- that would break the real-file-path
    parity with generalization.py this function is built to preserve.
    """
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
    rows = []
    for path in paths:
        for r in _read_jsonl(Path(path)):
            rows.append({"text": r["text"], "experiment": r["experiment"],
                         "participant_id": str(r.get("participant_id", r.get("participant")))})
    ds = Dataset.from_list(rows)

    if max_participants_per_study is not None:
        by_experiment: dict[str, set[str]] = {}
        for exp, pid in zip(ds["experiment"], ds["participant_id"]):
            by_experiment.setdefault(exp, set()).add(str(pid))
        keep = {exp: set(sorted(pids)[:max_participants_per_study])
                for exp, pids in by_experiment.items()}
        ds = ds.filter(lambda r: str(r["participant_id"]) in keep.get(r["experiment"], set()))
    return ds
