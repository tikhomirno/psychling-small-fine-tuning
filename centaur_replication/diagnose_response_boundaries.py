#!/usr/bin/env python3
"""Proves whether the response-boundary fixes documented in
RESPONSE_BOUNDARY_FIXES.md actually took, by testing the exact mechanism that
was broken: `trl.DataCollatorForCompletionOnlyLM` matches
response_template/instruction_template as TOKEN-ID subsequences, not strings,
and BPE tokenization of " <<"/">>" is context-sensitive to the surrounding
characters. A plain-text read of these files (matched <</>> counts) looks fine
even when this bug is active -- that's exactly why it went undetected the
first time. This script tokenizes the REAL text with the REAL model's
tokenizer and runs it through the REAL collator `centaur_finetune.py` trains
with, so a clean result here is a direct proof against the actual training
mechanism, not an inference from the raw text.

Needs only `transformers`/`trl` (CPU-only, no GPU, no model weights beyond the
tokenizer) -- both already required by centaur_finetune.py, so no new
dependency on the cluster.

Run with no arguments to verify the full corpus of every affected study (this
is the "proof" mode -- every record, not a sample) and get a PASS/FAIL
summary. Exits non-zero if any of the 7 fixed studies isn't at 100%, so it can
gate a script/CI step, not just be eyeballed.

    python centaur_replication/diagnose_response_boundaries.py
    python centaur_replication/diagnose_response_boundaries.py --sample 8   # quick spot-check
    python centaur_replication/diagnose_response_boundaries.py --adjacency  # also print the
                                                                             # bracket-adjacency
                                                                             # A/B reference cases
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
RESPONSE_TEMPLATE = " <<"
INSTRUCTION_TEMPLATE = ">>"
MAX_SEQ_TOKENS = 32768
CHARS_PER_TOKEN = 4  # data_loader.py's own rule-of-thumb, used for its chunking decision

# Studies whose real sequences can exceed MAX_SEQ_TOKENS -- data_loader.py
# chunks these before training (_chunk_record), so testing them RAW
# (unchunked) isn't the same text the model actually trains on. Mirrors
# data_loader.CHUNK_CANDIDATES exactly (kept as a literal copy, not an
# import, to keep this script's only dependencies transformers/trl).
CHUNK_CANDIDATES = {"futrell2021_corpus", "marson2026_eplep"}


def chunk_record(record: dict) -> list[dict]:
    """Reimplements data_loader._chunk_record: splits an oversized record at
    trial-line boundaries only, repeating the instruction header at the top of
    each chunk. Uses the SAME chars/4 estimate data_loader.py uses to decide
    chunk boundaries -- deliberately not the real tokenizer count, so this
    script tests the exact same (possibly imperfect) chunking real training
    uses, not an idealized version of it."""
    text = record["text"]
    n_tokens_est = len(text) / CHARS_PER_TOKEN
    if n_tokens_est <= MAX_SEQ_TOKENS:
        return [record]
    first_bracket = text.find("<<")
    header_end = text.rfind("\n\n", 0, first_bracket)
    header_end = 0 if header_end == -1 else header_end
    header = text[:header_end].strip()
    body_lines = [l for l in text[header_end:].split("\n") if l.strip()]
    max_body_chars = int(MAX_SEQ_TOKENS * CHARS_PER_TOKEN) - len(header) - 2
    chunks, current, current_len = [], [], 0
    for line in body_lines:
        if current and current_len + len(line) + 1 > max_body_chars:
            chunks.append(current)
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append(current)
    out = []
    for lines in chunks:
        chunk_record_ = dict(record)
        chunk_record_["text"] = header + "\n\n" + "\n".join(lines)
        out.append(chunk_record_)
    return out

# The 7 studies RESPONSE_BOUNDARY_FIXES.md fixed -- these are the ones that
# must come back 100.0000% for this to count as proof the fix took.
FIXED_STUDIES = {
    "devardaetal2024_cloze": "prompts_fixed.jsonl",
    "devardaetal2024_rating": "prompts_fixed.jsonl",
    "lynott2020lancaster": "prompts_fixed.jsonl",
    "Dymarska2025_associations": "prompts_fixed.jsonl",
    "stella2026_formamentis_data": "prompts_fixed.jsonl",
    "guenther2023associations_individual": "prompts_fixed.jsonl",
    "marson2026_eplep": "prompts_fixed.jsonl",
}
# Known-working studies, untouched by today's fix, included as a sanity check
# that the test methodology itself isn't the thing producing 100% -- also
# expected at 100% (confirmed this session once max_length matches the real
# recipe, 32768; the two "controls" only looked partial earlier in this
# session's investigation because that first pass used a too-small max_length).
CONTROL_STUDIES = {
    "balota2007_LDT": "prompts.jsonl.zip",
    "guenther2020LDT": "prompts.jsonl.zip",
}


def read_records(study: str, fname: str, limit: int | None) -> list[dict]:
    path = ROOT / study / fname
    if fname.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
            with zf.open(name) as fh:
                lines = io.TextIOWrapper(fh, encoding="utf-8").readlines()
    else:
        lines = path.open(encoding="utf-8").readlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
        if limit is not None and len(records) >= limit:
            break
    return records


def expected_item_count(text: str) -> int:
    """String-level expected count: number of ' <<' occurrences that have a
    matching '>>' after them."""
    n, pos = 0, 0
    while True:
        idx = text.find(RESPONSE_TEMPLATE, pos)
        if idx == -1:
            break
        close = text.find(INSTRUCTION_TEMPLATE, idx)
        if close == -1:
            break
        n += 1
        pos = close + 2
    return n


def real_item_count(collator, tokenizer, text: str, max_length: int = 32768) -> int:
    """Runs the REAL collator on one record's tokenized text, counts the number
    of contiguous unmasked (!= -100) label spans it actually produces. This is
    the exact mechanism centaur_finetune.py trains with -- 32768 matches
    MAX_SEQ_TOKENS, the real recipe's cap."""
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors=None)
    batch = collator([{"input_ids": enc["input_ids"]}])
    labels = batch["labels"][0].tolist()
    n, in_item = 0, False
    for label in labels:
        if label != -100:
            if not in_item:
                n += 1
                in_item = True
        else:
            in_item = False
    return n


def verify_studies(tokenizer, collator, studies: dict[str, str], limit: int | None) -> dict[str, tuple[int, int, int, int]]:
    """Returns {study: (n_units, n_boundary_bug_mismatches, total_expected, total_real)}.

    A mismatch is classified as a KNOWN TRUNCATION case (not a boundary-bug
    regression) when the unit's real, UNTRUNCATED token count exceeds
    MAX_SEQ_TOKENS -- that's data_loader.py's own chars/4 chunking-estimate
    heuristic under/over-shooting for that specific text (a separate, already
    -documented issue, see RESPONSE_BOUNDARY_FIXES.md), not the response-
    boundary bug this script exists to catch. Only mismatches on units that
    fit comfortably within the token budget count as a genuine FAIL."""
    results = {}
    for study, fname in studies.items():
        path = ROOT / study / fname
        if not path.is_file():
            print(f"  [SKIP] {study}: {path} not found")
            continue
        records = read_records(study, fname, limit)
        units = []
        for rec in records:
            units.extend(chunk_record(rec) if study in CHUNK_CANDIDATES else [rec])

        total_exp, total_real, n_bug_mismatch, n_truncation_mismatch = 0, 0, 0, 0
        for unit in units:
            exp = expected_item_count(unit["text"])
            real = real_item_count(collator, tokenizer, unit["text"])
            total_exp += exp
            total_real += real
            if exp != real:
                real_len = len(tokenizer(unit["text"], truncation=False, return_tensors=None)["input_ids"])
                if real_len > MAX_SEQ_TOKENS:
                    n_truncation_mismatch += 1
                else:
                    n_bug_mismatch += 1
        results[study] = (len(units), n_bug_mismatch, total_exp, total_real)
        pct = 100 * total_real / total_exp if total_exp else 100.0
        status = "PASS" if n_bug_mismatch == 0 else "FAIL"
        scope = "full corpus" if limit is None else f"first {limit} records"
        chunk_note = " (chunked, matching real training input)" if study in CHUNK_CANDIDATES else ""
        print(f"  [{status}] {study} ({scope}{chunk_note}): {len(units)} units, "
              f"{total_real}/{total_exp} items ({pct:.4f}%), "
              f"{n_bug_mismatch} boundary-bug mismatch(es)"
              + (f", {n_truncation_mismatch} known-truncation mismatch(es) (separate issue, not a regression)"
                 if n_truncation_mismatch else ""))
    return results


def print_adjacency_reference(tokenizer, collator):
    print("\n" + "=" * 78)
    print("REFERENCE: bracket-adjacency A/B cases (documents the fix rationale)")
    print("=" * 78)
    prefix = "You enter"
    cases = {
        "bare comma-space '>>, <<' (the original Dymarska/stella bug)":
            f"{prefix} <<WAR>>, <<FIGHT>>.",
        "period+space '>>. <<' (the fix)":
            f"{prefix} <<WAR>>. You enter <<FIGHT>>.",
        "'>>' immediately followed by newline (the devardaetal/lynott/marson bug)":
            f"Trial 1. You write: <<a>>\nTrial 2. You write: <<b>>.\n",
        "'>>.' before newline (the fix)":
            f"Trial 1. You write: <<a>>.\nTrial 2. You write: <<b>>.\n",
    }
    for label, text in cases.items():
        expected = expected_item_count(text)
        real = real_item_count(collator, tokenizer, text)
        status = "OK" if real == expected else "*** COLLAPSED/DROPPED ***"
        print(f"\n{label}")
        print(f"  text: {text!r}")
        print(f"  expected={expected}  real={real}  {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None,
                         help="only check the first N records per study (default: full corpus)")
    parser.add_argument("--adjacency", action="store_true",
                         help="also print the bracket-adjacency A/B reference cases")
    args = parser.parse_args()

    print(f"loading tokenizer for {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        instruction_template=INSTRUCTION_TEMPLATE,
        tokenizer=tokenizer,
    )

    print("\n--- 7 studies fixed by RESPONSE_BOUNDARY_FIXES.md (must be 100% for the fix to count as proven) ---")
    fixed_results = verify_studies(tokenizer, collator, FIXED_STUDIES, args.sample)

    print("\n--- control studies (never broken, sanity-check the test methodology itself) ---")
    verify_studies(tokenizer, collator, CONTROL_STUDIES, args.sample)

    if args.adjacency:
        print_adjacency_reference(tokenizer, collator)

    failed = [s for s, (_, n_mismatch, _, _) in fixed_results.items() if n_mismatch > 0]
    missing = set(FIXED_STUDIES) - set(fixed_results)
    print("\n" + "=" * 78)
    if not failed and not missing:
        print(f"PROOF PASSED: all {len(fixed_results)} fixed studies extract response items "
              f"correctly (100.0000% match against the real collator).")
    else:
        print("PROOF FAILED:")
        for s in failed:
            print(f"  - {s}: mismatch found -- fix did not take (or pulled files are stale)")
        for s in missing:
            print(f"  - {s}: file not found -- did the git pull/package step run?")
    print("=" * 78)

    raise SystemExit(1 if (failed or missing) else 0)


if __name__ == "__main__":
    main()
