#!/usr/bin/env python3
"""Per-participant generalization delta (fine-tuned vs. base), for later
correlation against age -- in the spirit of Wulff's aging-semantic-network
work. Run ONCE on the cluster, against the study sweep's already-computed
item_nlls.csv files (no new training/eval, just aggregation).

Deliberately does NOT touch age or any other demographic data -- age is
participant-linked and lives only in this repo's local raw per-study data
files, and should never be pushed to the cluster or the public
cluster_bundle repo. This script outputs only (participant_id, delta_nll,
n_items) per study; the age join happens locally afterward.

STUDIES here is restricted to the 21 (of 26) active studies confirmed this
session to have genuinely populated numeric age data in their own raw data
file -- not just an `age` column documented in CODEBOOK.csv, which turned
out to not be sufficient (pexman2016_calgary documents an age column that is
actually empty in the real data). If a study's sweep hasn't finished
evaluating both base and ours, it's skipped with a warning, not silently
dropped.

Usage: python compute_participant_deltas.py
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Confirmed this session (real raw data files, not just CODEBOOK.csv) to have
# genuinely populated numeric age values. Excluded: Dymarska2025_associations,
# futrell2021_corpus, guenther2023ViSpa, keuleers2011_britishlexiconproject
# (no age-bearing raw file found), pexman2016_calgary (age column present but
# entirely empty in the real data).
AGE_ELIGIBLE_STUDIES = [
    "balota2007_LDT", "balota2007_naming", "brysbaert2014_Concreteness",
    "devardaetal2024_cloze", "devardaetal2024_rating", "frank2013_reading",
    "guenther2020LDT", "guenther2020TS", "guenther2022relational",
    "guenther2023associations_individual", "guenther2023grammaticality",
    "guenther2024comprehension", "guenther2024substitutions",
    "hilton2021_comprehension", "hutchison2013_semantic",
    "kyroelaeinen2022_valence", "lynott2020lancaster", "marson2026_eplep",
    "pissani2026_metaphor", "stella2026_formamentis_data",
    "tsaregorodtseva2026_mousetracking",
]


def main():
    out_frames = []
    for study in AGE_ELIGIBLE_STUDIES:
        base_path = RESULTS_DIR / f"base_{study}_item_nlls.csv"
        ours_path = RESULTS_DIR / f"loo_{study}_item_nlls.csv"
        if not base_path.is_file() or not ours_path.is_file():
            print(f"WARNING: missing item_nlls for {study} (base={base_path.is_file()}, "
                  f"ours={ours_path.is_file()}) -- skipping, has this study's sweep eval finished?")
            continue

        base_df = pd.read_csv(base_path)
        ours_df = pd.read_csv(ours_path)
        base_df["participant_id"] = base_df["participant_id"].astype(str)
        ours_df["participant_id"] = ours_df["participant_id"].astype(str)

        base_per_participant = base_df.groupby("participant_id")["nll"].agg(["mean", "size"])
        ours_per_participant = ours_df.groupby("participant_id")["nll"].agg(["mean", "size"])

        merged = base_per_participant.join(ours_per_participant, lsuffix="_base", rsuffix="_ours", how="inner")
        merged["delta_nll"] = merged["mean_base"] - merged["mean_ours"]
        merged["n_items"] = merged[["size_base", "size_ours"]].min(axis=1)
        merged = merged.reset_index()[["participant_id", "delta_nll", "n_items"]]
        merged["study"] = study
        out_frames.append(merged)
        print(f"  {study}: {len(merged)} participants with both base and ours evaluated")

    out_df = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "participant_deltas.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(out_df)} participant rows across "
          f"{out_df['study'].nunique() if not out_df.empty else 0} studies) -- small, paste-back-able")


if __name__ == "__main__":
    main()
