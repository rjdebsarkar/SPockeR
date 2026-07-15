#!/bin/bash
# Run the scripts in this folder (single-PDB, argparse-driven versions of
# the SPOCKER pipeline) on one PDB. This is what
# demo_spocker/serve_pockets.py shells out to; see Script1-8 in this same
# folder for the actual algorithm.
#
# Unlike the old legacy/ pipeline (demo_spocker/run_pipeline_legacy.sh),
# these scripts already take --input_dir/--fields_dir/--analysis_dir/
# --pdb_file etc. directly on the command line, so there's no need for a
# BASE_DIR multi-PDB directory layout -- each stage is just pointed at the
# previous stage's output dir.
#
# Field generation is NOT done via Pipeline1_Fields_Generation.sh /
# Pipeline2_Fields_Generation.sh in this folder: those call
# `volgrids smiffer rna ...` + `volgrids vgtools unpack`/`convert`, which no
# longer exist in volgrids 1.0.0 (see demo_spocker/legacy/README.md -- the
# exact same break was already hit and worked around for legacy/, via
# demo_spocker/_legacy_prepare_inputs.py). _new_spocker_prepare_fields.py (in
# this folder) applies the same fix here, via the local _fields.py/
# _residues.py/_structure.py -- copies of demo_spocker/pipeline's modules of
# the same name, kept in this folder so everything needed to run a PDB
# through this pipeline lives right here, with no runtime dependency on
# demo_spocker/.
#
# Usage: bash run_pipeline_new_spocker.sh <input.pdb> <output_dir> [--keep-intermediate]
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input.pdb> <output_dir> [--keep-intermediate]"
    exit 1
fi

PDB_PATH=$(realpath "$1")
OUT_DIR=$(realpath -m "$2")
KEEP_INTERMEDIATE=0
[[ "${3:-}" == "--keep-intermediate" ]] && KEEP_INTERMEDIATE=1

# _new_spocker_prepare_fields.py and Script1/2/3/6/7/8 all derive the PDB
# identifier from the input filename's stem, so the caller's chosen filename
# IS the identifier used throughout. serve_pockets.py names this file
# "<job_id>.pdb" (a dot-free uuid4 hex), which is already safe here.
PDB_ID=$(basename "$PDB_PATH")
PDB_ID="${PDB_ID%.*}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NEW_SPOCKER_DIR="$SCRIPT_DIR"
WORK="$SCRIPT_DIR/testdata/new_spocker_work"

# Always start from a clean work dir: a previous --keep-intermediate run (or
# an interrupted one) can leave stale per-PDB field/analysis files behind
# that would otherwise get silently reused instead of regenerated.
rm -rf "$WORK"
mkdir -p "$WORK"

FIELDS1_DIR="$WORK/Fields_Pipeline1_${PDB_ID}"
FIELDS2_DIR="$WORK/Fields_Pipeline2_${PDB_ID}"
ANALYSIS1_DIR="$WORK/Analysis_Pipeline1_${PDB_ID}"
ANALYSIS2_DIR="$WORK/Analysis_Pipeline2_${PDB_ID}"

run_step() {
    echo ">>> Running $1"
    shift
    python3 "$@"
}

echo ">>> [1/3] Generating fields (APBS, stacking, hydrophobic; HBond fields if applicable) for $PDB_ID"
run_step Prepare "$SCRIPT_DIR/_new_spocker_prepare_fields.py" "$PDB_PATH" "$WORK" "$PDB_ID"
if ! compgen -G "$FIELDS1_DIR/*.mrc" > /dev/null; then
    echo "!!! Field generation produced no .mrc files in $FIELDS1_DIR"
    exit 1
fi

echo ">>> [2/3] Running the Pipeline 1 hotspot / pocket-volume stages"
run_step Script1 "$NEW_SPOCKER_DIR/Script1_Pipeline1_Slope_Derived_Fixed_Iso_Values_for_Hotspot.py" \
    --input_dir "$FIELDS1_DIR" --pdb_id "$PDB_ID" --output_dir "$ANALYSIS1_DIR"
run_step Script2 "$NEW_SPOCKER_DIR/Script2_Pipeline1_Detection_of_Binding_Site_Hotspots.py" \
    --fields_dir "$FIELDS1_DIR" --pdb_file "$PDB_PATH" --analysis_dir "$ANALYSIS1_DIR" --pdb_id "$PDB_ID"
run_step Script3 "$NEW_SPOCKER_DIR/Script3_Pipeline1_Making_Pocket_Volume_Using_Hotspots.py" \
    --pdb_file "$PDB_PATH" --analysis_dir "$ANALYSIS1_DIR" --pdb_id "$PDB_ID"

run_step Script6 "$NEW_SPOCKER_DIR/Script6_Trimming_APBS_for_Scoring_Unique_Pockets.py" \
    --fields_dir "$FIELDS1_DIR" --pdb_file "$PDB_PATH" --pdb_id "$PDB_ID"
run_step Script7 "$NEW_SPOCKER_DIR/Script7_Trimming_Hydrophobic_for_Scoring_Unique_Pockets.py" \
    --fields_dir "$FIELDS1_DIR" --pdb_id "$PDB_ID"

if [[ -f "$FIELDS2_DIR/${PDB_ID}.apbs.mrc" ]]; then
    echo ">>> [3/3] Running the Pipeline 2 H-bond pocket stages"
    run_step Script4 "$NEW_SPOCKER_DIR/Script4_Pipeline2_Hydrogen_Bond_Pocket_Hotspots_Using_HBA_HBD_ELE_Fields.py" \
        --fields_dir "$FIELDS2_DIR" --analysis_dir "$ANALYSIS2_DIR" --pdb_id "$PDB_ID"
    run_step Script5 "$NEW_SPOCKER_DIR/Script5_Pipeline2_Making_Hydrogen_Bond_Pocket_Volume.py" \
        --pdb_file "$PDB_PATH" --analysis_dir "$ANALYSIS2_DIR" --pdb_id "$PDB_ID"
else
    echo ">>> No non-canonical residues / HBond fields; skipping Script4/5"
fi

echo ">>> Merging pockets and scoring"
run_step Script8 "$NEW_SPOCKER_DIR/Script8_Making_Unique_Pockets_Using_All_Previous_Pockets.py" \
    --analysis1_dir "$ANALYSIS1_DIR" --analysis2_dir "$ANALYSIS2_DIR" --fields_dir "$FIELDS1_DIR" \
    --pdb_file "$PDB_PATH" --pdb_id "$PDB_ID" --output_dir "$OUT_DIR"

if [[ ! -f "$OUT_DIR/${PDB_ID}_field_contributions.csv" ]]; then
    echo "!!! Script8 produced no field-contributions CSV in $OUT_DIR"
    exit 1
fi

if [[ "$KEEP_INTERMEDIATE" -eq 0 ]]; then
    rm -rf "$WORK"
fi
