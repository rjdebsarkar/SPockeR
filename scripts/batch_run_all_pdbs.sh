#!/bin/bash
# Batch-runs the SPockeR pipeline (volgrids 1.0.0 native workflow) over
# every .pdb file in a given directory.
#
# For each <id>.pdb found in <pdb_directory>:
#   1. Runs 0_fix_pdb.sh to produce a cleaned/fixed structure, saved to
#      ../data/fixed_pdbs/<id>_fixed.pdb
#   2. Runs run_pipeline_new_spocker.sh on the fixed structure, saving
#      final ranked pockets to ../Analysis_Unique_Pockets_<id>/
#
# Any PDB that fails at either step is logged (with the reason) to
# ../failed_pdbs.txt and the batch continues with the next PDB -- a single
# problematic structure will not abort the whole run.
#
# Usage:
#   ./batch_run_all_pdbs.sh <pdb_directory>
#
# Example:
#   ./batch_run_all_pdbs.sh /path/to/pdb_directory
set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <pdb_directory>"
    exit 1
fi

PDB_DIR=$(realpath "$1")
if [[ ! -d "$PDB_DIR" ]]; then
    echo "!!! Not a directory: $PDB_DIR"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

FIXED_DIR="../data/fixed_pdbs"
mkdir -p "$FIXED_DIR"

FAILED_LOG="../failed_pdbs.txt"
> "$FAILED_LOG"

TOTAL=0
SUCCEEDED=0
FAILED=0

shopt -s nullglob
PDB_FILES=("$PDB_DIR"/*.pdb)
shopt -u nullglob

if [[ ${#PDB_FILES[@]} -eq 0 ]]; then
    echo "!!! No .pdb files found in $PDB_DIR"
    exit 1
fi

echo ">>> Found ${#PDB_FILES[@]} PDB file(s) in $PDB_DIR"

for pdb in "${PDB_FILES[@]}"; do
    id=$(basename "$pdb" .pdb)
    fixed="$FIXED_DIR/${id}_fixed.pdb"
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "=================================================================="
    echo ">>> [$TOTAL/${#PDB_FILES[@]}] Processing $id"
    echo "=================================================================="

    if ! ./0_fix_pdb.sh "$pdb" "$fixed"; then
        echo "!!! $id: 0_fix_pdb.sh failed"
        echo "$id  (0_fix_pdb.sh failed)" >> "$FAILED_LOG"
        FAILED=$((FAILED + 1))
        continue
    fi

    if ! bash run_pipeline_new_spocker.sh "$fixed" "../Analysis_Unique_Pockets_${id}"; then
        echo "!!! $id: run_pipeline_new_spocker.sh failed"
        echo "$id  (run_pipeline_new_spocker.sh failed)" >> "$FAILED_LOG"
        FAILED=$((FAILED + 1))
        continue
    fi

    SUCCEEDED=$((SUCCEEDED + 1))
done

echo ""
echo "=================================================================="
echo ">>> Batch complete: $SUCCEEDED/$TOTAL succeeded, $FAILED failed."
if [[ "$FAILED" -gt 0 ]]; then
    echo ">>> Failures logged in $FAILED_LOG:"
    cat "$FAILED_LOG"
else
    echo ">>> No failures."
fi
echo "=================================================================="
