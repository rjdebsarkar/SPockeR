#!/bin/bash
# =============================================================================
# Pipeline2_Fields_Generation.sh
# Generates H-bond fields (HBA, HBD, ELE) for non-canonical residues of a
# single PDB. Uses indices_rna_hb_available_modified.py to detect the
# non-canonical residue indices.
#
# Usage: ./Pipeline2_Fields_Generation.sh <path_to_fixed_pdb>
# Example: ./Pipeline2_Fields_Generation.sh 1AJU_fixed.pdb
#
# Output: Creates ./Fields_Pipeline2_<PDB_ID>/ in the current working
#         directory, containing the H-bond field .mrc grids and the
#         RNApolis annotation CSV.
#
# Requirement: indices_rna_hb_available_modified.py must be in the same
#              directory as this script (or set PY_SCRIPT below manually).
# =============================================================================
set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path_to_fixed_pdb>"
    echo "Example: $0 1AJU_fixed.pdb"
    exit 1
fi

INPUT_PDB="$1"

if [[ ! -f "$INPUT_PDB" ]]; then
    echo "ERROR: Input PDB not found: $INPUT_PDB"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/indices_rna_hb_available_modified.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "ERROR: Required Python script not found: $PY_SCRIPT"
    echo "indices_rna_hb_available_modified.py must be in the same directory as this script."
    exit 1
fi

PDB_DIR="$(cd "$(dirname "$INPUT_PDB")" && pwd)"
FILE_NAME="$(basename "$INPUT_PDB")"
NAME="${FILE_NAME%.*}"

RUN_DIR="$(pwd)"
OUT_DIR="$RUN_DIR/Fields_Pipeline2_${NAME}"
FAILED_LOG="$RUN_DIR/failed_pdbs_pipeline2.txt"

mkdir -p "$OUT_DIR"

echo "=================================================="
echo "Pipeline 2 — Processing $NAME"
echo "Input structure : $PDB_DIR/$FILE_NAME"
echo "Output folder   : $OUT_DIR"
echo "=================================================="

(
    set -e

    # ------------------------------------------------------------------
    # Step 0: Get non-canonical residue indices via Python script
    # ------------------------------------------------------------------
    echo "    [0/3] Detecting non-canonical residue indices for $NAME"
    csv_out="$OUT_DIR/${NAME}_annotation.csv"
    residues_nobp=$(python "$PY_SCRIPT" "$PDB_DIR/$FILE_NAME" "$csv_out")

    if [[ -z "$residues_nobp" ]]; then
        echo "    [WARNING] No non-canonical indices found for $NAME — skipping volgrids."
        exit 0
    fi

    # ------------------------------------------------------------------
    # All volgrids commands must run from pdb_dir with bare filenames.
    # volgrids writes intermediate files (*.pdb.mrc, *.pdb.dx, *.cmap)
    # next to the PDB in CWD.
    # ------------------------------------------------------------------
    cd "$PDB_DIR"

    echo "residues_nobp" $residues_nobp

    config_hbonds="GRID_FORMAT_OUTPUT=MRC DO_SMIF_APBS=true DO_SMIF_HBA=true DO_SMIF_HBD=true DO_SMIF_HYDROPHILIC=false DO_SMIF_HYDROPHOBIC=false DO_SMIF_STACKING=false HBONDS_ONLY_NUCLEOBASE=true"

    # ------------------------------------------------------------------
    # Step 1: SMIF — H-bond fields for non-canonical residues only
    # ------------------------------------------------------------------
    echo "    [1/2] Running SMIF (H-bond, non-canonical) for $NAME"
    yes Y | volgrids smiffer rna "$FILE_NAME" \
        -r "$residues_nobp" \
        -o "$OUT_DIR" \
        -c "$config_hbonds"

    # ------------------------------------------------------------------
    # Step 2: Clean up intermediate files volgrids wrote into pdb_dir
    # ------------------------------------------------------------------
    echo "    [2/2] Cleaning up intermediate files for $NAME"
    rm -f \
        "${PDB_DIR}/${FILE_NAME}.mrc" \
        "${PDB_DIR}/${FILE_NAME}.dx" \
        "${PDB_DIR}/${NAME}.cmap"

    echo "    Done -> $OUT_DIR"

) && echo "Pipeline 2 SUCCESS: $NAME" || {
    EXIT_CODE=$?
    echo "    [ERROR] Processing FAILED for $NAME (exit code $EXIT_CODE)."
    echo "$NAME" >> "$FAILED_LOG"

    rm -f \
        "${PDB_DIR}/${FILE_NAME}.mrc" \
        "${PDB_DIR}/${FILE_NAME}.dx" \
        "${PDB_DIR}/${NAME}.cmap"

    echo "Pipeline 2 FAILED: $NAME"
}
