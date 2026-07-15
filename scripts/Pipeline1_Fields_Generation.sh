#!/bin/bash
# =============================================================================
# Pipeline1_Fields_Generation.sh
# Generates full SMIF fields (ELE, STK, HPb, HPhi, HBA, HBD) for a single PDB
#
# Usage: ./Pipeline1_Fields_Generation.sh <path_to_fixed_pdb>
# Example: ./Pipeline1_Fields_Generation.sh 1AJU.pdb
#          ./Pipeline1_Fields_Generation.sh /home/user/data/1AJU.pdb
#
# Output: Creates ./Fields_Pipeline1_<PDB_ID>/ in the current working
#         directory, containing the final per-field .mrc grids.
# =============================================================================
set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path_to_fixed_pdb>"
    echo "Example: $0 1AJU.pdb"
    exit 1
fi

INPUT_PDB="$1"

if [[ ! -f "$INPUT_PDB" ]]; then
    echo "ERROR: Input PDB not found: $INPUT_PDB"
    exit 1
fi

PDB_DIR="$(cd "$(dirname "$INPUT_PDB")" && pwd)"
FILE_NAME="$(basename "$INPUT_PDB")"
ID="${FILE_NAME%.*}"

RUN_DIR="$(pwd)"
FIELDS_DIR="$RUN_DIR/Fields_Pipeline1_${ID}"
TMP_GRID_DIR="$FIELDS_DIR/_tmp_unpacked"

mkdir -p "$FIELDS_DIR" "$TMP_GRID_DIR"

echo "=================================================="
echo "Pipeline 1 — Processing $ID"
echo "Input structure : $PDB_DIR/$FILE_NAME"
echo "Output folder   : $FIELDS_DIR"
echo "=================================================="

(
    set -eo pipefail
    cd "$PDB_DIR"

    # ----------------------------------------------------------------
    # Step 1: APBS electrostatics
    # ----------------------------------------------------------------
    echo "[1/4] Running APBS for $ID"
    yes Y | volgrids apbs "$FILE_NAME" --mrc

    APBS_MRC="${PDB_DIR}/${FILE_NAME}.mrc"
    if [[ ! -f "$APBS_MRC" ]]; then
        echo "ERROR: APBS output not found: $APBS_MRC"
        exit 1
    fi

    # ----------------------------------------------------------------
    # Step 2: SMIF — all fields, whole structure
    # ----------------------------------------------------------------
    echo "[2/4] Running SMIF (all fields) for $ID"
    yes Y | volgrids smiffer rna "$FILE_NAME" -a "${FILE_NAME}.mrc" -o "$TMP_GRID_DIR"

    CMAP_FILE="$TMP_GRID_DIR/${ID}.cmap"
    if [[ ! -f "$CMAP_FILE" ]]; then
        echo "ERROR: Expected CMAP file not found: $CMAP_FILE"
        ls -la "$TMP_GRID_DIR/" || true
        exit 1
    fi

    # ----------------------------------------------------------------
    # Step 3: Unpack combined CMAP into per-field CMAPs
    # ----------------------------------------------------------------
    echo "[3/4] Unpacking CMAP for $ID"
    yes Y | volgrids vgtools unpack "$CMAP_FILE"

    # ----------------------------------------------------------------
    # Step 4: Convert each unpacked field CMAP -> MRC (saved into FIELDS_DIR)
    # ----------------------------------------------------------------
    echo "[4/4] Converting CMAP to MRC for $ID"
    FOUND_GRID=0
    for f in "$TMP_GRID_DIR"/${ID}.*.cmap; do
        [[ -e "$f" ]] || continue
        FOUND_GRID=1
        OUT_MRC="$FIELDS_DIR/$(basename "${f%.cmap}.mrc")"
        echo "  Converting: $(basename "$f") -> $(basename "$OUT_MRC")"
        yes Y | volgrids vgtools convert "$f" -m "$OUT_MRC"
    done

    if [[ "$FOUND_GRID" -eq 0 ]]; then
        echo "ERROR: No unpacked CMAP grid files found in $TMP_GRID_DIR"
        exit 1
    fi

    MRC_COUNT="$(find "$FIELDS_DIR" -maxdepth 1 -name "*.mrc" | wc -l)"
    if [[ "$MRC_COUNT" -eq 0 ]]; then
        echo "ERROR: convert ran but no .mrc files found in $FIELDS_DIR"
        exit 1
    fi

    # Cleanup intermediates written next to the PDB and the temp unpack dir
    rm -f "${PDB_DIR}/${FILE_NAME}.mrc" "${PDB_DIR}/${FILE_NAME}.dx" "${PDB_DIR}/${ID}.cmap"
    rm -rf "$TMP_GRID_DIR"

    echo "Finished $ID — $MRC_COUNT .mrc file(s) saved in: $FIELDS_DIR"

) && echo "Pipeline 1 SUCCESS: $ID" || echo "Pipeline 1 FAILED: $ID"
