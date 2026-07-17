#!/bin/bash
# =============================================================================
# fix_pdb_apo.sh
# Downloads raw PDB behavior: selects model 1, strips ligands/heteroatoms,
# then runs pdbfixer to produce a clean APO RNA structure.
# Usage: ./fix_pdb_apo.sh <raw_input.pdb> <output_fixed.pdb>
# =============================================================================
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <raw_input.pdb> <output_fixed.pdb>"
    exit 1
fi

RAW_PDB="$1"
FIXED_PDB="$2"
TMP_MODEL1="$(mktemp --suffix=.pdb)"
TMP_APO="$(mktemp --suffix=.pdb)"

echo "[1/3] Selecting first model (if multi-model NMR structure)..."
pdb_selmodel -1 "$RAW_PDB" > "$TMP_MODEL1"

echo "[2/3] Removing all HETATM records (ligands, ions, waters)..."
pdb_delhetatm "$TMP_MODEL1" > "$TMP_APO"

echo "[3/3] Running pdbfixer on APO structure..."
pdbfixer "$TMP_APO" \
    --replace-nonstandard \
    --keep-heterogens none \
    --output "$FIXED_PDB"

rm -f "$TMP_MODEL1" "$TMP_APO"

echo "Done. Clean APO fixed PDB saved to: $FIXED_PDB"
