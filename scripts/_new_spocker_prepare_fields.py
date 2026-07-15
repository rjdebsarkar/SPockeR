#!/usr/bin/env python3
"""
Generate SMIF fields for a single PDB using the current volgrids (1.0.0)
CLI (via the local _fields.py / _residues.py in this folder), and place them
under the Fields_Pipeline1_<PDB_ID>/ / Fields_Pipeline2_<PDB_ID>/ directory
+ filename convention that this folder's Script1-8 expect.

Pipeline1_Fields_Generation.sh and Pipeline2_Fields_Generation.sh (also in
this folder) call `volgrids smiffer rna ...` followed by `volgrids vgtools
unpack`/`convert`, and Pipeline2's config uses DO_SMIF_*-style keys -- none
of which exist in volgrids 1.0.0 anymore (see demo_spocker/legacy/README.md;
the same break was already worked around for legacy/ via
demo_spocker/_legacy_prepare_inputs.py, which this script mirrors).
volgrids 1.0.0's `volgrids smiffer <pdb> ...` writes per-field .mrc files
directly, so no unpack/convert step is needed at all -- _fields.py already
wraps the current CLI correctly (it's a local copy of
demo_spocker/pipeline/fields.py, kept in this folder so new_spocker/ has no
runtime dependency outside itself), so field generation here just reuses it.
Only field generation is reused -- Script1-8 (the actual hotspot /
pocket-detection / scoring algorithms) never call volgrids themselves, only
read the resulting .mrc files, so they run untouched.

Usage: python3 _new_spocker_prepare_fields.py <input.pdb> <work_dir> <pdb_id>
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fields as fields
import _residues as residues
import _structure as structure

# semantic field name (_fields.py) -> new_spocker file suffix (matches
# Script1-8's filename parsing, e.g. Script1's parse_name() and Script4's
# "<name>.hbacceptors.mrc" expectations).
NEW_SPOCKER_FIELD_NAME = {
    "apbs": "apbs",
    "stacking": "stacking",
    "hydrophobic": "hydrophobic",
    "hba": "hbacceptors",
    "hbd": "hbdonors",
}


def _place(semantic_paths: dict, dest_dir: Path, pdb_id: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, path in semantic_paths.items():
        shutil.copy(path, dest_dir / f"{pdb_id}.{NEW_SPOCKER_FIELD_NAME[name]}.mrc")


def main():
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <input.pdb> <work_dir> <pdb_id>")
    pdb_path = Path(sys.argv[1]).resolve()
    work_dir = Path(sys.argv[2]).resolve()
    pdb_id = sys.argv[3]

    fields1_dir = work_dir / f"Fields_Pipeline1_{pdb_id}"
    fields2_dir = work_dir / f"Fields_Pipeline2_{pdb_id}"

    field_work = work_dir / "_field_generation"
    local_pdb = fields.prepare_workdir(pdb_path, field_work)
    apbs_cache = fields.compute_apbs(local_pdb)

    print(f"[prepare-fields] generating whole-structure fields for {pdb_id}")
    whole_paths = fields.compute_whole_structure_fields(local_pdb, apbs_cache, field_work / "whole")
    _place(whole_paths, fields1_dir, pdb_id)

    struct = structure.load_structure(local_pdb)
    selectors = residues.non_canonical_residue_selectors(local_pdb, struct)
    if selectors:
        print(f"[prepare-fields] generating hydrogen-bond fields for "
              f"{len(selectors)} non-canonical residue(s)")
        hb_paths = fields.compute_hbond_subset_fields(local_pdb, selectors, apbs_cache, field_work / "hbond")
        _place(hb_paths, fields2_dir, pdb_id)
    else:
        print("[prepare-fields] no non-canonical residues found; HBond fields skipped")


if __name__ == "__main__":
    main()
