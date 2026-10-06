#!/usr/bin/env python3
"""
Generate SMIF fields for a single PDB using volgrids 1.0.0's own CLI
workflow end-to-end, including its native `smutils res_nobp` for
non-canonical residue selection.

The HBond-subset call no longer produces its own APBS field (SMIF_APBS is
forced false there -- see _fields.py's module docstring for why). If
downstream scoring scripts (Script4/5/6) expect an APBS file inside
Fields_Pipeline2_*, the whole-structure one is copied over as a stand-in;
remove that copy step below if it turns out not to be needed.

Usage: python3 _new_spocker_prepare_fields.py <pdb> <work_dir> <pdb_id>
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fields as fields

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
        sys.exit(f"Usage: {sys.argv[0]} <pdb> <work_dir> <pdb_id>")
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

    indices = fields.compute_non_canonical_indices(local_pdb)
    if indices:
        n = len(indices.split())
        print(f"[prepare-fields] generating hydrogen-bond fields for {n} non-base-paired residue(s): {indices}")
        hb_paths = fields.compute_hbond_subset_fields(local_pdb, indices, field_work / "hbond")
        _place(hb_paths, fields2_dir, pdb_id)

        # HBond-subset call produces no APBS field of its own (SMIF_APBS=false
        # there); reuse the whole-structure one in Fields_Pipeline2_ in case
        # Script4/5/6 expect an .apbs.mrc file alongside hbacceptors/hbdonors.
        pipeline1_apbs = fields1_dir / f"{pdb_id}.apbs.mrc"
        pipeline2_apbs = fields2_dir / f"{pdb_id}.apbs.mrc"
        if pipeline1_apbs.exists():
            shutil.copy(pipeline1_apbs, pipeline2_apbs)
    else:
        print("[prepare-fields] no non-base-paired residues found; HBond fields skipped")


if __name__ == "__main__":
    main()
