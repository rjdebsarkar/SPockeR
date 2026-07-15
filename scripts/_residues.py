"""
Identify non-canonically-paired residues (the ones hydrogen-bond pockets are
computed for). A residue is "non-canonical" here if RNApolis does not report
it as a Watson-Crick (cWW) base pair partner.

RNApolis is an optional dependency: if it is missing or annotation fails for
a structure, every residue is conservatively treated as non-canonical (with a
warning) rather than aborting the run.

Local copy of demo_spocker/pipeline/residues.py, kept so new_spocker/ has no
runtime dependency outside this folder -- see _new_spocker_prepare_fields.py.
"""

import re
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

from _structure import Structure

try:
    from rnapolis.annotator import (
        extract_base_interactions,
        handle_input_file,
        read_3d_structure,
        write_csv,
    )
    _HAVE_RNAPOLIS = True
except ImportError:
    _HAVE_RNAPOLIS = False


def _parse_resid(nt_label: str):
    """RNApolis nt labels look like "A.123", "A.G.123" or "A.123A"; take the
    last contiguous run of digits, robust to insertion codes / missing chains."""
    m = re.search(r"(\d+)\D*$", nt_label)
    return int(m.group(1)) if m else None


def _canonical_pair_resids(pdb_path: Path) -> set:
    """Residue sequence numbers involved in a cWW base pair, per RNApolis."""
    if not _HAVE_RNAPOLIS:
        print("[WARNING] rnapolis not installed; treating all residues as "
              "non-canonical.", file=sys.stderr)
        return set()

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "annotation.csv"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure3d = read_3d_structure(handle_input_file(pdb_path), None)
                base_interactions = extract_base_interactions(structure3d)
                structure2d, _ = structure3d.extract_secondary_structure(
                    base_interactions, False, False
                )
                write_csv(csv_path, structure2d)
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARNING] RNApolis annotation failed ({e}); treating all "
                  f"residues as non-canonical.", file=sys.stderr)
            return set()

    required = {"type", "classification-1", "classification-2", "nt1", "nt2"}
    if not required.issubset(df.columns):
        return set()

    df = df[
        (df["type"] == "base pair") &
        (df["classification-1"] == "cWW") &
        (df["classification-2"].isin(["XIX", "XX"]))
    ]

    resids = set()
    for col in ("nt1", "nt2"):
        for label in df[col].dropna():
            resid = _parse_resid(str(label))
            if resid is not None:
                resids.add(resid)
    return resids


def non_canonical_residue_selectors(pdb_path, structure: Structure) -> list:
    """Return volgrids '-r' selectors ("chain.resid") for residues that are
    not part of a canonical (cWW) base pair."""
    canonical = _canonical_pair_resids(Path(pdb_path))
    seen = set()
    selectors = []
    for chain, resid in structure.residues:
        if resid in canonical:
            continue
        key = (chain, resid)
        if key in seen:
            continue
        seen.add(key)
        chain_label = chain if chain else "_"
        selectors.append(f"{chain_label}.{resid}")
    return selectors
