import sys
import io
import logging
import warnings
import re
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import MDAnalysis as mda
import pandas as pd

from rnapolis.annotator import (
    extract_base_interactions,
    handle_input_file,
    read_3d_structure,
    write_csv,
)

# ------------------------------------------------------------------------------
def run_rnapolis(path_pdb: Path, path_csv: Path):
    try:
        file = handle_input_file(path_pdb)
        structure3d = read_3d_structure(file, None)
        base_interactions = extract_base_interactions(structure3d)
        structure2d, _ = structure3d.extract_secondary_structure(
            base_interactions
        )
        write_csv(path_csv, structure2d)
    except Exception as e:
        print(f"[WARNING] RNApolis annotation failed ({e}); "
              f"all residues will be treated as non-canonical.", file=sys.stderr)
        pd.DataFrame(
            columns=["type", "classification-1", "classification-2", "nt1", "nt2"]
        ).to_csv(path_csv, index=False)


# ------------------------------------------------------------------------------
def _parse_chain_resid(nt_label: str):
    """
    Extract (chain, resid) from an RNApolis nt label.

    Observed formats:
      "A.123"     -> ("A", 123)
      "A.123A"    -> ("A", 123)   (insertion code stripped)
      "A.G.123"   -> ("A", 123)   (chain . resname . resid)
      ".123"      -> (None, 123)  (empty chain ID)
    """
    parts = nt_label.split(".")
    chain = parts[0].strip() if parts[0].strip() else None
    m = re.search(r'(\d+)\D*$', nt_label)
    if m:
        return chain, int(m.group(1))
    return chain, None


# ------------------------------------------------------------------------------
def get_idxs_canonical(path_csv: Path) -> set:
    try:
        df = pd.read_csv(path_csv)
    except Exception:
        return set()

    required = {"type", "classification-1", "classification-2", "nt1", "nt2"}
    if not required.issubset(df.columns):
        return set()

    df = df[
        (df["type"] == "base pair") &
        (df["classification-1"] == "cWW") &
        (
            (df["classification-2"] == "XIX") |
            (df["classification-2"] == "XX")
        )
    ]

    idxs_canonical = set()
    for col in ("nt1", "nt2"):
        for label in df[col].dropna():
            chain, resid = _parse_chain_resid(str(label))
            if resid is not None:
                idxs_canonical.add((chain, resid))

    return idxs_canonical


# ------------------------------------------------------------------------------
def get_mda_universe_quiet(path_pdb) -> mda.Universe:
    buf = io.StringIO()
    logger = logging.getLogger("MDAnalysis")
    old_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(buf), redirect_stderr(buf):
                u = mda.Universe(str(path_pdb))
    finally:
        logger.setLevel(old_level)
    return u


# ------------------------------------------------------------------------------
def get_idxs_all_mda(path_pdb) -> set:
    """Primary path: use MDAnalysis to collect (chain, resid) pairs."""
    u = get_mda_universe_quiet(path_pdb)
    idxs = set()
    for res in u.residues:
        try:
            chain = res.segid.strip() if res.segid.strip() else (
                res.atoms.chainIDs[0] if hasattr(res.atoms, "chainIDs") else None
            )
        except Exception:
            chain = None
        idxs.add((chain, int(res.resid)))
    return idxs


def get_idxs_all_manual(path_pdb) -> set:
    """
    Fallback: parse ATOM/HETATM lines directly from the PDB text.
    PDB column layout (1-based, fixed-width):
      cols  1- 6  record type
      col      22  chain ID
      cols 23-26  residue sequence number (resSeq)
    """
    idxs = set()
    with open(path_pdb, "r", errors="replace") as fh:
        for line in fh:
            if line[:6].strip() not in ("ATOM", "HETATM"):
                continue
            try:
                chain = line[21:22].strip() or None
                resid = int(line[22:26].strip())
                idxs.add((chain, resid))
            except ValueError:
                pass
    return idxs


def get_idxs_all(path_pdb) -> set:
    try:
        idxs = get_idxs_all_mda(path_pdb)
        if idxs:
            return idxs
    except Exception as e:
        print(f"[WARNING] MDAnalysis failed ({e}); falling back to manual PDB parse.",
              file=sys.stderr)

    return get_idxs_all_manual(path_pdb)


# ------------------------------------------------------------------------------
def main():
    run_rnapolis(PATH_PDB, PATH_CSV)
    idxs_canonical = get_idxs_canonical(PATH_CSV)
    idxs_all       = get_idxs_all(PATH_PDB)

    # Match canonical/all sets even if chain info is missing on one side
    canonical_resids_only = {resid for (_, resid) in idxs_canonical}

    idxs_available = set()
    for chain, resid in idxs_all:
        if (chain, resid) in idxs_canonical:
            continue
        if chain is None and resid in canonical_resids_only:
            continue
        idxs_available.add((chain if chain else "A", resid))

    formatted = sorted(idxs_available, key=lambda x: (x[0], x[1]))
    print(' '.join(f"{chain}.{resid}" for chain, resid in formatted))


################################################################################
if __name__ == "__main__":
    warnings.filterwarnings("ignore", module="MDAnalysis.*")
    PATH_PDB = Path(sys.argv[1])
    PATH_CSV = (Path(sys.argv[2]) if len(sys.argv) > 2
                else PATH_PDB.with_suffix(".csv"))
    main()
