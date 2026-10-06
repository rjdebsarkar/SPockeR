"""
Wraps the `volgrids` CLI to generate SMIF grids for one structure, using
volgrids 1.0.0's native workflow throughout.

Field file names produced by volgrids (v1.0.0): apbs, stk (stacking),
hphob (hydrophobic), hphil (hydrophilic, unused downstream), hba/hbd
(H-bond acceptor/donor).

CONFIRMED ROOT CAUSE (isolated by direct testing on 2F4U):
Requesting multiple SMIF field types in ONE `smiffer` call (e.g. the
original single call with -a + all SMIF_* left at their true defaults,
only SMIF_HPHIL=false) produces CORRECT apbs/hphob fields but WRONG
stk/hba/hbd fields -- their density collapses onto the same wrong ~8
junction residues (e.g. A/B.1411, 1412, 1488, 1489/1492) regardless of
residue restriction, APBS cache use, or the -r/-a flags. This reproduced
reliably every time multiple fields were requested together.

Running each field type in its OWN separate `smiffer` call (one call per
SMIF_* flag, all others false) was confirmed via ChimeraX to always place
every field -- apbs, hphob, stk, hba, hbd -- correctly across the whole
structure. This is therefore the only currently-verified-correct calling
convention, and is what this module now uses throughout: every field is
computed via its own isolated `smiffer` invocation, never combined.

Non-canonical / non-base-paired residue selection for HBond fields is
still delegated to volgrids' own `smutils res_nobp`. The residue-
restricted (-r) HBond-subset calls also each request ONLY hba, then ONLY
hbd, as separate calls, for the same reason.
"""

import shutil
import subprocess
from pathlib import Path

# One entry per field: which SMIF_* flag must be true, and which must be
# forced false alongside it (every other SMIF_* flag). Verified via
# isolated single-field `smiffer` calls to always place fields correctly.
WHOLE_STRUCTURE_FIELD_FLAGS = {
    "apbs":   {"SMIF_APBS": "true",  "SMIF_HBA": "false", "SMIF_HBD": "false", "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "false"},
    "hphob":  {"SMIF_APBS": "false", "SMIF_HBA": "false", "SMIF_HBD": "false", "SMIF_HPHIL": "false", "SMIF_HPHOB": "true",  "SMIF_STK": "false"},
    "stk":    {"SMIF_APBS": "false", "SMIF_HBA": "false", "SMIF_HBD": "false", "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "true"},
    "hba":    {"SMIF_APBS": "false", "SMIF_HBA": "true",  "SMIF_HBD": "false", "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "false"},
    "hbd":    {"SMIF_APBS": "false", "SMIF_HBA": "false", "SMIF_HBD": "true",  "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "false"},
}

# HBond-subset (-r restricted) calls: same one-field-per-call rule, plus
# SMIF_HB_ONLY_NBASE=true. SMIF_APBS is never requested here (the
# whole-structure apbs field, computed separately above, is reused
# downstream instead).
HBOND_SUBSET_FIELD_FLAGS = {
    "hba": {"SMIF_APBS": "false", "SMIF_HBA": "true",  "SMIF_HBD": "false", "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "false", "SMIF_HB_ONLY_NBASE": "true"},
    "hbd": {"SMIF_APBS": "false", "SMIF_HBA": "false", "SMIF_HBD": "true",  "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "false", "SMIF_HB_ONLY_NBASE": "true"},
}

FIELD_NAME_MAP = {
    "apbs": "apbs",
    "stk": "stacking",
    "hphob": "hydrophobic",
    "hba": "hba",
    "hbd": "hbd",
}


class FieldGenerationError(RuntimeError):
    pass


def _run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FieldGenerationError(
            f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}"
        )
    return result


def _config_args(overrides: dict) -> list:
    if not overrides:
        return []
    pairs = " ".join(f"{k}={v}" for k, v in overrides.items())
    return ["-c", pairs]


def prepare_workdir(pdb_path: Path, work_dir: Path) -> Path:
    """Copy the input structure into an isolated working directory, since
    volgrids writes its intermediate/output files next to the input file."""
    work_dir.mkdir(parents=True, exist_ok=True)
    local_path = work_dir / pdb_path.name
    shutil.copy(pdb_path, local_path)
    return local_path


def compute_apbs(local_pdb: Path) -> Path:
    """Precompute the APBS potential once, for reuse (via -a) by every
    subsequent per-field smiffer call below."""
    _run(["volgrids", "apbs", local_pdb.name, "--mrc"], cwd=local_pdb.parent)
    apbs_cache = local_pdb.parent / f"{local_pdb.name}.mrc"
    if not apbs_cache.exists():
        raise FieldGenerationError(f"APBS cache not produced: {apbs_cache}")
    return apbs_cache


def compute_non_canonical_indices(local_pdb: Path) -> str:
    """Run volgrids' own `smutils res_nobp` to get the non-base-paired
    residue index string (space-separated "chain.resid" tokens). Returns
    "" if none are found."""
    result = _run(["volgrids", "smutils", "res_nobp", local_pdb.name], cwd=local_pdb.parent)
    return result.stdout.strip()


def compute_whole_structure_fields(local_pdb: Path, apbs_cache: Path, out_dir: Path) -> dict:
    """Issues one separate `smiffer` call per field type (see module
    docstring for why this is required -- combining fields in one call
    corrupts stk/hba/hbd). Each call writes to its own subfolder to avoid
    any risk of one call's intermediate files being picked up by another."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for field, flags in WHOLE_STRUCTURE_FIELD_FLAGS.items():
        field_out = out_dir / field
        field_out.mkdir(parents=True, exist_ok=True)
        cmd = ["volgrids", "smiffer", local_pdb.name, "-a", str(apbs_cache), "-o", str(field_out)]
        cmd += _config_args(flags)
        _run(cmd, cwd=local_pdb.parent)
        candidate = field_out / f"{local_pdb.stem}.{field}.mrc"
        if candidate.exists():
            paths[FIELD_NAME_MAP[field]] = candidate
    return paths


def compute_hbond_subset_fields(local_pdb: Path, indices: str, out_dir: Path) -> dict:
    """`indices` must be the raw output string of compute_non_canonical_indices,
    passed whole to `-r`. Issues separate calls for hba and hbd (see module
    docstring)."""
    if not indices:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for field, flags in HBOND_SUBSET_FIELD_FLAGS.items():
        field_out = out_dir / field
        field_out.mkdir(parents=True, exist_ok=True)
        cmd = ["volgrids", "smiffer", local_pdb.name, "-r", indices, "-o", str(field_out)]
        cmd += _config_args(flags)
        _run(cmd, cwd=local_pdb.parent)
        candidate = field_out / f"{local_pdb.stem}.{field}.mrc"
        if candidate.exists():
            paths[FIELD_NAME_MAP[field]] = candidate
    return paths
