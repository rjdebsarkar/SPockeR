#!/usr/bin/env python3
"""
Hydrophobic field trimming by removing overlap with stacking field
=================================================================

Inputs (from --fields_dir, e.g. Fields_Pipeline1_<PDB_ID>):
    <PDB_ID>.hydrophobic.mrc
    <PDB_ID>.stacking.mrc

Removes the hydrophobic voxels that overlap with the stacking field.

Output (written back into --fields_dir):
    <PDB_ID>.hydrophobic_nonoverlap_trimmed.mrc
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import mrcfile

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# PARAMETERS (UNCHANGED)
# =============================================================================
EPSILON = 1.0e-6   # values with abs(value) <= EPSILON are treated as zero

# =============================================================================
# MRC UTILITIES (UNCHANGED)
# =============================================================================

def load_mrc(path: Path):
    with mrcfile.open(str(path), mode="r", permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32).copy()
        voxel = np.array([
            float(mrc.voxel_size.x),
            float(mrc.voxel_size.y),
            float(mrc.voxel_size.z)
        ], dtype=np.float64)

        try:
            origin = np.array([
                float(mrc.header.origin.x),
                float(mrc.header.origin.y),
                float(mrc.header.origin.z)
            ], dtype=np.float64)
        except Exception:
            origin = np.zeros(3, dtype=np.float64)

        if np.allclose(origin, 0.0):
            try:
                ox = float(mrc.header.nxstart) * voxel[0]
                oy = float(mrc.header.nystart) * voxel[1]
                oz = float(mrc.header.nzstart) * voxel[2]
                if not (ox == 0.0 and oy == 0.0 and oz == 0.0):
                    origin = np.array([ox, oy, oz], dtype=np.float64)
            except Exception:
                pass

    return data, voxel, origin


def write_mrc(path: Path, data: np.ndarray, voxel: np.ndarray, origin: np.ndarray):
    d = np.asarray(data, dtype=np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(d)
        mrc.voxel_size = tuple(float(v) for v in voxel)
        try:
            mrc.header.origin.x = float(origin[0])
            mrc.header.origin.y = float(origin[1])
            mrc.header.origin.z = float(origin[2])
        except Exception:
            pass
        mrc.update_header_from_data()
        mrc.update_header_stats()

# =============================================================================
# FILE DISCOVERY (MODIFIED — searches within --fields_dir instead of a fixed
# ANALYSIS_DIR/<pdb_id> subfolder)
# =============================================================================

def find_field_mrc(fields_dir: Path, pdb_id: str, field_name: str):
    primary = fields_dir / f"{pdb_id}.{field_name}.mrc"
    if primary.exists():
        return primary

    for f in sorted(fields_dir.glob("*.mrc")):
        name = f.name.lower()
        if field_name.lower() in name and "trimmed" not in name and "nonoverlap" not in name:
            return f
    return None

# =============================================================================
# OVERLAP REMOVAL (UNCHANGED)
# =============================================================================

def validate_same_grid(h_data, h_voxel, h_origin, s_data, s_voxel, s_origin):
    if h_data.shape != s_data.shape:
        raise ValueError(
            f"Grid shape mismatch: hydrophobic {h_data.shape} vs stacking {s_data.shape}"
        )
    if not np.allclose(h_voxel, s_voxel, atol=1e-6):
        raise ValueError(
            f"Voxel size mismatch: hydrophobic {h_voxel} vs stacking {s_voxel}"
        )
    if not np.allclose(h_origin, s_origin, atol=1e-6):
        raise ValueError(
            f"Origin mismatch: hydrophobic {h_origin} vs stacking {s_origin}"
        )


def remove_hydrophobic_overlap(hydrophobic_data, stacking_data, epsilon):
    hydrophobic_present = np.abs(hydrophobic_data) > epsilon
    stacking_present    = np.abs(stacking_data) > epsilon

    overlap_mask = hydrophobic_present & stacking_present

    trimmed = hydrophobic_data.copy()
    trimmed[overlap_mask] = 0.0

    return trimmed, overlap_mask, hydrophobic_present, stacking_present

# =============================================================================
# PER-PDB PROCESSING (MODIFIED for single-PDB, explicit fields_dir)
# =============================================================================

def process_pdb(pdb_id: str, fields_dir: Path, epsilon: float, verbose: bool = True) -> bool:
    if not fields_dir.is_dir():
        if verbose:
            print(f"  [SKIP] Fields directory not found: {fields_dir}")
        return False

    hydrophobic_path = find_field_mrc(fields_dir, pdb_id, "hydrophobic")
    if hydrophobic_path is None:
        if verbose:
            print(f"  [SKIP] No hydrophobic MRC found in {fields_dir}")
        return False

    stacking_path = find_field_mrc(fields_dir, pdb_id, "stacking")
    if stacking_path is None:
        if verbose:
            print(f"  [SKIP] No stacking MRC found in {fields_dir}")
        return False

    if verbose:
        print(f"  Hydrophobic MRC : {hydrophobic_path.name}")
        print(f"  Stacking MRC    : {stacking_path.name}")
        print(f"  Zero threshold  : {epsilon:.2e}")

    hydrophobic_data, h_voxel, h_origin = load_mrc(hydrophobic_path)
    stacking_data,   s_voxel, s_origin = load_mrc(stacking_path)

    validate_same_grid(hydrophobic_data, h_voxel, h_origin,
                       stacking_data, s_voxel, s_origin)

    if verbose:
        print(f"  MRC shape       : {hydrophobic_data.shape} ({hydrophobic_data.size:,} voxels)")
        print(f"  Voxel size      : {h_voxel[0]:.3f}, {h_voxel[1]:.3f}, {h_voxel[2]:.3f} Å")
        print(f"  Origin          : ({h_origin[0]:.2f}, {h_origin[1]:.2f}, {h_origin[2]:.2f})")
        print("  Removing hydrophobic voxels overlapping with stacking ...")

    trimmed, overlap_mask, hydrophobic_present, stacking_present = remove_hydrophobic_overlap(
        hydrophobic_data, stacking_data, epsilon
    )

    output_path = fields_dir / f"{pdb_id}.hydrophobic_nonoverlap_trimmed.mrc"
    write_mrc(output_path, trimmed, h_voxel, h_origin)

    if verbose:
        n_h = int(np.sum(hydrophobic_present))
        n_s = int(np.sum(stacking_present))
        n_overlap = int(np.sum(overlap_mask))
        n_kept = int(np.sum(np.abs(trimmed) > epsilon))
        pct_removed = (100.0 * n_overlap / n_h) if n_h > 0 else 0.0

        print(f"  Hydrophobic voxels : {n_h:,}")
        print(f"  Stacking voxels    : {n_s:,}")
        print(f"  Overlap removed    : {n_overlap:,} ({pct_removed:.1f}% of hydrophobic)")
        print(f"  Voxels kept        : {n_kept:,}")
        print(f"  Output             : {output_path.name}")

    return True

# =============================================================================
# MAIN (MODIFIED — single-PDB driver, replaces multi-PDB discovery loop)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Remove hydrophobic field overlap with stacking field, "
                    "for a single PDB."
    )
    parser.add_argument(
        "--fields_dir", required=True,
        help="Path to Fields_Pipeline1_<PDB_ID> folder containing "
             "<PDB_ID>.hydrophobic.mrc and <PDB_ID>.stacking.mrc. "
             "The trimmed output is written back into this same folder."
    )
    parser.add_argument(
        "--pdb_id", default=None,
        help="PDB identifier used to locate/name files (e.g. 1AJU_fixed). "
             "If not provided, it is automatically derived from whichever "
             "*.hydrophobic.mrc file is found in --fields_dir."
    )
    parser.add_argument(
        "--epsilon", type=float, default=EPSILON,
        help=f"Absolute value threshold for treating voxels as non-zero. Default: {EPSILON:.1e}"
    )
    parser.add_argument(
        "--show-params", action="store_true",
        help="Print current parameters and exit"
    )
    args = parser.parse_args()

    if args.show_params:
        print(f"Zero threshold        : {args.epsilon:.2e}")
        return

    fields_dir = Path(args.fields_dir)
    if not fields_dir.is_dir():
        print(f"ERROR: Fields dir not found: {fields_dir}")
        sys.exit(1)

    if args.pdb_id:
        pdb_id = args.pdb_id
    else:
        candidates = sorted(fields_dir.glob("*.hydrophobic.mrc"))
        if not candidates:
            print(f"ERROR: No *.hydrophobic.mrc file found in: {fields_dir}")
            print("  Provide --pdb_id explicitly, or ensure "
                  "<PDB_ID>.hydrophobic.mrc exists in --fields_dir.")
            sys.exit(1)
        pdb_id = candidates[0].name[:-len(".hydrophobic.mrc")]

    print(f"[{pdb_id}]")
    print(f"Hydrophobic/stacking overlap threshold: {args.epsilon:.2e}")
    print("=" * 64)

    try:
        ok = process_pdb(pdb_id, fields_dir=fields_dir, epsilon=args.epsilon, verbose=True)
    except Exception as exc:
        import traceback
        print(f"  [ERROR] {exc}")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 64)
    if ok:
        print("Done. Success.")
    else:
        print("Done. Skipped (see reason above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
