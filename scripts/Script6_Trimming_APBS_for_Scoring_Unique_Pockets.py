#!/usr/bin/env python3
"""
Generic APBS trimming around all RNA heavy atoms
===============================================

Input (from --fields_dir, e.g. Fields_Pipeline1_<PDB_ID>):
    <PDB_ID>.apbs.mrc

Keeps the APBS map only around all RNA heavy atoms using:
    keep radius = VDW(atom element) + cutoff

Default cutoff:
    6.0 Å

Output (written back into --fields_dir):
    <PDB_ID>.apbs_rna_trimmed.mrc
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import mrcfile
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# PARAMETERS (UNCHANGED)
# =============================================================================
CUTOFF_A = 5.0   # Å

VDW_RADIUS = {
    "H":  1.20, "C":  1.70, "N":  1.55, "O":  1.52,
    "P":  1.80, "S":  1.80, "F":  1.47, "CL": 1.75,
    "BR": 1.85, "I":  1.98, "MG": 1.73, "ZN": 1.39,
    "CA": 2.31, "NA": 2.27, "K":  2.75,
}
VDW_DEFAULT = 1.70

RNA_RES = {
    "A", "C", "G", "U", "I",
    "DA", "DC", "DG", "DT", "DI",
    "ADE", "CYT", "GUA", "URI",
}

# =============================================================================
# PDB PARSING (UNCHANGED)
# =============================================================================

def _parse_pdb_line(line: str):
    rec = line[:6].strip().upper()
    if rec not in ("ATOM", "HETATM"):
        return None

    try:
        aname   = line[12:16].strip()
        altloc  = line[16:17].strip()
        resname = line[17:20].strip().upper()

        try:
            resseq = int(line[22:26].strip())
            chain  = line[21:22].strip()
            icode  = line[26:27].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            resseq = int(line[21:25].strip())
            chain  = line[20:21].strip()
            icode  = line[25:26].strip()
            x = float(line[26:34])
            y = float(line[34:42])
            z = float(line[42:50])

        element = ""
        if len(line) >= 78:
            element = line[76:78].strip().upper()
        elif len(line) >= 70:
            element = line[68:70].strip().upper()

        if not element:
            stripped = aname.lstrip("0123456789 ")
            element = stripped[:2].strip().upper()
            if not element:
                element = "C"
            if len(element) == 2 and element not in VDW_RADIUS:
                element = element[0]

        if element == "H":
            return None

    except (ValueError, IndexError):
        return None

    if altloc not in ("", "A"):
        return None

    return {
        "name": aname,
        "resname": resname,
        "chain": chain,
        "resseq": resseq,
        "icode": icode,
        "element": element,
        "xyz": np.array([x, y, z], dtype=np.float64),
    }


def parse_atoms(pdb_path: Path):
    atoms = []
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            a = _parse_pdb_line(line)
            if a is not None:
                atoms.append(a)
    return atoms


def collect_rna_heavy_atoms(atoms):
    rna_atoms = []
    for a in atoms:
        if a["resname"] in RNA_RES and a["element"] != "H":
            rna_atoms.append((a["xyz"], a["element"]))
    return rna_atoms

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
# KD-TREE MASK BUILDER (UNCHANGED)
# =============================================================================

def _build_mask_kdtree(data_shape, voxel, origin, atom_xyz, atom_rads):
    """
    Return boolean mask of voxels to KEEP:
    True where voxel center lies within any atom-specific radius.
    """
    nz, ny, nx = data_shape

    iz, iy, ix = np.mgrid[0:nz, 0:ny, 0:nx]
    vox_coords = np.column_stack([
        origin[0] + ix.ravel() * voxel[0],
        origin[1] + iy.ravel() * voxel[1],
        origin[2] + iz.ravel() * voxel[2],
    ]).astype(np.float32)

    r_max = float(atom_rads.max())
    r_min = float(atom_rads.min())

    tree = cKDTree(atom_xyz)
    dist_near, _ = tree.query(vox_coords, k=1, workers=-1)
    dist_near = dist_near.astype(np.float32)

    primary = dist_near <= r_max

    needs_refine = primary & (dist_near > r_min)
    if needs_refine.any():
        cand_indices = np.where(needs_refine)[0]
        cand_coords  = vox_coords[cand_indices]
        neighbours   = tree.query_ball_point(cand_coords, r=r_max, workers=-1)

        keep = np.zeros(len(cand_indices), dtype=bool)
        for vi, atom_list in enumerate(neighbours):
            for ai in atom_list:
                dx = cand_coords[vi, 0] - atom_xyz[ai, 0]
                dy = cand_coords[vi, 1] - atom_xyz[ai, 1]
                dz = cand_coords[vi, 2] - atom_xyz[ai, 2]
                if dx*dx + dy*dy + dz*dz <= atom_rads[ai] * atom_rads[ai]:
                    keep[vi] = True
                    break

        primary[cand_indices[~keep]] = False

    return primary.reshape(nz, ny, nx)


def build_rna_keep_mask(data_shape, voxel, origin, rna_atoms, cutoff_a):
    atom_xyz = np.array([xyz for xyz, _ in rna_atoms], dtype=np.float32)
    atom_rads = np.array(
        [VDW_RADIUS.get(el, VDW_DEFAULT) + cutoff_a for _, el in rna_atoms],
        dtype=np.float32
    )
    return _build_mask_kdtree(data_shape, voxel, origin, atom_xyz, atom_rads)

# =============================================================================
# FILE DISCOVERY (MODIFIED — searches within --fields_dir instead of a fixed
# ANALYSIS_DIR/<pdb_id> subfolder)
# =============================================================================

def find_apbs_mrc(fields_dir: Path, pdb_id: str):
    primary = fields_dir / f"{pdb_id}.apbs.mrc"
    if primary.exists():
        return primary

    for f in sorted(fields_dir.glob("*.mrc")):
        name = f.name.lower()
        if "apbs" in name and "trimmed" not in name and "marker" not in name:
            return f
    return None

# =============================================================================
# PER-PDB PROCESSING (MODIFIED for single-PDB, explicit fields_dir/pdb_file)
# =============================================================================

def process_pdb(pdb_id: str, fields_dir: Path, pdb_file: Path,
                cutoff_a: float, verbose: bool = True) -> bool:

    if not fields_dir.is_dir():
        if verbose:
            print(f"  [SKIP] Fields directory not found: {fields_dir}")
        return False

    apbs_path = find_apbs_mrc(fields_dir, pdb_id)
    if apbs_path is None:
        if verbose:
            print(f"  [SKIP] No APBS MRC found in {fields_dir}")
        return False

    if pdb_file is None or not pdb_file.is_file():
        if verbose:
            print(f"  [SKIP] No PDB/CIF file found: {pdb_file}")
        return False

    if verbose:
        print(f"  PDB file        : {pdb_file.name}")
        print(f"  APBS MRC        : {apbs_path.name}")
        print(f"  Cutoff          : {cutoff_a:.1f} Å")
        print(f"  Keep radii      : "
              f"P={VDW_RADIUS['P']+cutoff_a:.2f} Å  "
              f"C={VDW_RADIUS['C']+cutoff_a:.2f} Å  "
              f"N={VDW_RADIUS['N']+cutoff_a:.2f} Å  "
              f"O={VDW_RADIUS['O']+cutoff_a:.2f} Å")

    atoms = parse_atoms(pdb_file)
    rna_atoms = collect_rna_heavy_atoms(atoms)

    if not rna_atoms:
        if verbose:
            print("  [SKIP] No RNA heavy atoms found")
        return False

    if verbose:
        print(f"  RNA heavy atoms : {len(rna_atoms)}")

    data, voxel, origin = load_mrc(apbs_path)
    if verbose:
        print(f"  MRC shape       : {data.shape} ({data.size:,} voxels)")
        print(f"  Voxel size      : {voxel[0]:.3f}, {voxel[1]:.3f}, {voxel[2]:.3f} Å")
        print(f"  Origin          : ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")
        print("  Building RNA heavy-atom keep mask ...")

    keep_mask = build_rna_keep_mask(data.shape, voxel, origin, rna_atoms, cutoff_a)

    trimmed = np.zeros_like(data, dtype=np.float32)
    trimmed[keep_mask] = data[keep_mask]

    trimmed_path = fields_dir / f"{pdb_id}.apbs_rna_trimmed.mrc"
    write_mrc(trimmed_path, trimmed, voxel, origin)

    if verbose:
        n_nz_orig = int(np.sum(data != 0.0))
        n_kept = int(np.sum(trimmed != 0.0))
        pct = (100.0 * n_kept / n_nz_orig) if n_nz_orig > 0 else 0.0
        print(f"  Voxels kept     : {n_kept:,} / {n_nz_orig:,} non-zero ({pct:.1f}%)")
        print(f"  Output          : {trimmed_path.name}")

    return True

# =============================================================================
# MAIN (MODIFIED — single-PDB driver, replaces multi-PDB discovery loop)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Keep raw APBS only within a cutoff around all RNA heavy atoms "
                    "for a single PDB."
    )
    parser.add_argument(
        "--fields_dir", required=True,
        help="Path to Fields_Pipeline1_<PDB_ID> folder containing "
             "<PDB_ID>.apbs.mrc. The trimmed output is written back into "
             "this same folder."
    )
    parser.add_argument(
        "--pdb_file", required=True,
        help="Path to the fixed PDB/CIF structure file (e.g. 1AJU_fixed.pdb)."
    )
    parser.add_argument(
        "--pdb_id", default=None,
        help="Optional override for the PDB identifier used in filenames. "
             "If not provided, it is automatically derived from --pdb_file "
             "(filename without extension)."
    )
    parser.add_argument(
        "--cutoff", type=float, default=CUTOFF_A,
        help=f"Distance in Å added to VDW radius for keeping APBS. Default: {CUTOFF_A:.1f}"
    )
    parser.add_argument(
        "--show-params", action="store_true",
        help="Print current parameters and exit"
    )
    args = parser.parse_args()

    if args.show_params:
        c = args.cutoff
        print(f"Cutoff               : {c:.1f} Å")
        print(f"Keep radii           : "
              f"P={VDW_RADIUS['P']+c:.2f}  "
              f"C={VDW_RADIUS['C']+c:.2f}  "
              f"N={VDW_RADIUS['N']+c:.2f}  "
              f"O={VDW_RADIUS['O']+c:.2f} Å")
        return

    fields_dir = Path(args.fields_dir)
    pdb_file   = Path(args.pdb_file)
    pdb_id     = args.pdb_id if args.pdb_id else pdb_file.stem

    if not fields_dir.is_dir():
        print(f"ERROR: Fields dir not found: {fields_dir}")
        sys.exit(1)

    if not pdb_file.is_file():
        print(f"ERROR: PDB file not found: {pdb_file}")
        sys.exit(1)

    print(f"[{pdb_id}]")
    print(f"RNA heavy-atom keep cutoff: {args.cutoff:.1f} Å")
    print("=" * 64)

    try:
        ok = process_pdb(pdb_id, fields_dir=fields_dir, pdb_file=pdb_file,
                         cutoff_a=args.cutoff, verbose=True)
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
