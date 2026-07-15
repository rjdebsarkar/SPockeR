"""
  Inputs:
    --pdb_file                                        — RNA structure (.pdb)
    --analysis_dir/<name>.HBond_Site_N_marker.mrc     — marker MRCs (from Script4)

  Outputs (written to --analysis_dir):
    <name>.HBond_Site_N_pocket_volume.mrc
    <name>.pocket_volume_summary.txt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import mrcfile
from scipy.ndimage import label


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

# Radius of the imaginary seed sphere drawn around the binding-site centre (Å)
SEED_SPHERE_RADIUS_A = 8.0

# Exclusion radius around each heavy atom (Å)
ATOM_EXCLUSION_RADIUS_A = 3.0

# Fragment connectivity after trimming: 3 = 26-connectivity (recommended)
POCKET_CONNECTIVITY = 3

# Maximum number of marker MRC files expected per PDB (Site 1 … Site N)
MAX_SITES = 2

# Fallback voxel size (Å) — used only when MRC header is unreadable
VOXEL_SIZE_A = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# MRC I/O (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def load_mrc(path):
    with mrcfile.open(path, mode='r', permissive=True) as mrc:
        data   = mrc.data.copy().astype(np.float32)
        vs     = np.array([float(mrc.voxel_size.x),
                           float(mrc.voxel_size.y),
                           float(mrc.voxel_size.z)], dtype=np.float64)
        origin = np.array([float(mrc.header.origin.x),
                           float(mrc.header.origin.y),
                           float(mrc.header.origin.z)], dtype=np.float64)
        if np.allclose(origin, 0.0):
            origin = np.array([int(mrc.header.nxstart) * vs[0],
                               int(mrc.header.nystart) * vs[1],
                               int(mrc.header.nzstart) * vs[2]],
                              dtype=np.float64)
        if not np.all(vs > 0.0):
            print(f"    WARNING: implausible voxel size in "
                  f"{os.path.basename(path)}, using fallback {VOXEL_SIZE_A} A")
            vs[:] = VOXEL_SIZE_A
        nz, ny, nx = data.shape
        print(f"    {os.path.basename(path)}: shape=({nz},{ny},{nx})  "
              f"voxel=({vs[0]:.3f},{vs[1]:.3f},{vs[2]:.3f}) A  "
              f"origin=({origin[0]:.2f},{origin[1]:.2f},{origin[2]:.2f})  "
              f"nonzero={np.count_nonzero(data):,}/{data.size:,}")
    return data, vs, origin


def save_mrc(out_path, data_3d, vs, origin):
    data_3d      = data_3d.astype(np.float32)
    nz, ny, nx   = data_3d.shape
    nonzero_vals = data_3d[data_3d != 0.0]
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(data_3d)
        mrc.voxel_size = (float(vs[0]), float(vs[1]), float(vs[2]))
        mrc.header.origin.x = float(origin[0])
        mrc.header.origin.y = float(origin[1])
        mrc.header.origin.z = float(origin[2])
        mrc.header.mx = nx
        mrc.header.my = ny
        mrc.header.mz = nz
        mrc.header.cella.x = float(nx * vs[0])
        mrc.header.cella.y = float(ny * vs[1])
        mrc.header.cella.z = float(nz * vs[2])
        mrc.header.mapc  = 1
        mrc.header.mapr  = 2
        mrc.header.maps  = 3
        mrc.header.mode  = 2
        mrc.header.dmin  = float(data_3d.min())
        mrc.header.dmax  = float(data_3d.max())
        mrc.header.dmean = (float(nonzero_vals.mean())
                            if len(nonzero_vals) > 0 else 0.0)
    print(f"    Saved: {os.path.basename(out_path)}  "
          f"({np.count_nonzero(data_3d):,} nonzero voxels)")


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTIVITY STRUCTURE (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def make_conn_struct(connectivity):
    if connectivity == 1:
        return np.array([[[0,0,0],[0,1,0],[0,0,0]],
                         [[0,1,0],[1,1,1],[0,1,0]],
                         [[0,0,0],[0,1,0],[0,0,0]]], dtype=bool)
    elif connectivity == 2:
        return np.array([[[0,1,0],[1,1,1],[0,1,0]],
                         [[1,1,1],[1,1,1],[1,1,1]],
                         [[0,1,0],[1,1,1],[0,1,0]]], dtype=bool)
    else:
        return np.ones((3, 3, 3), dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# PDB PARSER  —  v2: robust column-slice reader (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def _is_hydrogen(atom_name_raw, element_raw):
    elem = element_raw.strip().upper()
    if elem == 'H':
        return True
    name = atom_name_raw.strip()
    if name and name[0].isdigit():
        name = name[1:]
    if name and name[0].upper() == 'H':
        return True
    return False


def parse_pdb_heavy_atoms(pdb_path):
    coords        = []
    n_lines_total = 0
    n_atom_lines  = 0
    n_coord_fail  = 0
    n_h_skipped   = 0

    with open(pdb_path, 'r') as fh:
        for line in fh:
            n_lines_total += 1
            rec = line[:6]
            if rec not in ('ATOM  ', 'HETATM'):
                continue
            n_atom_lines += 1
            if len(line) < 54:
                n_coord_fail += 1
                continue
            atom_name_raw = line[12:16] if len(line) > 15 else '    '
            element_raw   = line[76:78] if len(line) > 77 else '  '
            if _is_hydrogen(atom_name_raw, element_raw):
                n_h_skipped += 1
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                n_coord_fail += 1
                continue
            coords.append((x, y, z))

    print(f"    PDB {os.path.basename(pdb_path)}:")
    print(f"      Total lines       : {n_lines_total:,}")
    print(f"      ATOM/HETATM lines : {n_atom_lines:,}")
    print(f"      Hydrogen skipped  : {n_h_skipped:,}")
    print(f"      Coord parse fail  : {n_coord_fail:,}")
    print(f"      Heavy atoms kept  : {len(coords):,}")

    if not coords:
        print(f"    DIAGNOSTIC — first 10 ATOM/HETATM lines of "
              f"{os.path.basename(pdb_path)}:")
        count = 0
        with open(pdb_path, 'r') as fh:
            for line in fh:
                if line[:6] in ('ATOM  ', 'HETATM'):
                    print(f"      [{len(line):3d} chars] {repr(line[:80])}")
                    count += 1
                    if count >= 10:
                        break
        if count == 0:
            print("      (no ATOM/HETATM records found at all — check file format)")
        raise ValueError(
            f"No heavy atoms parsed from {pdb_path}. "
            "See diagnostic output above."
        )

    return np.array(coords, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# MARKER CENTRE EXTRACTION (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def marker_centre_angstrom(marker_data, vs, origin):
    zi, yi, xi = np.where(marker_data != 0.0)
    if len(zi) == 0:
        raise ValueError("Marker MRC has no nonzero voxels — cannot determine centre.")
    xi_f = xi.mean()
    yi_f = yi.mean()
    zi_f = zi.mean()
    cx = origin[0] + xi_f * vs[0]
    cy = origin[1] + yi_f * vs[1]
    cz = origin[2] + zi_f * vs[2]
    ix_c = int(round(xi_f))
    iy_c = int(round(yi_f))
    iz_c = int(round(zi_f))
    return np.array([cx, cy, cz], dtype=np.float64), (iz_c, iy_c, ix_c)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  —  BUILD SEED SPHERE (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def build_seed_sphere(centre_a, vs, origin, grid_shape,
                      radius_a=SEED_SPHERE_RADIUS_A):
    nz, ny, nx = grid_shape
    ci_x = (centre_a[0] - origin[0]) / vs[0]
    ci_y = (centre_a[1] - origin[1]) / vs[1]
    ci_z = (centre_a[2] - origin[2]) / vs[2]
    rx = radius_a / vs[0]
    ry = radius_a / vs[1]
    rz = radius_a / vs[2]
    ix_lo = max(0,    int(np.floor(ci_x - rx)))
    ix_hi = min(nx-1, int(np.ceil (ci_x + rx)))
    iy_lo = max(0,    int(np.floor(ci_y - ry)))
    iy_hi = min(ny-1, int(np.ceil (ci_y + ry)))
    iz_lo = max(0,    int(np.floor(ci_z - rz)))
    iz_hi = min(nz-1, int(np.ceil (ci_z + rz)))
    sphere_mask = np.zeros(grid_shape, dtype=bool)
    iz_r = np.arange(iz_lo, iz_hi + 1)
    iy_r = np.arange(iy_lo, iy_hi + 1)
    ix_r = np.arange(ix_lo, ix_hi + 1)
    iz_g, iy_g, ix_g = np.meshgrid(iz_r, iy_r, ix_r, indexing='ij')
    dx_a    = (ix_g - ci_x) * vs[0]
    dy_a    = (iy_g - ci_y) * vs[1]
    dz_a    = (iz_g - ci_z) * vs[2]
    dist2_a = dx_a**2 + dy_a**2 + dz_a**2
    sphere_mask[iz_lo:iz_hi+1, iy_lo:iy_hi+1, ix_lo:ix_hi+1] = \
        dist2_a <= radius_a**2
    n_in = int(sphere_mask.sum())
    print(f"    Seed sphere: centre_vox=({ci_x:.1f},{ci_y:.1f},{ci_z:.1f})  "
          f"radius={radius_a:.1f} A  voxels_in_sphere={n_in:,}")
    return sphere_mask


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  —  TRIM SPHERE BY RNA ATOMS (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def trim_sphere_by_atoms(sphere_mask, atom_coords, vs, origin,
                         excl_radius_a=ATOM_EXCLUSION_RADIUS_A):
    nz, ny, nx = sphere_mask.shape
    rx = excl_radius_a / vs[0]
    ry = excl_radius_a / vs[1]
    rz = excl_radius_a / vs[2]
    r2 = excl_radius_a ** 2
    n_excluded = 0
    for ax, ay, az in atom_coords:
        ci_x = (ax - origin[0]) / vs[0]
        ci_y = (ay - origin[1]) / vs[1]
        ci_z = (az - origin[2]) / vs[2]
        ix_lo = max(0,    int(np.floor(ci_x - rx)))
        ix_hi = min(nx-1, int(np.ceil (ci_x + rx)))
        iy_lo = max(0,    int(np.floor(ci_y - ry)))
        iy_hi = min(ny-1, int(np.ceil (ci_y + ry)))
        iz_lo = max(0,    int(np.floor(ci_z - rz)))
        iz_hi = min(nz-1, int(np.ceil (ci_z + rz)))
        if ix_lo > ix_hi or iy_lo > iy_hi or iz_lo > iz_hi:
            continue
        sub = sphere_mask[iz_lo:iz_hi+1, iy_lo:iy_hi+1, ix_lo:ix_hi+1]
        if not sub.any():
            continue
        iz_r = np.arange(iz_lo, iz_hi + 1)
        iy_r = np.arange(iy_lo, iy_hi + 1)
        ix_r = np.arange(ix_lo, ix_hi + 1)
        iz_g, iy_g, ix_g = np.meshgrid(iz_r, iy_r, ix_r, indexing='ij')
        dx_a  = (ix_g - ci_x) * vs[0]
        dy_a  = (iy_g - ci_y) * vs[1]
        dz_a  = (iz_g - ci_z) * vs[2]
        dist2 = dx_a**2 + dy_a**2 + dz_a**2
        to_excl = (dist2 <= r2) & sub
        n_excluded += int(to_excl.sum())
        sphere_mask[iz_lo:iz_hi+1, iy_lo:iy_hi+1, ix_lo:ix_hi+1][to_excl] = False
    return sphere_mask, n_excluded


# ─────────────────────────────────────────────────────────────────────────────
# STEPS 5-6  —  LABEL FRAGMENTS AND SELECT POCKET PIECE (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pocket_fragment(trimmed_mask, centre_a, centre_vox,
                            vs, origin, connectivity=POCKET_CONNECTIVITY):
    struct = make_conn_struct(connectivity)
    labeled, n_labels = label(trimmed_mask, structure=struct)
    labeled = labeled.astype(np.int32)
    print(f"    Fragments after atom trimming: {n_labels}")

    if n_labels == 0:
        print("    WARNING: Seed sphere entirely occupied by RNA atoms. "
              "Try increasing SEED_SPHERE_RADIUS_A.")
        return np.zeros_like(trimmed_mask), 0, 0, 'none'

    if n_labels == 1:
        pocket_mask = trimmed_mask.copy()
        n_pocket    = int(pocket_mask.sum())
        print(f"    Single fragment: {n_pocket:,} voxels (no selection needed)")
        return pocket_mask, n_pocket, 1, 'centre_voxel'

    iz_c, iy_c, ix_c = centre_vox
    nz, ny, nx = trimmed_mask.shape
    iz_c = int(np.clip(iz_c, 0, nz - 1))
    iy_c = int(np.clip(iy_c, 0, ny - 1))
    ix_c = int(np.clip(ix_c, 0, nx - 1))

    centre_label = int(labeled[iz_c, iy_c, ix_c])
    if centre_label != 0:
        pocket_mask = (labeled == centre_label)
        n_pocket    = int(pocket_mask.sum())
        print(f"    Centre voxel ({iz_c},{iy_c},{ix_c}) → label={centre_label}  "
              f"pocket={n_pocket:,} voxels  method=centre_voxel")
        return pocket_mask, n_pocket, n_labels, 'centre_voxel'

    print(f"    Centre voxel ({iz_c},{iy_c},{ix_c}) was trimmed by an atom. "
          "Using nearest-centroid fragment.")
    best_dist  = np.inf
    best_label = -1
    for sid in range(1, n_labels + 1):
        frag_mask = labeled == sid
        zi_f, yi_f, xi_f = np.where(frag_mask)
        cx_f = origin[0] + xi_f.mean() * vs[0]
        cy_f = origin[1] + yi_f.mean() * vs[1]
        cz_f = origin[2] + zi_f.mean() * vs[2]
        dist = float(np.sqrt((cx_f - centre_a[0])**2 +
                             (cy_f - centre_a[1])**2 +
                             (cz_f - centre_a[2])**2))
        if dist < best_dist:
            best_dist  = dist
            best_label = sid

    pocket_mask = (labeled == best_label)
    n_pocket    = int(pocket_mask.sum())
    print(f"    Nearest-centroid fragment: label={best_label}  "
          f"pocket={n_pocket:,} voxels  "
          f"dist_to_centre={best_dist:.2f} A  method=nearest_centroid")
    return pocket_mask, n_pocket, n_labels, 'nearest_centroid'


# ─────────────────────────────────────────────────────────────────────────────
# PER-STRUCTURE PIPELINE (MODIFIED for single-PDB, explicit pdb_file/analysis_dir)
# ─────────────────────────────────────────────────────────────────────────────

def process_structure(name, pdb_path, analysis_dir):
    """
    Run the full pocket-volume pipeline for one PDB.

    Parameters
    ----------
    name         : str  — PDB identifier
    pdb_path     : str  — path to the <name>.pdb structure file
    analysis_dir : str  — folder holding the marker MRCs (from Script4) and
                          where all Script5 outputs are written
                          (Analysis_Pipeline2_<PDB_ID>)
    """
    print(f"\n{'='*60}")
    print(f"  Processing pocket volumes: {name}")
    print(f"{'='*60}")

    if not os.path.isfile(pdb_path):
        print(f"  [SKIP] PDB not found: {pdb_path}")
        return

    if not os.path.isdir(analysis_dir):
        print(f"  [SKIP] Analysis directory not found: {analysis_dir}")
        return

    # Collect marker MRC files
    marker_files = []
    for site_n in range(1, MAX_SITES + 1):
        mrc_name = f"{name}.HBond_Site_{site_n}_marker.mrc"
        mrc_path = os.path.join(analysis_dir, mrc_name)
        if os.path.isfile(mrc_path):
            marker_files.append((site_n, mrc_path))
        else:
            print(f"  [INFO] Marker file not found (site {site_n}): {mrc_name}")

    if not marker_files:
        print(f"  [SKIP] No marker MRC files found in {analysis_dir}")
        return

    print(f"  Found {len(marker_files)} marker file(s): "
          f"{[os.path.basename(m) for _, m in marker_files]}")

    # Load heavy atoms from PDB (shared across all sites for this structure)
    print(f"\n  Loading PDB heavy atoms from {pdb_path} ...")
    try:
        atom_coords = parse_pdb_heavy_atoms(pdb_path)
    except ValueError as e:
        print(f"  [SKIP] PDB parse error: {e}")
        return

    vox_vol_A3 = None

    summary_lines = [
        f"HBond Pocket Volumes — {name}",
        f"PDB             : {pdb_path}",
        f"Heavy atoms     : {len(atom_coords):,}",
        f"Seed radius     : {SEED_SPHERE_RADIUS_A:.1f} A",
        f"Atom excl. rad. : {ATOM_EXCLUSION_RADIUS_A:.1f} A",
        f"Connectivity    : {POCKET_CONNECTIVITY} (26-conn)",
        "",
        f"{'Site':<18} {'Seed vox':>10} {'Excl vox':>10} "
        f"{'Pocket vox':>12} {'Volume (A^3)':>14} "
        f"{'Frags':>6} {'Method':<20}  Output MRC",
    ]

    for site_n, marker_path in marker_files:
        site_label   = f"HBond_Site_{site_n}"
        out_mrc_name = f"{name}.{site_label}_pocket_volume.mrc"
        out_mrc_path = os.path.join(analysis_dir, out_mrc_name)

        print(f"\n  --- {site_label} ---")
        print(f"  Marker : {os.path.basename(marker_path)}")

        # Step 1-2: Load marker, find centre
        print("    Step 1-2: Loading marker MRC and finding centre ...")
        marker_data, vs, origin = load_mrc(marker_path)
        centre_a, centre_vox   = marker_centre_angstrom(marker_data, vs, origin)
        print(f"    Marker centre: ({centre_a[0]:.3f}, {centre_a[1]:.3f}, "
              f"{centre_a[2]:.3f}) A  "
              f"nearest voxel: ({centre_vox[0]},{centre_vox[1]},{centre_vox[2]})")

        if vox_vol_A3 is None:
            vox_vol_A3 = vs[0] * vs[1] * vs[2]

        # Step 3: Seed sphere
        print(f"    Step 3: Building seed sphere (r={SEED_SPHERE_RADIUS_A} A) ...")
        sphere_mask = build_seed_sphere(centre_a, vs, origin,
                                        grid_shape=marker_data.shape,
                                        radius_a=SEED_SPHERE_RADIUS_A)
        n_seed = int(sphere_mask.sum())

        # Step 4: Trim by RNA atoms
        print(f"    Step 4: Trimming by {len(atom_coords):,} heavy atoms "
              f"(excl_r={ATOM_EXCLUSION_RADIUS_A} A) ...")
        sphere_mask, n_excluded = trim_sphere_by_atoms(
            sphere_mask, atom_coords, vs, origin,
            excl_radius_a=ATOM_EXCLUSION_RADIUS_A)
        n_after_trim = int(sphere_mask.sum())
        print(f"    After trimming: {n_after_trim:,} voxels remain  "
              f"({n_excluded:,} excluded by atoms)")

        if n_after_trim == 0:
            print(f"    WARNING: All seed sphere voxels excluded by atoms "
                  f"for {site_label}. "
                  "Try increasing SEED_SPHERE_RADIUS_A or "
                  "decreasing ATOM_EXCLUSION_RADIUS_A.")
            summary_lines.append(
                f"{site_label:<18} {n_seed:>10,} {n_excluded:>10,} "
                f"{'0':>12} {'0.00':>14} "
                f"{'0':>6} {'empty—all excluded':<20}  {out_mrc_name}"
            )
            continue

        # Steps 5-6: Fragment labelling and selection
        print("    Steps 5-6: Labelling fragments and selecting pocket piece ...")
        pocket_mask, n_pocket, n_frags, method = extract_pocket_fragment(
            sphere_mask, centre_a, centre_vox, vs, origin,
            connectivity=POCKET_CONNECTIVITY)

        # Step 7: Write pocket volume MRC
        pocket_data = pocket_mask.astype(np.float32)
        save_mrc(out_mrc_path, pocket_data, vs, origin)

        vol_A3      = n_pocket * vox_vol_A3
        pct_of_seed = 100.0 * n_pocket / n_seed if n_seed > 0 else 0.0

        print(f"    {site_label}: seed={n_seed:,}  excluded={n_excluded:,}  "
              f"pocket={n_pocket:,} vox  "
              f"vol={vol_A3:.2f} A^3  ({pct_of_seed:.1f}% of seed)  "
              f"frags={n_frags}  method={method}")

        summary_lines.append(
            f"{site_label:<18} {n_seed:>10,} {n_excluded:>10,} "
            f"{n_pocket:>12,} {vol_A3:>14.2f} "
            f"{n_frags:>6} {method:<20}  {out_mrc_name}"
        )

    # Write summary into the analysis directory
    summary_path = os.path.join(analysis_dir, f"{name}.pocket_volume_summary.txt")
    with open(summary_path, 'w') as fh:
        fh.write('\n'.join(summary_lines) + '\n')
    print(f"\n  Summary written: {os.path.basename(summary_path)}")
    print(f"  {name} complete.")


# =====================================================================
# SINGLE-PDB DRIVER (generalized — replaces the old multi-PDB discovery loop)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Script5 (Pipeline 2): Making hydrogen-bond pocket volumes "
                    "from HBond site markers, for a single PDB."
    )
    parser.add_argument(
        "--pdb_file",
        required=True,
        help="Path to the fixed PDB structure file (e.g. 1AJU_fixed.pdb) "
             "in the current working directory.",
    )
    parser.add_argument(
        "--analysis_dir",
        required=True,
        help="Path to Analysis_Pipeline2_<PDB_ID> folder (output of Script4). "
             "Marker MRCs are read from here and all Script5 outputs are "
             "saved here.",
    )
    parser.add_argument(
        "--pdb_id",
        default=None,
        help="Optional override for the PDB identifier used in filenames. "
             "If not provided, it is automatically derived from --pdb_file "
             "(filename without extension).",
    )
    args = parser.parse_args()

    pdb_file_path = Path(args.pdb_file)
    name = args.pdb_id if args.pdb_id else pdb_file_path.stem

    analysis_dir = args.analysis_dir

    print("Script5_Pipeline2_Making_Hydrogen_Bond_Pocket_Volume.py")
    print(f"  PDB ID       : {name}")
    print(f"  PDB file     : {args.pdb_file}")
    print(f"  Analysis dir : {analysis_dir}")
    print(f"\nActive parameters:")
    print(f"  SEED_SPHERE_RADIUS_A    : {SEED_SPHERE_RADIUS_A} A")
    print(f"  ATOM_EXCLUSION_RADIUS_A : {ATOM_EXCLUSION_RADIUS_A} A")
    print(f"  POCKET_CONNECTIVITY     : {POCKET_CONNECTIVITY}  (26-conn)")
    print(f"  MAX_SITES               : {MAX_SITES}")

    process_structure(name, args.pdb_file, analysis_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
