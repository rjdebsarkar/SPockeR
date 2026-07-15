"""
─────────────────────────────────────────────────────────────────────────────
Input MRC files (inside --fields_dir):
  <name>.apbs.mrc          — ELE field
  <name>.hbacceptors.mrc   — HBA field
  <name>.hbdonors.mrc      — HBD field

Outputs written to --analysis_dir:
    <name>.ele_isovalue.mrc
    <name>.HBond_Site_1_marker.mrc
    <name>.HBond_Site_1.mrc
    <name>.HBond_Site_2_marker.mrc   (if a second site is found)
    <name>.HBond_Site_2.mrc
    <name>.hbond_sites_summary.txt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import mrcfile
from scipy.ndimage import label, binary_dilation
from scipy.interpolate import RegularGridInterpolator

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

# ── ELE isovalue ──────────────────────────────────────────────────────────────
ELE_CONSTANT = -2.0

# ── HBA/HBD overlap-detection threshold ──────────────────────────────────────
HB_INTENSITY_PERCENTILE = 50

# ── Triple-overlap dilation (voxels) ─────────────────────────────────────────
OVERLAP_EXPAND_VOXELS = 1

# ── Minimum centroid-to-centroid separation between sites (Å) ────────────────
MIN_SITE_SEPARATION_A = 10.0

# ── Marker sphere radius (voxels) ────────────────────────────────────────────
MARKER_RADIUS_VOXELS = 3

# ── Overlap clustering connectivity ──────────────────────────────────────────
CLUSTER_CONNECTIVITY = 2

# ── Maximum number of HBond Sites to output ──────────────────────────────────
MAX_SITES = 2

# ── Minimum overlap core size (voxels) ───────────────────────────────────────
MIN_SITE_VOXELS = 10


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
                               int(mrc.header.nzstart) * vs[2]], dtype=np.float64)
        nz, ny, nx = data.shape
        print(f"    {os.path.basename(path)}: shape=({nz},{ny},{nx})  "
              f"voxel={vs[0]:.3f} A  "
              f"origin=({origin[0]:.2f},{origin[1]:.2f},{origin[2]:.2f})  "
              f"nonzero={np.count_nonzero(data):,}/{data.size:,}  "
              f"min={data.min():.4f}  max={data.max():.4f}")
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
# GRID UTILITIES (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def make_axes(data, vs, origin):
    nz, ny, nx = data.shape
    xa = origin[0] + np.arange(nx, dtype=np.float64) * vs[0]
    ya = origin[1] + np.arange(ny, dtype=np.float64) * vs[1]
    za = origin[2] + np.arange(nz, dtype=np.float64) * vs[2]
    return xa, ya, za


def resample_to_master(src_data, src_vs, src_origin,
                       ref_data, ref_vs, ref_origin):
    nz_r, ny_r, nx_r = ref_data.shape
    nz_s, ny_s, nx_s = src_data.shape
    same = (nz_r == nz_s and ny_r == ny_s and nx_r == nx_s and
            np.allclose(ref_origin, src_origin, atol=1e-3) and
            np.allclose(ref_vs, src_vs, atol=1e-4))
    if same:
        return src_data.copy()
    xa_r, ya_r, za_r = make_axes(ref_data, ref_vs, ref_origin)
    xa_s, ya_s, za_s = make_axes(src_data, src_vs, src_origin)
    interp = RegularGridInterpolator(
        (za_s, ya_s, xa_s),
        src_data.astype(np.float64),
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    zz, yy, xx = np.meshgrid(za_r, ya_r, xa_r, indexing='ij')
    pts    = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    result = interp(pts).reshape(nz_r, ny_r, nx_r).astype(np.float32)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FIELD THRESHOLDING (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ele_isovalue(ele_data, constant=ELE_CONSTANT):
    nonzero_vals = ele_data[ele_data != 0.0]
    if len(nonzero_vals) == 0:
        return constant, 0.0, 0.0
    min_val  = float(nonzero_vals.min())
    max_val  = float(nonzero_vals.max())
    isovalue = (min_val + max_val) / 2.0 + constant
    print(f"    ELE distribution: min={min_val:.3f}  max={max_val:.3f}")
    print(f"    ELE isovalue = ({min_val:.3f} + {max_val:.3f}) / 2 + "
          f"({constant}) = {isovalue:.3f}")
    return isovalue, min_val, max_val


def threshold_ele(ele_data, isovalue):
    mask       = ele_data <= isovalue
    thres_data = np.where(mask, ele_data, 0.0).astype(np.float32)
    print(f"    ELE mask (value <= {isovalue:.3f}): {mask.sum():,} voxels pass")
    return mask, thres_data


def threshold_hb_field(data, percentile=HB_INTENSITY_PERCENTILE):
    nonzero_vals = data[data != 0.0]
    if len(nonzero_vals) == 0:
        return np.zeros(data.shape, dtype=bool)
    thr  = np.percentile(np.abs(nonzero_vals), percentile)
    mask = np.abs(data) >= thr
    print(f"    HB threshold (|v| >= p{percentile} = {thr:.5f}): "
          f"{mask.sum():,} voxels pass")
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# MASK DILATION (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def dilate_mask(mask_3d, expand_voxels):
    if expand_voxels <= 0:
        return mask_3d.copy()
    struct = np.ones((2 * expand_voxels + 1,) * 3, dtype=bool)
    return binary_dilation(mask_3d, structure=struct)


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTIVITY STRUCTURE BUILDER (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def make_conn_struct(connectivity):
    if connectivity == 1:
        return np.array([[[0,0,0],[0,1,0],[0,0,0]],
                         [[0,1,0],[1,1,1],[0,1,0]],
                         [[0,0,0],[0,1,0],[0,0,0]]], dtype=int)
    elif connectivity == 2:
        return np.array([[[0,1,0],[1,1,1],[0,1,0]],
                         [[1,1,1],[1,1,1],[1,1,1]],
                         [[0,1,0],[1,1,1],[0,1,0]]], dtype=int)
    else:
        return np.ones((3, 3, 3), dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# CENTROID HELPER (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def mask_centroid_angstrom(bool_mask, vs, origin):
    zi, yi, xi = np.where(bool_mask)
    return np.array([origin[0] + xi.mean() * vs[0],
                     origin[1] + yi.mean() * vs[1],
                     origin[2] + zi.mean() * vs[2]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# MARKER MRC GENERATOR (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def make_marker_mrc(centroid_a, vs, origin, grid_shape,
                    radius_voxels=MARKER_RADIUS_VOXELS):
    nz, ny, nx = grid_shape
    ci_x = (centroid_a[0] - origin[0]) / vs[0]
    ci_y = (centroid_a[1] - origin[1]) / vs[1]
    ci_z = (centroid_a[2] - origin[2]) / vs[2]
    iz, iy, ix = np.mgrid[0:nz, 0:ny, 0:nx]
    dist2 = ((ix - ci_x) ** 2 +
             (iy - ci_y) ** 2 +
             (iz - ci_z) ** 2)
    marker_data = np.where(dist2 <= radius_voxels ** 2,
                           1.0, 0.0).astype(np.float32)
    n_sphere = int(marker_data.sum())
    print(f"    Marker sphere: centroid_vox=({ci_x:.1f},{ci_y:.1f},{ci_z:.1f})  "
          f"radius={radius_voxels} vox  filled={n_sphere:,} voxels")
    return marker_data


# ─────────────────────────────────────────────────────────────────────────────
# POCKET DETECTION: TRIPLE OVERLAP + SPATIALLY SEPARATED CLUSTERING (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def build_full_overlap_mask(ele_mask, hba_mask, hbd_mask,
                            expand=OVERLAP_EXPAND_VOXELS):
    d_ele = dilate_mask(ele_mask, expand)
    d_hba = dilate_mask(hba_mask, expand)
    d_hbd = dilate_mask(hbd_mask, expand)
    return d_ele & d_hba & d_hbd & ele_mask


def find_overlap_clusters(overlap_mask, vs, origin,
                          min_voxels=MIN_SITE_VOXELS,
                          connectivity=CLUSTER_CONNECTIVITY,
                          max_sites=MAX_SITES,
                          min_separation_a=MIN_SITE_SEPARATION_A):
    n_overlap = int(overlap_mask.sum())
    print(f"    Triple-field overlap voxels (ELE-anchored): {n_overlap:,}")

    if n_overlap == 0:
        print("    WARNING: No triple overlap found. "
              "Try increasing OVERLAP_EXPAND_VOXELS or "
              "lowering HB_INTENSITY_PERCENTILE.")
        return [], np.zeros_like(overlap_mask, dtype=np.int32)

    struct = make_conn_struct(connectivity)
    labeled_overlap, n_labels = label(overlap_mask, structure=struct)
    labeled_overlap = labeled_overlap.astype(np.int32)
    print(f"    Connected overlap components found: {n_labels}")

    candidates = []
    for sid in range(1, n_labels + 1):
        core_mask_i = labeled_overlap == sid
        nvox        = int(core_mask_i.sum())
        if nvox >= min_voxels:
            centroid = mask_centroid_angstrom(core_mask_i, vs, origin)
            candidates.append((nvox, core_mask_i, centroid, sid))

    candidates.sort(key=lambda x: -x[0])
    print(f"    Valid candidates (>= {min_voxels} voxels): {len(candidates)}")

    if not candidates:
        return [], labeled_overlap

    selected = [candidates[0]]
    print(f"    Site 1 selected: {candidates[0][0]:,} voxels  "
          f"label={candidates[0][3]}  "
          f"centroid=({candidates[0][2][0]:.1f}, "
          f"{candidates[0][2][1]:.1f}, "
          f"{candidates[0][2][2]:.1f}) A")

    if max_sites >= 2 and len(candidates) > 1:
        c1          = selected[0][2]
        found_site2 = False
        for nvox_j, mask_j, centroid_j, label_j in candidates[1:]:
            dist = float(np.linalg.norm(centroid_j - c1))
            if dist >= min_separation_a:
                selected.append((nvox_j, mask_j, centroid_j, label_j))
                print(f"    Site 2 selected: {nvox_j:,} voxels  "
                      f"label={label_j}  "
                      f"centroid=({centroid_j[0]:.1f}, "
                      f"{centroid_j[1]:.1f}, "
                      f"{centroid_j[2]:.1f}) A  "
                      f"separation from Site 1 = {dist:.1f} A")
                found_site2 = True
                break
        if not found_site2:
            print(f"    WARNING: No Site 2 found with separation "
                  f">= {min_separation_a} A from Site 1. "
                  f"Only Site 1 will be output. "
                  f"Lower MIN_SITE_SEPARATION_A if a second site is needed.")

    return selected, labeled_overlap


# ─────────────────────────────────────────────────────────────────────────────
# SITE PATCH: SINGLE CONNECTED TRIPLE-OVERLAP COMPONENT (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def extract_site_patch(overlap_label_id, labeled_overlap,
                       ele_thres, used_overlap_labels):
    if overlap_label_id in used_overlap_labels:
        print(f"    Overlap label {overlap_label_id} already claimed by a "
              "previous site. Skipping.")
        return np.zeros_like(ele_thres), 0, False

    component_mask = labeled_overlap == overlap_label_id
    nvox           = int(component_mask.sum())

    if nvox == 0:
        print(f"    WARNING: Overlap label {overlap_label_id} has 0 voxels.")
        return np.zeros_like(ele_thres), 0, False

    patch_data = np.where(component_mask, ele_thres, 0.0).astype(np.float32)
    used_overlap_labels.add(overlap_label_id)
    print(f"    Site patch: overlap label={overlap_label_id}  "
          f"voxels={nvox:,}  "
          f"(single connected triple-overlap component, ELE values)")
    return patch_data, nvox, True


# ─────────────────────────────────────────────────────────────────────────────
# PER-STRUCTURE PIPELINE (MODIFIED for single-PDB, separate fields/analysis dirs)
# ─────────────────────────────────────────────────────────────────────────────

def process_structure(name, fields_dir, analysis_dir):
    """
    Run the full pipeline for one PDB.

    Parameters
    ----------
    name         : str  — PDB identifier
    fields_dir   : str  — folder containing input .mrc field files
                          (Fields_Pipeline2_<PDB_ID>)
    analysis_dir : str  — folder where all outputs are written
                          (Analysis_Pipeline1_<PDB_ID>)
    """
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"{'='*60}")

    ele_file = os.path.join(fields_dir, f"{name}.apbs.mrc")
    hba_file = os.path.join(fields_dir, f"{name}.hbacceptors.mrc")
    hbd_file = os.path.join(fields_dir, f"{name}.hbdonors.mrc")

    for f in [ele_file, hba_file, hbd_file]:
        if not os.path.isfile(f):
            print(f"  [SKIP] Missing file: {f}")
            return

    out_subdir = analysis_dir
    os.makedirs(out_subdir, exist_ok=True)
    print(f"  Input directory : {fields_dir}")
    print(f"  Output directory: {out_subdir}")

    # ── Step 1: Load all three MRC fields ────────────────────────────────────
    print("\n  Step 1: Loading MRC fields ...")
    ele_data, ele_vs, ele_origin = load_mrc(ele_file)
    hba_data, hba_vs, hba_origin = load_mrc(hba_file)
    hbd_data, hbd_vs, hbd_origin = load_mrc(hbd_file)

    # ── Step 2: ELE isovalue and thresholding ─────────────────────────────────
    print("\n  Step 2: ELE isovalue and thresholding ...")
    isovalue, ele_min, ele_max = compute_ele_isovalue(ele_data, ELE_CONSTANT)
    ele_mask, ele_thres        = threshold_ele(ele_data, isovalue)
    ele_out_path = os.path.join(out_subdir, f"{name}.ele_isovalue.mrc")
    save_mrc(ele_out_path, ele_thres, ele_vs, ele_origin)

    # ── Step 3: Resample HBA and HBD onto ELE master grid ────────────────────
    print("\n  Step 3: Resampling HBA and HBD onto ELE master grid ...")
    hba_on_ele = resample_to_master(hba_data, hba_vs, hba_origin,
                                    ele_data, ele_vs, ele_origin)
    hbd_on_ele = resample_to_master(hbd_data, hbd_vs, hbd_origin,
                                    ele_data, ele_vs, ele_origin)
    print(f"    HBA resampled: nonzero={np.count_nonzero(hba_on_ele):,}")
    print(f"    HBD resampled: nonzero={np.count_nonzero(hbd_on_ele):,}")

    # ── Step 4: Threshold HBA and HBD (overlap detection only) ───────────────
    print("\n  Step 4: Thresholding HBA and HBD (overlap detection only) ...")
    hba_mask = threshold_hb_field(hba_on_ele)
    hbd_mask = threshold_hb_field(hbd_on_ele)
    if hba_mask.sum() == 0 or hbd_mask.sum() == 0:
        print("  WARNING: HBA or HBD mask empty after thresholding. "
              "Lower HB_INTENSITY_PERCENTILE.")
        return

    # ── Step 5: Build full triple-overlap mask ────────────────────────────────
    print(f"\n  Step 5: Building full triple-overlap mask "
          f"(OVERLAP_EXPAND_VOXELS={OVERLAP_EXPAND_VOXELS}) ...")
    overlap_mask = build_full_overlap_mask(ele_mask, hba_mask, hbd_mask,
                                           expand=OVERLAP_EXPAND_VOXELS)

    # ── Step 6: Cluster overlap; select top-2 spatially separated cores ───────
    print(f"\n  Step 6: Clustering overlap; selecting sites "
          f"(min separation {MIN_SITE_SEPARATION_A} A) ...")
    clusters, labeled_overlap = find_overlap_clusters(
        overlap_mask, vs=ele_vs, origin=ele_origin)

    if not clusters:
        print(f"  No HBond pocket sites found for {name}.")
        summary_path = os.path.join(out_subdir, f"{name}.hbond_sites_summary.txt")
        with open(summary_path, 'w') as fh:
            fh.write(f"HBond Pocket Sites — {name}\n")
            fh.write(f"ELE isovalue : {isovalue:.3f}  "
                     f"(distribution: {ele_min:.3f} to {ele_max:.3f}, "
                     f"c = {ELE_CONSTANT})\n")
            fh.write("No pocket sites found.\n")
        return

    # ── Step 7: Marker MRCs + site patch MRCs ────────────────────────────────
    print(f"\n  Step 7: Writing marker MRCs and site patch MRCs "
          f"for {len(clusters)} site(s) ...")

    vox_vol_A3          = ele_vs[0] * ele_vs[1] * ele_vs[2]
    used_overlap_labels = set()

    summary_lines = [
        f"HBond Pocket Sites — {name}",
        f"ELE isovalue      : {isovalue:.4f}  "
        f"(distribution: {ele_min:.4f} to {ele_max:.4f}, c = {ELE_CONSTANT})",
        f"Voxel volume      : {vox_vol_A3:.4f} A^3  "
        f"({ele_vs[0]:.3f} x {ele_vs[1]:.3f} x {ele_vs[2]:.3f} A)",
        f"Total sites       : {len(clusters)}  (max allowed: {MAX_SITES})",
        f"Min site sep.     : {MIN_SITE_SEPARATION_A} A (centroid-centroid)",
        f"Overlap expand    : {OVERLAP_EXPAND_VOXELS} voxels dilation",
        f"Marker radius     : {MARKER_RADIUS_VOXELS} voxels",
        f"Site patch        : single connected triple-overlap component "
        f"(ELE field values)",
        "",
        f"{'Site':<14} {'Core vox':>10} {'Patch vox':>10} "
        f"{'Volume (A^3)':>14}  {'Centroid (A)':>30}  MRC files",
    ]

    for rank, (core_nvox, core_mask, centroid, olap_lbl) in             enumerate(clusters, start=1):

        site_name       = f"HBond_Site_{rank}"
        marker_mrc_name = f"{name}.{site_name}_marker.mrc"
        patch_mrc_name  = f"{name}.{site_name}.mrc"
        marker_mrc_path = os.path.join(out_subdir, marker_mrc_name)
        patch_mrc_path  = os.path.join(out_subdir, patch_mrc_name)

        print(f"\n    --- {site_name}  "
              f"(core={core_nvox:,} vox, overlap_label={olap_lbl}, "
              f"centroid=({centroid[0]:.1f},{centroid[1]:.1f},"
              f"{centroid[2]:.1f}) A) ---")

        # 7a: Write spherical marker MRC at site centroid
        marker_data = make_marker_mrc(centroid, ele_vs, ele_origin,
                                      grid_shape=ele_data.shape,
                                      radius_voxels=MARKER_RADIUS_VOXELS)
        save_mrc(marker_mrc_path, marker_data, ele_vs, ele_origin)

        # 7b: Extract site patch = single connected overlap component
        patch_data, nvox_patch, found = extract_site_patch(
            olap_lbl, labeled_overlap, ele_thres, used_overlap_labels)
        save_mrc(patch_mrc_path, patch_data, ele_vs, ele_origin)

        vol_A3       = nvox_patch * vox_vol_A3
        centroid_str = (f"({centroid[0]:.1f}, {centroid[1]:.1f}, "
                        f"{centroid[2]:.1f})")
        summary_lines.append(
            f"{site_name:<14} {core_nvox:>10,} {nvox_patch:>10,} "
            f"{vol_A3:>14.2f}  {centroid_str:>30}  "
            f"{patch_mrc_name}  |  {marker_mrc_name}"
        )
        print(f"    {site_name}: core={core_nvox:,}  patch={nvox_patch:,} vox  "
              f"vol={vol_A3:.2f} A^3  centroid={centroid_str}  "
              f"->  {patch_mrc_name}  +  {marker_mrc_name}")

    summary_path = os.path.join(out_subdir, f"{name}.hbond_sites_summary.txt")
    with open(summary_path, 'w') as fh:
        fh.write('\n'.join(summary_lines) + '\n')
    print(f"\n  Summary written: {os.path.basename(summary_path)}")
    print(f"\n  {name} complete.")


# =====================================================================
# SINGLE-PDB DRIVER (generalized — replaces the old multi-PDB discovery loop)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Script4 (Pipeline 2): Hydrogen-bond pocket hotspots "
                    "using HBA/HBD/ELE fields, for a single PDB."
    )
    parser.add_argument(
        "--fields_dir",
        required=True,
        help="Path to the Fields_Pipeline2_<PDB_ID> folder containing "
             "<name>.apbs.mrc, <name>.hbacceptors.mrc, <name>.hbdonors.mrc.",
    )
    parser.add_argument(
        "--analysis_dir",
        required=True,
        help="Path to Analysis_Pipeline2_<PDB_ID> folder. All Script4 outputs "
             "(ele_isovalue.mrc, HBond_Site_*.mrc, summary .txt) are saved here, "
             "alongside the existing Pipeline 1 results.",
    )
    parser.add_argument(
        "--pdb_id",
        default=None,
        help="Optional override for the PDB identifier used in filenames. "
             "If not provided, it is automatically derived from the "
             "*.apbs.mrc filename found in --fields_dir.",
    )
    args = parser.parse_args()

    fields_dir = args.fields_dir
    analysis_dir = args.analysis_dir

    if args.pdb_id:
        name = args.pdb_id
    else:
        apbs_candidates = sorted(Path(fields_dir).glob("*.apbs.mrc"))
        if not apbs_candidates:
            print(f"\n[ERROR] No *.apbs.mrc file found in: {fields_dir}")
            print("  Provide --pdb_id explicitly, or ensure "
                  "<name>.apbs.mrc exists in --fields_dir.")
            raise SystemExit(1)
        name = apbs_candidates[0].name[:-len(".apbs.mrc")]

    print("Script4_Pipeline2_Hydrogen_Bond_Pocket_Hotspots_Using_HBA_HBD_ELE_Fields.py")
    print(f"  PDB ID       : {name}")
    print(f"  Fields dir   : {fields_dir}")
    print(f"  Analysis dir : {analysis_dir}")
    print(f"\nActive parameters:")
    print(f"  ELE_CONSTANT            : {ELE_CONSTANT}  "
          f"(isovalue = (min+max)/2 + c)")
    print(f"  HB_INTENSITY_PERCENTILE : {HB_INTENSITY_PERCENTILE}  "
          f"(overlap detection only)")
    print(f"  OVERLAP_EXPAND_VOXELS   : {OVERLAP_EXPAND_VOXELS}  "
          f"(dilation for triple-overlap detection)")
    print(f"  MIN_SITE_SEPARATION_A   : {MIN_SITE_SEPARATION_A} A  "
          f"(min centroid-centroid distance between sites)")
    print(f"  MARKER_RADIUS_VOXELS    : {MARKER_RADIUS_VOXELS}  "
          f"(sphere radius of centroid marker MRC)")
    print(f"  CLUSTER_CONNECTIVITY    : {CLUSTER_CONNECTIVITY}  "
          f"(connectivity for overlap clustering)")
    print(f"  MAX_SITES               : {MAX_SITES}")
    print(f"  MIN_SITE_VOXELS         : {MIN_SITE_VOXELS}  "
          f"(discard overlap cores smaller than this)")

    process_structure(name, fields_dir, analysis_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
