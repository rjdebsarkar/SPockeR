import argparse
from pathlib import Path
import shlex

import mrcfile
import numpy as np
from scipy import ndimage
from scipy.ndimage import label as scipy_label

try:
    import MDAnalysis as mda
except ImportError:
    mda = None


POCKET_SPECS = [
    {
        "tag":    "stk_hpb_first",
        "suffix": ".stacking_hydrophobic_pocket.mrc",
        "label":  "first STK-HPb pocket",
    },
    {
        "tag":    "stk_hpb_second",
        "suffix": ".stacking_hydrophobic_second_pocket.mrc",
        "label":  "second STK-HPb pocket",
    },
    {
        "tag":    "stk_ele",
        "suffix": ".stacking_electrostatic_pocket.mrc",
        "label":  "STK-ELE pocket",
    },
    {
        "tag":    "mixed_fields",
        "suffix": ".mixed_fields_pocket.mrc",
        "label":  "mixed fields pocket (STK+HPb+ELE common overlap)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────
BINARY_EPS = 1e-12

PATCH_CONNECTIVITY  = 2
POCKET_CONNECTIVITY = 3

MARKER_DILATION_ITERS = 1

SEED_SPHERE_RADIUS_A    = 8.0
ATOM_EXCLUSION_RADIUS_A = 3.0

CENTER_CORE_PERCENTILE = 95.0
MAX_PATCH_SAMPLES      = 4000
CENTER_CUBE_RADIUS_VOX = 1

DNA_RESNAMES = {
    "DA", "DC", "DG", "DT", "DI",
    "ADE", "CYT", "GUA", "THY", "URI", "URA"
}

RNA_RESNAMES = {
    "A", "C", "G", "U", "I",
    "RA", "RC", "RG", "RU", "RI",
    "ADE", "CYT", "GUA", "URA", "URI",
    "PSU", "H2U", "5MU", "5MC", "OMC", "OMG", "1MA", "2MA",
    "M2G", "1MG", "7MG", "YG", "YYG", "G7M", "A2M", "4SU",
    "OMU", "MIA", "5BU", "2MG"
}


# ─────────────────────────────────────────────────────────────────────────────
# HYDROGEN FILTER  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def is_hydrogen_name(atomname: str) -> bool:
    name = atomname.strip().upper().replace("*", "'")
    stripped = name.lstrip("0123456789")
    if not stripped:
        return False
    return stripped[0] in ("H", "D")


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def voxel_center_xyz(idx, voxel_size_zyx, origin_xyz):
    arr = np.asarray(idx, dtype=float)
    sz, sy, sx = voxel_size_zyx[0], voxel_size_zyx[1], voxel_size_zyx[2]
    oz, oy, ox = origin_xyz[2],     origin_xyz[1],     origin_xyz[0]

    if arr.ndim == 1:
        z_a = (arr[0] + 0.5) * sz + oz
        y_a = (arr[1] + 0.5) * sy + oy
        x_a = (arr[2] + 0.5) * sx + ox
        return np.array([x_a, y_a, z_a], dtype=float)

    z_a = (arr[:, 0] + 0.5) * sz + oz
    y_a = (arr[:, 1] + 0.5) * sy + oy
    x_a = (arr[:, 2] + 0.5) * sx + ox
    return np.column_stack([x_a, y_a, z_a])


def xyz_to_voxel_idx_zyx(xyz, voxel_size_zyx, origin_xyz):
    sz, sy, sx = voxel_size_zyx[0], voxel_size_zyx[1], voxel_size_zyx[2]
    oz, oy, ox = origin_xyz[2],     origin_xyz[1],     origin_xyz[0]
    x, y, z    = float(xyz[0]),     float(xyz[1]),     float(xyz[2])
    iz_f = (z - oz) / sz - 0.5
    iy_f = (y - oy) / sy - 0.5
    ix_f = (x - ox) / sx - 0.5
    return iz_f, iy_f, ix_f


# ─────────────────────────────────────────────────────────────────────────────
# MRC I/O (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def load_mrc(path):
    with mrcfile.open(path, mode="r") as mrc:
        data = np.asarray(mrc.data, dtype=np.float32).copy()
        vx = float(mrc.voxel_size.x)
        vy = float(mrc.voxel_size.y)
        vz = float(mrc.voxel_size.z)
        voxel_size_xyz = np.array([vx, vy, vz], dtype=float)
        voxel_size_zyx = np.array([vz, vy, vx], dtype=float)
        try:
            origin_xyz = np.array(
                [float(mrc.header.origin.x),
                 float(mrc.header.origin.y),
                 float(mrc.header.origin.z)],
                dtype=float
            )
        except Exception:
            origin_xyz = np.array([0.0, 0.0, 0.0], dtype=float)
    return data, voxel_size_xyz, voxel_size_zyx, origin_xyz


def save_mask_as_mrc(mask, ref_path, out_path):
    with mrcfile.open(ref_path, mode="r") as ref:
        voxel_size_xyz = np.array(
            [float(ref.voxel_size.x),
             float(ref.voxel_size.y),
             float(ref.voxel_size.z)],
            dtype=float
        )
        try:
            origin_xyz = np.array(
                [float(ref.header.origin.x),
                 float(ref.header.origin.y),
                 float(ref.header.origin.z)],
                dtype=float
            )
        except Exception:
            origin_xyz = np.array([0.0, 0.0, 0.0], dtype=float)
        try:
            nxstart = int(ref.header.nxstart)
            nystart = int(ref.header.nystart)
            nzstart = int(ref.header.nzstart)
        except Exception:
            nxstart, nystart, nzstart = 0, 0, 0
        try:
            mapc = int(ref.header.mapc)
            mapr = int(ref.header.mapr)
            maps = int(ref.header.maps)
        except Exception:
            mapc, mapr, maps = 1, 2, 3

    out_data = mask.astype(np.float32)
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(out_data)
        mrc.voxel_size = tuple(voxel_size_xyz.tolist())
        try:
            mrc.header.origin.x = origin_xyz[0]
            mrc.header.origin.y = origin_xyz[1]
            mrc.header.origin.z = origin_xyz[2]
        except Exception:
            pass
        try:
            mrc.header.nxstart = nxstart
            mrc.header.nystart = nystart
            mrc.header.nzstart = nzstart
        except Exception:
            pass
        try:
            mrc.header.mapc = mapc
            mrc.header.mapr = mapr
            mrc.header.maps = maps
        except Exception:
            pass
        mrc.update_header_from_data()
        mrc.update_header_stats()


# ─────────────────────────────────────────────────────────────────────────────
# MASKS / COMPONENTS / CENTRE (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def connected_components(mask, conn=2):
    structure = ndimage.generate_binary_structure(3, conn)
    return ndimage.label(mask, structure=structure)


def choose_single_patch(mask, voxel_size_zyx, conn=2):
    labels, nlab    = connected_components(mask, conn=conn)
    voxel_volume_a3 = float(np.prod(voxel_size_zyx))
    if nlab == 0:
        return np.zeros_like(mask, dtype=bool), 0, 0, 0.0
    counts = np.bincount(labels.ravel())
    if nlab == 1:
        nvox = int(counts[1])
        return (labels == 1), 1, nvox, nvox * voxel_volume_a3
    sizes = [(lab, int(counts[lab])) for lab in range(1, nlab + 1)]
    largest_lab, largest_voxels = max(sizes, key=lambda x: x[1])
    chosen_mask      = (labels == largest_lab)
    chosen_volume_a3 = largest_voxels * voxel_volume_a3
    return chosen_mask, nlab, int(largest_voxels), float(chosen_volume_a3)


def choose_patch_seed_center(mask, voxel_size_zyx, origin_xyz,
                              core_percentile=95.0,
                              max_patch_samples=4000):
    idx = np.column_stack(np.where(mask))
    if len(idx) == 0:
        return None, None, None
    dist      = ndimage.distance_transform_edt(mask, sampling=voxel_size_zyx)
    dist_vals = dist[tuple(idx.T)]
    max_dist  = float(dist_vals.max())
    if max_dist <= 0.0:
        pts  = voxel_center_xyz(idx, voxel_size_zyx, origin_xyz)
        cog  = pts.mean(axis=0)
        d2   = np.sum((pts - cog[None, :]) ** 2, axis=1)
        best = int(np.argmin(d2))
        return idx[best].astype(int), pts[best], 0.0
    cutoff        = float(np.percentile(dist_vals, core_percentile))
    candidate_idx = idx[dist_vals >= cutoff]
    if len(candidate_idx) == 0:
        best       = int(np.argmax(dist_vals))
        center_idx = idx[best].astype(int)
        center_xyz = voxel_center_xyz(center_idx, voxel_size_zyx, origin_xyz)
        return center_idx, center_xyz, float(dist_vals[best])
    patch_pts = voxel_center_xyz(idx, voxel_size_zyx, origin_xyz)
    if len(patch_pts) > max_patch_samples:
        sel       = np.linspace(0, len(patch_pts) - 1, max_patch_samples).astype(int)
        patch_pts = patch_pts[sel]
    cand_pts = voxel_center_xyz(candidate_idx, voxel_size_zyx, origin_xyz)
    d2       = np.sum((cand_pts[:, None, :] - patch_pts[None, :, :]) ** 2, axis=2)
    mean_d2  = d2.mean(axis=1)
    cand_clearance = dist[tuple(candidate_idx.T)]
    max_clear      = float(cand_clearance.max())
    penalty        = (max_clear - cand_clearance) / (max_clear + 1e-12)
    scores         = mean_d2 + penalty * (np.min(voxel_size_zyx) ** 2)
    best             = int(np.argmin(scores))
    center_idx       = candidate_idx[best].astype(int)
    center_xyz_out   = cand_pts[best]
    center_clearance = float(cand_clearance[best])
    return center_idx, center_xyz_out, center_clearance


def make_center_marker(shape, center_idx, dilation_iters=1):
    marker = np.zeros(shape, dtype=bool)
    if center_idx is None:
        return marker
    i, j, k = center_idx.tolist()
    marker[i, j, k] = True
    if dilation_iters > 0:
        structure = ndimage.generate_binary_structure(3, 1)
        marker    = ndimage.binary_dilation(marker, structure=structure,
                                             iterations=dilation_iters)
    return marker


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def find_structure_file(pdb_file_arg):
    """
    MODIFIED for single-PDB mode: the fixed PDB path is supplied explicitly
    via --pdb_file, rather than searched for in a fixed PDB_BASE directory.
    """
    candidate = Path(pdb_file_arg)
    if candidate.exists():
        return candidate
    return None


def normalize_atomname(atomname):
    return atomname.strip().upper().replace("*", "'")


def is_rna_residue(resname, atom_names):
    res     = resname.strip().upper()
    atomset = {normalize_atomname(a) for a in atom_names}
    if res in DNA_RESNAMES:
        return False
    if "O2'" in atomset:
        return True
    if res in RNA_RESNAMES:
        return True
    if res.startswith("R") and len(res) <= 4:
        return True
    return False


def grouped_atoms_to_xyz(grouped_atoms):
    atoms = []
    for (_, _, _, resname), atom_list in grouped_atoms.items():
        atom_names = [a[0] for a in atom_list]
        if not is_rna_residue(resname, atom_names):
            continue
        for atomname, x, y, z in atom_list:
            if is_hydrogen_name(atomname):
                continue
            atoms.append((x, y, z))
    if not atoms:
        return np.zeros((0, 3), dtype=float)
    return np.array(atoms, dtype=float)


def load_structure_atoms_xyz_manual_pdb(structure_path):
    grouped = {}
    with open(structure_path, "r") as fh:
        for line in fh:
            rec = line[:6].strip().upper()
            if rec != "ATOM":
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            atomname = line[12:16].strip()
            resname  = line[17:20].strip().upper()
            chain    = line[21].strip()
            resseq   = line[22:26].strip()
            icode    = line[26].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            key = (chain, resseq, icode, resname)
            grouped.setdefault(key, []).append((atomname, x, y, z))
    return grouped_atoms_to_xyz(grouped)


def load_structure_atoms_xyz_manual_cif(structure_path):
    with open(structure_path, "r") as fh:
        lines = fh.readlines()

    grouped = {}
    in_loop = False
    headers = []
    rows    = []

    for line in lines:
        s = line.strip()
        if s == "loop_":
            in_loop = True
            headers = []
            rows    = []
            continue
        if in_loop and s.startswith("_atom_site."):
            headers.append(s)
            continue
        if in_loop and headers:
            if not s or s.startswith("#"):
                if rows:
                    break
                continue
            if s.startswith("_"):
                break
            rows.append(shlex.split(s, posix=False))

    if not headers or not rows:
        return np.zeros((0, 3), dtype=float)

    name_to_idx = {h: i for i, h in enumerate(headers)}

    def pick(row, *keys, default=""):
        for k in keys:
            if k in name_to_idx and name_to_idx[k] < len(row):
                return row[name_to_idx[k]]
        return default

    for row in rows:
        group_pdb = pick(row, "_atom_site.group_PDB", default="ATOM").upper()
        if group_pdb != "ATOM":
            continue
        altloc = pick(row, "_atom_site.label_alt_id",
                       "_atom_site.pdbx_PDB_alt_id", default=".")
        if altloc not in (".", "?", "A", ""):
            continue
        atomname = pick(row, "_atom_site.label_atom_id",
                         "_atom_site.auth_atom_id", default="")
        resname  = pick(row, "_atom_site.label_comp_id",
                         "_atom_site.auth_comp_id", default="").upper()
        chain    = pick(row, "_atom_site.auth_asym_id",
                         "_atom_site.label_asym_id", default="")
        resseq   = pick(row, "_atom_site.auth_seq_id",
                         "_atom_site.label_seq_id", default="")
        icode    = pick(row, "_atom_site.pdbx_PDB_ins_code", default="")
        try:
            x = float(pick(row, "_atom_site.Cartn_x"))
            y = float(pick(row, "_atom_site.Cartn_y"))
            z = float(pick(row, "_atom_site.Cartn_z"))
        except Exception:
            continue
        key = (chain, resseq, icode, resname)
        grouped.setdefault(key, []).append((atomname, x, y, z))

    return grouped_atoms_to_xyz(grouped)


def load_structure_atoms_xyz_mdanalysis(structure_path):
    u     = mda.Universe(str(structure_path))
    atoms = []
    for res in u.residues:
        atom_names = [a.name for a in res.atoms]
        if not is_rna_residue(res.resname, atom_names):
            continue
        for atom in res.atoms:
            altloc = getattr(atom, "altLoc", "")
            if altloc not in ("", "A", None):
                continue
            if is_hydrogen_name(atom.name):
                continue
            x, y, z = map(float, atom.position)
            atoms.append((x, y, z))
    if not atoms:
        return np.zeros((0, 3), dtype=float)
    return np.array(atoms, dtype=float)


def load_structure_atoms_xyz(structure_path):
    if mda is not None:
        xyz = load_structure_atoms_xyz_mdanalysis(structure_path)
        if len(xyz) > 0:
            return xyz
    ext = structure_path.suffix.lower()
    if ext == ".pdb":
        return load_structure_atoms_xyz_manual_pdb(structure_path)
    if ext == ".cif":
        return load_structure_atoms_xyz_manual_cif(structure_path)
    return np.zeros((0, 3), dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME ALGORITHM (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def build_seed_sphere(centre_xyz, voxel_size_zyx, origin_xyz, grid_shape,
                       radius_a=SEED_SPHERE_RADIUS_A):
    nz, ny, nx = grid_shape
    sz, sy, sx = voxel_size_zyx[0], voxel_size_zyx[1], voxel_size_zyx[2]
    oz = origin_xyz[2]
    oy = origin_xyz[1]
    ox = origin_xyz[0]
    cx, cy, cz = centre_xyz[0], centre_xyz[1], centre_xyz[2]
    ci_z = (cz - oz) / sz - 0.5
    ci_y = (cy - oy) / sy - 0.5
    ci_x = (cx - ox) / sx - 0.5
    rz_vox = radius_a / sz
    ry_vox = radius_a / sy
    rx_vox = radius_a / sx
    iz_lo = max(0,    int(np.floor(ci_z - rz_vox)))
    iz_hi = min(nz-1, int(np.ceil (ci_z + rz_vox)))
    iy_lo = max(0,    int(np.floor(ci_y - ry_vox)))
    iy_hi = min(ny-1, int(np.ceil (ci_y + ry_vox)))
    ix_lo = max(0,    int(np.floor(ci_x - rx_vox)))
    ix_hi = min(nx-1, int(np.ceil (ci_x + rx_vox)))
    sphere_mask = np.zeros(grid_shape, dtype=bool)
    iz_r = np.arange(iz_lo, iz_hi + 1)
    iy_r = np.arange(iy_lo, iy_hi + 1)
    ix_r = np.arange(ix_lo, ix_hi + 1)
    iz_g, iy_g, ix_g = np.meshgrid(iz_r, iy_r, ix_r, indexing="ij")
    z_a = (iz_g + 0.5) * sz + oz
    y_a = (iy_g + 0.5) * sy + oy
    x_a = (ix_g + 0.5) * sx + ox
    dist2 = (x_a - cx)**2 + (y_a - cy)**2 + (z_a - cz)**2
    sphere_mask[iz_lo:iz_hi+1,
                iy_lo:iy_hi+1,
                ix_lo:ix_hi+1] = dist2 <= radius_a ** 2
    return sphere_mask


def trim_sphere_by_atoms(sphere_mask, atom_xyz, voxel_size_zyx, origin_xyz,
                          excl_radius_a=ATOM_EXCLUSION_RADIUS_A):
    nz, ny, nx = sphere_mask.shape
    sz, sy, sx = voxel_size_zyx[0], voxel_size_zyx[1], voxel_size_zyx[2]
    oz = origin_xyz[2]
    oy = origin_xyz[1]
    ox = origin_xyz[0]
    r2     = excl_radius_a ** 2
    rz_vox = excl_radius_a / sz
    ry_vox = excl_radius_a / sy
    rx_vox = excl_radius_a / sx
    n_excluded = 0
    for ax, ay, az in atom_xyz:
        ci_z = (az - oz) / sz - 0.5
        ci_y = (ay - oy) / sy - 0.5
        ci_x = (ax - ox) / sx - 0.5
        iz_lo = max(0,    int(np.floor(ci_z - rz_vox)))
        iz_hi = min(nz-1, int(np.ceil (ci_z + rz_vox)))
        iy_lo = max(0,    int(np.floor(ci_y - ry_vox)))
        iy_hi = min(ny-1, int(np.ceil (ci_y + ry_vox)))
        ix_lo = max(0,    int(np.floor(ci_x - rx_vox)))
        ix_hi = min(nx-1, int(np.ceil (ci_x + rx_vox)))
        if iz_lo > iz_hi or iy_lo > iy_hi or ix_lo > ix_hi:
            continue
        sub = sphere_mask[iz_lo:iz_hi+1, iy_lo:iy_hi+1, ix_lo:ix_hi+1]
        if not sub.any():
            continue
        iz_r = np.arange(iz_lo, iz_hi + 1)
        iy_r = np.arange(iy_lo, iy_hi + 1)
        ix_r = np.arange(ix_lo, ix_hi + 1)
        iz_g, iy_g, ix_g = np.meshgrid(iz_r, iy_r, ix_r, indexing="ij")
        z_a = (iz_g + 0.5) * sz + oz
        y_a = (iy_g + 0.5) * sy + oy
        x_a = (ix_g + 0.5) * sx + ox
        dist2   = (x_a - ax)**2 + (y_a - ay)**2 + (z_a - az)**2
        to_excl = (dist2 <= r2) & sub
        n_excluded += int(to_excl.sum())
        sphere_mask[iz_lo:iz_hi+1,
                    iy_lo:iy_hi+1,
                    ix_lo:ix_hi+1][to_excl] = False
    return sphere_mask, n_excluded


def make_conn_struct_3d(connectivity):
    if connectivity == 1:
        s = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                       [[0,1,0],[1,1,1],[0,1,0]],
                       [[0,0,0],[0,1,0],[0,0,0]]], dtype=bool)
    elif connectivity == 2:
        s = np.array([[[0,1,0],[1,1,1],[0,1,0]],
                       [[1,1,1],[1,1,1],[1,1,1]],
                       [[0,1,0],[1,1,1],[0,1,0]]], dtype=bool)
    else:
        s = np.ones((3, 3, 3), dtype=bool)
    return s


def extract_pocket_fragment(trimmed_mask, centre_xyz, centre_idx,
                             voxel_size_zyx, origin_xyz,
                             connectivity=POCKET_CONNECTIVITY):
    struct           = make_conn_struct_3d(connectivity)
    labeled, n_labels = scipy_label(trimmed_mask, structure=struct)
    labeled           = labeled.astype(np.int32)
    if n_labels == 0:
        return np.zeros_like(trimmed_mask, dtype=bool), 0, 0, "none"
    if n_labels == 1:
        n_pocket = int(trimmed_mask.sum())
        return trimmed_mask.copy(), n_pocket, 1, "centre_voxel"
    nz, ny, nx = trimmed_mask.shape
    iz_c = int(np.clip(centre_idx[0], 0, nz - 1))
    iy_c = int(np.clip(centre_idx[1], 0, ny - 1))
    ix_c = int(np.clip(centre_idx[2], 0, nx - 1))
    centre_label = int(labeled[iz_c, iy_c, ix_c])
    if centre_label != 0:
        pocket_mask = (labeled == centre_label)
        n_pocket    = int(pocket_mask.sum())
        return pocket_mask, n_pocket, n_labels, "centre_voxel"
    sz, sy, sx = voxel_size_zyx[0], voxel_size_zyx[1], voxel_size_zyx[2]
    oz = origin_xyz[2]
    oy = origin_xyz[1]
    ox = origin_xyz[0]
    best_dist  = np.inf
    best_label = -1
    for sid in range(1, n_labels + 1):
        frag_mask           = (labeled == sid)
        zi_f, yi_f, xi_f   = np.where(frag_mask)
        cz_f = (zi_f.mean() + 0.5) * sz + oz
        cy_f = (yi_f.mean() + 0.5) * sy + oy
        cx_f = (xi_f.mean() + 0.5) * sx + ox
        dist = float(np.sqrt((cx_f - centre_xyz[0])**2 +
                              (cy_f - centre_xyz[1])**2 +
                              (cz_f - centre_xyz[2])**2))
        if dist < best_dist:
            best_dist  = dist
            best_label = sid
    pocket_mask = (labeled == best_label)
    n_pocket    = int(pocket_mask.sum())
    return pocket_mask, n_pocket, n_labels, "nearest_centroid"


# ─────────────────────────────────────────────────────────────────────────────
# POCKET PROCESSING (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def process_single_pocket(pdb_id, pocket_path, pocket_tag, pocket_label,
                           structure_file, atom_xyz, out_dir):

    pocket_data, voxel_size_xyz, voxel_size_zyx, origin_xyz = load_mrc(pocket_path)

    patch_mask = np.isfinite(pocket_data) & (pocket_data > BINARY_EPS)
    if not np.any(patch_mask):
        raise RuntimeError("No positive patch voxels found in the input MRC.")

    chosen_mask, component_count, chosen_voxels, chosen_volume_a3 = \
        choose_single_patch(patch_mask, voxel_size_zyx=voxel_size_zyx,
                             conn=PATCH_CONNECTIVITY)

    if not np.any(chosen_mask):
        raise RuntimeError("Failed to choose a valid volumetric patch.")

    center_idx, center_xyz, center_clearance_a = choose_patch_seed_center(
        chosen_mask,
        voxel_size_zyx=voxel_size_zyx,
        origin_xyz=origin_xyz,
        core_percentile=CENTER_CORE_PERCENTILE,
        max_patch_samples=MAX_PATCH_SAMPLES
    )

    if center_idx is None:
        raise RuntimeError("Failed to choose a geometric seed center inside the patch.")

    center_marker_mask = make_center_marker(
        chosen_mask.shape, center_idx, dilation_iters=MARKER_DILATION_ITERS
    )

    sphere_mask = build_seed_sphere(
        centre_xyz=center_xyz,
        voxel_size_zyx=voxel_size_zyx,
        origin_xyz=origin_xyz,
        grid_shape=chosen_mask.shape,
        radius_a=SEED_SPHERE_RADIUS_A
    )
    n_seed = int(sphere_mask.sum())

    if n_seed == 0:
        raise RuntimeError(
            f"Seed sphere (r={SEED_SPHERE_RADIUS_A} Å) produced zero voxels.")

    sphere_mask, n_excluded = trim_sphere_by_atoms(
        sphere_mask=sphere_mask,
        atom_xyz=atom_xyz,
        voxel_size_zyx=voxel_size_zyx,
        origin_xyz=origin_xyz,
        excl_radius_a=ATOM_EXCLUSION_RADIUS_A
    )
    n_after_trim = int(sphere_mask.sum())

    if n_after_trim == 0:
        raise RuntimeError(
            f"All {n_seed:,} seed-sphere voxels excluded by RNA atoms "
            f"(excl_r={ATOM_EXCLUSION_RADIUS_A} Å).")

    final_pocket_mask, n_pocket, n_frags, select_method = \
        extract_pocket_fragment(
            trimmed_mask=sphere_mask,
            centre_xyz=center_xyz,
            centre_idx=center_idx,
            voxel_size_zyx=voxel_size_zyx,
            origin_xyz=origin_xyz,
            connectivity=POCKET_CONNECTIVITY
        )

    if not np.any(final_pocket_mask):
        raise RuntimeError("Failed to isolate a connected pocket fragment around the centre.")

    voxel_volume_a3        = float(np.prod(voxel_size_zyx))
    final_pocket_voxels    = int(np.count_nonzero(final_pocket_mask))
    final_pocket_volume_a3 = final_pocket_voxels * voxel_volume_a3

    center_marker_mrc = out_dir / f"{pdb_id}.{pocket_tag}.chosen_patch_center_marker.mrc"
    final_pocket_mrc  = out_dir / f"{pdb_id}.{pocket_tag}.final_trimmed_smif_identical_rna_only_pocket.mrc"
    volume_txt        = out_dir / f"{pdb_id}.{pocket_tag}.final_pocket_volume.txt"

    save_mask_as_mrc(center_marker_mask, pocket_path, center_marker_mrc)
    save_mask_as_mrc(final_pocket_mask,  pocket_path, final_pocket_mrc)

    with open(volume_txt, "w") as fh:
        fh.write(f"PDB_ID: {pdb_id}\n")
        fh.write(f"Pocket tag: {pocket_tag}\n")
        fh.write(f"Pocket label: {pocket_label}\n")
        fh.write(f"Input pocket MRC: {pocket_path}\n")
        fh.write(f"Input RNA structure: {structure_file}\n")
        fh.write(f"Voxel size XYZ (A): {voxel_size_xyz.tolist()}\n")
        fh.write(f"Voxel size ZYX (A): {voxel_size_zyx.tolist()}\n")
        fh.write(f"Voxel volume (A^3): {voxel_volume_a3}\n")
        fh.write(f"Chosen patch components: {component_count}\n")
        fh.write(f"Chosen patch voxels: {chosen_voxels}\n")
        fh.write(f"Chosen patch volume (A^3): {chosen_volume_a3}\n")
        fh.write(f"Chosen center ijk (ZYX): {center_idx.tolist()}\n")
        fh.write(f"Chosen center xyz (A): {center_xyz.tolist()}\n")
        fh.write(f"Chosen center clearance (A): {center_clearance_a}\n")
        fh.write(f"Seed sphere radius (A): {SEED_SPHERE_RADIUS_A}\n")
        fh.write(f"Atom exclusion radius (A): {ATOM_EXCLUSION_RADIUS_A}\n")
        fh.write(f"Pocket fragment connectivity: {POCKET_CONNECTIVITY} (26-conn)\n")
        fh.write(f"RNA heavy atom count used for trimming: {len(atom_xyz)}\n")
        fh.write(f"Seed sphere voxels (before trimming): {n_seed}\n")
        fh.write(f"Voxels excluded by RNA atoms: {n_excluded}\n")
        fh.write(f"Voxels after trimming: {n_after_trim}\n")
        fh.write(f"Fragments after trimming: {n_frags}\n")
        fh.write(f"Fragment selection method: {select_method}\n")
        fh.write(f"Final pocket voxels: {final_pocket_voxels}\n")
        fh.write(f"Final pocket volume (A^3): {final_pocket_volume_a3}\n")

    return {
        "pocket_tag":             pocket_tag,
        "pocket_label":           pocket_label,
        "pocket_file":            str(pocket_path),
        "center_marker_mrc":      str(center_marker_mrc),
        "final_pocket_mrc":       str(final_pocket_mrc),
        "volume_txt":             str(volume_txt),
        "final_pocket_voxels":    final_pocket_voxels,
        "final_pocket_volume_a3": final_pocket_volume_a3,
        "chosen_center_ijk":      center_idx.tolist(),
        "chosen_center_xyz":      center_xyz.tolist(),
    }


# =====================================================================
# SINGLE-PDB DRIVER (generalized — replaces the old multi-PDB BASE loop)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Script3 (Pipeline 1): Making pocket volumes from detected hotspots, for a single PDB."
    )
    parser.add_argument(
        "--pdb_file",
        required=True,
        help="Path to the fixed PDB/CIF structure file (e.g. 1AJU_fixed.pdb) in the current working directory.",
    )
    parser.add_argument(
        "--analysis_dir",
        required=True,
        help="Path to Analysis_Pipeline1_<PDB_ID> folder (output of Script1/Script2). "
             "Pocket .mrc inputs are read from here and all Script3 outputs are saved here.",
    )
    parser.add_argument(
        "--pdb_id",
        default=None,
        help="Optional override for the PDB identifier used in filenames. "
             "If not provided, it is automatically derived from --pdb_file (filename without extension).",
    )
    args = parser.parse_args()

    pdb_file_path = Path(args.pdb_file)
    pdb_id = args.pdb_id if args.pdb_id else pdb_file_path.stem

    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    structure_file = find_structure_file(args.pdb_file)

    if structure_file is None:
        print(f"ERROR: missing RNA structure file: {args.pdb_file}")
        return

    atom_xyz = load_structure_atoms_xyz(structure_file)
    if len(atom_xyz) == 0:
        print(f"ERROR: no RNA heavy atoms parsed from structure file {structure_file}")
        return

    existing_pockets = []
    missing_pockets  = []

    for spec in POCKET_SPECS:
        pocket_path = analysis_dir / f"{pdb_id}{spec['suffix']}"
        if pocket_path.exists():
            existing_pockets.append({
                "tag":   spec["tag"],
                "label": spec["label"],
                "path":  pocket_path
            })
        else:
            missing_pockets.append({
                "tag":   spec["tag"],
                "label": spec["label"],
                "path":  pocket_path
            })

    if not existing_pockets:
        print(f"ERROR: no eligible pocket MRC files found in {analysis_dir}")
        return

    results  = []
    failures = []

    for pocket_info in existing_pockets:
        try:
            res = process_single_pocket(
                pdb_id=pdb_id,
                pocket_path=pocket_info["path"],
                pocket_tag=pocket_info["tag"],
                pocket_label=pocket_info["label"],
                structure_file=structure_file,
                atom_xyz=atom_xyz,
                out_dir=analysis_dir
            )
            results.append(res)
            print("Center marker MRC:",       res["center_marker_mrc"])
            print("Final pocket MRC:",        res["final_pocket_mrc"])
            print("Final pocket volume TXT:", res["volume_txt"])
            print("Final pocket volume (A^3):", res["final_pocket_volume_a3"])
        except Exception as exc:
            failures.append({
                "tag":   pocket_info["tag"],
                "label": pocket_info["label"],
                "path":  str(pocket_info["path"]),
                "error": str(exc)
            })
            print(f"FAILED for {pocket_info['path']}: {exc}")

    summary_txt = analysis_dir / f"{pdb_id}.all_processed_pocket_volumes_summary.txt"
    with open(summary_txt, "w") as fh:
        fh.write(f"PDB_ID: {pdb_id}\n")
        fh.write(f"Input RNA structure: {structure_file}\n")
        fh.write(f"RNA heavy atom count used for trimming: {len(atom_xyz)}\n")
        fh.write(f"Seed sphere radius (A): {SEED_SPHERE_RADIUS_A}\n")
        fh.write(f"Atom exclusion radius (A): {ATOM_EXCLUSION_RADIUS_A}\n")
        fh.write(f"Pocket fragment connectivity: {POCKET_CONNECTIVITY} (26-conn)\n")
        fh.write(f"PATCH_CONNECTIVITY: {PATCH_CONNECTIVITY}\n")
        fh.write(f"CENTER_CORE_PERCENTILE: {CENTER_CORE_PERCENTILE}\n")
        fh.write(f"MAX_PATCH_SAMPLES: {MAX_PATCH_SAMPLES}\n")
        fh.write(f"CENTER_CUBE_RADIUS_VOX: {CENTER_CUBE_RADIUS_VOX}\n")
        fh.write("\n")
        fh.write("Processed pocket files successfully:\n")
        if results:
            for res in results:
                fh.write(f"- Pocket tag: {res['pocket_tag']}\n")
                fh.write(f"  Pocket label: {res['pocket_label']}\n")
                fh.write(f"  Input pocket MRC: {res['pocket_file']}\n")
                fh.write(f"  Center marker MRC: {res['center_marker_mrc']}\n")
                fh.write(f"  Final pocket MRC: {res['final_pocket_mrc']}\n")
                fh.write(f"  Volume TXT: {res['volume_txt']}\n")
                fh.write(f"  Chosen center ijk (ZYX): {res['chosen_center_ijk']}\n")
                fh.write(f"  Chosen center xyz (A): {res['chosen_center_xyz']}\n")
                fh.write(f"  Final pocket voxels: {res['final_pocket_voxels']}\n")
                fh.write(f"  Final pocket volume (A^3): {res['final_pocket_volume_a3']}\n")
                fh.write("\n")
        else:
            fh.write("- None\n\n")
        fh.write("Pocket files missing and skipped:\n")
        if missing_pockets:
            for miss in missing_pockets:
                fh.write(f"- Pocket tag: {miss['tag']}\n")
                fh.write(f"  Pocket label: {miss['label']}\n")
                fh.write(f"  Expected path: {miss['path']}\n\n")
        else:
            fh.write("- None\n\n")
        fh.write("Pocket files present but failed during calculation:\n")
        if failures:
            for fail in failures:
                fh.write(f"- Pocket tag: {fail['tag']}\n")
                fh.write(f"  Pocket label: {fail['label']}\n")
                fh.write(f"  Input path: {fail['path']}\n")
                fh.write(f"  Error: {fail['error']}\n\n")
        else:
            fh.write("- None\n")

    print("Summary TXT:", summary_txt)
    print(f"\nScript3 complete for {pdb_id}. Results saved in: {analysis_dir}")


if __name__ == "__main__":
    main()
