import argparse
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

# -------------------------------------------------
# Parameters (UNCHANGED)
# -------------------------------------------------
MIN_COMPONENT_VOXELS          = 8
CLOSE_DISTANCE_ANGSTROM       = 3.5
VERY_CLOSE_DISTANCE_ANGSTROM  = 1.5

REAL_BURIEDNESS_RADIUS_A          = 2.5
REAL_BURIEDNESS_NEIGHBOR_RADIUS_A = 10.0
REAL_BURIEDNESS_MIN               = 0.20

ACCESSIBLE_POINT_FAR_A      = 3.0
CENTROID_ENCLOSURE_RADIUS_A = 8.0

SECOND_POCKET_MAX_ATOM_DISTANCE_A = 6.0

POCKET_SCORE_WEIGHTS = {
    "stacking_rel":    1.0,
    "hydrophobic_rel": 1.0,
    "apbs_rel":        1.0,
    "buriedness":      1.2,
    "proximity_bonus": 0.5,
}

MIXED_WEIGHTS = {
    "triple_overlap":  8.0,
    "pair_overlap":    3.5,
    "proximity_bonus": 4.0,
    "sum_stk":         0.25,
    "sum_hyd":         0.20,
    "sum_apbs":        0.25,
    "density_stk":     0.10,
    "density_hyd":     0.10,
    "density_apbs":    0.15,
    "buriedness":      8.0,
}

PAIR_WEIGHTS = {
    "overlap_voxels":   6.0,
    "overlap_fraction": 4.0,
    "distance_bonus":   4.0,
    "sum_a":            0.30,
    "sum_b":            0.30,
    "density_a":        0.15,
    "density_b":        0.15,
    "buriedness":       8.0,
}

SINGLE_WEIGHTS = {
    "nvox":      0.8,
    "sum":       0.25,
    "density":   0.25,
    "buriedness":10.0,
}

# -------------------------------------------------
# Utilities (UNCHANGED)
# -------------------------------------------------
def find_field_file(pdb_dir, pdb, field):
    candidates = sorted(pdb_dir.glob(f"{pdb}*.mrc"))
    for path in candidates:
        low = path.name.lower()
        if "thresholded" in low or "pocket" in low or "center_marker" in low:
            continue
        if field == "stacking" and "stacking" in low and "electrostatic" not in low:
            return path
        if field == "hydrophobic" and "hydrophobic" in low:
            return path
        if field == "apbs" and (
            ".apbs.mrc" in low or
            low.endswith(".apbs.mrc") or
            "electrostatic" in low or
            ".ele." in low or
            low.endswith(".ele.mrc")
        ):
            return path
    return None

def find_pdb_file(pdb_path_arg, pdb):
    """
    MODIFIED for single-PDB mode: the fixed PDB path is now supplied
    explicitly via --pdb_file, rather than being searched for in a
    fixed PDB_BASE directory across many PDBs.
    """
    candidate = Path(pdb_path_arg)
    if candidate.exists():
        return candidate
    return None

def find_iso_table(analysis_dir, pdb):
    preferred = analysis_dir / f"{pdb}_stk_hp_ele_selected_isovalues.csv"
    if preferred.exists():
        return preferred
    fallback = sorted(analysis_dir.glob("*_stk_hp_ele_selected_isovalues.csv"))
    if fallback:
        return fallback[0]
    return None

def load_mrc(path):
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32).copy()
        voxel_size = np.array(
            [float(mrc.voxel_size.x),
             float(mrc.voxel_size.y),
             float(mrc.voxel_size.z)],
            dtype=float
        )
        try:
            origin = np.array(
                [float(mrc.header.origin.x),
                 float(mrc.header.origin.y),
                 float(mrc.header.origin.z)],
                dtype=float
            )
        except Exception:
            origin = np.array([0.0, 0.0, 0.0], dtype=float)
    return data, voxel_size, origin

def save_mrc_from_mask(mask, ref_path, out_path):
    _, voxel_size, origin = load_mrc(ref_path)
    out_data = mask.astype(np.float32)
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(out_data)
        mrc.voxel_size = tuple(voxel_size.tolist())
        try:
            mrc.header.origin.x = origin[0]
            mrc.header.origin.y = origin[1]
            mrc.header.origin.z = origin[2]
        except Exception:
            pass
        mrc.update_header_from_data()
        mrc.update_header_stats()

def threshold_mask(data, iso, field):
    if field == "apbs":
        return np.isfinite(data) & (data <= iso)
    return np.isfinite(data) & (data >= iso)

def connected_components(mask):
    structure = ndimage.generate_binary_structure(3, 2)
    labels, nlab = ndimage.label(mask, structure=structure)
    return labels, nlab

# -------------------------------------------------
# RNA residue / hydrogen helpers (UNCHANGED)
# -------------------------------------------------
_DNA_RESNAMES = {
    "DA", "DC", "DG", "DT", "DI",
    "ADE", "CYT", "GUA", "THY",
}

_RNA_RESNAMES = {
    "A",   "C",   "G",   "U",   "I",
    "RA",  "RC",  "RG",  "RU",  "RI",
    "ADE", "CYT", "GUA", "URA", "URI",
    "PSU", "H2U", "5MU", "5MC", "OMC", "OMG",
    "1MA", "2MA", "M2G", "1MG", "7MG",
    "YG",  "YYG", "G7M", "A2M",
    "4SU", "OMU", "MIA", "5BU", "2MG",
}

def _is_hydrogen_name(atomname: str) -> bool:
    name = atomname.strip().upper().replace("*", "'")
    stripped = name.lstrip("0123456789")
    if not stripped:
        return False
    return stripped[0] in ("H", "D")

def _is_rna_residue(resname: str, atom_names: list) -> bool:
    res     = resname.strip().upper()
    atomset = {a.strip().upper().replace("*", "'") for a in atom_names}
    if "O2'" in atomset:
        return True
    if res in _DNA_RESNAMES:
        return False
    if res in _RNA_RESNAMES:
        return True
    if res.startswith("R") and len(res) <= 4:
        return True
    return False

# -------------------------------------------------
# Structure loader (UNCHANGED)
# -------------------------------------------------
def load_structure_atoms(pdb_path):
    atoms = []
    ext   = pdb_path.suffix.lower()

    if ext == ".pdb":
        grouped = {}
        with open(pdb_path, "r") as fh:
            for line in fh:
                rec = line[:6].strip().upper()
                if rec != "ATOM":
                    continue
                altloc = line[16].strip()
                if altloc not in ("", "A"):
                    continue
                atomname = line[12:16].strip()
                if _is_hydrogen_name(atomname):
                    continue
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

        for (_, _, _, resname), atom_list in grouped.items():
            if not _is_rna_residue(resname, [a[0] for a in atom_list]):
                continue
            for _, x, y, z in atom_list:
                atoms.append((x, y, z))

    elif ext == ".cif":
        in_loop = False
        headers = []
        rows    = []
        with open(pdb_path, "r") as fh:
            for line in fh:
                s = line.strip()
                if s == "loop_":
                    in_loop = True; headers = []; rows = []; continue
                if in_loop and s.startswith("_atom_site."):
                    headers.append(s); continue
                if in_loop and headers and s and not s.startswith("_") and not s.startswith("#"):
                    rows.append(s.split()); continue
                if in_loop and headers and (s.startswith("#") or
                        (s.startswith("_") and not s.startswith("_atom_site."))):
                    break

        if headers and rows:
            n2i = {h: i for i, h in enumerate(headers)}
            req  = ["_atom_site.Cartn_x", "_atom_site.Cartn_y",
                    "_atom_site.Cartn_z", "_atom_site.label_atom_id",
                    "_atom_site.label_comp_id"]
            if all(k in n2i for k in req):
                grouped = {}
                for row in rows:
                    try:
                        if "_atom_site.group_PDB" in n2i:
                            if row[n2i["_atom_site.group_PDB"]].upper() != "ATOM":
                                continue
                        atomname = row[n2i["_atom_site.label_atom_id"]]
                        if _is_hydrogen_name(atomname):
                            continue
                        resname  = row[n2i["_atom_site.label_comp_id"]].upper()
                        chain    = row[n2i["_atom_site.auth_asym_id"]] \
                            if "_atom_site.auth_asym_id" in n2i else ""
                        resseq   = row[n2i["_atom_site.auth_seq_id"]] \
                            if "_atom_site.auth_seq_id" in n2i else ""
                        icode    = row[n2i["_atom_site.pdbx_PDB_ins_code"]] \
                            if "_atom_site.pdbx_PDB_ins_code" in n2i else ""
                        x = float(row[n2i["_atom_site.Cartn_x"]])
                        y = float(row[n2i["_atom_site.Cartn_y"]])
                        z = float(row[n2i["_atom_site.Cartn_z"]])
                        key = (chain, resseq, icode, resname)
                        grouped.setdefault(key, []).append((atomname, x, y, z))
                    except Exception:
                        continue

                for (_, _, _, resname), atom_list in grouped.items():
                    if not _is_rna_residue(resname, [a[0] for a in atom_list]):
                        continue
                    for _, x, y, z in atom_list:
                        atoms.append((x, y, z))

    if not atoms:
        return np.zeros((0, 3), dtype=float)
    return np.array(atoms, dtype=float)

# -------------------------------------------------
# Geometry helpers (UNCHANGED)
# -------------------------------------------------
def mask_indices_to_xyz(idx, voxel_size, origin):
    ijk = np.column_stack(idx).astype(float)
    return ijk * voxel_size + origin

def component_mask_from_indices(shape, idx):
    m = np.zeros(shape, dtype=bool)
    m[idx] = True
    return m

def build_union_mask(shape, components):
    out = np.zeros(shape, dtype=bool)
    for comp in components:
        out[comp["indices"]] = True
    return out

def overlap_mask_from_indices(shape, idx1, idx2):
    return (component_mask_from_indices(shape, idx1) &
            component_mask_from_indices(shape, idx2))

def count_overlap_voxels(shape, idx1, idx2):
    ov = overlap_mask_from_indices(shape, idx1, idx2)
    return int(np.count_nonzero(ov)), ov

def triple_overlap_mask(shape, idx1, idx2, idx3):
    m = (component_mask_from_indices(shape, idx1) &
         component_mask_from_indices(shape, idx2) &
         component_mask_from_indices(shape, idx3))
    return m, int(np.count_nonzero(m))

def min_distance_between_components(idx1, idx2, voxel_size):
    a = np.column_stack(idx1).astype(float) * voxel_size
    b = np.column_stack(idx2).astype(float) * voxel_size
    if len(a) == 0 or len(b) == 0:
        return np.inf
    min_d2 = np.inf
    chunk = 1500
    for i in range(0, len(a), chunk):
        aa = a[i:i + chunk]
        d2 = ((aa[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        local = np.min(d2)
        if local < min_d2:
            min_d2 = local
    return float(np.sqrt(min_d2))

def pair_distance_bonus(dist):
    if dist > CLOSE_DISTANCE_ANGSTROM:
        return None
    if dist <= VERY_CLOSE_DISTANCE_ANGSTROM:
        return 1.0
    return max(
        0.0,
        1.0 - (dist - VERY_CLOSE_DISTANCE_ANGSTROM) /
              (CLOSE_DISTANCE_ANGSTROM - VERY_CLOSE_DISTANCE_ANGSTROM)
    )

def localized_proximity_mask(shape, idx1, idx2, voxel_size, cutoff):
    xyz1 = np.column_stack(idx1).astype(float) * voxel_size
    xyz2 = np.column_stack(idx2).astype(float) * voxel_size
    if len(xyz1) == 0 or len(xyz2) == 0:
        return np.zeros(shape, dtype=bool)
    tree2 = cKDTree(xyz2)
    d12, _ = tree2.query(xyz1, k=1, workers=-1)
    keep1 = d12 <= cutoff
    tree1 = cKDTree(xyz1)
    d21, _ = tree1.query(xyz2, k=1, workers=-1)
    keep2 = d21 <= cutoff
    mask = np.zeros(shape, dtype=bool)
    if np.any(keep1):
        mask[tuple(arr[keep1] for arr in idx1)] = True
    if np.any(keep2):
        mask[tuple(arr[keep2] for arr in idx2)] = True
    return mask

def build_stk_ele_pocket_mask(shape, stk_comp, apbs_comp, voxel_size):
    ov_nvox, ov_mask = count_overlap_voxels(shape,
                                             stk_comp["indices"],
                                             apbs_comp["indices"])
    if ov_nvox > 0:
        return ov_mask.copy(), "stk_ele_overlap_only"
    dist   = min_distance_between_components(stk_comp["indices"],
                                             apbs_comp["indices"], voxel_size)
    dbonus = pair_distance_bonus(dist)
    if dbonus is None:
        return None, None
    prox_mask = localized_proximity_mask(
        shape, stk_comp["indices"], apbs_comp["indices"],
        voxel_size, CLOSE_DISTANCE_ANGSTROM
    )
    if np.count_nonzero(prox_mask) == 0:
        return None, None
    return prox_mask, "stk_ele_proximity_localized"

# -------------------------------------------------
# Buriedness / scoring (UNCHANGED)
# -------------------------------------------------
def estimate_real_buriedness_from_atoms(component_xyz, atom_xyz):
    if len(component_xyz) == 0 or len(atom_xyz) == 0:
        return 0.0
    tree = cKDTree(atom_xyz)
    dists, _ = tree.query(component_xyz, k=1, workers=-1)
    close_frac = float(np.mean(dists <= REAL_BURIEDNESS_RADIUS_A))
    far_frac   = float(np.mean(dists >= ACCESSIBLE_POINT_FAR_A))
    centroid   = component_xyz.mean(axis=0)
    local_idx  = tree.query_ball_point(centroid, REAL_BURIEDNESS_NEIGHBOR_RADIUS_A)
    local_density = min(1.0, len(local_idx) / 80.0)
    shell_idx = tree.query_ball_point(centroid, CENTROID_ENCLOSURE_RADIUS_A)
    centroid_enclosure = min(1.0, len(shell_idx) / 120.0)
    buriedness = (
        0.35 * close_frac +
        0.25 * (1.0 - far_frac) +
        0.20 * local_density +
        0.20 * centroid_enclosure
    )
    return float(np.clip(buriedness, 0.0, 1.0))

def component_stats(labels, nlab, data, voxel_size, origin, atom_xyz, field_name):
    comps = []
    for lab in range(1, nlab + 1):
        idx  = np.where(labels == lab)
        nvox = len(idx[0])
        if nvox < MIN_COMPONENT_VOXELS:
            continue
        vals       = data[idx]
        vals_score = np.abs(vals) if field_name == "apbs" else vals
        total      = float(np.sum(vals_score))
        density    = total / nvox
        centroid_idx = np.array([np.mean(idx[0]), np.mean(idx[1]),
                                 np.mean(idx[2])], dtype=float)
        centroid_xyz = centroid_idx * voxel_size + origin
        comp_xyz     = mask_indices_to_xyz(idx, voxel_size, origin)
        real_buriedness = estimate_real_buriedness_from_atoms(comp_xyz, atom_xyz)
        comps.append({
            "label": lab, "field": field_name, "indices": idx,
            "mask": component_mask_from_indices(data.shape, idx),
            "nvox": nvox, "sum": total, "density": density,
            "centroid_idx": centroid_idx, "centroid_xyz": centroid_xyz,
            "real_buriedness": real_buriedness,
        })
    return comps

def score_single(comp):
    return (
        SINGLE_WEIGHTS["nvox"]       * comp["nvox"] +
        SINGLE_WEIGHTS["sum"]        * comp["sum"] +
        SINGLE_WEIGHTS["density"]    * comp["density"] +
        SINGLE_WEIGHTS["buriedness"] * comp["real_buriedness"]
    )

def score_two_field_pair(comp_a, comp_b, shape, voxel_size):
    ov_nvox, ov_mask = count_overlap_voxels(shape,
                                             comp_a["indices"],
                                             comp_b["indices"])
    smaller = min(comp_a["nvox"], comp_b["nvox"])
    ov_frac = ov_nvox / smaller if smaller > 0 else 0.0
    dist    = 0.0 if ov_nvox > 0 else min_distance_between_components(
        comp_a["indices"], comp_b["indices"], voxel_size)
    dbonus  = 1.0 if ov_nvox > 0 else pair_distance_bonus(dist)
    if dbonus is None:
        return None
    buriedness = max(comp_a["real_buriedness"], comp_b["real_buriedness"])
    score = (
        PAIR_WEIGHTS["overlap_voxels"]   * ov_nvox +
        PAIR_WEIGHTS["overlap_fraction"] * ov_frac +
        PAIR_WEIGHTS["distance_bonus"]   * dbonus +
        PAIR_WEIGHTS["sum_a"]            * comp_a["sum"] +
        PAIR_WEIGHTS["sum_b"]            * comp_b["sum"] +
        PAIR_WEIGHTS["density_a"]        * comp_a["density"] +
        PAIR_WEIGHTS["density_b"]        * comp_b["density"] +
        PAIR_WEIGHTS["buriedness"]       * buriedness
    )
    return {
        "score": score, "ov_nvox": ov_nvox, "ov_frac": ov_frac,
        "dist": dist, "distance_bonus": dbonus,
        "buriedness": buriedness, "ov_mask": ov_mask,
    }

def score_three_field_triplet(stk_comp, hyd_comp, apbs_comp, shape, voxel_size):
    tri_mask, tri_ov = triple_overlap_mask(shape, stk_comp["indices"],
                                           hyd_comp["indices"],
                                           apbs_comp["indices"])
    sh_ov, _ = count_overlap_voxels(shape, stk_comp["indices"], hyd_comp["indices"])
    sa_ov, _ = count_overlap_voxels(shape, stk_comp["indices"], apbs_comp["indices"])
    ha_ov, _ = count_overlap_voxels(shape, hyd_comp["indices"], apbs_comp["indices"])

    d_sh = 0.0 if sh_ov > 0 else min_distance_between_components(
        stk_comp["indices"], hyd_comp["indices"], voxel_size)
    d_sa = 0.0 if sa_ov > 0 else min_distance_between_components(
        stk_comp["indices"], apbs_comp["indices"], voxel_size)
    d_ha = 0.0 if ha_ov > 0 else min_distance_between_components(
        hyd_comp["indices"], apbs_comp["indices"], voxel_size)

    b_sh = pair_distance_bonus(d_sh)
    b_sa = pair_distance_bonus(d_sa)
    b_ha = pair_distance_bonus(d_ha)
    if b_sh is None or b_sa is None or b_ha is None:
        return None

    proximity_bonus = max(b_sh, b_sa, b_ha)
    pair_overlap    = sh_ov + sa_ov + ha_ov
    buriedness      = max(stk_comp["real_buriedness"],
                          hyd_comp["real_buriedness"],
                          apbs_comp["real_buriedness"])
    score = (
        MIXED_WEIGHTS["triple_overlap"]  * tri_ov +
        MIXED_WEIGHTS["pair_overlap"]    * pair_overlap +
        MIXED_WEIGHTS["proximity_bonus"] * proximity_bonus +
        MIXED_WEIGHTS["sum_stk"]         * stk_comp["sum"] +
        MIXED_WEIGHTS["sum_hyd"]         * hyd_comp["sum"] +
        MIXED_WEIGHTS["sum_apbs"]        * apbs_comp["sum"] +
        MIXED_WEIGHTS["density_stk"]     * stk_comp["density"] +
        MIXED_WEIGHTS["density_hyd"]     * hyd_comp["density"] +
        MIXED_WEIGHTS["density_apbs"]    * apbs_comp["density"] +
        MIXED_WEIGHTS["buriedness"]      * buriedness
    )
    return {
        "score": score,
        "triple_overlap_voxels": tri_ov,
        "pair_overlap_voxels":   pair_overlap,
        "proximity_bonus":       proximity_bonus,
        "d_sh": d_sh, "d_sa": d_sa, "d_ha": d_ha,
        "buriedness": buriedness,
        "tri_mask":   tri_mask,
    }

def field_integral_in_mask(data, mask, field):
    vals = data[mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0
    return float(np.sum(np.abs(vals)) if field == "apbs" else np.sum(vals))

def compute_relevance_scores(mask, stk_data, hyd_data, apbs_data):
    stk_total  = field_integral_in_mask(stk_data,  np.isfinite(stk_data),  "stacking")
    hyd_total  = field_integral_in_mask(hyd_data,  np.isfinite(hyd_data),  "hydrophobic")
    apbs_total = field_integral_in_mask(apbs_data, np.isfinite(apbs_data), "apbs")

    stk_in  = field_integral_in_mask(stk_data,  mask, "stacking")
    hyd_in  = field_integral_in_mask(hyd_data,  mask, "hydrophobic")
    apbs_in = field_integral_in_mask(apbs_data, mask, "apbs")

    stk_rel  = stk_in  / stk_total  if stk_total  > 0 else 0.0
    hyd_rel  = hyd_in  / hyd_total  if hyd_total  > 0 else 0.0
    apbs_rel = apbs_in / apbs_total if apbs_total > 0 else 0.0

    rel_sum = stk_rel + hyd_rel + apbs_rel
    if rel_sum > 0:
        stk_rel_norm  = stk_rel  / rel_sum
        hyd_rel_norm  = hyd_rel  / rel_sum
        apbs_rel_norm = apbs_rel / rel_sum
    else:
        stk_rel_norm = hyd_rel_norm = apbs_rel_norm = 0.0

    return {
        "stacking_rel":         stk_rel,
        "hydrophobic_rel":      hyd_rel,
        "apbs_rel":             apbs_rel,
        "stacking_rel_norm":    stk_rel_norm,
        "hydrophobic_rel_norm": hyd_rel_norm,
        "apbs_rel_norm":        apbs_rel_norm,
    }

def compute_pocket_score(relevance, buriedness, proximity_bonus=0.0):
    return (
        POCKET_SCORE_WEIGHTS["stacking_rel"]    * relevance["stacking_rel"] +
        POCKET_SCORE_WEIGHTS["hydrophobic_rel"] * relevance["hydrophobic_rel"] +
        POCKET_SCORE_WEIGHTS["apbs_rel"]        * relevance["apbs_rel"] +
        POCKET_SCORE_WEIGHTS["buriedness"]      * buriedness +
        POCKET_SCORE_WEIGHTS["proximity_bonus"] * proximity_bonus
    )

def summarize_pocket(mask, components, pocket_type, shape, voxel_size, origin,
                     atom_xyz, stk_data, hyd_data, apbs_data):
    if mask is None or np.count_nonzero(mask) == 0:
        return None
    coords       = np.column_stack(np.where(mask))
    centroid_idx = coords.mean(axis=0)
    centroid_xyz = centroid_idx * voxel_size + origin
    pocket_xyz   = coords * voxel_size + origin
    pocket_buriedness = estimate_real_buriedness_from_atoms(pocket_xyz, atom_xyz)
    relevance = compute_relevance_scores(mask, stk_data, hyd_data, apbs_data)

    proximity_bonus = 0.0
    if len(components) >= 2:
        dvals = []
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                ov_n, _ = count_overlap_voxels(shape,
                                               components[i]["indices"],
                                               components[j]["indices"])
                if ov_n > 0:
                    dvals.append(1.0)
                else:
                    d = min_distance_between_components(
                        components[i]["indices"], components[j]["indices"], voxel_size)
                    b = pair_distance_bonus(d)
                    if b is not None:
                        dvals.append(b)
        if dvals:
            proximity_bonus = max(dvals)

    pocket_score = compute_pocket_score(relevance, pocket_buriedness, proximity_bonus)

    ligand_type_map = {
        "mixed_fields":               "mixed_aromatic_hydrophobic_charge_supported",
        "stacking_electrostatic":     "aromatic_charge_supported",
        "stacking_hydrophobic":       "aromatic_hydrophobic_preferred",
        "stacking_hydrophobic_second":"aromatic_hydrophobic_preferred",
        "electrostatic":              "non_aromatic_charge_or_ion_preferred",
    }
    ligand_type = ligand_type_map.get(pocket_type, "mixed")

    return {
        "pocket_type":          pocket_type,
        "mask":                 mask,
        "components":           components,
        "centroid_x":           float(centroid_xyz[0]),
        "centroid_y":           float(centroid_xyz[1]),
        "centroid_z":           float(centroid_xyz[2]),
        "nvox":                 int(np.count_nonzero(mask)),
        "real_buriedness":      float(pocket_buriedness),
        "stacking_rel":         relevance["stacking_rel"],
        "hydrophobic_rel":      relevance["hydrophobic_rel"],
        "apbs_rel":             relevance["apbs_rel"],
        "stacking_rel_norm":    relevance["stacking_rel_norm"],
        "hydrophobic_rel_norm": relevance["hydrophobic_rel_norm"],
        "apbs_rel_norm":        relevance["apbs_rel_norm"],
        "pocket_score":         float(pocket_score),
        "ligand_type":          ligand_type,
    }

def min_distance_mask_to_atoms(mask, voxel_size, origin, atom_xyz):
    if mask is None or np.count_nonzero(mask) == 0 or len(atom_xyz) == 0:
        return np.inf
    coords = np.column_stack(np.where(mask)).astype(float)
    xyz    = coords * voxel_size + origin
    tree   = cKDTree(atom_xyz)
    dists, _ = tree.query(xyz, k=1, workers=-1)
    return float(np.min(dists)) if len(dists) > 0 else np.inf

def mask_is_near_structure(mask, voxel_size, origin, atom_xyz, max_distance):
    return min_distance_mask_to_atoms(mask, voxel_size, origin, atom_xyz) <= max_distance

def subtract_mask(mask, exclude_mask):
    if mask is None:
        return None
    out = mask.copy()
    if exclude_mask is not None:
        out[exclude_mask] = False
    return out

def component_from_mask(mask, data, voxel_size, origin, atom_xyz,
                        field_name, label_seed):
    if mask is None or np.count_nonzero(mask) < MIN_COMPONENT_VOXELS:
        return None
    idx        = np.where(mask)
    nvox       = len(idx[0])
    vals       = data[idx]
    vals_score = np.abs(vals) if field_name == "apbs" else vals
    total      = float(np.sum(vals_score))
    density    = total / nvox
    centroid_idx = np.array([np.mean(idx[0]), np.mean(idx[1]),
                             np.mean(idx[2])], dtype=float)
    centroid_xyz = centroid_idx * voxel_size + origin
    comp_xyz     = mask_indices_to_xyz(idx, voxel_size, origin)
    real_buriedness = estimate_real_buriedness_from_atoms(comp_xyz, atom_xyz)
    return {
        "label": int(label_seed), "field": field_name, "indices": idx,
        "mask": mask.copy(), "nvox": nvox, "sum": total, "density": density,
        "centroid_idx": centroid_idx, "centroid_xyz": centroid_xyz,
        "real_buriedness": real_buriedness,
    }

def split_components_after_exclusion(comps, exclude_mask, data,
                                     voxel_size, origin, atom_xyz, field_name):
    out        = []
    label_seed = 1
    for comp in comps:
        trimmed = subtract_mask(comp["mask"], exclude_mask)
        if trimmed is None or np.count_nonzero(trimmed) < MIN_COMPONENT_VOXELS:
            continue
        labels, nlab = connected_components(trimmed)
        for lab in range(1, nlab + 1):
            submask = labels == lab
            if np.count_nonzero(submask) < MIN_COMPONENT_VOXELS:
                continue
            new_comp = component_from_mask(submask, data, voxel_size, origin,
                                           atom_xyz, field_name, label_seed)
            if new_comp is not None:
                out.append(new_comp)
                label_seed += 1
    return out

def select_best_sh_from_component_sets(
        stk_comps, hyd_comps, shape, voxel_size, origin, atom_xyz,
        stk_data, hyd_data, apbs_data, pocket_type,
        enforce_structure_nearness=False,
        max_atom_distance=SECOND_POCKET_MAX_ATOM_DISTANCE_A,
):
    sh_overlap_candidates = []
    sh_close_candidates   = []
    sh_single_candidates  = []

    for scomp in stk_comps:
        for hcomp in hyd_comps:
            pair_score = score_two_field_pair(scomp, hcomp, shape, voxel_size)
            if pair_score is None:
                continue
            pair_mask = build_union_mask(shape, [scomp, hcomp])
            if enforce_structure_nearness and not mask_is_near_structure(
                    pair_mask, voxel_size, origin, atom_xyz, max_atom_distance):
                continue
            pocket = summarize_pocket(
                pair_mask, [scomp, hcomp], pocket_type,
                shape, voxel_size, origin, atom_xyz, stk_data, hyd_data, apbs_data)
            if pocket is None:
                continue
            packed = {
                "pocket":         pocket,
                "mask":           pair_mask.copy(),
                "ov_nvox":        pair_score["ov_nvox"],
                "volume":         int(np.count_nonzero(pair_mask)),
                "combined_score": pair_score["score"] + 20.0 * pocket["pocket_score"],
                "buriedness":     pocket["real_buriedness"],
                "dist":           pair_score["dist"],
                "selection_rule": ("stk_hyd_overlap_union"
                                   if pair_score["ov_nvox"] > 0
                                   else "stk_hyd_close_union"),
            }
            if pair_score["ov_nvox"] > 0:
                sh_overlap_candidates.append(packed)
            else:
                sh_close_candidates.append(packed)

    for comp in stk_comps + hyd_comps:
        if comp["real_buriedness"] < REAL_BURIEDNESS_MIN:
            continue
        if enforce_structure_nearness and not mask_is_near_structure(
                comp["mask"], voxel_size, origin, atom_xyz, max_atom_distance):
            continue
        pocket = summarize_pocket(
            comp["mask"].copy(), [comp], pocket_type,
            shape, voxel_size, origin, atom_xyz, stk_data, hyd_data, apbs_data)
        if pocket is None:
            continue
        packed = {
            "pocket":         pocket,
            "mask":           comp["mask"].copy(),
            "volume":         comp["nvox"],
            "buriedness":     comp["real_buriedness"],
            "combined_score": score_single(comp) + 20.0 * pocket["pocket_score"],
            "field":          comp["field"],
            "selection_rule": "largest_single_buried_patch",
        }
        sh_single_candidates.append(packed)

    if sh_overlap_candidates:
        sh_overlap_candidates.sort(
            key=lambda x: (x["volume"], x["combined_score"],
                           x["buriedness"], x["ov_nvox"]), reverse=True)
        return sh_overlap_candidates[0]
    if sh_close_candidates:
        sh_close_candidates.sort(
            key=lambda x: (x["volume"], x["combined_score"],
                           x["buriedness"], -x["dist"]), reverse=True)
        return sh_close_candidates[0]
    if sh_single_candidates:
        sh_single_candidates.sort(
            key=lambda x: (x["volume"], x["buriedness"],
                           x["combined_score"]), reverse=True)
        return sh_single_candidates[0]
    return None


# =====================================================================
# SINGLE-PDB DRIVER (generalized — replaces the old multi-PDB BASE loop)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Script2 (Pipeline 1): Detection of binding-site hotspots, for a single PDB."
    )
    parser.add_argument(
        "--fields_dir",
        required=True,
        help="Path to the Fields_Pipeline1_<PDB_ID> folder containing the .mrc field files.",
    )
    parser.add_argument(
        "--pdb_file",
        required=True,
        help="Path to the fixed PDB structure file (e.g. 1AJU_fixed.pdb) in the current working directory.",
    )
    parser.add_argument(
        "--analysis_dir",
        required=True,
        help="Path to Analysis_Pipeline1_<PDB_ID> folder (output of Script1), used to read the selected isovalues CSV and to save all Script2 outputs.",
    )
    parser.add_argument(
        "--pdb_id",
        default=None,
        help="Optional override for the PDB identifier used in filenames. "
             "If not provided, it is automatically derived from --pdb_file (filename without extension).",
    )
    args = parser.parse_args()

    pdb_file_path = Path(args.pdb_file)
    pdb = args.pdb_id if args.pdb_id else pdb_file_path.stem

    fields_dir = Path(args.fields_dir)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    stk_file  = find_field_file(fields_dir, pdb, "stacking")
    hyd_file  = find_field_file(fields_dir, pdb, "hydrophobic")
    apbs_file = find_field_file(fields_dir, pdb, "apbs")
    pdb_file  = find_pdb_file(args.pdb_file, pdb)
    iso_table = find_iso_table(analysis_dir, pdb)

    summary_rows = []

    if stk_file is None or hyd_file is None or apbs_file is None:
        print(f"[SKIP] {pdb}: missing stacking, hydrophobic, or apbs .mrc in {fields_dir}")
        return
    if pdb_file is None:
        print(f"[SKIP] {pdb}: missing 3D structure file: {args.pdb_file}")
        return
    if iso_table is None:
        print(f"[SKIP] {pdb}: missing selected isovalues CSV in {analysis_dir}")
        return

    iso_df = pd.read_csv(iso_table).set_index("pdb")
    if pdb not in iso_df.index:
        print(f"[SKIP] {pdb}: missing row in {iso_table.name}")
        return

    row_iso  = iso_df.loc[pdb]
    stk_iso  = float(row_iso["stacking_iso"])    if "stacking_iso"    in row_iso else np.nan
    hyd_iso  = float(row_iso["hydrophobic_iso"]) if "hydrophobic_iso" in row_iso else np.nan
    apbs_iso = float(row_iso["apbs_iso"])        if "apbs_iso"        in row_iso else np.nan

    if not (np.isfinite(stk_iso) and np.isfinite(hyd_iso) and np.isfinite(apbs_iso)):
        print(f"[SKIP] {pdb}: invalid selected isovalues")
        return

    stk_data,  stk_voxel,  stk_origin  = load_mrc(stk_file)
    hyd_data,  hyd_voxel,  hyd_origin  = load_mrc(hyd_file)
    apbs_data, apbs_voxel, apbs_origin = load_mrc(apbs_file)

    if stk_data.shape != hyd_data.shape or stk_data.shape != apbs_data.shape:
        print(f"[SKIP] {pdb}: field grids have different shapes "
              f"stk={stk_data.shape} hyd={hyd_data.shape} apbs={apbs_data.shape}")
        return

    if not np.allclose(stk_voxel, hyd_voxel, atol=0.01) or \
       not np.allclose(stk_voxel, apbs_voxel, atol=0.01):
        print(f"[WARN] {pdb}: voxel sizes differ slightly (FP rounding) — "
              f"using stacking voxel as reference.\n"
              f"  stk={stk_voxel} hyd={hyd_voxel} apbs={apbs_voxel}")

    if not np.allclose(stk_origin, hyd_origin, atol=0.01) or \
       not np.allclose(stk_origin, apbs_origin, atol=0.01):
        print(f"[WARN] {pdb}: origins differ slightly (FP rounding) — "
              f"using stacking origin as reference.\n"
              f"  stk={stk_origin} hyd={hyd_origin} apbs={apbs_origin}")

    shape      = stk_data.shape
    voxel_size = stk_voxel
    origin     = stk_origin
    atom_xyz   = load_structure_atoms(pdb_file)

    stk_mask  = threshold_mask(stk_data,  stk_iso,  "stacking")
    hyd_mask  = threshold_mask(hyd_data,  hyd_iso,  "hydrophobic")
    apbs_mask = threshold_mask(apbs_data, apbs_iso, "apbs")

    save_mrc_from_mask(stk_mask,  stk_file,  analysis_dir / f"{pdb}.stacking_thresholded.mrc")
    save_mrc_from_mask(hyd_mask,  hyd_file,  analysis_dir / f"{pdb}.hydrophobic_thresholded.mrc")
    save_mrc_from_mask(apbs_mask, apbs_file, analysis_dir / f"{pdb}.apbs_thresholded.mrc")

    stk_labels,  stk_nlab  = connected_components(stk_mask)
    hyd_labels,  hyd_nlab  = connected_components(hyd_mask)
    apbs_labels, apbs_nlab = connected_components(apbs_mask)

    stk_comps  = component_stats(stk_labels,  stk_nlab,  stk_data,
                                 voxel_size, origin, atom_xyz, "stacking")
    hyd_comps  = component_stats(hyd_labels,  hyd_nlab,  hyd_data,
                                 voxel_size, origin, atom_xyz, "hydrophobic")
    apbs_comps = component_stats(apbs_labels, apbs_nlab, apbs_data,
                                 voxel_size, origin, atom_xyz, "apbs")

    # ── Pocket 1: mixed-fields ───────────────────────────────────────────────
    best_mixed = None
    best_mixed_mask = None
    mixed_overlap_candidates   = []
    mixed_proximity_candidates = []

    for scomp in stk_comps:
        for hcomp in hyd_comps:
            for acomp in apbs_comps:
                tri = score_three_field_triplet(scomp, hcomp, acomp, shape, voxel_size)
                if tri is None or tri["buriedness"] < REAL_BURIEDNESS_MIN:
                    continue

                mixed_union_mask = build_union_mask(shape, [scomp, hcomp, acomp])

                if tri["triple_overlap_voxels"] > 0:
                    mixed_mask_for_summary = tri["tri_mask"]
                    mixed_mask_for_save    = tri["tri_mask"]
                    mixed_selection_rule   = "mixed_triple_overlap_only"
                else:
                    mixed_mask_for_summary = mixed_union_mask
                    mixed_mask_for_save    = mixed_union_mask
                    mixed_selection_rule   = "mixed_proximity_union"

                pocket = summarize_pocket(
                    mixed_mask_for_summary, [scomp, hcomp, acomp], "mixed_fields",
                    shape, voxel_size, origin, atom_xyz,
                    stk_data, hyd_data, apbs_data)
                if pocket is None:
                    continue

                packed = {
                    "pocket":                pocket,
                    "mask":                  mixed_mask_for_save.copy(),
                    "volume":                int(np.count_nonzero(mixed_mask_for_save)),
                    "buriedness":            pocket["real_buriedness"],
                    "combined_score":        tri["score"] + 20.0 * pocket["pocket_score"],
                    "triple_overlap_voxels": tri["triple_overlap_voxels"],
                    "selection_rule":        mixed_selection_rule,
                }

                if tri["triple_overlap_voxels"] > 0:
                    mixed_overlap_candidates.append(packed)
                else:
                    mixed_proximity_candidates.append(packed)

    if mixed_overlap_candidates:
        mixed_overlap_candidates.sort(
            key=lambda x: (x["volume"], x["buriedness"],
                           x["combined_score"], x["triple_overlap_voxels"]),
            reverse=True)
        best = mixed_overlap_candidates[0]
    elif mixed_proximity_candidates:
        mixed_proximity_candidates.sort(
            key=lambda x: (x["volume"], x["buriedness"], x["combined_score"]),
            reverse=True)
        best = mixed_proximity_candidates[0]
    else:
        best = None

    if best is not None:
        best_mixed = best["pocket"]
        best_mixed["combined_score"] = best["combined_score"]
        best_mixed["selection_rule"] = best["selection_rule"]
        best_mixed_mask              = best["mask"]

    if best_mixed_mask is not None:
        save_mrc_from_mask(best_mixed_mask, stk_file,
                           analysis_dir / f"{pdb}.mixed_fields_pocket.mrc")

    # ── Pocket 2: stacking-electrostatic ────────────────────────────────────
    best_se = None; best_se_mask = None
    se_overlap_candidates = []; se_close_candidates = []

    for scomp in stk_comps:
        for acomp in apbs_comps:
            pair_score = score_two_field_pair(scomp, acomp, shape, voxel_size)
            if pair_score is None or pair_score["buriedness"] < REAL_BURIEDNESS_MIN:
                continue
            pair_mask, se_rule = build_stk_ele_pocket_mask(
                shape, scomp, acomp, voxel_size)
            if pair_mask is None or np.count_nonzero(pair_mask) == 0:
                continue
            pocket = summarize_pocket(
                pair_mask, [scomp, acomp], "stacking_electrostatic",
                shape, voxel_size, origin, atom_xyz,
                stk_data, hyd_data, apbs_data)
            if pocket is None:
                continue
            packed = {
                "pocket":         pocket,
                "mask":           pair_mask.copy(),
                "ov_nvox":        pair_score["ov_nvox"],
                "volume":         int(np.count_nonzero(pair_mask)),
                "buriedness":     pocket["real_buriedness"],
                "combined_score": pair_score["score"] + 20.0 * pocket["pocket_score"],
                "dist":           pair_score["dist"],
                "selection_rule": se_rule,
            }
            if pair_score["ov_nvox"] > 0:
                se_overlap_candidates.append(packed)
            else:
                se_close_candidates.append(packed)

    if se_overlap_candidates:
        se_overlap_candidates.sort(
            key=lambda x: (x["volume"], x["buriedness"],
                           x["combined_score"], x["ov_nvox"]), reverse=True)
        best = se_overlap_candidates[0]
    elif se_close_candidates:
        se_close_candidates.sort(
            key=lambda x: (x["volume"], x["buriedness"],
                           x["combined_score"], -x["dist"]), reverse=True)
        best = se_close_candidates[0]
    else:
        best = None

    if best is not None:
        best_se = best["pocket"]
        best_se["combined_score"] = best["combined_score"]
        best_se["selection_rule"] = best["selection_rule"]
        best_se_mask              = best["mask"]

    if best_se_mask is not None:
        save_mrc_from_mask(best_se_mask, stk_file,
                           analysis_dir / f"{pdb}.stacking_electrostatic_pocket.mrc")

    # ── Pocket 3: stacking-hydrophobic (first + second) ─────────────────────
    best_sh = None;   best_sh_mask   = None
    second_sh = None; second_sh_mask = None

    first_ranked_sh = select_best_sh_from_component_sets(
        stk_comps, hyd_comps, shape, voxel_size, origin, atom_xyz,
        stk_data, hyd_data, apbs_data,
        pocket_type="stacking_hydrophobic",
        enforce_structure_nearness=False,
    )
    if first_ranked_sh is not None:
        best_sh = first_ranked_sh["pocket"]
        best_sh["combined_score"] = first_ranked_sh["combined_score"]
        best_sh["selection_rule"] = first_ranked_sh["selection_rule"]
        best_sh_mask              = first_ranked_sh["mask"]

    if best_sh_mask is not None:
        save_mrc_from_mask(best_sh_mask, stk_file,
                           analysis_dir / f"{pdb}.stacking_hydrophobic_pocket.mrc")

    stk_comps_second = split_components_after_exclusion(
        stk_comps, best_sh_mask, stk_data,
        voxel_size, origin, atom_xyz, "stacking")
    hyd_comps_second = split_components_after_exclusion(
        hyd_comps, best_sh_mask, hyd_data,
        voxel_size, origin, atom_xyz, "hydrophobic")

    second_ranked_sh = select_best_sh_from_component_sets(
        stk_comps_second, hyd_comps_second, shape, voxel_size, origin, atom_xyz,
        stk_data, hyd_data, apbs_data,
        pocket_type="stacking_hydrophobic_second",
        enforce_structure_nearness=True,
        max_atom_distance=SECOND_POCKET_MAX_ATOM_DISTANCE_A,
    )
    if second_ranked_sh is not None:
        second_sh = second_ranked_sh["pocket"]
        second_sh["combined_score"] = second_ranked_sh["combined_score"]
        second_sh["selection_rule"] = second_ranked_sh["selection_rule"]
        second_sh_mask              = second_ranked_sh["mask"]

    if second_sh_mask is not None:
        save_mrc_from_mask(second_sh_mask, stk_file,
                           analysis_dir / f"{pdb}.stacking_hydrophobic_second_pocket.mrc")

    # ── Pocket 4: electrostatic ──────────────────────────────────────────────
    best_ele = None; best_ele_mask = None

    if apbs_comps:
        all_apbs_mask = build_union_mask(shape, apbs_comps)
        pocket = summarize_pocket(
            all_apbs_mask, apbs_comps, "electrostatic",
            shape, voxel_size, origin, atom_xyz,
            stk_data, hyd_data, apbs_data)
        if pocket is not None:
            best_ele = pocket
            best_ele["combined_score"] = 20.0 * pocket["pocket_score"]
            best_ele["selection_rule"] = "all_apbs_thresholded_components_union"
            best_ele_mask              = all_apbs_mask.copy()

    if best_ele_mask is not None:
        save_mrc_from_mask(best_ele_mask, stk_file,
                           analysis_dir / f"{pdb}.electrostatic_pocket.mrc")

    # ── Summary rows ─────────────────────────────────────────────────────────
    pocket_results = [
        ("mixed_fields",                best_mixed),
        ("stacking_electrostatic",      best_se),
        ("stacking_hydrophobic",        best_sh),
        ("stacking_hydrophobic_second", second_sh),
        ("electrostatic",               best_ele),
    ]

    per_pdb_rows = []
    apbs_method  = row_iso["apbs_method"] if "apbs_method" in row_iso else None

    for pocket_name, pocket in pocket_results:
        if pocket is None:
            row = {
                "pdb": pdb, "pocket_type": pocket_name,
                "stacking_iso": stk_iso, "hydrophobic_iso": hyd_iso,
                "apbs_iso": apbs_iso, "apbs_method": apbs_method,
                "predicted": False,
                "centroid_x": np.nan, "centroid_y": np.nan, "centroid_z": np.nan,
                "final_mask_voxels": 0, "real_buriedness": np.nan,
                "stacking_rel": np.nan, "hydrophobic_rel": np.nan,
                "apbs_rel": np.nan,
                "stacking_rel_norm": np.nan, "hydrophobic_rel_norm": np.nan,
                "apbs_rel_norm": np.nan, "pocket_score": np.nan,
                "ligand_type": None, "selection_rule": None,
            }
        else:
            row = {
                "pdb": pdb, "pocket_type": pocket_name,
                "stacking_iso": stk_iso, "hydrophobic_iso": hyd_iso,
                "apbs_iso": apbs_iso, "apbs_method": apbs_method,
                "predicted": True,
                "centroid_x":           pocket["centroid_x"],
                "centroid_y":           pocket["centroid_y"],
                "centroid_z":           pocket["centroid_z"],
                "final_mask_voxels":    pocket["nvox"],
                "real_buriedness":      pocket["real_buriedness"],
                "stacking_rel":         pocket["stacking_rel"],
                "hydrophobic_rel":      pocket["hydrophobic_rel"],
                "apbs_rel":             pocket["apbs_rel"],
                "stacking_rel_norm":    pocket["stacking_rel_norm"],
                "hydrophobic_rel_norm": pocket["hydrophobic_rel_norm"],
                "apbs_rel_norm":        pocket["apbs_rel_norm"],
                "pocket_score":         pocket["pocket_score"],
                "ligand_type":          pocket["ligand_type"],
                "selection_rule":       pocket.get("selection_rule", None),
            }
        summary_rows.append(row)
        per_pdb_rows.append(row)

    per_pdb_df = pd.DataFrame(per_pdb_rows).sort_values(["pdb", "pocket_type"])
    per_pdb_df.to_csv(
        analysis_dir / f"{pdb}.predicted_binding_pockets_summary.csv", index=False)

    print(f"Script2 complete for {pdb}. Results saved in: {analysis_dir}")
    print("Saved files:")
    for f in [
        f"{pdb}.stacking_thresholded.mrc",
        f"{pdb}.hydrophobic_thresholded.mrc",
        f"{pdb}.apbs_thresholded.mrc",
        f"{pdb}.mixed_fields_pocket.mrc",
        f"{pdb}.stacking_electrostatic_pocket.mrc",
        f"{pdb}.stacking_hydrophobic_pocket.mrc",
        f"{pdb}.stacking_hydrophobic_second_pocket.mrc",
        f"{pdb}.electrostatic_pocket.mrc",
        f"{pdb}.predicted_binding_pockets_summary.csv",
    ]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
