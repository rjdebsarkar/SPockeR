#!/usr/bin/env python3
"""
Unique Pocket Volume Generation & Field Contributions
=================================================================
For a single PDB, using the outputs of Pipeline 1, Pipeline 2, and the
trimmed scoring fields:

  POCKET VOLUME FORMATION RULES
  ──────────────────────────────
  • Detects pocket MRC files for:
      Mixed field, STK-ELE, First ranked STK-HPb,
      Second ranked STK-HPb, Third ranked STK-HPb,
      Fourth ranked STK-HPb, HBond Site1, HBond Site2
  • NOTE: The ELE pocket (*.electrostatic_pocket.mrc) is intentionally
      EXCLUDED from pocket volume formation. Including ELE causes pockets
      to become unrealistically large and can merge spatially distant
      sites (e.g. HBond Site1 and HBond Site2) via spurious ELE overlap.
      ELE volumetric information is therefore NOT used in overlap
      detection, clustering, or merged mask construction.
  • Clusters overlapping pocket masks using STRICT mutual-overlap logic:
      every pair inside a merged group must satisfy bidirectional
      overlap ≥20% of each mask's own voxel count.
  • Merges each accepted group mask by voxel-wise union of all members.
  • After merging, each pocket mask is TRIMMED: any voxel whose nearest
    RNA heavy atom is > 5 Å away is removed from the pocket volume.
  • Re-evaluates the TRIMMED merged masks and iteratively merges the strongest
    remaining pair whenever both bidirectional overlap fractions are ≥20%.
    This closes the group-union gap left by the strict source-level clique
    analysis without changing the established 20% significance criterion.
  • After trimming, each pocket is checked for proximity to RNA TERMINAL
    RESIDUES.  A voxel is terminal-associated only when it lies within
    TERMINAL_ZONE_CUTOFF_A of a terminal atom AND that terminal atom is
    closer than every non-terminal RNA atom.  This prevents a genuine
    multi-chain/interface pocket from being rejected merely because the end
    of one chain happens to be nearby.
  • A pocket is also discarded when its representative core (the occupied
    voxel nearest the geometric centre) lies within the terminal cutoff and
    is closer to terminal than non-terminal RNA atoms.
    This catches terminal-centred expanded volumes whose whole-mask terminal
    fraction is diluted by pocket construction.
  • After preliminary scoring, any residual overlap below the 20% merge
    criterion is assigned to the higher-scoring pocket.  Scores and ranks are
    then recomputed, so the saved final masks are strictly voxel-disjoint.
  • Saves each accepted trimmed mask as:
        <output_dir>/<pdb>.Pocket1_Volume.mrc
        <output_dir>/<pdb>.Pocket2_Volume.mrc  … etc.
    Pockets are numbered highest-score-first (Pocket1 = highest score).

  TERMINAL POCKET FILTERING
  ──────────────────────────
  • RNA residues are separated from protein/ligand/water/ion residues before
    terminal analysis.  Terminal residues are the FIRST and LAST RNA residues
    in file/polymer order for every RNA chain (residue insertion codes are
    preserved; numerical min/max residue numbering is not used).
  • Chains with fewer than MIN_RNA_CHAIN_RESIDUES_FOR_TERMINAL_FILTER residues
    are not used for terminal rejection because they have no meaningful
    terminal-versus-interior distinction.
  • A pocket is considered a TERMINAL POCKET (and therefore discarded)
    if MORE THAN terminal_voxel_fraction (default 0.60) of its voxels
    are terminal-associated: within TERMINAL_ZONE_CUTOFF_A (default 5.0 Å)
    of a terminal atom and closer to terminal than non-terminal RNA atoms.
  • This step occurs AFTER distance-based trimming but BEFORE scoring,
    so only genuine interior pockets proceed to field weight calculation.

  SCORING & FIELD CONTRIBUTIONS
  ──────────────────────────────
  • Reads field files (from --fields_dir) used for weight calculation and scoring:

      Field         File used for scoring                        Internal name
      ──────────    ──────────────────────────────────────────── ─────────────
      APBS          <pdb>.apbs_rna_trimmed.mrc                  "apbs"
                    (fallback: <pdb>.apbs.mrc if trimmed not found)
      Hydrophobic   <pdb>.hydrophobic_nonoverlap_trimmed.mrc    "hydrophobic"
                    (fallback: <pdb>.hydrophobic.mrc if trimmed not found)
      HB-Acceptors  <pdb>.hbacceptors.mrc                       "hbacceptors"
      HB-Donors     <pdb>.hbdonors.mrc                          "hbdonors"
      Stacking      <pdb>.stacking.mrc                          "stacking"

  • NORMALIZATION RULE:
        – Stacking and Hydrophobic are BOTH normalised by the COMBINED total:
              stk_hyd_combined_total = total_grid_integral(stacking)
                                     + total_grid_integral(hydrophobic)
          where hydrophobic uses hydrophobic_nonoverlap_trimmed.mrc.
        – APBS, HB-Donors, HB-Acceptors are each normalised by their OWN
          total grid integral (unchanged).

  • For each unique pocket and each field:
        field_integral_in_pocket  = Σ |field_values| at voxels inside the
                                    TRIMMED pocket mask
        normalization_constant    = stk_hyd_combined_total  (stacking & hydrophobic)
                                  = own total grid integral  (all other fields)
        field_weight              = field_integral_in_pocket / normalization_constant
  • pocket_score (raw) = Σ_fields  field_weight[field]
  • Scores are normalised so they sum to 1.0 across all pockets of that PDB.
  • Pockets are ordered by score (highest first): Pocket1 = highest score.

  OUTPUTS (inside --output_dir, e.g. Analysis_Unique_Pockets_<PDB>)
  ──────────────────────────────────────────
  • Previous MRC/CSV/PNG outputs for the same PDB identifier are removed after
    input validation and before analysis, preventing stale-result reuse.
  • <pdb>.Pocket1_Volume.mrc, <pdb>.Pocket2_Volume.mrc, …
  • <pdb>_field_contributions.csv
        columns: pocket_index, pocket_name, pocket_score, pocket_volume_A3,
                 contributing_fields,
                 <field>_integral_in_pocket, <field>_normalization_constant,
                 <field>_ratio
                 (for stacking, hydrophobic, hbdonors, hbacceptors, apbs)
  • <pdb>_field_contributions.png  – stacked bar chart of field weights
        (Times New Roman fonts, 300 dpi, upright x-axis labels)
"""

import argparse
from pathlib import Path
import csv
from collections import defaultdict
from itertools import combinations
import shlex

import mrcfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree

# -------------------------------------------------
# GLOBAL FONT: Times New Roman throughout all plots (UNCHANGED)
# -------------------------------------------------
matplotlib.rcParams["font.family"]      = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"]      = "Times New Roman"
matplotlib.rcParams["mathtext.it"]      = "Times New Roman:italic"
matplotlib.rcParams["mathtext.bf"]      = "Times New Roman:bold"

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------
BINARY_EPS                   = 1e-12
MAX_HBOND_SITES              = 2
SIGNIFICANT_OVERLAP_FRACTION = 0.20
RNA_TRIM_CUTOFF_A            = 5.0

# Terminal-pocket filtering parameters
TERMINAL_ZONE_CUTOFF_A       = 5.0   # Å radius around terminal-residue atoms
TERMINAL_VOXEL_FRACTION      = 0.60  # discard pocket if >60% of voxels are in terminal zone
TERMINAL_CENTRE_CUTOFF_A     = 5.0   # terminal-dominant pocket-core cutoff
MIN_RNA_CHAIN_RESIDUES_FOR_TERMINAL_FILTER = 3

# Standard residue names expected after 0_fix_pdb.sh/PDBFixer.  The structural
# ribose-atom test below additionally recognises RNA residues whose names are
# not in this compact set, while excluding ordinary amino-acid ATOM records.
RNA_RESIDUE_NAMES = {
    "A", "C", "G", "U", "I",
    "RA", "RC", "RG", "RU", "RI",
    "ADE", "CYT", "GUA", "URI", "URA",
}
RNA_RIBOSE_SIGNATURE = {"C1'", "C2'", "C3'", "C4'", "O2'", "O3'", "O4'"}

RAW_FIELD_SPECS = [
    (".apbs_rna_trimmed.mrc",                ".apbs.mrc",         "apbs"),
    (".hydrophobic_nonoverlap_trimmed.mrc", ".hydrophobic.mrc",  "hydrophobic"),
    (".hbacceptors.mrc",                     None,                 "hbacceptors"),
    (".hbdonors.mrc",                        None,                 "hbdonors"),
    (".stacking.mrc",                        None,                 "stacking"),
]

FIELD_NAMES_CSV = ["stacking", "hydrophobic", "hbdonors", "hbacceptors", "apbs"]

FIELD_COLORS = {
    "stacking":    "#55dd55",
    "hydrophobic": "#88ccff",
    "hbdonors":    "#ff9933",
    "hbacceptors": "#9933cc",
    "apbs":        "#2244cc",
}
FIELD_LABELS_DISPLAY = {
    "stacking":    "Stacking",
    "hydrophobic": "Hydrophobic",
    "hbdonors":    "HB-Donors",
    "hbacceptors": "HB-Acceptors",
    "apbs":        "APBS (magnitude)",
}

POCKET_SPECS = [
    (".mixed_fields.final_trimmed_smif_identical_rna_only_pocket.mrc",  "Mixed field"),
    (".stk_ele.final_trimmed_smif_identical_rna_only_pocket.mrc",        "STK-ELE"),
    (".stk_hpb_first.final_trimmed_smif_identical_rna_only_pocket.mrc",  "First ranked STK-HPb"),
    (".stk_hpb_second.final_trimmed_smif_identical_rna_only_pocket.mrc", "Second ranked STK-HPb"),
    (".stk_hpb_third.final_trimmed_smif_identical_rna_only_pocket.mrc",  "Third ranked STK-HPb"),
    (".stk_hpb_fourth.final_trimmed_smif_identical_rna_only_pocket.mrc", "Fourth ranked STK-HPb"),
]


# -------------------------------------------------
# MRC I/O (UNCHANGED)
# -------------------------------------------------
def _load_mrc_data(path):
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        data_raw = np.asarray(mrc.data, dtype=np.float32).copy()
        vx = float(mrc.voxel_size.x) or 1.0
        vy = float(mrc.voxel_size.y) or 1.0
        vz = float(mrc.voxel_size.z) or 1.0
        ox = float(mrc.header.origin.x)
        oy = float(mrc.header.origin.y)
        oz = float(mrc.header.origin.z)
        if ox == 0.0 and oy == 0.0 and oz == 0.0:
            ox = float(mrc.header.nxstart) * vx
            oy = float(mrc.header.nystart) * vy
            oz = float(mrc.header.nzstart) * vz
        mapc = int(mrc.header.mapc)
        mapr = int(mrc.header.mapr)
        maps = int(mrc.header.maps)
        cart_of_axis = {0: maps, 1: mapr, 2: mapc}
        axis_of_cart = {v: k for k, v in cart_of_axis.items()}
        try:
            perm = (axis_of_cart[1], axis_of_cart[2], axis_of_cart[3])
            data = np.transpose(data_raw, perm)
        except KeyError:
            data = np.transpose(data_raw, (2, 1, 0))
    return data, np.array([vx, vy, vz], dtype=float), np.array([ox, oy, oz], dtype=float)


def load_mrc_mask(path):
    data, voxel, origin = _load_mrc_data(path)
    mask = np.isfinite(data) & (data > BINARY_EPS)
    return mask, voxel, origin


def load_mrc_mask_hbond(path):
    data, voxel, origin = _load_mrc_data(path)
    mask = np.isfinite(data) & (np.abs(data) > BINARY_EPS)
    return mask, voxel, origin


def write_mrc(path, data, voxel, origin):
    data = np.asarray(data, dtype=np.float32)
    data_out = np.transpose(data, (2, 1, 0))
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data_out)
        mrc.voxel_size = (float(voxel[0]), float(voxel[1]), float(voxel[2]))
        mrc.header.mapc = 1
        mrc.header.mapr = 2
        mrc.header.maps = 3
        try:
            mrc.header.origin.x = float(origin[0])
            mrc.header.origin.y = float(origin[1])
            mrc.header.origin.z = float(origin[2])
        except Exception:
            pass
        mrc.update_header_from_data()
        mrc.update_header_stats()


def trim_mask_by_rna_distance(mask, voxel, origin, atom_xyz,
                              cutoff_a=RNA_TRIM_CUTOFF_A):
    if mask is None or not np.any(mask):
        return mask
    if atom_xyz is None or len(atom_xyz) == 0:
        print("  [WARN]  No RNA atoms available for distance trimming — pocket mask returned untrimmed.")
        return mask

    coords = np.column_stack(np.where(mask))
    world = coords * voxel[None, :] + origin[None, :]

    tree = cKDTree(atom_xyz)
    dists, _ = tree.query(world, k=1, workers=-1)

    keep = dists <= cutoff_a
    trimmed = np.zeros_like(mask, dtype=bool)
    trimmed[coords[keep, 0], coords[keep, 1], coords[keep, 2]] = True
    return trimmed


def compute_pocket_volume_A3(mask, voxel):
    if mask is None:
        return 0.0
    voxel_vol = float(voxel[0]) * float(voxel[1]) * float(voxel[2])
    return float(np.count_nonzero(mask)) * voxel_vol


# -------------------------------------------------
# HBOND HELPERS (MODIFIED — hbond_dir supplied directly, no search)
# -------------------------------------------------
def collect_hbond_sites(pdb, hbond_dir):
    sites = []
    stems = [pdb]
    for n in range(1, MAX_HBOND_SITES + 1):
        site_mrc = None
        for stem in stems:
            c = hbond_dir / f"{stem}.HBond_Site_{n}.mrc"
            if c.exists():
                site_mrc = c
                break
        if site_mrc is None:
            break
        label = f"HBond Site{n}"
        vol_mrc = None
        for stem in stems:
            c = hbond_dir / f"{stem}.HBond_Site_{n}_pocket_volume.mrc"
            if c.exists():
                vol_mrc = c
                break
        sites.append({"label": label, "site_mrc": site_mrc, "vol_mrc": vol_mrc})
        print(f"  [HBond] Found site file : {site_mrc}")
        if vol_mrc:
            print(f"  [HBond] Found vol  file : {vol_mrc}")
        else:
            print(f"  [HBond] WARNING: no pocket_volume file for {label}")
    return sites


def load_all_label_masks(pdb, analysis1_dir, hbond_dir):
    all_found_labels = []
    label_masks = {}
    label_voxels = {}
    label_origins = {}

    for suffix, label in POCKET_SPECS:
        f = analysis1_dir / f"{pdb}{suffix}"
        if not f.exists():
            continue
        try:
            mask, voxel, origin = load_mrc_mask(f)
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
            continue
        if not np.any(mask):
            continue
        all_found_labels.append(label)
        label_masks[label] = mask
        label_voxels[label] = voxel
        label_origins[label] = origin

    hbond_sites = collect_hbond_sites(pdb, hbond_dir) if hbond_dir else []
    for site in hbond_sites:
        label = site["label"]
        use_mrc = site["vol_mrc"] if (site["vol_mrc"] and site["vol_mrc"].exists()) else site["site_mrc"]
        try:
            mask, voxel, origin = load_mrc_mask_hbond(use_mrc)
        except Exception as e:
            print(f"  [ERROR] loading {use_mrc.name}: {e}")
            continue
        if not np.any(mask):
            print(f"  [HBond] {label} mask empty for {pdb}")
            continue
        all_found_labels.append(label)
        label_masks[label] = mask
        label_voxels[label] = voxel
        label_origins[label] = origin
        print(f"  [HBond] {pdb} {label}: active voxels = {int(np.sum(mask))}")

    return all_found_labels, label_masks, label_voxels, label_origins


def align_masks_to_common_grid(all_found_labels, label_masks, label_voxels,
                                label_origins):
    """
    Pocket masks are produced by independent MRC-generation runs (whole-
    structure fields vs. the HBond-specific fields sub-pipeline), so they
    can live on grids with different shapes/origins even when the voxel
    size matches. Boolean ops (overlap, union) require identical shapes,
    so re-embed every mask into one common world-space grid before any
    mask-vs-mask comparison happens.
    """
    if not all_found_labels:
        return label_masks, label_voxels, label_origins

    voxel_ref = None
    for lbl in all_found_labels:
        v = label_voxels[lbl]
        if voxel_ref is None:
            voxel_ref = v
        elif not np.allclose(v, voxel_ref, atol=1e-3):
            raise ValueError(
                f"Cannot align pocket masks: '{lbl}' has voxel size {v}, "
                f"expected {voxel_ref} (masks must share a voxel size to "
                f"be merged onto a common grid)"
            )

    origins = np.array([label_origins[lbl] for lbl in all_found_labels], dtype=float)
    shapes = np.array([label_masks[lbl].shape for lbl in all_found_labels], dtype=int)

    global_origin = origins.min(axis=0)
    offsets = np.rint((origins - global_origin[None, :]) / voxel_ref[None, :]).astype(int)
    global_shape = tuple((offsets + shapes).max(axis=0))

    aligned_masks = {}
    for lbl, offset, shape in zip(all_found_labels, offsets, shapes):
        full = np.zeros(global_shape, dtype=bool)
        sl = tuple(slice(o, o + s) for o, s in zip(offset, shape))
        full[sl] = label_masks[lbl]
        aligned_masks[lbl] = full

    aligned_voxels = {lbl: voxel_ref for lbl in all_found_labels}
    aligned_origins = {lbl: global_origin for lbl in all_found_labels}
    return aligned_masks, aligned_voxels, aligned_origins


# -------------------------------------------------
# SIGNIFICANT OVERLAP (UNCHANGED)
# -------------------------------------------------
def compute_pairwise_overlap_metrics(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return None

    size_a = int(np.count_nonzero(mask_a))
    size_b = int(np.count_nonzero(mask_b))
    if size_a == 0 or size_b == 0:
        return None

    intersection = int(np.count_nonzero(mask_a & mask_b))
    if intersection == 0:
        return {
            "size_a": size_a,
            "size_b": size_b,
            "intersection": 0,
            "frac_a": 0.0,
            "frac_b": 0.0,
        }

    return {
        "size_a": size_a,
        "size_b": size_b,
        "intersection": intersection,
        "frac_a": intersection / size_a,
        "frac_b": intersection / size_b,
    }


def masks_significantly_overlap(mask_a, mask_b,
                                threshold=SIGNIFICANT_OVERLAP_FRACTION):
    metrics = compute_pairwise_overlap_metrics(mask_a, mask_b)
    if metrics is None:
        return False
    return (metrics["frac_a"] >= threshold) and (metrics["frac_b"] >= threshold)


# -------------------------------------------------
# COMPLETE-LINKAGE / MAXIMAL CLIQUE GROUPING (UNCHANGED)
# -------------------------------------------------
def _build_overlap_adjacency(all_found_labels, label_masks):
    n = len(all_found_labels)
    adj = np.zeros((n, n), dtype=bool)
    pair_metrics = {}

    for i in range(n):
        adj[i, i] = True
        for j in range(i + 1, n):
            li, lj = all_found_labels[i], all_found_labels[j]
            metrics = compute_pairwise_overlap_metrics(
                label_masks.get(li), label_masks.get(lj)
            )
            pair_metrics[(i, j)] = metrics
            if metrics is not None and \
               (metrics["frac_a"] >= SIGNIFICANT_OVERLAP_FRACTION) and \
               (metrics["frac_b"] >= SIGNIFICANT_OVERLAP_FRACTION):
                adj[i, j] = True
                adj[j, i] = True
    return adj, pair_metrics


def _neighbors(adj, v):
    return {i for i, flag in enumerate(adj[v]) if flag and i != v}


def _bron_kerbosch(R, P, X, adj, cliques):
    if not P and not X:
        if R:
            cliques.append(set(R))
        return

    if P or X:
        pivot = max(P | X, key=lambda v: len(_neighbors(adj, v) & P))
        pivot_neighbors = _neighbors(adj, pivot)
    else:
        pivot_neighbors = set()

    for v in list(P - pivot_neighbors):
        neigh_v = _neighbors(adj, v)
        _bron_kerbosch(R | {v}, P & neigh_v, X & neigh_v, adj, cliques)
        P.remove(v)
        X.add(v)


def _maximal_cliques_from_adjacency(adj):
    cliques = []
    _bron_kerbosch(set(), set(range(len(adj))), set(), adj, cliques)
    maximal = []
    for c in cliques:
        if not any(c < other for other in cliques):
            maximal.append(c)
    return maximal


def _assign_disjoint_groups_from_cliques(maximal_cliques):
    assigned = set()
    groups = []
    for clique in sorted(maximal_cliques, key=lambda c: (-len(c), tuple(sorted(c)))):
        available = clique - assigned
        if available:
            groups.append(sorted(available))
            assigned |= available
    return groups, assigned


# -------------------------------------------------
# BUILD UNIQUE POCKETS (UNCHANGED)
# -------------------------------------------------
def build_unique_pockets(all_found_labels, label_masks, label_voxels,
                         label_origins, atom_xyz):
    n = len(all_found_labels)
    if n == 0:
        return []

    adj, pair_metrics = _build_overlap_adjacency(all_found_labels, label_masks)

    print("  Pairwise overlap check (bidirectional fractions):")
    for i, j in combinations(range(n), 2):
        li, lj = all_found_labels[i], all_found_labels[j]
        metrics = pair_metrics.get((i, j))
        if metrics is None:
            print(f"    {li} ↔ {lj}: overlap could not be evaluated")
            continue
        print(
            f"    {li} ↔ {lj}: "
            f"intersection={metrics['intersection']} voxels; "
            f"{li} covered={100.0 * metrics['frac_a']:.1f}%; "
            f"{lj} covered={100.0 * metrics['frac_b']:.1f}%"
            + ("  [MERGE-ELIGIBLE]" if adj[i, j] else "  [SEPARATE]")
        )

    maximal_cliques = _maximal_cliques_from_adjacency(adj)
    groups, assigned = _assign_disjoint_groups_from_cliques(maximal_cliques)

    for i in range(n):
        if i not in assigned:
            groups.append([i])

    groups = sorted(groups, key=lambda g: (g[0], len(g)))

    raw_pockets = []
    for indices in groups:
        constituent_labels = [all_found_labels[i] for i in indices]
        contributing_fields = ", ".join(constituent_labels)

        merged_mask = None
        ref_voxel = None
        ref_origin = None

        for lbl in constituent_labels:
            m = label_masks.get(lbl)
            if m is None:
                continue
            if merged_mask is None:
                merged_mask = m.copy().astype(bool)
                ref_voxel = label_voxels[lbl]
                ref_origin = label_origins[lbl]
            else:
                merged_mask |= m.astype(bool)

        if merged_mask is not None and ref_voxel is not None:
            n_before = int(np.count_nonzero(merged_mask))
            merged_mask = trim_mask_by_rna_distance(
                merged_mask, ref_voxel, ref_origin, atom_xyz,
                cutoff_a=RNA_TRIM_CUTOFF_A)
            n_after = int(np.count_nonzero(merged_mask))
            if n_before != n_after:
                print(f"  [Trim]  {contributing_fields}: "
                      f"{n_before} → {n_after} voxels "
                      f"({n_before - n_after} removed beyond {RNA_TRIM_CUTOFF_A} Å)")

        n_voxels = int(np.count_nonzero(merged_mask)) if merged_mask is not None else 0

        raw_pockets.append({
            "labels": constituent_labels,
            "contributing_fields": contributing_fields,
            "merged_mask": merged_mask,
            "voxel": ref_voxel,
            "origin": ref_origin,
            "n_voxels": n_voxels,
            "name": "",
            "filename": "",
        })

    return raw_pockets


def _merge_two_pocket_dicts(pocket_a, pocket_b):
    """Return the voxel-wise union of two already aligned pocket dictionaries."""
    if not np.allclose(pocket_a["voxel"], pocket_b["voxel"], atol=1e-6) or \
       not np.allclose(pocket_a["origin"], pocket_b["origin"], atol=1e-6):
        raise ValueError("Cannot consolidate pockets on different grids")

    labels = []
    for label in pocket_a.get("labels", []) + pocket_b.get("labels", []):
        if label not in labels:
            labels.append(label)

    merged_mask = (pocket_a["merged_mask"].astype(bool) |
                   pocket_b["merged_mask"].astype(bool))
    return {
        "labels": labels,
        "contributing_fields": ", ".join(labels),
        "merged_mask": merged_mask,
        "voxel": pocket_a["voxel"],
        "origin": pocket_a["origin"],
        "n_voxels": int(np.count_nonzero(merged_mask)),
        "name": "",
        "filename": "",
    }


def consolidate_trimmed_pocket_unions(
        pockets, threshold=SIGNIFICANT_OVERLAP_FRACTION):
    """
    Iteratively merge significantly overlapping *group unions*.

    The initial complete-linkage grouping compares individual source masks.
    Once several sources have been unioned, an ungrouped candidate can overlap
    the final union significantly without overlapping every source member.
    This deterministic agglomerative pass closes that gap after RNA trimming.
    At each iteration the strongest eligible pair is merged first.
    """
    work = list(pockets)
    print("  Post-trimming group-union overlap consolidation:")

    while True:
        eligible = []
        for i, j in combinations(range(len(work)), 2):
            metrics = compute_pairwise_overlap_metrics(
                work[i]["merged_mask"], work[j]["merged_mask"])
            if metrics is None:
                continue
            if (metrics["frac_a"] >= threshold and
                    metrics["frac_b"] >= threshold):
                # Strongest mutual coverage first; remaining terms provide
                # deterministic tie-breaking without favouring input order.
                strength = (min(metrics["frac_a"], metrics["frac_b"]),
                            max(metrics["frac_a"], metrics["frac_b"]),
                            metrics["intersection"], -i, -j)
                eligible.append((strength, i, j, metrics))

        if not eligible:
            break

        _, i, j, metrics = max(eligible, key=lambda item: item[0])
        label_i = work[i]["contributing_fields"]
        label_j = work[j]["contributing_fields"]
        print(
            f"    MERGE group unions: ({label_i}) + ({label_j}); "
            f"intersection={metrics['intersection']} voxels; "
            f"coverage={100.0 * metrics['frac_a']:.1f}%/"
            f"{100.0 * metrics['frac_b']:.1f}%"
        )
        merged = _merge_two_pocket_dicts(work[i], work[j])
        work = [p for k, p in enumerate(work) if k not in (i, j)]
        work.append(merged)

    if len(work) == len(pockets):
        print("    No additional significant group-union overlaps.")
    else:
        print(f"    Consolidated {len(pockets)} trimmed groups into "
              f"{len(work)} unique group unions.")
    return work


# -------------------------------------------------
# RNA STRUCTURE AND TERMINAL-RESIDUE IDENTIFICATION
# -------------------------------------------------
def _is_hydrogen(atom_name):
    name = atom_name.strip().lstrip("0123456789")
    return name.startswith("H") or name.startswith("D")


def _normalise_atom_name(atom_name):
    """Use prime notation consistently (some PDB files use * for ribose atoms)."""
    return atom_name.strip().upper().replace("*", "'")


def _new_residue(chain_key, chain_label, resseq, icode, resname):
    return {
        "chain_key": chain_key,
        "chain_label": chain_label,
        "resseq": str(resseq).strip(),
        "icode": "" if str(icode).strip() in (".", "?") else str(icode).strip(),
        "resname": str(resname).strip().upper(),
        "atom_names": set(),
        "atoms": [],
        "has_atom_record": False,
    }


def _residue_is_rna(residue):
    """Recognise standard/fixed RNA while rejecting protein ATOM residues."""
    if residue["resname"] in RNA_RESIDUE_NAMES:
        return True
    # Structural fallback for renamed RNA residues.  Requiring an ATOM record,
    # O2' and at least five ribose atoms avoids classifying free nucleotide-like
    # HETATM ligands as part of the RNA polymer.
    ribose = residue["atom_names"] & RNA_RIBOSE_SIGNATURE
    return (residue["has_atom_record"] and "O2'" in ribose and len(ribose) >= 5)


def _append_atom(residue, record_type, atom_name, xyz):
    residue["atom_names"].add(_normalise_atom_name(atom_name))
    residue["atoms"].append(tuple(float(v) for v in xyz))
    residue["has_atom_record"] |= (record_type == "ATOM")


def _parse_pdb_residues(fp):
    """Return residues in PDB file/polymer order, respecting TER boundaries."""
    residues = {}
    ordered_keys = []
    segment = 0

    with open(fp, "r") as fh:
        for line in fh:
            record_type = line[:6].strip().upper()
            if record_type == "TER":
                segment += 1
                continue
            if record_type not in ("ATOM", "HETATM"):
                continue

            atom_name = line[12:16].strip()
            altloc = line[16].strip()
            if _is_hydrogen(atom_name) or altloc not in ("", "A", "1"):
                continue

            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip()
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue

            chain_key = (chain, segment)
            key = (chain_key, resseq, icode, resname)
            if key not in residues:
                residues[key] = _new_residue(
                    chain_key, chain, resseq, icode, resname)
                ordered_keys.append(key)
            _append_atom(residues[key], record_type, atom_name, xyz)

    return [residues[key] for key in ordered_keys]


def _parse_cif_residues(fp):
    """Parse the atom_site loop of a conventional coordinate mmCIF file."""
    headers, rows = [], []
    in_loop = False
    in_atom = False
    with open(fp, "r") as fh:
        for line in fh:
            s = line.strip()
            if s == "loop_":
                if in_atom and headers and rows:
                    break
                in_loop, in_atom = True, False
                headers, rows = [], []
                continue
            if in_loop and s.startswith("_atom_site."):
                headers.append(s)
                in_atom = True
                continue
            if in_atom and s and not s.startswith("_") and not s.startswith("#"):
                rows.append(shlex.split(s, posix=True))
                continue
            if in_atom and (s.startswith("#") or
                            (s.startswith("_") and not s.startswith("_atom_site."))):
                break

    n2i = {name: i for i, name in enumerate(headers)}
    required = ("_atom_site.Cartn_x", "_atom_site.Cartn_y",
                "_atom_site.Cartn_z", "_atom_site.label_atom_id")
    if not rows or not all(name in n2i for name in required):
        return []

    def column(*names):
        for name in names:
            if name in n2i:
                return n2i[name]
        return None

    ix_group = column("_atom_site.group_PDB")
    ix_atom = n2i["_atom_site.label_atom_id"]
    ix_alt = column("_atom_site.label_alt_id", "_atom_site.auth_alt_id")
    ix_resname = column("_atom_site.auth_comp_id", "_atom_site.label_comp_id")
    ix_chain = column("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
    ix_seq = column("_atom_site.auth_seq_id", "_atom_site.label_seq_id")
    ix_icode = column("_atom_site.pdbx_PDB_ins_code")
    ix_x, ix_y, ix_z = (n2i["_atom_site.Cartn_x"],
                         n2i["_atom_site.Cartn_y"],
                         n2i["_atom_site.Cartn_z"])

    residues = {}
    ordered_keys = []
    for parts in rows:
        try:
            record_type = (parts[ix_group].upper() if ix_group is not None
                           else "ATOM")
            if record_type not in ("ATOM", "HETATM"):
                continue
            atom_name = parts[ix_atom]
            altloc = parts[ix_alt] if ix_alt is not None else "."
            if _is_hydrogen(atom_name) or altloc not in (".", "?", "A", "1"):
                continue
            chain = parts[ix_chain].strip() if ix_chain is not None else "_"
            chain = chain if chain not in (".", "?") else "_"
            resseq = parts[ix_seq] if ix_seq is not None else "0"
            icode = parts[ix_icode] if ix_icode is not None else ""
            resname = parts[ix_resname] if ix_resname is not None else ""
            xyz = (float(parts[ix_x]), float(parts[ix_y]), float(parts[ix_z]))
        except (ValueError, IndexError):
            continue

        chain_key = (chain, 0)
        key = (chain_key, resseq, icode, resname)
        if key not in residues:
            residues[key] = _new_residue(
                chain_key, chain, resseq, icode, resname)
            ordered_keys.append(key)
        _append_atom(residues[key], record_type, atom_name, xyz)

    return [residues[key] for key in ordered_keys]


def _residue_label(residue):
    return (f"{residue['chain_label']}:{residue['resname']}"
            f"{residue['resseq']}{residue['icode']}")


def load_rna_structure(pdb_file):
    """
    Return (all RNA atoms, terminal RNA atoms, non-terminal RNA atoms).

    Only RNA residues participate in RNA-distance trimming and terminal-pocket
    rejection.  First/last residues are selected independently for each chain
    in file/polymer order, so single-chain and multi-chain structures use the
    same logic and insertion codes/irregular numbering remain valid.
    """
    fp = Path(pdb_file)
    empty = np.zeros((0, 3), dtype=float)
    if not fp.exists():
        print(f"  [WARN]  Structure file not found: {fp}")
        return empty, empty.copy(), empty.copy()

    try:
        if fp.suffix.lower() == ".pdb":
            residues = _parse_pdb_residues(fp)
        elif fp.suffix.lower() in (".cif", ".mmcif"):
            residues = _parse_cif_residues(fp)
        else:
            raise ValueError(f"unsupported structure extension '{fp.suffix}'")
    except Exception as exc:
        print(f"  [WARN]  Could not parse RNA structure from {fp.name}: {exc}")
        return empty, empty.copy(), empty.copy()

    chains = defaultdict(list)
    chain_order = []
    for residue in residues:
        if not _residue_is_rna(residue):
            continue
        key = residue["chain_key"]
        if key not in chains:
            chain_order.append(key)
        chains[key].append(residue)

    if not chain_order:
        print(f"  [WARN]  No RNA residues recognised in {fp.name}; "
              "RNA-distance and terminal filtering will be skipped.")
        return empty, empty.copy(), empty.copy()

    all_atoms = []
    terminal_atoms = []
    nonterminal_atoms = []

    for chain_key in chain_order:
        chain_residues = chains[chain_key]
        for residue in chain_residues:
            all_atoms.extend(residue["atoms"])

        chain_label = chain_residues[0]["chain_label"]
        n_residues = len(chain_residues)
        if n_residues < MIN_RNA_CHAIN_RESIDUES_FOR_TERMINAL_FILTER:
            # With fewer than three residues there is no defensible interior
            # reference.  Retain these atoms for RNA trimming/support, but do
            # not let the short chain cause terminal-pocket rejection.
            for residue in chain_residues:
                nonterminal_atoms.extend(residue["atoms"])
            print(f"  [RNA]   Chain {chain_label}: {n_residues} RNA residue(s); "
                  "too short for terminal-artifact filtering")
            continue

        first_residue = chain_residues[0]
        last_residue = chain_residues[-1]
        terminal_atoms.extend(first_residue["atoms"])
        terminal_atoms.extend(last_residue["atoms"])
        for residue in chain_residues[1:-1]:
            nonterminal_atoms.extend(residue["atoms"])
        print(f"  [RNA]   Chain {chain_label}: {n_residues} RNA residues; "
              f"terminals {_residue_label(first_residue)}, "
              f"{_residue_label(last_residue)}")

    def xyz_array(values):
        return (np.asarray(values, dtype=float).reshape((-1, 3))
                if values else empty.copy())

    all_xyz = xyz_array(all_atoms)
    terminal_xyz = xyz_array(terminal_atoms)
    nonterminal_xyz = xyz_array(nonterminal_atoms)
    print(f"  [RNA]   Loaded {len(all_xyz)} RNA heavy atoms from {fp.name}; "
          f"terminal={len(terminal_xyz)}, non-terminal={len(nonterminal_xyz)}")
    return all_xyz, terminal_xyz, nonterminal_xyz


# -------------------------------------------------
# TERMINAL POCKET FILTERING
# -------------------------------------------------
def filter_terminal_pockets(unique_pockets,
                             terminal_atom_xyz,
                             nonterminal_atom_xyz,
                             cutoff_a=TERMINAL_ZONE_CUTOFF_A,
                             voxel_fraction=TERMINAL_VOXEL_FRACTION,
                             centre_cutoff_a=TERMINAL_CENTRE_CUTOFF_A):
    """
    Discard pockets whose voxels are predominantly terminal-associated.

    A voxel is terminal-associated only when (1) it is within *cutoff_a* Å
    of a terminal atom and (2) its nearest terminal atom is closer than its
    nearest non-terminal RNA atom.  The second requirement protects genuine
    interior and multi-chain-interface pockets that merely happen to have a
    chain end nearby.  A terminal-dominant pocket core provides a second,
    complementary rejection condition for expanded pocket volumes.

    Parameters
    ----------
    unique_pockets      : list of pocket dicts (output of build_unique_pockets)
    terminal_atom_xyz   : np.ndarray (N, 3) — terminal residue heavy-atom coords
    nonterminal_atom_xyz: np.ndarray (M, 3) — all other RNA heavy-atom coords
    cutoff_a            : distance threshold in Å (default TERMINAL_ZONE_CUTOFF_A)
    voxel_fraction      : fraction threshold (default TERMINAL_VOXEL_FRACTION)
    centre_cutoff_a     : representative-core-to-terminal cutoff in Å

    Returns
    -------
    accepted : list of pocket dicts that passed the filter
    rejected : list of pocket dicts that were discarded as terminal artefacts
    """
    if terminal_atom_xyz is None or len(terminal_atom_xyz) == 0:
        print("  [Term]  No terminal-atom coordinates available — "
              "terminal pocket filtering skipped.")
        return unique_pockets, []

    term_tree = cKDTree(terminal_atom_xyz)
    internal_tree = (cKDTree(nonterminal_atom_xyz)
                     if nonterminal_atom_xyz is not None and
                     len(nonterminal_atom_xyz) > 0 else None)
    if internal_tree is None:
        print("  [WARN]  No non-terminal RNA atoms are available — "
              "terminal filtering skipped to avoid rejecting an entire "
              "short RNA as a terminal artefact.")
        return unique_pockets, []

    accepted = []
    rejected = []

    for pocket in unique_pockets:
        mask   = pocket["merged_mask"]
        voxel  = pocket["voxel"]
        origin = pocket["origin"]

        if mask is None or voxel is None or not np.any(mask):
            # Empty pocket — keep it (will be naturally skipped downstream)
            accepted.append(pocket)
            continue

        n_voxels = int(np.count_nonzero(mask))
        coords   = np.column_stack(np.where(mask))
        world    = coords * voxel[None, :] + origin[None, :]

        # A terminal residue must be both nearby and the nearest RNA region.
        terminal_dists, _ = term_tree.query(world, k=1, workers=-1)
        internal_dists, _ = internal_tree.query(world, k=1, workers=-1)
        terminal_associated = ((terminal_dists <= cutoff_a) &
                               (terminal_dists < internal_dists))
        n_terminal_voxels = int(np.count_nonzero(terminal_associated))
        frac_terminal     = n_terminal_voxels / n_voxels if n_voxels > 0 else 0.0

        geometric_centre = np.mean(world, axis=0)
        # The arithmetic centre of a crescent-shaped mask can lie outside the
        # pocket.  Use the occupied voxel nearest that centre as an in-pocket
        # representative core for terminal classification.
        core_index = int(np.argmin(np.sum(
            (world - geometric_centre[None, :]) ** 2, axis=1)))
        pocket_core = world[core_index]
        centre_terminal_dist = float(term_tree.query(pocket_core, k=1)[0])
        centre_internal_dist = float(internal_tree.query(pocket_core, k=1)[0])
        centre_is_terminal = (
            centre_terminal_dist <= centre_cutoff_a and
            centre_terminal_dist < centre_internal_dist
        )

        label = pocket["contributing_fields"]
        fraction_is_terminal = frac_terminal > voxel_fraction
        if fraction_is_terminal or centre_is_terminal:
            reasons = []
            if fraction_is_terminal:
                reasons.append("terminal-dominant voxel fraction")
            if centre_is_terminal:
                reasons.append("terminal-dominant pocket core")
            print(f"  [Term]  DISCARD pocket ({label}): "
                  f"{n_terminal_voxels}/{n_voxels} voxels "
                  f"({100.0 * frac_terminal:.1f}%) terminal-associated "
                  f"within {cutoff_a} Å; core distances "
                  f"terminal/internal={centre_terminal_dist:.2f}/"
                  f"{centre_internal_dist:.2f} Å; "
                  f"reason={'+'.join(reasons)}")
            rejected.append(pocket)
        else:
            print(f"  [Term]  KEEP    pocket ({label}): "
                  f"{n_terminal_voxels}/{n_voxels} voxels "
                  f"({100.0 * frac_terminal:.1f}%) terminal-associated; "
                  f"core distances terminal/internal="
                  f"{centre_terminal_dist:.2f}/{centre_internal_dist:.2f} Å")
            accepted.append(pocket)

    return accepted, rejected


# -------------------------------------------------
# SCORING (UNCHANGED)
# -------------------------------------------------
def rank_pockets_by_score(unique_pockets, pocket_scores, pocket_field_integrals):
    combined = sorted(
        zip(pocket_scores, pocket_field_integrals, unique_pockets),
        key=lambda t: t[0],
        reverse=True,
    )
    sorted_scores = [t[0] for t in combined]
    sorted_integrals = [t[1] for t in combined]
    sorted_pockets = [t[2] for t in combined]

    for rank, pocket in enumerate(sorted_pockets, start=1):
        pocket["name"] = f"Pocket{rank}"
        pocket["filename"] = f"Pocket{rank}_Volume.mrc"

    return sorted_pockets, sorted_scores, sorted_integrals


def enforce_disjoint_masks_by_score(
        ranked_pockets,
        min_exclusive_fraction=SIGNIFICANT_OVERLAP_FRACTION):
    """
    Give every residual shared voxel to the higher preliminary-score pocket.

    Significant overlaps have already been consolidated, so this step affects
    only sub-threshold boundary intersections.  Pockets must be supplied in
    descending preliminary-score order.  If less than
    *min_exclusive_fraction* of a lower-ranked mask remains, that residual is
    not considered an independent pocket and is dropped.  The caller
    recomputes all integrals, normalised scores and final ranks afterwards.
    """
    if not ranked_pockets:
        return [], False

    reference_shape = ranked_pockets[0]["merged_mask"].shape
    occupied = np.zeros(reference_shape, dtype=bool)
    kept = []
    changed = False

    print("  Final residual-overlap ownership (higher preliminary score wins):")
    for pocket in ranked_pockets:
        mask = pocket["merged_mask"].astype(bool)
        n_original = int(np.count_nonzero(mask))
        shared = mask & occupied
        n_shared = int(np.count_nonzero(shared))
        if n_shared > 0:
            changed = True
            mask = mask & ~occupied
            print(f"    {pocket['name']} ({pocket['contributing_fields']}): "
                  f"removed {n_shared} shared voxel(s)")

        n_remaining = int(np.count_nonzero(mask))
        if n_remaining == 0:
            changed = True
            print(f"    DROP {pocket['name']} ({pocket['contributing_fields']}): "
                  "no exclusive voxels remain")
            continue

        exclusive_fraction = n_remaining / n_original if n_original else 0.0
        if n_shared > 0 and exclusive_fraction < min_exclusive_fraction:
            changed = True
            print(f"    DROP {pocket['name']} ({pocket['contributing_fields']}): "
                  f"only {100.0 * exclusive_fraction:.1f}% exclusive voxels "
                  f"remain [minimum={100.0 * min_exclusive_fraction:.0f}%]")
            continue

        pocket["merged_mask"] = mask
        pocket["n_voxels"] = n_remaining
        occupied |= mask
        kept.append(pocket)

    if not changed:
        print("    No residual shared voxels detected.")
    return kept, changed


def validate_final_masks_are_disjoint(pockets):
    """Fail loudly if any final ranked masks still share a voxel."""
    for i, j in combinations(range(len(pockets)), 2):
        overlap = int(np.count_nonzero(
            pockets[i]["merged_mask"] & pockets[j]["merged_mask"]))
        if overlap:
            raise RuntimeError(
                f"Final pockets {i + 1} and {j + 1} still share "
                f"{overlap} voxel(s)"
            )
    print("  [Check] Final pocket masks are voxel-disjoint.")


def find_field_file(fields_dir, pdb, primary_suffix, fallback_suffix):
    f = fields_dir / f"{pdb}{primary_suffix}"
    if f.exists():
        return f, "trimmed"
    if fallback_suffix is not None:
        f = fields_dir / f"{pdb}{fallback_suffix}"
        if f.exists():
            return f, "raw_fallback"
    return None, None


def load_raw_field_data(fields_dir, pdb):
    fields = {}
    for primary_suffix, fallback_suffix, fname in RAW_FIELD_SPECS:
        fp, variant = find_field_file(fields_dir, pdb, primary_suffix, fallback_suffix)
        if fp is None:
            print(f"  [WARN] Field '{fname}': no file found "
                  f"(tried {primary_suffix}"
                  + (f", {fallback_suffix}" if fallback_suffix else "") + ")")
            continue
        try:
            data, voxel, origin = _load_mrc_data(fp)
        except Exception as e:
            print(f"  [WARN] Cannot load field file {fp.name}: {e}")
            continue
        abs_data = np.where(np.isfinite(data), np.abs(data), 0.0)
        fields[fname] = (abs_data, voxel, origin)
        tag = "(trimmed)" if variant == "trimmed" else "(raw fallback)"
        print(f"  [Field] {fname:14s} ← {fp.name}  {tag}")
    return fields


def total_grid_integral(raw_field_data):
    return {
        fname: float(np.sum(data))
        for fname, (data, _, _) in raw_field_data.items()
    }


def build_normalization_constants(grid_totals):
    norm = {}
    stk_total = grid_totals.get("stacking", 0.0)
    hyd_total = grid_totals.get("hydrophobic", 0.0)
    stk_hyd_combined = stk_total + hyd_total

    for fname, total in grid_totals.items():
        if fname in ("stacking", "hydrophobic"):
            norm[fname] = stk_hyd_combined
        else:
            norm[fname] = total
    return norm


def field_integral_in_mask(raw_field_data, pocket_mask, pocket_voxel, pocket_origin):
    pocket_coords = np.column_stack(np.where(pocket_mask))
    if len(pocket_coords) == 0:
        return {fname: 0.0 for fname in raw_field_data}

    world = pocket_coords * pocket_voxel[None, :] + pocket_origin[None, :]

    integrals = {}
    for fname, (data, fvoxel, forigin) in raw_field_data.items():
        frac = (world - forigin[None, :]) / fvoxel[None, :]
        ix = np.rint(frac[:, 0]).astype(int)
        iy = np.rint(frac[:, 1]).astype(int)
        iz = np.rint(frac[:, 2]).astype(int)
        in_b = (
            (ix >= 0) & (ix < data.shape[0]) &
            (iy >= 0) & (iy < data.shape[1]) &
            (iz >= 0) & (iz < data.shape[2])
        )
        val = float(np.sum(data[ix[in_b], iy[in_b], iz[in_b]])) if np.any(in_b) else 0.0
        integrals[fname] = val
    return integrals


def compute_pocket_scores(unique_pockets, raw_field_data, grid_totals):
    norm_constants = build_normalization_constants(grid_totals)

    pocket_field_integrals = []
    for pocket in unique_pockets:
        mask = pocket["merged_mask"]
        voxel = pocket["voxel"]
        origin = pocket["origin"]
        if mask is None or voxel is None or not raw_field_data:
            pocket_field_integrals.append({})
        else:
            pocket_field_integrals.append(
                field_integral_in_mask(raw_field_data, mask, voxel, origin))

    raw_scores = []
    for pfi in pocket_field_integrals:
        s = sum(
            pfi[fname] / norm_constants[fname]
            for fname in pfi
            if norm_constants.get(fname, 0.0) > 0.0
        )
        raw_scores.append(s)

    total = sum(raw_scores)
    norm_scores = [s / total for s in raw_scores] if total > 0.0 else [0.0] * len(raw_scores)
    return norm_scores, pocket_field_integrals


# -------------------------------------------------
# OUTPUT: PLOT & CSV (UNCHANGED)
# -------------------------------------------------
def generate_field_contribution_plot(unique_pockets, pocket_field_integrals,
                                     grid_totals, out_path):
    if not unique_pockets:
        return

    norm_constants = build_normalization_constants(grid_totals)

    n = len(unique_pockets)
    pocket_names = [p["name"] for p in unique_pockets]
    x = np.arange(n)
    width = 0.60

    data_matrix = []
    for pfi in pocket_field_integrals:
        row = []
        for fname in FIELD_NAMES_CSV:
            integral = pfi.get(fname, 0.0)
            norm_c = norm_constants.get(fname, 0.0)
            row.append(integral / norm_c if norm_c > 0.0 else 0.0)
        data_matrix.append(row)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.6 + 2), 6), dpi=300)
    bottoms = np.zeros(n)
    patches = []
    for col_idx, fname in enumerate(FIELD_NAMES_CSV):
        heights = np.array([data_matrix[pi][col_idx] for pi in range(n)])
        color = FIELD_COLORS.get(fname, "#888888")
        disp = FIELD_LABELS_DISPLAY.get(fname, fname)
        ax.bar(x, heights, width, bottom=bottoms, color=color,
               edgecolor="white", linewidth=0.4)
        patches.append(mpatches.Patch(color=color, label=disp))
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(pocket_names, fontsize=11, rotation=0, ha="center",
                       fontfamily="Times New Roman")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlabel("Unique Pockets", fontsize=13, labelpad=8,
                  fontfamily="Times New Roman")
    ax.set_ylabel("Weightage of Each Field", fontsize=13, labelpad=8,
                  fontfamily="Times New Roman")
    ax.yaxis.get_major_formatter().set_powerlimits((-3, 3))
    ax.legend(handles=patches, loc="upper right", framealpha=0.85,
              fontsize=10, ncol=3, bbox_to_anchor=(1.0, 1.15),
              prop={"family": "Times New Roman"})
    ax.set_xlim(-0.5, n - 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot]  Saved: {out_path}")


def save_field_contributions_csv(pdb, unique_pockets, pocket_scores,
                                 pocket_field_integrals, grid_totals, out_path):
    norm_constants = build_normalization_constants(grid_totals)

    with open(out_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        header = (
            ["pocket_index", "pocket_name", "pocket_score", "pocket_volume_A3",
             "contributing_fields"] +
            [f"{fn}_integral_in_pocket" for fn in FIELD_NAMES_CSV] +
            [f"{fn}_normalization_constant" for fn in FIELD_NAMES_CSV] +
            [f"{fn}_ratio" for fn in FIELD_NAMES_CSV]
        )
        writer.writerow(header)
        for pi, (pocket, score, pfi) in enumerate(
                zip(unique_pockets, pocket_scores, pocket_field_integrals), 1):

            vol_A3 = (compute_pocket_volume_A3(pocket["merged_mask"], pocket["voxel"])
                      if pocket["voxel"] is not None else 0.0)

            in_vals = [pfi.get(fn, 0.0) for fn in FIELD_NAMES_CSV]
            norm_vals = [norm_constants.get(fn, 0.0) for fn in FIELD_NAMES_CSV]
            ratios = [iv / nv if nv > 0.0 else 0.0 for iv, nv in zip(in_vals, norm_vals)]
            writer.writerow(
                [pi, pocket["name"], f"{score:.6f}", f"{vol_A3:.3f}",
                 pocket["contributing_fields"]] +
                [f"{v:.6e}" for v in in_vals] +
                [f"{v:.6e}" for v in norm_vals] +
                [f"{r:.6e}" for r in ratios]
            )
    print(f"  [CSV]   Saved: {out_path}")


def clear_previous_outputs(output_dir, pdb):
    """
    Remove only this PDB's outputs from an earlier Script8 run.

    Cleanup is performed after input validation but before analysis begins, so
    an early return (for example, all candidates being rejected as terminal
    artefacts) cannot leave stale MRC/CSV/PNG files that look like fresh output.
    """
    patterns = [
        f"{pdb}.Pocket*_Volume.mrc",
        f"{pdb}_field_contributions.csv",
        f"{pdb}_field_contributions.png",
    ]

    removed = []
    for pattern in patterns:
        for path in sorted(output_dir.glob(pattern)):
            try:
                path.unlink()
                removed.append(path.name)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not remove previous Script8 output '{path}': {exc}"
                ) from exc

    if removed:
        print("  [Clean] Removed previous Script8 output(s): "
              + ", ".join(removed))
    else:
        print("  [Clean] No previous Script8 outputs found for this PDB.")


# =====================================================================
# SINGLE-PDB DRIVER (replaces old multi-PDB ANALYSIS_BASE discovery loop)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Script8: Make unique ligand-binding pockets by merging "
                    "pipeline1/pipeline2 pocket masks, trim by RNA distance "
                    "and terminal-residue proximity, then score and rank "
                    "using field contributions — for a single PDB."
    )
    parser.add_argument(
        "--analysis1_dir", required=True,
        help="Path to Analysis_Pipeline1_<PDB_ID> folder (output of Script1-3). "
             "Used to locate the STK-HPb / STK-ELE / Mixed-field pocket MRCs."
    )
    parser.add_argument(
        "--analysis2_dir", required=True,
        help="Path to Analysis_Pipeline2_<PDB_ID> folder (output of Script4-5). "
             "Used to locate the HBond_Site_N pocket MRCs."
    )
    parser.add_argument(
        "--fields_dir", required=True,
        help="Path to Fields_Pipeline1_<PDB_ID> folder. Used to locate the "
             "raw/trimmed field MRCs (apbs, hydrophobic, hbacceptors, "
             "hbdonors, stacking) for scoring."
    )
    parser.add_argument(
        "--pdb_file", required=True,
        help="Path to the fixed PDB/CIF structure file (e.g. 1AJU_fixed.pdb), "
             "used for RNA heavy-atom distance trimming and terminal-residue "
             "detection."
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Path to the output folder for unique pockets "
             "(e.g. Analysis_Unique_Pockets_<PDB_ID>). "
             "Defaults to 'Analysis_Unique_Pockets_<pdb_id>' in the current "
             "working directory if not provided."
    )
    parser.add_argument(
        "--pdb_id", default=None,
        help="Optional override for the PDB identifier used in filenames. "
             "If not provided, it is automatically derived from --pdb_file "
             "(filename without extension)."
    )
    args = parser.parse_args()

    pdb_file = Path(args.pdb_file)
    pdb = args.pdb_id if args.pdb_id else pdb_file.stem

    analysis1_dir = Path(args.analysis1_dir)
    analysis2_dir = Path(args.analysis2_dir)
    fields_dir    = Path(args.fields_dir)

    output_dir = (Path(args.output_dir) if args.output_dir
                 else Path(f"Analysis_Unique_Pockets_{pdb}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {pdb} ===")
    print(f"  Analysis1 dir (Pipeline1 pockets) : {analysis1_dir}")
    print(f"  Analysis2 dir (Pipeline2 HBond)   : {analysis2_dir}")
    print(f"  Fields dir    (scoring fields)    : {fields_dir}")
    print(f"  PDB file                          : {pdb_file}")
    print(f"  Output dir    (unique pockets)    : {output_dir}\n")

    if not analysis1_dir.is_dir():
        print(f"  [ERROR] Analysis1 dir not found: {analysis1_dir}")
        return
    if not analysis2_dir.is_dir():
        print(f"  [WARN]  Analysis2 dir not found: {analysis2_dir} "
              "— HBond sites will be skipped")
        analysis2_dir = None
    if not fields_dir.is_dir():
        print(f"  [ERROR] Fields dir not found: {fields_dir}")
        return
    if not pdb_file.exists():
        print(f"  [ERROR] PDB file not found: {pdb_file}")
        return

    # Clear prior results only after every required input path has passed
    # validation.  This prevents stale outputs from surviving any later early
    # return and being mistaken for results from the current run.
    clear_previous_outputs(output_dir, pdb)

    atom_xyz, terminal_atom_xyz, nonterminal_atom_xyz = \
        load_rna_structure(pdb_file)

    all_found_labels, label_masks, label_voxels, label_origins = \
        load_all_label_masks(pdb, analysis1_dir, analysis2_dir)

    if not all_found_labels:
        print("  [SKIP]  No pocket MRC files found\n")
        return
    print(f"  Found labels: {all_found_labels}")

    label_masks, label_voxels, label_origins = align_masks_to_common_grid(
        all_found_labels, label_masks, label_voxels, label_origins)

    # ── Step 1: Build unique pockets (overlap-cluster + RNA-distance trim) ──
    unique_pockets = build_unique_pockets(
        all_found_labels, label_masks, label_voxels, label_origins, atom_xyz)

    # Re-evaluate final group unions after RNA trimming.  This is essential
    # because complete-linkage source grouping can leave two significantly
    # overlapping union masks in separate cliques.
    unique_pockets = consolidate_trimmed_pocket_unions(unique_pockets)

    # ── Step 2: Filter out terminal-region pockets ──────────────────────
    print(f"  Terminal-pocket filtering "
          f"(cutoff={TERMINAL_ZONE_CUTOFF_A} Å, "
          f"discard_if_terminal_dominant_fraction>"
          f"{TERMINAL_VOXEL_FRACTION:.0%}):")
    unique_pockets, discarded_pockets = filter_terminal_pockets(
        unique_pockets,
        terminal_atom_xyz,
        nonterminal_atom_xyz,
        cutoff_a=TERMINAL_ZONE_CUTOFF_A,
        voxel_fraction=TERMINAL_VOXEL_FRACTION,
        centre_cutoff_a=TERMINAL_CENTRE_CUTOFF_A,
    )
    if discarded_pockets:
        print(f"  [Term]  {len(discarded_pockets)} terminal pocket(s) discarded: "
              + "; ".join(p["contributing_fields"] for p in discarded_pockets))
    else:
        print("  [Term]  No terminal pockets detected — all pockets retained.")

    if not unique_pockets:
        print("  [SKIP]  All pockets were discarded as terminal artefacts.\n")
        return

    # ── Step 3: Preliminary score/rank for residual-voxel ownership ─────
    raw_field_data = load_raw_field_data(fields_dir, pdb)
    grid_totals = total_grid_integral(raw_field_data)
    if not raw_field_data:
        print("  [WARN]  No field MRC files found; scores will be 0")
    else:
        print(f"  Fields loaded for scoring: {list(raw_field_data.keys())}")
        for fn, tot in grid_totals.items():
            print(f"    {fn}: total grid integral = {tot:.4e}")
        stk_hyd_combined = (grid_totals.get("stacking", 0.0) +
                            grid_totals.get("hydrophobic", 0.0))
        print(f"    stacking+hydrophobic combined normalization = {stk_hyd_combined:.4e}"
              f"  (denominator for both stacking and hydrophobic weights)")

    preliminary_scores, preliminary_integrals = compute_pocket_scores(
        unique_pockets, raw_field_data, grid_totals)

    unique_pockets, preliminary_scores, preliminary_integrals = \
        rank_pockets_by_score(
            unique_pockets, preliminary_scores, preliminary_integrals)

    # Significant group-union overlaps were merged above.  Resolve any minor
    # sub-threshold intersections deterministically so saved "unique" pocket
    # masks never share voxels.  Higher preliminary score has priority.
    unique_pockets, masks_changed = enforce_disjoint_masks_by_score(
        unique_pockets)

    if not unique_pockets:
        print("  [SKIP]  No non-overlapping pocket voxels remain.\n")
        return

    # Removing shared boundary voxels can increase terminal dominance of a
    # residual pocket, so apply the same terminal test once more.
    if masks_changed:
        print("  Rechecking terminal status after residual-overlap removal:")
        unique_pockets, newly_discarded = filter_terminal_pockets(
            unique_pockets,
            terminal_atom_xyz,
            nonterminal_atom_xyz,
            cutoff_a=TERMINAL_ZONE_CUTOFF_A,
            voxel_fraction=TERMINAL_VOXEL_FRACTION,
            centre_cutoff_a=TERMINAL_CENTRE_CUTOFF_A,
        )
        if newly_discarded:
            print(f"  [Term]  {len(newly_discarded)} additional pocket(s) "
                  "discarded after de-overlap: "
                  + "; ".join(p["contributing_fields"]
                              for p in newly_discarded))

    if not unique_pockets:
        print("  [SKIP]  All remaining pockets were terminal artefacts.\n")
        return

    # ── Step 4: Recompute final scores/ranks from final disjoint masks ───
    pocket_scores, pocket_field_integrals = compute_pocket_scores(
        unique_pockets, raw_field_data, grid_totals)
    unique_pockets, pocket_scores, pocket_field_integrals = \
        rank_pockets_by_score(
            unique_pockets, pocket_scores, pocket_field_integrals)

    validate_final_masks_are_disjoint(unique_pockets)

    print(f"  Unique pockets ({len(unique_pockets)}) ordered by score (highest first):")
    for p, score in zip(unique_pockets, pocket_scores):
        vol = (compute_pocket_volume_A3(p["merged_mask"], p["voxel"])
               if p["voxel"] is not None else 0.0)
        print(f"    {p['name']}  score={score:.6f}  "
              f"({p['n_voxels']} voxels, {vol:.1f} Å³)  "
              f"← {p['contributing_fields']}")

    for pocket in unique_pockets:
        if pocket["merged_mask"] is None or pocket["voxel"] is None:
            print(f"  [WARN]  '{pocket['name']}' has no mask — skipping MRC")
            continue
        out_path = output_dir / f"{pdb}.{pocket['filename']}"
        try:
            write_mrc(out_path,
                      pocket["merged_mask"].astype(np.float32),
                      pocket["voxel"],
                      pocket["origin"])
            print(f"  [MRC]   Saved: {out_path.name}")
        except Exception as e:
            print(f"  [ERROR] Writing {out_path.name}: {e}")

    contrib_csv = output_dir / f"{pdb}_field_contributions.csv"
    try:
        save_field_contributions_csv(
            pdb, unique_pockets, pocket_scores,
            pocket_field_integrals, grid_totals, contrib_csv)
    except Exception as e:
        print(f"  [ERROR] Writing CSV: {e}")

    if unique_pockets and any(pfi for pfi in pocket_field_integrals):
        plot_path = output_dir / f"{pdb}_field_contributions.png"
        try:
            generate_field_contribution_plot(
                unique_pockets, pocket_field_integrals, grid_totals, plot_path)
        except Exception as e:
            print(f"  [ERROR] Generating chart for {pdb}: {e}")

    print("\nScript8 complete.")


if __name__ == "__main__":
    main()
