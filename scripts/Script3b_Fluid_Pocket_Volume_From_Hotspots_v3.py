#!/usr/bin/env python3
"""
Fluid pocket volumes from SPOCKER hotspots  ("fluid_single_T40") -- v3
=====================================================================

Runs in the volgrids conda env (needs scikit-image installed there):
    conda run -n volgrids python3 Script3b_Fluid_Pocket_Volume_From_Hotspots_v3.py \
        --pdb_file X.pdb --fields_dir Fields_Pipeline1_X \
        --analysis1_dir Analysis_Pipeline1_X --analysis2_dir Analysis_Pipeline2_X \
        --output_dir Fluid_Pockets_X [--travel_time 40] [--stride 2] [--pipeline_names]
    python3 Script3b_Fluid_Pocket_Volume_From_Hotspots_v3.py --selftest
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import volgrids as vg
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from Script8_Making_Unique_Pockets_Using_All_Previous_Pockets import (
    load_rna_structure, find_field_file, RAW_FIELD_SPECS)

vg.CFG.out_overwrite_ok = True   # never prompt on overwrite

# Parameters (values selected in pocket_shape_study/), identical to v1 / v2
TRAVEL_TIME     = 40.0   # fluid travel-time limit (A / speed units)
STRIDE          = 2      # work grid = field grid subsampled by this factor
EXCL_A          = 3.0    # no pocket voxel closer than this to an RNA heavy atom
SHELL_A         = 14.0   # ... nor farther than this, not sure if it's going to be crossed at some point
VDW_A           = 1.8    # atom radius used for ray hits
RAY_LEN_A       = 12.0
N_RAYS          = 30
MIN_BURIEDNESS  = 0.4    # domain gate
SPEED_FLOOR     = 0.1    # speed = bur^2 * (SPEED_FLOOR + P)
SEED_SNAP_A     = 4.0    # if the centre is not in the domain, start from the nearest domain voxel within this, not very important.

P1_TAGS = ["stk_hpb_first", "stk_hpb_second", "stk_hpb_third", "stk_hpb_fourth",
           "stk_ele", "mixed_fields"]


def fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def buriedness(occ, vox):
    """Fraction of N_RAYS directions in which a ray from the voxel hits `occ` within RAY_LEN_A (LIGSITE-like)."""
    # ndimage.binary_dilation with a ray kernel is the obvious library call but ~1000x slower
    s = int(RAY_LEN_A / vox[0])
    pad = np.pad(occ, s)
    nx, ny, nz = occ.shape
    count = np.zeros(occ.shape, np.uint8)
    for u in fib_dirs(N_RAYS):
        hit = np.zeros_like(occ)
        for i, j, k in {tuple(o) for o in np.rint(np.outer(np.arange(1, s + 1), u)).astype(int)}:
            hit |= pad[s + i:s + i + nx, s + j:s + j + ny, s + k:s + k + nz]   # occ[v + offset]
        count += hit
    return count.astype(np.float32) / N_RAYS


def fluid_pockets(atom_xyz, fields, centers_xyz, travel_time=TRAVEL_TIME):
    """fields: {name: vg.Grid of |field|}, all on the same box.
    Returns (one bool vg.Grid mask per centre, buriedness vg.Grid)."""
    box = next(iter(fields.values())).box
    shape, vox, org = tuple(box.resolution), box.deltas, box.min_coords
    pts = vg.Math.get_coords_array(box.resolution, vox, org).reshape(-1, 3)
    d_atom = np.minimum(cKDTree(atom_xyz).query(pts, distance_upper_bound=20, workers=-1)[0], 20)
    d_atom = d_atom.reshape(shape).astype(np.float32)
    del pts

    bur = buriedness(d_atom <= VDW_A, vox)
    near = (d_atom > EXCL_A) & (d_atom <= SHELL_A)
    P = np.mean([np.clip(f.arr / max(np.percentile(f.arr[near], 99), 1e-9), 0, 1) for f in fields.values()], axis=0)
    domain = near & (bur >= MIN_BURIEDNESS)
    cost = np.where(domain, 1.0 / np.maximum(bur ** 2 * (SPEED_FLOOR + P), 1e-3), np.inf)

    masks = [vg.Grid(box, dtype=bool) for _ in centers_xyz]
    if domain.any():
        coords = np.argwhere(domain)
        tree = cKDTree(coords)
        for m, c in zip(masks, centers_xyz):
            ijk = np.clip(np.rint((np.asarray(c) - org) / vox).astype(int), 0, np.array(shape) - 1)
            if not domain[tuple(ijk)]:
                d, k = tree.query(ijk, distance_upper_bound=SEED_SNAP_A / vox[0])
                if not np.isfinite(d):
                    continue
                ijk = coords[k]
            # ponytail: floods the whole domain (max_cumulative_cost early stop is deprecated in skimage 0.26)
            t, _ = MCP_Geometric(cost, sampling=tuple(vox)).find_costs([tuple(ijk)])
            m.arr = t <= travel_time
    g_bur = vg.Grid(box, init_grid=False)
    g_bur.arr = bur
    return masks, g_bur


def find_hotspots(pdb_id, a1, a2):
    """[(tag, centre_xyz, pipeline_output_path)] from the Script3 / Script4 centre markers."""
    cand = [(t, a1 / f"{pdb_id}.{t}.chosen_patch_center_marker.mrc",
             a1 / f"{pdb_id}.{t}.final_trimmed_smif_identical_rna_only_pocket.mrc") for t in P1_TAGS]
    if a2 is not None:
        cand += [(f"HBond_Site_{n}", a2 / f"{pdb_id}.HBond_Site_{n}_marker.mrc",
                  a2 / f"{pdb_id}.HBond_Site_{n}_pocket_volume.mrc") for n in (1, 2)]
    res = []
    for tag, mk, target in cand:
        if mk.exists():
            g = vg.Grid.load(mk)
            res.append((tag, np.array(ndimage.center_of_mass(g.arr > 0)) * g.box.deltas + g.box.min_coords, target))
    return res


def coarsen(g, s):
    """|g| subsampled every s voxels, on a box with s-times larger deltas."""
    arr = np.nan_to_num(abs(g).arr, nan=0, posinf=0, neginf=0)[::s, ::s, ::s]
    out = vg.Grid(vg.Box(g.box.min_coords, arr.shape, g.box.deltas * s), init_grid=False)
    out.arr = arr.astype(np.float32)
    return out


def refine(mask, s, full_box):
    """Coarse bool mask -> float mask on the full field box."""
    out = vg.Grid(full_box, init_grid=False)
    out.arr = mask.arr.repeat(s, 0).repeat(s, 1).repeat(s, 2)[tuple(slice(n) for n in full_box.resolution)].astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description="Fluid pocket volumes from SPOCKER hotspots (replaces the 8 A seed sphere).")
    ap.add_argument("--pdb_file")
    ap.add_argument("--fields_dir")
    ap.add_argument("--analysis1_dir")
    ap.add_argument("--analysis2_dir", default=None)
    ap.add_argument("--output_dir")
    ap.add_argument("--pdb_id", default=None, help="default: --pdb_file stem")
    ap.add_argument("--travel_time", type=float, default=TRAVEL_TIME)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--pipeline_names", action="store_true",
                    help="also overwrite the Script3/Script5 pocket files so Script8 uses the fluid pockets")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    for k in ("pdb_file", "fields_dir", "analysis1_dir", "output_dir"):
        if getattr(a, k) is None:
            ap.error(f"--{k} is required")

    pdb_id = a.pdb_id or Path(a.pdb_file).stem
    fdir, a1 = Path(a.fields_dir), Path(a.analysis1_dir)
    a2 = Path(a.analysis2_dir) if a.analysis2_dir and Path(a.analysis2_dir).exists() else None
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields, s = {}, a.stride
    for prim, fb, name in RAW_FIELD_SPECS:
        fp, _ = find_field_file(fdir, pdb_id, prim, fb)
        if fp is None:
            print(f"  [WARN] field '{name}' not found, left out of P")
            continue
        g = vg.Grid.load(fp)
        full_box = g.box
        fields[name] = coarsen(g, s)
        print(f"  [Field] {name:12s} <- {fp.name}")
    if not fields:
        raise SystemExit(f"No field files for {pdb_id} in {fdir}")

    atoms = load_rna_structure(a.pdb_file)[0]
    if len(atoms) == 0:
        raise SystemExit(f"No heavy atoms read from {a.pdb_file}")
    hot = find_hotspots(pdb_id, a1, a2)
    if not hot:
        raise SystemExit(f"No hotspot centre markers in {a1} / {a2}")

    masks, bur = fluid_pockets(atoms, fields, [c for _, c, _ in hot], a.travel_time)

    rows = []
    for (tag, c, target), m in zip(hot, masks):
        full = refine(m, s, full_box)
        vol = float(m.arr.sum() * np.prod(m.box.deltas))
        rows.append(dict(hotspot=tag, cx=round(c[0], 3), cy=round(c[1], 3), cz=round(c[2], 3),
                         volume_A3=round(vol, 1),
                         mean_buriedness=round(float(bur.arr[m.arr].mean()), 3) if m.arr.any() else ""))
        if m.arr.any():
            full.save(out / f"{pdb_id}.{tag}.fluid_pocket.mrc")
            print(f"  [Pocket] {tag:15s} {vol:7.1f} A^3")
        else:
            print(f"  [WARN] {tag}: no buried solvent within {SEED_SNAP_A} A of the centre, no pocket")
        if a.pipeline_names:
            full.save(target)   # empty mask -> Script8 skips the old sphere pocket

    pd.DataFrame(rows).to_csv(out / f"{pdb_id}.fluid_pockets_summary.csv", index=False)
    print(f"\nFluid pockets for {pdb_id} saved in {out}")


def selftest():
    """RNA "cup" open towards +z: the fluid must fill it and not leak out."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(6000, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    shell = np.concatenate([u * r for r in (8.0, 9.0, 10.0)])
    atoms = shell[shell[:, 2] < 6.0]
    vox, org, n = np.full(3, 0.5), np.full(3, -16.0), 64
    g = vg.Grid(vg.Box(org, (n, n, n), vox), init_grid=False)
    g.arr = np.ones((n, n, n), np.float32)
    (m,), _ = fluid_pockets(atoms, {"flat": g}, [np.zeros(3)])
    pts = np.argwhere(m.arr) * vox + org
    r = np.linalg.norm(pts, axis=1)
    inside = (r < 4.5).sum() / (4 / 3 * np.pi * 4.5 ** 3 / 0.125)
    outside = r > 10
    assert inside > 0.95, f"fluid did not fill the cup ({inside:.0%})"
    assert outside.mean() < 0.02, f"{outside.mean():.0%} of the pocket is outside the cup"
    assert not (outside & (pts[:, 2] < 6)).any(), "fluid ran down the outer wall"

    print(f"selftest OK: cavity {inside:.0%} filled, {outside.mean():.1%} of voxels just past the mouth")


if __name__ == "__main__":
    main()
