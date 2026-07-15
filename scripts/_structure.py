"""
Structure parsing shared by every stage that needs RNA atom coordinates:
buriedness estimation, seed-sphere trimming, terminal-residue filtering and
building volgrids' "-r chain.resid" residue selections.

Supports plain-text .pdb and .cif files without depending on MDAnalysis, since
raw RCSB files, pdbfixer-processed files and neutron-diffraction structures
each spell hydrogens differently and this parser has to treat them all alike.

Local copy of demo_spocker/pipeline/structure.py, kept so new_spocker/ has no
runtime dependency outside this folder -- see _new_spocker_prepare_fields.py.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_DNA_RESNAMES = {"DA", "DC", "DG", "DT", "DI", "ADE", "CYT", "GUA", "THY"}

_RNA_RESNAMES = {
    "A", "C", "G", "U", "I",
    "RA", "RC", "RG", "RU", "RI",
    "ADE", "CYT", "GUA", "URA", "URI",
    # common modified nucleotides found in raw RCSB PDBs
    "PSU", "H2U", "5MU", "5MC", "OMC", "OMG",
    "1MA", "2MA", "M2G", "1MG", "7MG",
    "YG", "YYG", "G7M", "A2M",
    "4SU", "OMU", "MIA", "5BU", "2MG",
}


def is_hydrogen_name(atom_name: str) -> bool:
    """True for hydrogen/deuterium atom names, including the digit-first
    convention used by pdbfixer (e.g. "1H5'") and neutron structures ("D2')."""
    name = atom_name.strip().upper().replace("*", "'")
    stripped = name.lstrip("0123456789")
    return bool(stripped) and stripped[0] in ("H", "D")


def is_rna_residue(resname: str, atom_names: list) -> bool:
    res = resname.strip().upper()
    atomset = {a.strip().upper().replace("*", "'") for a in atom_names}
    if "O2'" in atomset:
        return True
    if res in _DNA_RESNAMES:
        return False
    if res in _RNA_RESNAMES:
        return True
    return res.startswith("R") and len(res) <= 4


@dataclass
class Structure:
    heavy_rna_xyz: np.ndarray            # (N, 3) heavy-atom coords, RNA residues only
    terminal_xyz: np.ndarray             # (M, 3) heavy atoms of each chain's first/last residue
    residues: list = field(default_factory=list)  # [(chain, resid)] in file order, RNA only


def _group_pdb_atoms(path: Path):
    grouped = {}
    order = []
    with open(path, "r") as fh:
        for line in fh:
            rec = line[:6].strip().upper()
            if rec not in ("ATOM", "HETATM"):
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            atomname = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            try:
                resseq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            key = (chain, resseq, resname)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append((atomname, x, y, z))
    return grouped, order


def _group_cif_atoms(path: Path):
    headers, rows = [], []
    in_loop = False
    with open(path, "r") as fh:
        for line in fh:
            s = line.strip()
            if s == "loop_":
                in_loop, headers, rows = True, [], []
                continue
            if in_loop and s.startswith("_atom_site."):
                headers.append(s)
                continue
            if in_loop and headers and s and not s.startswith(("_", "#")):
                rows.append(s.split())
                continue
            if in_loop and headers and (s.startswith("#") or
                    (s.startswith("_") and not s.startswith("_atom_site."))):
                break

    grouped, order = {}, []
    if not headers or not rows:
        return grouped, order

    n2i = {h: i for i, h in enumerate(headers)}
    req = ["_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
           "_atom_site.label_atom_id", "_atom_site.label_comp_id"]
    if not all(k in n2i for k in req):
        return grouped, order

    for row in rows:
        try:
            if "_atom_site.group_PDB" in n2i and row[n2i["_atom_site.group_PDB"]].upper() != "ATOM":
                continue
            atomname = row[n2i["_atom_site.label_atom_id"]]
            resname = row[n2i["_atom_site.label_comp_id"]].upper()
            chain = (row[n2i["_atom_site.auth_asym_id"]]
                     if "_atom_site.auth_asym_id" in n2i else "")
            resseq_raw = (row[n2i["_atom_site.auth_seq_id"]]
                          if "_atom_site.auth_seq_id" in n2i else "0")
            resseq = int("".join(c for c in resseq_raw if c.isdigit() or c == "-") or "0")
            x = float(row[n2i["_atom_site.Cartn_x"]])
            y = float(row[n2i["_atom_site.Cartn_y"]])
            z = float(row[n2i["_atom_site.Cartn_z"]])
        except (KeyError, ValueError, IndexError):
            continue
        key = (chain, resseq, resname)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((atomname, x, y, z))
    return grouped, order


def load_structure(path) -> Structure:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".cif":
        grouped, order = _group_cif_atoms(path)
    else:
        grouped, order = _group_pdb_atoms(path)

    heavy_rna = []
    terminal = []
    residues = []

    chain_res_range = {}
    for chain, resseq, resname in order:
        atom_list = grouped[(chain, resseq, resname)]
        if not is_rna_residue(resname, [a[0] for a in atom_list]):
            continue
        lo, hi = chain_res_range.get(chain, (resseq, resseq))
        chain_res_range[chain] = (min(lo, resseq), max(hi, resseq))
        residues.append((chain, resseq))

    terminal_keys = set()
    for chain, (lo, hi) in chain_res_range.items():
        terminal_keys.add((chain, lo))
        terminal_keys.add((chain, hi))

    for chain, resseq, resname in order:
        atom_list = grouped[(chain, resseq, resname)]
        if not is_rna_residue(resname, [a[0] for a in atom_list]):
            continue
        is_terminal = (chain, resseq) in terminal_keys
        for atomname, x, y, z in atom_list:
            if is_hydrogen_name(atomname):
                continue
            heavy_rna.append((x, y, z))
            if is_terminal:
                terminal.append((x, y, z))

    heavy_rna_xyz = np.array(heavy_rna, dtype=float) if heavy_rna else np.zeros((0, 3))
    terminal_xyz = np.array(terminal, dtype=float) if terminal else np.zeros((0, 3))
    return Structure(heavy_rna_xyz=heavy_rna_xyz, terminal_xyz=terminal_xyz, residues=residues)
