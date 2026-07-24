# SPockeR

SPockeR is a SMIFs-based (Statistical Molecular Interaction Fields) pipeline for
detecting, volumetrically defining, and scoring ligand-binding pockets in
RNA 3D structures. It combines electrostatic (APBS), hydrophobic, base
stacking, and hydrogen-bond donor/acceptor fields to identify unique
druggable pockets, ranked by a physics-based composite score.

## Pipeline overview

**Pipeline 1 — Binding-Site Hotspot Detection**
![Pipeline 1](docs/Pipeline1.png)

**Pipeline 2 — Hydrogen-Bond Hotspot Detection**
![Pipeline 2](docs/Pipeline2.jpg)

**Pipeline 3 — Pocket Consolidation & Scoring**
![Pipeline 3](docs/Pipeline3.jpg)

## System requirements
- OS: Linux x86-64 (tested on Ubuntu 22.04)
- Python 3.11+
- No GPU required
- Key pinned dependencies: volgrids==1.0.0, rnapolis==0.11.5 (see `environment.yml`)

## Installation
```bash
git clone https://github.com/rjdebsarkar/SPockeR.git
cd SPockeR
conda env create -f environment.yml
conda activate SPockeR
export PYTHONNOUSERSITE=1
```

> **Note:** `conda activate SPockeR` must match the `name:` field at the top
> of `environment.yml`. If you rename the environment in `environment.yml`,
> update this command accordingly.

## Usage (single PDB, e.g. 1AJU)

Step 1 — clean/fix the raw PDB structure (required, run manually first):

```bash
cd scripts
./0_fix_pdb.sh ../data/example/1AJU.pdb 1AJU_fixed.pdb
```

Step 2 — run the full pipeline on the fixed PDB:

**Without keeping intermediate files** (default — recommended for large batch runs):
```bash
bash run_pipeline_new_spocker.sh 1AJU_fixed.pdb ../Analysis_Unique_Pockets_1AJU_fixed
```

**With intermediate files preserved** (useful for debugging / inspecting individual field/hotspot stages):
```bash
bash run_pipeline_new_spocker.sh 1AJU_fixed.pdb ../Analysis_Unique_Pockets_1AJU_fixed --keep-intermediate
```

With `--keep-intermediate`, the `Fields_Pipeline1_*/`, `Fields_Pipeline2_*/`,
`Analysis_Pipeline1_*/`, and `Analysis_Pipeline2_*/` directories under
`scripts/testdata/new_spocker_work/` are preserved instead of being deleted
automatically at the end of the run. Note that each new run of
`run_pipeline_new_spocker.sh` clears this work directory before starting, so
intermediates from one PDB are overwritten once the next PDB is processed —
copy them elsewhere first if you need to keep results from multiple PDBs.

## Batch usage (multiple PDBs)

To run SPockeR over a directory of PDB files, loop over Steps 1–2 above for
each structure, e.g.:

```bash
cd scripts
for pdb in /path/to/pdb_directory/*.pdb; do
    id=$(basename "$pdb" .pdb)
    ./0_fix_pdb.sh "$pdb" "${id}_fixed.pdb"
    bash run_pipeline_new_spocker.sh "${id}_fixed.pdb" "../Analysis_Unique_Pockets_${id}"
done
```

## Output
Final ranked pockets are saved in `Analysis_Unique_Pockets_<pdb_id>/`:
- `<pdb>.Pocket1_Volume.mrc`, `<pdb>.Pocket2_Volume.mrc`, ... (Pocket1 = highest scoring)
- `<pdb>_field_contributions.csv`
- `<pdb>_field_contributions.png`

## Repository structure
   SPockeR/
├── scripts/
│ ├── 0_fix_pdb.sh
│ ├── run_pipeline_new_spocker.sh
│ ├── _fields.py
│ ├── _residues.py
│ ├── _structure.py
│ ├── _new_spocker_prepare_fields.py
│ ├── Script1_Pipeline1_Slope_Derived_Fixed_Iso_Values_for_Hotspot.py
│ ├── Script2_Pipeline1_Detection_of_Binding_Site_Hotspots.py
│ ├── Script3_Pipeline1_Making_Pocket_Volume_Using_Hotspots.py
│ ├── Script4_Pipeline2_Hydrogen_Bond_Pocket_Hotspots_Using_HBA_HBD_ELE_Fields.py
│ ├── Script5_Pipeline2_Making_Hydrogen_Bond_Pocket_Volume.py
│ ├── Script6_Trimming_APBS_for_Scoring_Unique_Pockets.py
│ ├── Script7_Trimming_Hydrophobic_for_Scoring_Unique_Pockets.py
│ └── Script8_Making_Unique_Pockets_Using_All_Previous_Pockets.py
├── data/example/ # example PDB (1AJU) for quick testing
├── docs/ # pipeline diagrams and example output figures
├── environment.yml # conda environment specification (env name: SPockeR)
├── LICENSE
└── README.md

## Contact
For questions or issues, please open an issue on GitHub or contact
Raju Sarkar (rjdebsarkar@gmail.com).

## Acknowledgments
We thank the developers of the following open-source tools that SPockeR
relies on:
- volgrids
- pdb2pqr, APBS (electrostatics)
- pdbfixer, pdb-tools (structure preparation)
- RNApolis (RNA annotation)

## Citation
If you use SPockeR in your research, please cite:

    @article{2026spocker,
      author  = {Raju and ...},
      title   = {SPockeR: A SMIFs-based pipeline for RNA ligand-binding pocket detection},
      journal = {TBD},
      year    = {2026}
    }
