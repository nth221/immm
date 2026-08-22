<div align="center">

# IM<sup>3</sup>

### Interaction Region-Augmented Multimodal Molecular Modeling for Precision Prescription

[![Conference](https://img.shields.io/badge/CIKM-2026-B31B1B.svg?style=flat-square)](https://cikm2026.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3799682.3841136-1F6FEB.svg?style=flat-square)](https://doi.org/10.1145/3799682.3841136)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-EFB700.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0%2Bcu118-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)

<img src="assets/pipeline.jpg" width="100%" alt="Overview of the IM3 framework"/>

</div>

---

Official implementation of **"IM<sup>3</sup>: Interaction Region-Augmented Multimodal Molecular Modeling for Precision Prescription"**, to appear at **CIKM '26** (Rome, Italy).

IM<sup>3</sup> recommends medication combinations from longitudinal EHRs while explicitly suppressing drug-drug interaction (DDI) risk. It does so by encoding each drug through its **Interaction Region Indicator (IRI)** field, a real-space electronic descriptor derived from DFT, instead of relying on molecular topology or historical co-prescription statistics alone.

**Two ideas carry the method:**

- **Dual-level formulation.** DDIs arise between specific compounds, but prescriptions are judged by therapeutic class. IM<sup>3</sup> keeps these apart: interaction risk is modeled and penalized at the **CID level** (PubChem Compound ID), while therapeutic validity is supervised and evaluated at the **ATC3 level**. Molecular representations are never collapsed into class labels during inference.
- **Quantum-informed drug representations rather than topological drug representation.** Molecules with near-identical graph topology can interact very differently. IM<sup>3</sup> pairs a **global** encoder over DFT-derived IRI fields with a **local** encoder over BRICS substructures, capturing interaction propensity that static connectivity does not expose.

---

## Architecture

Three components, wired end to end in [`src/IM3.py`](src/IM3.py). Notation follows the paper: $\mathcal{M}$ is the CID-level medication vocabulary, $\mathcal{S}$ the BRICS fragment vocabulary, and $d$ the shared embedding dimension.

| Component | Role | Output |
|---|---|---|
| **Patient Context Encoder** | Diagnosis $\mathcal{D}^{(t)}$ and procedure $\mathcal{P}^{(t)}$ codes summed per visit, then two independent GRUs | $\mathbf{q}^{(t)}$ |
| **GQM Encoder** *(global)* | IRI field sliced at 0.25 Å into $\mathcal{I}_m$, encoded by a frozen ViViT and average-pooled | $\mathbf{V}_{\mathrm{G}}$ |
| **LSM Encoder** *(local)* | BRICS fragments over Uni-Mol2 atom embeddings, $(\mathbf{E}_{\mathrm{S}}\mathbf{B})^{\top}$ | $\mathbf{V}_{\mathrm{L}}$ |

**Fusion and scoring.** Both drug matrices are layer-normalized, fused element-wise, and scored against the patient embedding:

$$
\mathbf{E}_{\mathrm{d}} = \bar{\mathbf{V}}_{\mathrm{G}} \odot \bar{\mathbf{V}}_{\mathrm{L}}, \qquad \hat{\mathbf{o}}^{(t)} = \sigma\left(\mathbf{E}_{\mathrm{d}} \, \mathbf{q}^{(t)}\right) \in \mathbb{R}^{|\mathcal{M}|}
$$

**Dual-level output.** ATC3 logits pool the CID logits of each class's member compounds and add a term conditioned on $\mathbf{q}^{(t)}$. The CID head drives prediction and the DDI penalty; the ATC3 head carries the therapeutic supervision and is what the reported Jaccard, F1-score, AUPRC, and AUROC are computed on. Molecular representations are never aggregated during inference.

**Objective.** Prescription accuracy and interaction safety are balanced at the compound level, with auxiliary supervision at the class level:

$$
\mathcal{L}^{(t)} = \underbrace{\beta \mathcal{L}^{(t)}_{\mathrm{base}} + (1-\beta) \mathcal{L}^{(t)}_{\mathrm{DDI}}}_{\mathcal{L}^{(t)}_{\mathrm{CID}}} + \mathcal{L}^{(t)}_{\mathrm{ATC3}}
$$

where $\mathcal{L}^{(t)}_{\mathrm{base}} = \alpha \mathcal{L}^{(t)}_{\mathrm{bce}} + (1-\alpha) \mathcal{L}^{(t)}_{\mathrm{multi}}$, the DDI term penalizes interacting CID pairs from TWOSIDES, and $\beta$ decays from 1 once the running DDI rate exceeds $\gamma$. We set $\alpha = 0.95$ and $K_p = 0.05$ for both datasets, with $\gamma = 0.04$ for MIMIC-III and $\gamma = 0.01$ for MIMIC-IV.

## Repository layout

```
IM3/
├── src/
│   ├── IM3.py               # model: patient GRUs, GQM/LSM encoders, fusion, ATC3 head
│   ├── main.py              # training / evaluation loop, dual-level metrics, seed sweep
│   ├── utils.py             # DDI rate, multi-label metrics (CID & ATC3), MPNN helpers
│   └── baselines/           # reference implementations: RETAIN, SafeDrug, COGNet, SHAPE
├── data/
│   ├── processing.py        # MIMIC → NDC→RxCUI→CID normalization, DDI adjacency, BRICS masks
│   ├── mimic-iii/
│   │   ├── input/           # raw MIMIC tables + external mapping files (see below)
│   │   ├── output/          # generated .pkl artifacts
│   │   ├── MIMIC-III_250_IRI_ViViT_cls_embeddings.pkl
│   │   ├── MIMIC-III_250_BRICSFragVocab_UNIMOL2_{H,E_loc}_512.npy
│   │   └── final_used_cid_atc3_smiles_250.csv
│   └── mimic-iv/            # same structure, 597 medications
└── assets/
```

---

## Setup

<details open>
<summary><b>Environment</b>: install in this order to avoid dependency conflicts</summary>

```bash
# 1. conda environment
conda create -c conda-forge -n im3 python=3.9
conda activate im3

# 2. RDKit (required by data/processing.py for BRICS fragmentation)
conda install -c conda-forge rdkit
#   fallback if the conda build misbehaves:
#   pip install rdkit-pypi

# 3. pinned scientific stack
pip install numpy==1.22.4 pandas==1.3.0 scipy==1.13.1

# 4. PyTorch 2.3.0 + CUDA 11.8
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu118

# 5. remaining utilities
pip install dill scikit-learn matplotlib tqdm
```

</details>

---

## Data

Both datasets contain longitudinal ICU records: diagnoses, procedures, and prescribed medications. The medication vocabulary is CID-indexed, so counts refer to distinct chemical compounds rather than therapeutic classes.

| Items | MIMIC-III | MIMIC-IV |
|---|---|---|
| # of patients | 5,413 | 52,696 |
| # of visits | 14,057 | 177,145 |
| # of diagnoses | 1,942 | 3,000 |
| # of procedures | 1,399 | 2,000 |
| # of medications | 250 | 597 |
| avg # of visits | 2.60 | 3.36 |
| avg # of diagnoses | 13.72 | 13.12 |
| avg # of procedures | 4.58 | 2.62 |
| avg # of medications | 11.95 | 10.56 |
| # of substructures | 331 | 775 |
| empirical DDI rate | 0.2509 | 0.1704 |

The empirical DDI rate is a proportion, so 0.2509 corresponds to 25.09% of co-prescribed pairs in the observed records being flagged as interacting.

### 1. Obtain the source records

MIMIC-III and MIMIC-IV are credentialed datasets. Request access through [PhysioNet](https://physionet.org/) and place the tables under `data/<dataset>/input/`:

```
data/mimic-iv/input/MIMIC-iv/3.1/{prescriptions,diagnoses_icd,procedures_icd, admissions}.csv
```

### 2. Add the external mapping files

| File | Description |
|---|---|
| **`drug-DDI.csv`** | CID-indexed DDI records from [TWOSIDES](https://doi.org/10.1126/scitranslmed.3003377). Download from [Google Drive](https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing). |
| **`mimic{3,4}_ndc_rxcui_inn_cid_smiles_atc3.csv`** | NDC → RxCUI → active ingredient → CID → SMILES → ATC3 normalization table. Included in this repository. |


### 3. Run preprocessing

```bash
cd data
python processing.py
```

This writes the following into `data/<dataset>/output/`:

| Artifact | Contents |
|---|---|
| `records_final.pkl` | Per-patient visit sequences `[diagnoses, procedures, medications]` |
| `voc_final.pkl` | `diag_voc` / `pro_voc` / `med_voc` (medication vocabulary is CID-indexed) |
| `ddi_A_final.pkl` | Binary CID×CID DDI adjacency matrix |
| `ddi_mask_H.pkl` | Drug-substructure incidence mask |
| `cidtoSMILES.pkl`, `cidtoATC3.pkl` | Molecular and therapeutic-class lookups |

**Note.** `processing.py` currently resolves paths against `data/mimic-iv/`. Adjust `BASE_DIR` at the top of the file to regenerate the MIMIC-III artifacts.

---

## Training & evaluation

The active dataset is a module-level constant near the top of [`src/main.py`](src/main.py):

```python
DATASET = "mimic-iii"    # or "mimic-iv"
```

All data paths, including the precomputed IRI and BRICS tensors, are resolved from it, so the script runs from any working directory.

```bash
cd src
python main.py                                        # train (sweeps 6 seed sets)
python main.py --Test --resume_path [best_epoch.model]   # evaluate a checkpoint
```

Checkpoints land in `_saved/IM3_<dataset>_Set<i>__GQM1_LSM1/`, one file per epoch, tagged with its validation ATC3 Jaccard, CID Jaccard, and DDI rate. Test mode reports mean ± std over 10 bootstrap resamples (80% of the test split each).

### Configuration

Hyperparameters are `argparse` flags. Defaults target MIMIC-III; override the two starred rows for MIMIC-IV.

| Flag | MIMIC-III | MIMIC-IV | Meaning |
|---|---|---|---|
| `--lr` | `2e-4` | `2e-4` | Adam learning rate |
| `--dim` ★ | `128` | `512` | Embedding / hidden dimension $d$ |
| `--target_ddi` ★ | `0.04` | `0.01` | Target DDI threshold $\gamma$ |
| `--kp` | `0.05` | `0.05` | Adaptive feedback coefficient $K_p$ |
| `--cuda` | `0` | `0` | CUDA device index |
| `--fixed_thr_cid` | `None` | `None` | Fix the CID decision threshold at test time |
| `--fixed_thr_atc3` | `None` | `None` | Fix the ATC3 decision threshold at test time |

Training runs 50 epochs. Decision thresholds are **tuned on validation** by maximizing sample-averaged F1 over a 0.01 to 0.99 grid, then held fixed at test time, so pass them explicitly via `--fixed_thr_*` when evaluating a checkpoint.

### Ablations

The two molecular encoders are toggled independently, reproducing Table 4 of the paper. Checkpoints are kept in separate directories per configuration.

```bash
python main.py --use_gqm 0      # w/o GQM Encoder: no IRI, structure only
python main.py --use_lsm 0      # w/o LSM Encoder: no BRICS fragments
```

---

## Results

Dual-level protocol: **DDI Rate** and **Avg # Drugs** are measured at the CID level, while **Jaccard**, **F1-score**, **AUPRC**, and **AUROC** are evaluated at the ATC3 level. All baselines are trained and evaluated on the same CID-level medication vocabulary as IM<sup>3</sup>, rather than through post-hoc mapping from ATC-level outputs. Results are reported as mean ± std over six random seeds. For each metric, the best result and any result not statistically distinguishable from it are shown in **bold** (paired *t*-test, *p* ≥ 0.05).

<details open>
<summary><b>MIMIC-III</b>: ground truth 11.95 drugs per visit, empirical DDI rate 0.2509</summary>

<br/>

| Category | Model | DDI Rate (↓) | Jaccard (↑) | F1-score (↑) | AUPRC (↑) | AUROC (↑) | Avg # Drugs |
|---|---|---|---|---|---|---|---|
| Traditional machine learning | RF | 0.2616 ± 0.0042 | 0.4600 ± 0.0032 | 0.6162 ± 0.0035 | 0.6912 ± 0.0039 | 0.9316 ± 0.0014 | 11.5832 ± 0.0532 |
| Traditional machine learning | LR | 0.2567 ± 0.0052 | 0.4744 ± 0.0043 | 0.6297 ± 0.0046 | **0.7008 ± 0.0346** | **0.9360 ± 0.0012** | 12.3297 ± 0.8541 |
| Sequential EHR | RETAIN | 0.3075 ± 0.0077 | 0.4293 ± 0.0068 | 0.5850 ± 0.0070 | 0.6835 ± 0.0028 | 0.9297 ± 0.0008 | 6.4801 ± 0.1328 |
| Sequential EHR | LEAP | 0.2508 ± 0.0063 | 0.4237 ± 0.0022 | 0.5818 ± 0.0026 | 0.5264 ± 0.0059 | 0.8847 ± 0.0028 | 12.7824 ± 0.5204 |
| Sequential EHR | DMNC | 0.2581 ± 0.0098 | 0.3921 ± 0.0175 | 0.5487 ± 0.0179 | 0.5265 ± 0.0225 | 0.8806 ± 0.0081 | 12.2454 ± 1.1388 |
| Sequential EHR | COGNet | 0.2148 ± 0.0048 | 0.4631 ± 0.0018 | 0.6170 ± 0.0017 | 0.6476 ± 0.0047 | 0.9123 ± 0.0024 | 16.1053 ± 0.4031 |
| Interaction-aware | GAMENet | 0.2279 ± 0.0033 | 0.4270 ± 0.0031 | 0.5866 ± 0.0031 | 0.5557 ± 0.0034 | 0.9017 ± 0.0010 | 25.9611 ± 0.4898 |
| Interaction-aware | SHAPE | 0.2337 ± 0.0049 | 0.4729 ± 0.0037 | 0.6270 ± 0.0038 | 0.6346 ± 0.0033 | 0.9123 ± 0.0007 | 11.8267 ± 0.5519 |
| Interaction-aware | RASNet | 0.0838 ± 0.0007 | 0.4432 ± 0.0025 | 0.6006 ± 0.0025 | 0.5949 ± 0.0053 | 0.8977 ± 0.0011 | 13.4909 ± 0.2321 |
| Interaction-aware | StratMed | 0.0832 ± 0.0017 | 0.4241 ± 0.0084 | 0.5823 ± 0.0081 | 0.5808 ± 0.0034 | 0.8920 ± 0.0014 | 13.2899 ± 0.8666 |
| Interaction-aware | ARMR | 0.2280 ± 0.0009 | 0.4835 ± 0.0014 | **0.6397 ± 0.0012** | 0.6156 ± 0.0013 | 0.9166 ± 0.0004 | 18.8027 ± 0.1651 |
| Molecular structure-aware | SafeDrug | 0.0823 ± 0.0055 | 0.4226 ± 0.0088 | 0.5805 ± 0.0087 | 0.5992 ± 0.0139 | 0.8951 ± 0.0063 | 12.0011 ± 0.9702 |
| Molecular structure-aware | MoleRec | 0.0864 ± 0.0018 | 0.4530 ± 0.0014 | 0.6100 ± 0.0016 | 0.6065 ± 0.0054 | 0.8990 ± 0.0022 | 13.1202 ± 0.4911 |
| Molecular structure-aware | DNMDR | 0.0846 ± 0.0026 | 0.4488 ± 0.0026 | 0.6060 ± 0.0022 | 0.6004 ± 0.0082 | 0.8980 ± 0.0024 | 13.3673 ± 0.5803 |
| **Proposed approach** | **IM<sup>3</sup>** | **0.0295 ± 0.0043** | **0.4874 ± 0.0018** | **0.6421 ± 0.0016** | **0.7282 ± 0.0014** | **0.9356 ± 0.0005** | 12.8677 ± 2.0642 |

</details>

<details>
<summary><b>MIMIC-IV</b>: ground truth 10.56 drugs per visit, empirical DDI rate 0.1704</summary>

<br/>

| Category | Model | DDI Rate (↓) | Jaccard (↑) | F1-score (↑) | AUPRC (↑) | AUROC (↑) | Avg # Drugs |
|---|---|---|---|---|---|---|---|
| Traditional machine learning | RF | 0.1975 ± 0.0009 | 0.3851 ± 0.0014 | 0.5333 ± 0.0015 | 0.5904 ± 0.0010 | 0.9388 ± 0.0003 | 12.5826 ± 0.0382 |
| Traditional machine learning | LR | 0.1767 ± 0.0005 | 0.4239 ± 0.0016 | 0.5728 ± 0.0016 | 0.6382 ± 0.0016 | 0.9487 ± 0.0004 | 13.1033 ± 0.0601 |
| Sequential EHR | RETAIN | 0.1772 ± 0.0178 | 0.4259 ± 0.0050 | 0.5584 ± 0.0058 | **0.6560 ± 0.0128** | **0.9588 ± 0.0033** | 5.2978 ± 0.1538 |
| Sequential EHR | LEAP | 0.2322 ± 0.0069 | 0.3746 ± 0.0018 | 0.5164 ± 0.0022 | 0.4475 ± 0.0086 | 0.9145 ± 0.0014 | 11.5594 ± 0.1301 |
| Sequential EHR | DMNC | 0.2102 ± 0.0140 | 0.3823 ± 0.0069 | 0.5276 ± 0.0052 | 0.4389 ± 0.0334 | 0.9101 ± 0.0054 | 12.0870 ± 0.7796 |
| Sequential EHR | COGNet | 0.1604 ± 0.0025 | 0.4378 ± 0.0023 | 0.5801 ± 0.0025 | 0.5356 ± 0.0047 | 0.9313 ± 0.0016 | 12.2390 ± 0.1728 |
| Interaction-aware | GAMENet | 0.1681 ± 0.0023 | 0.3019 ± 0.0020 | 0.4469 ± 0.0022 | 0.4252 ± 0.0021 | 0.9151 ± 0.0007 | 39.3438 ± 0.6154 |
| Interaction-aware | SHAPE | 0.1716 ± 0.0040 | **0.4522 ± 0.0031** | **0.5938 ± 0.0034** | 0.6064 ± 0.0025 | 0.9458 ± 0.0006 | 10.3731 ± 0.3217 |
| Interaction-aware | RASNet | 0.0659 ± 0.0021 | 0.3942 ± 0.0048 | 0.5432 ± 0.0041 | 0.5229 ± 0.0052 | 0.9295 ± 0.0012 | 17.0710 ± 0.5247 |
| Interaction-aware | StratMed | 0.0616 ± 0.0003 | 0.3778 ± 0.0018 | 0.5276 ± 0.0018 | 0.5052 ± 0.0017 | 0.9260 ± 0.0003 | 21.1390 ± 0.0859 |
| Interaction-aware | ARMR | 0.1720 ± 0.0004 | 0.3753 ± 0.0005 | 0.5266 ± 0.0005 | 0.4972 ± 0.0006 | 0.9358 ± 0.0001 | 25.1805 ± 0.0959 |
| Molecular structure-aware | SafeDrug | 0.0671 ± 0.0028 | 0.3693 ± 0.0077 | 0.5197 ± 0.0080 | 0.5047 ± 0.0085 | 0.9275 ± 0.0031 | 17.3048 ± 0.7992 |
| Molecular structure-aware | MoleRec | 0.0669 ± 0.0012 | 0.3921 ± 0.0045 | 0.5433 ± 0.0041 | 0.5128 ± 0.0047 | 0.9284 ± 0.0015 | 20.6069 ± 0.6709 |
| Molecular structure-aware | DNMDR | 0.0647 ± 0.0012 | 0.3861 ± 0.0058 | 0.5365 ± 0.0062 | 0.5184 ± 0.0103 | 0.9297 ± 0.0023 | 18.7595 ± 1.0861 |
| **Proposed approach** | **IM<sup>3</sup>** | **0.0238 ± 0.0005** | 0.4145 ± 0.0014 | 0.5633 ± 0.0012 | 0.6298 ± 0.0016 | 0.9466 ± 0.0003 | 11.9522 ± 0.2001 |

</details>

<details>
<summary><b>Ablation</b> on MIMIC-III: what each encoder contributes</summary>

<br/>

| Model | DDI Rate (↓) | Jaccard (↑) | F1-score (↑) | AUPRC (↑) | AUROC (↑) | Avg # Drugs |
|---|---|---|---|---|---|---|
| w/o GQM Encoder | 0.0766 ± 0.0034 | 0.4843 ± 0.0017 | 0.6393 ± 0.0016 | 0.7242 ± 0.0025 | 0.9343 ± 0.0010 | 15.0076 ± 1.0824 |
| w/o LSM Encoder | 0.0392 ± 0.0116 | **0.4923 ± 0.0022** | **0.6463 ± 0.0022** | **0.7304 ± 0.0029** | **0.9361 ± 0.0008** | 15.6720 ± 1.5566 |
| **IM<sup>3</sup>** | **0.0295 ± 0.0043** | 0.4874 ± 0.0018 | 0.6421 ± 0.0016 | 0.7282 ± 0.0014 | 0.9356 ± 0.0005 | 12.8677 ± 2.0642 |

Removing GQM inflates the DDI rate 2.6×, so IRI carries interaction signal that structure alone does not. Removing LSM nudges accuracy up but loosens prescription size, so the fragment encoder acts as a structural regularizer. Neither encoder achieves the full model's balance in isolation.

</details>

IM<sup>3</sup> reaches the lowest DDI rate on both datasets, a reduction of over 60% against the strongest molecular structure-aware baseline, while staying competitive on therapeutic relevance and without shrinking the prescription set.

---

## Reproducibility

- `main.py` sweeps **six seed sets** in a single invocation (`(1001,…), (2001,…), …, (6001,…)`), covering `torch`, `numpy`, `random`, and CUDA, with `cudnn.deterministic = True`.
- Patients are split **4:1:1** (train / validation / test) at the patient level, not the visit level.
- Decision thresholds are selected on validation only; no test-set information enters threshold selection.
- The IRI and BRICS tensors are frozen buffers, so molecular representations are byte-identical across every run and every ablation.

---


## Citation

```bibtex
@inproceedings{kim2026im3,
  title     = {{IM}$^3$: Interaction Region-Augmented Multimodal Molecular Modeling for Precision Prescription},
  author    = {Kim, Yujin and Park, Seoeun and Hong, Charmgil},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  address   = {Rome, Italy},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3799682.3841136},
  isbn      = {979-8-4007-2539-5/2026/11}
}
```

<details>
<summary><b>Related work from this repository's lineage and dependencies</b></summary>

```bibtex
@article{lu2021iri,
  title   = {Interaction Region Indicator: A Simple Real Space Function Clearly
             Revealing Both Chemical Bonds and Weak Interactions},
  author  = {Lu, Tian and Chen, Qinxue},
  journal = {Chemistry--Methods},
  volume  = {1}, number = {5}, pages = {231--239}, year = {2021}
}

@inproceedings{arnab2021vivit,
  title     = {ViViT: A Video Vision Transformer},
  author    = {Arnab, Anurag and Dehghani, Mostafa and Heigold, Georg and Sun, Chen
               and Lu{\v{c}}i{\'c}, Mario and Schmid, Cordelia},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages     = {6836--6846}, year = {2021}
}

@article{ji2024unimol2,
  title   = {Uni-Mol2: Exploring Molecular Pretraining Model at Scale},
  author  = {Ji, Xiaohong and Wang, Zhen and Gao, Zhifeng and Zheng, Hang and
             Zhang, Linfeng and Ke, Guolin},
  journal = {arXiv preprint arXiv:2406.14969}, year = {2024}
}

@inproceedings{yang2021safedrug,
  title     = {SafeDrug: Dual Molecular Graph Encoders for Safe Drug Recommendations},
  author    = {Yang, Chaoqi and Xiao, Cao and Ma, Fenglong and Glass, Lucas and Sun, Jimeng},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence (IJCAI 2021)},
  year      = {2021}
}
```

</details>
