# Causally-anchored multi-omic deep learning for exercise-ageing gene prioritisation

Code and key data accompanying *"Causally-anchored multi-omic deep learning
recovers exercise-responsive and ageing-causal genes from human physical
activity."*

The pipeline combines **LD-aware, pleiotropy-filtered multi-omic Mendelian
randomisation (MR)** with a **supervised graph-attention network (GAT)** to
prioritise genes linking vigorous physical activity (VPA) to biological ageing,
and then subjects the eight triple-convergent genes to **systematic cis-MR with
colocalisation** against four ageing outcomes.

---

## What this repository contains

This repository provides the **model layer**, the **canonical model inputs**
(node and edge tables), the **model outputs** (per-seed rankings), the
**enrichment / convergence analysis** that produces Table 5.2, and the **full
CTSF causal-validation arm**, together with the per-layer MR scripts that were
saved as standalone files. It is reproducible from the provided node table
(`MTI_REBUILD_5LAYER.tsv`) onward. Upstream steps that were run interactively or
on separate data drives are described here and in the paper's Methods; their
outputs are included directly in the node table so that nothing downstream
depends on re-running them.

> **Corrections in this version.** Relative to an earlier exploratory run, this
> version (1) uses a correctly harmonised LD-aware MR step (an earlier run used
> an incorrect LD matrix), and (2) trains the GAT on multi-omic causal evidence
> and network topology **only** — AlphaFold-derived and PANTHER/InterPro/UniProt
> functional features are **not** used. The committed node table
> (`MTI_REBUILD_5LAYER.tsv`) contains only the five MR layers, the MTI score, and
> the multi-layer count; the trainer (`multiseed.py`) selects only the five MR
> feature columns. No structural or functional annotation enters the model.

---

## Pipeline

The analysis follows the five-stage sequence described in the paper's Methods:
ExWAS → multi-omic MR → MTI score → graph (construction + GAT) → enrichment,
with a separate cis-MR/colocalisation validation arm applied to the convergent
genes.

```
 1  ExWAS (VPA)                       86 lead-variant instruments
        │
 2  per-layer MR  ×5 layers           LD-aware, overlap-aware, Egger-filtered
        │                             → per-gene standardised betas (*_std)
 3  MTI score      compute_MTI.py     MTI(g) = sqrt(Σ_k (β_std,k)^2)
        │                             → node table (5 *_std betas + MTI)
 4  graph                             nodes = node table; edges = STRING ≥700
        │                             (17q21.31 excluded; connected N = 2,473)
        │           multiseed.py      GAT propagates MTI over the PPI graph
        │           multiseed_mlp.py  non-graph MLP baseline
        │           graph_ablation.py GAT-vs-MLP ablation
        │                             → rank_{GAT,MLP}_seed{0,1,7,42,123}.tsv
 5  enrichment     enrichment_comparison.R
                    rank-based GAT/MLP vs Ying & MoTrPAC (connected universe)
                    + model-free MR-FDR × Ying convergence (MR universe 2,959)

 validation arm   8 convergent genes → cis-MR (pQTL+eQTL) × 4 ageing outcomes
                  mr_CTSF_ageing_outcomes.R · coloc_longevity_CTSF_{pQTL,eQTL}.R
                  → colocalisation + MR-Steiger → CTSF (exceptional longevity)
```

1. **ExWAS** — exome-wide association of habitual VPA in UK Biobank (REGENIE,
   OQFE WES; p < 5×10⁻⁸; LocusZoom fine-mapping) → 86 lead-variant instruments.
2. **Per-layer MR** — LD-aware, overlap-aware, MR-Egger-filtered two-sample MR
   of VPA on each gene, in each of five molecular layers (protein, CpG, glycan,
   single-cell, bulk transcript), yielding per-gene standardised betas.
3. **MTI score** (`compute_MTI.py`) — per-layer standardised betas aggregated by
   Euclidean norm into the continuous multi-omic trait-importance (MTI) score,
   written into the GAT node table.
4. **Graph + GAT** — the node table (MTI layers) and the STRING edge table
   (high-confidence ≥700, 17q21.31 excluded; connected N = 2,473) define the
   graph; the GAT propagates the MTI signal over it (5 seeds), with an MLP
   baseline and a GAT-vs-MLP ablation.
5. **Enrichment / convergence** — GAT/MLP rankings tested for overrepresentation
   of the Ying ageing-causal and MoTrPAC exercise-responsive sets against the
   connected universe, plus a model-free MR-FDR × Ying convergence against the
   multi-omic MR universe (N = 2,959). This is Table 5.2.

**Validation arm** — the eight triple-convergent genes are tested by cis-MR
(cis-pQTL + cis-eQTL) against four ageing outcomes, with colocalisation and
MR-Steiger directionality; CTSF vs exceptional longevity is the single validated
pair.

---

## Scripts

Listed in pipeline order: MR → MTI → graph/GAT → enrichment, then the validation arm.

### Per-layer MR (`/mr`)

| Script | Layer |
|---|---|
| `MR_proteomic_FINAL.R` | Plasma proteomic (UKB-PPP), incl. overlap-aware correction |
| `MR_glycomic_FINAL_v2.R` | Plasma glycomic (TPNG N-glycome) |
| `MR_cpg_CLUMPED.R` | Whole-blood methylation (GoDMC) |

The **bulk-transcriptomic (eQTLGen)** and **single-cell (OneK1K)** layers were
computed with the same LD-aware MR procedure (Methods); their per-gene outputs
are included directly in the node table as the `transcript_std` and `sc_std`
columns. All downstream analyses are reproducible from the node table.

### MTI score (`/model`)

| Script | Role |
|---|---|
| `compute_MTI.py` | **MTI score.** Aggregates the five per-layer standardised MR betas (`protein_std, cpg_std, glycan_std, sc_std, transcript_std`) into the multi-omic trait-importance target `MTI(g) = sqrt(Σ_k (β_std,k)^2)` (Euclidean norm). Writes `MTI_score`, `MTI_sumsq`, `MTI_n_layers` into the node table. Reads only the five MR columns — no structural/functional (AlphaFold/PANTHER/InterPro/UniProt) features are read or used. The MTI is the GAT's regression target, never a node feature (a leakage guard excludes all MTI-derived columns from the feature matrix). |

### Graph + GAT (`/model`)

| Script | Role |
|---|---|
| `multiseed.py` | **Canonical GAT trainer.** Two-layer GATConv encoder, regression head (→ MTI) + classification head (→ multi-layer support, n_layers ≥ 2), hybrid ranking `z(MTI_pred) + α·z(σ(logit))`. Runs seeds {0, 1, 7, 42, 123} → `rank_GAT_seed*.tsv`. Reads only the five MR feature columns; no structural/functional features. |
| `multiseed_mlp.py` | Non-graph **MLP baseline**, same target and seeds → `rank_MLP_seed*.tsv`. |
| `graph_ablation.py` | **GAT-vs-MLP ablation** driver, isolating the contribution of the graph. |

### Enrichment / convergence (`/enrichment`)

| Script | Role | Verified output |
|---|---|---|
| `enrichment_comparison.R` | **Reproduces Table 5.2.** Rebuilds the n = 906 FDR-significant MR set, runs the model-free convergence, and regenerates the five-method × two-reference enrichment table. | MR-FDR × Ying: n = 906, x = 16, 1.6-fold, **p = 0.023**; GAT × Ying 0.024/0.024/0.021; GAT × MoTrPAC 0.007/0.014/0.071 |

See **"Table 5.2 — enrichment and convergence"** below for the exact inputs and
the definition of the n = 906 set.

### Validation arm (`/validation`)

| Script | Role | Verified output |
|---|---|---|
| `mr_CTSF_ageing_outcomes.R` | cis-MR of the convergent genes vs four ageing outcomes, both instrument arms | CTSF protein Wald β = 0.33, P = 1.8×10⁻³; expression IVW β = 0.19, P = 6.6×10⁻⁴, Q P = 0.99 |
| `coloc_longevity_CTSF_pQTL.R` | CTSF cis-pQTL × exceptional longevity colocalisation (UKB-PPP CTSF assay N = 33,822; longevity cc, s = 0.31) | nsnps = 900, PP.H3 = 0.182, PP.H4 = 0.632, **conditional PP.H4 = 0.78** |
| `coloc_longevity_CTSF_eQTL.R` | CTSF cis-eQTL × exceptional longevity colocalisation | nsnps = 2889, PP.H3 = 0.286, PP.H4 = 0.455, **conditional PP.H4 = 0.62** |

---

## Data provided

| File | Contents |
|---|---|
| `MTI_REBUILD_5LAYER.tsv` | **Node table.** `gene_symbol, MTI_score, MTI_n_layers, protein_std, cpg_std, glycan_std, sc_std, transcript_std`. The GAT's input — MR layers only, no annotations. |
| `STRING_edges_REBUILD.tsv` | STRING v12.0 high-confidence (≥700) edge list; 17q21.31 excluded. |
| `rank_GAT_seed{0,1,7,42,123}.tsv` | GAT rankings, five seeds. |
| `rank_MLP_seed{0,1,7,42,123}.tsv` | MLP baseline rankings, five seeds. |
| `ying_ageing_causal_genes.tsv` | Ying et al. ageing-causal reference (CausAge), 33 genes in the tested universe. |
| `motrpac_exercise_genes.tsv` | MoTrPAC exercise-responsive reference, 948 genes in the tested universe. |

The n = 906 FDR-significant MR set is **rebuilt by `enrichment_comparison.R`** from
the per-layer MR summaries (see below) rather than stored as a flat list, so the
definition is explicit and reproducible.

The single **validated gene (CTSF, exceptional longevity)** is not stored as a
separate list — it is the output of the validation-arm scripts
(`mr_CTSF_ageing_outcomes.R` + the two `coloc_longevity_CTSF_*.R`), the one
gene–outcome pair meeting all three criteria (FDR-significant MR, Steiger-
consistent direction, conditional PP.H4 > 0.7).

---

## Reproducing the results

### 1. MTI score (build the GAT regression target)
```bash
python compute_MTI.py --in_node MTI_REBUILD_5LAYER.tsv --out_node MTI_REBUILD_5LAYER.tsv
```
Recomputes `MTI_score`, `MTI_sumsq`, `MTI_n_layers` from the five per-layer
standardised betas already in the node table. The committed table already
carries these values; this step lets an examiner verify them from the betas.

### 2. Model (from the node + edge tables)
```bash
python multiseed.py      --node_path MTI_REBUILD_5LAYER.tsv --edge_path STRING_edges_REBUILD.tsv
python multiseed_mlp.py  --node_path MTI_REBUILD_5LAYER.tsv
python graph_ablation.py --node_path MTI_REBUILD_5LAYER.tsv --edge_path STRING_edges_REBUILD.tsv
```

### 3. Table 5.2 — enrichment and convergence
```bash
Rscript enrichment_comparison.R
```

**Two universes are kept separate.** The model-free convergence is tested
against the **multi-omic MR universe** (N = 2,959 MTI-scored genes); the GAT/MLP
rank rows are reported against the same MR universe with the reference sizes
CausAge = 33 and MoTrPAC = 948.

**The n = 906 drawn set** is the union of the per-layer FDR < 0.05 genes
(protein, single-cell, bulk-transcript) plus the glycan trait→gene map and the
single-cell gene `MEF2C`, intersected with the MR universe. The script builds it
from these inputs:

```
mr_outputs_PPP_REAL/PPP_PA_LDaware_REAL_summary.tsv   (protein,    fdr < 0.05)
mr_outputs_TX/TX_PA_KEEPALL_summary.tsv               (single-cell, fdr < 0.05)
transcript_MR_eQTLGen_FULL.csv                        (transcript, transcript_fdr < 0.05)
glycan_trait_gene_map.rds                             (glycan trait→gene map)
ying_targets.rds        →  CausAge (33 in universe)
motrpac_EE_blood_genes.rds  (948 in universe)
MTI_REBUILD_5LAYER.rds  →  multi-omic MR universe (2,959)
```

Expected output (matches the paper):

| Method | Ying K100/150/200 | MoTrPAC K100/150/200 |
|---|---|---|
| MR-FDR set (n = 906) | **0.023** / — / — | 0.969 / — / — |
| MR p-value rank | 0.307 / 0.505 / 0.665 | 0.971 / 0.994 / 0.985 |
| MTI (β) rank | 0.307 / 0.505 / 0.665 | 0.839 / 0.793 / 0.953 |
| MLP (no graph) | **0.024** / 0.083 / 0.180 | 0.118 / 0.212 / 0.243 |
| GAT (graph) | **0.024** / **0.024** / **0.021** | **0.007** / **0.014** / 0.071 |

The model-free convergence (MR-FDR × Ying) is **16 observed vs 10.1 expected,
1.6-fold, p = 0.023**. The rank rows report the representative initialisation
(seed 42); the GAT enrichments are robust across all five seeds (5/5 significant
at K = 100 for both reference sets).

### 4. CTSF validation (from public summary statistics)
```bash
Rscript mr_CTSF_ageing_outcomes.R     # → β=0.33 protein, β=0.19 expression
Rscript coloc_longevity_CTSF_pQTL.R   # → conditional PP.H4 = 0.78
Rscript coloc_longevity_CTSF_eQTL.R   # → conditional PP.H4 = 0.62
```

A gene–outcome pair is **validated** only if it satisfies all three criteria:
FDR-significant MR, Steiger-consistent direction, and conditional PP.H4 > 0.7.
CTSF vs exceptional longevity is the single pair meeting all three.

---

## Data sources

All summary statistics are publicly available; this repository contains code and
gene lists only, not redistributed GWAS data.

**Molecular layers:** UKB-PPP proteomics (Sun et al. 2023, doi:10.1038/s41586-023-06592-6);
GoDMC methylation QTL; TPNG plasma N-glycome; eQTLGen cis-eQTL (Võsa et al. 2021,
doi:10.1038/s41588-021-00913-z); OneK1K single-cell cis-eQTL (Yazar et al. 2022).
STRING v12.0 (Szklarczyk et al. 2023).

**Ageing outcomes:** exceptional longevity (Deelen et al. 2019,
doi:10.1038/s41467-019-11558-2); parental lifespan (Timmers et al. 2019,
doi:10.7554/eLife.39856); aging-GIP1 (Timmers et al. 2022,
doi:10.1038/s43587-021-00159-8); healthspan (Zenin et al. 2019,
doi:10.1038/s42003-019-0290-0).

**Reference gene sets:** ageing-causal (Ying et al. 2024,
doi:10.1038/s43587-023-00557-0); exercise-responsive (MoTrPAC Study Group, human
acute exercise).

**Methods:** coloc (Giambartolomei et al. 2014, doi:10.1371/journal.pgen.1004383);
conditional colocalisation (Wallace 2020, doi:10.1371/journal.pgen.1008720).

---

## Environment

**Python** (model layer): PyTorch, PyTorch Geometric, pandas, numpy.
**R** (MR / coloc / enrichment): `coloc` (5.2.3), `data.table`, `ieugwasr` (1.1.0).
LD clumping uses the 1000 Genomes EUR panel via OpenGWAS (set an `OPENGWAS_JWT` token).
