# Causally-anchored multi-omic deep learning for exercise-ageing gene prioritisation

Code and key data accompanying *"Causally-anchored multi-omic deep learning
recovers exercise-responsive and ageing-causal genes from human physical
activity."*

The pipeline combines **linkage disequilibrium (LD)-aware, overlap-aware, pleiotropy-filtered multi-omic
Mendelian randomisation (MR)** with a **supervised graph-attention network (GAT)**
to prioritise genes linking vigorous physical activity (VPA) to biological ageing,
then subjects the eight triple-convergent genes to **systematic cis-MR with
colocalisation** against four ageing outcomes.

---

## What this repository contains

The **model layer**, the **canonical model inputs** (node and edge tables), the
**model outputs** (per-seed rankings), the **enrichment / convergence analysis**
that produces Table 5.2, and the **full CTSF causal-validation arm**, together
with the per-layer MR scripts saved as standalone files. Reproducible from the
provided node table (`MTI_REBUILD_5LAYER.tsv`) onward. Upstream steps run
interactively or on separate data drives are described in the Methods; their
outputs are included directly in the node table, so nothing downstream depends on
re-running them.

> **Corrections in this version.** Relative to an earlier exploratory run, this
> version (1) uses a correctly harmonised LD-aware MR step (an earlier run used
> an incorrect LD matrix), and (2) trains the GAT on multi-omic causal evidence
> and network topology **only** — AlphaFold-derived and PANTHER/InterPro/UniProt
> functional features are **not** used. The committed node table contains only
> the five MR layers, the MTI score, and the multi-layer count; `multiseed.py`
> selects only the five MR feature columns. No structural or functional
> annotation enters the model.

---

## Pipeline

```
 1  ExWAS (VPA)                       86 lead-variant instruments
        │                             UKBB OQFE WES, n ≈ 75,000 accelerometry
        │                             REGENIE, p < 5×10⁻⁸, LocusZoom fine-mapping
 2  per-layer MR  ×5 layers           LD-aware, overlap-aware, Egger-filtered
        │                             → per-gene standardised betas (*_std)
 3  MTI score      compute_MTI.py     MTI(g) = sqrt(Σ_k (β_std,k)²)
        │                             → node table (5 *_std betas + MTI)
 4  graph                             nodes = node table; edges = STRING ≥700
        │                             genes with no high-confidence edge excluded
        │                             → 2,473 connected genes / 17,193 edges
        │           multiseed.py      GAT propagates MTI over the PPI graph
        │           multiseed_mlp.py  non-graph MLP baseline
        │           graph_ablation.py GAT-vs-MLP ablation
        │                             → rank_{GAT,MLP}_seed{0,1,7,42,123}.tsv
 5  enrichment     enrichment_comparison.R
                    GAT/MLP rank rows      → connected universe, N = 2,473
                    model-free convergence → multi-omic MR universe, N = 2,959

 validation arm   8 convergent genes → cis-MR (pQTL+eQTL) × 4 ageing outcomes
                  mr_CTSF_ageing_outcomes.R · coloc_longevity_CTSF_{pQTL,eQTL}.R
                  → colocalisation + MR-Steiger → CTSF (exceptional longevity)
```

---

## Molecular layers

| Layer | Tissue | Dataset | N | Ancestry |
|---|---|---|---|---|
| Proteomic | Plasma | UKBB Pharma Proteomics Project | ~54,000 | Predominantly European |
| Epigenomic (CpG) | Whole blood | GoDMC | ~27,750 | European |
| Glycomic | Plasma | TPNG GWAMA (N-glycome) | ~10,000 | European |
| Bulk transcriptomic | Whole blood | eQTLGen | ~31,684 | European |
| Single-cell transcriptomic | PBMCs | OneK1K | 982 | European |

**On the single-cell layer.** OneK1K (n = 982) is markedly smaller than the other
outcome datasets, so its cell-type eQTL are correspondingly less powered. The
LD-aware, pleiotropy-filtered MR retains only variants that are simultaneously VPA
instruments and significant cell-type eQTL, and under these stringent, uniformly
applied criteria the layer yields very few genes. This is an expected consequence
of honest instrument selection at this sample size, not a deficiency of the
pipeline. Only summary-level cell-type cis-eQTL were used; no individual-level
single-cell data were accessed.

---

## Scripts

### Per-layer MR (`/mr`)

| Script | Layer |
|---|---|
| `MR_proteomic_FINAL.R` | Plasma proteomic (UKB-PPP), incl. overlap-aware correction |
| `MR_glycomic_FINAL_v2.R` | Plasma glycomic (TPNG N-glycome) |
| `MR_cpg_CLUMPED.R` | Whole-blood methylation (GoDMC) |

The **bulk-transcriptomic (eQTLGen)** and **single-cell (OneK1K)** layers were
computed with the same LD-aware MR procedure (Methods 5.2.4); their per-gene
outputs are in the node table as `transcript_std` and `sc_std`.

### MTI score (`/model`)

| Script | Role |
|---|---|
| `compute_MTI.py` | Aggregates the five per-layer standardised MR betas (`protein_std, cpg_std, glycan_std, sc_std, transcript_std`) into `MTI(g) = sqrt(Σ_k (β_std,k)²)`, where `β_std = β / SD_k` within each layer. All layers weighted equally. Writes `MTI_score`, `MTI_sumsq`, `MTI_n_layers`. Reads only the five MR columns. |

### Graph + GAT (`/model`)

| Script | Role |
|---|---|
| `multiseed.py` | **Canonical GAT trainer.** GATConv → ELU → Dropout → GATConv encoder; regression head (→ MTI) + classification head (→ `MTI_n_layers ≥ 2`); loss `MSE + λ·BCEWithLogits` with `λ = 0.3` and `pos_weight` capped at 50; Adam lr = 1e-3, 300 epochs. Hybrid ranking `S_g = z(MTI_pred) + α·z(σ(logit))`, α = 1.0. Seeds {0, 1, 7, 42, 123} → `rank_GAT_seed*.tsv`. |
| `multiseed_mlp.py` | Non-graph **MLP baseline**, same target and seeds → `rank_MLP_seed*.tsv`. |
| `graph_ablation.py` | **GAT-vs-MLP ablation** driver. |

### Enrichment / convergence (`/enrichment`)

| Script | Role |
|---|---|
| `enrichment_comparison.R` | Reproduces Table 5.2: rebuilds the MR-anchored drawn set, runs the model-free convergence, and regenerates the five-method × two-reference enrichment table across K = 100/150/200 and all five seeds. |
| `degree_matched_null.py` | **Degree-matched null.** Bins genes by STRING degree, resamples each top-K matching its observed degree profile, and returns an empirical p-value alongside the hypergeometric one. Tests whether the enrichment is an artefact of node connectivity. |

### Validation arm (`/validation`)

| Script | Role | Verified output |
|---|---|---|
| `mr_CTSF_ageing_outcomes.R` | cis-MR of the convergent genes vs four ageing outcomes, both instrument arms | CTSF protein Wald β = 0.33, P = 1.8×10⁻³; expression IVW β = 0.19, P = 6.6×10⁻⁴, Q P = 0.99 |
| `coloc_longevity_CTSF_pQTL.R` | CTSF cis-pQTL × exceptional longevity (UKB-PPP CTSF assay N = 33,822; longevity cc, s = 0.31) | nsnps = 900, PP.H3 = 0.182, PP.H4 = 0.632, **conditional PP.H4 = 0.78** |
| `coloc_longevity_CTSF_eQTL.R` | CTSF cis-eQTL × exceptional longevity | nsnps = 2889, PP.H3 = 0.286, PP.H4 = 0.455, **conditional PP.H4 = 0.62** |

---

## Data provided

| File | Contents |
|---|---|
| `MTI_REBUILD_5LAYER.tsv` | **Node table.** `gene_symbol, MTI_score, MTI_n_layers, protein_std, cpg_std, glycan_std, sc_std, transcript_std`. MR layers only, no annotations. |
| `STRING_edges_REBUILD.tsv` | STRING v12.0 high-confidence (≥700) edge list; 17q21.31 excluded. |
| `rank_GAT_seed{0,1,7,42,123}.tsv` | GAT rankings, five seeds, 2,473 genes each. |
| `rank_MLP_seed{0,1,7,42,123}.tsv` | MLP baseline rankings, five seeds, 2,473 genes each. |
| `ying_ageing_causal_genes.tsv` | Ying et al. ageing-causal reference (CausAge); 33 genes in the MR universe, **30 in the connected graph**. |
| `motrpac_exercise_genes.tsv` | MoTrPAC exercise-responsive reference; 948 genes in the MR universe, **831 in the connected graph**. |
| `degree_matched_null_results.tsv` | Per-seed output of `degree_matched_null.py`: observed counts, hypergeometric and degree-matched expectations and p-values, and top-K median degree. |

### Inputs required by `enrichment_comparison.R`


The drawn MR-anchored set is rebuilt from the per-layer MR summaries rather than
read from a stored list, so its definition is explicit and auditable. The
following intermediates are **available on request** and are not committed:

```
mr_outputs_PPP_REAL/PPP_PA_LDaware_REAL_summary.tsv   (protein,     fdr < 0.05)
mr_outputs_TX/TX_PA_KEEPALL_summary.tsv               (single-cell, fdr < 0.05)
transcript_MR_eQTLGen_FULL.csv                        (transcript,  transcript_fdr < 0.05)
<CpG MR summary>                                      (CpG,         fdr < 0.05)
glycan_trait_gene_map.rds                             (glycan trait→gene map)
MR_FDR_top1000_2026-06-12.tsv                         (MR ranking input)
ying_targets.rds / motrpac_EE_blood_genes.rds         (reference sets)
MTI_REBUILD_5LAYER.rds                                (multi-omic MR universe)
```

---

## Reproducing the results

### 1. MTI score
```bash
python compute_MTI.py \
    --in_node  MTI_REBUILD_5LAYER.tsv \
    --out_node MTI_REBUILD_5LAYER_recomputed.tsv
```
Recomputes `MTI_score`, `MTI_sumsq`, `MTI_n_layers` from the standardised betas
already in the node table. Writing to a distinct path keeps the committed table
intact for comparison.

### 2. Model
```bash
python multiseed.py      --node_path MTI_REBUILD_5LAYER.tsv --edge_path STRING_edges_REBUILD.tsv
python multiseed_mlp.py  --node_path MTI_REBUILD_5LAYER.tsv
python graph_ablation.py --node_path MTI_REBUILD_5LAYER.tsv --edge_path STRING_edges_REBUILD.tsv
```

### 3. Enrichment and convergence
```bash
Rscript enrichment_comparison.R
```

#### Background universes

Two universes, kept separate, because the two tests draw from different pools:

| Test | Universe | N | Reference sizes |
|---|---|---|---|
| GAT / MLP rank-based enrichment | Connected STRING graph | 2,473 | CausAge 30, MoTrPAC 831 |
| Model-free MR-anchored × CausAge convergence | Multi-omic MR universe | 2,959 | CausAge 33 |

The rank files contain only the 2,473 connected genes, so a gene without a
high-confidence STRING edge can never enter the top K. The hypergeometric null for
rank-based tests must therefore use the connected universe, with reference sets
re-intersected against it (30 of 33 CausAge genes and 831 of 948 MoTrPAC genes are
connected). The model-free convergence is not graph-based and correctly uses the
full MTI-scored universe.

> **Status.** The committed `enrichment_comparison.R` sets `N` once from the
> 2,959-gene MTI universe and applies it to every row, including the rank rows.
> The corrected rank-row figures (N = 2,473, M = 30 / 831) are tabulated below;
> the script is being updated to match.


#### The drawn MR-anchored set


Union of the per-layer FDR < 0.05 genes across all five molecular layers,
intersected with the multi-omic MR universe.

> **Pending re-analysis.** The committed script builds this union from four
> layers (protein, single-cell, bulk transcript, glycan) and appends `MEF2C` as
> a literal; the CpG layer is absent. The set is being rebuilt across all five
> layers without the literal. Because the drawn-set size *n* enters the
> expectation `E[X] = n x 33 / 2,959`, the convergence figures below will change
> and should be treated as provisional until the rebuild is complete.

#### Rank-based enrichment — corrected universe (N = 2,473)

Median across the five seeds {0, 1, 7, 42, 123}; range in brackets; **sig** counts
seeds reaching p < 0.05.

**CausAge, ageing-causal (M = 30)**

| Method | K = 100 | K = 150 | K = 200 |
|---|---|---|---|
| GAT | **0.031** [0.006–0.031] · 5/5 | **0.032** [0.008–0.032] · 5/5 | **0.030** [0.030–0.030] · 5/5 |
| MLP | **0.031** [0.031–0.031] · 5/5 | 0.105 · 0/5 | 0.221 · 0/5 |

**MoTrPAC, exercise-responsive (M = 831)**

| Method | K = 100 | K = 150 | K = 200 |
|---|---|---|---|
| GAT | **0.018** [0.010–0.070] · 4/5 | 0.037 [0.001–0.181] · 3/5 | 0.098 [0.002–0.250] · 2/5 |
| MLP | 0.199 [0.145–0.264] · 0/5 | 0.352 [0.289–0.419] · 0/5 | 0.358 [0.302–0.417] · 0/5 |

Observed counts per seed, in order {0, 1, 7, 42, 123}:

| Method | Reference | K = 100 | K = 150 | K = 200 |
|---|---|---|---|---|
| GAT | CausAge | 4, 5, 4, 4, 4 | 5, 6, 6, 5, 5 | 6, 6, 6, 6, 6 |
| MLP | CausAge | 4, 4, 4, 4, 4 | 4, 4, 4, 4, 4 | 4, 4, 4, 4, 4 |
| GAT | MoTrPAC | 42, 44, 45, 44, 41 | 56, 68, 61, 61, 59 | 76, 87, 81, 74, 72 |
| MLP | MoTrPAC | 39, 39, 37, 38, 38 | 53, 52, 53, 53, 54 | 69, 70, 71, 69, 70 |

Expected under the null: CausAge 1.21 / 1.82 / 2.43; MoTrPAC 33.60 / 50.40 / 67.21.

#### Degree-matched null

The hypergeometric null assumes every gene in the connected universe is equally
likely to enter the top K. If the ranking favoured well-connected genes, and the
reference sets were themselves degree-biased, that null could be rejected without
any biological signal. The degree-matched null holds connectivity fixed: genes are
binned by STRING degree, and each simulated top-K draws the same number from each
bin as the observed top-K contains.

```bash
python degree_matched_null.py \
    --edges STRING_edges_REBUILD.tsv --ranks . \
    --refs ying_ageing_causal_genes.tsv motrpac_exercise_genes.tsv \
    --nsim 10000 --bins 10 --seed 0
```

Median across the five seeds; 10,000 simulations; **sig** counts seeds at p < 0.05.

| Reference | Method | K | observed | hypergeom p | degree-matched p [range] | sig |
|---|---|---|---|---|---|---|
| CausAge | GAT | 100 | 4 | 0.031 | **0.039** [0.013–0.053] | 4/5 |
| CausAge | GAT | 150 | 5 | 0.032 | **0.034** [0.015–0.044] | 5/5 |
| CausAge | GAT | 200 | 6 | 0.030 | **0.036** [0.023–0.048] | 5/5 |
| CausAge | MLP | 100 | 4 | 0.031 | **0.031** [0.030–0.034] | 5/5 |
| CausAge | MLP | 150 | 4 | 0.105 | 0.102 [0.100–0.106] | 0/5 |
| CausAge | MLP | 200 | 4 | 0.221 | 0.235 [0.221–0.240] | 0/5 |
| MoTrPAC | GAT | 100 | 44 | 0.018 | **0.012** [0.005–0.041] | 5/5 |
| MoTrPAC | GAT | 150 | 61 | 0.037 | **0.030** [0.001–0.183] | 3/5 |
| MoTrPAC | GAT | 200 | 76 | 0.098 | 0.121 [0.002–0.289] | 2/5 |
| MoTrPAC | MLP | 100 | 38 | 0.199 | 0.178 [0.138–0.259] | 0/5 |
| MoTrPAC | MLP | 150 | 53 | 0.352 | 0.339 [0.281–0.419] | 0/5 |
| MoTrPAC | MLP | 200 | 70 | 0.358 | 0.315 [0.274–0.370] | 0/5 |

**Every conclusion is unchanged, and the exercise-responsive result strengthens**
— MoTrPAC at K = 100 moves from p = 0.018 to p = 0.012 and from four to five of
five seeds significant. CausAge at K = 100 loses one seed (0.053).

The reason the correction is small is that the ranking is not degree-biased:

| | median STRING degree |
|---|---|
| All 2,473 connected genes | 7 |
| GAT top 100 (seed 42) | 6 |
| CausAge reference (M = 30) | 5 |
| MoTrPAC reference (M = 831) | 8 |

The top-ranked genes are marginally *less* connected than average and the
ageing-causal reference is degree-poor, so the degree-matched expectation (1.36 at
K = 100) barely differs from the uniform one (1.21).

**What this does and does not establish.** It rules out connectivity as an
explanation for the enrichment — the most obvious confound in network-based gene
prioritisation. It does not establish that the *specific* biological wiring
matters: degree-matched sampling holds connectivity fixed while randomising which
genes are drawn, whereas degree-preserving rewiring holds connectivity fixed while
randomising the network itself and retraining. The latter remains outstanding.

Caveats: 10,000 simulations at one RNG seed, degree binned into deciles. The
CausAge arm rests on 4 observed genes against ~1.3 expected and is underpowered
under any null; the MoTrPAC arm carries the resolution.

#### As published (N = 2,959) — for comparison

| Method | CausAge K=100/150/200 | MoTrPAC K=100/150/200 |
|---|---|---|
| MR p-value rank | 0.307 / 0.505 / 0.665 | 0.971 / 0.994 / 0.985 |
| MTI (β) rank | 0.307 / 0.505 / 0.665 | 0.839 / 0.793 / 0.953 |
| MLP (no graph) | **0.024** / 0.083 / 0.180 | 0.118 / 0.212 / 0.243 |
| GAT (graph) | **0.024** / **0.024** / **0.021** | **0.007** / **0.014** / 0.071 |

All twelve GAT/MLP cells reproduce exactly from seed 42 at N = 2,959.

The MR p-value and MTI rank rows are shown at N = 2,959 only. Both are far from
significance there (p >= 0.31 and p >= 0.79), and the universe correction moves
p upward, so neither becomes significant under N = 2,473; they will be
recomputed for internal consistency when the corrected table is published.

> **On seed choice.** The published table reports seed 42, which is the **median
> or the most conservative** of the five seeds in every cell at K = 100 and
> K = 150 against CausAge it is the *least* significant seed in the set. The
> published figures are therefore not a favourable selection. The median-and-range
> tables above are given so this can be verified directly from the committed rank
> files.


#### What changes under the correction

- **The ageing-causal result is unaffected.** The GAT holds significance at all
  three ranking depths in 5/5 seeds; the MLP is significant only at K = 100 and
  loses it as the ranking widens. The depth-robustness contrast reported in §5.3.1
  survives intact.
- **The MoTrPAC result weakens but stands.** The GAT median moves from p = 0.007
  to p = 0.018 and from 5/5 to 4/5 seeds at K = 100, and depth-robustness is lost
  (3/5 at K = 150, 2/5 at K = 200). The MLP remains non-significant at every depth
  under both universes, so the qualitative claim that only the graph model recovers
  the exercise-responsive set is unchanged. Wording should become "recovered at
  K = 100 in four of five initialisations" rather than "robustly".

#### Ranking stability across seeds

| | top-100 Jaccard, median [range] | full-ranking Spearman, median [range] |
|---|---|---|
| GAT | 0.455 [0.361–0.562] | 0.869 [0.791–0.931] |
| MLP | 0.942 [0.905–0.980] | 0.999 [0.995–1.000] |

GAT and MLP share only 34–42 of their top 100 genes at the same seed.

The MLP is close to deterministic across random initialisations, which is expected
given that its target is a closed-form function of its inputs (see *Scope of the
model*). The GAT is not, because message passing makes each node's prediction
depend on its neighbourhood. The divergence between these two stability profiles
is the graph's operative contribution.

### 4. CTSF validation
```bash
Rscript mr_CTSF_ageing_outcomes.R     # → β = 0.33 protein, β = 0.19 expression
Rscript coloc_longevity_CTSF_pQTL.R   # → conditional PP.H4 = 0.78
Rscript coloc_longevity_CTSF_eQTL.R   # → conditional PP.H4 = 0.62
```

A gene–outcome pair is **validated** only if it satisfies all three criteria:
FDR-significant MR, Steiger-consistent direction, and conditional
PP.H4 = PP.H4 / (PP.H3 + PP.H4) > 0.7. CTSF vs exceptional longevity is the single
pair meeting all three.

---

## Scope of the model

Stated explicitly, because the committed node table makes it directly checkable.

The GAT's regression target is the Euclidean norm of the same five standardised MR
betas that constitute its node features, and the classification target counts their
non-missing entries. The MTI-derived columns are excluded from the feature matrix
by a leakage guard, but the components that determine them are retained, so **the
target is a deterministic function of the node features** and the regression head
is not a prediction task in the usual sense.

The model's operative role is **propagation**: message passing prevents each node
from reading off its own composite score, so the learned ranking is a
neighbourhood-smoothed MTI. The framework is best understood as network propagation
with learned edge weights, performing denoising rather than causal discovery: the
supervision target derives from the upstream MR, and the model cannot generate
causal information the MR did not contain. Attention weights are treated as
context-dependent indicators of relative importance, not evidence of mechanism.

### Analyses not yet run

- **A model-free hybrid score**, `z(MTI) + α·z(1[n_layers ≥ 2])`, computed directly
  from the node table. This is the correct null for the MLP row.
- **A degree-preserving edge-rewiring null.** Degree-matched sampling (above)
  shows the enrichment is not explained by node connectivity; rewiring goes
  further by shuffling the edges themselves while holding every node's degree
  fixed, retraining, and asking whether the specific biological wiring matters.
- **Diffusion / random-walk-with-restart on STRING seeded with MTI**, a more natural
  comparator for a propagation method than a non-graph MLP.
- **A STRING graph restricted to the experimental and co-expression channels.** The
  combined score at ≥700 aggregates curated-database and text-mining evidence, so
  pathway co-membership and literature co-occurrence can enter through the edges
  even though annotations were removed from the nodes.


---

## Data sources

All summary statistics are publicly available; this repository contains code and
gene lists only, not redistributed GWAS data.

**Molecular layers:** UKB-PPP proteomics (Sun et al. 2023,
doi:10.1038/s41586-023-06592-6); GoDMC methylation QTL; TPNG plasma N-glycome;
eQTLGen cis-eQTL (Võsa et al. 2021, doi:10.1038/s41588-021-00913-z); OneK1K
single-cell cis-eQTL (Yazar et al. 2022). STRING v12.0 (Szklarczyk et al. 2023).

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

