# Causal Multi-omic Deep Learning for Exercise Genomics
### Supervised MR-GAT | Multi-omic Trait Importance | Exercise-responsive Ageing Pathways

Graph neural network framework integrating causal inference across four molecular
layers with supervised graph attention learning to prioritise exercise-responsive
genes and map their intersection with human biological ageing.

**Paper:**
Juan CG, Ntasis L. *Multi-omic deep learning identifies exercise-responsive ageing
pathways in humans.* medRxiv. 2026.
https://doi.org/10.64898/2025.12.26.25343061

---

## What this repository does

Translates sparse polygenic GWAS signals for vigorous physical activity (VPA) into
coherent pathway-level mechanisms by combining:

- Linkage disequilibrium-aware Mendelian randomisation across four omic layers
  to establish causal gene associations with habitual VPA
- Multi-omic Trait Importance (MTI) scoring to aggregate causal evidence into
  a continuous gene-level signal
- Supervised Graph Attention Network (MR-GAT) to learn graph-informed,
  topology-aware gene representations that improve on raw MTI rankings
- Hybrid gene ranking combining predicted MTI and multi-layer support probability

---

## Pipeline

```
GWAS (UKBB whole-exome sequencing, n~75,000)
        ↓  LD-based clumping (PLINK2)
  87 VPA-associated SNPs
        ↓  LD-aware overlap-aware IVW MR
  ┌─────────────────────────────────────────┐
  │  proteomics   2,304 genes  (UKBB)       │
  │  epigenomics     19 genes  (GoDMC)      │
  │  glycomics       31 genes  (TPNG)       │
  │  scRNA           35 genes  (OneK1K)     │
  └─────────────────────────────────────────┘
        ↓
  recompute_MTI_beta_from_fixedMR.py
        ↓
  MTI scores  (2,370 unique genes)
        ↓
  gnn_trainer.py
        ↓
  GNN hybrid gene rankings
        ↓
  Ageing clock annotation + pathway analysis
```

---

## Repository structure

```
├── README.md                              this file
├── recompute_MTI_beta_from_fixedMR.py     MTI score computation
└── gnn_trainer.py                         Supervised MR-GAT training
```

---

## Script 1 — `recompute_MTI_beta_from_fixedMR.py`

### What it does

Computes Multi-omic Trait Importance (MTI) scores from MR effect sizes across
four omic layers. Produces the node table used as input to `gnn_trainer.py`.

### Formula

The paper describes MTI as a weighted sum of standardised MR effect sizes.
The implementation computes the Euclidean norm (L2 norm) of SD-normalised
effect sizes across layers:

```
MTI(g) = sqrt( sum_{k in K_g} ( beta_{g,k} / SD_k )^2 )
```

| Symbol | Meaning |
|---|---|
| g | gene |
| k | omic layer index |
| K_g | layers with available MR evidence for gene g |
| beta_{g,k} | MR IVW effect size for gene g in layer k |
| SD_k | standard deviation of beta across all genes in layer k |

All layers are weighted equally. The paper's methods section has been updated
to reflect this implementation.

### Normalisation and aggregation

Each layer's MR betas are divided by their within-layer standard deviation
(population SD, ddof=0), placing all four layers on a common scale regardless
of their different units, sample sizes, and variance. The mean is not
subtracted — SD normalisation is equivalent to z-scoring when the layer mean
is zero, which is approximately true for MR betas distributed around the null.

The Euclidean norm is always non-negative — genes with strong effects in any
direction receive high scores. This is appropriate for gene prioritisation
where direction is secondary to the magnitude of causal association with VPA.

### Steps

**Step 1 — Promote `_fixed` columns.**
The protein MR layer was recomputed with LD-aware overlap correction after
initial node table assembly. Corrected values in `protein_MR_*_fixed` are
promoted into canonical `protein_MR_*` columns via `combine_first` (fills
missing values only, does not overwrite existing ones).

**Step 2 — Define four omic layers.**

| Layer | Column | Dataset | n |
|---|---|---|---|
| protein | `protein_MR_beta` | UKBB plasma proteomics | ~54,000 |
| cpg | `cpg_MR_beta` | GoDMC whole-blood CpG methylation | 27,750 |
| glycan | `glycan_MR_beta` | TPNG GWAMA plasma glycan peaks | 10,172 |
| sc | `sc_MR_beta` | OneK1K PBMCs scRNA | 982 |

**Step 3 — SD-normalise betas within each layer.**
Genes absent from a layer (NaN beta) receive NaN and are excluded from the
MTI summation for that gene.

**Step 4 — Compute MTI.**
`np.nansum` sums squared SD-normalised betas treating NaN as zero. Square root
produces the final score. `MTI_n_layers` counts layers with finite normalised
betas per gene.

**Step 5 — Overwrite canonical MTI columns.**
Updates `MTI_score`, `MTI_sumsq`, `MTI_n_layers` in place so `gnn_trainer.py`
uses updated scores without input changes. Comment out these three lines to
preserve both old and new values for comparison.

### Input

Node table TSV with required columns:

| Column | Description |
|---|---|
| `gene_symbol` | HGNC gene symbol |
| `protein_MR_beta` | Protein MR IVW effect size (or `protein_MR_beta_fixed`) |
| `cpg_MR_beta` | CpG methylation MR effect size |
| `glycan_MR_beta` | Glycan MR effect size |
| `sc_MR_beta` | scRNA MR effect size |

All other columns are passed through unchanged.

### Output

| Column | Description |
|---|---|
| `MTI_score` | Primary MTI — Euclidean norm of SD-normalised betas |
| `MTI_sumsq` | Sum of squared SD-normalised betas (diagnostic) |
| `MTI_n_layers` | Number of layers with MR evidence for this gene |
| `protein_beta_std_recalc` | SD-normalised beta, proteomics |
| `cpg_beta_std_recalc` | SD-normalised beta, epigenomics |
| `glycan_beta_std_recalc` | SD-normalised beta, glycomics |
| `sc_beta_std_recalc` | SD-normalised beta, scRNA |

Expected output numbers from the node table used for analysis:

| Metric | Value |
|---|---|
| Total genes | 3,089 |
| MTI_n_layers = 0 | 19 |
| MTI_n_layers = 1 | 3,046 |
| MTI_n_layers = 2 | 22 |
| MTI_n_layers = 3 | 2 |

### Usage

Update `IN_NODE` and `OUT_NODE` paths at the top of the script, then:

```bash
python recompute_MTI_beta_from_fixedMR.py
```

No command-line arguments — paths are set directly in the script.

---

## Script 2 — `gnn_trainer.py`

### What it does

Trains a two-layer Graph Attention Network with a shared encoder and two
task-specific heads: regression (MTI prediction) and classification
(multi-layer support). Produces hybrid gene rankings combining both signals.

### Model architecture

```
Input features (numeric node table columns, z-scored)
        ↓
GATConv(in_dim → hidden_dim × heads)   [4 attention heads]
        ↓  ELU → Dropout(0.2)
GATConv(hidden_dim × heads → out_dim)  [1 head]
        ↓
  ┌─────┴──────┐
  │            │
Linear       Linear
  │            │
MTI pred    multi_logit
(regression) (classification)
```

### Loss function

```
L = MSE(mti_pred, MTI_score)                         [on finite MTI rows only]
  + lambda_multi × BCEWithLogits(multi_logit, MTI_n_layers >= 2)
```

Class imbalance: `pos_weight = min(n_neg / n_pos, 50)` applied to BCE term.

### Ranking outputs

MTI-only ranking — genes sorted by predicted MTI score descending.

Hybrid ranking:
```
hybrid_score = z(mti_pred) + alpha × z(sigmoid(multi_logit))
```
where `z(·)` denotes z-scoring across all genes.

MR-nominal flag — detects MR p-value columns by name heuristic (containing
`mr`, `protein`, `cpg`, `glycan`, or `sc_`, ending in `_p` or `_mr_p`).
`MR_nominal_any = True` if any such column has p < 0.05 for that gene.

### Design decisions

| Decision | Choice | Alternative rejected | Reason |
|---|---|---|---|
| Graph operator | GAT | GCN | Attention weights allow each gene to learn which neighbours are most relevant rather than aggregating all kNN neighbours equally |
| Encoder depth | 2 layers | 1 or 3+ | 2 layers captures 2-hop neighbourhoods; deeper layers risk over-smoothing on small graphs |
| Dual-head loss | MSE + BCE | MSE only | BCE head provides a separate gradient signal rewarding genes with consistent cross-layer causal evidence |
| pos_weight cap | min(n_neg/n_pos, 50) | Uncapped | Raw ratio ~146 with 21 positives in 3,089 genes caused training instability |
| Graph topology | Precomputed kNN (k=15) | STRING edges | kNN from the node feature space keeps topology consistent with what the model learns from |
| Missing features | NaN → 0 after z-scoring | Mean imputation | Equivalent to mean imputation for z-scored data; avoids information leakage |

### Circularity testing

Because the regression target (MTI) is derived from MR outputs, the model
was retrained with all MR-derived node features removed (MR-ablated model).
Global rank concordance between full and MR-ablated models was Spearman
ρ = 0.875, with 57–61% Jaccard overlap at top K = 100–1000, demonstrating
the model integrates orthogonal structural, functional, and topological
information beyond MR signal.

### Limitations

- MTI is derived from MR outputs — the GNN is a graph-informed denoising and
  representation learning model, not an independent causal discovery engine
- Full-graph training (~3,089 nodes per step) would require mini-batch
  neighbourhood sampling at 10–100× scale
- kNN graph topology is fixed during training
- Experimental validation cohort (n=13) limits statistical power for the
  glycomic and gene expression validation

### Inputs

Node TSV — must contain:

| Column | Required | Description |
|---|---|---|
| `gene_symbol` | yes | Gene identifier |
| `MTI_score` | yes | Regression target from `recompute_MTI_beta_from_fixedMR.py` |
| `MTI_n_layers` | optional | Enables multi-layer classification loss |
| `MTI_sumsq` | optional | Excluded from features automatically |
| `af_*` | optional | AlphaFold v6 structural descriptors |
| InterPro / UniProt PCs | optional | Any additional numeric descriptors |
| MR-derived columns | optional | MR effect sizes and p-values |

Legacy AlphaFold columns (`pLDDT*`, `n_atoms`) are automatically dropped
in favour of `af_*` columns.

Edge TSV — gene–gene kNN edges, column names auto-detected:

| Accepted aliases | Mapped to |
|---|---|
| `from`, `source`, `gene1` | source node |
| `to`, `target`, `gene2` | target node |

### Output files

| File | Description |
|---|---|
| `GNN_embeddings_EXERCISEONLY_GAT_SUP_MTI_MULTI_AFv6_HYBRID_v2.tsv` | Full per-gene predictions and ranks |
| `GNN_ranking_EXERCISEONLY_FULL_deduped_v2.tsv` | Deduplicated full ranking |
| `GNN_ranking_EXERCISEONLY_MR_nominal_deduped_v2.tsv` | Deduplicated MR-nominal subset |
| `GNN_top{N}_EXERCISEONLY_FULL_v2.tsv` | Top-N from full ranking |
| `GNN_top{N}_EXERCISEONLY_MRnominal_v2.tsv` | Top-N from MR-nominal ranking |

### Output columns

| Column | Description |
|---|---|
| `gene_symbol` | Gene identifier |
| `MTI_score` | Input MTI score |
| `GNN_EXERCISE_SUP_MTI_MULTI_pred` | Predicted MTI (regression head) |
| `GNN_EXERCISE_SUP_MTI_MULTI_multi_prob` | Multi-layer support probability |
| `GNN_EXERCISE_SUP_MTI_MULTI_rank_MTIonly` | Rank by predicted MTI (1 = highest) |
| `GNN_EXERCISE_SUP_MTI_MULTI_rank_MTIonly_scaled` | MTI-only rank scaled to [0, 1] |
| `GNN_EXERCISE_SUP_MTI_MULTI_hybrid_score` | Hybrid z-score |
| `GNN_EXERCISE_SUP_MTI_MULTI_rank` | Final hybrid rank (1 = highest) |
| `GNN_EXERCISE_SUP_MTI_MULTI_rank_scaled` | Hybrid rank scaled to [0, 1] |
| `MR_nominal_any` | True if any MR p-value column < 0.05 |

### Usage

```bash
python gnn_trainer.py \
    --node_path path/to/node_table.tsv \
    --edge_path path/to/edges.tsv \
    --out_dir   path/to/output_dir \
    --epochs    300 \
    --lr        1e-3 \
    --lambda_multi 0.3 \
    --alpha     1.0 \
    --top_n     300
```

| Argument | Default | Description |
|---|---|---|
| `--node_path` | required | Node TSV path |
| `--edge_path` | required | Edge TSV path |
| `--out_dir` | same dir as node_path | Output directory |
| `--epochs` | 300 | Training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--lambda_multi` | 0.3 | Weight for multi-layer BCE loss |
| `--alpha` | 1.0 | Weight for multi-omic support in hybrid score |
| `--top_n` | 300 | Genes in Top-N output files |

---

## Scalability

Current scale: ~3,089 nodes, ~46,000 kNN edges, full-graph training per step.
Scaling to 100× would require:

- Mini-batch neighbourhood sampling (GraphSAINT, ClusterGCN) rather than
  full-graph forward passes
- Approximate kNN construction (FAISS) above ~100,000 genes
- Distributed MR computation across omic layers (embarrassingly parallel)

The masked MTI target handling and dual-head architecture are scale-agnostic.

---

## Reproducibility

Fixed seed (42) applied to Python, NumPy, PyTorch, and CUDA. All outputs
are fully deterministic given the same input node and edge tables.

Input data access is subject to the conditions of the source datasets:
UK Biobank, GoDMC, TPNG GWAMA, and OneK1K.

---

## Dependencies

```
python >= 3.9
torch >= 2.0
torch-geometric
numpy
pandas
```

---

## Citation

```
Juan CG, Ntasis L. Multi-omic deep learning identifies exercise-responsive
ageing pathways in humans. medRxiv. 2026.
doi: 10.64898/2025.12.26.25343061
```
