# Supervised MR-GAT-GNN Trainer (Exercise-only | MTI + Multi-omic Hybrid | AFv6)

Supervised Graph Attention Network (GAT) training script for **exercise-only** gene prioritisation using an **Mendelian randomisation-derived node table** (MTI scores + precomputed features) and a **kNN gene–gene edge list**. The model predicts:

1) **Continuous MTI score** (regression; MSE)  
2) **Multi-layer support label** (classification; `MTI_n_layers >= 2`, optional; BCE)

After training, it produces both:
- **MTI-only ranking** (by predicted MTI)
- **Hybrid ranking** (z(predicted MTI) + `alpha`·z(multi_prob))

It also generates an **MR-nominal subset** defined as *any* detected MR p-value column < 0.05.

---

## What this script does

### Inputs
- **Node TSV** containing:
  - `gene_symbol` (required)
  - `MTI_score` (required; regression target)
  - optionally `MTI_n_layers` (enables multi-layer classification loss)
  - numeric feature columns including (already computed upstream):
    - **AlphaFold v6** features (expects `af_*` columns)
    - **InterPro / UniProt** PCs or other numeric descriptors
    - any additional MR-derived / omics-derived features you’ve stored in the node table

- **Edge TSV** containing gene–gene edges (kNN graph) with columns:
  - `from`, `to`  
  or compatible aliases: `source/target`, `gene1/gene2` (auto-renamed)

### Training
- Reproducible seed (42) for Python / NumPy / Torch (+ CUDA if available)
- **GAT encoder**: 2-layer GATConv → ELU → dropout
- **Multi-task heads**:
  - regression head → predicted MTI
  - classification head → multi-layer support logit

Loss:
- `MSE(mti_pred, MTI_score)` on finite MTI rows
- `+ lambda_multi * BCEWithLogits(multi_logit, MTI_n_layers>=2)` (if `MTI_n_layers` exists and has positives)

Class imbalance handling:
- Computes `pos_weight = min(n_neg/n_pos, 50)` for BCE

### Ranking outputs
- **MTI-only ranks**: descending predicted MTI
- **Hybrid score**:
  - `hybrid = z(mti_pred) + alpha * z(sigmoid(multi_logit))`
- **MR-nominal flag**:
  - detects MR p-value columns with heuristics (e.g., `*_MR_p`, `*_p` with “mr/protein/cpg/glycan/sc_” in name)
  - `MR_nominal_any = any(p < 0.05)` across detected columns



