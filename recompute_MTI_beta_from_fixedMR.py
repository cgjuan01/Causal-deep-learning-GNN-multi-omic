#!/usr/bin/env python3
"""
Multi-omic Trait Importance (MTI) score computation.

Implements the MTI formulation used in:
  Juan & Ntasis (2026). Multi-omic deep learning identifies exercise-responsive
  ageing pathways in humans. medRxiv. https://doi.org/10.64898/2025.12.26.25343061

Formula:
  MTI(g) = sqrt( sum_{k in K_g} (beta_{g,k} / SD_k)^2 )

Where:
  beta_{g,k}  = MR effect size for gene g in omic layer k
  SD_k        = standard deviation of beta across all genes in layer k
  K_g         = set of layers with available MR evidence for gene g

Normalisation note:
  Each layer's MR effect sizes are divided by their standard deviation,
  placing all four omic layers on a common scale before aggregation.
  The mean is not subtracted — this is SD normalisation rather than
  full z-scoring. The two are equivalent when the layer mean is zero,
  which is approximately true for MR betas distributed around the null.

The aggregation is the Euclidean norm (L2 norm) of the normalised effect
size vector across layers. This is always non-negative and rewards genes
with strong effects in any direction across multiple layers.

Three MTI variants are stored:
  MTI_score    PRIMARY  -- Euclidean norm of SD-normalised betas
  MTI_sumsq    DIAGNOSTIC -- sum of squared SD-normalised betas (before sqrt)
  MTI_n_layers DIAGNOSTIC -- number of layers with MR evidence for this gene

Inputs:
  Node table TSV containing per-gene MR results across four omic layers,
  plus structural and functional annotation features. Must contain:
    gene_symbol          (str)
    protein_MR_beta      (float, or protein_MR_beta_fixed as fallback)
    cpg_MR_beta          (float)
    glycan_MR_beta       (float)
    sc_MR_beta           (float)

Outputs:
  <OUT_NODE>  -- same node table with MTI_score, MTI_sumsq, MTI_n_layers
                 updated in place, plus per-layer *_beta_std_recalc columns

Usage:
  Set IN_NODE and OUT_NODE paths below, then:
    python recompute_MTI_beta_from_fixedMR.py
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — update as needed
# ---------------------------------------------------------------------------

BASE     = "/Users/ciara/Downloads/LDaware_MR_proteins"
IN_NODE  = f"{BASE}/GNN_nodes_EXERCISEONLY_MR_SC_STRUCT_PANTHER_INTERPRO_UNIPROT_AFv6_PROTEINMRfixed.tsv"
OUT_NODE = f"{BASE}/GNN_nodes_EXERCISEONLY_MR_SC_STRUCT_PANTHER_INTERPRO_UNIPROT_AFv6_PROTEINMRfixed_MTIbetaRecalc.tsv"

# ---------------------------------------------------------------------------
# Load node table
# ---------------------------------------------------------------------------

df = pd.read_csv(IN_NODE, sep="\t")

if "gene_symbol" not in df.columns:
    raise ValueError("Node table must contain 'gene_symbol' column.")

print(f"Loaded: {len(df)} genes x {len(df.columns)} columns")

# ---------------------------------------------------------------------------
# Step 1 — Promote *_fixed columns into canonical protein MR columns
#
# The protein MR layer was recomputed with LD-aware overlap correction
# after the initial node table was assembled. The corrected values are
# stored in protein_MR_*_fixed columns. This step fills any missing
# values in the canonical columns with the corrected estimates.
# ---------------------------------------------------------------------------

for col in ["beta", "se", "p", "nsnp"]:
    fixed = f"protein_MR_{col}_fixed"
    base  = f"protein_MR_{col}"
    if fixed in df.columns:
        if base not in df.columns:
            df[base] = np.nan
        df[base] = df[fixed].combine_first(df[base])
        print(f"  Promoted {fixed} -> {base}")

# ---------------------------------------------------------------------------
# Step 2 — Define the four omic layers
# ---------------------------------------------------------------------------

layers = {
    "protein": "protein_MR_beta",   # UKBB plasma proteomics,       n~54,000
    "cpg":     "cpg_MR_beta",       # GoDMC whole-blood CpG,        n=27,750
    "glycan":  "glycan_MR_beta",    # TPNG GWAMA plasma glycans,    n=10,172
    "sc":      "sc_MR_beta",        # OneK1K PBMCs scRNA,           n=982
}

# ---------------------------------------------------------------------------
# Step 3 — SD-normalise betas within each layer
#
# Divides each gene's MR effect size by the standard deviation of that
# layer's betas across all genes. This puts all four layers on a common
# scale regardless of their different units, sample sizes, and variance.
# Genes with no MR evidence in a layer (NaN beta) receive NaN and are
# excluded from the MTI summation for that gene.
# ---------------------------------------------------------------------------

beta_std_cols = []

for layer, bcol in layers.items():
    outcol = f"{layer}_beta_std_recalc"
    beta_std_cols.append(outcol)

    if bcol not in df.columns:
        print(f"  [{layer}] column '{bcol}' not found — setting to NaN")
        df[outcol] = np.nan
        continue

    x  = pd.to_numeric(df[bcol], errors="coerce").to_numpy(dtype=float)
    sd = np.nanstd(x)

    if not np.isfinite(sd) or sd == 0:
        print(f"  [{layer}] SD is zero or non-finite — setting to NaN")
        df[outcol] = np.nan
    else:
        df[outcol] = x / sd
        n_finite   = np.isfinite(x).sum()
        print(f"  [{layer}] SD={sd:.4f}  n_finite={n_finite}  "
              f"normalised range=[{(x/sd)[np.isfinite(x)].min():.3f}, "
              f"{(x/sd)[np.isfinite(x)].max():.3f}]")

# ---------------------------------------------------------------------------
# Step 4 — Compute MTI components
#
# MTI is the Euclidean norm (L2 norm) of the SD-normalised beta vector.
# Squaring before summing means sign is not preserved — genes with strong
# effects in any direction (positive or negative) receive high scores.
# This is appropriate for gene prioritisation where direction is secondary
# to the magnitude of causal association with vigorous physical activity.
#
# np.nansum treats NaN as zero, so layers without MR evidence for a gene
# contribute nothing to that gene's MTI score. MTI_n_layers records how
# many layers actually contributed.
# ---------------------------------------------------------------------------

M       = df[beta_std_cols].to_numpy(dtype=float)
present = np.isfinite(M)

df["MTI_n_layers_recalc"] = present.sum(axis=1).astype(int)
df["MTI_sumsq_recalc"]    = np.nansum(M ** 2, axis=1)
df["MTI_score_recalc"]    = np.sqrt(df["MTI_sumsq_recalc"].to_numpy(dtype=float))

# ---------------------------------------------------------------------------
# Step 5 — Overwrite canonical MTI columns
#
# Replaces the MTI columns in the node table with the recomputed values
# so downstream scripts (gnn_trainer.py) use the updated scores without
# any changes to their input handling.
# Comment out these three lines to preserve the original MTI alongside
# the recalculated values for comparison.
# ---------------------------------------------------------------------------

df["MTI_n_layers"] = df["MTI_n_layers_recalc"]
df["MTI_sumsq"]    = df["MTI_sumsq_recalc"]
df["MTI_score"]    = df["MTI_score_recalc"]

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

df.to_csv(OUT_NODE, sep="\t", index=False)
print(f"\nWrote: {OUT_NODE}")
print(f"  Genes: {len(df)}")
print(f"  MTI_score range: [{df['MTI_score'].min():.4f}, {df['MTI_score'].max():.4f}]")
print(f"  MTI_n_layers distribution:")
print(df["MTI_n_layers"].value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# Sanity check — key genes from paper
# ---------------------------------------------------------------------------

check     = ["SIRT1", "SIRT2", "B4GALT1", "FUT8", "ST6GAL1", "MGAT3", "SIRT6"]
cols_show = ["gene_symbol", "protein_MR_beta", "protein_MR_p",
             "MTI_n_layers", "MTI_score"]

print("\nSanity check (key paper genes):")
print(df.loc[df["gene_symbol"].isin(check), cols_show].to_string(index=False))
