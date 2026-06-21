#!/usr/bin/env python3
"""
compute_MTI.py — Multi-omic Trait Importance (MTI) score computation.

Implements the MTI formulation used in:
  Juan & Ntasis (2026). Causally-anchored multi-omic deep learning recovers
  exercise-responsive and ageing-causal genes from human physical activity.

Formula:
  MTI(g) = sqrt( sum_{k in K_g} (beta_{g,k} / SD_k)^2 )

Where:
  beta_{g,k}  = MR effect size for gene g in omic layer k
  SD_k        = standard deviation of beta across all genes in layer k
  K_g         = set of layers with available MR evidence for gene g

The aggregation is the Euclidean (L2) norm of the per-layer SD-normalised
effect-size vector. It is non-negative and rewards genes with strong causal
association in any direction across multiple layers. The MTI score is the
supervised regression target for the graph model; it is NOT used as a node
feature (a leakage guard excludes all MTI-derived columns from the features).

Normalisation note:
  Each layer's MR effect sizes are divided by their standard deviation,
  placing all five omic layers on a common scale before aggregation. The mean
  is not subtracted — this is SD normalisation rather than full z-scoring; the
  two are equivalent when the layer mean is ~0, which holds for MR betas
  distributed around the null.

Five molecular layers:
  protein     UKB-PPP plasma proteomics
  cpg         GoDMC whole-blood methylation
  glycan      TPNG plasma N-glycome
  sc          OneK1K PBMC single-cell cis-eQTL
  transcript  eQTLGen whole-blood bulk cis-eQTL

Inputs:
  A node table (TSV) containing the per-gene, per-layer standardised MR betas:
    gene_symbol, protein_std, cpg_std, glycan_std, sc_std, transcript_std
  This script reads ONLY these columns. No structural or functional annotation
  (AlphaFold, PANTHER, InterPro, UniProt) is read or used — the final model
  rests on multi-omic causal evidence and network topology alone.

Outputs:
  A node table with MTI_score, MTI_sumsq, MTI_n_layers (re)computed in place.

Usage:
  python compute_MTI.py --in_node MTI_REBUILD_5LAYER.tsv --out_node MTI_REBUILD_5LAYER.tsv
"""

import argparse
import numpy as np
import pandas as pd

# Per-layer standardised-beta columns (the only inputs this script reads)
LAYER_COLS = {
    "protein":    "protein_std",     # UKB-PPP plasma proteomics
    "cpg":        "cpg_std",         # GoDMC whole-blood methylation
    "glycan":     "glycan_std",      # TPNG plasma N-glycome
    "sc":         "sc_std",          # OneK1K PBMC single-cell cis-eQTL
    "transcript": "transcript_std",  # eQTLGen bulk whole-blood cis-eQTL
}


def compute_mti(df):
    """Compute MTI_score, MTI_sumsq, MTI_n_layers from the per-layer std betas.

    The input columns are already SD-normalised per layer. If a raw (un-
    normalised) beta column is supplied instead, set RENORMALISE=True below.
    """
    cols = []
    for layer, col in LAYER_COLS.items():
        if col not in df.columns:
            print(f"  [{layer}] column '{col}' not found — treated as absent")
            continue
        cols.append(col)

    if not cols:
        raise ValueError(
            "None of the expected per-layer columns were found: "
            + ", ".join(LAYER_COLS.values())
        )

    M       = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    present = np.isfinite(M)

    n_layers = present.sum(axis=1).astype(int)
    sumsq    = np.nansum(M ** 2, axis=1)   # NaN layers contribute 0
    score    = np.sqrt(sumsq)

    df["MTI_n_layers"] = n_layers
    df["MTI_sumsq"]    = sumsq
    df["MTI_score"]    = score
    return df, cols


def main():
    ap = argparse.ArgumentParser(description="Compute the multi-omic trait-importance (MTI) score.")
    ap.add_argument("--in_node",  required=True, help="input node table (TSV)")
    ap.add_argument("--out_node", required=True, help="output node table (TSV)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_node, sep="\t")
    if "gene_symbol" not in df.columns:
        raise ValueError("Node table must contain a 'gene_symbol' column.")
    print(f"Loaded: {len(df)} genes x {len(df.columns)} columns")

    df, used = compute_mti(df)
    print(f"Layers used: {', '.join(used)}")

    df.to_csv(args.out_node, sep="\t", index=False)
    print(f"\nWrote: {args.out_node}")
    print(f"  Genes: {len(df)}")
    print(f"  MTI_score range: [{df['MTI_score'].min():.4f}, {df['MTI_score'].max():.4f}]")
    print("  MTI_n_layers distribution:")
    print(df["MTI_n_layers"].value_counts().sort_index().to_string())

    # Sanity check — eight triple-convergent genes
    check = ["CTSD", "CTSF", "FADS1", "FADS2", "HEXIM1", "IGFBP7", "LTBP3", "RHOC"]
    show  = [c for c in ["gene_symbol", "MTI_n_layers", "MTI_score"] if c in df.columns]
    sub   = df.loc[df["gene_symbol"].isin(check), show]
    if len(sub):
        print("\nSanity check (eight convergent genes):")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
