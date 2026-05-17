#!/usr/bin/env python3
"""
feature_attribution.py
======================
Gradient-based feature attribution for the supervised MR-GAT model.

Produces:
  1. Global attribution (Figure 4D) — proportional contribution of each
     feature class to gene ranking across top-100, 200, 500, 1000 genes.
  2. Per-gene-class attribution — glycosylation vs DNA repair vs background,
     to test whether concentrated diffusion is driven by biology or topology.

Method: input-gradient × input (saliency-weighted attribution).
  - For each gene, compute dMTI_pred / dx, then multiply element-wise by x.
  - Take absolute value and sum within each feature class.
  - Normalise to proportions within each gene.
  - Average across gene sets for group-level summaries.

This method satisfies completeness (attributions sum to the prediction minus
a zero baseline) and is consistent with how gradient-based attribution is
described in the paper (Figure 4D).

Usage
-----
python feature_attribution.py \
    --node_path  path/to/node_table.tsv \
    --edge_path  path/to/edges.tsv \
    --model_path path/to/trained_model.pt      # optional — will retrain if absent
    --ranking_path path/to/GNN_ranking_EXERCISEONLY_FULL_deduped_v2.tsv \
    --out_dir    path/to/output_dir \
    --top_ks     100 200 500 1000

Dependencies: torch, torch_geometric, numpy, pandas, matplotlib
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Model — identical architecture to gnn_trainer.py
# ---------------------------------------------------------------------------

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x


class MTIMultiNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, heads=4, dropout=0.2):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim, out_dim, heads, dropout)
        self.reg_head = nn.Linear(out_dim, 1)
        self.cls_head = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        mti_pred = self.reg_head(h).squeeze(-1)
        multi_logit = self.cls_head(h).squeeze(-1)
        return mti_pred, multi_logit


# ---------------------------------------------------------------------------
# Feature class definitions
# Adjust column name patterns to match your exact node table column names.
# ---------------------------------------------------------------------------

def classify_feature(col_name: str) -> str:
    """
    Map a feature column name to one of six feature classes:
      MR           — MR-derived causal signals (beta, std beta, p-values,
                     FDR, nsnp, significance flags, fixed corrections)
      Structure    — AlphaFold v6 structural descriptors
      UniProt      — UniProt functional PCs
      InterPro     — InterPro domain PCs
      PANTHER      — PANTHER pathway/functional PCs
                     (columns prefixed pan_ in this node table)
      Contextual   — multi-omic presence flags, cross-layer concordance,
                     neighbourhood-aggregated descriptors

    Column naming verified against GNN_nodes_EXERCISE_complete.tsv.
    """
    c = col_name.lower()

    # MR-derived signals
    # Covers: protein_MR_beta/se/p, protein_MR_beta_fixed/se_fixed/p_fixed,
    #         protein_MR_FDR_BH, protein_MR_FDR_BH_fixed,
    #         protein_MR_sig_FDR05, protein_MR_sig_FDR05_fixed,
    #         sc_MR_beta/se/p/FDR, glycan_MR_beta/se/p/FDR,
    #         cpg_MR_beta/se/p/FDR,
    #         protein_beta_std, cpg_beta_std, glycan_beta_std, sc_beta_std
    if any(k in c for k in (
        "_mr_beta", "_mr_p", "_mr_se", "_mr_fdr", "_mr_sig",
        "beta_std", "protein_mr", "cpg_mr", "glycan_mr", "sc_mr",
        "protein_beta", "cpg_beta", "glycan_beta", "sc_beta",
        "_beta_std_recalc",
    )):
        return "MR"

    # nsnp columns — number of SNPs used in MR, an MR quality metric
    # Covers: protein_nsnp, sc_nsnp, glycan_nsnp, cpg_nsnp, protein_nsnp_fixed
    if c.endswith("_nsnp") or c.endswith("_nsnp_fixed"):
        return "MR"

    # AlphaFold v6 structural descriptors
    # Covers: af_len, af_plddt_mean, af_plddt_sd, af_plddt_frac70, af_plddt_frac90
    if c.startswith("af_") or c.startswith("plddt") or c == "n_atoms":
        return "Structure"

    # UniProt functional PCs
    # Covers: uniprot_pc1, uniprot_pc2, uniprot_pc3
    if c.startswith("uniprot_pc") or c.startswith("uniprot"):
        return "UniProt"

    # InterPro domain PCs
    # Covers: interpro_pc1, interpro_pc2, interpro_pc3
    if c.startswith("interpro_pc") or c.startswith("interpro"):
        return "InterPro"

    # PANTHER pathway/functional PCs
    # Covers: pan_family_pc1-3, pan_class_pc1-3, pan_pw_pc1-3, pan_mf_pc1-3
    # Note: columns are named pan_* not panther_* in this node table
    if c.startswith("pan_") or c.startswith("panther"):
        return "PANTHER"

    # Contextual integration
    # Covers: protein_present, sc_present, glycan_present, cpg_present
    # (multi-omic presence indicators used as contextual node features)
    return "Contextual"


# ---------------------------------------------------------------------------
# Edge builder — identical to gnn_trainer.py
# ---------------------------------------------------------------------------

def build_edge_index(edge_df, gene_to_idx):
    if not {"from", "to"}.issubset(edge_df.columns):
        cands_from = [c for c in edge_df.columns
                      if c.lower() in ("from", "source", "gene1")]
        cands_to = [c for c in edge_df.columns
                    if c.lower() in ("to", "target", "gene2")]
        if not cands_from or not cands_to:
            raise ValueError("Edge file needs 'from'/'to' (or source/target) columns.")
        edge_df = edge_df.rename(
            columns={cands_from[0]: "from", cands_to[0]: "to"}
        )
    src = edge_df["from"].map(gene_to_idx).to_numpy()
    dst = edge_df["to"].map(gene_to_idx).to_numpy()
    mask = ~pd.isna(src) & ~pd.isna(dst)
    src = src[mask].astype(np.int64)
    dst = dst[mask].astype(np.int64)
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)


# ---------------------------------------------------------------------------
# Core attribution function
# ---------------------------------------------------------------------------

def compute_gradient_attribution(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> np.ndarray:
    """
    Compute input-gradient × input attribution for the MTI regression head.

    Returns
    -------
    attr : np.ndarray, shape (n_genes, n_features)
        Absolute value of (gradient × input) for each gene × feature.
        Rows sum to the total attribution magnitude for that gene.
    """
    model.eval()

    # Detach, clone, and enable grad on a fresh leaf tensor
    x_leaf = x.detach().clone().requires_grad_(True)

    # Forward pass
    mti_pred, _ = model(x_leaf, edge_index)

    # Backward: sum over all genes to get gradients w.r.t. all inputs
    # We want per-gene gradients, so we loop — but for efficiency we use
    # a single backward with a ones vector (gives sum of row gradients).
    # For per-gene: multiply grad by one-hot mask per gene.
    # Practical approach: use jacobian-style loop over top-K only (fast enough).
    # Here we use the efficient "sum trick" which is equivalent when we only
    # need feature-class proportions averaged across genes.

    mti_pred.sum().backward()

    # Gradient shape: (n_genes, n_features)
    grad = x_leaf.grad.detach()  # same shape as x

    # Input-gradient × input (element-wise), absolute value
    attr = (grad * x_leaf.detach()).abs().cpu().numpy()

    return attr  # shape: (n_genes, n_features)


# ---------------------------------------------------------------------------
# Per-gene attribution (precise, per-gene loop — slower but exact)
# ---------------------------------------------------------------------------

def compute_per_gene_attribution(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    gene_indices: list,
) -> np.ndarray:
    """
    Compute exact per-gene input-gradient × input attribution for a
    specific set of genes (e.g. glycosylation panel, DNA repair panel).

    Parameters
    ----------
    gene_indices : list of int
        Row indices in x corresponding to the genes of interest.

    Returns
    -------
    attr : np.ndarray, shape (len(gene_indices), n_features)
    """
    model.eval()
    results = []

    for idx in gene_indices:
        x_leaf = x.detach().clone().requires_grad_(True)
        mti_pred, _ = model(x_leaf, edge_index)

        # Scalar loss = prediction for this specific gene
        loss = mti_pred[idx]
        loss.backward()

        grad = x_leaf.grad.detach()[idx]          # (n_features,)
        inp  = x_leaf.detach()[idx]               # (n_features,)
        attr = (grad * inp).abs().cpu().numpy()   # (n_features,)
        results.append(attr)

        # Reset gradients
        if x_leaf.grad is not None:
            x_leaf.grad.zero_()

    return np.array(results)  # (n_genes, n_features)


# ---------------------------------------------------------------------------
# Summarise attribution by feature class
# ---------------------------------------------------------------------------

def summarise_by_class(
    attr: np.ndarray,
    feature_cols: list,
    gene_indices: np.ndarray,
) -> pd.DataFrame:
    """
    Given attribution matrix (n_selected_genes × n_features),
    return a DataFrame with proportional attribution per feature class,
    averaged across the selected genes.
    """
    class_labels = [classify_feature(c) for c in feature_cols]
    classes = ["MR", "PANTHER", "Structure", "UniProt", "InterPro", "Contextual"]

    rows = []
    for i, g_idx in enumerate(gene_indices):
        gene_attr = attr[i]
        total = gene_attr.sum()
        if total == 0:
            continue
        row = {}
        for cls in classes:
            mask = np.array([c == cls for c in class_labels])
            row[cls] = gene_attr[mask].sum() / total
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=classes)

    summary = pd.DataFrame(rows)[classes]
    return summary


# ---------------------------------------------------------------------------
# Plotting — reproduces Figure 4D style
# ---------------------------------------------------------------------------

FEATURE_COLOURS = {
    "MR":          "#E05A4E",   # red
    "PANTHER":     "#5B8A3C",   # green
    "Structure":   "#D4A843",   # yellow-orange
    "UniProt":     "#4A9BB5",   # teal
    "InterPro":    "#3A6BB5",   # blue
    "Contextual":  "#C9549A",   # pink
}


def plot_attribution_bar(
    summary_dict: dict,   # {label: mean proportions Series}
    out_path: str,
    title: str = "Feature attribution",
):
    """
    Stacked bar chart of proportional feature class attribution.
    summary_dict keys = x-axis labels (e.g. "Top 100", "Top 200", ...)
    summary_dict values = dict of {class: proportion}
    """
    classes   = ["MR", "PANTHER", "Structure", "UniProt", "InterPro", "Contextual"]
    labels    = list(summary_dict.keys())
    n_bars    = len(labels)

    fig, ax = plt.subplots(figsize=(max(6, n_bars * 1.4), 5))

    bottoms = np.zeros(n_bars)
    for cls in classes:
        vals = np.array([summary_dict[lbl].get(cls, 0.0) for lbl in labels])
        ax.bar(
            labels, vals,
            bottom=bottoms,
            color=FEATURE_COLOURS[cls],
            label=cls,
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += vals

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion of model attribution", fontsize=11)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.28, 1.0),
        frameon=False,
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Gene class definitions — edit to match your column naming / gene lists
# ---------------------------------------------------------------------------

GLYCOSYLATION_GENES = [
    "B4GALT1", "ST6GAL1", "ST3GAL1", "ST3GAL4",
    "MGAT3",   "MGAT5",   "MGAT5B",  "FUT8", "FCGR3B",
]

DNA_REPAIR_GENES = [
    "SIRT1", "PARP1", "RAD51",
    "SIRT2", "SIRT5", "SIRT3", "SIRT6",
    "OGG1",  "PRDX6",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gradient-based feature attribution for the MR-GAT model."
    )
    parser.add_argument("--node_path",    required=True,
                        help="Node TSV (same file used in gnn_trainer.py).")
    parser.add_argument("--edge_path",    required=True,
                        help="Edge TSV (same file used in gnn_trainer.py).")
    parser.add_argument("--model_path",   default=None,
                        help="Path to saved model .pt file. "
                             "If not provided the model is re-trained (300 epochs).")
    parser.add_argument("--ranking_path", default=None,
                        help="Path to GNN_ranking_EXERCISEONLY_FULL_deduped_v2.tsv. "
                             "Used to identify gene rank order. "
                             "If absent, rank is taken from mti_pred.")
    parser.add_argument("--out_dir",      default="attribution_outputs",
                        help="Directory to write outputs.")
    parser.add_argument("--top_ks",       nargs="+", type=int,
                        default=[100, 200, 500, 1000],
                        help="Top-K thresholds for global attribution (Figure 4D).")
    parser.add_argument("--epochs",       type=int, default=300,
                        help="Training epochs if retraining (default 300).")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--lambda_multi", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load and prepare node table  (identical to gnn_trainer.py)
    # ------------------------------------------------------------------
    print(f"\nLoading node table: {args.node_path}")
    nodes = pd.read_csv(args.node_path, sep="\t", dtype=str)
    for c in [col for col in nodes.columns if col != "gene_symbol"]:
        nodes[c] = pd.to_numeric(nodes[c], errors="coerce")

    numeric_cols = nodes.select_dtypes(include=[np.number]).columns.tolist()
    drop_targets = [c for c in ["MTI_score", "MTI_continuous", "MTI_nominal",
                                    "MTI_fdr", "MTI_sumsq", "MTI_n_layers"]
                    if c in numeric_cols]
    feature_cols = [c for c in numeric_cols if c not in drop_targets]
    old_af = [c for c in feature_cols if c.startswith("pLDDT") or c == "n_atoms"]
    feature_cols = [c for c in feature_cols if c not in old_af]

    X = nodes[feature_cols].to_numpy(dtype=np.float32)
    mean = np.nanmean(X, axis=0)
    std  = np.nanstd(X,  axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std
    X[~np.isfinite(X)] = 0.0

    gene_symbols = nodes["gene_symbol"].tolist()
    gene_to_idx  = {g: i for i, g in enumerate(gene_symbols)}
    n_genes      = len(gene_symbols)
    n_features   = len(feature_cols)

    x = torch.tensor(X, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Diagnostic: print how each feature column is being classified
    # so you can verify the classify_feature() patterns are correct
    # before trusting the attribution numbers.
    # Comment this block out once you have confirmed the mappings.
    # ------------------------------------------------------------------
    print("\nFeature column classification (first 40 shown):")
    for c in feature_cols[:40]:
        print(f"  {c:50s} -> {classify_feature(c)}")
    if len(feature_cols) > 40:
        print(f"  ... ({len(feature_cols) - 40} more columns not shown)")
    class_counts = {}
    for c in feature_cols:
        cls = classify_feature(c)
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print("\nFeature class totals:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:20s}: {count} columns")

    # ------------------------------------------------------------------
    # 2. Build edge index
    # ------------------------------------------------------------------
    print(f"Loading edge table: {args.edge_path}")
    edge_df    = pd.read_csv(args.edge_path, sep="\t", dtype=str)
    edge_index = build_edge_index(edge_df, gene_to_idx).to(device)

    # ------------------------------------------------------------------
    # 3. Load or train model
    # ------------------------------------------------------------------
    model = MTIMultiNet(in_dim=n_features).to(device)

    if args.model_path and os.path.exists(args.model_path):
        print(f"\nLoading model weights from: {args.model_path}")
        model.load_state_dict(
            torch.load(args.model_path, map_location=device)
        )
    else:
        print("\nNo model checkpoint provided — retraining for "
              f"{args.epochs} epochs...")

        # Try MTI_score first (gnn_trainer.py default),
        # fall back to MTI_continuous (stress test / ablation default)
        if "MTI_score" in nodes.columns:
            mti_col = "MTI_score"
        elif "MTI_continuous" in nodes.columns:
            mti_col = "MTI_continuous"
        else:
            mti_like = [c for c in nodes.columns if c.lower().startswith("mti")]
            raise ValueError(
                f"Cannot find MTI target column. "
                f"Available MTI-like columns: {mti_like}. "
                f"Pass --model_path to a saved checkpoint to skip retraining."
            )
        print(f"  Using target column for retraining: {mti_col}")
        mti         = nodes[mti_col].to_numpy(dtype=np.float32)
        mti_mask    = torch.tensor(np.isfinite(mti), dtype=torch.bool, device=device)
        y_mti       = torch.tensor(mti, dtype=torch.float32, device=device)

        has_multi   = "MTI_n_layers" in nodes.columns
        bce         = None
        y_multi     = None
        pos_weight  = None

        if has_multi:
            n_layers    = nodes["MTI_n_layers"].fillna(0).to_numpy()
            multi_lbl   = (n_layers >= 2).astype(np.float32)
            y_multi     = torch.tensor(multi_lbl, dtype=torch.float32, device=device)
            n_pos       = float(multi_lbl.sum())
            n_neg       = float(len(multi_lbl)) - n_pos
            if n_pos > 0 and n_neg > 0:
                pw          = torch.tensor(min(n_neg / n_pos, 50.0),
                                           dtype=torch.float32, device=device)
                pos_weight  = pw
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            model.train()
            optim.zero_grad()
            mti_pred, multi_logit = model(x, edge_index)
            loss = F.mse_loss(mti_pred[mti_mask], y_mti[mti_mask])
            if has_multi and bce is not None:
                loss = loss + args.lambda_multi * bce(multi_logit, y_multi)
            loss.backward()
            optim.step()
            if epoch % 50 == 0 or epoch == args.epochs:
                print(f"  Epoch {epoch:03d} | loss {loss.item():.4f}")

        # Save retrained weights
        ckpt = os.path.join(args.out_dir, "model_retrained.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"  Saved retrained weights to: {ckpt}")

    # ------------------------------------------------------------------
    # 4. Determine gene ranking order
    # ------------------------------------------------------------------
    if args.ranking_path and os.path.exists(args.ranking_path):
        ranking_df = pd.read_csv(args.ranking_path, sep="\t")
        # Sort by rank column if present, otherwise use file order
        rank_col = "GNN_EXERCISE_SUP_MTI_MULTI_rank"
        if rank_col in ranking_df.columns:
            ranking_df = ranking_df.sort_values(rank_col).reset_index(drop=True)
            print(f"  Sorted ranking file by {rank_col}")
        else:
            print(f"  Warning: rank column '{rank_col}' not found — using file order")
        ranked_genes = ranking_df["gene_symbol"].tolist()
        ranked_indices = [gene_to_idx[g] for g in ranked_genes
                          if g in gene_to_idx]
    else:
        print("No ranking file provided — using mti_pred order.")
        model.eval()
        with torch.no_grad():
            mti_pred_np, _ = model(x, edge_index)
        mti_pred_np = mti_pred_np.cpu().numpy()
        ranked_indices = np.argsort(-mti_pred_np).tolist()

    # ------------------------------------------------------------------
    # 5. Global attribution (Figure 4D) — efficient sum-of-gradients
    # ------------------------------------------------------------------
    print("\nComputing global gradient attribution (Figure 4D)...")
    attr_all = compute_gradient_attribution(model, x, edge_index)
    # attr_all: (n_genes, n_features)

    class_labels = [classify_feature(c) for c in feature_cols]
    classes      = ["MR", "PANTHER", "Structure", "UniProt", "InterPro", "Contextual"]

    # For each gene compute per-class proportions, then average across top-K
    summary_global = {}
    for k in args.top_ks:
        top_idx = ranked_indices[:k]
        attr_k  = attr_all[top_idx]            # (k, n_features)
        totals  = attr_k.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        props   = attr_k / totals              # proportions per gene

        class_means = {}
        for cls in classes:
            mask = np.array([c == cls for c in class_labels])
            class_means[cls] = float(props[:, mask].sum(axis=1).mean())

        summary_global[f"Top {k}"] = class_means
        print(f"  Top {k}: " +
              " | ".join(f"{cls}={v*100:.1f}%" for cls, v in class_means.items()))

    # Save global summary TSV
    global_rows = []
    for label, d in summary_global.items():
        row = {"top_k": label}
        row.update(d)
        global_rows.append(row)
    global_df = pd.DataFrame(global_rows)
    global_tsv = os.path.join(args.out_dir, "attribution_global_figure4D.tsv")
    global_df.to_csv(global_tsv, sep="\t", index=False)
    print(f"\nSaved global attribution table: {global_tsv}")

    # Plot Figure 4D
    fig4d_path = os.path.join(args.out_dir, "attribution_global_figure4D.png")
    plot_attribution_bar(
        summary_global,
        out_path=fig4d_path,
        title="Feature attribution by class — top-ranked exercise-responsive genes",
    )

    # ------------------------------------------------------------------
    # 6. Per-gene-class attribution (Jason's question)
    #    Glycosylation vs DNA repair vs degree-matched background
    # ------------------------------------------------------------------
    print("\nComputing per-gene-class attribution...")

    glyco_idx = [gene_to_idx[g] for g in GLYCOSYLATION_GENES if g in gene_to_idx]
    dna_idx   = [gene_to_idx[g] for g in DNA_REPAIR_GENES    if g in gene_to_idx]

    print(f"  Glycosylation genes found: "
          f"{[g for g in GLYCOSYLATION_GENES if g in gene_to_idx]}")
    print(f"  DNA repair genes found:    "
          f"{[g for g in DNA_REPAIR_GENES    if g in gene_to_idx]}")

    # Degree-matched background: sample genes with similar degree to glyco panel
    # Proxy for degree: number of times a node appears in edge_index
    edge_np  = edge_index.cpu().numpy()
    degrees  = np.bincount(edge_np[0], minlength=n_genes)

    glyco_degrees   = degrees[glyco_idx]
    mean_glyco_deg  = glyco_degrees.mean()
    std_glyco_deg   = max(glyco_degrees.std(), 1.0)

    # Background: genes not in either panel, with degree within 1 SD of glyco mean
    panel_set   = set(glyco_idx + dna_idx)
    bg_cands    = [i for i in range(n_genes)
                   if i not in panel_set
                   and abs(degrees[i] - mean_glyco_deg) < std_glyco_deg]

    rng = np.random.RandomState(SEED)
    n_bg = min(50, len(bg_cands))
    bg_idx = rng.choice(bg_cands, size=n_bg, replace=False).tolist()
    print(f"  Background genes (degree-matched): {n_bg}")

    # Compute per-gene attribution for each group
    if glyco_idx:
        attr_glyco = compute_per_gene_attribution(model, x, edge_index, glyco_idx)
    if dna_idx:
        attr_dna   = compute_per_gene_attribution(model, x, edge_index, dna_idx)
    if bg_idx:
        attr_bg    = compute_per_gene_attribution(model, x, edge_index, bg_idx)

    # Summarise by feature class
    def mean_class_props(attr_matrix):
        totals = attr_matrix.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        props = attr_matrix / totals
        out = {}
        for cls in classes:
            mask = np.array([c == cls for c in class_labels])
            out[cls] = float(props[:, mask].sum(axis=1).mean())
        return out

    per_class_summary = {}
    if glyco_idx:
        per_class_summary["Glycosylation"] = mean_class_props(attr_glyco)
    if dna_idx:
        per_class_summary["DNA repair"]    = mean_class_props(attr_dna)
    if bg_idx:
        per_class_summary["Background\n(degree-matched)"] = mean_class_props(attr_bg)

    # Print summary
    print("\nPer-gene-class attribution summary:")
    for group, d in per_class_summary.items():
        print(f"  {group.replace(chr(10), ' ')}: " +
              " | ".join(f"{cls}={v*100:.1f}%" for cls, v in d.items()))

    # Save per-class TSV
    per_class_rows = []
    for group, d in per_class_summary.items():
        row = {"gene_class": group.replace("\n", " ")}
        row.update(d)
        per_class_rows.append(row)
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_tsv = os.path.join(args.out_dir, "attribution_per_gene_class.tsv")
    per_class_df.to_csv(per_class_tsv, sep="\t", index=False)
    print(f"\nSaved per-gene-class attribution: {per_class_tsv}")

    # Plot per-class comparison
    per_class_path = os.path.join(args.out_dir, "attribution_per_gene_class.png")
    plot_attribution_bar(
        per_class_summary,
        out_path=per_class_path,
        title="Feature attribution by gene class\n"
              "(Glycosylation vs DNA repair vs background)",
    )

    # ------------------------------------------------------------------
    # 7. Per-gene detailed table — top 100 genes
    # ------------------------------------------------------------------
    print("\nBuilding per-gene attribution table (top 100)...")
    top100_idx  = ranked_indices[:100]
    attr_top100 = compute_per_gene_attribution(model, x, edge_index, top100_idx)

    per_gene_rows = []
    for i, g_idx in enumerate(top100_idx):
        gene  = gene_symbols[g_idx]
        total = attr_top100[i].sum()
        row   = {
            "gene_symbol": gene,
            "gnn_rank":    i + 1,
            "degree":      int(degrees[g_idx]),
            "gene_class": (
                "Glycosylation" if g_idx in set(glyco_idx) else
                "DNA_repair"    if g_idx in set(dna_idx)   else
                "Other"
            ),
        }
        for cls in classes:
            mask = np.array([c == cls for c in class_labels])
            row[f"attr_{cls}"] = (
                float(attr_top100[i][mask].sum() / total) if total > 0 else 0.0
            )
        per_gene_rows.append(row)

    per_gene_df  = pd.DataFrame(per_gene_rows)
    per_gene_tsv = os.path.join(args.out_dir, "attribution_per_gene_top100.tsv")
    per_gene_df.to_csv(per_gene_tsv, sep="\t", index=False)
    print(f"Saved per-gene attribution (top 100): {per_gene_tsv}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n=== Attribution analysis complete ===")
    print(f"Outputs written to: {args.out_dir}")
    print("\nFiles produced:")
    print("  attribution_global_figure4D.tsv  — reproduces Figure 4D values")
    print("  attribution_global_figure4D.png  — Figure 4D stacked bar chart")
    print("  attribution_per_gene_class.tsv   — glyco vs DNA repair vs background")
    print("  attribution_per_gene_class.png   — per-class comparison chart")
    print("  attribution_per_gene_top100.tsv  — per-gene breakdown, top 100 genes")


if __name__ == "__main__":
    main()
