#!/usr/bin/env python3
"""
train_GNN_EXERCISE_SUP_MTI_MULTI_AFv6_HYBRID_v2.py

Updated supervised GAT trainer:
 - robust to different MR p column names and '_fixed' patched MR columns
 - integrates AlphaFold v6 / InterPro / UniProt PCs already stored in node table
 - computes MTI-only and hybrid (MTI + multi-omic multi_prob) rankings
 - writes:
     * full GNN ranking table (deduplicated)
     * MR-nominal subset (any MR p < 0.05) (deduplicated)
     * Top-N TSVs (full and MR-nominal)
 - logging and safety checks included.

STRESS-TEST EXTENSIONS (no model change):
 - --feature_list : restrict X columns (for FULL / NO_MR / MR_ONLY ablations)
 - --target_col   : choose training target (MTI_continuous / MTI_nominal / MTI_fdr ...)
 - --run_name     : suffix outputs so multiple runs don't overwrite each other
"""
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import os

# -------------------------
# Repro
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# Model
# -------------------------
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


# -------------------------
# Utils
# -------------------------
def build_edge_index(edge_df, gene_to_idx):
    if not {"from", "to"}.issubset(edge_df.columns):
        candidates_from = [c for c in edge_df.columns if c.lower() in ("from", "source", "gene1")]
        candidates_to   = [c for c in edge_df.columns if c.lower() in ("to", "target", "gene2")]
        if not candidates_from or not candidates_to:
            raise ValueError("Edge file must contain 'from' and 'to' (or source/target) columns.")
        edge_df = edge_df.rename(columns={candidates_from[0]: "from", candidates_to[0]: "to"})

    src = edge_df["from"].map(gene_to_idx).to_numpy()
    dst = edge_df["to"].map(gene_to_idx).to_numpy()
    mask = ~pd.isna(src) & ~pd.isna(dst)
    src = src[mask].astype(np.int64)
    dst = dst[mask].astype(np.int64)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    return edge_index

def find_mr_p_columns(df):
    # Prefer obvious MR p-value columns
    cand = [c for c in df.columns if c.lower().endswith("_mr_p") or c.lower().endswith("_p")]
    mr_cand = [c for c in cand if any(k in c.lower() for k in ("mr", "protein", "cpg", "glycan", "sc_"))]
    return mr_cand if mr_cand else cand

def any_mr_p_below(df, cols, thresh=0.05):
    if not cols:
        return pd.Series([False] * len(df), index=df.index)
    mask = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns:
            mask = mask | (pd.to_numeric(df[c], errors="coerce") < thresh)
    return mask.fillna(False)

def dedupe_keep_first(df, key="gene_symbol"):
    return df.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)

def read_feature_list(path):
    """
    Read a plain-text feature list file.
    One column name per line. Lines beginning with '#' are comments and ignored.
    Used to restrict the GNN input feature matrix for ablation runs
    (e.g. NO_MR: exclude all MR-derived columns; MR_ONLY: include only MR columns).
    """
    cols = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            cols.append(ln)
    return cols


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--node_path", required=True,
                        help="Node TSV path (node table with MTI + features).")
    parser.add_argument("--edge_path", required=True,
                        help="Edge TSV path (kNN edges).")

    parser.add_argument("--feature_list", default=None,
                        help="Optional text file with one feature column name per line.")
    parser.add_argument("--target_col", default="MTI_continuous",
                        help="Target column to predict.")
    parser.add_argument("--run_name", default="",
                        help="Suffix for outputs (FULL / NO_MR / MR_ONLY).")

    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_multi", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top_n", type=int, default=300)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    node_path = args.node_path
    edge_path = args.edge_path
    out_dir = args.out_dir if args.out_dir else os.path.dirname(node_path)
    os.makedirs(out_dir, exist_ok=True)

    suffix = f"_{args.run_name}" if args.run_name else ""

    print(f"Loading node table:\n   {node_path}")
    nodes = pd.read_csv(node_path, sep="\t")
    print(f"Node rows: {len(nodes)}  Cols: {len(nodes.columns)}")

    if "gene_symbol" not in nodes.columns:
        raise ValueError("node table must contain 'gene_symbol' column.")

    # target column
    if args.target_col not in nodes.columns:
        mti_like = [c for c in nodes.columns if c.startswith("MTI")]
        raise ValueError(f"--target_col '{args.target_col}' not found. Available MTI-like columns: {mti_like}")

    # Coerce everything except gene_symbol to numeric (strings -> NaN)
    for c in nodes.columns:
        if c == "gene_symbol":
            continue
        nodes[c] = pd.to_numeric(nodes[c], errors="coerce")

    # Build y
    mti = nodes[args.target_col].to_numpy(dtype=np.float32)
    mti_mask = np.isfinite(mti)

    # Build feature columns
    if args.feature_list is not None:
        feature_cols = read_feature_list(args.feature_list)
        missing = [c for c in feature_cols if c not in nodes.columns]
        if missing:
            raise ValueError(f"Feature columns missing from node table: {missing[:25]}{'...' if len(missing)>25 else ''}")
    else:
        # Default behaviour (original): numeric cols excluding key MTI summary columns (but NOT excluding target_col by default)
        numeric_cols = nodes.select_dtypes(include=[np.number]).columns.tolist()
        drop_targets = [c for c in ["MTI_score", "MTI_sumsq", "MTI_n_layers"] if c in numeric_cols]
        feature_cols = [c for c in numeric_cols if c not in drop_targets]

    # Drop old AF columns to prefer af_* (original behaviour)
    old_af_cols = [c for c in feature_cols if c.startswith("pLDDT") or c == "n_atoms"]
    feature_cols = [c for c in feature_cols if c not in old_af_cols]
    afv6_cols = [c for c in feature_cols if c.startswith("af_")]

    print("\nTarget column:", args.target_col)
    print("Using feature columns for GNN (first 20):")
    print("  ", feature_cols[:20], "...")
    print("Total feature dims:", len(feature_cols))
    print("Dropped old AF columns:", old_af_cols if old_af_cols else "None")
    print("AFv6 columns detected:", afv6_cols if afv6_cols else "None")
    if args.feature_list:
        print("Feature list file:", args.feature_list)

    # Build X
    X = nodes[feature_cols].to_numpy(dtype=np.float32)

    # Multi-layer label handling
    has_multi = "MTI_n_layers" in nodes.columns
    multi_labels = None
    pos_weight = None
    if has_multi:
        n_layers = nodes["MTI_n_layers"].fillna(0).to_numpy()
        multi_labels = (n_layers >= 2).astype(np.float32)
        n_pos = float(np.nansum(multi_labels))
        n_tot = float(len(multi_labels))
        n_neg = n_tot - n_pos
        print(f"\nMulti-layer positives (MTI_n_layers >= 2): {int(n_pos)} / {int(n_tot)}")
        if n_pos > 0 and n_neg > 0:
            raw_pw = n_neg / n_pos
            capped_pw = min(raw_pw, 50.0)
            pos_weight = torch.tensor(capped_pw, dtype=torch.float32, device=device)
            print(f"Using pos_weight for BCE: raw={raw_pw:.2f}, capped={capped_pw:.2f}")
        else:
            print("No positive multi-layer examples; disabling multi-layer loss.")
            has_multi = False

    # Standardise features
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std
    X[~np.isfinite(X)] = 0.0

    x = torch.tensor(X, dtype=torch.float32, device=device)
    y_mti = torch.tensor(mti, dtype=torch.float32, device=device)
    mti_mask_t = torch.tensor(mti_mask, dtype=torch.bool, device=device)

    if has_multi:
        y_multi = torch.tensor(multi_labels, dtype=torch.float32, device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        y_multi = None
        bce = None

    # edges
    print(f"\nLoading edge table:\n   {edge_path}")
    edge_df = pd.read_csv(edge_path, sep="\t", dtype=str)
    gene_symbols = nodes["gene_symbol"].tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    edge_index = build_edge_index(edge_df, gene_to_idx)
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge index range src [{edge_index[0].min().item()}..{edge_index[0].max().item()}], "
          f"dst [{edge_index[1].min().item()}..{edge_index[1].max().item()}]")

    data = Data(x=x, edge_index=edge_index).to(device)

    # model
    model = MTIMultiNet(in_dim=len(feature_cols), hidden_dim=64, out_dim=64, heads=4, dropout=0.2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("\nStarting training…\n")
    lambda_multi = args.lambda_multi if has_multi else 0.0
    print(f"lambda_multi = {lambda_multi}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optim.zero_grad()
        mti_pred, multi_logit = model(data.x, data.edge_index)

        mti_loss = F.mse_loss(mti_pred[mti_mask_t], y_mti[mti_mask_t])
        if has_multi and lambda_multi > 0.0:
            multi_loss = bce(multi_logit, y_multi)
            loss = mti_loss + lambda_multi * multi_loss
        else:
            multi_loss = torch.tensor(0.0, device=device)
            loss = mti_loss

        if torch.isnan(loss):
            print(f"Epoch {epoch:03d} | NaN loss - aborting.")
            break

        loss.backward()
        optim.step()

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | MTI MSE: {mti_loss.item():.4f} | "
                  f"Multi BCE: {multi_loss.item():.4f} | Total: {loss.item():.4f}")

    print("\nTraining complete.\n")

    # eval + ranking
    model.eval()
    with torch.no_grad():
        mti_pred, multi_logit = model(data.x, data.edge_index)
        multi_prob = torch.sigmoid(multi_logit)

    mti_pred_np = mti_pred.cpu().numpy()
    multi_prob_np = multi_prob.cpu().numpy()
    n = len(mti_pred_np)

    # MTI-only ranks
    order_mti = np.argsort(-mti_pred_np)
    ranks_mti = np.empty_like(order_mti)
    ranks_mti[order_mti] = np.arange(1, n + 1)
    rank_mti_scaled = (n - ranks_mti) / (n - 1.0)

    # Hybrid score: z(MTI_pred) + alpha * z(multi_prob)
    def zscore(arr):
        s = arr.std()
        return (arr - arr.mean()) / s if s > 0 else arr - arr.mean()

    hybrid_score = zscore(mti_pred_np) + args.alpha * zscore(multi_prob_np)

    order_h = np.argsort(-hybrid_score)
    ranks_h = np.empty_like(order_h)
    ranks_h[order_h] = np.arange(1, n + 1)
    rank_h_scaled = (n - ranks_h) / (n - 1.0)

    # assemble output DF
    out = nodes[["gene_symbol"]].copy()
    out[args.target_col] = nodes[args.target_col]
    out["GNN_EXERCISE_SUP_MTI_MULTI_pred"] = mti_pred_np
    out["GNN_EXERCISE_SUP_MTI_MULTI_multi_prob"] = multi_prob_np
    out["GNN_EXERCISE_SUP_MTI_MULTI_rank_MTIonly"] = ranks_mti.astype(int)
    out["GNN_EXERCISE_SUP_MTI_MULTI_rank_MTIonly_scaled"] = rank_mti_scaled
    out["GNN_EXERCISE_SUP_MTI_MULTI_hybrid_score"] = hybrid_score
    out["GNN_EXERCISE_SUP_MTI_MULTI_rank"] = ranks_h.astype(int)
    out["GNN_EXERCISE_SUP_MTI_MULTI_rank_scaled"] = rank_h_scaled

    # write embedding/rank file
    out_path = os.path.join(out_dir, f"GNN_embeddings_EXERCISEONLY_GAT_SUP_MTI_MULTI_AFv6_HYBRID_v2{suffix}.tsv")
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Saved embeddings + ranks to:\n   {out_path}")

    # dedupe
    out_sorted = out.sort_values("GNN_EXERCISE_SUP_MTI_MULTI_rank").reset_index(drop=True)
    out_dedup = dedupe_keep_first(out_sorted, key="gene_symbol")

    # MR-nominal detection (from node table p columns)
    mr_p_cols = find_mr_p_columns(nodes)
    print(f"Detected MR p columns (candidates): {mr_p_cols}")
    nom_mask = any_mr_p_below(nodes, mr_p_cols, thresh=0.05)
    out_dedup["MR_nominal_any"] = nom_mask.astype(bool).values

    # write full ranking deduped
    full_out_path = os.path.join(out_dir, f"GNN_ranking_EXERCISEONLY_FULL_deduped_v2{suffix}.tsv")
    out_dedup.to_csv(full_out_path, sep="\t", index=False)
    print(f"Saved deduped full ranking to:\n   {full_out_path}  Rows: {len(out_dedup)}")

    # MR-nominal subset
    mr_nominal_df = out_dedup[out_dedup["MR_nominal_any"]].reset_index(drop=True)
    mr_nom_path = os.path.join(out_dir, f"GNN_ranking_EXERCISEONLY_MR_nominal_deduped_v2{suffix}.tsv")
    mr_nominal_df.to_csv(mr_nom_path, sep="\t", index=False)
    print(f"Saved MR-nominal deduped ranking to:\n   {mr_nom_path}  Rows (MR-nominal): {len(mr_nominal_df)}")

    # Top-N files
    topn = args.top_n
    topn_full = out_dedup.sort_values("GNN_EXERCISE_SUP_MTI_MULTI_rank").head(topn)
    topn_full_path = os.path.join(out_dir, f"GNN_top{topn}_EXERCISEONLY_FULL_v2{suffix}.tsv")
    topn_full.to_csv(topn_full_path, sep="\t", index=False)

    topn_nom = mr_nominal_df.sort_values("GNN_EXERCISE_SUP_MTI_MULTI_rank").head(topn)
    topn_nom_path = os.path.join(out_dir, f"GNN_top{topn}_EXERCISEONLY_MRnominal_v2{suffix}.tsv")
    topn_nom.to_csv(topn_nom_path, sep="\t", index=False)

    print(f"Saved Top {topn} full: {topn_full_path}  Rows: {len(topn_full)}")
    print(f"Saved Top {topn} MR-nominal: {topn_nom_path}  Rows: {len(topn_nom)}")

    print("\nTop 10 (hybrid-ranked):")
    print(out_dedup[["gene_symbol", "GNN_EXERCISE_SUP_MTI_MULTI_rank", "GNN_EXERCISE_SUP_MTI_MULTI_rank_scaled"]]
          .sort_values("GNN_EXERCISE_SUP_MTI_MULTI_rank")
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()

