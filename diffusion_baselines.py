#!/usr/bin/env python3
"""
Propagation baselines for the GAT ranking.

Three comparators, all computed without any learning, on the same connected
STRING graph and the same MTI seed vector the GAT is trained on:

  1. degree-only     rank genes by STRING degree; no MR signal at all
  2. PPR             personalised PageRank seeded with MTI, swept over the
                     restart parameter alpha (alpha -> 0 recovers the MTI
                     ranking; alpha -> 1 converges toward degree)
  3. rewiring null   PPR on degree-preserving rewired graphs, to ask whether
                     the specific biological wiring carries signal beyond the
                     degree sequence

Note this rewires the graph under *diffusion*, not under the GAT. Retraining
multiseed.py on rewired graphs is a separate, still-outstanding control.

Usage:
    python diffusion_baselines.py --edges STRING_edges_REBUILD.tsv \
        --nodes MTI_REBUILD_5LAYER.tsv --ranks . \
        --refs ying_ageing_causal_genes.tsv motrpac_exercise_genes.tsv \
        --nrewire 100 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

KS = [100, 150, 200]
ALPHAS = [0.15, 0.30, 0.50, 0.70, 0.85, 0.95]


def norm(s):
    return s.astype(str).str.strip().str.upper()


def adjacency(edges, n):
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    return A


def ppr(edges, seed_vec, alpha, n, iters=150):
    """Symmetrically normalised personalised PageRank by power iteration."""
    A = adjacency(edges, n)
    deg = np.asarray(A.sum(1)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(deg, 1))
    S = sp.diags(inv) @ A @ sp.diags(inv)
    y0 = seed_vec / seed_vec.sum()
    f = y0.copy()
    for _ in range(iters):
        f = alpha * (S @ f) + (1 - alpha) * y0
    return f


def rewire(edges, rng, rounds=10):
    """Degree-preserving double edge swap. Each swap replaces (a,b),(c,d) with
    (a,d),(c,b), leaving every node's degree unchanged."""
    E = edges.copy()
    for _ in range(rounds * len(E) // 2):
        i, j = rng.integers(0, len(E), 2)
        if i == j:
            continue
        a, b = E[i]
        c, d = E[j]
        if len({a, b, c, d}) < 4:
            continue
        E[i] = [a, d]
        E[j] = [c, b]
    return E


def top_overlap(score, gene_arr, ref, K):
    return len(set(gene_arr[np.argsort(-score)][:K]) & ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--ranks", default=".")
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--nrewire", type=int, default=100)
    ap.add_argument("--rewire-alpha", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)

    genes = norm(pd.read_csv(Path(a.ranks) / "rank_GAT_seed42.tsv",
                             sep="\t").iloc[:, 0]).tolist()
    idx = {g: i for i, g in enumerate(genes)}
    gene_arr = np.array(genes)
    N = len(genes)

    e = pd.read_csv(a.edges, sep="\t")
    e.columns = ["a", "b"]
    E0 = np.c_[norm(e.a).map(idx).values, norm(e.b).map(idx).values]

    nodes = pd.read_csv(a.nodes, sep="\t")
    nodes["g"] = norm(nodes.gene_symbol)
    y = nodes.set_index("g").MTI_score.reindex(genes).fillna(0).values

    refs = {Path(p).stem: set(norm(pd.read_csv(p, sep="\t").iloc[:, 0])) & set(genes)
            for p in a.refs}

    deg = np.asarray(adjacency(E0, N).sum(1)).ravel()

    def report(label, score):
        cells = []
        for name, ref in refs.items():
            for K in KS:
                x = top_overlap(score, gene_arr, ref, K)
                p = hypergeom.sf(x - 1, N, len(ref), K)
                cells.append(f"{name[:7]} K={K}: x={x:3d} p={p:.3f}")
        print(f"  {label:<16} " + " | ".join(cells))

    print(f"universe {N} genes, {len(E0)} edges\n")
    print("baseline rankings (no learning):")
    report("MTI only", y)
    report("degree only", deg.astype(float))
    for alpha in ALPHAS:
        report(f"PPR alpha={alpha}", ppr(E0, y, alpha, N))

    print(f"\ndegree-preserving rewiring null "
          f"(PPR alpha={a.rewire_alpha}, {a.nrewire} rewires):")
    real = ppr(E0, y, a.rewire_alpha, N)
    null = {(nm, K): [] for nm in refs for K in KS}
    for _ in range(a.nrewire):
        f = ppr(rewire(E0, rng), y, a.rewire_alpha, N)
        for nm, ref in refs.items():
            for K in KS:
                null[(nm, K)].append(top_overlap(f, gene_arr, ref, K))
    for nm, ref in refs.items():
        for K in KS:
            obs = top_overlap(real, gene_arr, ref, K)
            arr = np.array(null[(nm, K)])
            print(f"  {nm[:7]:<8} K={K:<4} real={obs:3d}  "
                  f"rewired mean={arr.mean():6.1f}  empirical p={np.mean(arr >= obs):.3f}")


if __name__ == "__main__":
    main()
