#!/usr/bin/env python3
"""
Degree-matched null for GAT/MLP rank-based enrichment.

The hypergeometric null in enrichment_comparison.R assumes every gene in the
connected universe is equally likely to enter the top K. That is not the only
alternative to a blind draw: if the ranking favoured well-connected genes, and
the reference sets were themselves degree-biased, the hypergeometric null could
be rejected without any biological signal.

This script tests that directly. Genes are binned by STRING degree; each
simulated top-K draws the same number of genes from each bin as the observed
top-K contains, so connectivity is held fixed while gene identity is
randomised. The empirical p-value is the fraction of simulations reaching the
observed overlap.

Usage:
    python degree_matched_null.py \
        --edges STRING_edges_REBUILD.tsv \
        --ranks . \
        --refs ying_ageing_causal_genes.tsv motrpac_exercise_genes.tsv \
        --nsim 10000 --bins 10 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

SEEDS = [0, 1, 7, 42, 123]
KS = [100, 150, 200]


def norm(s):
    return s.astype(str).str.strip().str.upper()


def degree_vector(edge_path, genes):
    e = pd.read_csv(edge_path, sep="\t")
    e.columns = ["a", "b"]
    d = pd.concat([norm(e.a), norm(e.b)]).value_counts()
    return d.reindex(genes).fillna(0).astype(int).values


def bin_by_degree(deg, n_bins):
    """Equal-frequency bins. Ties at a boundary fall into the same bin, so bin
    sizes are approximate rather than exact -- this is intended, since splitting
    tied degrees would break the degree matching."""
    edges = np.quantile(deg, np.linspace(0, 1, n_bins + 1))[1:-1]
    return np.searchsorted(edges, deg, side="right")


def degree_matched_p(top_idx, ref_mask, bins, pools, rng, nsim):
    """Empirical upper-tail p under a degree-matched null."""
    per_bin = np.bincount(bins[top_idx], minlength=len(pools))
    observed = int(ref_mask[top_idx].sum())
    totals = np.zeros(nsim, dtype=int)
    for b, k in enumerate(per_bin):
        if k:
            # sampled with replacement across simulations, without within a draw
            draw = rng.choice(pools[b], size=(nsim, k))
            totals += ref_mask[draw].sum(axis=1)
    return observed, totals.mean(), float(np.mean(totals >= observed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--ranks", default=".", help="directory holding rank_*.tsv")
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--nsim", type=int, default=10000)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="degree_matched_null_results.tsv")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    rank_dir = Path(a.ranks)

    genes = norm(pd.read_csv(rank_dir / "rank_GAT_seed42.tsv", sep="\t").iloc[:, 0]).tolist()
    idx = {g: i for i, g in enumerate(genes)}
    N = len(genes)

    deg = degree_vector(a.edges, genes)
    bins = bin_by_degree(deg, a.bins)
    pools = [np.where(bins == b)[0] for b in range(bins.max() + 1)]

    print(f"universe {N} genes | degree median {np.median(deg):.0f} "
          f"mean {deg.mean():.1f} max {deg.max()}")

    rows = []
    for ref_path in a.refs:
        name = Path(ref_path).stem
        ref = set(norm(pd.read_csv(ref_path, sep="\t").iloc[:, 0])) & set(genes)
        mask = np.array([g in ref for g in genes])
        M = int(mask.sum())
        print(f"\n{name}: M = {M} in connected universe | "
              f"degree median {np.median(deg[mask]):.0f} (all genes {np.median(deg):.0f})")

        for method in ("GAT", "MLP"):
            for K in KS:
                for s in SEEDS:
                    r = norm(pd.read_csv(rank_dir / f"rank_{method}_seed{s}.tsv",
                                         sep="\t").iloc[:, 0]).tolist()
                    top = np.array([idx[g] for g in r[:K]])
                    obs, dmn_e, dmn_p = degree_matched_p(top, mask, bins, pools,
                                                         rng, a.nsim)
                    rows.append(dict(
                        reference=name, method=method, K=K, seed=s, observed=obs,
                        hyper_expected=K * M / N,
                        hyper_p=hypergeom.sf(obs - 1, N, M, K),
                        degree_expected=dmn_e, degree_p=dmn_p,
                        top_k_median_degree=float(np.median(deg[top])),
                    ))

    df = pd.DataFrame(rows)
    df.to_csv(a.out, sep="\t", index=False)

    summ = (df.groupby(["reference", "method", "K"])
              .agg(observed=("observed", "median"),
                   hyper_p=("hyper_p", "median"),
                   degree_p=("degree_p", "median"),
                   degree_p_min=("degree_p", "min"),
                   degree_p_max=("degree_p", "max"),
                   sig_seeds=("degree_p", lambda x: int((x < 0.05).sum())))
              .round(4))
    print(f"\nmedian across {len(SEEDS)} seeds, {a.nsim} simulations:\n")
    print(summ.to_string())
    print(f"\nper-seed results written to {a.out}")


if __name__ == "__main__":
    main()
