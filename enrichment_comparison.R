###############################################################################
# enrichment_comparison.R  -- reproduces Table 5.2 in full
#
# Universe for ALL enrichment tests: the multi-omic MR universe (N = 2,959
# MTI-scored genes). Reference sizes: CausAge 33, MoTrPAC 948.
#   (This is what the original 14-June run used for every row, including the
#    GAT/MLP rank rows -- NOT the connected 2,473 graph. The rank files contain
#    the connected genes, but the hypergeometric background is the MTI universe.)
#
# Rank rows report the representative initialisation seed 42 (the seed whose
# values appear in the paper; 5/5-seed robustness is reported separately in text).
#
# MR-FDR drawn set (n = 906) rebuilt as in the original run.
#
# Run from:  C:/Users/coach/Downloads/LDaware_MR_proteins
###############################################################################

setwd("C:/Users/coach/Downloads/LDaware_MR_proteins")
toU <- function(x) toupper(trimws(as.character(x)))

## --- universe + references --------------------------------------------------
mti      <- readRDS("MTI_REBUILD_5LAYER.rds")
mti$gu   <- toU(mti$gene_symbol)
universe <- unique(mti$gu)                 # N = 2,959
N        <- length(universe)

yt      <- readRDS("ying_targets.rds")
causage <- intersect(toU(yt$CausAge), universe)                            # 33
motrpac <- intersect(toU(readRDS("motrpac_EE_blood_genes.rds")), universe) # 948

## --- MR-FDR drawn set (n = 906) --------------------------------------------
fdr_sig <- function(f,g,fc,csv=FALSE){
  d <- if (csv) read.csv(f, stringsAsFactors=FALSE) else read.delim(f, stringsAsFactors=FALSE)
  d$G <- toU(d[[g]]); unique(d$G[!is.na(d[[fc]]) & d[[fc]] < 0.05])
}
prot <- fdr_sig("mr_outputs_PPP_REAL/PPP_PA_LDaware_REAL_summary.tsv","gene","fdr")
sc   <- fdr_sig("mr_outputs_TX/TX_PA_KEEPALL_summary.tsv",           "gene","fdr")
tx   <- fdr_sig("transcript_MR_eQTLGen_FULL.csv",                   "gene","transcript_fdr",csv=TRUE)
gmap <- readRDS("glycan_trait_gene_map.rds"); gly <- toU(sub("_ENSG.*$","",gmap$gene))
mr_fdr <- intersect(unique(c(prot, sc, tx, "MEF2C", gly)), universe)

cat(sprintf("universe N=%d | CausAge=%d | MoTrPAC=%d | MR-FDR=%d (expect 2959/33/948/906)\n",
            N, length(causage), length(motrpac), length(mr_fdr)))

## --- hypergeometric: p of >=x of target in a drawn set of size n ------------
enr_p <- function(drawn, target){
  M <- length(target); n <- length(drawn); x <- length(intersect(drawn, target))
  phyper(x-1, M, N-M, n, lower.tail=FALSE)
}

## --- ROW 1: model-free MR-FDR convergence ----------------------------------
cat("\n=== MR-FDR convergence ===\n")
xy <- length(intersect(mr_fdr,causage))
cat(sprintf("MR-FDR x CausAge : n=%d x=%d exp=%.1f fold=%.2f p=%.4f  (expect 16/10.1/1.6x/0.023)\n",
            length(mr_fdr), xy, length(mr_fdr)*33/N, xy/(length(mr_fdr)*33/N), enr_p(mr_fdr,causage)))
cat(sprintf("MR-FDR x MoTrPAC : p=%.4f  (expect 0.969)\n", enr_p(mr_fdr,motrpac)))

## --- rankings ---------------------------------------------------------------
read_rank <- function(path) toU(read.delim(path, header=TRUE, stringsAsFactors=FALSE)[[1]])
SEED <- 42

# GAT / MLP from rank files (seed 42)
gat <- read_rank(sprintf("rank_GAT_seed%d.tsv", SEED))
mlp <- read_rank(sprintf("rank_MLP_seed%d.tsv", SEED))

# MTI (beta) rank: order universe by MTI_score desc
mti_rank <- mti$gu[order(-mti$MTI_score)]

# MR p-value rank: order the top-1000 MR file by best_MR_fdr asc (proxy for MR p)
mr1k <- read.delim("MR_FDR_top1000_2026-06-12.tsv", stringsAsFactors=FALSE)
mrp_rank <- toU(mr1k$gene_symbol[order(mr1k$best_MR_fdr)])

KS <- c(100,150,200)
row <- function(label, ranking, target) sprintf("%-14s %7.3f %7.3f %7.3f",
            label, enr_p(head(ranking,KS[1]),target),
                   enr_p(head(ranking,KS[2]),target),
                   enr_p(head(ranking,KS[3]),target))

cat("\n=== RANK-BASED (universe 2959, seed 42) ===\n")
cat("--- CausAge (M=33) ---\n")
cat(row("MR p-value",  mrp_rank, causage), "\n")
cat(row("MTI (beta)",  mti_rank, causage), "\n")
cat(row("MLP",         mlp,      causage), "\n")
cat(row("GAT",         gat,      causage), "\n")
cat("--- MoTrPAC (M=948) ---\n")
cat(row("MR p-value",  mrp_rank, motrpac), "\n")
cat(row("MTI (beta)",  mti_rank, motrpac), "\n")
cat(row("MLP",         mlp,      motrpac), "\n")
cat(row("GAT",         gat,      motrpac), "\n")

cat("\nExpected (paper Table 5.2):\n")
cat("  MR p-value CausAge 0.307/0.505/0.665   MoTrPAC 0.971/0.994/0.985\n")
cat("  MTI (beta) CausAge 0.307/0.505/0.665   MoTrPAC 0.839/0.793/0.953\n")
cat("  MLP        CausAge 0.024/0.083/0.180   MoTrPAC 0.118/0.212/0.243\n")
cat("  GAT        CausAge 0.024/0.024/0.021   MoTrPAC 0.007/0.014/0.071\n")
