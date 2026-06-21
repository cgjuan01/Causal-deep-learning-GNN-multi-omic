###############################################################################
# coloc_longevity_CTSF_eQTL.R
#
# Colocalisation of CTSF cis-eQTL (eQTLGen whole blood) with exceptional
# longevity (Deelen et al. 2019, survival beyond the 90th percentile).
#
# Reproduces the expression-arm colocalisation reported in the manuscript:
#   conditional PP.H4 = 0.615  (manuscript: 0.62)
#
# Inputs (place paths to your local copies below):
#   - eqtlgen_8genes.txt : eQTLGen cis-eQTL for the eight convergent genes,
#       columns: SNP, SNPChr, SNPPos, AssessedAllele, OtherAllele, Zscore,
#                GeneSymbol, NrSamples   (GRCh37/hg19; Z-scores, not beta)
#   - lon : exceptional-longevity GWAS (Deelen 2019),
#       file Results_90th_percentile.txt.gz,
#       columns: SNP, CHR, pos, oEA, oOA, EAF, oB, oSE, P-value, Effective_N
#
# Method: coloc.abf (coloc v5.2.3). eQTLGen Z-scores converted to beta/SE via
#   beta = Z / sqrt(2*f*(1-f)*(N + Z^2)),  SE = 1 / sqrt(2*f*(1-f)*(N + Z^2)),
#   with minor-allele frequency f taken from the matched outcome variant (EAF).
#   Exposure type="quant" (eQTLGen N); outcome type="cc", s=0.31, per-SNP
#   effective N from the GWAS. Conditional PP.H4 = PP.H4/(PP.H3+PP.H4).
###############################################################################

library(coloc)       # v5.2.3
library(data.table)

## ---- load inputs -----------------------------------------------------------
# lon <- fread("C:/Users/coach/Downloads/Results_90th_percentile.txt.gz")

eq <- fread("C:/Users/coach/Downloads/eqtlgen_8genes.txt")
setnames(eq,
  c("SNP","SNPChr","SNPPos","AssessedAllele","OtherAllele","Zscore","GeneSymbol","NrSamples"),
  c("rsid","chr","pos","EA","OA","Z","gene","N"), skip_absent = TRUE)

## ---- CTSF cis region (hg19): chr11 65.5-67.6 Mb ----------------------------
cis_eq <- eq[gene == "CTSF" & chr == 11 & pos > 65500000 & pos < 67600000]
cis_eq <- cis_eq[!duplicated(pos)]
cat("CTSF cis-eQTL variants in region:", nrow(cis_eq), "\n")

## ---- longevity outcome (dedup on position) ---------------------------------
lonc <- lon[CHR == 11, .(pos19 = pos, oEA, oOA,
                         beta_y = oB, se_y = oSE, Neff = Effective_N)]
lonc <- lonc[!duplicated(pos19)]
eaf  <- lon[CHR == 11, .(pos19 = pos, eaf = EAF)][!duplicated(pos19)]

## ---- merge exposure + outcome on hg19 position -----------------------------
mc <- merge(cis_eq[, .(pos19 = pos, ea = EA, oa = OA, Z, Nexp = N)],
            lonc, by = "pos19")
cat("merged variants:", nrow(mc), "\n")

## ---- harmonise alleles -----------------------------------------------------
same <- toupper(mc$oEA) == toupper(mc$ea) & toupper(mc$oOA) == toupper(mc$oa)
flip <- toupper(mc$oEA) == toupper(mc$oa) & toupper(mc$oOA) == toupper(mc$ea)
mc <- mc[same | flip]
mc[, beta_y := ifelse(toupper(oEA) == toupper(oa) & toupper(oOA) == toupper(ea),
                      -beta_y, beta_y)]

## ---- eQTLGen Z -> beta/SE, using MAF from matched outcome variant ----------
mc <- merge(mc, eaf, by = "pos19")
mc[, maf := pmin(eaf, 1 - eaf)]
mc <- mc[maf > 0 & maf < 1 & se_y > 0]
mc[, beta_x := Z / sqrt(2 * maf * (1 - maf) * (Nexp + Z^2))]
mc[, se_x   := 1 / sqrt(2 * maf * (1 - maf) * (Nexp + Z^2))]
cat("variants in coloc region after harmonisation:", nrow(mc), "\n")

## ---- colocalisation --------------------------------------------------------
D1 <- list(beta = mc$beta_x, varbeta = mc$se_x^2, snp = as.character(mc$pos19),
           MAF = mc$maf, type = "quant", N = round(mean(mc$Nexp)))     # eQTLGen
D2 <- list(beta = mc$beta_y, varbeta = mc$se_y^2, snp = as.character(mc$pos19),
           type = "cc", s = 0.31, N = round(mean(mc$Neff)))            # longevity cc

res <- coloc.abf(D1, D2)
condH4 <- res$summary["PP.H4.abf"] /
          (res$summary["PP.H3.abf"] + res$summary["PP.H4.abf"])

cat(sprintf(paste0("\nCTSF eQTL vs longevity coloc:\n",
                   "  nsnps=%d  PP.H3=%.3f  PP.H4=%.3f  conditional PP.H4=%.3f\n"),
            res$summary["nsnps"], res$summary["PP.H3.abf"],
            res$summary["PP.H4.abf"], condH4))

## Expected output:
##   nsnps=2889  PP.H3=0.286  PP.H4=0.455  conditional PP.H4=0.615
