###############################################################################
# coloc_longevity_CTSF_pQTL.R
#
# Colocalisation of CTSF cis-pQTL (UKB-PPP) with exceptional longevity
# (Deelen et al. 2019, survival beyond the 90th percentile).
#
# Reproduces the protein-arm colocalisation reported in the manuscript:
#   conditional PP.H4 = 0.777  (manuscript: 0.77)
#
# Inputs (place paths to your local copies below):
#   - cis_sig2 : CTSF cis-pQTL, UKB-PPP (GRCh38 lifted to GRCh37/hg19),
#                columns: id_chr, pos19, ALLELE0, ALLELE1, BETA, SE, A1FREQ
#                (object built in the LD-clumping pipeline; or rebuild from the
#                 UKB-PPP CTSF discovery file CTSF_Q9UBX1_OID21500_v1_*.tar)
#   - lon      : exceptional-longevity GWAS (Deelen 2019),
#                file Results_90th_percentile.txt.gz,
#                columns: SNP, CHR, pos, oEA, oOA, EAF, oB, oSE, P-value, Effective_N
#
# Method: coloc.abf (coloc v5.2.3). Exposure type="quant" (UKB-PPP N=33,822);
#   outcome type="cc" with case proportion s = 11,262/36,745 = 0.31 and per-SNP
#   effective N from the GWAS (mean ~ 11,473 across the CTSF cis region).
#   Conditional PP.H4 = PP.H4 / (PP.H3 + PP.H4); >0.7 taken as colocalisation.
###############################################################################

library(coloc)       # v5.2.3
library(data.table)

## ---- load inputs -----------------------------------------------------------
## If cis_sig2 and lon are not already in the session, read them here, e.g.:
# lon      <- fread("C:/Users/coach/Downloads/Results_90th_percentile.txt.gz")
# cis_sig2 <- fread("C:/Users/coach/Downloads/CTSF_cis_pQTL_hg19.csv")  # your lifted pQTL

## ---- CTSF cis region (hg19): chr11 65.5-67.6 Mb ----------------------------
cisreg <- cis_sig2[id_chr == 11 & pos19 > 65500000 & pos19 < 67600000]
cisreg <- cisreg[!duplicated(pos19)]                       # dedup exposure by position

lonc <- lon[CHR == 11, .(pos19 = pos, oEA, oOA,
                         beta_y = oB, se_y = oSE, Neff = Effective_N)]
lonc <- lonc[!duplicated(pos19)]                           # dedup outcome by position

## ---- merge on hg19 position ------------------------------------------------
mc <- merge(cisreg[, .(pos19, ea = ALLELE1, oa = ALLELE0,
                       beta_x = BETA, se_x = SE, eaf = A1FREQ)],
            lonc, by = "pos19")
cat("merged rows:", nrow(mc), "\n")

## ---- harmonise alleles (flip outcome beta where alleles reversed) ----------
same <- toupper(mc$oEA) == toupper(mc$ea) & toupper(mc$oOA) == toupper(mc$oa)
flip <- toupper(mc$oEA) == toupper(mc$oa) & toupper(mc$oOA) == toupper(mc$ea)
mc <- mc[same | flip]
mc[, beta_y := ifelse(toupper(oEA) == toupper(oa) & toupper(oOA) == toupper(ea),
                      -beta_y, beta_y)]
mc <- mc[eaf > 0 & eaf < 1 & se_x > 0 & se_y > 0]
cat("variants in coloc region after harmonisation:", nrow(mc), "\n")

## ---- colocalisation --------------------------------------------------------
D1 <- list(beta = mc$beta_x, varbeta = mc$se_x^2, snp = as.character(mc$pos19),
           MAF = pmin(mc$eaf, 1 - mc$eaf), type = "quant", N = 33822)   # CTSF pQTL
D2 <- list(beta = mc$beta_y, varbeta = mc$se_y^2, snp = as.character(mc$pos19),
           type = "cc", s = 0.31, N = round(mean(mc$Neff)))             # longevity cc

res <- coloc.abf(D1, D2)
condH4 <- res$summary["PP.H4.abf"] /
          (res$summary["PP.H3.abf"] + res$summary["PP.H4.abf"])

cat(sprintf(paste0("\nCTSF pQTL vs longevity coloc:\n",
                   "  nsnps=%d  PP.H3=%.3f  PP.H4=%.3f  conditional PP.H4=%.3f\n"),
            res$summary["nsnps"], res$summary["PP.H3.abf"],
            res$summary["PP.H4.abf"], condH4))
cat("  longevity effective N used:", round(mean(mc$Neff)), "\n")

## Expected output:
##   nsnps=900  PP.H3=0.182  PP.H4=0.632  conditional PP.H4=0.777
