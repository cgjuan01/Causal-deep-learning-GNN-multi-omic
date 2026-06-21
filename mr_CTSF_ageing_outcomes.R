###############################################################################
# mr_CTSF_ageing_outcomes.R
#
# Two-sample cis-MR of CTSF against four ageing outcomes, in two instrument
# arms (cis-pQTL, UKB-PPP; cis-eQTL, eQTLGen), on LD-clumped instruments.
# Reproduces the CTSF MR table reported in the manuscript.
#
# Outcomes:
#   longevity   - Deelen et al. 2019 (survival > 90th percentile; case-control)
#   aging-GIP1  - Timmers et al. 2022 (quantitative)
#   healthspan  - Zenin  et al. 2019 (quantitative)
#   lifespan    - Timmers et al. 2019 (parental lifespan; log-hazard)
#
# Instruments: LD-clumped (r2 < 0.001, 10 Mb window, 1000G EUR via OpenGWAS)
#   from the companion clumping step. The clumped rsID sets are hard-coded
#   below so this script is reproducible offline without re-clumping; re-run
#   the clumping pipeline to regenerate them.
#
# Matching: longevity is reported as chr:pos and matched on hg19 position;
#   the rsID-bearing outcomes (GIP1, healthspan, lifespan) are matched on rsID.
#   This is why the instrument count (k) differs by outcome.
#
# eQTL arm: eQTLGen Z-scores -> beta/SE via
#   beta = Z / sqrt(2*f*(1-f)*(N + Z^2)),  SE = 1 / sqrt(2*f*(1-f)*(N + Z^2)),
#   with f = MAF of the matched outcome variant.
# pQTL arm: BETA/SE used directly (UKB-PPP, lifted to hg19).
#
# Causal estimate: IVW (k >= 2) or Wald ratio (k = 1). Heterogeneity by
#   Cochran's Q. Steiger direction and colocalisation are computed in separate
#   scripts (coloc_longevity_CTSF_*.R); a pair is validated only if FDR-sig MR
#   AND Steiger-consistent AND conditional PP.H4 > 0.7.
###############################################################################

library(data.table)

## ===========================================================================
## 1. Clumped instrument sets (output of the LD-clumping pipeline)
## ===========================================================================
clumped_eqtl_rsid <- c("rs12786563","rs35965170","rs75898802","rs6560",
                       "rs7932285","rs149455081","rs35750459")          # 7
clumped_pqtl_rsid <- c("rs4930384","rs577949342","rs181934483")          # 3

## ===========================================================================
## 2. Exposures
## ===========================================================================
## --- eQTL exposure: eQTLGen CTSF cis-eQTL (hg19; Z-scores) ------------------
eq <- fread("C:/Users/coach/Downloads/eqtlgen_8genes.txt")
setnames(eq,
  c("SNP","SNPChr","SNPPos","AssessedAllele","OtherAllele","Zscore","GeneSymbol","NrSamples"),
  c("rsid","chr","pos","EA","OA","Z","gene","N"), skip_absent = TRUE)
ctsf_eq <- eq[gene == "CTSF" & rsid %in% clumped_eqtl_rsid][!duplicated(rsid)]

## --- pQTL exposure: UKB-PPP CTSF cis-pQTL (lifted to hg19; BETA/SE) ---------
## cis_sig2 is the harmonised CTSF cis-pQTL object built in the clumping
## pipeline, with columns: rsid, pos19, ALLELE1 (EA), ALLELE0 (OA), BETA, SE.
## If not in the session, rebuild it from CTSF_Q9UBX1_OID21500_v1_*.tar.
ctsf_pq <- cis_sig2[rsid %in% clumped_pqtl_rsid][!duplicated(rsid)]

## ===========================================================================
## 3. Outcomes -> normalised to (rsid, pos19, oEA, oOA, oB, oSE, ofrq)
## ===========================================================================
load_outcome <- function(nm){
  if (nm == "longevity"){
    d <- fread("C:/Users/coach/Downloads/Results_90th_percentile.txt.gz")
    d[CHR == 11, .(rsid = SNP, pos19 = pos, oEA, oOA,
                   oB, oSE, ofrq = EAF, match = "pos")]
  } else if (nm == "gip1"){
    d <- fread("C:/Users/coach/Downloads/2021_07_06_human_ageing_gip1.tsv.gz")
    d[, .(rsid, pos19 = NA_integer_, oEA = a1, oOA = a0,
          oB = beta1, oSE = se, ofrq = freq1, match = "rsid")]
  } else if (nm == "healthspan"){
    d <- fread("C:/Users/coach/Downloads/healthspan_summary.csv.gz")
    d[, .(rsid = SNPID, pos19 = NA_integer_, oEA = EA, oOA = RA,
          oB = beta, oSE = se, ofrq = EAF, match = "rsid")]
  } else if (nm == "lifespan"){
    d <- fread("C:/Users/coach/Downloads/lifegen_phase2_bothpl_alldr_2017_09_18.tsv.gz")
    d[, .(rsid, pos19 = NA_integer_, oEA = a1, oOA = a0,
          oB = beta1, oSE = se, ofrq = freq1, match = "rsid")]
  }
}

## ===========================================================================
## 4. Estimators
## ===========================================================================
ivw <- function(bx, bxse, by, byse){
  if (length(bx) == 1){                                  # Wald ratio
    b <- by/bx; se <- abs(byse/bx); z <- b/se
    return(list(k = 1, b = b, se = se, p = 2*pnorm(-abs(z)), Qp = NA))
  }
  w  <- 1/byse^2
  b  <- sum(w*bx*by)/sum(w*bx^2)
  se <- sqrt(1/sum(w*bx^2)); z <- b/se
  Q  <- sum(w*(by - b*bx)^2); Qp <- pchisq(Q, length(bx)-1, lower.tail = FALSE)
  list(k = length(bx), b = b, se = se, p = 2*pnorm(-abs(z)), Qp = Qp)
}

harmonise <- function(m){
  same <- toupper(m$oEA) == toupper(m$ea) & toupper(m$oOA) == toupper(m$oa)
  flip <- toupper(m$oEA) == toupper(m$oa) & toupper(m$oOA) == toupper(m$ea)
  m <- m[same | flip]
  m[, oB := ifelse(toupper(oEA) == toupper(oa) & toupper(oOA) == toupper(ea), -oB, oB)]
  m
}

## ===========================================================================
## 5. Run: 4 outcomes x 2 arms
## ===========================================================================
cat(sprintf("%-11s %-5s %2s  %-8s %-10s %-6s\n",
            "outcome","arm","k","beta","p","Q_p"))
cat(strrep("-", 46), "\n")

for (nm in c("longevity","gip1","healthspan","lifespan")){
  out <- load_outcome(nm)
  key <- out$match[1]                                   # "pos" or "rsid"

  ## ---- eQTL arm (convert Z -> beta/se with outcome MAF) -------------------
  ex <- ctsf_eq[, .(rsid, pos19 = pos, ea = EA, oa = OA, Z, Nexp = N)]
  m  <- if (key == "pos") merge(ex, out, by = "pos19")
        else               merge(ex, out, by = "rsid")
  if (nrow(m) > 0){
    m <- harmonise(m)
    m[, maf := pmin(ofrq, 1-ofrq)]; m <- m[!is.na(maf) & maf > 0 & maf < 1 & oSE > 0]
    m[, bx   := Z / sqrt(2*maf*(1-maf)*(Nexp + Z^2))]
    m[, bxse := 1 / sqrt(2*maf*(1-maf)*(Nexp + Z^2))]
    r <- ivw(m$bx, m$bxse, m$oB, m$oSE)
    cat(sprintf("%-11s %-5s %2d  %+8.3f %.2e   %s\n", nm, "eQTL", r$k, r$b, r$p,
                ifelse(is.na(r$Qp), "-", sprintf("%.2f", r$Qp))))
  }

  ## ---- pQTL arm (BETA/SE used directly) ----------------------------------
  ex <- ctsf_pq[, .(rsid, pos19, ea = ALLELE1, oa = ALLELE0, bx = BETA, bxse = SE)]
  mp <- if (key == "pos") merge(ex, out, by = "pos19")
        else               merge(ex, out, by = "rsid")
  if (nrow(mp) > 0){
    mp <- harmonise(mp)
    mp <- mp[oSE > 0]
    r <- ivw(mp$bx, mp$bxse, mp$oB, mp$oSE)
    cat(sprintf("%-11s %-5s %2d  %+8.3f %.2e   %s\n", nm, "pQTL", r$k, r$b, r$p,
                ifelse(is.na(r$Qp), "-", sprintf("%.2f", r$Qp))))
  }
}

## Expected output (matches manuscript):
##   outcome     arm    k  beta      p          Q_p
##   ----------------------------------------------
##   longevity   eQTL    6   +0.189  6.63e-04   0.99
##   longevity   pQTL    1   +0.331  1.77e-03   -
##   gip1        eQTL    6   +0.034  5.00e-04   0.06
##   gip1        pQTL    1   +0.044  2.40e-02   -
##   healthspan  eQTL    7   -0.003  8.10e-01   (ns)
##   healthspan  pQTL    2   +0.051  4.60e-02   0.71
##   lifespan    eQTL    7   +0.005  6.10e-01   (ns)
##   lifespan    pQTL    2   -0.045  2.90e-02   0.96
##
## Only longevity colocalises (conditional PP.H4 = 0.78 protein / 0.62 expression;
## see coloc_longevity_CTSF_*.R); the other three are weaker, directionally
## inconsistent across arms, and non-colocalising -> longevity-specific signal.
