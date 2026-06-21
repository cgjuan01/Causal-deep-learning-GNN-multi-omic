#!/usr/bin/env Rscript
# =====================================================================
# LD-aware MR  -  GLYCOMIC layer (TPNG GWAMA EUR 10k, Zenodo 15057709)
# Hand-computed GLS IVW + LD-aware MR-Egger + Cochran's Q  (no MendelianRandomization pkg)
# Real 1000G-EUR LD matrix (GS_LD_square_REAL.csv) with identity guard + ridge.
# FAST: extracts the archive ONCE to a folder, then reads loose CSVs.
# TPNG = independent cohort (not UKBB) -> no sample overlap -> rho = 0.
# =====================================================================
suppressPackageStartupMessages({library(data.table);library(Matrix);library(MASS)})
toU<-function(x)toupper(as.character(x)); toL<-function(x)tolower(as.character(x)); num<-function(x)suppressWarnings(as.numeric(x))
logmsg<-function(...) cat(format(Sys.time(),"%H:%M:%S"),"-",...,"\n")

# ---------------- CONFIG (EDIT PATHS) ----------------
ARCHIVE      <- "C:/Users/coach/Downloads/TPNG_GWAMA_eur_10k.tar.gz"
EXTRACT_DIR  <- "C:/Users/coach/Downloads/TPNG_extracted"          # archive unpacked here once
EXPOSURE_CSV <- "C:/Users/coach/Downloads/fine-mapped-pa-snps.csv"
LD_CSV       <- "C:/Users/coach/Downloads/GS_LD_square_REAL.csv"
TRAIT_MAP    <- "C:/Users/coach/Downloads/glycan_trait_map.csv"
OUTDIR       <- "C:/Users/coach/Downloads/LDaware_MR_proteins/mr_outputs_glyco_REAL"
RIDGE        <- 1e-3   # stabilises near-singular LD (17q21 block r~0.96); report in methods
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

# ---------------- EXPOSURE ----------------
exp<-fread(EXPOSURE_CSV,check.names=FALSE)
ex_df<-data.frame(SNP=toL(exp$SNP),beta=num(exp$beta_pa),se=num(exp$se_pa),
  effect_allele=toU(exp$effect_allele),other_allele=toU(exp$other_allele),
  eaf=num(exp$eaf),pval=num(exp$p_value))
exp_snps<-ex_df$SNP
logmsg("Exposure SNPs:",nrow(ex_df))

# ---------------- LD MATRIX (named, guarded) ----------------
ld<-fread(LD_CSV,check.names=FALSE); rn<-toL(ld[[1]])
LD<-as.matrix(type.convert(as.data.frame(ld[,-1,with=FALSE]),as.is=TRUE))
rownames(LD)<-rn; colnames(LD)<-rn; LD<-(LD+t(LD))/2; diag(LD)<-1
offd<-LD[upper.tri(LD)]
if (max(abs(offd))<1e-6) stop("LD matrix is identity. Wrong file - aborting.")
logmsg(sprintf("LD OK: %dx%d max|offdiag|=%.3f mean|offdiag|=%.3f",nrow(LD),ncol(LD),max(abs(offd)),mean(abs(offd))))

# ---------------- EXTRACT ARCHIVE ONCE (the speed-up) ----------------
if (!dir.exists(EXTRACT_DIR) || length(list.files(EXTRACT_DIR,pattern="_done\\.csv$",recursive=TRUE))==0){
  logmsg("Extracting archive once to", EXTRACT_DIR, "(one-time, slow)...")
  dir.create(EXTRACT_DIR, showWarnings=FALSE, recursive=TRUE)
  utils::untar(ARCHIVE, exdir=EXTRACT_DIR)
  logmsg("Extraction done.")
} else logmsg("Using already-extracted files in", EXTRACT_DIR)

files <- list.files(EXTRACT_DIR, pattern="pgp[0-9]+.*_done\\.csv$", recursive=TRUE, full.names=TRUE)
logmsg("Found", length(files), "glycan files")

# ---------------- LD-aware estimators (hand-computed) ----------------
fit_ld <- function(bx, by, byse, R, ridge=RIDGE){
  R<-(R+t(R))/2; diag(R)<-1+ridge
  D<-diag(byse); Sigma<-D%*%R%*%D
  Si<-tryCatch(solve(Sigma),error=function(e) MASS::ginv(Sigma))
  # --- IVW (no intercept) ---
  XtSiX<-as.numeric(t(bx)%*%Si%*%bx)
  b_ivw<-as.numeric(t(bx)%*%Si%*%by)/XtSiX
  se_ivw<-sqrt(1/XtSiX); p_ivw<-2*pnorm(-abs(b_ivw/se_ivw))
  # --- Cochran's Q (heterogeneity about IVW slope) ---
  r<-by-b_ivw*bx; Q<-as.numeric(t(r)%*%Si%*%r); Qdf<-length(bx)-1; Qp<-pchisq(Q,Qdf,lower.tail=FALSE)
  # --- MR-Egger (GLS with intercept): [intercept, slope] ---
  X<-cbind(1,bx)
  XtSiX2<-t(X)%*%Si%*%X
  XtSiXi<-tryCatch(solve(XtSiX2),error=function(e) MASS::ginv(as.matrix(XtSiX2)))
  coef<-as.numeric(XtSiXi%*%t(X)%*%Si%*%by)
  ses <-sqrt(diag(XtSiXi))
  egg_int<-coef[1]; egg_int_se<-ses[1]; egg_int_p<-2*pnorm(-abs(egg_int/egg_int_se))
  egg_slope<-coef[2]; egg_slope_se<-ses[2]; egg_slope_p<-2*pnorm(-abs(egg_slope/egg_slope_se))
  list(b_ivw=b_ivw,se_ivw=se_ivw,p_ivw=p_ivw,Q=Q,Qdf=Qdf,Qp=Qp,
       egg_int=egg_int,egg_int_se=egg_int_se,egg_int_p=egg_int_p,
       egg_slope=egg_slope,egg_slope_se=egg_slope_se,egg_slope_p=egg_slope_p,nsnp=length(bx))
}

run_one<-function(f){
  outname<-sub("_done\\.csv$","",basename(f))
  dt<-tryCatch(fread(f,showProgress=FALSE),error=function(e) NULL)
  if(is.null(dt)||!nrow(dt)) return(NULL)
  names(dt)<-tolower(names(dt))
  out<-data.frame(SNP=toL(dt$rs_id),beta=num(dt$beta),se=num(dt$se),
    effect_allele=toU(dt$ea),other_allele=toU(dt$ra),pval=num(dt$p))
  out<-out[!duplicated(out$SNP)&out$SNP%in%exp_snps,]
  if(nrow(out)<2) return(NULL)
  m<-merge(ex_df,out,by="SNP",suffixes=c(".exp",".out"))
  flip<-m$effect_allele.exp==m$other_allele.out & m$other_allele.exp==m$effect_allele.out
  keep<-(m$effect_allele.exp==m$effect_allele.out & m$other_allele.exp==m$other_allele.out)|flip
  m<-m[keep,]; if(nrow(m)<2) return(NULL); m$beta.out[flip[keep]]<- -m$beta.out[flip[keep]]
  pal<-(m$effect_allele.exp%in%c("A","T")&m$other_allele.exp%in%c("A","T"))|(m$effect_allele.exp%in%c("C","G")&m$other_allele.exp%in%c("C","G"))
  m<-m[!pal,]; if(nrow(m)<2) return(NULL)
  common<-intersect(m$SNP,rownames(LD)); if(length(common)<2) return(NULL)
  m<-m[m$SNP%in%common,]; R<-LD[m$SNP,m$SNP,drop=FALSE]
  ft<-tryCatch(fit_ld(m$beta.exp,m$beta.out,m$se.out,R),error=function(e) NULL)
  if(is.null(ft)) return(NULL)
  data.table(outcome=outname,nsnp=ft$nsnp,
    beta=ft$b_ivw,se=ft$se_ivw,pval=ft$p_ivw,
    Q=ft$Q,Q_df=ft$Qdf,Q_pval=ft$Qp,
    egger_intercept=ft$egg_int,egger_intercept_se=ft$egg_int_se,egger_intercept_p=ft$egg_int_p,
    egger_slope=ft$egg_slope,egger_slope_p=ft$egg_slope_p)
}

all_res<-rbindlist(lapply(seq_along(files),function(i){
  if(i%%10==1||i==length(files)) logmsg(sprintf("...%d/%d %s",i,length(files),basename(files[i])))
  tryCatch(run_one(files[i]),error=function(e) NULL)
}),fill=TRUE)

if(!nrow(all_res)){logmsg("No results.");quit(save="no",status=1)}

# attach trait names
tm<-tryCatch(fread(TRAIT_MAP),error=function(e) NULL)
if(!is.null(tm)){all_res[,pgp_id:=tolower(outcome)]; all_res<-merge(all_res,tm,by="pgp_id",all.x=TRUE,sort=FALSE)}
all_res[,`:=`(ci_low=beta-1.96*se,ci_high=beta+1.96*se,fdr=p.adjust(pval,"BH"))]

fout<-file.path(OUTDIR,"TPNG_PA_LDaware_REAL_summary.tsv")
fwrite(all_res,fout,sep="\t")
logmsg("WROTE:",fout)
logmsg(sprintf("Glycans: %d | nominal p<0.05: %d (%.1f%%) | FDR<0.05: %d (%.1f%%)",
  nrow(all_res),sum(all_res$pval<0.05,na.rm=TRUE),100*mean(all_res$pval<0.05,na.rm=TRUE),
  sum(all_res$fdr<0.05,na.rm=TRUE),100*mean(all_res$fdr<0.05,na.rm=TRUE)))
logmsg(sprintf("High heterogeneity (Q_pval<0.05): %d of %d glycans",
  sum(all_res$Q_pval<0.05,na.rm=TRUE),nrow(all_res)))
