#!/usr/bin/env Rscript
# =====================================================================
# CLUMPED LD-aware MR  -  EPIGENOMIC / CpG layer (GoDMC)
# Instruments CLUMPED to independent leads (r2<0.5 -> ~6 leads), then LD-aware MR
# on whatever leads are in cis to each CpG:
#    >=2 leads -> full LD-aware IVW + MR-Egger + Cochran's Q
#     1 lead   -> Wald ratio (flagged single_instrument=TRUE)
# Produces VALID, stable betas (well-conditioned matrix; no singular-matrix blowups).
# MR here is a hypothesis-generating PRIOR for the network layer, not a standalone causal claim.
# =====================================================================
suppressPackageStartupMessages({library(data.table);library(Matrix);library(MASS)})
toL<-function(x)tolower(as.character(x)); toU<-function(x)toupper(as.character(x)); num<-function(x)suppressWarnings(as.numeric(x))
logmsg<-function(...) cat(format(Sys.time(),"%H:%M:%S"),"-",...,"\n")

# ---------------- CONFIG ----------------
GODMC_FILE   <- "C:/Users/coach/Downloads/assoc_meta_all.csv.gz"
SNPS_BRIDGE  <- "C:/Users/coach/Downloads/snps.csv.gz"
CPG_TARGETS  <- "C:/Users/coach/Downloads/PA_SNPs_matched_CpGs.csv"
EXPOSURE_CSV <- "C:/Users/coach/Downloads/fine-mapped-pa-snps.csv"
LD_CSV       <- "C:/Users/coach/Downloads/GS_LD_square_REAL.csv"
OUTDIR       <- "C:/Users/coach/Downloads/LDaware_MR_proteins/mr_outputs_CPG_CLUMPED"
R2_CLUMP     <- 0.5     # -> ~6 independent leads, condition number ~5 (stable)
RIDGE        <- 1e-4    # tiny, just for safety; matrix is already well-conditioned
CHUNK        <- 1e6
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

# ---------------- EXPOSURE ----------------
exp<-fread(EXPOSURE_CSV,check.names=FALSE)
ex_df<-data.table(SNP=toL(exp$SNP),beta_exp=num(exp$beta_pa),se_exp=num(exp$se_pa),
  ea_exp=toU(exp$effect_allele),oa_exp=toU(exp$other_allele),pval=num(exp$p_value))
ex_df<-ex_df[SNP!="." & SNP!="" & !is.na(SNP)]
logmsg("Exposure SNPs:",nrow(ex_df))

# ---------------- LD MATRIX ----------------
ld<-fread(LD_CSV,check.names=FALSE); rn<-toL(ld[[1]])
LD<-as.matrix(type.convert(as.data.frame(ld[,-1,with=FALSE]),as.is=TRUE))
rownames(LD)<-rn; colnames(LD)<-rn; LD<-(LD+t(LD))/2; diag(LD)<-1
if (max(abs(LD[upper.tri(LD)]))<1e-6) stop("LD matrix is identity - aborting.")

# ---------------- CLUMP to independent leads ----------------
clump<-function(snps, pvals, R, r2=R2_CLUMP){
  snps<-snps[snps %in% rownames(R)]
  dt<-data.table(snp=snps, p=pvals[match(snps, snps)]); setorder(dt,p); ord<-dt$snp
  leads<-character(0); remaining<-ord
  while(length(remaining)){
    lead<-remaining[1]; leads<-c(leads,lead)
    rr<-R[lead, remaining]^2; remaining<-remaining[rr<r2]; remaining<-setdiff(remaining,lead)
  }
  leads
}
leads<-clump(ex_df$SNP, ex_df$pval, LD)
logmsg(sprintf("Clumped %d instruments -> %d independent leads (r2<%.1f): %s",
  nrow(ex_df), length(leads), R2_CLUMP, paste(leads,collapse=", ")))
ex_df<-ex_df[SNP %in% leads]                     # keep only leads as instruments
LDl<-LD[leads,leads,drop=FALSE]
logmsg(sprintf("Lead LD condition number: %.1f (stable if <1000)", kappa(LDl)))

# ---------------- TARGETS + BRIDGE ----------------
tg<-fread(CPG_TARGETS); target_cpgs<-unique(tg$cpg)
br<-fread(SNPS_BRIDGE, select=c("name","rsid")); br[,rsid:=toL(rsid)]
br<-br[rsid %in% leads]                            # bridge only for our LEADS now
name2rsid<-setNames(br$rsid, br$name); godmc_names<-br$name
logmsg("Lead SNPs matched in bridge:",length(godmc_names))

# ---------------- STREAM GoDMC (quote-safe fread, chunked) ----------------
hdr<-names(fread(GODMC_FILE, nrows=0))
keep_list<-list(); skip<-1L; total<-0L
repeat{
  ch<-tryCatch(fread(GODMC_FILE, skip=skip, nrows=CHUNK, header=FALSE, col.names=hdr, showProgress=FALSE),
               error=function(e) NULL)
  if(is.null(ch)||nrow(ch)==0) break
  total<-total+nrow(ch); short<-nrow(ch)<CHUNK
  k<-(ch$snp %in% godmc_names) & (ch$cpg %in% target_cpgs)
  if(any(k)) keep_list[[length(keep_list)+1]]<-ch[k,.(cpg,snp,beta_a1,se,allele1,allele2)]
  nkept<-sum(vapply(keep_list,nrow,integer(1))); rm(ch,k); gc(verbose=FALSE)
  if(total %% 1e7 < CHUNK) logmsg(sprintf("  read %d lines, kept %d rows", total, nkept))
  if(short) break
  skip<-skip+CHUNK
}
god<-rbindlist(keep_list,fill=TRUE)
if(!nrow(god)){logmsg("No matching rows."); quit(save="no",status=1)}
god[,rsid:=name2rsid[snp]]
logmsg("GoDMC rows kept:",nrow(god)," across",length(unique(god$cpg)),"CpGs")

# ---------------- estimators ----------------
fit_multi<-function(bx,bxse,by,byse,R,ridge=RIDGE){
  R<-(R+t(R))/2; diag(R)<-1+ridge
  D<-diag(byse); Si<-tryCatch(solve(D%*%R%*%D),error=function(e) MASS::ginv(D%*%R%*%D))
  XtSiX<-as.numeric(t(bx)%*%Si%*%bx)
  b<-as.numeric(t(bx)%*%Si%*%by)/XtSiX; se<-sqrt(1/XtSiX); p<-2*pnorm(-abs(b/se))
  r<-by-b*bx; Q<-as.numeric(t(r)%*%Si%*%r); Qdf<-length(bx)-1; Qp<-pchisq(Q,Qdf,lower.tail=FALSE)
  X<-cbind(1,bx); XtXi<-tryCatch(solve(t(X)%*%Si%*%X),error=function(e) MASS::ginv(as.matrix(t(X)%*%Si%*%X)))
  co<-as.numeric(XtXi%*%t(X)%*%Si%*%by); vd<-diag(XtXi); vd[vd<=0]<-NA; ses<-sqrt(vd)
  list(b=b,se=se,p=p,Q=Q,Qdf=Qdf,Qp=Qp,ei=co[1],eip=if(is.na(ses[1])) NA_real_ else 2*pnorm(-abs(co[1]/ses[1])))
}
wald<-function(bx,by,byse){            # single-instrument Wald ratio
  b<-by/bx; se<-abs(byse/bx); list(b=b,se=se,p=2*pnorm(-abs(b/se)))
}

run_cpg<-function(cg){
  d<-god[cpg==cg]; d<-d[!duplicated(rsid) & rsid %in% leads]
  if(nrow(d)<1) return(NULL)
  m<-merge(ex_df,d,by.x="SNP",by.y="rsid")
  if(nrow(m)<1) return(NULL)
  a1<-toU(m$allele1); a2<-toU(m$allele2)
  flip<- m$ea_exp==a2 & m$oa_exp==a1
  keep<-(m$ea_exp==a1 & m$oa_exp==a2)|flip
  m<-m[keep]; if(nrow(m)<1) return(NULL)
  by<-m$beta_a1; fl<-flip[keep]; by[fl]<- -by[fl]
  pal<-(m$ea_exp%in%c("A","T")&m$oa_exp%in%c("A","T"))|(m$ea_exp%in%c("C","G")&m$oa_exp%in%c("C","G"))
  m<-m[!pal]; by<-by[!pal]; if(nrow(m)<1) return(NULL)
  if(nrow(m)==1){
    w<-wald(m$beta_exp, by, m$se)
    return(data.table(cpg=cg,nsnp=1L,single_instrument=TRUE,beta=w$b,se=w$se,pval=w$p,
      Q=NA_real_,Q_pval=NA_real_,egger_intercept=NA_real_,egger_intercept_p=NA_real_))
  }
  R<-LD[toL(m$SNP),toL(m$SNP),drop=FALSE]
  ft<-tryCatch(fit_multi(m$beta_exp,m$se_exp,by,m$se,R),error=function(e) NULL)
  if(is.null(ft)) return(NULL)
  data.table(cpg=cg,nsnp=nrow(m),single_instrument=FALSE,beta=ft$b,se=ft$se,pval=ft$p,
    Q=ft$Q,Q_pval=ft$Qp,egger_intercept=ft$ei,egger_intercept_p=ft$eip)
}

cpgs<-unique(god$cpg)
logmsg("Running clumped MR on",length(cpgs),"CpGs...")
res<-rbindlist(lapply(seq_along(cpgs),function(i){
  if(i%%200==1||i==length(cpgs)) logmsg(sprintf("...%d/%d",i,length(cpgs)))
  tryCatch(run_cpg(cpgs[i]),error=function(e) NULL)
}),fill=TRUE)
if(!nrow(res)){logmsg("No results.");quit(save="no",status=1)}
res[,fdr:=p.adjust(pval,"BH")]
fout<-file.path(OUTDIR,"CpG_PA_CLUMPED_summary.tsv")
fwrite(res,fout,sep="\t")
logmsg("WROTE:",fout)
logmsg(sprintf("CpGs: %d | multi-instrument: %d | single-instrument(Wald): %d",
  nrow(res),sum(!res$single_instrument),sum(res$single_instrument)))
logmsg(sprintf("nominal p<0.05: %d (%.1f%%) | FDR<0.05: %d (%.1f%%)",
  sum(res$pval<0.05,na.rm=TRUE),100*mean(res$pval<0.05,na.rm=TRUE),
  sum(res$fdr<0.05,na.rm=TRUE),100*mean(res$fdr<0.05,na.rm=TRUE)))
logmsg(sprintf("beta range: [%.3f, %.3f]  (sane if within ~ +/-1; no 12s = singularity fixed)",
  min(res$beta,na.rm=TRUE),max(res$beta,na.rm=TRUE)))
