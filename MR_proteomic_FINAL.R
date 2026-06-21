#!/usr/bin/env Rscript
# =====================================================================
# LD-aware, OVERLAP-aware MR  -  PROTEOMIC layer (UKB-PPP, 2947 proteins)
# UKB-PPP format: per-chromosome discovery_chr*.gz, cols:
#   CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P
#   -> ALLELE1 = effect allele; p = 10^(-LOG10P); match by CHR:POS (GRCh38)
# Hand-computed GLS IVW + LD-aware MR-Egger + Cochran's Q (no MR package).
# Real 1000G-EUR LD matrix + identity guard + ridge. Overlap-aware rho.
# Only reads chromosomes that contain instruments (5,9,12,15,17) -> faster.
# =====================================================================
suppressPackageStartupMessages({library(data.table);library(Matrix);library(MASS)})
toU<-function(x)toupper(as.character(x)); toL<-function(x)tolower(as.character(x)); num<-function(x)suppressWarnings(as.numeric(x))
logmsg<-function(...) cat(format(Sys.time(),"%H:%M:%S"),"-",...,"\n")

# ---------------- CONFIG (EDIT PATHS) ----------------
PROT_ROOT    <- "D:/"                                              # contains the 2947 protein folders
EXPOSURE_CSV <- "C:/Users/coach/Downloads/fine-mapped-pa-snps.csv"
INSTR_POS    <- "C:/Users/coach/Downloads/instruments_chrpos.csv"  # SNP,chr,pos,ref,alt (GRCh38)
LD_CSV       <- "C:/Users/coach/Downloads/GS_LD_square_REAL.csv"
OUTDIR       <- "C:/Users/coach/Downloads/LDaware_MR_proteins/mr_outputs_PPP_REAL"
RIDGE        <- 1e-3
RHO          <- 0.06        # conservative overlap correlation (UKBB exposure x UKB-PPP outcome)
N_EXP        <- 91184       # exposure GWAS N
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

# ---------------- EXPOSURE + INSTRUMENT POSITIONS ----------------
exp<-fread(EXPOSURE_CSV,check.names=FALSE)
ex_df<-data.frame(SNP=toL(exp$SNP),beta=num(exp$beta_pa),se=num(exp$se_pa),
  effect_allele=toU(exp$effect_allele),other_allele=toU(exp$other_allele),pval=num(exp$p_value))
ip<-fread(INSTR_POS); ip[,SNP:=toL(SNP)]; ip[,chr:=as.character(chr)]
ex_df<-merge(ex_df, ip[,.(SNP,chr,pos)], by="SNP")
need_chr<-sort(unique(ex_df$chr))
logmsg("Exposure SNPs:",nrow(ex_df)," on chromosomes:",paste(need_chr,collapse=","))

# ---------------- LD MATRIX (named, guarded) ----------------
ld<-fread(LD_CSV,check.names=FALSE); rn<-toL(ld[[1]])
LD<-as.matrix(type.convert(as.data.frame(ld[,-1,with=FALSE]),as.is=TRUE))
rownames(LD)<-rn; colnames(LD)<-rn; LD<-(LD+t(LD))/2; diag(LD)<-1
offd<-LD[upper.tri(LD)]
if (max(abs(offd))<1e-6) stop("LD matrix is identity. Wrong file - aborting.")
logmsg(sprintf("LD OK: %dx%d max|offdiag|=%.3f mean|offdiag|=%.3f",nrow(LD),ncol(LD),max(abs(offd)),mean(abs(offd))))

run_protein<-function(folder){
  gene<-sub("_.*","",basename(folder))
  oid <-basename(folder)
  # read only the chromosome files we need
  rows<-list()
  for (cc in need_chr){
    pat<-sprintf("discovery_chr%s_",cc)
    fl<-list.files(folder,pattern=paste0("^",pat),full.names=TRUE)
    fl<-fl[grepl("\\.gz$",fl)]
    if(!length(fl)) next
    d<-tryCatch(fread(fl[1],showProgress=FALSE,
        select=c("CHROM","GENPOS","ALLELE0","ALLELE1","BETA","SE","N","LOG10P")),
        error=function(e) NULL)
    if(is.null(d)||!nrow(d)) next
    inst_c<-ex_df[ex_df$chr==cc,]
    d2<-d[GENPOS %in% inst_c$pos]
    if(nrow(d2)) rows[[length(rows)+1]]<-d2
  }
  if(!length(rows)) return(NULL)
  pp<-rbindlist(rows,fill=TRUE)
  pp[,chr:=as.character(CHROM)]; pp[,pos:=GENPOS]
  pp[,p:=10^(-LOG10P)]
  # match to instruments by chr:pos
  m<-merge(ex_df, pp[,.(chr,pos,ALLELE0,ALLELE1,BETA,SE,N)], by=c("chr","pos"))
  if(nrow(m)<2) return(NULL)
  # ALLELE1 = effect allele in REGENIE; align to exposure effect allele
  eff_out<-toU(m$ALLELE1); oth_out<-toU(m$ALLELE0)
  flip<- m$effect_allele==oth_out & m$other_allele==eff_out
  keep<-(m$effect_allele==eff_out & m$other_allele==oth_out)|flip
  m<-m[keep,]; if(nrow(m)<2) return(NULL)
  by<-m$BETA; by[flip[keep]]<- -by[flip[keep]]
  byse<-m$SE; bx<-m$beta; bxse<-m$se
  # palindromic drop
  pal<-(m$effect_allele%in%c("A","T")&m$other_allele%in%c("A","T"))|(m$effect_allele%in%c("C","G")&m$other_allele%in%c("C","G"))
  m<-m[!pal,]; by<-by[!pal]; byse<-byse[!pal]; bx<-bx[!pal]; bxse<-bxse[!pal]
  if(nrow(m)<2) return(NULL)
  common<-intersect(toL(m$SNP),rownames(LD)); if(length(common)<2) return(NULL)
  idx<-toL(m$SNP)%in%common
  m<-m[idx,]; by<-by[idx]; byse<-byse[idx]; bx<-bx[idx]; bxse<-bxse[idx]
  R<-LD[toL(m$SNP),toL(m$SNP),drop=FALSE]; R<-(R+t(R))/2; diag(R)<-1+RIDGE
  # overlap-aware GLS
  Dout<-diag(byse); Dexp<-diag(bxse)
  Sigma<-Dout%*%R%*%Dout + (RHO^2)*(Dexp%*%R%*%Dexp)
  Si<-tryCatch(solve(Sigma),error=function(e) MASS::ginv(Sigma))
  XtSiX<-as.numeric(t(bx)%*%Si%*%bx)
  b_ivw<-as.numeric(t(bx)%*%Si%*%by)/XtSiX
  se_ivw<-sqrt(1/XtSiX); p_ivw<-2*pnorm(-abs(b_ivw/se_ivw))
  r<-by-b_ivw*bx; Q<-as.numeric(t(r)%*%Si%*%r); Qdf<-length(bx)-1; Qp<-pchisq(Q,Qdf,lower.tail=FALSE)
  X<-cbind(1,bx); XtX<-t(X)%*%Si%*%X; XtXi<-tryCatch(solve(XtX),error=function(e) MASS::ginv(as.matrix(XtX)))
  co<-as.numeric(XtXi%*%t(X)%*%Si%*%by)
  vdiag<-diag(XtXi); vdiag[vdiag<=0]<-NA            # guard: negative variance -> NA, no NaN warning
  ses<-sqrt(vdiag)
  egg_int<-co[1]
  egg_int_p<-if(is.na(ses[1])) NA_real_ else 2*pnorm(-abs(co[1]/ses[1]))
  data.table(gene=gene,protein=oid,nsnp=nrow(m),
    beta=b_ivw,se=se_ivw,pval=p_ivw,Q=Q,Q_df=Qdf,Q_pval=Qp,
    egger_intercept=egg_int,egger_intercept_p=egg_int_p)
}

# ---------------- LOOP OVER PROTEINS ----------------
folders<-list.dirs(PROT_ROOT,recursive=FALSE)
folders<-folders[grepl("_OID[0-9]+_",basename(folders))]   # skip junk dirs
logmsg("Protein folders:",length(folders))

res<-vector("list",length(folders))
for(i in seq_along(folders)){
  res[[i]]<-tryCatch(run_protein(folders[i]),error=function(e) NULL)
  if(i%%100==0||i==length(folders)) logmsg(sprintf("...%d/%d",i,length(folders)))
}
all_res<-rbindlist(res,fill=TRUE)
if(!nrow(all_res)){logmsg("No results.");quit(save="no",status=1)}
all_res[,fdr:=p.adjust(pval,"BH")]
fout<-file.path(OUTDIR,"PPP_PA_LDaware_REAL_summary.tsv")
fwrite(all_res,fout,sep="\t")
logmsg("WROTE:",fout)
logmsg(sprintf("Proteins: %d | nominal p<0.05: %d (%.1f%%) | FDR<0.05: %d (%.1f%%)",
  nrow(all_res),sum(all_res$pval<0.05,na.rm=TRUE),100*mean(all_res$pval<0.05,na.rm=TRUE),
  sum(all_res$fdr<0.05,na.rm=TRUE),100*mean(all_res$fdr<0.05,na.rm=TRUE)))
logmsg(sprintf("High heterogeneity (Q_pval<0.05): %d | sig Egger intercept: %d",
  sum(all_res$Q_pval<0.05,na.rm=TRUE),sum(all_res$egger_intercept_p<0.05,na.rm=TRUE)))
