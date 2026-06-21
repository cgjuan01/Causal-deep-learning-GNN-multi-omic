import argparse,os,random,numpy as np,pandas as pd,torch
import torch.nn as nn,torch.nn.functional as F
MR=["protein_std","cpg_std","glycan_std","sc_std","transcript_std"]
def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def z(x): m,s=np.nanmean(x,0),np.nanstd(x,0);s[s==0]=1;return (x-m)/s
class MLP(nn.Module):
    def __init__(s,d,h=64,o=32,p=0.2):
        super().__init__();s.l1=nn.Linear(d,h);s.l2=nn.Linear(h,o)
        s.reg=nn.Linear(o,1);s.cls=nn.Linear(o,1);s.p=p
    def forward(s,x):
        h=F.elu(s.l1(x));h=F.dropout(h,p=s.p,training=s.training);h=F.elu(s.l2(h))
        return s.reg(h).squeeze(-1),s.cls(h).squeeze(-1)
def run_seed(nodes,sd):
    seed(sd)
    x=torch.tensor(np.nan_to_num(z(nodes[MR].to_numpy(float)),nan=0.0),dtype=torch.float)
    y=torch.tensor(nodes["MTI_score"].to_numpy(float),dtype=torch.float);yf=torch.isfinite(y)
    ml=torch.tensor((nodes["MTI_n_layers"].to_numpy()>=2).astype(float),dtype=torch.float)
    m=MLP(x.shape[1]);opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    npos=float(ml.sum());pw=torch.tensor(min((len(ml)-npos)/max(npos,1),50.0))
    m.train()
    for ep in range(300):
        opt.zero_grad();r,lg=m(x)
        loss=F.mse_loss(r[yf],y[yf])+0.3*F.binary_cross_entropy_with_logits(lg,ml,pos_weight=pw)
        loss.backward();opt.step()
    m.eval()
    with torch.no_grad(): r,lg=m(x);pred=r.numpy();prob=torch.sigmoid(lg).numpy()
    def zz(v): v=np.asarray(v,float);sd2=v.std();return (v-v.mean())/(sd2 if sd2>0 else 1)
    return pd.Series(zz(pred)+zz(prob),index=nodes["gene_symbol"]).sort_values(ascending=False)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--node_path");a=ap.parse_args()
    nodes=pd.read_csv(a.node_path,sep="\t");nodes["gene_symbol"]=nodes["gene_symbol"].astype(str).str.upper()
    for sd in [0,1,7,42,123]:
        r=run_seed(nodes,sd);r.to_csv("rank_MLP_seed"+str(sd)+".tsv",sep="\t");print("MLP seed",sd,"done")

