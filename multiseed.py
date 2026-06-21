import argparse,os,random,numpy as np,pandas as pd,torch
import torch.nn as nn,torch.nn.functional as F
from torch_geometric.nn import GATConv
MR=["protein_std","cpg_std","glycan_std","sc_std","transcript_std"]
def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def z(x): m,s=np.nanmean(x,0),np.nanstd(x,0);s[s==0]=1;return (x-m)/s
class GAT(nn.Module):
    def __init__(s,d,h=64,hd=4,o=32,p=0.2):
        super().__init__();s.g1=GATConv(d,h,heads=hd,dropout=p);s.g2=GATConv(h*hd,o,heads=1,dropout=p)
        s.reg=nn.Linear(o,1);s.cls=nn.Linear(o,1);s.p=p
    def forward(s,x,ei):
        h=F.elu(s.g1(x,ei));h=F.dropout(h,p=s.p,training=s.training);h=s.g2(h,ei)
        return s.reg(h).squeeze(-1),s.cls(h).squeeze(-1)
def run_seed(nodes,edges,sd):
    seed(sd)
    idx={g:i for i,g in enumerate(nodes["gene_symbol"])}
    x=torch.tensor(np.nan_to_num(z(nodes[MR].to_numpy(float)),nan=0.0),dtype=torch.float)
    y=torch.tensor(nodes["MTI_score"].to_numpy(float),dtype=torch.float);yf=torch.isfinite(y)
    ml=torch.tensor((nodes["MTI_n_layers"].to_numpy()>=2).astype(float),dtype=torch.float)
    ec=list(edges.columns)[:2];e=edges[(edges[ec[0]].isin(idx))&(edges[ec[1]].isin(idx))]
    s_=e[ec[0]].map(idx).to_numpy();d_=e[ec[1]].map(idx).to_numpy()
    ei=torch.tensor(np.vstack([np.concatenate([s_,d_]),np.concatenate([d_,s_])]),dtype=torch.long)
    m=GAT(x.shape[1]);opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    npos=float(ml.sum());pw=torch.tensor(min((len(ml)-npos)/max(npos,1),50.0))
    m.train()
    for ep in range(300):
        opt.zero_grad();r,lg=m(x,ei)
        loss=F.mse_loss(r[yf],y[yf])+0.3*F.binary_cross_entropy_with_logits(lg,ml,pos_weight=pw)
        loss.backward();opt.step()
    m.eval()
    with torch.no_grad(): r,lg=m(x,ei);pred=r.numpy();prob=torch.sigmoid(lg).numpy()
    def zz(v): v=np.asarray(v,float);sd2=v.std();return (v-v.mean())/(sd2 if sd2>0 else 1)
    hyb=zz(pred)+zz(prob)
    return pd.Series(hyb,index=nodes["gene_symbol"]).sort_values(ascending=False)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--node_path");ap.add_argument("--edge_path")
    a=ap.parse_args()
    nodes=pd.read_csv(a.node_path,sep="\t");nodes["gene_symbol"]=nodes["gene_symbol"].astype(str).str.upper()
    edges=pd.read_csv(a.edge_path,sep="\t")
    for c in list(edges.columns)[:2]: edges[c]=edges[c].astype(str).str.upper()
    for sd in [0,1,7,42,123]:
        r=run_seed(nodes,edges,sd)
        r.to_csv("rank_GAT_seed"+str(sd)+".tsv",sep="\t")
        print("seed",sd,"done")

