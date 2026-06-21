import argparse, os, random
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv

MR_FEATS=["protein_std","cpg_std","glycan_std","sc_std","transcript_std"]
def seed(s=42): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def z(x): m,s=np.nanmean(x,0),np.nanstd(x,0); s[s==0]=1; return (x-m)/s

class GAT(nn.Module):
    def __init__(s,d,h=64,heads=4,o=32,p=0.2):
        super().__init__(); s.g1=GATConv(d,h,heads=heads,dropout=p); s.g2=GATConv(h*heads,o,heads=1,dropout=p)
        s.reg=nn.Linear(o,1); s.cls=nn.Linear(o,1); s.p=p
    def forward(s,x,ei):
        h=F.elu(s.g1(x,ei)); h=F.dropout(h,p=s.p,training=s.training); h=s.g2(h,ei)
        return s.reg(h).squeeze(-1), s.cls(h).squeeze(-1)

class MLP(nn.Module):
    def __init__(s,d,h=64,o=32,p=0.2):
        super().__init__(); s.l1=nn.Linear(d,h); s.l2=nn.Linear(h,o)
        s.reg=nn.Linear(o,1); s.cls=nn.Linear(o,1); s.p=p
    def forward(s,x,ei=None):
        h=F.elu(s.l1(x)); h=F.dropout(h,p=s.p,training=s.training); h=F.elu(s.l2(h))
        return s.reg(h).squeeze(-1), s.cls(h).squeeze(-1)

def run(nodes,edges,mt,out):
    seed(42)
    idx={g:i for i,g in enumerate(nodes["gene_symbol"])}
    feats=[c for c in MR_FEATS if c in nodes.columns]
    x=torch.tensor(np.nan_to_num(z(nodes[feats].to_numpy(dtype=float)),nan=0.0),dtype=torch.float)
    y=torch.tensor(nodes["MTI_score"].to_numpy(dtype=float),dtype=torch.float); yf=torch.isfinite(y)
    multi=torch.tensor((nodes["MTI_n_layers"].to_numpy()>=2).astype(float),dtype=torch.float)
    ec=list(edges.columns)[:2]
    e=edges[(edges[ec[0]].isin(idx))&(edges[ec[1]].isin(idx))]
    s_=e[ec[0]].map(idx).to_numpy(); d_=e[ec[1]].map(idx).to_numpy()
    ei=torch.tensor(np.vstack([np.concatenate([s_,d_]),np.concatenate([d_,s_])]),dtype=torch.long)
    model=GAT(x.shape[1]) if mt=="GAT" else MLP(x.shape[1])
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    npos=float(multi.sum()); pw=torch.tensor(min((len(multi)-npos)/max(npos,1),50.0))
    model.train()
    for ep in range(300):
        opt.zero_grad(); reg,logit=model(x,ei)
        loss=F.mse_loss(reg[yf],y[yf])+0.3*F.binary_cross_entropy_with_logits(logit,multi,pos_weight=pw)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): reg,logit=model(x,ei); pred=reg.numpy(); prob=torch.sigmoid(logit).numpy()
    def zz(v): v=np.asarray(v,float); sd=v.std(); return (v-v.mean())/(sd if sd>0 else 1)
    res=pd.DataFrame({"gene_symbol":nodes["gene_symbol"],"MTI_score":nodes["MTI_score"],
        "MTI_n_layers":nodes["MTI_n_layers"],"pred_MTI":pred,"hybrid_score":zz(pred)+zz(prob)})
    res["rank_hybrid"]=res["hybrid_score"].rank(ascending=False,method="first").astype(int)
    res=res.sort_values("rank_hybrid")
    res.to_csv(os.path.join(out,"rank_"+mt+"_MRbetas.tsv"),sep="\t",index=False)
    print(mt,"loss=%.4f saved rank_"%loss.item()+mt+"_MRbetas.tsv")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--node_path",required=True)
    ap.add_argument("--edge_path",required=True); ap.add_argument("--out_dir",default="."); a=ap.parse_args()
    nodes=pd.read_csv(a.node_path,sep="\t"); nodes["gene_symbol"]=nodes["gene_symbol"].astype(str).str.upper()
    edges=pd.read_csv(a.edge_path,sep="\t")
    for c in list(edges.columns)[:2]: edges[c]=edges[c].astype(str).str.upper()
    run(nodes,edges,"GAT",a.out_dir); run(nodes,edges,"MLP",a.out_dir)

