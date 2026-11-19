#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,torch
from train_risk import Data
from risk import RiskNet
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('em_opm/data/risk_v1'));ap.add_argument('--checkpoint',type=Path,default=Path('em_opm/runs/risk_v1/best.pt'));ap.add_argument('--output',type=Path,default=Path('em_opm/runs/risk_v1/test_metrics.json'));a=ap.parse_args();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=RiskNet().to(dev);m.load_state_dict(torch.load(a.checkpoint,map_location=dev,weights_only=True)['model']);m.eval();ds=Data(a.data,[7]);p=[];y=[]
 with torch.no_grad():
  for i in range(0,len(ds),256):
   q=[ds[j] for j in range(i,min(i+256,len(ds)))];b,s,g,t,z=[torch.stack([x[k] for x in q]).to(dev) for k in range(5)];p.extend(torch.sigmoid(m(b,s,g,t)).cpu().numpy()[:,0]);y.extend(z.cpu().numpy()[:,0])
 p=np.asarray(p);y=np.asarray(y);bins=np.linspace(0,1,11);ece=0
 for lo,hi in zip(bins[:-1],bins[1:]):
  q=(p>=lo)&(p<(hi if hi<1 else hi+1e-6));ece+=q.mean()*abs(p[q].mean()-y[q].mean()) if q.any() else 0
 order=np.argsort(p);ranks=np.empty(len(p),float);ranks[order]=np.arange(1,len(p)+1);npos=y.sum();nneg=len(y)-npos;auc=(ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg);order=np.argsort(-p);ys=y[order];precision=np.cumsum(ys)/np.arange(1,len(y)+1);ap=(precision*ys).sum()/npos
 out={'test_sequence':'M0S7','pairs':len(y),'positive_rate':float(y.mean()),'AUROC':float(auc),'AUPRC':float(ap),'Brier':float(np.mean((p-y)**2)),'ECE_10':float(ece)};a.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
