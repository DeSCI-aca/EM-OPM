#!/usr/bin/env python3
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

class CollisionNet(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(20,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
 def forward(self,c,a):return self.net(torch.cat([c.flatten(1),a.flatten(1)],1))[:,0]
def data(path):
 z=np.load(path);return TensorDataset(torch.from_numpy(z['candidates']),torch.from_numpy(z['actors']),torch.from_numpy(z['labels']))
def metrics(p,y):
 bins=np.linspace(0,1,11);ece=0.
 for lo,hi in zip(bins[:-1],bins[1:]):
  q=(p>=lo)&(p<(hi if hi<1 else hi+1e-8));ece+=q.mean()*abs(p[q].mean()-y[q].mean()) if q.any() else 0
 order=np.argsort(p);ranks=np.empty(len(p),float);ranks[order]=np.arange(1,len(p)+1);npos=y.sum();nneg=len(y)-npos;auc=(ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg)
 order=np.argsort(-p);ys=y[order];precision=np.cumsum(ys)/np.arange(1,len(y)+1);ap=(precision*ys).sum()/npos
 return {'pairs':len(y),'positive_rate':float(y.mean()),'AUROC':float(auc),'AUPRC':float(ap),'Brier':float(np.mean((p-y)**2)),'ECE_10':float(ece)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('em_opm/data/collision_v2'));ap.add_argument('--output',type=Path,default=Path('em_opm/runs/collision_v2'));ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--batch-size',type=int,default=1024);ap.add_argument('--seed',type=int,default=31);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tr,va,te=[data(a.data/f'{x}.npz') for x in ('train','val','test')];tl=DataLoader(tr,a.batch_size,shuffle=True,num_workers=2,pin_memory=True);vl=DataLoader(va,a.batch_size*2);m=CollisionNet().to(dev);y=np.load(a.data/'train.npz')['labels'];pw=torch.tensor((len(y)-y.sum())/max(y.sum(),1),device=dev);lossfn=nn.BCEWithLogitsLoss(pos_weight=pw);opt=torch.optim.AdamW(m.parameters(),2e-4,weight_decay=1e-4);best=1e9;hist=[]
 for e in range(1,a.epochs+1):
  m.train();ls=[]
  for c,x,z in tl:c,x,z=c.to(dev),x.to(dev),z.to(dev);l=lossfn(m(c,x),z);opt.zero_grad();l.backward();opt.step();ls.append(l.item())
  m.eval();vv=[]
  with torch.no_grad():
   for c,x,z in vl:vv.append(lossfn(m(c.to(dev),x.to(dev)),z.to(dev)).item())
  row={'epoch':e,'train_bce':float(np.mean(ls)),'val_bce':float(np.mean(vv))};hist.append(row);print(row,flush=True)
  if row['val_bce']<best:best=row['val_bce'];torch.save({'model':m.state_dict(),'epoch':e},a.output/'best.pt')
 m.load_state_dict(torch.load(a.output/'best.pt',map_location=dev,weights_only=True)['model']);m.eval();p=[];yy=[]
 with torch.no_grad():
  for c,x,z in DataLoader(te,a.batch_size*2):p.extend(torch.sigmoid(m(c.to(dev),x.to(dev))).cpu().numpy());yy.extend(z.numpy())
 result=metrics(np.asarray(p),np.asarray(yy));result.update({'device':str(dev),'seed':a.seed,'best_val_bce':best,'scope':'controlled collision replay using measured target speeds, longitudinal actions, and augmented placements'});(a.output/'test_metrics.json').write_text(json.dumps(result,indent=2));(a.output/'history.json').write_text(json.dumps(hist,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
