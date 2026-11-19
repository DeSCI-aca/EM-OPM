#!/usr/bin/env python3
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from risk import RiskNet
class Data(Dataset):
 def __init__(self,root,ids):
  self.p=[];self.ix=[]
  for sid in ids:
   b=np.load(root/f'M0S{sid}_bev.npy',mmap_mode='r')
   with np.load(root/f'M0S{sid}_labels.npz') as z:y={k:z[k] for k in ('states','goals','candidates','labels')}
   self.p.append((b,y));self.ix.extend((len(self.p)-1,i,j) for i in range(len(b)) for j in range(25))
 def __len__(self):return len(self.ix)
 def __getitem__(self,n):
  p,i,j=self.ix[n];b,y=self.p[p];return tuple(torch.from_numpy(np.array(x,np.float32)) for x in (b[i],y['states'][i],y['goals'][i],y['candidates'][i,j],y['labels'][i,j,:1]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('em_opm/data/risk_v1'));ap.add_argument('--output',type=Path,default=Path('em_opm/runs/risk_v1'));ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--batch-size',type=int,default=128);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);random.seed(11);np.random.seed(11);torch.manual_seed(11);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tr=Data(a.data,[0,1,2,3,5]);va=Data(a.data,[6]);tl=DataLoader(tr,a.batch_size,shuffle=True,num_workers=4,pin_memory=True);vl=DataLoader(va,a.batch_size,num_workers=4,pin_memory=True);m=RiskNet().to(dev)
 all_y=np.concatenate([p[1]['labels'][...,:1].reshape(-1,1) for p in tr.p]);pos=(len(all_y)-all_y.sum(0))/np.maximum(all_y.sum(0),1);lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,dtype=torch.float32,device=dev));opt=torch.optim.AdamW(m.parameters(),2e-4,weight_decay=1e-4);hist=[];best=1e9
 for e in range(1,a.epochs+1):
  vals=[];m.train()
  for b,s,g,t,y in tl:b,s,g,t,y=[x.to(dev) for x in (b,s,g,t,y)];loss=lossfn(m(b,s,g,t),y);opt.zero_grad();loss.backward();opt.step();vals.append(loss.item())
  m.eval();vv=[]
  with torch.no_grad():
   for b,s,g,t,y in vl:b,s,g,t,y=[x.to(dev) for x in (b,s,g,t,y)];vv.append(lossfn(m(b,s,g,t),y).item())
  row={'epoch':e,'train_bce':float(np.mean(vals)),'val_bce':float(np.mean(vv))};hist.append(row);print(row,flush=True)
  if row['val_bce']<best:best=row['val_bce'];torch.save({'model':m.state_dict(),'epoch':e},a.output/'best.pt')
 (a.output/'history.json').write_text(json.dumps({'device':str(dev),'pos_weight':pos.tolist(),'history':hist},indent=2))
if __name__=='__main__':main()
