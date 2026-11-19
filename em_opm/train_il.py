#!/usr/bin/env python3
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from il_core import make_model
class Shards(Dataset):
 def __init__(self,root,ids):
  self.parts=[];self.index=[]
  for sid in ids:
   b=np.load(root/f'M0S{sid}_bev.npy',mmap_mode='r')
   with np.load(root/f'M0S{sid}_labels.npz') as z:
    y={k:z[k] for k in ('states','goals','waypoints')}
   self.parts.append((b,y));self.index.extend((len(self.parts)-1,i) for i in range(len(b)))
 def __len__(self):return len(self.index)
 def __getitem__(self,i):
  p,j=self.index[i];b,y=self.parts[p];return torch.from_numpy(np.array(b[j],np.float32)),torch.from_numpy(y['states'][j]),torch.from_numpy(y['goals'][j]),torch.from_numpy(y['waypoints'][j])
def run(model,loader,device,opt=None):
 model.train(opt is not None);tot=n=0
 for b,s,g,y in loader:
  b,s,g,y=[x.to(device) for x in (b,s,g,y)];m,ls=model(b,s,g);loss=(.5*((y-m)/ls.exp())**2+ls).mean()
  if opt:opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  tot+=loss.item()*len(b);n+=len(b)
 return tot/n
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('em_opm/data/il_v1'));ap.add_argument('--output',type=Path,default=Path('em_opm/runs/il_v1'));ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 random.seed(7);np.random.seed(7);torch.manual_seed(7);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=make_model().to(device);train=Shards(a.data,[0,1,2,3,5]);val=Shards(a.data,[6]);tl=DataLoader(train,a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=device.type=='cuda');vl=DataLoader(val,a.batch_size,num_workers=a.workers,pin_memory=device.type=='cuda');opt=torch.optim.AdamW(model.parameters(),2e-4,weight_decay=1e-4);history=[];best=float('inf')
 for e in range(1,a.epochs+1):
  tr=run(model,tl,device,opt);va=run(model,vl,device);row={'epoch':e,'train_nll':tr,'val_nll':va};history.append(row);print(row,flush=True);torch.save({'model':model.state_dict(),'epoch':e},a.output/'last.pt')
  if va<best:best=va;torch.save({'model':model.state_dict(),'epoch':e,'val_nll':va},a.output/'best.pt')
 (a.output/'history.json').write_text(json.dumps({'device':str(device),'history':history,'best_val_nll':best},indent=2))
if __name__=='__main__':main()
