#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,torch
from train_il import Shards
from il_core import make_model,HORIZONS
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('em_opm/data/il_v1'));ap.add_argument('--checkpoint',type=Path,default=Path('em_opm/runs/il_v1/best.pt'));ap.add_argument('--output',type=Path,default=Path('em_opm/runs/il_v1/test_metrics.json'));ap.add_argument('--batch-size',type=int,default=64);a=ap.parse_args();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=make_model().to(dev);m.load_state_dict(torch.load(a.checkpoint,map_location=dev,weights_only=True)['model']);m.eval();ds=Shards(a.data,[7]);errs=[]
 with torch.no_grad():
  for i in range(0,len(ds),a.batch_size):
   q=[ds[j] for j in range(i,min(i+a.batch_size,len(ds)))];b,s,g,y=[torch.stack([x[k] for x in q]).to(dev) for k in range(4)];pred,_=m(b,s,g);errs.append(torch.linalg.norm(pred-y,dim=-1).cpu().numpy())
 e=np.concatenate(errs)
 truth=np.stack([ds[i][3].numpy() for i in range(len(ds))]);state=np.stack([ds[i][1].numpy() for i in range(len(ds))]);goal=np.stack([ds[i][2].numpy() for i in range(len(ds))])
 cv=state[:,:2,None]*HORIZONS[None,None,:];cv=np.transpose(cv,(0,2,1));gl=goal[:,None,:]*(HORIZONS[None,:,None]/8.)
 def score(p):
  z=np.linalg.norm(p-truth,axis=-1);return {'ADE_m':float(z.mean()),'FDE_m':float(z[:,-1].mean()),'p95_FDE_m':float(np.percentile(z[:,-1],95))}
 metrics={'test_sequence':'M0S7','samples':len(e),'learned':{'ADE_m':float(e.mean()),'FDE_m':float(e[:,-1].mean()),'per_horizon_m':{str(int(h)):float(e[:,i].mean()) for i,h in enumerate(HORIZONS)},'p95_FDE_m':float(np.percentile(e[:,-1],95))},'constant_velocity':score(cv),'linear_to_8s_goal':score(gl),'scope':'open-loop imitation baseline; not closed-loop navigation'};a.output.write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
