#!/usr/bin/env python3
import argparse, carla, gymnasium, torch
from stable_baselines3 import PPO
ap=argparse.ArgumentParser();ap.add_argument('--connect',action='store_true');ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=2000);a=ap.parse_args()
print('CARLA API:',carla.Client);print('Gymnasium:',gymnasium.__version__);print('PyTorch:',torch.__version__,'CUDA:',torch.cuda.is_available(),'GPUs:',torch.cuda.device_count());print('SB3:',PPO.__name__)
if a.connect:
 c=carla.Client(a.host,a.port);c.set_timeout(10);w=c.get_world();print('Server map:',w.get_map().name,'actors:',len(w.get_actors()))
