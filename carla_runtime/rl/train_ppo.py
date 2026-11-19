#!/usr/bin/env python3
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from carla_mine_env import CarlaMineEnv

ap=argparse.ArgumentParser(); ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=2000)
ap.add_argument('--steps',type=int,default=1000000);ap.add_argument('--device',default='cuda:0');ap.add_argument('--map',default=None)
ap.add_argument('--navigation-root',type=Path,default=Path('carla_mine/outputs/carla_route_buffer_1km_utm_aligned/lidar_all_points_1m/navigation'))
ap.add_argument('--output',type=Path,default=Path('carla_runtime/runs/ppo_mine'));a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
env=Monitor(CarlaMineEnv(a.host,a.port,map_name=a.map,navigation_root=a.navigation_root),str(a.output/'monitor.csv'))
model=PPO('MlpPolicy',env,verbose=1,device=a.device,tensorboard_log=str(a.output/'tb'),n_steps=2048,batch_size=256)
try:model.learn(a.steps,progress_bar=True);model.save(a.output/'final')
finally:env.close()
