"""Gymnasium environment for continuous-control training in CARLA."""
from __future__ import annotations
import math
import random
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import carla
from mine_navigation import MineNavigation


class CarlaMineEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, host="127.0.0.1", port=2000, timeout=10.0,
                 max_steps=1200, fixed_delta=0.05, map_name=None,
                 navigation_root=None):
        super().__init__(); self.max_steps=max_steps; self.fixed_delta=fixed_delta
        self.client=carla.Client(host,port); self.client.set_timeout(timeout)
        self.world=self.client.get_world()
        if map_name and map_name not in self.world.get_map().name:
            self.world=self.client.load_world(map_name)
        self.original_settings=self.world.get_settings()
        settings=self.world.get_settings(); settings.synchronous_mode=True
        settings.fixed_delta_seconds=fixed_delta; settings.no_rendering_mode=True
        self.world.apply_settings(settings)
        self.action_space=spaces.Box(np.array([-1.,-1.],np.float32),np.array([1.,1.],np.float32))
        self.observation_space=spaces.Box(-np.inf,np.inf,(14,),np.float32)
        self.vehicle=None; self.sensors=[]; self.collision=False; self.lane_events=0
        self.steps=0; self.goal=None; self.last_distance=0.
        self.navigation=MineNavigation(navigation_root) if navigation_root else None

    def _destroy(self):
        for actor in self.sensors+[self.vehicle]:
            if actor is not None:
                try: actor.destroy()
                except RuntimeError: pass
        self.sensors=[]; self.vehicle=None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._destroy(); self.collision=False; self.lane_events=0; self.steps=0
        if self.navigation:
            start_id,goal_id=self.navigation.choose_pair(random.Random(seed))
            spawn=[self.navigation.transform(start_id)]
            custom_goal=self.navigation.transform(goal_id).location
        else:
            spawn=list(self.world.get_map().get_spawn_points());custom_goal=None
        if not spawn or (not self.navigation and len(spawn)<2):
            raise RuntimeError("CARLA map needs at least two spawn points or generated mine navigation assets")
        rng=random.Random(seed); rng.shuffle(spawn)
        bp=self.world.get_blueprint_library().filter("vehicle.*")[0]
        for transform in spawn:
            self.vehicle=self.world.try_spawn_actor(bp,transform)
            if self.vehicle: break
        if not self.vehicle: raise RuntimeError("unable to spawn ego vehicle")
        start=self.vehicle.get_location()
        if custom_goal is not None:self.goal=custom_goal
        else:
            goals=sorted(spawn,key=lambda t:start.distance(t.location),reverse=True)
            self.goal=goals[0].location
        self.last_distance=start.distance(self.goal)
        collision_bp=self.world.get_blueprint_library().find("sensor.other.collision")
        collision=self.world.spawn_actor(collision_bp,carla.Transform(),attach_to=self.vehicle)
        collision.listen(lambda _:setattr(self,"collision",True)); self.sensors.append(collision)
        lane_bp=self.world.get_blueprint_library().find("sensor.other.lane_invasion")
        lane=self.world.spawn_actor(lane_bp,carla.Transform(),attach_to=self.vehicle)
        lane.listen(lambda _:setattr(self,"lane_events",self.lane_events+1)); self.sensors.append(lane)
        self.world.tick(); return self._observation(),{"goal_distance":self.last_distance}

    def _observation(self):
        t=self.vehicle.get_transform(); v=self.vehicle.get_velocity(); a=self.vehicle.get_acceleration(); w=self.vehicle.get_angular_velocity()
        dx=self.goal.x-t.location.x; dy=self.goal.y-t.location.y
        yaw=math.radians(t.rotation.yaw); c,s=math.cos(yaw),math.sin(yaw)
        rel_x=c*dx+s*dy; rel_y=-s*dx+c*dy
        return np.asarray([v.x,v.y,v.z,a.x,a.y,a.z,w.x,w.y,w.z,rel_x,rel_y,
                           math.hypot(dx,dy),float(self.collision),float(self.lane_events>0)],np.float32)

    def step(self, action):
        steer=float(np.clip(action[0],-1,1)); longitudinal=float(np.clip(action[1],-1,1))
        self.vehicle.apply_control(carla.VehicleControl(steer=steer,
            throttle=max(0.,longitudinal),brake=max(0.,-longitudinal)))
        self.world.tick(); self.steps+=1; obs=self._observation(); distance=float(obs[11])
        progress=self.last_distance-distance; self.last_distance=distance
        speed=float(np.linalg.norm(obs[:3])); reward=2.0*progress+0.03*speed-0.01*abs(steer)
        if self.lane_events: reward-=0.2; self.lane_events=0
        terminated=self.collision or distance<5.0
        if self.collision: reward-=100.0
        if distance<5.0: reward+=100.0
        truncated=self.steps>=self.max_steps
        return obs,float(reward),terminated,truncated,{"goal_distance":distance,"speed":speed,"collision":self.collision}

    def close(self):
        self._destroy()
        try:self.world.apply_settings(self.original_settings)
        except RuntimeError:pass

