"""Load the generated conservative mine navigation graph and CARLA transforms."""
from __future__ import annotations
import csv, json, math, random
from pathlib import Path
import carla

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "carla_mine/outputs/carla_route_buffer_1km_utm_aligned/lidar_all_points_1m/navigation"


class MineNavigation:
    def __init__(self, root=DEFAULT_ROOT):
        self.root=Path(root)
        self.graph=json.loads((self.root/"navigation_graph.json").read_text())
        with (self.root/"spawn_points.csv").open() as f:self.spawns=list(csv.DictReader(f))

    def transform(self, index):
        s=self.spawns[int(index)]
        return carla.Transform(
            carla.Location(x=float(s["carla_x_m"]),y=float(s["carla_y_m"]),z=float(s["z_relative_m"])),
            carla.Rotation(yaw=float(s.get("carla_yaw_deg", -float(s["yaw_deg"])))))

    def choose_pair(self, rng=None, minimum_distance=100.):
        rng=rng or random.Random(); order=list(range(len(self.spawns)));rng.shuffle(order)
        start=order[0]; a=self.transform(start).location
        valid=[i for i in order[1:] if a.distance(self.transform(i).location)>=minimum_distance]
        return start,(rng.choice(valid) if valid else order[-1])

    def nearest_node(self, location):
        return min(self.graph["nodes"],key=lambda n:math.hypot(n["carla_x_m"]-location.x,n["carla_y_m"]-location.y))

