#!/usr/bin/env python3
"""Run a short low-speed collision test on each measured mine route."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import carla


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--navigation-root", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--routes", default="",
                        help="Comma-separated route names; default tests all routes")
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    graph = json.loads((args.navigation_root / "navigation_graph.json").read_text())
    nodes = {int(n["id"]): n for n in graph["nodes"]}

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    old_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    vehicle_bp = world.get_blueprint_library().filter("vehicle.*model3*")
    vehicle_bp = (vehicle_bp or world.get_blueprint_library().filter("vehicle.*"))[0]
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    selected = {name.strip() for name in args.routes.split(",") if name.strip()}
    results = []
    diagnostics = {}
    try:
        for route_name, raw_ids in sorted(graph["route_node_ids"].items()):
            if selected and route_name not in selected:
                continue
            ids = [int(i) for i in raw_ids]
            ids = [value for i, value in enumerate(ids) if i == 0 or value != ids[i - 1]]
            path = [nodes[i] for i in ids]
            if len(path) < 3:
                results.append((route_name, "insufficient_path", 0.0, 0))
                continue
            a, b = path[0], path[1]
            yaw = math.degrees(math.atan2(b["carla_y_m"] - a["carla_y_m"],
                                          b["carla_x_m"] - a["carla_x_m"]))
            transform = carla.Transform(
                carla.Location(x=a["carla_x_m"], y=a["carla_y_m"],
                               z=a["z_relative_m"] + 3.0),
                carla.Rotation(yaw=yaw),
            )
            vehicle = world.try_spawn_actor(vehicle_bp, transform)
            if vehicle is None:
                results.append((route_name, "spawn_failed", 0.0, 0))
                continue
            hits = []
            sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
            sensor.listen(lambda event, store=hits: store.append({
                "other_type": event.other_actor.type_id,
                "other_id": int(event.other_actor.id),
                "impulse": math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 +
                                     event.normal_impulse.z ** 2),
                "x": float(event.transform.location.x),
                "y": float(event.transform.location.y),
                "z": float(event.transform.location.z),
            }))
            vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            for _ in range(30):
                world.tick()
            # Ignore initial tyre/terrain contact while the actor settles.
            hits.clear()
            start = vehicle.get_location()
            travelled = 0.0
            steps = 0
            max_lateral = 0.0
            samples = []
            try:
                for step in range(500):
                    world.tick()
                    location = vehicle.get_location()
                    travelled = start.distance(location)
                    if travelled >= args.distance:
                        break
                    nearest = min(range(len(path)), key=lambda i:
                                  (path[i]["carla_x_m"] - location.x) ** 2 +
                                  (path[i]["carla_y_m"] - location.y) ** 2)
                    nearest_node = path[nearest]
                    lateral = math.hypot(nearest_node["carla_x_m"] - location.x,
                                         nearest_node["carla_y_m"] - location.y)
                    max_lateral = max(max_lateral, lateral)
                    if step % 10 == 0:
                        samples.append({"step": step, "x": location.x, "y": location.y,
                                        "z": location.z, "lateral_m": lateral,
                                        "route_z_m": nearest_node["z_relative_m"]})
                    target = path[min(nearest + 3, len(path) - 1)]
                    desired = math.atan2(target["carla_y_m"] - location.y,
                                         target["carla_x_m"] - location.x)
                    current = math.radians(vehicle.get_transform().rotation.yaw)
                    steer = max(-0.65, min(0.65, 1.2 * wrap(desired - current)))
                    velocity = vehicle.get_velocity()
                    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                    error = args.speed - speed
                    throttle = max(0.0, min(0.45, 0.28 * error))
                    brake = max(0.0, min(0.5, -0.35 * error))
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=throttle, brake=brake, steer=steer))
                    steps = step + 1
                significant = [hit for hit in hits if hit["impulse"] >= 100.0]
                status = "ok" if travelled >= args.distance and not significant else (
                    "collision" if significant else "incomplete")
                actors = sorted({hit["other_type"] for hit in significant})
                results.append((route_name, status, round(travelled, 2),
                                len(significant), actors, steps, round(max_lateral, 3)))
                enriched_hits = []
                for hit in significant:
                    nearest_hit = min(path, key=lambda n:
                        (n["carla_x_m"] - hit["x"]) ** 2 +
                        (n["carla_y_m"] - hit["y"]) ** 2)
                    item = dict(hit)
                    item["centerline_distance_m"] = math.hypot(
                        nearest_hit["carla_x_m"] - hit["x"],
                        nearest_hit["carla_y_m"] - hit["y"])
                    item["route_z_m"] = nearest_hit["z_relative_m"]
                    item["actor_minus_route_z_m"] = hit["z"] - nearest_hit["z_relative_m"]
                    enriched_hits.append(item)
                diagnostics[route_name] = {
                    "status": status, "travelled_m": travelled, "steps": steps,
                    "max_lateral_error_m": max_lateral,
                    "significant_collision_count": len(significant),
                    "collisions": enriched_hits, "samples": samples,
                }
            finally:
                sensor.stop()
                sensor.destroy()
                vehicle.destroy()
                world.tick()
    finally:
        world.apply_settings(old_settings)

    for result in results:
        print(result)
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(json.dumps(diagnostics, indent=2))
        print("diagnostics:", args.diagnostics)
    passed = sum(r[1] == "ok" for r in results)
    print(f"MINE_ROUTE_DRIVE_VALIDATION passed={passed}/{len(results)}")
    raise SystemExit(0 if passed == len(results) else 2)


if __name__ == "__main__":
    main()
