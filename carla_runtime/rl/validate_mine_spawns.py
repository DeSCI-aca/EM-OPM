#!/usr/bin/env python3
"""Validate custom mine spawn transforms against CARLA terrain collision."""
import argparse
import csv
import time
from pathlib import Path

import carla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--navigation-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=7)
    args = parser.parse_args()

    with (args.navigation_root / "spawn_points.csv").open() as stream:
        points = list(csv.DictReader(stream))
    # Cover the full ordered route extent instead of checking adjacent samples.
    indices = sorted({round(i * (len(points) - 1) / max(1, args.count - 1)) for i in range(args.count)})

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    blueprint = world.get_blueprint_library().filter("vehicle.*model3*")
    if not blueprint:
        blueprint = world.get_blueprint_library().filter("vehicle.*")
    blueprint = blueprint[0]

    results = []
    for index in indices:
        row = points[index]
        expected = float(row["z_relative_m"])
        transform = carla.Transform(
            carla.Location(
                x=float(row["carla_x_m"]),
                y=float(row["carla_y_m"]),
                z=expected + 3.0,
            ),
            carla.Rotation(yaw=float(row["carla_yaw_deg"])),
        )
        vehicle = None
        for _ in range(3):
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                break
            transform.location.z += 2.0
            time.sleep(0.25)
        if vehicle is None:
            results.append((index, row["source_route"], expected, None, "spawn_failed"))
            continue
        try:
            # Let physics settle. Sleeping is intentional: the server runs asynchronously.
            time.sleep(2.5)
            location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
            clearance = location.z - expected
            # The exported collision mesh is simplified from the 1 m DEM, and CARLA
            # reports the vehicle actor origin rather than its tyre contact point.
            # A stable actor within this bounded vertical residual validates mapping
            # and collision; the residual is reported separately for refinement.
            status = "ok" if -1.5 <= clearance <= 3.5 and speed < 1.0 else "check"
            results.append((index, row["source_route"], expected, location.z, status, clearance, speed))
        finally:
            vehicle.destroy()
            time.sleep(0.2)

    print("map:", world.get_map().name)
    for result in results:
        print(result)
    passed = sum(result[4] == "ok" for result in results)
    print(f"MINE_SPAWN_VALIDATION passed={passed}/{len(results)}")
    raise SystemExit(0 if passed == len(results) else 2)


if __name__ == "__main__":
    main()
