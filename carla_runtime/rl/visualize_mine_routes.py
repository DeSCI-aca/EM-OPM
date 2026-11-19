#!/usr/bin/env python3
"""Draw the seven measured routes, navigation graph, and spawns in CARLA."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import carla


COLORS = [
    (255, 60, 60),    # M0S0 red
    (255, 150, 30),   # M0S1 orange
    (255, 235, 40),   # M0S2 yellow
    (60, 220, 80),    # M0S3 green
    (20, 210, 230),   # M0S5 cyan
    (70, 110, 255),   # M0S6 blue
    (210, 70, 255),   # M0S7 magenta
]


def location(node, lift):
    return carla.Location(
        x=float(node["carla_x_m"]),
        y=float(node["carla_y_m"]),
        z=float(node["z_relative_m"]) + lift,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--navigation-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "carla_mine/outputs/carla_route_buffer_1km_utm_aligned/lidar_all_points_1m/navigation",
    )
    parser.add_argument("--life-time", type=float, default=3600.0)
    parser.add_argument("--lift", type=float, default=1.5)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-spawns", action="store_true")
    args = parser.parse_args()

    graph = json.loads((args.navigation_root / "navigation_graph.json").read_text())
    nodes = {int(node["id"]): node for node in graph["nodes"]}
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    debug = world.debug

    # The merged conservative navigation graph is a thin white underlay.
    if not args.no_graph:
        for edge in graph["edges"]:
            debug.draw_line(
                location(nodes[int(edge["source"])], args.lift - 0.35),
                location(nodes[int(edge["target"])], args.lift - 0.35),
                thickness=0.06,
                color=carla.Color(235, 235, 235),
                life_time=args.life_time,
                persistent_lines=True,
            )

    route_names = sorted(graph["route_node_ids"])
    for route_index, route_name in enumerate(route_names):
        ids = graph["route_node_ids"][route_name]
        color = carla.Color(*COLORS[route_index % len(COLORS)])
        for left, right in zip(ids[:-1], ids[1:]):
            if left == right:
                continue
            debug.draw_line(
                location(nodes[int(left)], args.lift),
                location(nodes[int(right)], args.lift),
                thickness=0.18,
                color=color,
                life_time=args.life_time,
                persistent_lines=True,
            )
        if ids:
            start = location(nodes[int(ids[0])], args.lift + 1.0)
            debug.draw_string(
                start,
                route_name,
                draw_shadow=True,
                color=color,
                life_time=args.life_time,
                persistent_lines=True,
            )

    spawn_count = 0
    if not args.no_spawns:
        with (args.navigation_root / "spawn_points.csv").open() as stream:
            spawns = list(csv.DictReader(stream))
        spawn_count = len(spawns)
        for row in spawns:
            point = carla.Location(
                x=float(row["carla_x_m"]),
                y=float(row["carla_y_m"]),
                z=float(row["z_relative_m"]) + args.lift,
            )
            debug.draw_point(
                point,
                size=0.14,
                color=carla.Color(255, 255, 255),
                life_time=args.life_time,
                persistent_lines=True,
            )

    # Put the spectator above the route system for an immediate overview.
    used = [nodes[int(i)] for ids in graph["route_node_ids"].values() for i in ids]
    min_x, max_x = min(n["carla_x_m"] for n in used), max(n["carla_x_m"] for n in used)
    min_y, max_y = min(n["carla_y_m"] for n in used), max(n["carla_y_m"] for n in used)
    max_z = max(n["z_relative_m"] for n in used)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    extent = max(max_x - min_x, max_y - min_y)
    world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(x=center_x, y=center_y, z=max_z + max(180.0, extent * 0.55)),
            carla.Rotation(pitch=-90.0),
        )
    )
    print(
        f"MINE_ROUTE_VISUALIZATION map={world.get_map().name} "
        f"routes={len(route_names)} graph_edges={len(graph['edges'])} spawns={spawn_count} "
        f"life_time={args.life_time}s"
    )


if __name__ == "__main__":
    main()
