#!/usr/bin/env python3
"""Summarize the dynamic tracks used by the paper split."""
import json
from pathlib import Path

ROOT = Path("em_opm/data/dynamic_actors_v1")
SCENES = (0, 1, 2, 3, 5, 6, 7)


def main():
    summary = {}
    for sid in SCENES:
        path = ROOT / f"scene_{sid}_tracks.json"
        data = json.loads(path.read_text())
        tracks = data["tracks"]
        usable = [
            track for track in tracks
            if track["dynamic"] and track["duration_s"] >= 2
            and track["speed_mps"] <= 15
        ]
        summary[f"scene_{sid}"] = {
            "tracks": len(tracks),
            "dynamic_tracks": sum(track["dynamic"] for track in tracks),
            "usable_tracks": len(usable),
            "usable_observations": sum(track["observations"] for track in usable),
        }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

