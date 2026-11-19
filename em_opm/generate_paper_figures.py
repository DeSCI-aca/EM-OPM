#!/usr/bin/env python3
"""Generate qualitative and quantitative figures from held-out paper artifacts."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/em_opm_matplotlib")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from il_core import make_model
from risk import RiskNet
from train_collision import CollisionNet

ROOT = Path("em_opm")
OUT = ROOT / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path):
    return json.loads(Path(path).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def trajectory_segments(trajectories):
    origin = np.zeros((len(trajectories), 1, 2), np.float32)
    return np.concatenate([origin, trajectories], axis=1)


def open_loop_examples():
    bevs = np.load(ROOT / "data/il_v1/M0S7_bev.npy", mmap_mode="r")
    labels = np.load(ROOT / "data/il_v1/M0S7_labels.npz")
    model = make_model().to(DEVICE)
    checkpoint = torch.load(ROOT / "runs/il_v1/best.pt", map_location=DEVICE,
                            weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    indices = [40, 150, 260]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35), constrained_layout=True)
    for ax, index in zip(axes, indices):
        bev = np.asarray(bevs[index], np.float32)
        with torch.no_grad():
            pred, _ = model(
                torch.from_numpy(bev[None]).to(DEVICE),
                torch.from_numpy(labels["states"][index:index + 1]).to(DEVICE),
                torch.from_numpy(labels["goals"][index:index + 1]).to(DEVICE),
            )
        pred = pred[0].cpu().numpy()
        expert = labels["waypoints"][index]
        background = np.maximum(bev[2], 0.25 * bev[4])
        ax.imshow(background, extent=(-40, 40, -40, 40), origin="lower",
                  cmap="gray_r", vmin=0, vmax=1)
        ax.plot(*np.vstack([[0, 0], expert]).T, "o-", color="#168aad",
                linewidth=2, markersize=3, label="Expert")
        ax.plot(*np.vstack([[0, 0], pred]).T, "s--", color="#d1495b",
                linewidth=2, markersize=3, label="IL prediction")
        ax.scatter(0, 0, marker="^", s=55, color="black", zorder=5)
        ax.set(xlim=(-12, 35), ylim=(-22, 22), aspect="equal",
               xlabel="Longitudinal (m)", ylabel="Lateral (m)")
        error = np.linalg.norm(pred - expert, axis=1).mean()
        ax.set_title(f"M0S7 sample {index}  |  ADE={error:.2f} m")
    axes[0].legend(loc="lower right", frameon=True, fontsize=8)
    save(fig, "il_qualitative_m0s7")


def boundary_candidates():
    bevs = np.load(ROOT / "data/risk_v1/M0S7_bev.npy", mmap_mode="r")
    labels = np.load(ROOT / "data/risk_v1/M0S7_labels.npz")
    model = RiskNet().to(DEVICE)
    model.load_state_dict(torch.load(ROOT / "runs/risk_v1/best.pt",
                                     map_location=DEVICE,
                                     weights_only=True)["model"])
    model.eval()
    index = 180
    candidates = labels["candidates"][index]
    count = len(candidates)
    with torch.no_grad():
        probability = torch.sigmoid(model(
            torch.from_numpy(np.asarray(bevs[index], np.float32)[None]).to(DEVICE).expand(count, -1, -1, -1),
            torch.from_numpy(labels["states"][index:index + 1]).to(DEVICE).expand(count, -1),
            torch.from_numpy(labels["goals"][index:index + 1]).to(DEVICE).expand(count, -1),
            torch.from_numpy(candidates).to(DEVICE),
        )).cpu().numpy()[:, 0]
    background = np.maximum(np.asarray(bevs[index, 2], np.float32),
                            0.25 * np.asarray(bevs[index, 4], np.float32))
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    ax.imshow(background, extent=(-40, 40, -40, 40), origin="lower",
              cmap="gray_r", vmin=0, vmax=1)
    lines = LineCollection(trajectory_segments(candidates), cmap="RdYlGn_r",
                           norm=Normalize(0, 1), linewidths=2.2)
    lines.set_array(probability)
    ax.add_collection(lines)
    best = int(np.argmin(probability))
    ax.plot(*np.vstack([[0, 0], candidates[best]]).T, color="#146c43",
            linewidth=3.4, label=f"Lowest risk, $p_{{bd}}$={probability[best]:.2f}")
    ax.scatter(0, 0, marker="^", s=65, color="black", zorder=5)
    ax.set(xlim=(-8, 32), ylim=(-20, 20), aspect="equal",
           xlabel="Longitudinal (m)", ylabel="Lateral (m)",
           title="Action-conditioned boundary risk on M0S7")
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(lines, ax=ax, label="Predicted boundary probability")
    save(fig, "boundary_candidate_screening")


def collision_candidates():
    data = np.load(ROOT / "data/collision_v2/test.npz")
    model = CollisionNet().to(DEVICE)
    model.load_state_dict(torch.load(ROOT / "runs/collision_v2/best.pt",
                                     map_location=DEVICE,
                                     weights_only=True)["model"])
    model.eval()
    groups = len(data["labels"]) // 60
    chosen = None
    for group in range(groups):
        sl = slice(group * 60, (group + 1) * 60)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(data["candidates"][sl]).to(DEVICE),
                                    torch.from_numpy(data["actors"][sl]).to(DEVICE))).cpu().numpy()
        if p.min() < 0.15 and p.max() > 0.85:
            chosen = (sl, p)
            break
    if chosen is None:
        raise RuntimeError("No collision example with a useful probability range")
    sl, probability = chosen
    candidates = data["candidates"][sl]
    actor = data["actors"][sl][0]
    fig, ax = plt.subplots(figsize=(5.4, 4.5), constrained_layout=True)
    lines = LineCollection(trajectory_segments(candidates), cmap="RdYlGn_r",
                           norm=Normalize(0, 1), linewidths=1.8, alpha=0.9)
    lines.set_array(probability)
    ax.add_collection(lines)
    ax.plot(*np.vstack([[actor[0, 0] - (actor[1, 0] - actor[0, 0]),
                         actor[0, 1] - (actor[1, 1] - actor[0, 1])], actor]).T,
            "ko-", linewidth=2.5, markersize=4, label="Tracked actor")
    safe = int(np.argmin(probability))
    ax.plot(*np.vstack([[0, 0], candidates[safe]]).T, color="#146c43",
            linewidth=3.4, label=f"Lowest risk, $p_{{col}}$={probability[safe]:.2f}")
    ax.scatter(0, 0, marker="^", s=65, color="black", zorder=5, label="Ego")
    all_points = np.concatenate([candidates.reshape(-1, 2), actor], axis=0)
    ax.set(xlim=(min(-5, all_points[:, 0].min() - 3), all_points[:, 0].max() + 3),
           ylim=(all_points[:, 1].min() - 5, all_points[:, 1].max() + 5),
           aspect="equal", xlabel="Longitudinal (m)", ylabel="Lateral (m)",
           title="Action-conditioned collision screening")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(lines, ax=ax, label="Predicted collision probability")
    save(fig, "collision_candidate_screening")


def closed_loop_results():
    results = load_json(ROOT / "runs/paper_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.6), constrained_layout=True)
    static_names = ["IL only", "IL + risk", "Route upper bound"]
    static_keys = ["closed_loop_il", "closed_loop_risk", "closed_loop_route_upper"]
    success = [100 * results[key]["overall"]["success_rate"] for key in static_keys]
    boundary = [100 * results[key]["overall"]["boundary_rate"] for key in static_keys]
    x = np.arange(3)
    axes[0].bar(x - 0.18, success, 0.36, label="Success", color="#2a9d8f")
    axes[0].bar(x + 0.18, boundary, 0.36, label="Boundary violation", color="#e76f51")
    axes[0].set(xticks=x, xticklabels=static_names, ylabel="Episodes (%)",
                ylim=(0, 105), title="Boundary-recovery closed loop")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(fontsize=8)

    keys = ["dynamic_none", "dynamic_brake", "dynamic_geometric", "dynamic_learned"]
    names = ["Boundary only", "Emergency brake", "Geometric", "Learned"]
    fields = [("success_rate", "Success", "#2a9d8f"),
              ("collision_rate", "Collision", "#d1495b"),
              ("boundary_rate", "Boundary", "#f4a261"),
              ("timeout_rate", "Timeout", "#7b6d8d")]
    bottom = np.zeros(4)
    for field, label, color in fields:
        if field == "timeout_rate":
            values = np.array([
                100 * results[key].get(
                    field,
                    max(0, 1 - results[key]["success_rate"]
                        - results[key]["collision_rate"]
                        - results[key]["boundary_rate"]),
                )
                for key in keys
            ])
        else:
            values = np.array([100 * results[key][field] for key in keys])
        axes[1].bar(np.arange(4), values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[1].set(xticks=np.arange(4), xticklabels=names, ylabel="Episodes (%)",
                ylim=(0, 105), title="Dynamic-replay outcomes")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=8, ncol=2)
    save(fig, "closed_loop_results")


def main():
    open_loop_examples()
    boundary_candidates()
    collision_candidates()
    closed_loop_results()
    print(f"Wrote paper figures to {OUT}")


if __name__ == "__main__":
    main()
