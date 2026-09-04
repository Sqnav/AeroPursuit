from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


SPLIT_ORDER = ("train", "held_out", "cross_city")
SPLIT_LABELS = {
    "train": "Train",
    "held_out": "Held-out",
    "cross_city": "Cross-city",
}
SPLIT_COLORS = {
    "train": "#2563eb",
    "held_out": "#dc2626",
    "cross_city": "#16a34a",
}
ACTION_KEYS = ("vx", "vy", "vz", "yaw_norm")
IMAGE_SIZE = 640.0
HEATMAP_BINS = 64
DIFFICULTY_ORDER = ("easy", "medium", "hard")
DIFFICULTY_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}
DIFFICULTY_COLORS = {
    "easy": "#16a34a",
    "medium": "#f59e0b",
    "hard": "#dc2626",
}


def _numeric_suffix(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else -1


def _split_name(city_index: int, trajectory_index: int) -> str:
    if city_index <= 27:
        return "train" if trajectory_index <= 450 else "held_out"
    return "cross_city"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_plan_metadata(path: Path) -> tuple[str, int]:
    """Read the top-level mode and point count without loading frame payloads."""
    mode: str | None = None
    num_points: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith('"mode"'):
                mode = str(json.loads(stripped.split(":", 1)[1].rstrip(","))).lower()
            elif stripped.startswith('"num_points"'):
                num_points = int(json.loads(stripped.split(":", 1)[1].rstrip(",")))
            if mode is not None and num_points is not None:
                break
    if mode is None or num_points is None:
        raise ValueError(f"missing mode or num_points in plan metadata: {path}")
    return mode, num_points


def _load_plan_length_plot_data(
    plan_root: Path,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, list[int]]]:
    split_lengths = {
        name: {"lengths": []} for name in SPLIT_ORDER
    }
    difficulty_lengths = {name: [] for name in DIFFICULTY_ORDER}
    city_dirs = sorted(
        (path for path in plan_root.glob("city_*") if path.is_dir()),
        key=lambda path: _numeric_suffix(path.name),
    )
    for city_dir in city_dirs:
        city_index = _numeric_suffix(city_dir.name)
        plan_paths = sorted(
            city_dir.glob("trajectory_*.json"),
            key=lambda path: _numeric_suffix(path.stem),
        )
        for plan_path in plan_paths:
            trajectory_index = _numeric_suffix(plan_path.stem)
            mode, num_points = _load_plan_metadata(plan_path)
            split_lengths[_split_name(city_index, trajectory_index)]["lengths"].append(
                num_points
            )
            if mode not in difficulty_lengths:
                raise ValueError(f"unknown difficulty mode {mode!r}: {plan_path}")
            difficulty_lengths[mode].append(num_points)
    return split_lengths, difficulty_lengths


def _serialize_difficulty_modes(
    difficulty_lengths: dict[str, list[int]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "label": DIFFICULTY_LABELS[name],
            "trajectories": len(difficulty_lengths[name]),
            "trajectory_length": _summary(difficulty_lengths[name]),
        }
        for name in DIFFICULTY_ORDER
    }


def _new_split_stats() -> dict[str, Any]:
    return {
        "trajectories": 0,
        "valid_trajectories": 0,
        "declared_frames": 0,
        "frames": 0,
        "lengths": [],
        "distances": [],
        "box_areas": [],
        "box_center_x": [],
        "box_center_y": [],
        "valid_boxes": 0,
        "total_boxes": 0,
        "target_assets": Counter(),
        "jammer_counts": Counter(),
        "jammer_assets": Counter(),
        "capture_modes": Counter(),
        "actions": {key: [] for key in ACTION_KEYS},
    }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "p05": None,
                "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"count": 0, "mean": None, "std": None, "min": None, "p05": None,
                "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    quantiles = np.percentile(finite, [5, 25, 50, 75, 95])
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(finite.max()),
    }


def analyze_dataset(
    dataset_root: Path,
    plan_root: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    split_stats = {name: _new_split_stats() for name in SPLIT_ORDER}
    difficulty_lengths: dict[str, list[int]] = {
        name: [] for name in DIFFICULTY_ORDER
    }
    unknown_difficulty_trajectories = 0
    city_stats: list[dict[str, Any]] = []
    center_heatmap = np.zeros((HEATMAP_BINS, HEATMAP_BINS), dtype=np.int64)
    errors: list[str] = []

    city_dirs = sorted(
        (path for path in dataset_root.glob("City_*") if path.is_dir()),
        key=lambda path: _numeric_suffix(path.name),
    )
    for city_position, city_dir in enumerate(city_dirs, start=1):
        city_index = _numeric_suffix(city_dir.name)
        trajectory_dirs = sorted(
            (path for path in city_dir.glob("trajectory_*") if path.is_dir()),
            key=lambda path: _numeric_suffix(path.name),
        )
        city_frames = 0
        city_lengths: list[int] = []
        city_errors = 0
        for trajectory_dir in trajectory_dirs:
            trajectory_index = _numeric_suffix(trajectory_dir.name)
            split = _split_name(city_index, trajectory_index)
            stats = split_stats[split]
            stats["trajectories"] += 1
            try:
                uav_data = _load_json(trajectory_dir / "uav_trajectory.json")
                box_data = _load_json(trajectory_dir / "target_boxes.json")
            except Exception as exc:
                city_errors += 1
                errors.append(f"{trajectory_dir}: {type(exc).__name__}: {exc}")
                continue

            jammer_path = trajectory_dir / "jammer_trajectories.json"
            try:
                jammer_data = _load_json(jammer_path) if jammer_path.is_file() else {
                    "num_jammers": 0,
                    "jammer_asset_names": {},
                }
            except Exception as exc:
                city_errors += 1
                errors.append(f"{trajectory_dir}: {type(exc).__name__}: {exc}")
                continue

            frames = uav_data.get("trajectory")
            if not isinstance(frames, list):
                city_errors += 1
                errors.append(f"{trajectory_dir}: invalid uav trajectory")
                continue
            frame_count = len(frames)
            declared_frame_count = int(uav_data.get("num_frames", frame_count) or 0)
            stats["declared_frames"] += declared_frame_count
            boxes = box_data.get("boxes_xywh")
            uav_target_asset = str(uav_data.get("target_asset_name", "unknown"))
            box_target_asset = str(box_data.get("target_asset_name", "unknown"))
            integrity_problems: list[str] = []
            if frame_count <= 0 or frame_count != declared_frame_count:
                integrity_problems.append(
                    f"declared_frames={declared_frame_count}, trajectory_frames={frame_count}"
                )
            if not isinstance(boxes, list) or len(boxes) != frame_count:
                box_count = len(boxes) if isinstance(boxes, list) else "invalid"
                integrity_problems.append(f"trajectory_frames={frame_count}, boxes={box_count}")
            if uav_target_asset != box_target_asset:
                integrity_problems.append(
                    f"uav_target={uav_target_asset}, box_target={box_target_asset}"
                )
            if integrity_problems:
                city_errors += 1
                errors.append(f"{trajectory_dir}: " + "; ".join(integrity_problems))
                continue

            stats["valid_trajectories"] += 1
            stats["frames"] += frame_count
            stats["lengths"].append(frame_count)
            plan_path = (
                plan_root
                / f"city_{city_index}"
                / f"trajectory_{trajectory_index:04d}.json"
            )
            try:
                difficulty, _ = _load_plan_metadata(plan_path)
            except Exception as exc:
                difficulty = ""
                errors.append(
                    f"{trajectory_dir}: difficulty metadata unavailable from "
                    f"{plan_path}: {type(exc).__name__}: {exc}"
                )
            if difficulty in difficulty_lengths:
                difficulty_lengths[difficulty].append(frame_count)
            else:
                unknown_difficulty_trajectories += 1
            city_frames += frame_count
            city_lengths.append(frame_count)
            stats["target_assets"][uav_target_asset] += 1
            stats["capture_modes"][str(uav_data.get("capture_mode", "unknown"))] += 1

            jammer_count = int(jammer_data.get("num_jammers", 0) or 0)
            stats["jammer_counts"][jammer_count] += 1
            jammer_assets = jammer_data.get("jammer_asset_names")
            if isinstance(jammer_assets, dict):
                stats["jammer_assets"].update(str(value) for value in jammer_assets.values())

            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                distance = frame.get("distance")
                if isinstance(distance, (int, float)) and math.isfinite(float(distance)):
                    stats["distances"].append(float(distance))
                delta = frame.get("body_frame_delta")
                yaw = frame.get("body_frame_yaw_delta")
                if isinstance(delta, dict) and all(axis in delta for axis in ("x", "y", "z")):
                    values = [float(delta["x"]), float(delta["y"]), float(delta["z"])]
                    if all(math.isfinite(value) for value in values):
                        stats["actions"]["vx"].append(values[0])
                        stats["actions"]["vy"].append(values[1])
                        stats["actions"]["vz"].append(values[2])
                        yaw_value = float(yaw or 0.0) / 15.0
                        stats["actions"]["yaw_norm"].append(yaw_value)

            stats["total_boxes"] += len(boxes)
            valid_boxes: list[list[float]] = []
            for box in boxes:
                if not isinstance(box, (list, tuple)) or len(box) < 4:
                    continue
                try:
                    x, y, width, height = (float(value) for value in box[:4])
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(value) for value in (x, y, width, height)):
                    continue
                if width <= 0.0 or height <= 0.0:
                    continue
                valid_boxes.append([x, y, width, height])
            stats["valid_boxes"] += len(valid_boxes)
            if valid_boxes:
                array = np.asarray(valid_boxes, dtype=np.float64)
                center_x = np.clip((array[:, 0] + 0.5 * array[:, 2]) / IMAGE_SIZE, 0.0, 1.0)
                center_y = np.clip((array[:, 1] + 0.5 * array[:, 3]) / IMAGE_SIZE, 0.0, 1.0)
                area = (array[:, 2] * array[:, 3]) / (IMAGE_SIZE * IMAGE_SIZE)
                stats["box_areas"].extend(area.tolist())
                stats["box_center_x"].extend(center_x.tolist())
                stats["box_center_y"].extend(center_y.tolist())
                x_indices = np.minimum((center_x * HEATMAP_BINS).astype(np.int64), HEATMAP_BINS - 1)
                y_indices = np.minimum((center_y * HEATMAP_BINS).astype(np.int64), HEATMAP_BINS - 1)
                np.add.at(center_heatmap, (y_indices, x_indices), 1)

        city_stats.append(
            {
                "city": city_dir.name,
                "city_index": city_index,
                "trajectories": len(trajectory_dirs),
                "valid_trajectories": len(city_lengths),
                "frames": city_frames,
                "mean_frames": float(np.mean(city_lengths)) if city_lengths else None,
                "min_frames": int(min(city_lengths)) if city_lengths else None,
                "max_frames": int(max(city_lengths)) if city_lengths else None,
                "errors": city_errors,
            }
        )
        print(
            f"[dataset-analysis] {city_position}/{len(city_dirs)} {city_dir.name}: "
            f"trajectories={len(city_lengths)} frames={city_frames} errors={city_errors}",
            flush=True,
        )

    serialized_splits: dict[str, Any] = {}
    for split, stats in split_stats.items():
        serialized_splits[split] = {
            "trajectories": int(stats["trajectories"]),
            "valid_trajectories": int(stats["valid_trajectories"]),
            "declared_frames": int(stats["declared_frames"]),
            "frames": int(stats["frames"]),
            "trajectory_length": _summary(stats["lengths"]),
            "distance_m": _summary(stats["distances"]),
            "box_area_ratio": _summary(stats["box_areas"]),
            "box_center_x": _summary(stats["box_center_x"]),
            "box_center_y": _summary(stats["box_center_y"]),
            "valid_boxes": int(stats["valid_boxes"]),
            "total_boxes": int(stats["total_boxes"]),
            "valid_box_ratio": (
                float(stats["valid_boxes"] / stats["total_boxes"])
                if stats["total_boxes"]
                else None
            ),
            "target_assets": dict(sorted(stats["target_assets"].items())),
            "jammer_counts": {str(key): value for key, value in sorted(stats["jammer_counts"].items())},
            "jammer_assets": dict(sorted(stats["jammer_assets"].items())),
            "capture_modes": dict(sorted(stats["capture_modes"].items())),
            "normalized_actions": {
                key: _summary(values) for key, values in stats["actions"].items()
            },
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "image_size": [int(IMAGE_SIZE), int(IMAGE_SIZE)],
        "cities": len(city_stats),
        "trajectories": sum(item["trajectories"] for item in serialized_splits.values()),
        "valid_trajectories": sum(
            item["valid_trajectories"] for item in serialized_splits.values()
        ),
        "declared_frames": sum(
            item["declared_frames"] for item in serialized_splits.values()
        ),
        "frames": sum(item["frames"] for item in serialized_splits.values()),
        "difficulty_modes": _serialize_difficulty_modes(difficulty_lengths),
        "unknown_difficulty_trajectories": unknown_difficulty_trajectories,
        "splits": serialized_splits,
        "city_statistics": city_stats,
        "errors": errors,
    }
    summary["_plot_data"] = split_stats
    summary["_difficulty_plot_data"] = difficulty_lengths
    return summary, center_heatmap


def _style_axis(axis) -> None:
    axis.grid(True, axis="y", alpha=0.2, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_trajectory_length_frequency(summary: dict[str, Any], output_path: Path) -> None:
    plot_data = summary["_plot_data"]
    difficulty_data = summary["_difficulty_plot_data"]
    lengths = np.concatenate(
        [np.asarray(plot_data[name]["lengths"], dtype=np.int64) for name in SPLIT_ORDER]
    )
    minimum = int(lengths.min())
    maximum = int(lengths.max())
    bin_width = 2.0
    bins = np.arange(minimum - 0.5, maximum + bin_width + 0.5, bin_width)
    all_counts, edges = np.histogram(lengths, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    figure, axis = plt.subplots(figsize=(12.5, 7.2), constrained_layout=True)
    cumulative_counts = np.zeros_like(all_counts)
    for name in DIFFICULTY_ORDER:
        mode_lengths = np.asarray(difficulty_data[name], dtype=np.int64)
        counts, _ = np.histogram(mode_lengths, bins=bins)
        axis.bar(
            centers,
            counts,
            bottom=cumulative_counts,
            width=0.9 * bin_width,
            color=DIFFICULTY_COLORS[name],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.9,
            label=(
                f"{DIFFICULTY_LABELS[name]}  |  N={len(mode_lengths):,}, "
                f"mean={mode_lengths.mean():.1f}, median={np.median(mode_lengths):.0f}"
            ),
            zorder=1,
        )
        cumulative_counts += counts
    if not np.array_equal(cumulative_counts, all_counts):
        raise RuntimeError("Difficulty frequencies do not sum to the full dataset.")
    axis.text(
        0.985,
        0.965,
        f"All trajectories: N={len(lengths):,}  |  "
        f"mean={lengths.mean():.1f}  |  median={np.median(lengths):.0f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        color="#374151",
    )
    axis.set_xlabel("Frames per Trajectory", fontsize=16)
    axis.set_ylabel("Frequency (Number of Trajectories)", fontsize=16)
    axis.set_xlim(minimum - 1.0, maximum + 1.0)
    axis.tick_params(axis="both", labelsize=14)
    axis.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.94,
        fontsize=13,
    )
    _style_axis(axis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_dataset_split_distribution(summary: dict[str, Any], output_path: Path) -> None:
    plot_data = summary["_plot_data"]
    display_labels = ("Training Set", "Seen Test Set", "Unseen Test Set")
    split_ranges = (
        "City 1-27, trajectory 0001-0450",
        "City 1-27, trajectory 0451-0500",
        "City 28-30, trajectory 0001-0500",
    )
    counts = [int(plot_data[name]["trajectories"]) for name in SPLIT_ORDER]
    frames = [int(plot_data[name]["frames"]) for name in SPLIT_ORDER]
    colors = [SPLIT_COLORS[name] for name in SPLIT_ORDER]

    figure, axis = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    wedges, _, percentage_labels = axis.pie(
        counts,
        colors=colors,
        startangle=90,
        counterclock=False,
        labels=None,
        autopct=lambda percentage: f"{percentage:.1f}%",
        pctdistance=0.78,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2.5},
        textprops={"fontsize": 13, "fontweight": "bold", "color": "white"},
    )
    for label in percentage_labels:
        label.set_bbox({"boxstyle": "round,pad=0.25", "facecolor": "#111827", "edgecolor": "none", "alpha": 0.72})
    axis.text(
        0.0,
        0.03,
        f"{sum(counts):,}",
        ha="center",
        va="center",
        fontsize=23,
        fontweight="bold",
        color="#111827",
    )
    axis.text(0.0, -0.12, "trajectories", ha="center", va="center", fontsize=12, color="#4b5563")
    legend_labels = [
        f"{label}: {count:,} trajectories, {frame:,} frames\n{split_range}"
        for label, count, frame, split_range in zip(display_labels, counts, frames, split_ranges)
    ]
    axis.legend(
        wedges,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
        fontsize=11,
        ncol=1,
        handlelength=1.4,
        labelspacing=1.0,
    )
    axis.set_title("Dataset Split Distribution", fontsize=17, fontweight="bold", pad=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_summary(summary: dict[str, Any], center_heatmap: np.ndarray, output_path: Path) -> None:
    plot_data = summary["_plot_data"]
    figure, axes = plt.subplots(3, 3, figsize=(22, 16), constrained_layout=True)
    figure.suptitle(
        "UAV Visual Pursuit Dataset: Coverage, Geometry, and Control Statistics\n"
        f"{summary['cities']:,} cities  |  {summary['trajectories']:,} trajectory directories  |  "
        f"{summary['valid_trajectories']:,} valid  |  {summary['frames']:,} aligned frames  |  640 x 640",
        fontsize=20,
        fontweight="bold",
    )

    axis = axes[0, 0]
    counts = [plot_data[name]["trajectories"] for name in SPLIT_ORDER]
    labels = [SPLIT_LABELS[name] for name in SPLIT_ORDER]
    colors = [SPLIT_COLORS[name] for name in SPLIT_ORDER]
    bars = axis.bar(labels, counts, color=colors, width=0.65)
    for bar, split in zip(bars, SPLIT_ORDER):
        frames = plot_data[split]["frames"]
        valid = plot_data[split]["valid_trajectories"]
        trajectory_text = f"{int(bar.get_height()):,} traj"
        if valid != int(bar.get_height()):
            trajectory_text += f" ({valid:,} valid)"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{trajectory_text}\n{frames / 1000:.1f}k aligned frames",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.set_ylabel("Trajectories")
    axis.set_ylim(0.0, max(counts) * 1.16)
    axis.set_title("A. Experimental split size", loc="left", fontweight="bold")
    _style_axis(axis)

    axis = axes[0, 1]
    cities = summary["city_statistics"]
    city_indices = np.asarray([item["city_index"] for item in cities])
    city_frames = np.asarray([item["frames"] for item in cities], dtype=np.float64) / 1000.0
    city_colors = [SPLIT_COLORS["train"] if index <= 27 else SPLIT_COLORS["cross_city"] for index in city_indices]
    axis.bar(city_indices, city_frames, color=city_colors, width=0.78)
    axis.axvline(27.5, color="#111827", linestyle="--", linewidth=1.2)
    axis.set_xlabel("City index")
    axis.set_ylabel("RGB frames (thousands)")
    axis.set_xticks([1, 5, 10, 15, 20, 25, 27, 28, 29, 30])
    axis.set_title("B. Frame coverage by city", loc="left", fontweight="bold")
    _style_axis(axis)

    axis = axes[0, 2]
    all_lengths = np.concatenate(
        [np.asarray(plot_data[name]["lengths"], dtype=np.float64) for name in SPLIT_ORDER]
    )
    minimum = int(all_lengths.min())
    maximum = int(all_lengths.max())
    bins = np.arange(minimum - 0.5, maximum + 1.5, max(1, int(math.ceil((maximum - minimum) / 35))))
    for split in SPLIT_ORDER:
        axis.hist(
            plot_data[split]["lengths"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
        )
    axis.set_xlabel("Frames per trajectory")
    axis.set_ylabel("Density")
    axis.set_title("C. Trajectory length distribution", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    _style_axis(axis)

    axis = axes[1, 0]
    assets = sorted(
        {asset for split in SPLIT_ORDER for asset in plot_data[split]["target_assets"]},
        key=_numeric_suffix,
    )
    x = np.arange(len(assets))
    for split in SPLIT_ORDER:
        total = max(plot_data[split]["valid_trajectories"], 1)
        percentages = [100.0 * plot_data[split]["target_assets"].get(asset, 0) / total for asset in assets]
        axis.plot(
            x,
            percentages,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
        )
    axis.set_xticks(x)
    axis.set_xticklabels(assets, rotation=45, ha="right")
    axis.set_ylabel("Trajectories (%)")
    axis.set_title("D. Target UAV asset balance", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=3, fontsize=9)
    _style_axis(axis)

    axis = axes[1, 1]
    jammer_values = sorted(
        {value for split in SPLIT_ORDER for value in plot_data[split]["jammer_counts"]}
    )
    width = 0.24
    x = np.arange(len(jammer_values))
    for offset, split in enumerate(SPLIT_ORDER):
        total = max(plot_data[split]["valid_trajectories"], 1)
        percentages = [100.0 * plot_data[split]["jammer_counts"].get(value, 0) / total for value in jammer_values]
        axis.bar(
            x + (offset - 1) * width,
            percentages,
            width=width,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
        )
    axis.set_xticks(x)
    axis.set_xticklabels([str(value) for value in jammer_values])
    axis.set_xlabel("Jammers per trajectory")
    axis.set_ylabel("Trajectories (%)")
    axis.set_title("E. Distractor density", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=9)
    _style_axis(axis)

    axis = axes[1, 2]
    positive = center_heatmap[center_heatmap > 0]
    normalization = LogNorm(vmin=max(1, int(positive.min())), vmax=int(positive.max())) if len(positive) else None
    image = axis.imshow(
        center_heatmap,
        origin="upper",
        extent=(0.0, 1.0, 1.0, 0.0),
        cmap="viridis",
        norm=normalization,
        aspect="equal",
    )
    axis.axvline(0.5, color="white", linewidth=1.0, alpha=0.75)
    axis.axhline(0.5, color="white", linewidth=1.0, alpha=0.75)
    axis.set_xlabel("Normalized target center x")
    axis.set_ylabel("Normalized target center y")
    axis.set_title("F. Target center density", loc="left", fontweight="bold")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Frame count (log)")

    axis = axes[2, 0]
    all_distances = np.concatenate(
        [np.asarray(plot_data[name]["distances"], dtype=np.float64) for name in SPLIT_ORDER]
    )
    distance_max = float(np.percentile(all_distances, 99.5))
    distance_bins = np.linspace(0.0, distance_max, 80)
    for split in SPLIT_ORDER:
        axis.hist(
            plot_data[split]["distances"],
            bins=distance_bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
        )
    axis.set_xlabel("UAV-target distance (m)")
    axis.set_ylabel("Density")
    axis.set_title("G. Pursuit distance distribution", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=9)
    _style_axis(axis)

    axis = axes[2, 1]
    all_areas = np.concatenate(
        [np.asarray(plot_data[name]["box_areas"], dtype=np.float64) for name in SPLIT_ORDER]
    )
    positive_areas = all_areas[all_areas > 0.0]
    area_min, area_max = np.percentile(positive_areas, [0.1, 99.9])
    area_bins = np.geomspace(area_min, area_max, 80)
    for split in SPLIT_ORDER:
        axis.hist(
            plot_data[split]["box_areas"],
            bins=area_bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Target box area / image area")
    axis.set_ylabel("Density (log)")
    axis.set_title("H. Apparent target scale", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=9)
    _style_axis(axis)

    axis = axes[2, 2]
    action_labels = ("vx", "vy", "vz", "yaw / 15")
    offsets = (-0.22, 0.0, 0.22)
    for split_offset, split in zip(offsets, SPLIT_ORDER):
        for action_index, key in enumerate(ACTION_KEYS):
            values = np.asarray(plot_data[split]["actions"][key], dtype=np.float64)
            p05, p25, median, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
            position = action_index + split_offset
            axis.plot([position, position], [p05, p95], color=SPLIT_COLORS[split], linewidth=1.2)
            axis.plot([position, position], [p25, p75], color=SPLIT_COLORS[split], linewidth=5.0)
            axis.scatter([position], [median], color="white", edgecolor=SPLIT_COLORS[split], s=28, zorder=3)
        axis.plot([], [], color=SPLIT_COLORS[split], linewidth=5.0, label=SPLIT_LABELS[split])
    axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.5)
    axis.set_xticks(np.arange(len(action_labels)))
    axis.set_xticklabels(action_labels)
    axis.set_ylabel("Normalized expert action\n(5-95% thin, IQR thick, median dot)")
    axis.set_title("I. Expert control distribution", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=9, ncol=3)
    _style_axis(axis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def save_outputs(summary: dict[str, Any], center_heatmap: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory_length_frequency(summary, output_dir / "trajectory_length_frequency.png")
    plot_dataset_split_distribution(summary, output_dir / "dataset_split_distribution.png")
    plot_data = summary.pop("_plot_data")
    difficulty_plot_data = summary.pop("_difficulty_plot_data")
    try:
        with (output_dir / "dataset_statistics.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        city_rows = summary["city_statistics"]
        with (output_dir / "city_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(city_rows[0].keys()))
            writer.writeheader()
            writer.writerows(city_rows)
    finally:
        summary["_plot_data"] = plot_data
        summary["_difficulty_plot_data"] = difficulty_plot_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the UAV pursuit dataset and draw a summary figure.")
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=None,
        help="Planned trajectories containing the easy/medium/hard mode metadata.",
    )
    parser.add_argument(
        "--length-plot-only",
        action="store_true",
        help="Regenerate only the trajectory-length plot from lightweight plan metadata.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/dataset_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_root = args.plan_root or args.dataset_root.resolve().parent / "Plandataset"
    if args.length_plot_only:
        split_lengths, difficulty_lengths = _load_plan_length_plot_data(plan_root)
        plot_summary = {
            "_plot_data": split_lengths,
            "_difficulty_plot_data": difficulty_lengths,
        }
        output_path = args.output_dir / "trajectory_length_frequency.png"
        plot_trajectory_length_frequency(plot_summary, output_path)
        statistics_path = args.output_dir / "dataset_statistics.json"
        if statistics_path.is_file():
            statistics = _load_json(statistics_path)
            statistics["difficulty_modes"] = _serialize_difficulty_modes(
                difficulty_lengths
            )
            statistics["unknown_difficulty_trajectories"] = 0
            with statistics_path.open("w", encoding="utf-8") as handle:
                json.dump(statistics, handle, indent=2, ensure_ascii=False)
        print(
            f"[dataset-analysis] length plot complete trajectories="
            f"{sum(len(values) for values in difficulty_lengths.values())} "
            f"output={output_path}",
            flush=True,
        )
        return
    summary, center_heatmap = analyze_dataset(args.dataset_root, plan_root)
    save_outputs(summary, center_heatmap, args.output_dir)
    print(
        f"[dataset-analysis] complete cities={summary['cities']} "
        f"trajectories={summary['trajectories']} frames={summary['frames']} "
        f"errors={len(summary['errors'])} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
