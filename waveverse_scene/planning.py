"""Convert scene-aware tasks to the original 64-point path representation."""

import json
import re
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from .config import save_json
from .scene_description import parse_scene
from .trajectory import resample_trajectory
from .geometry.floor_plan_2d import FloorPlan2D
from .geometry.holodeck_scene import HoloDeckScene
from .navigation import (
    adjust_point_with_margin_and_direction,
    find_path,
    get_object_coordinates,
)


def load_scene(path, config):
    scene = HoloDeckScene.load(Path(path))
    assets = config["scene_assets"]
    for attribute, key in (
        ("objaverse_assets_dir", "objaverse"),
        ("proc_thor_assets_dir", "procthor"),
        ("door_assets_dir", "doors"),
        ("window_assets_dir", "windows"),
    ):
        setattr(scene, attribute, Path(assets[key]))
    return scene


def parse_tasks(text):
    """Read the task format emitted by the original action-design prompt."""
    number = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
    pattern = re.compile(
        rf"from position\s*\(\s*({number})\s*,\s*({number})\s*\)"
        rf"\s*to position\s*\(\s*({number})\s*,\s*({number})\s*\)",
        re.IGNORECASE,
    )
    stationary = re.compile(
        rf"\bat position\s*\(\s*({number})\s*,\s*({number})\s*\)",
        re.IGNORECASE,
    )
    tasks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.search(line)
        if not match and re.search(r"\bin[\s-]+place\b", line, re.IGNORECASE):
            match = stationary.search(line)
        if not match:
            raise ValueError("Task is missing 'from position (x, z) to position (x, z)': " + line)
        description = re.sub(r"^\d+\.\s*", "", line[:match.start()])
        action = re.split(r"\bfrom\b", description, maxsplit=1, flags=re.IGNORECASE)[0]
        # Free-floor labels belong to placement, not the motion encoder's text.
        action = re.sub(r"\s+at\s+(?:the\s+)?'open floor[^']*'\s*[, .]*$",
                        "", action, flags=re.IGNORECASE).strip(" ,")
        points = [float(v) for v in match.groups()]
        if len(points) == 2:
            points *= 2
        objects = re.findall(r"'([^']+)'", line)
        tasks.append(
            {
                "action": action,
                "start": points[:2],
                "end": points[2:],
                "objects": objects if len(objects) == 2 else None,
            }
        )
    if not tasks:
        raise ValueError("No human tasks were provided.")
    return tasks


def read_tasks(path):
    path = Path(path)
    if path.suffix.lower() != ".json":
        return parse_tasks(path.read_text())
    tasks = json.loads(path.read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Tasks JSON must contain a nonempty list.")
    result = []
    for task in tasks:
        if isinstance(task, list) and len(task) == 2:
            task = {"action": task[0], "path": task[1]}
        if (
            not isinstance(task, dict)
            or not isinstance(task.get("action"), str)
            or not task["action"].strip()
        ):
            raise ValueError("Each task needs a nonempty action string.")
        result.append(task)
    return result


def in_place_path(floor, start, end, config):
    """Give stationary actions a short, obstacle-free heading cue in metres."""
    from .motion_safety import MotionSafetyError

    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    heading = end - start
    angle = np.arctan2(heading[1], heading[0]) if np.linalg.norm(heading) > 1e-6 else np.pi / 2
    distance = config.get("in_place_path_length_m", 0.2)
    radius = config["agent_radius"] + config.get("motion_safety", {}).get("clearance_m", 0.02)
    free = floor.get_navigable_space()
    # Prefer the requested heading; use nearby headings when it is blocked.
    for turn in (0, 1, -1, 2, -2, 3, -3, 4):
        direction = np.array([np.cos(angle + turn * np.pi / 4), np.sin(angle + turn * np.pi / 4)])
        for length in dict.fromkeys((distance, 0.15)):
            target = start + length * direction
            segment = LineString([start, target])
            footprint = segment.buffer(radius) if radius else segment
            if free.covers(footprint):
                return np.array([start, target])
    raise MotionSafetyError("No free space for an in-place motion")


def plan_paths(scene_path, task_path, output, config, *, return_paths=False):
    scene = load_scene(scene_path, config)
    floor = FloorPlan2D()
    floor.initialize_from_scene(scene)
    tasks = read_tasks(task_path)
    details = None
    planned = []
    for task in tasks:
        if "path" in task:
            path = np.asarray(task["path"], dtype=float)
            if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
                raise ValueError("A supplied path must have at least two finite [x, z] points.")
            if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < 1e-6:
                path = in_place_path(floor, path[0], path[-1], config)
        else:
            points = np.asarray([task.get("start"), task.get("end")], dtype=float)
            if points.shape != (2, 2) or not np.isfinite(points).all():
                raise ValueError("Each task needs finite start and end [x, z] coordinates.")
            start, end = points
            stationary = (
                re.search(r"\bin[\s-]+place\b", task["action"], re.IGNORECASE)
                or np.linalg.norm(end - start) < 1e-6
            )
            if stationary:
                path = in_place_path(floor, start, end, config)
                planned.append([task["action"], resample_trajectory(path, number=64).tolist()])
                continue
            if task.get("objects"):
                if details is None:
                    _, details = parse_scene(json.loads(Path(scene_path).read_text()))
                for i, name in enumerate(task["objects"]):
                    bbox = get_object_coordinates(details, name)
                    if bbox is not None and len(bbox) == 2:
                        bbox = [(p[0], p[2]) if len(p) == 3 else (p[0], p[1]) for p in bbox]
                        if i == 0:
                            start = adjust_point_with_margin_and_direction(
                                start, bbox, end, margin=0.05
                            )
                        else:
                            end = adjust_point_with_margin_and_direction(
                                end, bbox, start, margin=0.05
                            )
            path = np.asarray(
                find_path(floor, start, end, config["agent_radius"], config["resolution"])
            )
            if len(path) == 0:
                from .motion_safety import MotionSafetyError

                raise MotionSafetyError("No traversable path for task: " + task["action"])
        # Use the same arc-length interpolation as the original inference pipeline.
        path = resample_trajectory(path, number=64)
        planned.append([task["action"], path.tolist()])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid = floor.to_grid(config["resolution"])
    np.save(output / "occupancy_grid.npy", grid)
    save_json(
        output / "occupancy_grid.json",
        {
            "resolution": config["resolution"],
            "min_x": floor.bounds.min_x,
            "min_z": floor.bounds.min_y,
            "occupied": 1,
            "free": 0,
            "axis_order": ["z", "x"],
        },
    )
    result = {"tasks": len(planned), "grid_shape": list(grid.shape)}
    return (result, planned) if return_paths else result
