"""Conservative motion/scene collision checks and bounded placement repairs.

Scene obstacles use the bounds of the exported, Z-up meshes. These deliberately
overestimate concave furniture. This is a geometric filter, not a physics solver.
"""

import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np
from plyfile import PlyData
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .config import save_json
from .motion_files import joints_dir, meshes_dir, correspondences_dir, working_meshes_dir
from .trajectory import mesh_scene_transform


DEFAULTS = {
    "enabled": True,
    "clearance_m": 0.02,
    "contact_tolerance_m": 0.005,
    "max_penetration_m": 0.02,
    "max_collision_frame_fraction": 0.01,
    "max_shift_m": 0.30,
    "shift_step_m": 0.05,
    "limb_radius_m": 0.08,
    "torso_radius_m": 0.16,
    "head_radius_m": 0.12,
    "sample_spacing_m": 0.04,
    "temporal_spacing_m": 0.05,
    "floor_tolerance_m": 0.025,
    "max_locomotion_foot_height_m": 0.30,
    "max_locomotion_air_time_s": 0.40,
    "min_spin_rotation_degrees": 120.0,
    "max_attempts": 10,
}


class MotionSafetyError(RuntimeError):
    """A generated attempt failed the configured geometric acceptance criteria."""


def settings(config):
    supplied = config.get("motion_safety", {})
    if not isinstance(supplied, dict):
        raise ValueError("Unknown or invalid motion_safety settings")
    supplied = dict(supplied)
    # Preserve the budgets of existing saved configurations when resuming.
    if "max_regenerations" in supplied:
        retries = supplied.pop("max_regenerations")
        if (
            "max_attempts" in supplied
            or isinstance(retries, bool)
            or not isinstance(retries, int)
            or not 0 <= retries <= 10
        ):
            raise ValueError("Invalid legacy motion_safety.max_regenerations")
        supplied["max_attempts"] = retries + 1
    if set(supplied) - DEFAULTS.keys():
        raise ValueError("Unknown or invalid motion_safety settings")
    result = {**DEFAULTS, **supplied}
    if not isinstance(result["enabled"], bool):
        raise ValueError("motion_safety.enabled must be boolean")
    attempts = result["max_attempts"]
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or not 1 <= attempts <= 11
    ):
        raise ValueError(
            "motion_safety.max_attempts must be an integer from 1 to 11"
        )
    for name, value in result.items():
        if name in ("enabled", "max_attempts"):
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "motion_safety." + name + " must be finite and nonnegative"
            )
    for name in (
        "shift_step_m",
        "limb_radius_m",
        "torso_radius_m",
        "head_radius_m",
        "sample_spacing_m",
        "temporal_spacing_m",
    ):
        if result[name] <= 0:
            raise ValueError("motion_safety." + name + " must be positive")
    if result["max_collision_frame_fraction"] > 1:
        raise ValueError("motion_safety.max_collision_frame_fraction must be at most 1")
    if result["contact_tolerance_m"] > result["max_penetration_m"]:
        raise ValueError("Contact tolerance cannot exceed maximum penetration")
    if result["max_shift_m"] / result["shift_step_m"] > 20:
        raise ValueError("Shift search is limited to 20 steps per axis")
    return result


def read_mesh(path):
    ply = PlyData.read(str(path))
    vertices = np.column_stack([ply["vertex"][axis] for axis in "xyz"]).astype(float)
    faces = np.stack(ply["face"]["vertex_indices"])
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError("Invalid triangular mesh: " + str(path))
    return vertices, faces


class CollisionScene:
    def __init__(self, footprint, floor_z, ceiling_z, obstacles):
        self.footprint = footprint
        self.floor_z = floor_z
        self.ceiling_z = ceiling_z
        self.obstacles = obstacles

    @classmethod
    def from_run(cls, output):
        output = Path(output)
        scene = json.loads((output / "scene.json").read_text())
        polygons, heights = [], []
        for room in scene["rooms"]:
            points = room["floorPolygon"]
            polygons.append(Polygon([(p["x"], p["z"]) for p in points]))
            heights.extend(p["y"] for p in points)
        heights = np.asarray(heights, dtype=float)
        footprint = unary_union(polygons)
        if (
            footprint.is_empty
            or not footprint.is_valid
            or not np.isfinite(heights).all()
            or np.ptp(heights) > 1e-5
            or not np.allclose(heights, 0)
        ):
            raise ValueError(
                "Motion safety currently requires a valid, level floor at height zero"
            )
        height = float(scene.get("wall_height", scene.get("wallHeight", 2.8)))
        if not np.isfinite(height) or height <= 0:
            raise ValueError("Invalid room height")
        folder = output / "sionna_scene"
        materials = json.loads((folder / "materials.json").read_text())["objects"]
        obstacles = []
        for shape in ET.parse(folder / "scene.xml").getroot().findall("shape"):
            name = shape.get("id")
            kind = materials[name]["kind"]
            if kind in ("floor", "ceiling"):
                continue  # Test floor and ceiling explicitly, allowing normal foot contact.
            if kind not in ("object", "wall", "window", "door"):
                raise ValueError("Unknown collision geometry kind: " + kind)
            if shape.find("transform") is not None:
                raise ValueError(
                    "Collision meshes must already be in world coordinates"
                )
            path = (
                folder / shape.find("string[@name='filename']").get("value")
            ).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError("Scene mesh lies outside the exported scene folder")
            vertices, _ = read_mesh(path)
            obstacles.append((name, vertices.min(axis=0), vertices.max(axis=0)))
        return cls(footprint, float(heights[0]), float(heights[0] + height), obstacles)

    def penetrations(self, points, radii, cfg):
        """Depth inside buffered obstacle bounds; outside-room points also collide."""
        radius = np.broadcast_to(radii, (len(points),)) + cfg["clearance_m"]
        xy = shapely.points(points[:, :2])
        distance = shapely.distance(xy, self.footprint.boundary)
        signed = np.where(shapely.covers(self.footprint, xy), distance, -distance)
        depth = np.maximum(radius - signed, 0)
        sources = {"room_boundary": float(depth.max(initial=0))}
        for name, values in (
            ("floor", self.floor_z - points[:, 2] - cfg["floor_tolerance_m"]),
            ("ceiling", points[:, 2] + radius - self.ceiling_z),
        ):
            depth = np.maximum(depth, values)
            sources[name] = max(0.0, float(values.max(initial=0)))
        for name, low, high in self.obstacles:
            nearby = np.all(
                (points >= low - radius[:, None]) & (points <= high + radius[:, None]),
                axis=1,
            )
            if not nearby.any():
                continue
            # Signed Euclidean distance to an axis-aligned box (negative inside).
            delta = np.abs(points[nearby] - (low + high) / 2) - (high - low) / 2
            signed = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(
                delta.max(axis=1), 0
            )
            penetration = np.maximum(radius[nearby] - signed, 0)
            depth[nearby] = np.maximum(depth[nearby], penetration)
            sources[name] = float(penetration.max(initial=0))
        return depth, {
            k: v for k, v in sources.items() if v > cfg["contact_tolerance_m"]
        }


def joint_samples(joints, cfg):
    """Cover limbs with overlapping spheres, including between stored frames."""
    if (
        joints.ndim != 3
        or joints.shape[1:] != (22, 3)
        or not len(joints)
        or not np.isfinite(joints).all()
    ):
        raise ValueError("Expected finite joints of shape (frames, 22, 3)")
    bones = [
        (0, 2),
        (2, 5),
        (5, 8),
        (8, 11),
        (0, 1),
        (1, 4),
        (4, 7),
        (7, 10),
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
        (12, 15),
        (9, 14),
        (14, 17),
        (17, 19),
        (19, 21),
        (9, 13),
        (13, 16),
        (16, 18),
        (18, 20),
    ]
    clouds, radii = [], []
    for a, b in bones:
        length = np.linalg.norm(joints[:, b] - joints[:, a], axis=1).max()
        count = max(1, int(np.ceil(length / cfg["sample_spacing_m"])))
        if count > 100:
            raise ValueError("Implausibly long body segment")
        weight = np.linspace(0, 1, count + 1)[None, :, None]
        clouds.append(
            joints[:, a : a + 1] * (1 - weight) + joints[:, b : b + 1] * weight
        )
        kind = (
            "head" if b == 15 else ("torso" if b in (3, 6, 9, 12, 13, 14) else "limb")
        )
        # Covers the gaps between the sphere centers, plus temporal sampling gaps.
        radius = np.hypot(cfg[kind + "_radius_m"], length / (2 * count))
        radii.extend([radius + cfg["temporal_spacing_m"] / 2] * (count + 1))
    clouds = np.concatenate(clouds, axis=1)
    radii = np.asarray(radii)
    yield 0, clouds[0], radii
    for frame in range(1, len(joints)):
        movement = np.linalg.norm(clouds[frame] - clouds[frame - 1], axis=1).max()
        steps = max(1, int(np.ceil(movement / cfg["temporal_spacing_m"])))
        if steps > 200:
            raise ValueError("Motion jumps too far between frames to validate")
        for step in range(1, steps + 1):
            alpha = step / steps
            yield frame, clouds[frame - 1] * (1 - alpha) + clouds[frame] * alpha, radii


def mesh_samples(paths):
    for frame, path in enumerate(paths):
        vertices, faces = read_mesh(path)
        triangles = vertices[faces]
        centers = triangles.mean(axis=1)
        # Each sphere encloses an entire face, avoiding vertex-only missed crossings.
        radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
        yield frame, np.concatenate((vertices, centers)), np.concatenate(
            (np.zeros(len(vertices)), radii)
        )


def measure(
    scene, samples, frame_count, cfg, shift=(0.0, 0.0), *, stop_on_rejection=False
):
    maxima = np.zeros(frame_count)
    sampled = np.zeros(frame_count, dtype=bool)
    sources = {}
    for frame, points, radii in samples:
        points = points + np.array([shift[0], shift[1], 0.0])
        depth, hits = scene.penetrations(points, radii, cfg)
        maxima[frame] = max(maxima[frame], float(depth.max(initial=0)))
        sampled[frame] = True
        for name, value in hits.items():
            sources[name] = max(sources.get(name, 0), value)
        if stop_on_rejection and (
            maxima.max() > cfg["max_penetration_m"]
            or np.count_nonzero(maxima > cfg["contact_tolerance_m"]) / frame_count
            > cfg["max_collision_frame_fraction"]
        ):
            break
    if not sampled.all() and not stop_on_rejection:
        raise ValueError("Collision check did not cover every frame")
    frames = np.flatnonzero(maxima > cfg["contact_tolerance_m"])
    fraction, peak = len(frames) / frame_count, float(maxima.max())
    return {
        "accepted": bool(sampled.all())
        and peak <= cfg["max_penetration_m"]
        and fraction <= cfg["max_collision_frame_fraction"],
        "frames": frame_count,
        "checked_frames": int(sampled.sum()),
        "frame_penetration_m": maxima.tolist(),
        "collision_frames": frames.tolist(),
        "collision_frame_fraction": fraction,
        "max_penetration_m": peak,
        "obstacles": dict(sorted(sources.items(), key=lambda item: -item[1])),
    }


def candidate_shifts(cfg):
    limit, step = cfg["max_shift_m"], cfg["shift_step_m"]
    count = int(np.floor(limit / step + 1e-9))
    candidates = [
        (x * step, y * step)
        for x in range(-count, count + 1)
        for y in range(-count, count + 1)
        if 0 < np.hypot(x * step, y * step) <= limit + 1e-9
    ]
    return sorted(candidates, key=lambda delta: (np.hypot(*delta), delta))


def find_placement(scene, joints, cfg):
    samples = list(joint_samples(joints, cfg))
    before = measure(scene, samples, len(joints), cfg)
    best, offset, tried = before, (0.0, 0.0), 0
    if not before["accepted"]:
        for delta in candidate_shifts(cfg):
            tried += 1
            after = measure(
                scene, samples, len(joints), cfg, shift=delta, stop_on_rejection=True
            )
            if after["accepted"]:
                best, offset = after, delta
                break
    return {
        "before": before,
        "after": best,
        "shift_xy_m": list(offset),
        "candidates_tested": tried,
    }


def apply_placements(output, inference, offsets, *, segmentation=None):
    """Update placement only; token sequences, local joints, and SMPL poses stay intact."""
    output = Path(output)
    for motion, offset in zip(inference["motions"], offsets):
        offset = np.asarray(offset)
        if not np.any(offset):
            continue
        motion.setdefault("original_translation_xz", motion["translation_xz"])
        motion["translation_xz"] = (
            np.asarray(motion["translation_xz"]) + offset
        ).tolist()
        motion["safety_shift_xy_m"] = (
            np.asarray(motion.get("safety_shift_xy_m", [0.0, 0.0])) + offset
        ).tolist()
        raw = joints_dir(output)
        local = np.load(raw / (motion["name"] + ".npy"), allow_pickle=False)[0]
        transform = mesh_scene_transform(
            motion["translation_xz"], motion["rotation_y"],
            motion.get("coordinate_mapping", "legacy_yz_swap"))
        world = local @ transform[:3, :3].T + transform[:3, 3]
        np.save(raw / (motion["name"] + "_world.npy"), world)
        if segmentation is not None:
            import trimesh
            from .correspondence import generate_human_dict

            # Rebuild world meshes from the unchanged local meshes, so retries cannot
            # accumulate translations after an interrupted write.
            directory = meshes_dir(output, motion["name"])
            sources = sorted(working_meshes_dir(output, motion["name"]).glob("*.ply"))
            if not sources or {p.name for p in sources} != {
                p.name for p in directory.glob("*.ply")
            }:
                raise ValueError(
                    "Cannot repair meshes with missing or mismatched local/world frames"
                )
            for source in sources:
                mesh = trimesh.load(source, process=False)
                mesh.apply_transform(transform)
                mesh.export(directory / source.name)
            generate_human_dict(
                directory,
                correspondences_dir(output, motion["name"]),
                segmentation,
            )
    save_json(output / "motion/inference.json", inference)
    save_json(output / "motion_result.json", inference)


def check_joints(output, config, *, repair=False, report_path=None, collision_scene=None):
    output = Path(output)
    cfg = settings(config)
    scene = collision_scene if collision_scene is not None else CollisionScene.from_run(output)
    path = output / "motion/inference.json"
    inference = json.loads(path.read_text())
    if not inference["motions"]:
        raise ValueError("No motions to check")
    records = []
    for motion in inference["motions"]:
        world = np.load(
            joints_dir(output) / (motion["name"] + "_world.npy"),
            allow_pickle=False,
        )
        if len(world) != motion["frames"]:
            raise ValueError("World joint frame count differs from inference metadata")
        grounded = world.copy()
        grounded[..., 2] -= grounded[
            ..., 2
        ].min()  # Same sequence grounding as SMPL fitting.
        placement = find_placement(scene, grounded, cfg)
        records.append(
            {
                "name": motion["name"],
                "action": motion["action"],
                **placement,
                "total_shift_xy_m": (
                    np.asarray(motion.get("safety_shift_xy_m", [0.0, 0.0]))
                    + placement["shift_xy_m"]
                ).tolist(),
            }
        )
    accepted = all(record["after"]["accepted"] for record in records)
    report = {
        "accepted": (
            accepted if repair else all(r["before"]["accepted"] for r in records)
        ),
        "repairable": accepted,
        "applied": repair and accepted,
        "check": "joint_capsules_with_temporal_samples",
        "geometry": getattr(scene, "geometry_method", "exported_mesh_bounds"),
        "settings": cfg,
        "motions": records,
    }
    # Only apply a repair if the entire attempt passes, before recording the motion stage.
    if repair and accepted:
        apply_placements(
            output, inference, [record["shift_xy_m"] for record in records]
        )
    if report_path is not None:
        save_json(report_path, report)
    return report


def check_meshes(output, config, *, repair=False, report_path=None, collision_scene=None):
    output = Path(output)
    cfg = settings(config)
    scene = collision_scene if collision_scene is not None else CollisionScene.from_run(output)
    inference = json.loads((output / "motion/inference.json").read_text())
    fitted = json.loads((output / "motion/meshes.json").read_text())
    metadata = {record["name"]: record for record in fitted["motions"]}
    records = []
    for motion in inference["motions"]:
        count = metadata[motion["name"]]["frames"]
        if not 0 < count <= motion["frames"]:
            raise ValueError("Invalid fitted frame count")
        paths = [
            meshes_dir(output, motion["name"]) / f"{i:06d}.ply"
            for i in range(count)
        ]
        before = measure(scene, mesh_samples(paths), count, cfg)
        best, offset, tried = before, (0.0, 0.0), 0
        if not before["accepted"]:
            joints = np.load(
                joints_dir(output) / (motion["name"] + "_world.npy"),
                allow_pickle=False,
            )
            joints = joints.copy()
            joints[..., 2] -= metadata[motion["name"]]["height_offset"]
            samples = list(joint_samples(joints, cfg))
            current = np.asarray(motion.get("safety_shift_xy_m", [0.0, 0.0]))
            for total in [(0.0, 0.0)] + candidate_shifts(cfg):
                delta = np.asarray(total) - current
                if not np.any(delta):
                    continue
                tried += 1
                proxy = measure(
                    scene,
                    samples,
                    len(joints),
                    cfg,
                    shift=delta,
                    stop_on_rejection=True,
                )
                if not proxy["accepted"]:
                    continue
                after = measure(
                    scene,
                    mesh_samples(paths),
                    count,
                    cfg,
                    shift=delta,
                    stop_on_rejection=True,
                )
                if after["accepted"]:
                    best, offset = after, delta
                    break
        records.append(
            {
                "name": motion["name"],
                "action": motion["action"],
                "before": before,
                "after": best,
                "shift_xy_m": list(offset),
                "candidates_tested": tried,
                "total_shift_xy_m": (
                    np.asarray(motion.get("safety_shift_xy_m", [0.0, 0.0])) + offset
                ).tolist(),
            }
        )
    if not records:
        raise ValueError("No fitted motions to check")
    accepted = all(record["after"]["accepted"] for record in records)
    report = {
        "accepted": (
            accepted if repair else all(r["before"]["accepted"] for r in records)
        ),
        "repairable": accepted,
        "applied": repair and accepted,
        "partial": any(
            metadata[m["name"]]["frames"] != m["frames"] for m in inference["motions"]
        ),
        "check": "fitted_mesh_surface_envelopes_at_saved_frames",
        "geometry": getattr(scene, "geometry_method", "exported_mesh_bounds"),
        "settings": cfg,
        "motions": records,
    }
    if repair and accepted:
        segmentation = json.loads(
            Path(config["mesh_assets"]["segmentation"]).read_text()
        )
        random.seed(config.get("seed", 123))
        apply_placements(
            output,
            inference,
            [record["shift_xy_m"] for record in records],
            segmentation=segmentation,
        )
    if report_path is not None:
        save_json(report_path, report)
    return report


def guard_joints(output, config):
    report = check_joints(
        output,
        config,
        repair=True,
        report_path=Path(output) / "motion/safety_joints.json",
    )
    if not report["accepted"]:
        raise MotionSafetyError(
            "Generated joints cannot be placed within the collision thresholds"
        )
    return report


def guard_meshes(output, config):
    report = check_meshes(
        output,
        config,
        repair=True,
        report_path=Path(output) / "motion/safety_meshes.json",
    )
    if not report["accepted"]:
        raise MotionSafetyError("Fitted meshes exceed the collision thresholds")
    return report
