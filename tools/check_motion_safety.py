"""Audit an existing run without changing its joints, meshes, or tasks."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from waveverse_scene.config import DEFAULT_CONFIG, load_config, save_json
from waveverse_scene.motion_safety import check_joints, check_meshes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mesh", action="store_true", help="Also check fitted mesh surfaces"
    )
    parser.add_argument(
        "--report", type=Path, help="Default: <input>/safety_audit.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    results = {"joints": check_joints(args.input, config)}
    if args.mesh:
        results["meshes"] = check_meshes(args.input, config)
    target = args.report or args.input / "safety_audit.json"
    save_json(target, results)
    for mode, report in results.items():
        for motion in report["motions"]:
            before = motion["before"]
            print(
                f"{mode} {motion['name']}: peak={before['max_penetration_m']:.4f} m, "
                f"colliding frames={len(before['collision_frames'])}/{before['frames']}, "
                f"accepted={before['accepted']}, repairable={motion['after']['accepted']}, "
                f"suggested shift={motion['shift_xy_m']}"
            )
    print("Report:", target.resolve())
    return 0 if all(report["accepted"] for report in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
