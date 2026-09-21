"""Check the configured external inputs without running generation."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from waveverse_scene.config import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    missing = []

    def check(path, pattern=None):
        path = Path(path)
        ok = any(p.is_file() for p in path.glob(pattern)) if pattern else path.is_file()
        label = str(path / pattern) if pattern else str(path)
        print(("OK      " if ok else "MISSING ") + label)
        if not ok:
            missing.append(label)

    scene = {key: Path(value) for key, value in config["scene_assets"].items()}
    for key, pattern in (("objaverse", "*/*.pkl.gz"), ("procthor", "*.glb"),
                         ("windows", "*.glb")):
        check(scene[key], pattern)
    for part in ("frame", "door", "double", "handle"):
        check(scene["doors"], f"doorway_{part}_*.glb")
    base = scene["holodeck_base"]
    for relative in ("2023_09_23/annotations.json.gz", "2023_09_23/features/clip_features.pkl",
                     "2023_09_23/features/sbert_features.pkl",
                     "holodeck/2023_09_23/thor_object_data/annotations.json.gz",
                     "holodeck/2023_09_23/thor_object_data/clip_features.pkl",
                     "holodeck/2023_09_23/thor_object_data/sbert_features.pkl",
                     "holodeck/2023_09_23/materials/material-database.json",
                     "holodeck/2023_09_23/doors/door-database.json",
                     "holodeck/2023_09_23/windows/window-database.json"):
        check(base / relative)
    for value in config["motion_assets"].values():
        check(value)
    mesh = config["mesh_assets"]
    for value in [
        Path(mesh["body_models"]) / "smpl/SMPL_NEUTRAL.pkl",
        Path(mesh["prior"]) / "gmm_08.pkl",
        Path(mesh["prior"]) / "neutral_smpl_mean_params.h5",
        Path(mesh["segmentation"]),
    ]:
        check(value)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
