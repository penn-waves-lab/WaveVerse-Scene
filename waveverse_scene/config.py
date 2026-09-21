"""Repository-relative paths shared by the pipeline stages."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/scene.json"


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    config.pop("_comments", None)  # Documentation does not affect runtime or resume checks.
    for section in ("scene_assets", "motion_assets", "mesh_assets"):
        for key, value in config[section].items():
            p = Path(value).expanduser()
            config[section][key] = str((ROOT / p).resolve() if not p.is_absolute() else p)
    if config["resolution"] <= 0 or config["agent_radius"] < 0:
        raise ValueError("Resolution must be positive and agent radius nonnegative.")
    distance = config.setdefault("in_place_path_length_m", 0.2)
    if (
        isinstance(distance, bool) or not isinstance(distance, (int, float))
        or not math.isfinite(distance) or not 0.15 <= distance <= 0.20
    ):
        raise ValueError("in_place_path_length_m must be between 0.15 and 0.20 metres")
    from .motion_safety import settings

    config["motion_safety"] = settings(config)
    return config


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_files(paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))
