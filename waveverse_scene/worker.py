"""Optional process boundary for the historical stage environments."""

import argparse
import json
import sys
from pathlib import Path

from .config import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["room", "tasks", "motion", "meshes"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "room":
        from .llm import generate_room

        result = generate_room(args.query, args.output, config)
    elif args.stage == "tasks":
        from .llm import generate_tasks

        result = generate_tasks(args.output, config)
    elif args.stage == "motion":
        from .motion import generate_motion

        result = generate_motion(args.output, config, json.load(sys.stdin))
    else:
        from .meshes import fit_meshes

        result = fit_meshes(args.output, config, args.max_frames)
    save_json(args.output / (args.stage + "_result.json"), result)


if __name__ == "__main__":
    main()
