"""Export a generated scene or curated dataset in the WaveVerse-Sim data layout."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from waveverse_scene.config import DEFAULT_CONFIG, load_config
from waveverse_scene.sim_release import export_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="Private validation report outside the output")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.input, args.output, load_config(args.config), args.report), indent=2))


if __name__ == "__main__":
    main()
