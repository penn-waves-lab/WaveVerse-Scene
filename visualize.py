"""Preview generated motion without a GPU, checkpoints, or an API key."""

import argparse
from pathlib import Path

from waveverse_scene.visualization import visualize_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Generated run directory")
    parser.add_argument("--output", type=Path, help="Default: <input>/visualizations")
    parser.add_argument("--motion", type=int, help="Zero-based motion index; default: all")
    parser.add_argument("--format", choices=("html", "gif", "mp4"), default="html")
    parser.add_argument("--mesh", action="store_true", help="Show fitted SMPL meshes instead of joints")
    parser.add_argument("--no-scene", action="store_true", help="Hide the floor-plan/path panel")
    parser.add_argument("--stride", type=int, default=1, help="Show every Nth frame at the original speed")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    if args.motion is not None and args.motion < 0:
        parser.error("--motion must be nonnegative")
    try:
        gallery = visualize_run(
            args.input,
            output=args.output,
            motion_index=args.motion,
            file_format=args.format,
            mesh=args.mesh,
            show_scene=not args.no_scene,
            stride=args.stride,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f"Visualization failed: {error}\n")
    print("Open in your browser:", gallery)


if __name__ == "__main__":
    main()
