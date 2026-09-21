"""Build offline 3D viewers of fitted human motion inside the exported scenes."""

import argparse
import json
from pathlib import Path

from waveverse_scene.scene_rendering import export_viewer, write_gallery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A dataset root or one generated scene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter", default="")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--gallery-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing viewers")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.gallery_only:
        print(write_gallery(args.output))
        return
    if (args.input / "manifest.json").is_file():
        scenes = [args.input / s["path"] for s in json.loads((args.input / "manifest.json").read_text())["scenes"]]
    else:
        scenes = [args.input]
    for index, scene in enumerate(scenes):
        if index % args.shards != args.shard or args.filter not in scene.name:
            continue
        target = args.output / scene.name
        if not args.overwrite and (target / "render.json").is_file() and (target / "index.html").is_file():
            continue
        result = export_viewer(scene, target)
        try:
            with (target / "videos.js").open("x") as stream:
                stream.write("window.videoPreviews = [];\n")
        except FileExistsError:
            pass
        print(f"{index+1}/{len(scenes)} {scene.name}: {result['motion_count']} motions, {result['frames']} frames", flush=True)
    if args.shards == 1:
        print(write_gallery(args.output))


if __name__ == "__main__":
    main()
