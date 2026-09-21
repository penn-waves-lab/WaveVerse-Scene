"""Install the converted ProcTHOR, door, and window meshes under assets/."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waveverse_scene.downloads import install_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Previously downloaded scene asset ZIP")
    source.add_argument("--url", help="Google Drive URL for the published asset ZIP")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "assets")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/scene_assets.json").read_text())
    if args.archive:
        install_archive(args.archive, args.asset_root, manifest, args.overwrite)
        return
    url = args.url or manifest.get("url")
    if not url:
        parser.error("No download URL configured. Use --archive WaveVerse-Scene-assets.zip.")
    import gdown

    with tempfile.TemporaryDirectory(prefix="waveverse-scene-download-") as work:
        archive = Path(work) / manifest["filename"]
        if not gdown.download(url=url, output=str(archive), fuzzy=True, use_cookies=False):
            raise RuntimeError("Asset download failed; use --archive for a local copy.")
        install_archive(archive, args.asset_root, manifest, args.overwrite)


if __name__ == "__main__":
    main()
