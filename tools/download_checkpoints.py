"""Download and install the WaveVerse-HMG weight-only checkpoint archive."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waveverse_scene.downloads import install_archive

MANIFEST = ROOT / "configs/checkpoints.json"
DEFAULT_URL = "https://drive.google.com/file/d/1CBIdN_3oH6LlRQZcksORe2NMNDZAqDZG/view?usp=sharing"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--url", default=DEFAULT_URL, help="Override the published Google Drive checkpoint URL"
    )
    source.add_argument("--archive", type=Path, help="Install a ZIP already downloaded locally")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "assets")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace differing checkpoint files"
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.archive:
        install_archive(args.archive.resolve(), args.asset_root, manifest, args.overwrite)
        return
    try:
        import gdown
    except ImportError:
        parser.error(
            "Missing gdown. Install the release dependencies: pip install -r requirements.txt"
        )
    with tempfile.TemporaryDirectory(prefix="waveverse-hmg-download-") as work:
        archive = Path(work) / manifest["filename"]
        downloaded = gdown.download(
            url=args.url, output=str(archive), fuzzy=True, use_cookies=False
        )
        if not downloaded:
            raise RuntimeError(
                "Download failed. Check that the Google Drive file is publicly shared."
            )
        install_archive(archive, args.asset_root, manifest, args.overwrite)


if __name__ == "__main__":
    main()
