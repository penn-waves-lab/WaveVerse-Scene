"""Verified ZIP installation shared by the asset downloaders."""

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def destination(asset_root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Invalid asset path: " + name)
    target = asset_root.joinpath(*relative.parts)
    try:
        target.resolve().relative_to(asset_root.resolve())
    except ValueError:
        raise ValueError("Asset destination leaves the asset directory: " + str(target))
    return target


def install_archive(archive, asset_root, manifest, overwrite=False):
    """Verify every member before installing; preserve differing existing files."""
    if sha256(archive) != manifest["sha256"]:
        raise ValueError("Archive SHA-256 mismatch; use the published asset ZIP.")
    expected = manifest["files"]
    asset_root = Path(asset_root).resolve()
    targets = {name: destination(asset_root, name) for name in expected}
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != len(expected) or {info.filename for info in members} != set(expected):
            raise ValueError("Archive members do not match the asset manifest.")
        for info in members:
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Expected a regular asset file: " + info.filename)
            if info.file_size != expected[info.filename]["bytes"]:
                raise ValueError("Asset size mismatch: " + info.filename)
        asset_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".asset-install-", dir=str(asset_root)) as work:
            staging = Path(work)
            for name, metadata in expected.items():
                staged = staging / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, staged.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if sha256(staged) != metadata["sha256"]:
                    raise ValueError("Asset SHA-256 mismatch: " + name)
            pending = []
            for name, target in targets.items():
                if target.exists():
                    if not target.is_file():
                        raise FileExistsError(
                            "Asset destination is not a file: " + str(target)
                        )
                    if sha256(target) == expected[name]["sha256"]:
                        continue
                    if not overwrite:
                        raise FileExistsError(
                            "A different asset already exists: "
                            + str(target)
                            + ". Use a new --asset-root or --overwrite."
                        )
                pending.append((name, target))
            for name, target in pending:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(staging / name), str(target))
    print(f"Installed {len(pending)} files; {len(targets) - len(pending)} already present in {asset_root}")
    return targets
