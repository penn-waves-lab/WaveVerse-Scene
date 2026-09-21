"""Current motion folders, with read support for existing historical outputs."""

from pathlib import Path


def joints_dir(run):
    root = Path(run) / "motion"
    current, previous = root / "joints", root / "raw_human_mesh"
    return current if current.exists() or not previous.exists() else previous


def meshes_dir(run, name):
    root = Path(run) / "motion"
    current, previous = root / "meshes" / name, root / "refine_human_mesh" / name / "mesh"
    return current if current.exists() or not previous.exists() else previous


def correspondences_dir(run, name):
    root = Path(run) / "motion"
    current, previous = root / "correspondences" / name, root / "corr_human_mesh" / name
    return current if current.exists() or not previous.exists() else previous


def working_meshes_dir(run, name):
    """Local meshes needed only until collision repair finishes."""
    current = Path(run) / "motion/.mesh_work" / name
    previous = Path(run) / "motion/raw_human_mesh" / name / "mesh"
    return current if current.exists() or not previous.exists() else previous
