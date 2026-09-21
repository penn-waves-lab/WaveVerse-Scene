"""Conservative floor-support and action checks, separate from collision repair."""

import json
from pathlib import Path
import re

import numpy as np

from .config import save_json
from .motion_files import joints_dir
from .motion_safety import MotionSafetyError, settings


def floor_support(joints, action, fps, cfg):
    """Reject sustained floating in locomotion; short flight phases remain valid.

    This narrow heuristic does not certify text/action agreement or validate jumps.
    Input is the pipeline's Z-up HumanML3D skeleton, with 22 joints.
    """
    joints = np.asarray(joints)
    if (joints.ndim != 3 or joints.shape[1:] != (22, 3) or not len(joints)
            or not np.isfinite(joints).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Expected finite motion joints and a positive frame rate")
    locomotion = re.search(r"\b(?:walk\w*|jog\w*|run|runs|running|sidestep\w*|march\w*)\b",
                           action, re.IGNORECASE)
    aerial = re.search(r"\b(?:jump\w*|hop|hops|hopping|skip\w*|leap\w*)\b",
                       action, re.IGNORECASE)
    if not locomotion or aerial:
        return {"checked": False, "accepted": True}
    # Match the fitter's single sequence-level grounding, without flattening motion.
    height = joints[:, [7, 8, 10, 11], 2].min(axis=1) - joints[..., 2].min()
    high = height > cfg["max_locomotion_foot_height_m"]
    boundaries = np.flatnonzero(np.diff(np.r_[False, high, False]))
    longest = int((boundaries[1::2] - boundaries[::2]).max()) if len(boundaries) else 0
    duration = longest / float(fps)
    accepted = duration <= cfg["max_locomotion_air_time_s"]
    return {
        "checked": True,
        "accepted": accepted,
        "max_both_feet_height_m": float(height.max()),
        "longest_air_time_s": duration,
        "issues": [] if accepted else ["Locomotion rises above the level floor for too long"],
    }


def spin_rotation(joints, action, cfg):
    """A named spin must rotate the body, rather than only shuffle in place."""
    if not re.search(r"\b(?:spin|spins|spinning|twirl\w*|pirouette\w*)\b", action, re.IGNORECASE):
        return {"checked": False, "accepted": True}
    # Averaging hip and shoulder axes reduces changes from upper-body gestures.
    across = ((joints[:, 2, :2] - joints[:, 1, :2])
              + (joints[:, 17, :2] - joints[:, 16, :2]))
    valid = np.linalg.norm(across, axis=1) > .10
    heading = np.unwrap(np.arctan2(across[valid, 1], across[valid, 0]))
    rotation = float(np.rad2deg(np.ptp(heading))) if len(heading) else 0.
    accepted = rotation >= cfg["min_spin_rotation_degrees"]
    return {
        "checked": True, "accepted": accepted,
        "body_rotation_degrees": rotation,
        "issues": [] if accepted else ["The requested spin does not rotate the body sufficiently"],
    }


def guard_motion_quality(output, config):
    output = Path(output)
    cfg = settings(config)
    inference = json.loads((output / "motion/inference.json").read_text())
    records = []
    if not inference["motions"]:
        raise ValueError("No motions to check")
    for motion in inference["motions"]:
        joints = np.load(joints_dir(output) / (motion["name"] + "_world.npy"),
                         allow_pickle=False)
        checks = {
            "floor_support": floor_support(joints, motion["action"], inference["fps"], cfg),
            "spin_rotation": spin_rotation(joints, motion["action"], cfg),
        }
        records.append({
            "name": motion["name"], "action": motion["action"],
            "accepted": all(c["accepted"] for c in checks.values()),
            "issues": [issue for c in checks.values() for issue in c.get("issues", [])],
            "checks": checks,
        })
    report = {"accepted": all(r["accepted"] for r in records), "motions": records}
    save_json(output / ".internal/motion_quality.json", report)
    if not report["accepted"]:
        raise MotionSafetyError("Generated motion failed floor-support or action checks")
    return report
