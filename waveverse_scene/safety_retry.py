"""Persistent, bounded retries of human generation; the exported room stays fixed."""

import json
from pathlib import Path
import shutil

from .config import file_hash, save_json
from .motion_safety import MotionSafetyError, settings


HUMAN_STAGES = ("tasks", "paths", "motion", "meshes")
ATTEMPT_FILES = (
    "tasks.json",
    "human_task.txt",
    "tasks_result.json",
    "motion_result.json",
    "meshes_result.json",
    "occupancy_grid.npy",
    "occupancy_grid.json",
    "motion",
    "visualizations",
    "attempt_config.json",
    ".internal/attempt_config.json",
    ".internal/runtime.json",
    ".internal/motion_quality.json",
)


def finish_pending_retry(output, state, save_state):
    """Finish journaled cleanup safely, including after an interrupted cleanup."""
    pending = state.get("pending_safety_retry")
    if pending is None:
        return
    output = Path(output)
    save_json(output / ".internal/safety_feedback.json", pending["failure"])
    for name in ATTEMPT_FILES:
        source = output / name
        if source.is_dir() and not source.is_symlink():
            shutil.rmtree(source)
        else:
            source.unlink(missing_ok=True)
    (output / "safety_feedback.json").unlink(missing_ok=True)
    exhausted = pending.get("exhausted", False)
    state["safety_attempt"] = pending["attempt"] + (0 if exhausted else 1)
    state.pop("pending_safety_retry")
    if exhausted:
        state["safety_failure"] = pending["failure"]
    else:
        state.pop("safety_failure", None)
    state.pop("pending_mesh_placement", None)
    state["status"] = "rejected" if exhausted else "running"
    save_state()


def run_with_retries(output, config, state, save_state, run_attempt):
    """Retry rejected paths, collisions, and motion checks within one shared budget."""
    output = Path(output)
    cfg = settings(config)
    finish_pending_retry(output, state, save_state)
    if state.get("safety_failure"):
        attempt = state.get("safety_attempt", 0)
        if attempt + 1 >= cfg["max_attempts"]:
            raise MotionSafetyError("Motion generation failed.")
        # Resuming with a larger budget starts the next attempt, not attempt 0.
        state["pending_safety_retry"] = {
            "attempt": attempt, "failure": state["safety_failure"], "exhausted": False,
        }
        save_state()
        finish_pending_retry(output, state, save_state)

    def record_result(number, outcome):
        files = [output / "tasks.json"]
        files += sorted((output / "motion/joints").glob("*.npy"))
        state.setdefault("history", {})[str(number)] = {
            "outcome": outcome,
            "conditioning_sha256": state["stages"].get("paths", {}).get(
                "result", {}
            ).get("conditioning_sha256"),
            "files": {
                str(path.relative_to(output)): file_hash(path)
                for path in files if path.is_file()
            },
        }

    while True:
        attempt = state.get("safety_attempt", 0)
        try:
            result = run_attempt(attempt)
        except MotionSafetyError as error:
            if not cfg["enabled"]:
                raise
            record_result(attempt, "rejected")
            reports = {}
            for name in ("safety_joints", "safety_meshes"):
                path = output / "motion" / (name + ".json")
                if path.is_file():
                    reports[name] = json.loads(path.read_text())
            quality = output / ".internal/motion_quality.json"
            if quality.is_file():
                reports["motion_quality"] = json.loads(quality.read_text())
            failure = {"attempt": attempt, "message": str(error), "reports": reports}
            if (output / "tasks.json").is_file():
                failure["tasks"] = json.loads((output / "tasks.json").read_text())
            state["safety_failure"] = failure
            state["status"] = "rejected"
            exhausted = attempt + 1 >= cfg["max_attempts"]
            state["pending_safety_retry"] = {
                "attempt": attempt, "failure": failure, "exhausted": exhausted
            }
            for name in HUMAN_STAGES:
                state["stages"].pop(name, None)
            save_state()
            finish_pending_retry(output, state, save_state)
            if exhausted:
                raise MotionSafetyError("Motion generation failed.") from None
            continue
        record_result(attempt, "accepted")
        state["status"] = "complete"
        save_state()
        (output / ".internal/safety_feedback.json").unlink(missing_ok=True)
        return result
