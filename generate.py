"""Generate a room, human paths, motion, and animated SMPL meshes."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from waveverse_scene.config import (
    DEFAULT_CONFIG,
    ROOT,
    file_hash,
    load_config,
    save_json,
)
from waveverse_scene.materials import material_input_hashes
from waveverse_scene.motion_safety import MotionSafetyError, guard_joints, guard_meshes, settings
from waveverse_scene.motion_quality import guard_motion_quality
from waveverse_scene.motion_files import joints_dir, meshes_dir, correspondences_dir
from waveverse_scene.safety_retry import run_with_retries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", help="Room description for live Holodeck generation")
    source.add_argument("--scene", type=Path, help="Existing Holodeck scene JSON")
    parser.add_argument(
        "--tasks",
        type=Path,
        help="Tasks JSON or original human_task.txt; otherwise use the LLM",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scene")
    parser.add_argument(
        "--stop-after", choices=["room", "paths", "motion", "meshes"], default="meshes"
    )
    parser.add_argument(
        "--mesh-frames",
        type=int,
        help="Fit only this many frames per motion for a smoke test",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a matching run after validating its completed outputs",
    )
    args = parser.parse_args()
    if args.mesh_frames is not None and args.mesh_frames < 1:
        parser.error("--mesh-frames must be positive")
    config = load_config(args.config)
    output = args.output.expanduser().resolve()
    signature = {
        "query": args.query,
        "scene_sha256": file_hash(args.scene) if args.scene else None,
        "tasks_sha256": file_hash(args.tasks) if args.tasks else None,
        "config": config,
        "mesh_frames": args.mesh_frames,
        "material_input_sha256": material_input_hashes(config),
    }
    state_path = output / ".internal/pipeline.json"
    legacy_state = output / "pipeline.json"
    if not state_path.is_file() and legacy_state.is_file():
        state_path = legacy_state
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            parser.error(
                "Output directory is not empty. Choose another --output or use --resume."
            )
        if not state_path.is_file():
            parser.error("Output has no saved pipeline state; choose a new directory.")
        state = json.loads(state_path.read_text())
        previous = dict(state["inputs"])
        previous["config"] = dict(previous["config"])
        previous["config"]["motion_safety"] = settings(previous["config"])
        if previous != signature:
            old_limit = previous["config"]["motion_safety"]["max_attempts"]
            new_limit = config["motion_safety"]["max_attempts"]
            previous["config"]["motion_safety"]["max_attempts"] = new_limit
            if new_limit <= old_limit or previous != signature:
                parser.error("Inputs or settings changed. Use a new output directory.")
            # An explicit larger budget may continue an exhausted run. Keep
            # all prior attempt indices, conditioning hashes, and room outputs.
            state.setdefault("budget_updates", []).append({
                "from": old_limit, "to": new_limit,
                "recorded_attempts": len(state.get("history", {})),
            })
            state["inputs"] = signature
            save_json(state_path, state)
    else:
        output.mkdir(parents=True, exist_ok=True)
        state = {"inputs": signature, "stages": {}}
        save_json(state_path, state)
    internal = output / ".internal"
    internal.mkdir(exist_ok=True)
    print("Generating...", flush=True)
    with (internal / "generation.log").open("w", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                generate(args, config, output, state, state_path)
            except BaseException:
                traceback.print_exc()
                raise
    print("Generation completed:", output, flush=True)


def generate(args, config, output, state, state_path):
    config_path = output / ".internal/config.json"
    save_json(config_path, config)
    planned_paths = None

    def stage(name, run, artifacts):
        if name in state["stages"]:
            for relative, digest in state["stages"][name]["files"].items():
                path = output / relative
                if not path.is_file() or file_hash(path) != digest:
                    raise RuntimeError(
                        "Completed output is missing or changed: " + str(path)
                    )
            print("Already completed:", name, flush=True)
            return state["stages"][name]["result"]
        print("Running:", name, flush=True)
        started = time.monotonic()
        result = run()
        files = list(artifacts())
        if not files:
            raise RuntimeError("Stage produced no artifacts: " + name)
        state["stages"][name] = {
            "seconds": time.monotonic() - started,
            "result": result,
            "files": {str(p.relative_to(output)): file_hash(p) for p in files},
        }
        save_json(state_path, state)
        return result

    def worker(name):
        nonlocal planned_paths
        # Retrying changes the task set as well as the recorded inference seed.
        runtime_config = dict(
            config, seed=config["seed"] + state.get("safety_attempt", 0)
        )
        runtime_path = output / ".internal/runtime.json"
        save_json(runtime_path, runtime_config)
        key = (
            "holodeck_python"
            if name in ("room", "tasks")
            else ("motion_python" if name == "motion" else "mesh_python")
        )
        command = [
            config.get(key) or sys.executable,
            "-u",
            "-m",
            "waveverse_scene.worker",
            name,
            "--output",
            str(output),
            "--config",
            str(runtime_path),
        ]
        if name == "room":
            command += ["--query", args.query]
        if name == "meshes" and args.mesh_frames is not None:
            command += ["--max-frames", str(args.mesh_frames)]
        process_input = {}
        if name == "motion":
            if planned_paths is None:
                _, planned_paths = compute_paths()
            # Send model conditioning directly to the worker, without a path file.
            process_input = {"input": json.dumps(planned_paths), "text": True}
        subprocess.run(
            command, cwd=ROOT, check=True, stdout=sys.stdout, stderr=sys.stderr,
            **process_input,
        )
        return json.loads((output / (name + "_result.json")).read_text())

    def room():
        if args.scene:
            scene = json.loads(args.scene.read_text())
            if not scene.get("rooms"):
                raise ValueError("Scene JSON contains no rooms.")
            save_json(output / "scene.json", scene)
            return {"source": "existing scene", "rooms": len(scene["rooms"])}
        return worker("room")

    def tasks():
        if args.tasks and state.get("safety_attempt", 0) == 0:
            from waveverse_scene.planning import read_tasks

            parsed = read_tasks(args.tasks)
            # User-provided trajectories remain in the original input file.
            # The run keeps only descriptions and endpoint positions.
            descriptions = []
            for task in parsed:
                task = dict(task)
                path = task.pop("path", None)
                if path:
                    task.update(start=path[0], end=path[-1])
                descriptions.append(task)
            save_json(output / "tasks.json", descriptions)
            return {"source": "existing tasks", "tasks": len(parsed)}
        return worker("tasks")

    stage("room", room, lambda: [output / "scene.json"])
    from waveverse_scene.scene_export import export_scene

    stage(
        "export",
        lambda: export_scene(output / "scene.json", output, config),
        lambda: sorted((output / "sionna_scene").glob("*")),
    )
    if args.stop_after == "room":
        return
    from waveverse_scene.planning import plan_paths

    safety = config["motion_safety"]["enabled"]

    def compute_paths():
        task_path = args.tasks if args.tasks and state.get("safety_attempt", 0) == 0 else output / "tasks.json"
        return plan_paths(output / "scene.json", task_path, output, config, return_paths=True)

    def paths():
        nonlocal planned_paths
        result, planned_paths = compute_paths()
        result["conditioning_sha256"] = hashlib.sha256(
            json.dumps(planned_paths, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        return result

    def motion_artifacts():
        return (
            [output / "motion/inference.json"]
            + sorted(joints_dir(output).glob("*.npy"))
            + ([output / "motion/safety_joints.json"] if safety else [])
        )

    def motion():
        # A worker interrupted before the stage was committed may leave partial
        # frames. Rebuild this attempt without retaining stale motion/mesh files.
        for name in ("joints", "meshes", "correspondences", ".mesh_work",
                     "raw_human_mesh", "refine_human_mesh", "corr_human_mesh"):
            shutil.rmtree(output / "motion" / name, ignore_errors=True)
        for name in (
            "inference.json",
            "meshes.json",
            "safety_joints.json",
            "safety_meshes.json",
        ):
            (output / "motion" / name).unlink(missing_ok=True)
        worker("motion")
        if safety:
            guard_joints(output, config)
        return json.loads((output / "motion/inference.json").read_text())

    def meshes():
        if safety:
            # A mesh repair also moves world joints/placement metadata. Journal
            # that dependency so an interrupted repair is regenerated on resume.
            state["pending_mesh_placement"] = True
            save_json(state_path, state)
        try:
            result = worker("meshes")
            if safety:
                guard_meshes(output, config)
                state["stages"]["motion"]["result"] = json.loads(
                    (output / "motion/inference.json").read_text()
                )
                state["stages"]["motion"]["files"] = {
                    str(p.relative_to(output)): file_hash(p) for p in motion_artifacts()
                }
                state.pop("pending_mesh_placement")
                save_json(state_path, state)
            return result
        finally:
            # The fitting round-trip and any placement repair have finished.
            # Interrupted attempts regenerate these files when they resume.
            shutil.rmtree(output / "motion/.mesh_work", ignore_errors=True)

    def mesh_artifacts():
        records = json.loads((output / "motion/meshes.json").read_text())["motions"]
        files = [output / "motion/meshes.json"]
        if safety:
            files.append(output / "motion/safety_meshes.json")
        for record in records:
            files += sorted(meshes_dir(output, record["name"]).glob("*.ply"))
            files += sorted(correspondences_dir(output, record["name"]).glob("*.pkl"))
        return files

    def attempt(number):
        if state.pop("pending_mesh_placement", False):
            state["stages"].pop("motion", None)
            state["stages"].pop("meshes", None)
            save_json(state_path, state)
        stage("tasks", tasks, lambda: [output / "tasks.json"])
        stage(
            "paths",
            paths,
            lambda: [
                output / "occupancy_grid.npy",
                output / "occupancy_grid.json",
            ],
        )
        if args.stop_after == "paths":
            return
        stage("motion", motion, motion_artifacts)
        if safety:
            # Check resumed outputs too: older runs only checked collisions.
            guard_motion_quality(output, config)
        if args.stop_after == "motion":
            return
        stage(
            "meshes",
            meshes,
            mesh_artifacts,
        )

    run_with_retries(
        output, config, state, lambda: save_json(state_path, state), attempt
    )


def cli():
    try:
        main()
    except MotionSafetyError:
        print("Motion generation failed.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Generation interrupted.", file=sys.stderr)
        return 130
    except Exception:
        print("Generation failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli())
