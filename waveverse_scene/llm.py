"""Adapters for the local Holodeck version and its original task prompt."""

import json
import os
from pathlib import Path
import sys

from .config import ROOT, save_json


def require_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Live scene/task generation needs OPENAI_API_KEY. Existing scene and task files can be used without an API key."
        )
    return key


def generate_room(query, output, config):
    key = require_api_key()
    os.environ["OBJATHOR_ASSETS_BASE_DIR"] = config["scene_assets"]["holodeck_base"]
    os.environ["WAVEVERSE_SCENE_LLM_MODEL"] = config["llm_model"]
    sys.path.insert(0, str(ROOT / "third_party/holodeck"))
    from ai2holodeck.generation.holodeck import Holodeck

    model = Holodeck(
        openai_api_key=key,
        openai_org=os.environ.get("OPENAI_ORG"),
        objaverse_asset_dir=config["scene_assets"]["objaverse"],
        single_room=config["single_room"],
    )
    scene, _ = model.generate_scene(
        model.get_empty_scene(),
        query,
        str(Path(output) / "holodeck"),
        add_time=False,
    )
    save_json(Path(output) / "scene.json", scene)
    return {"query": query, "rooms": len(scene["rooms"])}


def generate_tasks(output, config):
    from langchain.chat_models import ChatOpenAI
    from .scene_description import parse_scene
    from .planning import parse_tasks
    from .prompts import ACTION_DESIGN_PROMPT

    key = require_api_key()
    output = Path(output)
    query, details = parse_scene(json.loads((output / "scene.json").read_text()))
    llm = ChatOpenAI(model_name=config["llm_model"], openai_api_key=key)
    prompt = ACTION_DESIGN_PROMPT.format(
        environment_prompt=query,
        environment_details=details,
        task_number=config["task_count"],
    )
    feedback = output / ".internal/safety_feedback.json"
    if not feedback.is_file():
        feedback = output / "safety_feedback.json"  # Older interrupted runs.
    if feedback.is_file():
        failed = json.loads(feedback.read_text())
        details = []
        for report in failed.get("reports", {}).values():
            for motion in report["motions"]:
                check = motion.get("after", motion)
                if not check["accepted"]:
                    details.append(
                        {
                            "action": motion["action"],
                            "obstacles": check.get("obstacles", {}),
                            "issues": check.get("issues", []),
                        }
                    )
        prompt += (
            "\n\nThe previous human-generation attempt failed motion checks. "
            "Keep the room unchanged, but design a completely new set of human tasks and "
            "start/end positions. Prefer spacious areas, generous clearance for arms and legs, "
            "and no contact with furniture. Avoid repeating the rejected tasks. "
            "Keep the original output format.\nFailure: "
            + failed["message"]
            + "\nRejected motions, obstacles, and motion issues: "
            + json.dumps(details)
            + "\nPrevious tasks to replace: "
            + json.dumps(failed.get("tasks", []))
        )
    text = llm.predict(prompt)
    tasks = parse_tasks(text)
    (output / "human_task.txt").write_text(text)
    save_json(output / "tasks.json", tasks)
    return {"tasks": len(tasks)}
