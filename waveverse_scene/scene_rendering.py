"""Read saved room geometry and fitted motion for world-space inspection renders."""

import base64
from html import escape
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from plyfile import PlyData

from .config import ROOT, save_json
from .motion_safety import read_mesh
from .motion_files import joints_dir, meshes_dir


def clean_id(value):
    return re.sub(r"\W+", " ", value).strip().replace(" ", "_")


def vertex_normals(vertices, faces):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    result = np.zeros_like(vertices, dtype=np.float32)
    for corner in range(3):
        np.add.at(result, faces[:, corner], normals)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result


def read_colored_mesh(path):
    ply = PlyData.read(str(path))
    vertices = np.column_stack([ply["vertex"][axis] for axis in "xyz"]).astype(np.float32)
    faces = np.stack(ply["face"]["vertex_indices"]).astype(np.uint32)
    if all(key in ply["vertex"].data.dtype.names for key in ("red", "green", "blue")):
        colors = np.column_stack([ply["vertex"][key] for key in ("red", "green", "blue")])
        if not np.isfinite(colors).all():
            raise ValueError("Invalid vertex colors: " + str(path))
        # Sionna exports normalized float RGB; conventional PLY uses byte RGB.
        if np.issubdtype(colors.dtype, np.floating) and colors.max() <= 1.:
            colors = colors * 255
        colors = np.rint(np.clip(colors, 0, 255)).astype(np.uint8)
    else:
        colors = np.tile(np.array([185, 190, 194], dtype=np.uint8), (len(vertices), 1))
    if not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Invalid render mesh: " + str(path))
    return vertices, faces, colors


def static_meshes(run):
    run = Path(run)
    scene = json.loads((run / "scene.json").read_text())
    identities = []
    for group, kind in (("rooms", "room"), ("walls", "wall"), ("doors", "door"),
                        ("windows", "window"), ("objects", "object")):
        identities.extend((clean_id(item["id"]), kind) for item in scene.get(group, []))
    identities.sort(key=lambda item: -len(item[0]))
    folder = run / "sionna_scene"
    materials_file = folder / "materials.json"
    materials = json.loads(materials_file.read_text())["objects"] if materials_file.exists() else {}
    meshes = []
    for shape in ET.parse(folder / "scene.xml").getroot().findall("shape"):
        name = shape.get("id")
        if shape.find("transform") is not None:
            raise ValueError("Render input must already be in world coordinates")
        kind = materials.get(name, {}).get("kind") or next(
            (kind for key, kind in identities if name == key or name.startswith(key + "_")), None)
        if kind is None:
            raise ValueError("Unknown scene object: " + name)
        if kind == "ceiling":
            continue
        path = (folder / shape.find("string[@name='filename']").get("value")).resolve()
        path.relative_to(folder.resolve())
        vertices, faces, colors = read_colored_mesh(path)
        if kind == "room":
            # Historical room PLYs combine floor and ceiling. Display the floor;
            # the source file is never rewritten or transformed.
            levels = vertices[faces, 2]
            threshold = float((levels.min() + levels.max()) / 2)
            if levels.max() - levels.min() > .2:
                faces = faces[np.all(levels < threshold, axis=1)]
            kind = "floor"
        if not len(faces):
            continue
        used, inverse = np.unique(faces, return_inverse=True)
        vertices, colors, faces = vertices[used], colors[used], inverse.reshape(-1, 3).astype(np.uint32)
        meshes.append({"name": name, "kind": kind, "vertices": vertices, "faces": faces,
                       "colors": colors, "normals": vertex_normals(vertices, faces)})
    return meshes


def motion_info(run):
    run = Path(run)
    inference = json.loads((run / "motion/inference.json").read_text())
    fitted = {m["name"]: m for m in json.loads((run / "motion/meshes.json").read_text())["motions"]}
    safety_path = run / "motion/safety_meshes.json"
    safety = json.loads(safety_path.read_text()) if safety_path.exists() else {}
    checks = {m["name"]: m for m in safety.get("motions", [])}
    check_state = "after" if safety.get("applied") else "before"
    result = []
    for index, motion in enumerate(inference["motions"]):
        name = motion["name"]
        count = fitted[name]["frames"]
        root = np.load(joints_dir(run) / (name + "_world.npy"), allow_pickle=False)[:count, 0].copy()
        root[:, 2] -= fitted[name]["height_offset"]
        check = checks.get(name, {}).get(check_state, {})
        overlaps = check.get("frame_penetration_m", [0.] * count)
        result.append({"name": name, "action": motion["action"], "frames": count,
                       "fps": inference["fps"], "roots": root.tolist(),
                       "shift_xy_m": motion.get("safety_shift_xy_m", [0., 0.]),
                       "frame_overlap_m": overlaps, "worst_frame": int(np.argmax(overlaps)),
                       "partial": count != motion["frames"]})
    return result


def encoded(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def export_viewer(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meshes = static_meshes(run)
    bounds = np.array([np.concatenate([m["vertices"] for m in meshes]).min(axis=0),
                       np.concatenate([m["vertices"] for m in meshes]).max(axis=0)])
    data = {"name": run.name, "query": json.loads((run / "scene.json").read_text()).get("query", run.name),
            "bounds": bounds.tolist(), "meshes": [], "motions": []}
    for mesh in meshes:
        data["meshes"].append({"name": mesh["name"], "kind": mesh["kind"],
                               "positions": encoded(mesh["vertices"], "<f4"),
                               "normals": encoded(mesh["normals"], "<f4"),
                               "colors": encoded(mesh["colors"], "u1"), "indices": encoded(mesh["faces"], "<u4")})
    records = motion_info(run)
    for record in records:
        directory = meshes_dir(run, record["name"])
        frames, faces_by_frame = [], []
        for index in range(record["frames"]):
            vertices, faces = read_mesh(directory / f"{index:06d}.ply")
            frames.append(vertices.astype(np.float32))
            faces_by_frame.append(faces.astype(np.uint32))
        count = len(frames[0])
        if any(len(vertices) != count for vertices in frames):
            raise ValueError("Variable vertex count is unsupported: " + record["name"])
        common_faces = all(np.array_equal(faces, faces_by_frame[0]) for faces in faces_by_frame)
        data["motions"].append({**record, "vertices_per_frame": count,
                                "positions": encoded(np.stack(frames), "<f4"),
                                "indices": encoded(faces_by_frame[0], "<u4"),
                                "indices_per_frame": None if common_faces else [encoded(f, "<u4") for f in faces_by_frame]})
    template = (ROOT / "waveverse_scene/templates/scene_motion.html").read_text()
    document = template.replace("__SCENE_TITLE__", escape(data["query"]))
    document = document.replace("__SCENE_DATA__", json.dumps(data, separators=(",", ":")).replace("<", "\\u003c"))
    (output / "index.html").write_text(document)
    result = {"scene": run.name, "query": data["query"], "motion_count": len(records),
              "frames": sum(m["frames"] for m in records), "source_geometry_unchanged": True,
              "render_coordinates": "saved Z-up world vertices; no placement transforms applied",
              "ceiling_hidden": True, "motions": records}
    save_json(output / "render.json", result)
    return result


def write_gallery(output):
    output = Path(output)
    entries = [json.loads(p.read_text()) for p in output.glob("*/render.json")]
    entries.sort(key=lambda entry: (-max(max(m["frame_overlap_m"]) for m in entry["motions"]), entry["scene"]))
    cards = []
    clip_count = 0
    for entry in entries:
        name = entry["scene"]
        peak = max(max(m["frame_overlap_m"]) for m in entry["motions"]) * 100
        poster = name + "/motion_000000.png"
        video_files = {}
        for metadata_name in ("media.json", "preview.json"):
            metadata = output / name / metadata_name
            if metadata.is_file():
                for record in json.loads(metadata.read_text())["motions"]:
                    filename = record.get("video")
                    if filename and (output / name / filename).is_file():
                        video_files[record["motion"]] = name + "/" + filename
        choices = []
        for index, motion in enumerate(entry["motions"]):
            if motion["name"] in video_files:
                filename = video_files[motion["name"]]
                image = name + "/" + motion["name"] + ".png"
                choices.append((filename, image, f"{index+1}. {motion['action']}", motion["action"]))
        clip_count += len(choices)
        if choices:
            options = ''.join(f'<option value="{escape(filename, quote=True)}" '
                              f'data-poster="{escape(image, quote=True)}" '
                              f'data-description="{escape(description, quote=True)}">{escape(label)}</option>'
                              for filename, image, label, description in choices)
            selector = (f'<label class="sequence">Sequence <select>{options}</select></label>'
                        if len(choices) > 1 else '')
            preview = (f'<video autoplay muted loop playsinline controls preload="none" '
                       f'data-src="{escape(choices[0][0], quote=True)}" '
                       f'poster="{escape(choices[0][1], quote=True)}" '
                       f'aria-label="{escape(entry["query"], quote=True)} motion preview"></video>'
                       '<button class="fallback" hidden>Play preview</button>'
                       f'{selector}'
                       '<p class="motion-description"><strong>Motion:</strong> '
                       f'<span>{escape(choices[0][3])}</span></p>')
        else:
            preview = (f'<img loading="lazy" src="{escape(poster, quote=True)}" alt="Human motion in the scene">'
                       if (output / poster).is_file() else '')
        viewer = escape(name, quote=True) + "/index.html"
        cards.append(f'<article data-scene="{escape(name, quote=True)}" '
                     f'data-search="{escape(entry["query"].lower(), quote=True)}">{preview}'
                     f'<h2><a href="{viewer}">{escape(entry["query"])}</a></h2>'
                     f'<p class="quality">{entry["motion_count"]} '
                     f'{"selected motion" if entry["motion_count"] == 1 else "independent sequences"} · '
                     f'recorded maximum overlap {peak:.1f} cm</p>'
                     f'<a href="{viewer}">Inspect in 3D</a></article>')
    document = (ROOT / "waveverse_scene/templates/scene_gallery.html").read_text()
    document = document.replace("__SCENE_COUNT__", str(len(entries)))
    document = document.replace("__MOTION_COUNT__", str(sum(e["motion_count"] for e in entries)))
    document = document.replace("__SEQUENCE_GUIDANCE__", (
        "One selected motion per scene." if entries and all(e["motion_count"] == 1 for e in entries)
        else "Use the sequence selector below a preview to compare available motions."))
    document = document.replace("__SCENE_CARDS__", ''.join(cards))
    temporary = output / "index.html.tmp"
    temporary.write_text(document)
    temporary.replace(output / "index.html")
    save_json(output / "manifest.json", {"scenes": len(entries), "sequences": sum(e["motion_count"] for e in entries),
              "overview_preview_sequences": clip_count,
              "entries": [{"scene": e["scene"], "viewer": e["scene"] + "/index.html"} for e in entries]})
    return output / "index.html"
