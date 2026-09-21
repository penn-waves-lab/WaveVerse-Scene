"""Export saved scenes as self-contained WaveVerse-Sim data, without run logs."""

from collections import Counter
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from .config import file_hash, save_json
from .materials import MaterialResolver, assign_xml_materials
from .motion_files import meshes_dir, correspondences_dir


def read(path):
    return json.loads(Path(path).read_text())


def source_file(root, relative):
    path = root / relative
    path.resolve().relative_to(root.resolve())
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or linked input: " + str(path))
    return path


def scene_assignments(root, scene, resolver):
    """Map both original exporter suffixes and current exact IDs to saved assets."""
    identities = []
    for group, kind in (("objects", "object"), ("walls", "wall"),
                        ("doors", "door"), ("windows", "window"), ("rooms", "floor")):
        for item in scene.get(group, []):
            name = re.sub(r"\W+", " ", item["id"]).strip().replace(" ", "_")
            identities.append((name, kind, item))
    identities.sort(key=lambda item: -len(item[0]))
    assignments = {}
    for shape in root.findall("shape"):
        name = shape.get("id")
        if not name or name in assignments or shape.get("type") != "ply":
            raise ValueError("Expected unique named PLY shapes")
        if shape.find("transform") is not None:
            raise ValueError("Expected already placed static meshes")
        matches = [x for x in identities if name == x[0] or name.startswith(x[0] + "_")]
        if not matches:
            raise ValueError("No source entity for " + name)
        matches = [x for x in matches if len(x[0]) == len(matches[0][0])]
        if len(matches) > 1:
            # The historical exporter appends the first six asset-ID characters.
            # Some original scene JSONs contain duplicate human-readable IDs.
            matches = [x for x in matches if x[2].get("assetId")
                       and name == x[0] + "_" + x[2]["assetId"][:6]]
        if len(matches) != 1:
            raise ValueError("Ambiguous source entity for " + name)
        key, kind, item = matches[0]
        if kind == "floor":
            kind = "ceiling" if name == key + "_ceiling" else "floor"
            # Preserve the original combined-floor/ceiling lookup convention.
            visual = item.get("floorMaterial")
            item = {"radioMaterial": item.get(kind + "RadioMaterial")}
        else:
            visual = item.get("material")
        assignments[name] = resolver.resolve(name, kind, item, visual)
    unused = set(resolver.overrides) - set(assignments)
    if unused:
        raise ValueError("Unmatched material overrides: " + str(sorted(unused)))
    return assignments


def export_scene(source, output, config):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists: " + str(output))
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Keep the release separate from the source scene")
    scene = read(source / "scene.json")
    inference = read(source / "motion/inference.json")
    fitted = read(source / "motion/meshes.json")["motions"]
    if len(fitted) != len(inference["motions"]) or inference["fps"] != 20:
        raise ValueError("Expected complete 20 FPS motions")
    tree = ET.parse(source_file(source, "sionna_scene/scene.xml"))
    root = tree.getroot()
    resolver = MaterialResolver(config)
    assignments = scene_assignments(root, scene, resolver)
    for bsdf in list(root.findall("bsdf")):
        root.remove(bsdf)
    assign_xml_materials(root, assignments)

    checksums_path = source / "checksums.json"
    expected = read(checksums_path) if checksums_path.exists() else {}
    before = {str(p.relative_to(source)): file_hash(p) for p in (
        source / "scene.json", source / "sionna_scene/scene.xml",
        source / "motion/inference.json")}
    copies = {}
    output.mkdir(parents=True)

    def copy_file(relative, destination):
        src = source_file(source, relative)
        digest = expected.get(str(relative)) or file_hash(src)
        dst = output / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if file_hash(dst) != digest:
            raise ValueError("Copied payload differs from source checksum: " + str(src))
        copies[str(destination)] = {"source": str(relative), "sha256": digest}

    copy_file("scene.json", "scene.json")
    filenames = set()
    for shape in root.findall("shape"):
        node = shape.find("string[@name='filename']")
        if node is None:
            raise ValueError("Missing mesh filename")
        original = Path("sionna_scene") / node.get("value")
        # All mesh paths are relative to the scene XML in the public copy.
        destination = Path("meshes") / original.name
        if destination in filenames:
            raise ValueError("Colliding mesh filenames")
        filenames.add(destination)
        copy_file(original, destination)
        node.set("value", destination.as_posix())
    xml = output / (output.name + ".xml")
    tree.write(xml, encoding="utf-8", xml_declaration=True)
    frames = 0
    for index, (motion, mesh) in enumerate(zip(inference["motions"], fitted)):
        name = f"motion_{index:06d}"
        count = motion["frames"]
        if motion["name"] != name or mesh["name"] != name or mesh["frames"] != count or count < 1:
            raise ValueError("Incomplete or unordered motion sequence")
        src_meshes = meshes_dir(source, name).relative_to(source)
        src_corr = correspondences_dir(source, name).relative_to(source)
        if {p.name for p in (source / src_meshes).glob("*.ply")} != {f"{i:06d}.ply" for i in range(count)}:
            raise ValueError("Missing or extra human mesh frames")
        if {p.name for p in (source / src_corr).glob("*.pkl")} != {f"{i:06d}.pkl" for i in range(count)}:
            raise ValueError("Missing or extra correspondence frames")
        description = output / "person" / name / "description.txt"
        description.parent.mkdir(parents=True, exist_ok=True)
        description.write_text(motion["action"].strip() + "\n")
        for frame in range(count):
            destination = Path("person") / name / f"{frame:06d}"
            copy_file(src_meshes / f"{frame:06d}.ply", destination / "pose.ply")
            copy_file(src_corr / f"{frame:06d}.pkl", destination / "correspondence.pkl")
        frames += count
    for name, digest in before.items():
        if file_hash(source / name) != digest:
            raise ValueError("Source metadata changed during export")
    return {"scene": output.name, "source": str(source), "motions": len(inference["motions"]),
            "frames": frames, "saved_paths": False, "static_shapes": len(assignments),
            "material_counts": dict(Counter(x["material"] for x in assignments.values())),
            "assignment_sources": dict(Counter(x["source"] for x in assignments.values())),
            "assignments": assignments, "source_checksums": before, "copied_files": copies,
            "geometry_and_correspondences_byte_identical": True}


def export_dataset(source, output, config, report_path):
    source, output, report_path = Path(source).resolve(), Path(output).resolve(), Path(report_path).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the source")
    if report_path == output or output in report_path.parents:
        raise ValueError("Keep validation records outside the metadata-free public data")
    if output.exists():
        raise ValueError("Output already exists; use an empty destination")
    stage = output.with_name(output.name + ".partial")
    if stage.exists():
        raise ValueError("A partial export already exists: " + str(stage))
    manifest = source / "manifest.json"
    if manifest.exists():
        entries = [(source / e["path"]).resolve() for e in read(manifest)["scenes"]]
    elif (source / "motion/inference.json").exists():
        entries = [source]
    else:
        raise ValueError("Input must be a curated release or a generated scene run")
    if len(set(p.name for p in entries)) != len(entries):
        raise ValueError("Duplicate scene directory names")
    report = {"output": str(output), "accepted": False, "scenes": []}
    for index, entry in enumerate(entries):
        entry.relative_to(source)
        report["scenes"].append(export_scene(entry, stage / "data" / entry.name, config))
        print(f"Exported and hash-checked {index + 1}/{len(entries)}: {entry.name}", flush=True)
    materials, origins = Counter(), Counter()
    for entry in report["scenes"]:
        materials.update(entry["material_counts"])
        origins.update(entry["assignment_sources"])
    report.update(accepted=True, scene_count=len(entries),
                  motion_count=sum(x["motions"] for x in report["scenes"]),
                  frames=sum(x["frames"] for x in report["scenes"]),
                  static_shapes=sum(x["static_shapes"] for x in report["scenes"]),
                  material_counts=dict(materials), assignment_sources=dict(origins),
                  material_input_sha256=MaterialResolver(config).inputs)
    template = Path(__file__).with_name("templates") / "sim_release_readme.md"
    (stage / "README.md").write_text(template.read_text().replace("__SCENES__", str(len(entries)))
                                       .replace("__MOTIONS__", str(report["motion_count"]))
                                       .replace("__FRAMES__", str(report["frames"])))
    shutil.copy2(Path(__file__).with_name("sim_data.py"), stage / "load_scene.py")
    stage.rename(output)
    save_json(report_path, report)
    return {k: v for k, v in report.items() if k != "scenes"}
