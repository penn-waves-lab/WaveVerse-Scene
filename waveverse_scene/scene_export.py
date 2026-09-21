"""Export the room in the Z-up coordinate frame used by WaveVerse-Sim."""

from pathlib import Path
import copy
import json
import xml.etree.ElementTree as ET

import numpy as np
from plyfile import PlyData, PlyElement

from .planning import load_scene
from .geometry.mitsuba_exporter import MitsubaExporter
from .materials import MaterialResolver, assign_xml_materials


def _split_floor_ceiling(root, target, room):
    """Partition existing triangles; preserve geometry, winding, and vertex colors."""
    shape = next(node for node in root.findall("shape") if node.get("id") == room.clean_id)
    ceiling_id = room.clean_id + "_ceiling"
    if any(node.get("id") == ceiling_id for node in root.findall("shape")):
        raise ValueError("Duplicate exported shape ID: " + ceiling_id)
    ply_path = target / (room.clean_id + ".ply")
    ply = PlyData.read(ply_path)
    # PlyData may memory-map the input, which the floor write replaces below.
    vertices = ply["vertex"].data.copy()
    face_records = ply["face"].data.copy()
    faces = np.stack(face_records["vertex_indices"])
    z = vertices["z"][faces]
    midpoint = (z.min() + z.max()) / 2
    floor = np.all(z < midpoint, axis=1)
    ceiling = np.all(z > midpoint, axis=1)
    if not floor.any() or not ceiling.any() or not np.all(floor | ceiling):
        raise ValueError("Cannot separate floor and ceiling for " + room.id)
    for shape_id, mask in ((room.clean_id, floor), (ceiling_id, ceiling)):
        selected = faces[mask]
        indices, inverse = np.unique(selected, return_inverse=True)
        records = face_records[mask].copy()
        for record, face in zip(records, inverse.reshape(-1, 3)):
            record["vertex_indices"] = face.astype(np.int32)
        PlyData(
            [
                PlyElement.describe(vertices[indices].copy(), "vertex"),
                PlyElement.describe(records, "face"),
            ],
            text=ply.text,
            byte_order=ply.byte_order,
        ).write(str(target / (shape_id + ".ply")))
    extra = copy.deepcopy(shape)
    extra.set("id", ceiling_id)
    extra.find("string[@name='filename']").set("value", str(target / (ceiling_id + ".ply")))
    root.append(extra)


def export_scene(scene_path, output, config):
    output = Path(output)
    scene = load_scene(scene_path, config)
    source = json.loads(Path(scene_path).read_text())
    resolver = MaterialResolver(config)
    entities = scene.furniture + scene.walls + scene.windows + scene.doors + scene.rooms
    ids = [entity.clean_id for entity in entities]
    if len(set(ids)) != len(ids):
        raise ValueError("Scene entity IDs collide after conversion to XML IDs")
    target = Path(output) / "sionna_scene"
    path = MitsubaExporter(
        scene, up="z", camera_origin=np.array([3, -6, 10]), camera_target=np.array([3, 3, 0])
    ).export(
        target,
        scene_name="scene",
        up="z",
        force_one_mesh=True,
        force_vertex_color=True,
        show_ceiling=config["show_ceiling"],
    )
    # The legacy exporter writes absolute mesh paths. Make the output movable.
    tree = ET.parse(path)
    root = tree.getroot()
    assignments = {}
    for kind, collection, key in (
        ("object", scene.furniture, "objects"),
        ("wall", scene.walls, "walls"),
        ("door", scene.doors, "doors"),
        ("window", scene.windows, "windows"),
    ):
        originals = {item["id"]: item for item in source.get(key, [])}
        for entity in collection:
            item = originals[entity.id]
            assignments[entity.clean_id] = resolver.resolve(
                entity.clean_id, kind, item, item.get("material")
            )
    rooms = {item["id"]: item for item in source["rooms"]}
    for room in scene.rooms:
        item = rooms[room.id]
        assignments[room.clean_id] = resolver.resolve(
            room.clean_id,
            "floor",
            {"radioMaterial": item.get("floorRadioMaterial")},
            item.get("floorMaterial"),
        )
        if config["show_ceiling"]:
            _split_floor_ceiling(root, target, room)
            ceiling_id = room.clean_id + "_ceiling"
            # The original exporter uses the floor lookup for the combined room mesh.
            assignments[ceiling_id] = resolver.resolve(
                ceiling_id,
                "ceiling",
                {"radioMaterial": item.get("ceilingRadioMaterial")},
                item.get("floorMaterial"),
            )
    manifest = resolver.write_manifest(target, assignments)
    assign_xml_materials(root, assignments)
    for node in tree.iter("string"):
        if node.get("name") == "filename":
            filename = Path(node.get("value"))
            node.set("value", filename.relative_to(target).as_posix())
    tree.write(path, encoding="unicode", xml_declaration=True)
    return {
        "xml": str(path.relative_to(output)),
        "meshes": len(list(target.glob("*.ply"))),
        "coordinate_frame": "z-up, metres",
        "materials": str((target / "materials.json").relative_to(output)),
        "assignment_counts": manifest["assignment_counts"],
    }
