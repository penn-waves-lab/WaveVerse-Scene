"""Load exported geometry and RF materials in WaveVerse-Sim's Sionna 0.19.x."""

from functools import partial
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from .materials import material_properties


def load_scene(xml_path, *, human_meshes=None, frequency_hz=None):
    """Load a room and optional {object_name: world-space human PLY} frame mapping.

    Call this from the simulator environment. Meshes are already in Z-up metres.
    The adjacent materials.json is required; no assets are inferred at load time.
    """
    import sionna
    from sionna import rt

    if not sionna.__version__.startswith("0.19."):
        raise RuntimeError("This loader targets WaveVerse-Sim's Sionna 0.19.x environment")
    xml_path = Path(xml_path).resolve()
    manifest = json.loads(xml_path.with_name("materials.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported material manifest schema")
    frequency = manifest["frequency_hz"] if frequency_hz is None else float(frequency_hz)
    assignments = {key: value["material"] for key, value in manifest["objects"].items()}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    shape_ids = [shape.get("id") for shape in root.findall("shape")]
    if len(shape_ids) != len(set(shape_ids)) or set(shape_ids) != set(assignments):
        raise ValueError("Scene shapes and material assignments do not match")
    for shape in root.findall("shape"):
        ref = shape.find("ref")
        if ref is None or ref.get("id") != "mat-" + assignments[shape.get("id")]:
            raise ValueError("XML material disagrees with manifest: " + shape.get("id"))
    for node in root.iter("string"):
        if node.get("name") == "filename":
            path = (xml_path.parent / node.get("value")).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            node.set("value", str(path))
    human_meshes = human_meshes or {}
    for name, mesh in human_meshes.items():
        if not name or name.startswith("mesh-") or name in assignments:
            raise ValueError(
                "Human object name must be unique and must not start with mesh-: " + name
            )
        mesh = Path(mesh).resolve()
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        assignments[name] = manifest["human"]["material"]
        shape = ET.SubElement(root, "shape", type="ply", id=name)
        ET.SubElement(shape, "string", name="filename", value=str(mesh))
        material_id = "mat-" + assignments[name]
        if not any(node.get("id") == material_id for node in root.findall("bsdf")):
            bsdf = ET.Element("bsdf", type="diffuse", id=material_id)
            ET.SubElement(bsdf, "rgb", name="reflectance", value="0.5 0.5 0.5")
            root.insert(0, bsdf)
        ET.SubElement(shape, "ref", id=material_id)
    used = set(assignments.values())
    for name in used:
        material_properties(manifest["definitions"][name], frequency)
    # Absolute mesh paths in this temporary document allow movable output folders.
    with tempfile.TemporaryDirectory(prefix="waveverse-scene-") as temporary:
        combined = Path(temporary) / "scene.xml"
        tree.write(combined, encoding="unicode", xml_declaration=True)
        scene = rt.load_scene(str(combined))
    if set(scene.objects) != set(assignments):
        raise ValueError("Sionna loaded unexpected object names")
    for name in used:
        callback = partial(material_properties, manifest["definitions"][name])
        material = scene.get(name)
        if material is None or material.is_placeholder:
            scene.add(rt.RadioMaterial(name, frequency_update_callback=callback))
            material = scene.get(name)
        else:
            material.frequency_update_callback = callback
        material.scattering_coefficient = manifest["scattering_coefficient"]
        material.scattering_pattern = rt.LambertianPattern()
    scene.frequency = frequency
    for name, material in assignments.items():
        scene.objects[name].radio_material = material
        if scene.objects[name].radio_material.is_placeholder:
            raise ValueError("Unresolved RF material for " + name)
    return scene
