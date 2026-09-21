"""Load a released room and optional articulated frame with its XML RF materials."""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def load_scene(directory, motion=None, frame=0, frequency_hz=77e9):
    """Use the patched Sionna 0.19.2 environment installed by WaveVerse-Sim.

    Human vertices are already placed in world coordinates. No centering,
    grounding, synthetic translation, or filename-based material remapping occurs.
    """
    from sionna import rt

    directory = Path(directory).resolve()
    tree = ET.parse(directory / (directory.name + ".xml"))
    root = tree.getroot()
    expected = {x.get("id"): x.find("ref").get("id")[4:] for x in root.findall("shape")}
    for node in root.iter("string"):
        if node.get("name") == "filename":
            path = (directory / node.get("value")).resolve()
            path.relative_to(directory)
            if not path.is_file():
                raise FileNotFoundError(path)
            node.set("value", str(path))
    if motion is not None:
        if type(motion) is not int or motion < 0 or type(frame) is not int or frame < 0:
            raise ValueError("Motion and frame must be nonnegative integers")
        folder = directory / "person" / f"motion_{motion:06d}" / f"{frame:06d}"
        if not (folder / "pose.ply").is_file() or not (folder / "correspondence.pkl").is_file():
            raise FileNotFoundError(folder)
        material = "mat-itu_fabric"
        if not any(x.get("id") == material for x in root.findall("bsdf")):
            bsdf = ET.Element("bsdf", type="diffuse", id=material)
            ET.SubElement(bsdf, "rgb", name="reflectance", value="0.5 0.5 0.5")
            root.insert(0, bsdf)
        if "human" in expected:
            raise ValueError("Reserved human object ID is already in use")
        shape = ET.SubElement(root, "shape", type="ply", id="human")
        ET.SubElement(shape, "string", name="filename", value=str(folder / "pose.ply"))
        ET.SubElement(shape, "ref", id=material, name="bsdf")
        expected["human"] = "itu_fabric"
    with tempfile.TemporaryDirectory(prefix="waveverse-sim-data-") as tmp:
        xml = Path(tmp) / "scene.xml"
        tree.write(xml, encoding="utf-8", xml_declaration=True)
        scene = rt.load_scene(str(xml))
    scene.frequency = frequency_hz
    if set(scene.objects) != set(expected):
        raise ValueError("Loaded scene object IDs differ from the XML")
    for name, material in expected.items():
        actual = scene.objects[name].radio_material
        if actual.is_placeholder or actual.name != material:
            raise ValueError("Unresolved or remapped RF material for " + name)
    return scene
