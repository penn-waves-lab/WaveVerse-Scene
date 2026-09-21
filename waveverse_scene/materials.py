"""Assign saved RF materials using the original asset and surface lookup table."""

from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from .config import ROOT, file_hash, save_json

CATALOG_PATH = ROOT / "resources/radio_materials.json"
CATALOG = json.loads(CATALOG_PATH.read_text())["materials"]
DATABASE_PATH = ROOT / "resources/asset_wall_floor_ceiling_material_database.csv"
DOOR_MATERIALS = ("wood", "glass", "metal", "plywood", "chipboard")


def material_name(value):
    """Accept category names and the names used by the patched simulator."""
    name = re.sub(r"[\s-]+", "_", value.strip().lower())
    if name.startswith("mat_itu_"):
        name = name[4:]
    if not name.startswith("itu_"):
        name = "itu_" + name
    if name not in CATALOG and name + "_ex" in CATALOG:
        name += "_ex"
    if name not in CATALOG:
        raise ValueError("Unknown RF material: " + value)
    return name


def material_properties(definition, frequency_hz):
    frequency = float(frequency_hz) / 1e9
    if math.isfinite(frequency):
        for band in definition["frequency_ranges_ghz"]:
            if band["min"] <= frequency <= band["max"]:
                a, b, c, d = band["coefficients"]
                return a * frequency**b, c * frequency**d
    raise ValueError("RF material is not defined at frequency " + str(frequency_hz))


def material_input_paths(config):
    settings = config.get("radio_materials", {})
    path = Path(settings.get("database", DATABASE_PATH)).expanduser()
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError("Missing RF material database: " + str(path))
    return {"database": path}


def material_input_hashes(config):
    return {key: file_hash(path) for key, path in material_input_paths(config).items()} | {
        "catalog": file_hash(CATALOG_PATH)
    }


def _read_database(path, column):
    if column not in ("builtin_mat", "extended_mat"):
        raise ValueError("material_column must be builtin_mat or extended_mat")
    assets, surfaces, seen = {}, {}, set()
    required = {"asset_id", "asset_type", "builtin_mat", "extended_mat"}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not required.issubset(fields) or len(set(fields)) != len(fields):
            raise ValueError("Material database needs asset_id, asset_type, builtin_mat, extended_mat")
        for row in reader:
            if None in row or any(not (row.get(key) or "").strip() for key in required):
                raise ValueError(f"Incomplete material database row at line {reader.line_num}")
            asset_id = row["asset_id"].strip()
            if asset_id in seen:
                raise ValueError("Duplicate material database ID: " + asset_id)
            seen.add(asset_id)
            kind = row["asset_type"].strip()
            if kind == "wall_material":
                destination = surfaces
            elif kind in ("objaverse", "procthor"):
                destination = assets
            else:
                raise ValueError("Unknown material database asset_type: " + kind)
            destination[asset_id] = material_name(row[column])
    if not seen:
        raise ValueError("Material database is empty: " + str(path))
    return assets, surfaces


class MaterialResolver:
    def __init__(self, config):
        self.settings = config.get("radio_materials", {})
        self.column = self.settings.get("material_column", "extended_mat")
        path = material_input_paths(config)["database"]
        self.labels, self.surface_labels = _read_database(path, self.column)
        self.inputs = material_input_hashes(config)
        self.overrides = {
            key: material_name(value)
            for key, value in self.settings.get("object_overrides", {}).items()
        }
        # The original MitsubaExporter uses default_rng(42) for door assignments.
        self.door_seed = self.settings.get("door_seed", 42)
        self.rng = np.random.default_rng(self.door_seed)
        self.frequency = float(self.settings.get("frequency_hz", 77e9))
        self.scattering = float(self.settings.get("scattering_coefficient", 0.0))
        if not 0 <= self.scattering <= 1:
            raise ValueError("scattering_coefficient must be between zero and one")

    def resolve(self, shape_id, kind, item, visual_material=None):
        if kind not in ("object", "wall", "floor", "ceiling", "door", "window", "human"):
            raise ValueError("Unknown scene object kind: " + kind)
        asset_id = item.get("assetId")
        material = self.overrides.get(shape_id)
        source, evidence = "object_override", shape_id
        if material is None and item.get("radioMaterial"):
            material = material_name(item["radioMaterial"])
            source, evidence = "scene_radio_material", item["radioMaterial"]
        if material is None:
            if kind == "window":
                material = material_name("glass")
                source, evidence = "window_glass", "glass"
            elif kind == "door":
                label = str(self.rng.choice(DOOR_MATERIALS))
                material = material_name(label)
                source, evidence = "door_selection", label
            elif kind == "human":
                label = self.settings.get("human_material", "fabric")
                material = material_name(label)
                source, evidence = "human_config", label
            else:
                if kind == "object":
                    key, labels, source = asset_id, self.labels, "asset_database"
                else:
                    key = (
                        visual_material.get("name")
                        if isinstance(visual_material, dict)
                        else visual_material
                    )
                    labels, source = self.surface_labels, "surface_database"
                if key not in labels:
                    raise ValueError(
                        f"No saved RF material for {kind} {shape_id!r} (lookup key {key!r}). "
                        "Provide an explicit radioMaterial or object_overrides entry."
                    )
                material, evidence = labels[key], key
        material_properties(CATALOG[material], self.frequency)
        return {
            "material": material,
            "source": source,
            "evidence": evidence,
            "kind": kind,
            "asset_id": asset_id,
        }

    def write_manifest(self, target, assignments):
        unused = set(self.overrides) - set(assignments)
        if unused:
            raise ValueError(
                "Object material overrides do not match exported shapes: " + str(sorted(unused))
            )
        human = self.resolve("human", "human", {})
        used = {value["material"] for value in assignments.values()} | {human["material"]}
        manifest = {
            "schema_version": 1,
            "frequency_hz": self.frequency,
            "scattering_coefficient": self.scattering,
            "scattering_pattern": "lambertian",
            "material_column": self.column,
            "door_seed": self.door_seed,
            "input_sha256": self.inputs,
            "definitions": {name: CATALOG[name] for name in sorted(used)},
            "objects": assignments,
            "human": human,
            "default_assignments": [],
            "assignment_counts": dict(Counter(value["source"] for value in assignments.values())),
        }
        save_json(Path(target) / "materials.json", manifest)
        return manifest


def assign_xml_materials(root, assignments):
    """Use shared named BSDFs so Sionna receives actual material names."""
    for name in sorted({record["material"] for record in assignments.values()}):
        bsdf = ET.Element("bsdf", type="diffuse", id="mat-" + name)
        texture = ET.SubElement(bsdf, "texture", type="mesh_attribute", name="reflectance")
        ET.SubElement(texture, "string", name="name", value="vertex_color")
        root.insert(0, bsdf)
    for shape in root.findall("shape"):
        for child in list(shape):
            if child.tag in ("bsdf", "ref"):
                shape.remove(child)
        ET.SubElement(shape, "ref", id="mat-" + assignments[shape.get("id")]["material"])
