"""Read the asset metadata used by scene generation."""

from typing import Dict, Any


def get_asset_metadata(obj_data: Dict[str, Any]):
    if "assetMetadata" in obj_data:
        return obj_data["assetMetadata"]
    elif "thor_metadata" in obj_data:
        return obj_data["thor_metadata"]["assetMetadata"]
    else:
        raise ValueError("Can not find assetMetadata in obj_data")

def get_annotations(obj_data: Dict[str, Any]):
    if "annotations" in obj_data:
        return obj_data["annotations"]
    else:
        # The assert here is just double-checking that a field that should exist does.
        assert "onFloor" in obj_data, f"Can not find annotations in obj_data {obj_data}"

        return obj_data

def get_bbox_dims(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)

    bbox_info = am["boundingBox"]

    if "x" in bbox_info:
        return bbox_info

    if "size" in bbox_info:
        return bbox_info["size"]

    mins = bbox_info["min"]
    maxs = bbox_info["max"]

    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}

def get_secondary_properties(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)
    return am["secondaryProperties"]
