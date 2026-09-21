import gzip
import pickle
from pathlib import Path
from typing import Union, Literal

import numpy as np
import trimesh
from PIL import Image
from trimesh import visual

from .base import BaseMeshEntity


class BasePklGzEntity(BaseMeshEntity):
    """
    Abstract class for entities with a mesh stored in a PklGz file (objaverse).
    """

    def __init__(self):
        super().__init__()

    def get_mesh(self, asset_dir: Union[str, Path]):
        if self._cached_mesh is not None:
            return self._cached_mesh

        asset_dir = Path(asset_dir)

        pkl_gz_path = asset_dir / f"{self.asset_id}.pkl.gz"
        if not pkl_gz_path.exists():
            raise FileNotFoundError(f"PklGz file not found: {pkl_gz_path}")

        # Load the PklGz file
        with gzip.open(pkl_gz_path, "rb") as f:
            loaded_data = pickle.load(f)

        # Extract vertices and triangles
        vertices = np.array([[v["x"], v["y"], v["z"]] for v in loaded_data["vertices"]])
        triangles = np.array(loaded_data["triangles"]).reshape(-1, 3)
        normals = np.array([[n["x"], n["y"], n["z"]] for n in loaded_data["normals"]])

        # Texture information
        uvs = np.array([[uv["x"], uv["y"]] for uv in loaded_data["uvs"]])

        albedo_map = Image.open(asset_dir / "albedo.jpg")
        normal_map = Image.open(asset_dir / "normal.jpg")

        material = visual.material.PBRMaterial(
            normalTexture=normal_map,
            baseColorTexture=albedo_map,
        )
        texture = visual.texture.TextureVisuals(
            uv=uvs,
            material=material,
        )

        # Create a trimesh object
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=triangles,
            vertex_normals=normals,
            visual=texture,
        )

        # Apply rotation offset
        rotation_offset = loaded_data["yRotOffset"]
        if rotation_offset != 0:
            # Rotate the mesh
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.radians(rotation_offset), [0, 1, 0])
            )
        # center the mesh
        self.center_mesh(mesh)

        # Cache the mesh
        self._cached_mesh = mesh

        return mesh

    def _get_mesh_basic_components(
        self,
        asset_dir: Union[str, Path],
        force_one_mesh: bool = False,
        force_vertex_color: bool = False,
    ) -> list[dict[str, np.ndarray]]:
        # get mesh data
        asset_dir = Path(asset_dir)
        mesh = self.get_mesh_in_world_frame(asset_dir)

        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.vertex_normals
        uvs = mesh.visual.uv

        # vertex data
        vertex_dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
        ]
        if force_vertex_color or uvs is None:
            vertex_dtype.extend([("red", "f4"), ("green", "f4"), ("blue", "f4")])
        else:
            vertex_dtype.extend([("s", "f4"), ("t", "f4")])

        vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
        vertex_data["x"] = vertices[:, 0]
        vertex_data["y"] = vertices[:, 1]
        vertex_data["z"] = vertices[:, 2]
        vertex_data["nx"] = normals[:, 0]
        vertex_data["ny"] = normals[:, 1]
        vertex_data["nz"] = normals[:, 2]

        if force_vertex_color or uvs is None:
            colors = mesh.visual.to_color().vertex_colors / 255.0
            vertex_data["red"] = colors[:, 0]
            vertex_data["green"] = colors[:, 1]
            vertex_data["blue"] = colors[:, 2]
        else:
            vertex_data["s"] = uvs[:, 0]
            vertex_data["t"] = uvs[:, 1]

        # face data
        face_dtype = [("vertex_indices", "i4", (3,))]
        face_data = np.empty(len(faces), dtype=face_dtype)
        face_data["vertex_indices"] = faces

        return [{"vertex_data": vertex_data, "face_data": face_data}]

    def to_mitsuba_xml(
        self,
        asset_dir: Union[str, Path] = "",
        ply_output_dir: Union[str, Path] = "",
        up: Literal["y", "z"] = "y",
        export_ply: bool = False,
        force_one_mesh: bool = False,
        force_vertex_color: bool = False,
    ) -> str:
        asset_dir = Path(asset_dir)
        ply_output_dir = Path(ply_output_dir)

        if export_ply:
            export_paths = self.to_ply(
                asset_dir,
                ply_output_dir,
                up=up,
                force_one_mesh=force_one_mesh,
                force_vertex_color=force_vertex_color,
            )
            assert (
                len(export_paths) == 1
            ), f"More than one mesh in the Objaverse asset {self.id}: {self.asset_id}."
            export_path = export_paths[0]
            ply_name = export_path["ply"]
        else:
            ply_name = ply_output_dir / f"{self.clean_id}.ply"

        if force_vertex_color:
            bsdf_string = """<bsdf type="diffuse">
                <texture type="mesh_attribute" name="reflectance">
                    <string name="name" value="vertex_color"/>
                </texture>
            </bsdf>"""
        else:
            bsdf_string = f"""<bsdf type="normalmap">
                <texture name="normalmap" type="bitmap">
                    <boolean name="raw" value="true"/>
                    <string name="filename" value="{asset_dir}/normal.jpg"/>
                </texture>
                <bsdf type="diffuse">
                    <texture name="reflectance" type="bitmap">
                        <string name="filename" value="{asset_dir}/albedo.jpg"/>
                    </texture>
                </bsdf>
            </bsdf>
            """

        xml = f"""<shape type="ply" id="{self.clean_id}">
            <string name="filename" value="{ply_name}"/>
            {bsdf_string}
        </shape>"""

        return xml
