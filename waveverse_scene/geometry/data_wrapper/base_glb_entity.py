from pathlib import Path
from typing import Union, Literal

import numpy as np
import trimesh

from .base import BaseMeshEntity


class BaseGLBEntity(BaseMeshEntity):
    """
    A class to represent entities with a GLB mesh (proc-thor converted).
    """

    def __init__(self):
        super().__init__()

    def get_mesh(self, asset_dir: Union[str, Path]):
        if self._cached_mesh is not None:
            return self._cached_mesh

        glb_path = Path(asset_dir) / f"{self.asset_id}.glb"
        if not glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")

        # Load the GLB file.
        #   it should be a scene. we force it to be a mesh.
        mesh = trimesh.load(glb_path, force="mesh")
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
        glb_path = Path(asset_dir) / f"{self.asset_id}.glb"
        if not glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")

        to_process = []
        if force_one_mesh:
            mesh = trimesh.load(glb_path, force="mesh")
            self.center_mesh(mesh)
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.radians(self.rotation.y), [0, 1, 0])
            )
            mesh.apply_translation([self.position.x, self.position.y, self.position.z])
            to_process.append(mesh)
        else:
            scene = trimesh.load(glb_path, force="scene")
            self.center_mesh(scene)
            scene.apply_transform(
                trimesh.transformations.rotation_matrix(np.radians(self.rotation.y), [0, 1, 0])
            )
            scene.apply_translation([self.position.x, self.position.y, self.position.z])
            to_process.extend(scene.geometry.values())

        results = []
        for mesh in to_process:
            # vertex data
            vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
            vertex_dtype.extend([("nx", "f4"), ("ny", "f4"), ("nz", "f4")])

            # trimesh.texture.TextureVisuals
            mesh_material = mesh.visual.material
            export_albedo = False
            export_normal = False
            #   process texture/albedo
            if force_vertex_color or mesh_material.baseColorTexture is None:
                # if no image texture available or force to use vertex color, use the main color
                vertex_dtype.extend([("red", "f4"), ("green", "f4"), ("blue", "f4")])
            else:
                # we need to export the texture image
                export_albedo = True
                vertex_dtype.extend([("s", "f4"), ("t", "f4")])

            vertex_data = np.empty(len(mesh.vertices), dtype=vertex_dtype)
            vertex_data["x"] = mesh.vertices[:, 0]
            vertex_data["y"] = mesh.vertices[:, 1]
            vertex_data["z"] = mesh.vertices[:, 2]
            vertex_data["nx"] = mesh.vertex_normals[:, 0]
            vertex_data["ny"] = mesh.vertex_normals[:, 1]
            vertex_data["nz"] = mesh.vertex_normals[:, 2]

            if force_vertex_color or mesh_material.baseColorTexture is None:
                vertex_data["red"] = mesh_material.main_color[0] / 255.0
                vertex_data["green"] = mesh_material.main_color[1] / 255.0
                vertex_data["blue"] = mesh_material.main_color[2] / 255.0
            else:
                # we need to export the uv data
                vertex_data["s"] = mesh.visual.uv[:, 0]
                vertex_data["t"] = mesh.visual.uv[:, 1]

            #   process normal texture
            if mesh_material.normalTexture is not None:
                # we need to export the normal texture image
                export_normal = True

            # face data
            face_dtype = [("vertex_indices", "i4", (3,))]
            face_data = np.empty(len(mesh.faces), dtype=face_dtype)
            face_data["vertex_indices"] = mesh.faces

            results.append(
                {
                    "vertex_data": vertex_data,
                    "face_data": face_data,
                    "texture_to_export": {
                        "albedo": mesh_material.baseColorTexture if export_albedo else None,
                        "normal": mesh_material.normalTexture if export_normal else None,
                    },
                }
            )

        return results

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
        else:
            if force_one_mesh:
                n_meshes = 1
            else:
                glb_path = Path(asset_dir) / f"{self.clean_id}.glb"
                if not glb_path.exists():
                    raise FileNotFoundError(f"GLB file not found: {glb_path}")

                scene = trimesh.load(glb_path, force="scene")
                n_meshes = len(scene.geometry)

            export_paths = [
                {
                    "ply": ply_output_dir
                    / "{}{}.ply".format(self.clean_id, f"_{i}" if i > 0 else ""),
                    "albedo": ply_output_dir
                    / "{}{}.albedo.png".format(self.clean_id, f"_{i}" if i > 0 else ""),
                    "normal": ply_output_dir
                    / "{}{}.normal.png".format(self.clean_id, f"_{i}" if i > 0 else ""),
                }
                for i in range(n_meshes)
            ]

        xml = ""
        for i, export_path in enumerate(export_paths):
            normal_map_exists = export_path["normal"] is not None and export_path["normal"].exists()
            albedo_map_exists = export_path["albedo"] is not None and export_path["albedo"].exists()

            if force_vertex_color:
                # if force_vertex_color, we don't need to manage the albedo/normal map
                bsdf_string = """<bsdf type="diffuse">
                    <texture type="mesh_attribute" name="reflectance">
                        <string name="name" value="vertex_color"/>
                    </texture>
                </bsdf>"""
            else:
                # manage the albedo/normal map
                if albedo_map_exists:
                    bsdf_string = f"""<bsdf type="diffuse">
                        <texture name="reflectance" type="bitmap">
                            <string name="filename" value="{export_path['albedo']}"/>
                        </texture>
                    </bsdf>"""
                else:
                    bsdf_string = """<bsdf type="diffuse">
                        <texture type="mesh_attribute" name="reflectance">
                            <string name="name" value="vertex_color"/>
                        </texture>
                    </bsdf>"""

                if normal_map_exists:
                    normal_bsdf_wrapper = f"""<bsdf type="normalmap">
                        <texture name="normalmap" type="bitmap">
                            <boolean name="raw" value="true"/>
                            <string name="filename" value="{export_path['normal']}"/>
                        </texture>
                        {bsdf_string}
                    </bsdf>"""
                    bsdf_string = normal_bsdf_wrapper

            xml += f"""<shape type="ply" id="{self.clean_id}">
                <string name="filename" value="{export_path['ply']}"/>
                {bsdf_string}
            </shape>"""

        return xml
