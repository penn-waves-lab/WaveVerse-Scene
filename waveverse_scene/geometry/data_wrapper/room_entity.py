from contextlib import contextmanager
from pathlib import Path
from typing import Union, Optional, Literal

import numpy as np
import trimesh
from PIL.Image import Image
from shapely.geometry import Polygon

from .base import Point3, Material, BaseMeshEntity


class RoomEntity(BaseMeshEntity):
    def __init__(self):
        super().__init__()

        # BaseEntity already defines an id
        # self.id: str = ""

        self.ceilings: list = []
        self.children: list = []
        self.vertices: list[list[float]] = []
        self.floor_material: Material = Material("")
        self.floor_polygon: list[Point3] = []

        self.room_type: str = ""
        self.floor_design: str = ""
        self.wall_design: str = ""
        self.full_vertices: list[list[float]] = []
        self.wall_material: Material = Material("")
        self.layer: str = ""

        self.wall_height = 2.8

        # manages whether a ceiling is present
        self._ceiling_state = True

    @contextmanager
    def temp_ceiling_state(self, ceiling: Optional[bool]):
        old_state = self._ceiling_state
        if ceiling is not None:
            self._ceiling_state = ceiling

        try:
            yield
        finally:
            self._ceiling_state = old_state

    @classmethod
    def load_from_dict(cls, dict_data: dict, wall_height: float = 2.8) -> "RoomEntity":
        instance = super().load_from_dict(dict_data)

        # Convert complex types after basic dict loading
        if "floor_material" in instance.__dict__:
            instance.floor_material = Material.load_from_dict(instance.floor_material)

        if "wall_material" in instance.__dict__:
            instance.wall_material = Material.load_from_dict(instance.wall_material)

        if "floor_polygon" in instance.__dict__:
            instance.floor_polygon = [
                Point3.load_from_dict(point) for point in instance.floor_polygon
            ]

        instance.wall_height = wall_height

        return instance

    def get_mesh(
        self, asset_dir: Union[str, Path], ceiling: Optional[bool] = None
    ) -> trimesh.Trimesh:
        use_ceiling = ceiling if ceiling is not None else self._ceiling_state

        if self._cached_mesh is not None:
            has_ceiling = self._cached_mesh.vertices[:, 1].max() > (self.wall_height / 2)
            if has_ceiling == use_ceiling:
                # ceiling state is the same as the cached mesh
                # no need to recompute
                return self._cached_mesh

        vertices = np.array([[v.x, v.y, v.z] for v in self.floor_polygon])
        vertices_2d = vertices[:, [0, 2]]
        mesh = trimesh.creation.extrude_polygon(
            Polygon(vertices_2d), height=0.01, engine="triangle"
        )
        mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
        mesh.apply_translation([0, -0.01, 0])

        if use_ceiling:
            ceiling_mesh = mesh.copy()
            ceiling_mesh.apply_translation([0, self.wall_height, 0])

            mesh = trimesh.util.concatenate([mesh, ceiling_mesh])

        self._cached_mesh = mesh
        return mesh

    def get_mesh_in_world_frame(
        self, asset_dir: Union[str, Path] = "", ceiling: Optional[bool] = None
    ) -> trimesh.Trimesh:
        """
        Get the mesh in the world frame.

        Args:
            asset_dir: path to the asset directory
            ceiling: whether to include the ceiling in the mesh. If None, ceiling is included.

        Returns:
            trimesh.Trimesh: the mesh in the world frame
        """
        # override the super method with the ceiling state managed
        with self.temp_ceiling_state(ceiling):
            return super().get_mesh_in_world_frame(asset_dir)

    def to_ply(
        self,
        asset_dir: Union[str, Path],
        ply_output_dir: Union[str, Path],
        up: Literal["y", "z"] = "y",
        force_one_mesh: bool = False,
        force_vertex_color: bool = False,
        ceiling: Optional[bool] = None,
    ) -> list[dict[str, Path]]:
        """
        Export mesh/meshes to PLY file(s).

        Args:
            asset_dir: path to the asset directory
            ply_output_dir: path to the output directory
            up: the up direction of the entity
            force_one_mesh: whether to force all sub-meshes into one PLY file. Used for entities with multiple meshes.
            force_vertex_color: whether to force vertex color export instead of texture export
            ceiling: whether to include the ceiling in the mesh. If None, ceiling is included.

        Returns:
            list[dict[str, Path]]: list of output paths for different types of data (e.g. ply, albedo, normal, etc)
        """
        # override the super method with the ceiling state managed
        with self.temp_ceiling_state(ceiling):
            return super().to_ply(asset_dir, ply_output_dir, up, force_one_mesh, force_vertex_color)

    def _get_mesh_basic_components(
        self,
        asset_dir: Union[str, Path],
        force_one_mesh: bool = False,
        force_vertex_color: bool = False,
        ceiling: Optional[bool] = None,
    ) -> list[dict[str, Union[np.ndarray, dict[str, Image]]]]:
        with self.temp_ceiling_state(ceiling):
            # get mesh data
            mesh = self.get_mesh_in_world_frame(asset_dir)
            vertices = mesh.vertices
            faces = mesh.faces
            # mitsuba xml requires the color to be in the range [0, 1]
            colors = mesh.visual.vertex_colors / 255.0

            # vertex data
            vertex_dtype = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "f4"),
                ("green", "f4"),
                ("blue", "f4"),
            ]
            vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
            vertex_data["x"] = vertices[:, 0]
            vertex_data["y"] = vertices[:, 1]
            vertex_data["z"] = vertices[:, 2]
            vertex_data["red"] = colors[:, 0]
            vertex_data["green"] = colors[:, 1]
            vertex_data["blue"] = colors[:, 2]

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
        ceiling: Optional[bool] = None,
    ) -> str:
        """
        Export the entity to Mitsuba XML format.

        Args:
            asset_dir: path to the asset directory
            ply_output_dir: path to the output directory
            up: the up direction of the entity
            export_ply: whether to export the mesh to PLY format
            force_one_mesh: whether to force all sub-meshes into one PLY file. Used for entities with multiple meshes
            force_vertex_color: whether to force vertex color export instead of texture export
            ceiling: whether to include the ceiling in the mesh. If None, ceiling is included.

        Returns:
            str: the XML string of the entity
        """
        with self.temp_ceiling_state(ceiling):
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
                assert len(export_paths) == 1, f"More than one mesh in the room {self.id}."
                export_path = export_paths[0]
                ply_name = export_path["ply"]
            else:
                ply_name = ply_output_dir / f"{self.clean_id}.ply"

            xml = f"""<shape type="ply" id="{self.clean_id}">
                <string name="filename" value="{ply_name}"/>
                <bsdf type="diffuse">
                    <texture type="mesh_attribute" name="reflectance">
                        <string name="name" value="vertex_color"/>
                    </texture>
                </bsdf>
            </shape>"""

            return xml
