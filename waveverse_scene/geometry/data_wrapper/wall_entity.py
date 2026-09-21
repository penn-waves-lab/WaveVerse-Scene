from pathlib import Path
from typing import Union, Sequence, Literal

import numpy as np
import trimesh
from PIL.Image import Image
from shapely.geometry import Polygon

from .base import Point3, Material, BaseMeshEntity


class WallEntity(BaseMeshEntity):
    def __init__(self):
        super().__init__()

        # BaseEntity already defines an id
        # self.id: str = ""

        self.room_id: str = ""
        self.material: Material = Material("")
        self.polygon: list[Point3] = []
        self.connected_rooms: list[str] = []
        self.width: float = 0.0
        self.height: float = 0.0
        self.direction: str = ""
        self.segment: list[list[float]] = []
        self.layer: str = ""

        # each opening is a tuple of two points (two diagonal corners)
        self.openings: Sequence[tuple[Point3, Point3]] = []

    @classmethod
    def load_from_dict(
        cls, dict_data: dict, openings: Sequence[tuple[Point3, Point3]] = ()
    ) -> "WallEntity":
        instance = super().load_from_dict(dict_data)

        # Convert complex types after basic dict loading
        if "material" in instance.__dict__:
            instance.material = Material.load_from_dict(instance.material)

        if "polygon" in instance.__dict__:
            instance.polygon = [Point3.load_from_dict(point) for point in instance.polygon]

        # wall opening information from doors and windows
        instance.openings = openings

        return instance


    def get_mesh(
        self,
        asset_dir: Union[str, Path],
    ) -> trimesh.Trimesh:
        if self._cached_mesh is not None:
            return self._cached_mesh

        vertices = np.array([[v.x, v.y, v.z] for v in self.polygon])
        center = np.mean(vertices, axis=0)
        variations = np.std(vertices, axis=0)

        # should only be x ot z
        constant_axis = np.argmin(variations)
        assert constant_axis == 0 or constant_axis == 2, "Wall should be vertical or horizontal."
        # y should always be the last axis (z, y) or (x, y)
        projected_axis = [2, 1] if constant_axis == 0 else [0, 1]

        # hole polygon is computed in the local frame
        vertices_2d = vertices[:, projected_axis]
        vertices_2d -= np.min(vertices_2d, axis=0)

        # remove openings
        wall_polygon = Polygon(vertices_2d)
        for opening in self.openings:
            opening_polygon = Polygon(
                [
                    [opening[0].x, opening[0].y],
                    [opening[1].x, opening[0].y],
                    [opening[1].x, opening[1].y],
                    [opening[0].x, opening[1].y],
                ]
            )
            wall_polygon = wall_polygon.difference(opening_polygon)

        # triangulate the wall polygon
        opened_wall_vertices, opened_wall_faces = trimesh.creation.triangulate_polygon(
            wall_polygon, engine="triangle"
        )

        # extrude the wall
        opened_wall_mesh = trimesh.creation.extrude_triangulation(
            vertices=opened_wall_vertices,
            faces=opened_wall_faces,
            height=0.01,
        )

        if constant_axis == 0:
            # the wall is located at the yz plane originally
            # therefore, original z, y are extruded x, y
            # we should change extruded x -> z, z -> x
            opened_wall_mesh.vertices[:, [0, 2]] = opened_wall_mesh.vertices[:, [2, 0]]
            # we swapped y and z in the previous projection and this is the fix
            opened_wall_mesh.apply_transform(
                trimesh.transformations.scale_matrix(-1, [0, 0, 0], [0, 0, 1])
            )
        elif constant_axis == 2:
            # the wall is located at the xy plane originally
            # no need to change axis
            pass

        opened_wall_mesh = self.center_mesh(opened_wall_mesh)
        opened_wall_mesh.apply_translation(center)

        # nudge a little bit to avoid collision
        if "north" in self.direction:
            opened_wall_mesh.apply_translation([0, 0, 0.01 / 2])
        elif "south" in self.direction:
            opened_wall_mesh.apply_translation([0, 0, -0.01 / 2])
        elif "east" in self.direction:
            opened_wall_mesh.apply_translation([0.01 / 2, 0, 0])
        elif "west" in self.direction:
            opened_wall_mesh.apply_translation([-0.01 / 2, 0, 0])

        self._cached_mesh = opened_wall_mesh
        return opened_wall_mesh

    def _get_mesh_basic_components(
        self,
        asset_dir: Union[str, Path],
        force_one_mesh: bool = False,
        force_vertex_color: bool = False,
    ) -> list[dict[str, Union[np.ndarray, dict[str, Image]]]]:
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
        force_vertex_color=False,
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
            assert len(export_paths) == 1, f"More than one mesh in the wall {self.id}."
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
