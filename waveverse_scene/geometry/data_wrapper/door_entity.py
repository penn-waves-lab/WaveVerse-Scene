import warnings
from pathlib import Path
from typing import Union, Optional, Literal

import numpy as np
import trimesh
from PIL.Image import Image

from .base import Point3, BaseMeshEntity, Rotation
from .base_glb_entity import BaseGLBEntity


class DoorEntity(BaseMeshEntity):
    """
    A class to represent a door entity.

    One can consider a ``DoorEntity`` as a wrapper over multiple ``BaseGLBEntity``.
    """

    def __init__(self):
        super().__init__()

        # BaseEntity already defines an id
        # self.id: str = ""

        # BaseGLBEntity already defines an asset_id
        # self.asset_id: str = ""

        self.openable: bool = False
        self.openness: float = 0.0
        self.room0: str = ""
        self.room1: str = ""
        self.wall0: str = ""
        self.wall1: str = ""
        self.hole_polygon: list[Point3] = []
        self.asset_position: Point3 = Point3(0.0, 0.0, 0.0)
        self.door_boxes: list[list[list[float]]] = []
        self.door_segment: list[list[float]] = []

        self.seed = None
        self.rng = None

    @property
    def door_type(self) -> str:
        return "double" if "double" in self.asset_id.lower() else "single"

    @property
    def needs_rotation(self) -> bool:
        return "west" in self.wall0.lower() or "west" in self.wall1.lower()

    @classmethod
    def load_from_dict(cls, dict_data: dict[str, any], seed: Optional[int] = None) -> "DoorEntity":
        dict_data["seed"] = seed
        instance = super().load_from_dict(dict_data)

        # Convert complex types after basic dict loading
        if "hole_polygon" in instance.__dict__:
            instance.hole_polygon = [
                Point3.load_from_dict(point) for point in instance.hole_polygon
            ]

        if "asset_position" in instance.__dict__:
            instance.asset_position = Point3.load_from_dict(instance.asset_position)

        instance.asset_id = instance.asset_id.lower()
        instance.rng = np.random.default_rng(seed)

        if instance.needs_rotation:
            instance.rotation = Rotation.load_from_dict({"x": 0.0, "y": 90.0, "z": 0.0})

        door_segment = instance.door_segment
        projection_dim_x = (door_segment[0][0] + door_segment[1][0]) / 2
        projection_dim_z = (door_segment[0][1] + door_segment[1][1]) / 2
        instance.position = Point3.load_from_dict(
            {"x": projection_dim_x, "y": instance.asset_position.y, "z": projection_dim_z}
        )

        return instance

    @staticmethod
    def _mirror_mesh(
        mesh: trimesh.Trimesh, mirror_plane: Literal["xy", "xz", "yz"] = "yz"
    ) -> trimesh.Trimesh:
        mirrored_mesh = mesh.copy()
        scale = np.array([1.0, 1.0, 1.0])
        if mirror_plane == "xy":
            scale[2] = -1.0
        elif mirror_plane == "xz":
            scale[1] = -1.0
        elif mirror_plane == "yz":
            scale[0] = -1.0

        mirrored_mesh.apply_transform(
            trimesh.transformations.scale_and_translate(scale=scale, translate=[0, 0, 0])
        )
        return mirrored_mesh


    def get_mesh(self, asset_dir: Union[str, Path]) -> trimesh.Trimesh:
        """
        Create a door mesh. The door mesh contains a door frame, a door, and a door handle.
        """
        if self._cached_mesh is not None:
            return self._cached_mesh

        if self.rng is None:
            self.rng = np.random.default_rng()
            warnings.warn(f"{self.id} has no seed to sample a mesh. Using a random seed.")

        asset_dir = Path(asset_dir)

        # load the door frame
        door_frame_mesh = self._get_door_frame(asset_dir)
        # door_frame_mesh = self._lift_mesh_to_xz_plane(door_frame_mesh)

        # load the door
        door_panel_and_handle = self._get_door_panel_and_handle(asset_dir)
        # door_panel_and_handle = self._lift_mesh_to_xz_plane(door_panel_and_handle)

        # form the entire door mesh
        if self.door_type == "single":
            door_mesh = trimesh.util.concatenate([door_frame_mesh, door_panel_and_handle])
        else:
            mirrored_door_panel_and_handle = self._mirror_mesh(door_panel_and_handle)
            # apply offsets.
            # the mirrored door panel and handle should be moved to the left and the origin to the right
            door_panel_x_extend = np.max(door_panel_and_handle.vertices[:, 0]) - np.min(
                door_panel_and_handle.vertices[:, 0]
            )
            door_panel_and_handle.apply_translation([door_panel_x_extend / 2, 0, 0])
            # move the mirrored door panel and handle to the left
            mirrored_door_panel_and_handle.apply_translation([-door_panel_x_extend / 2, 0, 0])

            door_mesh = trimesh.util.concatenate(
                [door_frame_mesh, door_panel_and_handle, mirrored_door_panel_and_handle]
            )

        # cache the mesh
        self._cached_mesh = door_mesh

        return door_mesh

    def _get_door_frame(self, asset_dir: Union[str, Path]) -> trimesh.Trimesh:
        """
        Sample a door frame from the asset directory.
        The door frame lies on the xy plane with y up.
        """
        # load the door frame
        if self.door_type == "single":
            asset_id_split = self.asset_id.split("_")
            door_frame_asset_id = f"{asset_id_split[0]}_frame_{asset_id_split[1]}"
        else:
            door_frame_asset_id = self.asset_id

        door_frame_entity = BaseGLBEntity.load_from_dict(
            {
                "asset_id": door_frame_asset_id,
            }
        )
        door_frame_mesh = door_frame_entity.get_mesh(asset_dir)
        return door_frame_mesh

    def _get_door_panel_and_handle(self, asset_dir: Union[str, Path]) -> trimesh.Trimesh:
        """
        Sample a door panel and a door handle from the asset directory.
        The door lies on the xy plane with y up. The door handle is attached to the lower-left of the door panel.
        """
        door_panel_asset_id = asset_dir.glob("doorway_door_*.glb")
        door_frame_asset_id = self.rng.choice([frame.stem for frame in door_panel_asset_id])

        all_handles = asset_dir.glob("doorway_handle_*.glb")
        door_handle_asset_id = self.rng.choice([handle.stem for handle in all_handles])

        door_panel_mesh = BaseGLBEntity.load_from_dict(
            {
                "asset_id": door_frame_asset_id,
            }
        ).get_mesh(asset_dir)

        door_handle_mesh = BaseGLBEntity.load_from_dict(
            {
                "asset_id": door_handle_asset_id,
            }
        ).get_mesh(asset_dir)

        # attach the handle to the door panel
        #   offsets are manually measured and do not fit all door panels
        door_panel_x_extend = np.max(door_panel_mesh.vertices[:, 0]) - np.min(
            door_panel_mesh.vertices[:, 0]
        )
        margin = 0.025

        door_handle_mesh_x_extend = np.max(door_handle_mesh.vertices[:, 0]) - np.min(
            door_handle_mesh.vertices[:, 0]
        )

        door_handle_mesh.apply_translation(
            [-door_panel_x_extend / 2 + door_handle_mesh_x_extend / 2 + margin, -0.204888, 0]
        )
        return trimesh.util.concatenate([door_panel_mesh, door_handle_mesh])

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
        colors = mesh.visual.to_color().vertex_colors / 255.0

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
            assert len(export_paths) == 1, f"More than one mesh in the door {self.id}."
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
