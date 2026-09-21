from abc import abstractmethod
from pathlib import Path
from typing import Optional, Literal

from .base import Point3, Rotation, BaseMeshEntity
from .base_glb_entity import BaseGLBEntity
from .base_pklgz_entity import BasePklGzEntity


class BaseFurnitureEntity(BaseMeshEntity):
    """
    Abstract class for furniture entities.
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def furniture_type(self) -> str:
        pass


class ObjaverseFurnitureEntity(BasePklGzEntity, BaseFurnitureEntity):
    """
    A class to represent an objaverse furniture entity.
    """

    def __init__(self):
        super().__init__()

        # BaseEntity already defines an id
        # self.id: str = ""

        # BaseGLBEntity already defines an asset_id, position, and rotation
        # self.asset_id: str = ""
        # self.position: Point3 = Point3(0.0, 0.0, 0.0)
        # self.rotation: Rotation = Rotation(0.0, 0.0, 0.0)

        self.kinematic: bool = True
        self.material: Optional[str] = None
        self.room_id: str = ""
        self.vertices: list[list[float]] = []
        self.object_name: str = ""
        self.layer: str = ""

    @classmethod
    def load_from_dict(cls, dict_data: dict[str, any]):
        instance = super().load_from_dict(dict_data)

        # Convert complex types after basic dict loading
        if "position" in instance.__dict__:
            instance.position = Point3.load_from_dict(instance.position)

        if "rotation" in instance.__dict__:
            instance.rotation = Rotation.load_from_dict(instance.rotation)

        return instance

    @property
    def furniture_type(self) -> str:
        return "objaverse"

    def get_mesh(self, asset_dir):
        # objaverse furniture assets are stored in a directory named after the asset_id
        if self.asset_id not in str(asset_dir):
            asset_dir = Path(asset_dir) / self.asset_id
        return super().get_mesh(asset_dir)

    def _get_mesh_basic_components(self, asset_dir, force_one_mesh=False, force_vertex_color=False):
        if self.asset_id not in str(asset_dir):
            asset_dir = Path(asset_dir) / self.asset_id
        return super()._get_mesh_basic_components(asset_dir, force_one_mesh, force_vertex_color)

    def to_mitsuba_xml(
        self,
        asset_dir="",
        ply_output_dir="",
        up: Literal["y", "z"] = "y",
        export_ply=False,
        force_one_mesh=False,
        force_vertex_color=False,
    ) -> str:
        if self.asset_id not in str(asset_dir):
            asset_dir = Path(asset_dir) / self.asset_id
        return super().to_mitsuba_xml(
            asset_dir, ply_output_dir, up, export_ply, force_one_mesh, force_vertex_color
        )


class ProcTHORFurnitureEntity(BaseGLBEntity, BaseFurnitureEntity):
    """
    A class to represent a procthor furniture entity.
    """

    def __init__(self):
        super().__init__()

        # BaseEntity already defines an id
        # self.id: str = ""

        # BaseGLBEntity already defines an asset_id, position, and rotation
        # self.asset_id: str = ""
        # self.position: Point3 = Point3(0.0, 0.0, 0.0)
        # self.rotation: Rotation = Rotation(0.0, 0.0, 0.0)

        self.kinematic: bool = True
        self.material: str | None = None
        self.room_id: str = ""
        self.layer: str = ""

    @classmethod
    def load_from_dict(cls, dict_data: dict) -> "ProcTHORFurnitureEntity":
        instance = super().load_from_dict(dict_data)

        # Convert complex types after basic dict loading
        if "position" in instance.__dict__:
            instance.position = Point3.load_from_dict(instance.position)

        if "rotation" in instance.__dict__:
            instance.rotation = Rotation.load_from_dict(instance.rotation)

        return instance

    @property
    def furniture_type(self) -> str:
        return "procthor"
