from .base import Point3, Rotation
from .base_glb_entity import BaseGLBEntity


class WindowEntity(BaseGLBEntity):
    def __init__(self):
        super().__init__()

        # BaseEntity already defines an id
        # self.id: str = ""

        # BaseGLBEntity already defines an asset_id
        # self.asset_id: str = ""

        self.room0: str = ""
        self.room1: str = ""
        self.wall0: str = ""
        self.wall1: str = ""
        self.hole_polygon: list[Point3] = []
        self.asset_position: Point3 = Point3(0.0, 0.0, 0.0)
        self.room_id: str = ""
        self.window_segment: list[list[float]] = []
        self.window_boxes: list[list[list[float]]] = []
        self.layer: str = ""

    @property
    def needs_rotation(self) -> bool:
        return "west" in self.wall0.lower() or "east" in self.wall1.lower()

    @classmethod
    def load_from_dict(cls, dict_data: dict) -> "WindowEntity":
        instance = super().load_from_dict(dict_data)

        # Convert complex types after basic dict loading
        if "hole_polygon" in instance.__dict__:
            instance.hole_polygon = [
                Point3.load_from_dict(point) for point in instance.hole_polygon
            ]

        if "asset_position" in instance.__dict__:
            instance.asset_position = Point3.load_from_dict(instance.asset_position)

        instance.asset_id = instance.asset_id.lower()

        if instance.needs_rotation:
            instance.rotation = Rotation.load_from_dict({"x": 0.0, "y": 90.0, "z": 0.0})

        window_segment = instance.window_segment
        projection_dim_x = (window_segment[0][0] + window_segment[1][0]) / 2
        projection_dim_z = (window_segment[0][1] + window_segment[1][1]) / 2
        instance.position = Point3.load_from_dict(
            {"x": projection_dim_x, "y": instance.asset_position.y, "z": projection_dim_z}
        )

        return instance
