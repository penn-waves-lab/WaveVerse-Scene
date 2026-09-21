import json
from pathlib import Path
from typing import Optional, Union

from .data_wrapper import (
    DoorEntity,
    BaseFurnitureEntity,
    ObjaverseFurnitureEntity,
    ProcTHORFurnitureEntity,
    RoomEntity,
    WallEntity,
    WindowEntity,
)


class HoloDeckScene:
    """
    A class to represent a holodeck scene description json.

    y-axis is up.
    """

    def __init__(
        self,
        objaverse_assets_dir: Optional[Union[str, Path]] = None,
        proc_thor_assets_dir: Optional[Union[str, Path]] = None,
        door_assets_dir: Optional[Union[str, Path]] = None,
        window_assets_dir: Optional[Union[str, Path]] = None,
    ):
        self.rooms: list[RoomEntity] = []
        self.doors: list[DoorEntity] = []
        self.objaverse_furniture: list[ObjaverseFurnitureEntity] = []
        self.proc_thor_furniture: list[ProcTHORFurnitureEntity] = []
        self.walls: list[WallEntity] = []
        self.windows: list[WindowEntity] = []

        self.wall_height: Optional[float] = None

        self.objaverse_assets_dir = (
            None if objaverse_assets_dir is None else Path(objaverse_assets_dir)
        )
        self.proc_thor_assets_dir = (
            None if proc_thor_assets_dir is None else Path(proc_thor_assets_dir)
        )
        self.door_assets_dir = None if door_assets_dir is None else Path(door_assets_dir)
        self.window_assets_dir = None if window_assets_dir is None else Path(window_assets_dir)

        self.filepath = None
        self.wall_openings: Optional[dict] = {}

    @classmethod
    def load(cls, filepath: Union[str, Path]):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        scene = cls()

        # furniture
        scene.objaverse_furniture = [
            ObjaverseFurnitureEntity.load_from_dict(furniture)
            for furniture in data.get("objects", [])
            if ("_" not in furniture.get("assetId", ""))
        ]
        scene.proc_thor_furniture = [
            ProcTHORFurnitureEntity.load_from_dict(furniture)
            for furniture in data.get("objects", [])
            if ("_" in furniture.get("assetId", ""))
        ]

        # objects on walls
        scene.doors = [DoorEntity.load_from_dict(door) for door in data.get("doors", [])]
        scene.windows = [WindowEntity.load_from_dict(window) for window in data.get("windows", [])]
        scene.gather_wall_openings(data)

        # floors and ceilings
        scene.wall_height = data.get("wall_height", data.get("wallHeight", 2.8))
        scene.rooms = [
            RoomEntity.load_from_dict(room, wall_height=scene.wall_height)
            for room in data.get("rooms", [])
        ]

        # walls. keep unique walls.
        scene.walls = scene._filter_unique_walls(
            [
                WallEntity.load_from_dict(wall, openings=scene.wall_openings[wall["id"]])
                for wall in data.get("walls", [])
                if "exterior" not in wall.get("id", "").lower()
            ]
        )

        scene.filepath = filepath

        return scene

    @property
    def furniture(self) -> list[BaseFurnitureEntity]:
        return self.objaverse_furniture + self.proc_thor_furniture

    @staticmethod
    def _filter_unique_walls(walls: list[WallEntity]) -> list[WallEntity]:
        """
        filter out walls with the same segment on the xz plane
        """

        def norm_segment(segments):
            return tuple(sorted(tuple(map(tuple, segments))))

        seen_segments = set()
        unique_walls = []
        for wall in walls:
            normed_segment = norm_segment(wall.segment)
            if normed_segment not in seen_segments:
                unique_walls.append(wall)
                seen_segments.add(normed_segment)
        return unique_walls

    def gather_wall_openings(self, scene_json_data: dict):
        """
        gather wall opening information from doors and windows
        """
        self.wall_openings = {}
        for wall in scene_json_data["walls"]:
            self.wall_openings[wall["id"]] = []

        for door in self.doors:
            if door.wall0 not in self.wall_openings or door.wall1 not in self.wall_openings:
                raise ValueError(f"Door {door.id} has invalid wall id")
            self.wall_openings[door.wall0].append(door.hole_polygon)
            self.wall_openings[door.wall1].append(door.hole_polygon)

        for window in self.windows:
            if window.wall0 not in self.wall_openings or window.wall1 not in self.wall_openings:
                raise ValueError(f"Window {window.id} has invalid wall id")
            self.wall_openings[window.wall0].append(window.hole_polygon)
            self.wall_openings[window.wall1].append(window.hole_polygon)
