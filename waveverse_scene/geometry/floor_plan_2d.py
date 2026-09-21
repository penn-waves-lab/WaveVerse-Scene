from dataclasses import dataclass
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, box

from .holodeck_scene import HoloDeckScene
from scipy.ndimage import distance_transform_edt


@dataclass
class Bounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float


class FloorPlan2D:
    def __init__(self):
        # Main boundary of the space (outer walls)
        self.space_boundary: Polygon = None

        # List of obstacle polygons (furniture, inner walls, etc)
        self.obstacles: list[Polygon] = []

        # Cached navigable space (boundary minus obstacles)
        self._navigable_space: MultiPolygon = None

        # Bounds of the entire space
        self.bounds: Bounds = None

    def initialize_from_scene(self, scene: HoloDeckScene):
        # Extract room boundaries and combine them
        room_polys = []
        for room in scene.rooms:
            # Convert vertices to 2D points (assuming vertices are [x, y])
            points = [(v[0], v[1]) for v in room.vertices]
            if len(points) >= 3:  # Need at least 3 points for a polygon
                room_polys.append(Polygon(points))

        # Combine all rooms into one boundary
        self.space_boundary = room_polys[0]
        for poly in room_polys[1:]:
            self.space_boundary = self.space_boundary.union(poly)

        # Process walls and doors
        self._process_walls_and_doors(scene)

        # Furniture
        for furniture in scene.furniture:
            projection = furniture.get_projection_xz_plane(
                scene.objaverse_assets_dir
                if furniture.furniture_type == "objaverse"
                else scene.proc_thor_assets_dir
            )

            if projection is not None:
                self.obstacles.append(projection)

        # Calculate bounds
        bounds = self.space_boundary.bounds  # (min_x, min_y, max_x, max_y)
        self.bounds = Bounds(bounds[0], bounds[2], bounds[1], bounds[3])

        # Cache the navigable space
        self._update_navigable_space()

    def _process_walls_and_doors(self, scene: HoloDeckScene):
        """Process walls and doors, creating wall obstacles with appropriate holes for doors."""
        # First, create a mapping of wall IDs to their polygons
        wall_polygons = {}  # Dict[str, Polygon]
        for wall in scene.walls:
            wall_id = wall.id  # Assuming there's an id field in wall objects
            points = [(p.x, p.z) for p in wall.polygon]
            if len(points) >= 3:
                wall_polygons[wall_id] = Polygon(points)

        # Process doors and create holes in walls
        for door in scene.doors:
            if not door.door_segment or len(door.door_segment) != 2:
                continue

            # Create a small buffer around the door segment to ensure proper opening
            door_p1, door_p2 = door.door_segment
            door_line = LineString([door_p1, door_p2])
            # Use a small buffer to create a polygon that represents the door opening
            door_opening = door_line.buffer(0.1)  # 10cm buffer

            # Subtract door opening from corresponding walls
            if door.wall0 in wall_polygons:
                wall_polygons[door.wall0] = wall_polygons[door.wall0].difference(door_opening)
            if door.wall1 in wall_polygons:
                wall_polygons[door.wall1] = wall_polygons[door.wall1].difference(door_opening)

        # Add processed wall polygons to obstacles
        for wall_poly in wall_polygons.values():
            if not wall_poly.is_empty:
                # Handle potential MultiPolygon results from difference operation
                if wall_poly.geom_type == "MultiPolygon":
                    self.obstacles.extend(list(wall_poly.geoms))
                else:
                    self.obstacles.append(wall_poly)

    def _update_navigable_space(self):
        """Update the cached navigable space by subtracting obstacles from boundary."""
        navigable = self.space_boundary
        for obstacle in self.obstacles:
            navigable = navigable.difference(obstacle)
        self._navigable_space = navigable


    def get_navigable_space(self) -> MultiPolygon:
        """Get the navigable space polygons."""
        return self._navigable_space

    def to_grid(self, resolution: float = 0.1) -> np.ndarray:
        """Convert to grid representation for algorithms like A*.
        A cell is considered non-navigable if it intersects with any obstacle
        or is outside the boundary."""

        width = int(np.ceil((self.bounds.max_x - self.bounds.min_x) / resolution))
        height = int(np.ceil((self.bounds.max_y - self.bounds.min_y) / resolution))

        grid = np.ones((height, width), dtype=np.uint8)  # Start with all occupied

        # For each cell, create a box and check intersection with navigable space
        for y in range(height):
            for x in range(width):
                # Get world coordinates of cell corners
                min_x = self.bounds.min_x + x * resolution
                min_y = self.bounds.min_y + y * resolution
                max_x = min_x + resolution
                max_y = min_y + resolution

                # Create a box representing this cell
                cell_box = box(min_x, min_y, max_x, max_y)

                # Cell is navigable if:
                # 1. It's completely inside the boundary
                # 2. It doesn't intersect with any obstacle
                if self.space_boundary.contains(cell_box):
                    is_intersecting = False
                    for obstacle in self.obstacles:
                        if cell_box.intersects(obstacle):
                            is_intersecting = True
                            break
                    if not is_intersecting:
                        grid[y, x] = 0

        return grid


    def snap_point_to_navigable(
        self, grid, point: tuple[float, float], resolution: float = 0.1
    ) -> tuple[float, float]:
        """
        Given a world coordinate point (x, y), if the point lies in an obstacle,
        snap it to the closest navigable cell based on the grid representation.

        Parameters:
            point (tuple): (x, y) world coordinates.
            resolution (float): The resolution used to generate the grid (default 0.1).

        Returns:
            tuple: Adjusted (x, y) world coordinates that are navigable.
        """
        # Convert world coordinates to grid indices.
        col = int((point[0] - self.bounds.min_x) / resolution)
        row = int((point[1] - self.bounds.min_y) / resolution)

        # Ensure indices are within grid bounds.
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return point  # Point outside grid, return as is.

        # If the cell is navigable (0), return the original point.
        if grid[row, col] == 0:
            return point

        # For free_grid, 0 means the cell is free (navigable), 1 means it's occupied.
        free_grid = grid
        distances, indices = distance_transform_edt(free_grid, return_indices=True)

        # indices has shape (ndim, height, width); get nearest free cell indices.
        nearest_row = indices[0, row, col]
        nearest_col = indices[1, row, col]

        # Convert grid indices back to world coordinates (center of cell)
        world_x = self.bounds.min_x + (nearest_col + 0.5) * resolution
        world_y = self.bounds.min_y + (nearest_row + 0.5) * resolution

        return (world_x, world_y)
