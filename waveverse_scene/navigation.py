from __future__ import annotations

import ast
import math
import re
import numpy as np
from scipy.ndimage import binary_dilation
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


def find_path(
    floorplan,
    from_point: tuple[float, float],
    to_point: tuple[float, float],
    agent_radius: float = 0.0,
    resolution: float = 0.1,
) -> list[tuple[float, float]]:
    """
    Find a path considering agent radius using morphological dilation and A* pathfinding.

    Args:
        floorplan: FloorPlan2D instance
        from_point: (x, y) start position in world coordinates
        to_point: (x, y) goal position in world coordinates
        agent_radius: radius of the agent in meters
        resolution: grid resolution in meters

    Returns:
        List of (x, y) waypoints in world coordinates, or empty list if no path found
    """
    base_grid = floorplan.to_grid(resolution)
    if agent_radius > 0:
        kernel_radius = int(np.ceil(agent_radius / resolution))
        kernel = np.ones((2 * kernel_radius + 1, 2 * kernel_radius + 1))
        dilated_grid = binary_dilation(base_grid, kernel)
        grid_for_path = dilated_grid.astype(int)
    else:
        grid_for_path = base_grid
    start_x = int((from_point[0] - floorplan.bounds.min_x) / resolution)
    start_y = int((from_point[1] - floorplan.bounds.min_y) / resolution)
    end_x = int((to_point[0] - floorplan.bounds.min_x) / resolution)
    end_y = int((to_point[1] - floorplan.bounds.min_y) / resolution)
    start_x = max(start_x, 0)
    start_x = min(start_x, grid_for_path.shape[1] - 1)
    start_y = max(start_y, 0)
    start_y = min(start_y, grid_for_path.shape[0] - 1)
    end_x = max(end_x, 0)
    end_x = min(end_x, grid_for_path.shape[1] - 1)
    end_y = max(end_y, 0)
    end_y = min(end_y, grid_for_path.shape[0] - 1)
    if grid_for_path[start_y, start_x] == 1:
        if from_point[0] <= floorplan.bounds.min_x:
            from_point = (floorplan.bounds.min_x + 2 * resolution, from_point[1])
        elif from_point[0] >= floorplan.bounds.max_x:
            from_point = (floorplan.bounds.max_x - 2 * resolution, from_point[1])
        if from_point[1] <= floorplan.bounds.min_y:
            from_point = (from_point[0], floorplan.bounds.min_y + 2 * resolution)
        elif from_point[1] >= floorplan.bounds.max_y:
            from_point = (from_point[0], floorplan.bounds.max_y - 2 * resolution)
        from_point = floorplan.snap_point_to_navigable(grid_for_path, from_point, resolution)
        start_x = int((from_point[0] - floorplan.bounds.min_x) / resolution)
        start_y = int((from_point[1] - floorplan.bounds.min_y) / resolution)
    if grid_for_path[end_y, end_x] == 1:
        if to_point[0] <= floorplan.bounds.min_x:
            to_point = (floorplan.bounds.min_x + 2 * resolution, to_point[1])
        elif to_point[0] >= floorplan.bounds.max_x:
            to_point = (floorplan.bounds.max_x - 2 * resolution, to_point[1])
        if to_point[1] <= floorplan.bounds.min_y:
            to_point = (to_point[0], floorplan.bounds.min_y + 2 * resolution)
        elif to_point[1] >= floorplan.bounds.max_y:
            to_point = (to_point[0], floorplan.bounds.max_y - 2 * resolution)
        to_point = floorplan.snap_point_to_navigable(grid_for_path, to_point, resolution)
        end_x = int((to_point[0] - floorplan.bounds.min_x) / resolution)
        end_y = int((to_point[1] - floorplan.bounds.min_y) / resolution)
    start_x = max(start_x, 0)
    start_x = min(start_x, grid_for_path.shape[1] - 1)
    start_y = max(start_y, 0)
    start_y = min(start_y, grid_for_path.shape[0] - 1)
    end_x = max(end_x, 0)
    end_x = min(end_x, grid_for_path.shape[1] - 1)
    end_y = max(end_y, 0)
    end_y = min(end_y, grid_for_path.shape[0] - 1)
    pathfinding_grid = Grid(matrix=(1 - grid_for_path).tolist())
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always, weight=1)
    (path, _) = finder.find_path(
        pathfinding_grid.node(start_x, start_y),
        pathfinding_grid.node(end_x, end_y),
        pathfinding_grid,
    )
    if not path:
        print("no path found")
        print(start_x, start_y, end_x, end_y)
        return []
    world_path = [
        (
            floorplan.bounds.min_x + (x + 0.5) * resolution,
            floorplan.bounds.min_y + (y + 0.5) * resolution,
        )
        for (x, y) in path
    ]
    return world_path


def get_object_coordinates(floor_plan_str, object_name):
    """
    Given a floor plan string and an object name, this function returns
    the coordinate list associated with that object as a list of tuples of floats.

    The floor plan string is expected to have lines with the following format:
      <object name> | <location> | [<coordinates>]

    Args:
        floor_plan_str (str): The full multiline floor plan description.
        object_name (str): The exact object name to search for (including any extra info).

    Returns:
        list of tuples: The coordinate list if found, or None if not found.
    """
    pattern = re.compile(re.escape(object_name) + ".*?(\\[[^\\]]+\\])")
    match = pattern.search(floor_plan_str)
    if match:
        coord_list_str = match.group(1)
        try:
            coords = ast.literal_eval(coord_list_str)
            parsed_coords = [tuple(map(float, c)) for c in coords]
            return parsed_coords
        except Exception as e:
            print(f"Error parsing coordinates: {e}")
            return None
    else:
        return None


def adjust_point_with_margin_and_direction(point, bbox, other_point, margin=0.2):
    """
    Adjust a point (on the x–z plane) so that it is outside the object's bounding box
    and at least 'margin' away from it. If the point lies inside the bbox, it is moved
    to exactly margin outside the box. If it lies outside the bbox but within the margin,
    it is nudged to the nearest safe boundary. In both cases, the adjustment is biased toward
    moving the point in the direction of other_point, when possible.

    Parameters:
      point (tuple): (x, z) coordinates to adjust.
      bbox (tuple): Two (x, z) tuples defining opposite corners of the bounding box.
      other_point (tuple): (x, z) coordinates of the other point (directional bias).
      margin (float): Desired minimal distance from the bounding box (default 0.1).

    Returns:
      tuple: The adjusted (x, z) coordinates.
    """
    (x, z) = point
    ((x1, z1), (x2, z2)) = bbox
    (min_x, max_x) = (min(x1, x2), max(x1, x2))
    (min_z, max_z) = (min(z1, z2), max(z1, z2))
    (safe_min_x, safe_max_x) = (min_x - margin, max_x + margin)
    (safe_min_z, safe_max_z) = (min_z - margin, max_z + margin)
    if x < safe_min_x or x > safe_max_x or z < safe_min_z or (z > safe_max_z):
        return point
    candidates = []
    d_vec = (other_point[0] - x, other_point[1] - z)
    if min_x <= x <= max_x and min_z <= z <= max_z:
        candidates.append((min_x - margin, z))
        candidates.append((max_x + margin, z))
        candidates.append((x, min_z - margin))
        candidates.append((x, max_z + margin))
    else:
        if x < min_x:
            candidates.append((safe_min_x, z))
        elif x > max_x:
            candidates.append((safe_max_x, z))
        if z < min_z:
            candidates.append((x, safe_min_z))
        elif z > max_z:
            candidates.append((x, safe_max_z))
        if x < min_x and z < min_z:
            candidates.append((safe_min_x, safe_min_z))
        if x < min_x and z > max_z:
            candidates.append((safe_min_x, safe_max_z))
        if x > max_x and z < min_z:
            candidates.append((safe_max_x, safe_min_z))
        if x > max_x and z > max_z:
            candidates.append((safe_max_x, safe_max_z))
    best_candidate = None
    best_mag = float("inf")
    candidates_in_direction = []
    for cand in candidates:
        (dx, dz) = (cand[0] - x, cand[1] - z)
        mag = math.hypot(dx, dz)
        dot = dx * d_vec[0] + dz * d_vec[1]
        if dot > 0:
            candidates_in_direction.append((cand, mag))
    if candidates_in_direction:
        (best_candidate, best_mag) = min(candidates_in_direction, key=lambda item: item[1])
    else:
        for cand in candidates:
            (dx, dz) = (cand[0] - x, cand[1] - z)
            mag = math.hypot(dx, dz)
            if mag < best_mag:
                best_candidate = cand
                best_mag = mag
    return best_candidate
