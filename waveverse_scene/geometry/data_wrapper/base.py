import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Literal

import numpy as np
import trimesh
from PIL.Image import Image
from plyfile import PlyData, PlyElement
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


@dataclass
class Point3:
    x: float
    y: float
    z: float

    @classmethod
    def load_from_dict(cls, data: dict) -> "Point3":
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            z=float(data.get("z", 0.0)),
        )


@dataclass
class Rotation:
    x: float
    y: float
    z: float

    @classmethod
    def load_from_dict(cls, data: dict) -> "Rotation":
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            z=float(data.get("z", 0.0)),
        )


@dataclass
class Material:
    name: str
    ambientcg: str = ""  # Optional field

    @classmethod
    def load_from_dict(cls, data: dict) -> "Material":
        return cls(name=data.get("name", ""), ambientcg=data.get("ambientcg", ""))


class BaseEntity(ABC):
    """
    Abstract class for all entities in the scene. Works as a container of information.
    """

    def __init__(self):
        self.id: str = ""

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def load_from_dict(cls, dict_data: dict[str, any]):
        new_instance = cls()

        # Convert camelCase to snake_case for each key
        for key, value in dict_data.items():
            snake_key = "".join(["_" + c.lower() if c.isupper() else c for c in key]).lstrip("_")

            setattr(new_instance, snake_key, value)

        return new_instance

    def copy(self):
        return copy.deepcopy(self)

    @abstractmethod
    def to_mitsuba_xml(
        self,
        asset_dir: Union[str, Path] = "",
        ply_output_dir: Union[str, Path] = "",
        up: Literal["y", "z"] = "y",
        export_ply: bool = False,
        force_one_mesh: bool = False,
        force_vertex_color: bool = False,
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

        Returns:
            str: the XML string of the entity
        """
        pass

    @property
    def clean_id(self) -> str:
        """
        Get a cleaned-up version of the entity's id.

        Returns:
            str: the clean id of the entity
        """
        return re.sub(r"\W+", " ", self.id).strip().replace(" ", "_")


class BaseMeshEntity(BaseEntity):
    """
    Abstract class for entities with meshes.
    """

    def __init__(self):
        super().__init__()

        self.asset_id: str = ""
        self.position: Point3 = Point3(0.0, 0.0, 0.0)
        self.rotation: Rotation = Rotation(0.0, 0.0, 0.0)

        self._cached_mesh = None
        self._cached_projection = None

    @staticmethod
    def center_mesh(mesh: trimesh.Trimesh):
        """
        Center the mesh at the origin.

        Args:
            mesh: the mesh to center
        """
        bound_center = mesh.bounding_box.centroid
        mesh.apply_translation(-bound_center)
        return mesh

    @abstractmethod
    def get_mesh(self, asset_dir: Union[str, Path]) -> trimesh.Trimesh:
        """
        Get the mesh of the entity.

        Args:
            asset_dir: path to the asset directory

        Returns:
            trimesh.Trimesh: the mesh of the entity
        """
        pass

    @abstractmethod
    def _get_mesh_basic_components(
        self,
        asset_dir: Union[str, Path],
        force_one_mesh: bool = False,
        force_vertex_color: bool = False,
    ) -> list[dict[str, Union[np.ndarray, dict[str, Image]]]]:
        """
        Get basic components of the mesh/meshes (vertex data and face data) to export to PLY.

        Internal method called by ``to_ply``. Should be implemented by subclasses.

        Args:
            asset_dir: path to the asset directory
            force_one_mesh: whether to force all sub-meshes into one PLY file. Used for entities with multiple meshes.
            force_vertex_color: whether to force vertex color export instead of texture export

        Returns:
            list[dict[str, np.ndarray]]: list of dictionaries with mesh components. Each dictionary should contain:
                - vertex_data: an (N, D) array of vertex data, where D is the number of data fields (e.g. 3 for XYZ)
                - face_data: an (M, 3) array of face data, where M is the number of faces
                - texture_to_export: a dict with texture data to export (optional)
        """
        pass

    def get_mesh_in_world_frame(self, asset_dir: Union[str, Path] = "") -> trimesh.Trimesh:
        """
        Get the mesh of the entity in world frame.

        Args:
            asset_dir: path to the asset directory

        Returns:
            trimesh.Trimesh: the mesh of the entity in world frame
        """
        mesh = self.get_mesh(asset_dir).copy()

        # Apply rotation
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(self.rotation.y), [0, 1, 0])
        )

        # Apply translation
        mesh.apply_translation([self.position.x, self.position.y, self.position.z])

        return mesh


    def get_projection_xz_plane(self, asset_dir: Union[str, Path]) -> Union[Polygon, MultiPolygon]:
        """
        Get the projection of the mesh to the x-z plane.

        Used to construct the 2D floor plan of the entire scene.

        Args:
            asset_dir: path to the asset directory

        Returns:
            Union[Polygon, MultiPolygon]: the projection of the mesh to the x-z plane
        """

        if self._cached_projection is not None:
            return self._cached_projection

        mesh = self.get_mesh_in_world_frame(asset_dir)

        # Get the vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces

        # Create polygons for each face
        face_polygons = []
        for face in faces:
            # face is an array of indices, e.g. [0, 1, 2]
            # vertices[face] gives us the actual 3D coordinates for this face
            face_verts_3d = vertices[face]

            # Project these vertices to 2D by taking only x and z coordinates
            face_verts_2d = face_verts_3d[:, [0, 2]]  # Keep only X and Z coordinates

            # Skip degenerate faces (where vertices are collinear after projection)
            if len(face_verts_2d) >= 3:
                try:
                    poly = Polygon(face_verts_2d)
                    if poly.is_valid and not poly.is_empty:
                        face_polygons.append(poly)
                except ValueError:
                    continue  # Skip invalid polygons

        # Merge all face polygons
        if not face_polygons:
            raise ValueError("No valid polygons created from mesh projection")

        # Union all polygons and simplify the result
        merged = unary_union(face_polygons)

        # The result might be a MultiPolygon if the projection has disconnected parts
        if isinstance(merged, MultiPolygon):
            # You might want to only keep the largest polygon
            # or handle multiple polygons based on your needs
            areas = [p.area for p in merged.geoms]
            largest_poly = merged.geoms[np.argmax(areas)]
            self._cached_projection = largest_poly
            return largest_poly

        # Cache the result
        self._cached_projection = merged
        return merged

    def to_ply(
        self,
        asset_dir: Union[str, Path],
        ply_output_dir: Union[str, Path],
        up: Literal["y", "z"] = "y",
        force_one_mesh: bool = False,
        force_vertex_color: bool = False,
    ) -> list[dict[str, Path]]:
        """
        Export mesh/meshes to PLY file(s).

        Args:
            asset_dir: path to the asset directory
            ply_output_dir: path to the output directory
            up: the up direction of the entity
            force_one_mesh: whether to force all sub-meshes into one PLY file. Used for entities with multiple meshes.
            force_vertex_color: whether to force vertex color export instead of texture export

        Returns:
            list[dict[str, Path]]: list of output paths for different types of data (e.g. ply, albedo, normal, etc)
        """
        asset_dir = Path(asset_dir)
        ply_output_dir = Path(ply_output_dir)

        mesh_component_list = self._get_mesh_basic_components(
            asset_dir, force_one_mesh=force_one_mesh, force_vertex_color=force_vertex_color
        )

        exported_paths = []
        for i, results in enumerate(mesh_component_list):
            vertex_data = results["vertex_data"]
            face_data = results["face_data"]

            if up == "z":
                old_z = vertex_data["z"].copy()
                old_y = vertex_data["y"].copy()

                vertex_data["z"] = old_y
                vertex_data["y"] = old_z

                if "nx" in vertex_data.dtype.names:
                    old_nz = vertex_data["nz"].copy()
                    old_ny = vertex_data["ny"].copy()

                    vertex_data["nz"] = old_ny
                    vertex_data["ny"] = old_nz

            # Create PLY object
            ply_data = PlyData(
                [
                    PlyElement.describe(vertex_data, "vertex"),
                    PlyElement.describe(face_data, "face"),
                ],
                text=False,
            )  # binary format

            # Save PLY file
            output_path = Path(ply_output_dir) / f"{self.clean_id}.ply"
            if i > 0:
                output_path = output_path.with_stem(f"{output_path.stem}_{i}")
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)
            ply_data.write(str(output_path))

            exported_path = {"ply": output_path, "albedo": None, "normal": None}

            # check whether there is texture data to export
            if "texture_to_export" in results:
                texture_data = results.get("texture_to_export", None)
                if texture_data is not None:
                    albedo_data = texture_data.get("albedo", None)
                    if albedo_data is not None:
                        albedo_path = Path(ply_output_dir) / f"{self.clean_id}_albedo.png"
                        if i > 0:
                            albedo_path = albedo_path.with_stem(f"{albedo_path.stem}_{i}")
                        albedo_data.save(str(albedo_path))
                        exported_path["albedo"] = albedo_path

                    normal_data = texture_data.get("normal", None)
                    if normal_data is not None:
                        normal_path = Path(ply_output_dir) / f"{self.clean_id}_normal.png"
                        if i > 0:
                            normal_path = normal_path.with_stem(f"{normal_path.stem}_{i}")
                        normal_data.save(str(normal_path))
                        exported_path["normal"] = normal_path

            exported_paths.append(exported_path)

        return exported_paths
