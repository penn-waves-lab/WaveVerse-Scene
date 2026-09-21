from pathlib import Path
from typing import Union, Literal

import numpy as np

from .holodeck_scene import HoloDeckScene


class MitsubaExporter:
    coordinate_frames = """
        <!-- X-axis (Red) -->
        <shape type="cube">
            <transform name="to_world">
                <scale x="5" y="0.05" z="0.05"/>  <!-- Long along X, thin in Y and Z -->
                <translate x="5" y="0" z="0"/>  <!-- Position along X-axis -->
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="1, 0, 0"/>  <!-- Red -->
            </bsdf>
        </shape>
        
        <!-- Y-axis (Green) -->
        <shape type="cube">
            <transform name="to_world">
                <scale x="0.05" y="3" z="0.05"/>  <!-- Long along Y, thin in X and Z -->
                <translate x="0" y="3" z="0"/>  <!-- Position along Y-axis -->
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0, 1, 0"/>  <!-- Green -->
            </bsdf>
        </shape>
        
        <!-- Z-axis (Blue) -->
        <shape type="cube">
            <transform name="to_world">
                <scale x="0.05" y="0.05" z="5"/>  <!-- Long along Z, thin in X and Y -->
                <translate x="0" y="0" z="5"/>  <!-- Position along Z-axis -->
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0, 0, 1"/>  <!-- Blue -->
            </bsdf>
        </shape>
    """

    template = """<scene version="3.0.0">
    	<default name="integrator" value="path" />
        <default name="spp" value="{spp}" />
        <default name="resy" value="{resx}" />
        <default name="resx" value="{resy}" />
        <default name="max_depth" value="{max_depth}" />
        <integrator type="$integrator">
            <integer name="max_depth" value="$max_depth" />
        </integrator>
        <emitter type="constant">
            <rgb name="radiance" value="1.0"/>
        </emitter>
        <sensor type="perspective">
            <float name="fov" value="55" />
            <transform name="to_world">
                <lookat origin="{camera_origin}" target="{camera_target}" up="{up}"/>
            </transform>
            <sampler type="independent">
                <integer name="sample_count" value="$spp" />
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="$resx" />
                <integer name="height" value="$resy" />
                <string name="file_format" value="openexr" />
                <string name="pixel_format" value="rgb" />
                <rfilter type="tent" />
            </film>
        </sensor>
                
        {geometry}
        
	</scene>"""

    def __init__(
        self,
        scene: HoloDeckScene,
        spp: int = 128,
        resx: int = 1024,
        resy: int = 1720,
        max_depth: int = 5,
        camera_origin: np.ndarray = np.array([0, 10, 0]),
        camera_target: np.ndarray = np.array([0, -1, 0.01]),
        up: Literal["x", "y", "z"] = "y",
    ):
        self.scene = scene

        self.spp = spp
        self.resx = resx
        self.resy = resy
        self.max_depth = max_depth
        self.camera_origin = camera_origin
        self.camera_target = camera_target
        self.up = up

    @property
    def up_vector(self):
        v = np.zeros(3)
        v["xyz".index(self.up)] = 1
        return " ".join(map(str, v))

    def export(
        self,
        output_dir: Union[str, Path],
        scene_name: str = "scene",
        up: Literal["y", "z"] = "y",
        force_one_mesh: bool = False,
        force_vertex_color: bool = False,
        show_coordinate_frames: bool = False,
        show_ceiling: bool = True,
    ) -> Path:
        """
        Export the scene to a Mitsuba XML file.

        Args:
            output_dir: directory to save the XML file
            scene_name: name of the scene
            up: the up direction of the scene
            force_one_mesh: whether to force all sub-meshes into one PLY file. Used for entities with multiple meshes
            force_vertex_color: whether to force vertex color export instead of texture export. Toggle if Sionna is to be used.
            show_coordinate_frames: include coordinate frames in the scene
            show_ceiling: include the ceiling in the scene

        Returns:
            Path to the exported XML file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        geometry = ""
        for furniture in self.scene.furniture:
            geometry += furniture.to_mitsuba_xml(
                asset_dir=(
                    self.scene.objaverse_assets_dir
                    if furniture.furniture_type == "objaverse"
                    else self.scene.proc_thor_assets_dir
                ),
                ply_output_dir=output_dir,
                up=up,
                export_ply=True,
                force_one_mesh=force_one_mesh,
                force_vertex_color=force_vertex_color,
            )

        for wall in self.scene.walls:
            geometry += wall.to_mitsuba_xml(
                asset_dir="",
                ply_output_dir=output_dir,
                up=up,
                export_ply=True,
                force_one_mesh=force_one_mesh,
                force_vertex_color=force_vertex_color,
            )

        for window in self.scene.windows:
            geometry += window.to_mitsuba_xml(
                asset_dir=self.scene.window_assets_dir,
                ply_output_dir=output_dir,
                up=up,
                export_ply=True,
                force_one_mesh=force_one_mesh,
                force_vertex_color=force_vertex_color,
            )

        for door in self.scene.doors:
            geometry += door.to_mitsuba_xml(
                asset_dir=self.scene.door_assets_dir,
                ply_output_dir=output_dir,
                up=up,
                export_ply=True,
                force_one_mesh=force_one_mesh,
                force_vertex_color=force_vertex_color,
            )

        for room in self.scene.rooms:
            geometry += room.to_mitsuba_xml(
                asset_dir="",
                ply_output_dir=output_dir,
                up=up,
                export_ply=True,
                force_one_mesh=force_one_mesh,
                force_vertex_color=force_vertex_color,
                ceiling=show_ceiling,
            )

        if show_coordinate_frames:
            geometry += self.coordinate_frames

        with open(output_dir / f"{scene_name}.xml", "w") as f:
            f.write(
                self.template.format(
                    spp=self.spp,
                    resx=self.resx,
                    resy=self.resy,
                    max_depth=self.max_depth,
                    camera_origin=" ".join(map(str, self.camera_origin)),
                    camera_target=" ".join(map(str, self.camera_target)),
                    geometry=geometry,
                    up=self.up_vector,
                )
            )

        return output_dir / f"{scene_name}.xml"
