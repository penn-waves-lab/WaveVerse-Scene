"""GPU-render room/human posters and optional synchronized overview/detail videos."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender

from waveverse_scene.config import save_json
from waveverse_scene.motion_safety import read_mesh
from waveverse_scene.motion_files import joints_dir, meshes_dir
from waveverse_scene.scene_rendering import static_meshes, motion_info, vertex_normals


def look_at(eye, target):
    eye, target = np.asarray(eye), np.asarray(target)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([x, y, z])
    matrix[:3, 3] = eye
    return matrix


def render_mesh(vertices, faces, colors=None):
    rgba = None if colors is None else np.column_stack([colors.astype(np.float32) / 255, np.ones(len(colors))])
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[.96, .26, .10, 1.] if colors is None else [1., 1., 1., 1.],
        metallicFactor=0., roughnessFactor=.8, doubleSided=True)
    return pyrender.Mesh(primitives=[pyrender.Primitive(
        positions=vertices.astype(np.float32), indices=faces.astype(np.uint32),
        normals=vertex_normals(vertices, faces), color_0=rgba, material=material)])


def choose_follow_angles(meshes, motions, run):
    """Pick stable camera azimuths with visibility of the human over the sequence."""
    import open3d as o3d

    rays = o3d.t.geometry.RaycastingScene(nthreads=1)
    for mesh in meshes:
        rays.add_triangles(o3d.core.Tensor(mesh["vertices"].astype(np.float32)),
                           o3d.core.Tensor(mesh["faces"].astype(np.uint32)))
    all_vertices = np.concatenate([m["vertices"] for m in meshes])
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2
    result = []
    for motion in motions:
        world = np.load(joints_dir(run) / (motion["name"] + "_world.npy"))[:motion["frames"]].copy()
        roots = np.asarray(motion["roots"])
        world[..., 2] -= world[0, 0, 2] - roots[0, 2]
        indices = sorted(set(np.linspace(0, len(world)-1, min(7, len(world))).astype(int).tolist() + [motion["worst_frame"]]))
        scores = []
        preferred = np.arctan2(center[1] - roots[:, 1].mean(), center[0] - roots[:, 0].mean())
        for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
            targets = world[indices].reshape(-1, 3)
            aim = np.column_stack([roots[indices, :2], np.full(len(indices), .9)])
            offset = 4.3 * np.array([np.cos(.8)*np.cos(angle), np.cos(.8)*np.sin(angle), np.sin(.8)])
            eyes = np.repeat(aim + offset, 22, axis=0)
            difference = targets - eyes
            distances = np.linalg.norm(difference, axis=1)
            vectors = difference / np.maximum(distances[:, None], 1e-8)
            tensor = o3d.core.Tensor(np.column_stack([eyes, vectors]).astype(np.float32))
            depth = rays.cast_rays(tensor, nthreads=1)["t_hit"].numpy()
            visible = np.mean(depth >= distances - .035)
            scores.append((visible + .001*np.cos(angle-preferred), angle))
        result.append(float(max(scores)[1]))
    return result


def font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()


def render_scene(run, output, renderer, video=False, preview=False):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meshes = static_meshes(run)
    motions = motion_info(run)
    if preview and (output / "media.json").is_file():
        previous = json.loads((output / "media.json").read_text())
        saved_angles = {m["motion"]: m["follow_azimuth"] for m in previous["motions"]}
        angles = [saved_angles[m["name"]] for m in motions]
    else:
        angles = choose_follow_angles(meshes, motions, run)
    vertices = np.concatenate([m["vertices"] for m in meshes])
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) / 2
    radius = max(2., float(np.linalg.norm(high-low)/2))
    overview_target = np.array([center[0], center[1], .9])
    distance = radius / np.sin(np.deg2rad(25)) * 1.08
    overview_eye = overview_target + distance*np.array([np.cos(1.)*np.cos(-np.pi/3), np.cos(1.)*np.sin(-np.pi/3), np.sin(1.)])
    overview_pose = look_at(overview_eye, overview_target)
    scene = pyrender.Scene(bg_color=[.94, .96, .98, 1.], ambient_light=[.5, .5, .5])
    for mesh in meshes:
        scene.add(render_mesh(mesh["vertices"], mesh["faces"], mesh["colors"]))
    camera = scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(50), znear=.03, zfar=200), pose=overview_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.), pose=look_at(center+[6, -8, 10], center))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=.8), pose=look_at(center+[-5, 4, 7], center))
    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    if not preview:
        flags |= pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    query = json.loads((run / "scene.json").read_text()).get("query", run.name)
    body = None
    video_entries = []
    records = []
    size = renderer.viewport_width
    def frame_image(motion, index, angle):
        nonlocal body
        if body is not None:
            scene.remove_node(body)
        mesh_file = meshes_dir(run, motion["name"]) / f"{index:06d}.ply"
        v, f = read_mesh(mesh_file)
        body = scene.add(render_mesh(v, f))
        scene.set_pose(camera, overview_pose)
        overview, _ = renderer.render(scene, flags=flags)
        root = motion["roots"][index]
        target = np.array([root[0], root[1], .9])
        offset = 4.3*np.array([np.cos(.8)*np.cos(angle), np.cos(.8)*np.sin(angle), np.sin(.8)])
        scene.set_pose(camera, look_at(target+offset, target))
        detail, _ = renderer.render(scene, flags=flags)
        if preview:
            image = Image.new("RGB", (size*2, size))
            image.paste(Image.fromarray(overview), (0, 0))
            image.paste(Image.fromarray(detail), (size, 0))
            return image
        image = Image.new("RGB", (size*2, size+80), (242, 246, 249))
        image.paste(Image.fromarray(overview), (0, 48))
        image.paste(Image.fromarray(detail), (size, 48))
        draw = ImageDraw.Draw(image)
        title = textwrap.shorten(query + " · " + motion["action"], width=95, placeholder="…")
        draw.text((18, 8), title, fill=(31, 50, 67), font=font(19))
        draw.text((18, 32), "Scene overview · ceiling hidden", fill=(77, 95, 112), font=font(12))
        draw.text((size+18, 32), "Human and nearby furniture", fill=(77, 95, 112), font=font(12))
        label = f"Frame {index+1}/{motion['frames']}  |  {index/motion['fps']:.2f} s  |  "
        label += f"placement shift {np.linalg.norm(motion['shift_xy_m'])*100:.1f} cm  |  recorded overlap {motion['frame_overlap_m'][index]*100:.1f} cm"
        draw.text((18, size+56), label, fill=(48, 66, 85), font=font(14))
        return image
    for motion, angle in zip(motions, angles):
        if not preview:
            frame_image(motion, motion["worst_frame"], angle).save(output / (motion["name"] + ".png"))
        record = {"motion": motion["name"], "poster_frame": motion["worst_frame"], "follow_azimuth": angle,
                  "overview_eye": overview_eye.tolist(), "rendered_frames": 1}
        if video or preview:
            target = output / (motion["name"] + ("_preview.mp4" if preview else "_scene.mp4"))
            temporary = target.with_name(target.stem + ".partial.mp4")
            stride = 2 if preview else 1
            frames = list(range(0, motion["frames"], stride))
            fps = motion["fps"] / stride
            height = size if preview else size+80
            command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                       f"{size*2}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                       "-threads", "2", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for index in frames:
                    process.stdin.write(np.asarray(frame_image(motion, index, angle)).tobytes())
                process.stdin.close()
                if process.wait():
                    raise RuntimeError("ffmpeg failed")
            except BaseException:
                process.kill()
                process.wait()
                raise
            temporary.replace(target)
            video_entries.append({"motion": motion["name"], "file": target.name})
            record.update(video=target.name, rendered_frames=len(frames), fps=fps,
                          source_frames=motion["frames"], source_fps=motion["fps"], stride=stride)
        records.append(record)
        print(run.name + ": " + motion["name"] + (" preview" if preview else " video" if video else " poster"), flush=True)
    if video_entries and not preview:
        (output / "videos.js").write_text("window.videoPreviews = " + json.dumps(video_entries) + ";\n")
    save_json(output / ("preview.json" if preview else "media.json"), {"scene": run.name, "ceiling_hidden": True,
              "all_walls_and_furniture_rendered": True, "source_geometry_unchanged": True, "motions": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter", default="")
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--previews", action="store_true", help="Small looping gallery clips at half the source frame rate")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing media")
    parser.add_argument("--size", type=int, help="Width of each view; defaults to 320 for previews, 640 otherwise")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if (args.input / "manifest.json").exists():
        scenes = [args.input / s["path"] for s in json.loads((args.input / "manifest.json").read_text())["scenes"]]
    else:
        scenes = [args.input]
    size = args.size or (320 if args.previews else 640)
    renderer = pyrender.OffscreenRenderer(size, size)
    try:
        for index, scene in enumerate(scenes):
            if index % args.shards != args.shard or args.filter not in scene.name:
                continue
            output = args.output / scene.name
            record_file = output / ("preview.json" if args.previews else "media.json")
            if not args.overwrite and record_file.exists():
                previous = json.loads(record_file.read_text())
                if args.previews or not args.videos or all("video" in m for m in previous["motions"]):
                    continue
            render_scene(scene, output, renderer, video=args.videos, preview=args.previews)
    finally:
        renderer.delete()


if __name__ == "__main__":
    main()
