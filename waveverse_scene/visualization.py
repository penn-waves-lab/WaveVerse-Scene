"""CPU previews of saved Z-up motion, with optional fitted meshes and floor plans."""

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import re
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation, pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from plyfile import PlyData

from .config import save_json
from .motion_files import joints_dir, meshes_dir

# HumanML3D's 22-joint topology, also used by the original motion renderer.
CHAINS = ((0, 2, 5, 8, 11), (0, 1, 4, 7, 10), (0, 3, 6, 9, 12, 15),
          (9, 14, 17, 19, 21), (9, 13, 16, 18, 20))
COLORS = ("#e27455", "#3377b5", "#304455", "#e27455", "#3377b5")


@dataclass
class Motion:
    name: str
    action: str
    fps: float
    joints: np.ndarray
    path: np.ndarray | None
    meshes: list[Path] | None


def load_motion(run, index, mesh=False):
    """Read world coordinates directly; never rotate or swap the world array again."""
    run = Path(run)
    inference_path = run / "motion/inference.json"
    if not inference_path.is_file():
        raise FileNotFoundError("Run generation through --stop-after motion first: " + str(run))
    inference = json.loads(inference_path.read_text())
    records = inference["motions"]
    if not 0 <= index < len(records):
        raise ValueError(f"Motion index {index} is outside the {len(records)} saved motions")
    record = records[index]
    name = record["name"]
    if not re.fullmatch(r"motion_\d+", name):
        raise ValueError("Invalid saved motion name: " + name)
    fps = float(inference["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Motion fps must be finite and positive")
    joints = np.load(joints_dir(run) / (name + "_world.npy"), allow_pickle=False)
    if joints.shape != (record["frames"], 22, 3) or len(joints) == 0 or not np.isfinite(joints).all():
        raise ValueError("Expected finite world joints with shape (frames, 22, 3): " + name)
    target = run / "motion/motion_task.json"
    path = None
    if target.is_file():
        path = np.asarray(json.loads(target.read_text())[index][1], dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or not len(path) or not np.isfinite(path).all():
            raise ValueError("Invalid requested path for " + name)
    files = None
    if mesh:
        manifest = run / "motion/meshes.json"
        if not manifest.is_file():
            raise FileNotFoundError("Finish the mesh stage first, or omit --mesh")
        fitted = next((r for r in json.loads(manifest.read_text())["motions"] if r["name"] == name), None)
        if fitted is None or not 0 < fitted["frames"] <= len(joints):
            raise ValueError("No valid fitted frame count for " + name)
        files = [meshes_dir(run, name) / f"{i:06d}.ply"
                 for i in range(fitted["frames"])]
        missing = next((p for p in files if not p.is_file()), None)
        if missing:
            raise FileNotFoundError("Missing fitted frame: " + str(missing))
        joints = joints[:len(files)].copy()
        joints[..., 2] -= float(fitted["height_offset"])
    return Motion(name, str(record["action"]), fps, joints, path, files)


def load_floor_plan(run):
    run = Path(run)
    grid_path, meta_path = run / "occupancy_grid.npy", run / "occupancy_grid.json"
    if not grid_path.exists() and not meta_path.exists():
        return None
    grid = np.load(grid_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text())
    resolution = float(meta["resolution"])
    if (grid.ndim != 2 or not grid.size or not np.isfinite(grid).all()
            or not np.isfinite(resolution) or resolution <= 0
            or meta["axis_order"] != ["z", "x"]):
        raise ValueError("Invalid saved occupancy grid")
    x, y = float(meta["min_x"]), float(meta["min_z"])
    extent = (x, x + grid.shape[1] * resolution, y, y + grid.shape[0] * resolution)
    if not np.isfinite(extent).all():
        raise ValueError("Invalid occupancy grid extent")
    return grid, extent


def _mesh_triangles(path, root_xy):
    ply = PlyData.read(path)
    vertices = np.column_stack([ply["vertex"][axis] for axis in "xyz"])
    faces = np.stack(ply["face"]["vertex_indices"])
    if faces.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Expected a finite triangular mesh: " + str(path))
    vertices[:, :2] -= root_xy
    return vertices[faces]


def make_figure(motion, floor_plan=None, show_scene=True):
    """The pose camera follows root XY; the floor-plan panel retains world XY."""
    fig = plt.figure(figsize=(10.8 if show_scene else 5.6, 5.6), dpi=100, facecolor="#f8fafc")
    ax = fig.add_subplot(1, 2 if show_scene else 1, 1, projection="3d")
    ax.set_facecolor("#f8fafc")
    ax.set_title("Fitted mesh" if motion.meshes else "Generated joints", fontsize=11, pad=8)
    centered = motion.joints.copy()
    centered[..., :2] -= motion.joints[:, :1, :2]
    radius = max(1.05, float(np.abs(centered[..., :2]).max()) + 0.15)
    zmin = min(-0.05, float(centered[..., 2].min()) - 0.05)
    zmax = max(2.05, float(centered[..., 2].max()) + 0.15)
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(zmin, zmax),
           xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.set_box_aspect((2 * radius, 2 * radius, zmax - zmin))
    ax.view_init(elev=18, azim=-55)
    ax.tick_params(labelsize=8)
    ground = [[(-radius, -radius, 0), (radius, -radius, 0),
               (radius, radius, 0), (-radius, radius, 0)]]
    ax.add_collection3d(Poly3DCollection(ground, facecolor="#dce5eb", alpha=0.3))
    lines, body = [], None
    if motion.meshes:
        body = Poly3DCollection([], facecolor="#83b6cd", edgecolor="none", linewidth=0)
        ax.add_collection3d(body)
    else:
        lines = [ax.plot([], [], [], color=color, linewidth=3, marker="o", markersize=3)[0]
                 for color in COLORS]
    trail = marker = None
    if show_scene:
        plan = fig.add_subplot(1, 2, 2)
        plan.set_facecolor("#ffffff")
        bounds = motion.joints[:, 0, :2]
        if motion.path is not None:
            bounds = np.concatenate((bounds, motion.path))
            plan.plot(*motion.path.T, "--", color="#bd7d16", linewidth=2, label="Requested path")
        if floor_plan is not None:
            grid, extent = floor_plan
            plan.imshow(grid, origin="lower", extent=extent, interpolation="nearest", vmin=0, vmax=1,
                        cmap=ListedColormap(("#ffffff", "#d5dde5")), zorder=0)
            bounds = np.concatenate((bounds, [[extent[0], extent[2]], [extent[1], extent[3]]]))
        margin = max(0.35, float(np.ptp(bounds, axis=0).max()) * 0.05)
        lo, hi = bounds.min(axis=0) - margin, bounds.max(axis=0) + margin
        plan.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="x (m)", ylabel="y (m)",
                 title="Room floor plan" if floor_plan else "World trajectory")
        plan.set_aspect("equal")
        plan.grid(alpha=0.15)
        trail, = plan.plot([], [], color="#3377b5", linewidth=2.5, label="Generated root")
        marker, = plan.plot([], [], "o", color="#df6847", markersize=8, label="Current position")
        plan.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle(fill(motion.action, 75), fontsize=14, y=0.97, color="#263848")
    status = fig.text(0.5, 0.035, "", ha="center", fontsize=10, color="#526575")
    fig.subplots_adjust(left=0.045, right=0.96, bottom=0.15, top=0.83, wspace=0.25)

    def update(index):
        if body is not None:
            triangles = _mesh_triangles(motion.meshes[index], motion.joints[index, 0, :2])
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            light = np.array([1., -1., 2.]) / np.sqrt(6)
            brightness = 0.4 + 0.6 * np.clip(normals @ light, 0, 1)
            body.set_verts(triangles)
            body.set_facecolor(brightness[:, None] * np.array([0.51, 0.71, 0.80]))
        else:
            for line, chain in zip(lines, CHAINS):
                line.set_data_3d(*centered[index, list(chain)].T)
        if trail is not None:
            trail.set_data(*motion.joints[:index + 1, 0, :2].T)
            marker.set_data([motion.joints[index, 0, 0]], [motion.joints[index, 0, 1]])
        status.set_text(f"Frame {index + 1} / {len(motion.joints)}    |    "
                        f"{index / motion.fps:.2f} s    |    Pose camera follows the root")
        return lines + ([body] if body is not None else [])

    update(0)
    return fig, update


def _offline_html(path, title):
    # Matplotlib embeds its playback code and frames, but uses a CDN for button icons.
    # Text buttons make the saved page fully offline and keep the controls accessible.
    fragment = path.read_text()
    fragment = re.sub(r'<link\b[^>]*href="https://maxcdn.bootstrapcdn.com/font-awesome/[^>]*>', "", fragment)
    fragment = re.sub(r'(<button\b[^>]*title="([^"]+)"[^>]*>)\s*<i\b[^>]*></i>',
                      lambda m: m[1] + escape(m[2]), fragment)
    path.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width, initial-scale=1">'
                    f'<title>{escape(title)} · WaveVerse</title>'
                    '<style>body{font:15px system-ui;background:#f8fafc;color:#263848;'
                    'max-width:1120px;margin:24px auto;padding:0 16px}img{max-width:100%}'
                    'button{padding:6px 9px;cursor:pointer}.anim-buttons{display:flex;'
                    'flex-wrap:wrap;justify-content:center;gap:4px}</style>'
                    f'<h1>{escape(title)}</h1><p>Play, pause, scrub frames, or change speed. '
                    'This preview works offline.</p>' + fragment + '</html>')


def _write_gallery(output, entries):
    cards = []
    for entry in entries:
        cards.append(f'<article><a href="{escape(entry["file"], quote=True)}">'
                     f'<img src="{escape(entry["poster"], quote=True)}" alt="Motion preview">'
                     f'<h2>{escape(entry["action"])}</h2></a>'
                     f'<p>{entry["frames"]} frames · {entry["fps"]:g} fps · '
                     f'{escape(entry["representation"])} · {escape(entry["format"])}</p></article>')
    path = output / "index.html"
    path.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width, initial-scale=1">'
                    '<title>WaveVerse motion previews</title><style>'
                    'body{font:16px system-ui;background:#f8fafc;color:#263848;max-width:1100px;'
                    'margin:32px auto;padding:0 20px}article{background:white;border:1px solid '
                    '#d5dde5;border-radius:12px;padding:16px;margin:20px 0}img{width:100%}'
                    'a{color:inherit;text-decoration:none}h2{font-size:20px}</style>'
                    '<h1>WaveVerse · Motion previews</h1><p>Select a motion to view it.</p>'
                    + ''.join(cards) + '</html>')
    return path


def visualize_run(run, *, output=None, motion_index=None, file_format="html", mesh=False,
                  show_scene=True, stride=1):
    if file_format not in ("html", "gif", "mp4") or stride < 1:
        raise ValueError("Choose html, gif, or mp4 and a positive frame stride")
    if file_format == "mp4" and not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("MP4 export needs ffmpeg on PATH; use --format html or gif instead")
    run = Path(run).expanduser().resolve()
    inference_path = run / "motion/inference.json"
    if not inference_path.is_file():
        raise FileNotFoundError("Run generation through --stop-after motion first: " + str(run))
    count = len(json.loads(inference_path.read_text())["motions"])
    if not count:
        raise ValueError("No saved motions to visualize")
    indices = range(count) if motion_index is None else [motion_index]
    output = Path(output).expanduser().resolve() if output else run / "visualizations"
    floor_plan = load_floor_plan(run) if show_scene else None
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "previews.json"
    entries = json.loads(manifest_path.read_text())["previews"] if manifest_path.exists() else []
    for index in indices:
        motion = load_motion(run, index, mesh=mesh)
        representation = "mesh" if mesh else "joints"
        stem = motion.name + "_" + representation + ("" if show_scene else "_solo")
        target = output / (stem + "." + file_format)
        poster = output / (stem + ".png")
        frames = range(0, len(motion.joints), stride)
        fps = motion.fps / stride
        print(f"Rendering {motion.name}: {len(frames)} {representation} frames -> {target.name}", flush=True)
        fig, update = make_figure(motion, floor_plan=floor_plan, show_scene=show_scene)
        try:
            fig.savefig(poster, dpi=100)
            anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps,
                                           blit=False, cache_frame_data=False)
            if file_format == "html":
                writer = animation.HTMLWriter(fps=fps, embed_frames=True, embed_limit=float("inf"))
            elif file_format == "gif":
                writer = animation.PillowWriter(fps=fps)
            else:
                writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
            anim.save(target, writer=writer, dpi=100)
            if file_format == "html":
                _offline_html(target, motion.action)
        finally:
            plt.close(fig)
        entries = [entry for entry in entries if entry["file"] != target.name]
        entries.append({"file": target.name, "poster": poster.name, "motion": motion.name,
                        "action": motion.action, "representation": representation,
                        "frames": len(motion.joints), "rendered_frames": len(frames),
                        "fps": motion.fps, "playback_fps": fps, "stride": stride,
                        "format": file_format, "show_scene": show_scene})
        save_json(manifest_path, {"previews": entries})
        _write_gallery(output, entries)
    return output / "index.html"
