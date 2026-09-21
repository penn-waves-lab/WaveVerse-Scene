"""SMPL sequence fitting, placement, and body-part correspondence export."""

import json
import random
from pathlib import Path

import h5py
import numpy as np
import smplx
import torch
import trimesh

from third_party.smplify import config as fit_config
from third_party.smplify.smplify import SMPLify3D
from .config import require_files, save_json
from .correspondence import generate_human_dict
from .motion_files import joints_dir
from .trajectory import mesh_scene_transform


def fit_meshes(output, config, max_frames=None):
    output = Path(output)
    assets = config["mesh_assets"]
    body = Path(assets["body_models"])
    prior = Path(assets["prior"])
    require_files(
        [
            body / "smpl/SMPL_NEUTRAL.pkl",
            prior / "gmm_08.pkl",
            prior / "neutral_smpl_mean_params.h5",
            assets["segmentation"],
        ]
    )
    fit_config.GMM_MODEL_DIR = str(prior)
    fit_config.Part_Seg_DIR = str(prior / "smplx_parts_segm.pkl")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    model = smplx.create(
        str(body), model_type="smpl", gender="neutral", ext="pkl", batch_size=1
    ).to(device)
    with h5py.File(prior / "neutral_smpl_mean_params.h5", "r") as h5:
        mean_pose = torch.from_numpy(h5["pose"][:]).float().to(device)
        mean_shape = torch.from_numpy(h5["shape"][:]).float().to(device)
    fitter = SMPLify3D(
        smplxmodel=model,
        batch_size=1,
        joints_category="AMASS",
        num_iters=config["smpl_iterations"],
        device=device,
        shared_shape=True,
        temporal_pose_weight=20.0,
        floor_height=0.0,
    )
    segments = json.loads(Path(assets["segmentation"]).read_text())
    inference = json.loads((output / "motion/inference.json").read_text())
    records = []
    for motion in inference["motions"]:
        name = motion["name"]
        joints = np.load(joints_dir(output) / (name + ".npy"))[0].copy()
        # The historical renderer grounds the complete sequence before fitting.
        height_offset = float(joints[..., 1].min())
        joints[..., 1] -= height_offset
        if max_frames is not None:
            joints = joints[:max_frames]
        frames = len(joints)
        pose = mean_pose[None].repeat(frames, 1)
        shape = mean_shape[None].repeat(frames, 1)
        camera = torch.zeros(frames, 3, device=device)
        keypoints = torch.tensor(joints, dtype=torch.float32, device=device)
        _, _, pose, shape, camera, loss = fitter(
            pose.detach(),
            shape.detach(),
            camera.detach(),
            keypoints,
            conf_3d=torch.ones(22, device=device),
        )
        with torch.no_grad():
            fitted = model(
                betas=shape,
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:],
                transl=camera[:, 0, :],
                return_verts=True,
            )
            vertices = fitted.vertices.cpu().numpy()
            joint_errors = torch.linalg.vector_norm(
                fitted.joints[:, :22] - keypoints, dim=-1
            ).cpu().numpy()
        if not np.isfinite(vertices).all():
            raise ValueError("Nonfinite SMPL vertices for " + name)
        raw_dir = output / "motion/.mesh_work" / name
        world_dir = output / "motion/meshes" / name
        corr_dir = output / "motion/correspondences" / name
        for directory in (raw_dir, world_dir, corr_dir):
            directory.mkdir(parents=True, exist_ok=True)
        transform = mesh_scene_transform(
            motion["translation_xz"], motion["rotation_y"],
            motion.get("coordinate_mapping", "legacy_yz_swap"))
        for index, verts in enumerate(vertices):
            mesh = trimesh.Trimesh(vertices=verts, faces=model.faces, process=False)
            filename = f"{index:06d}.ply"
            mesh.export(raw_dir / filename)
            # Preserve SMPL vertex IDs and outward winding, including when
            # refitting a historical run that used the reflected convention.
            mesh.apply_transform(transform)
            mesh.export(world_dir / filename)
        generate_human_dict(world_dir, corr_dir, segments)
        records.append(
            {
                "name": name,
                "frames": frames,
                "vertices": int(vertices.shape[1]),
                "faces": int(len(model.faces)),
                "loss": float(loss.detach().cpu()),
                "height_offset": height_offset,
                "mean_joint_error_m": float(joint_errors.mean()),
            }
        )
    result = {
        "motions": records,
        "coordinate_frame": "z-up, metres",
        "fps": inference["fps"],
        "max_frames": max_frames,
        "smpl_iterations": config["smpl_iterations"],
        "shared_shape": True,
        "temporal_pose_weight": 20.0,
        "floor_height_m": 0.0,
    }
    save_json(output / "motion/meshes.json", result)
    return result
