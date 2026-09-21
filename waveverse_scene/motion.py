"""Inference with the verified, state-aware WaveVerse-HMG paper checkpoint."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .config import ROOT, file_hash, require_files, save_json
from .trajectory import normalize_scene_path, mesh_scene_transform
from .hmg.models.vqvae import HumanVQVAE
from .hmg.models.t2m_trans import Text2Motion_Transformer
from .hmg.utils.checkpoint import model_state
from .hmg.utils.motion_process import recover_from_ric


class Normalization:
    """The same inverse transforms as the training evaluator, without its dataset."""

    def __init__(self, mean, std):
        self.mean = np.load(mean)
        self.std = np.load(std)
        if (
            self.mean.shape != (263,)
            or self.std.shape != (263,)
            or not np.isfinite(self.mean).all()
            or not (self.std > 0).all()
        ):
            raise ValueError("Expected finite HumanML3D mean/std vectors of length 263.")
        self.mean_torch = torch.from_numpy(self.mean)
        self.std_torch = torch.from_numpy(self.std)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def inv_transform_torch(self, data):
        return data * self.std_torch.to(data.device) + self.mean_torch.to(data.device)


def load_models(config):
    import clip

    assets = config["motion_assets"]
    require_files(assets.values())
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select an available device in configs/scene.json.")
    cfg = SimpleNamespace(**json.loads((ROOT / "configs/hmg.json").read_text()))
    vq = HumanVQVAE(
        cfg,
        cfg.nb_code,
        cfg.code_dim,
        cfg.output_emb_width,
        cfg.down_t,
        cfg.stride_t,
        cfg.width,
        cfg.depth,
        cfg.dilation_growth_rate,
    )
    trans = Text2Motion_Transformer(
        num_vq=cfg.nb_code,
        embed_dim=cfg.embed_dim_gpt,
        clip_dim=cfg.clip_dim,
        block_size=cfg.block_size,
        num_layers=cfg.num_layers,
        n_head=cfg.n_head_gpt,
        drop_out_rate=cfg.drop_out_rate,
        fc_rate=cfg.ff_rate,
    )
    for model, path, key in [(vq, assets["vq"], "net"), (trans, assets["transformer"], "trans")]:
        model.load_state_dict(
            model_state(torch.load(path, map_location="cpu", weights_only=True), key), strict=True
        )
        model.to(device).eval().requires_grad_(False)
    encoder, _ = clip.load("ViT-B/32", device=device, jit=False)
    encoder.eval().requires_grad_(False)
    return vq, trans, encoder, Normalization(assets["mean"], assets["std"])


@torch.no_grad()
def generate_motion(output, config, tasks):
    import clip

    output = Path(output)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    vq, trans, encoder, normalization = load_models(config)
    device = config["device"]
    raw = output / "motion/joints"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (action, path) in enumerate(tasks):
        translation, rotation, normalized = normalize_scene_path(np.asarray(path))
        text = clip.tokenize([action], truncate=True).to(device)
        features = encoder.encode_text(text).float()
        paths = torch.tensor(normalized[None], dtype=torch.float32, device=device)
        tokens = trans.sample(features, paths, vq, normalization, config["categorical"])
        poses = vq.forward_decoder(tokens)
        denormalized = normalization.inv_transform(poses.cpu().numpy())
        joints = (
            recover_from_ric(torch.from_numpy(denormalized).float().to(device), 22).cpu().numpy()
        )
        if not np.isfinite(joints).all():
            raise ValueError("Nonfinite HMG output for " + action)
        name = f"motion_{i:06d}"
        np.save(raw / (name + ".npy"), joints)
        np.save(raw / (name + "_tokens.npy"), tokens.cpu().numpy())
        transform = mesh_scene_transform(translation[0], rotation, "right_handed")
        world = joints[0] @ transform[:3, :3].T + transform[:3, 3]
        np.save(raw / (name + "_world.npy"), world)
        records.append(
            {
                "name": name,
                "action": action,
                "frames": joints.shape[1],
                "translation_xz": translation[0].tolist(),
                "rotation_y": float(rotation),
                "coordinate_mapping": "right_handed",
            }
        )
    result = {
        "motions": records,
        "fps": 20,
        "seed": config["seed"],
        "categorical": config["categorical"],
        "local_joints_up": "y",
        "checkpoint_sha256": {
            key: file_hash(path) for key, path in config["motion_assets"].items()
        },
    }
    save_json(output / "motion/inference.json", result)
    return result
