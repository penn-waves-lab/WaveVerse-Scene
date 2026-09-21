# WaveVerse-Scene: Scene and Human Motion Generation

<a href="https://arxiv.org/abs/2508.12176"><img src="https://img.shields.io/badge/arXiv-2508.12176-b31b1b.svg" alt="arXiv"></a>
<a href="https://waves.seas.upenn.edu/projects/waveverse/"><img src="https://img.shields.io/badge/Project-Website-green" alt="Project Page"></a>

WaveVerse-Scene provides the scene and human motion generation pipeline for
*Scalable RF Simulation in Generative 4D Worlds*. Starting from a short description
of an indoor environment, it creates a furnished 3D room, proposes human activities,
and plans paths through the room. The motion model uses the activity descriptions
and paths to generate human movement, which is fitted to body meshes and placed
in the scene.

The exported scenes include static geometry with RF material assignments and
per-frame human meshes with body-part mappings for
[WaveVerse-Sim](https://github.com/penn-waves-lab/WaveVerse-Sim). This repository also
provides motion and scene visualization tools, plus a downloadable library of
100 scenes with optional human motion. For training the motion model, see
[WaveVerse-HMG](https://github.com/penn-waves-lab/WaveVerse-HMG).

## 🛠️ Installation

Requires Linux, Conda, Git, and an NVIDIA GPU. Clone the repository:

```bash
git clone https://github.com/penn-waves-lab/WaveVerse-Scene.git
cd WaveVerse-Scene
```

On Ubuntu, install the system tools:

```bash
sudo apt-get install -y build-essential git curl unzip ffmpeg libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 xvfb
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate waveverse-scene
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install setuptools==70.3.0 wheel numpy==1.23.5
python -m pip install --no-build-isolation -r requirements.txt -r requirements-holodeck.txt --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+8524eadda94df0ab2dbb2ef5a577e4d37c712897
```


## 📦 Assets

Installed assets occupy approximately **26 GB**:

| Assets | Installed size |
| --- | --- |
| Objathor and Holodeck data | 23.2 GB |
| Converted furniture, doors, and windows | 2.1 GB |
| Motion checkpoints, SMPL model, and fitting priors | 0.31 GB |

First use also caches about **3.6 GB** for Unity and pretrained encoders. Allow
**60 GB of free space** during asset setup for downloads and extraction, excluding
the Conda environment and generated outputs.

Run the commands below from the repository root. They create and populate the
required folders under `assets/`.

Download the Holodeck/Objathor data:

```bash
mkdir -p assets/objathor
TMPDIR="$PWD/assets/objathor" python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23 --path assets/objathor
python -m objathor.dataset.download_assets --version 2023_09_23 --path assets/objathor
python -m objathor.dataset.download_annotations --version 2023_09_23 --path assets/objathor
TMPDIR="$PWD/assets/objathor" python -m objathor.dataset.download_features --version 2023_09_23 --path assets/objathor
```

Download and install the converted furniture, door, and window
[bundle](https://drive.google.com/file/d/1R8gkWg5Bct5yU08vT3zKTLi5tzUJVh8C/view?usp=sharing) (1.91 GB):

```bash
python tools/download_scene_assets.py
```

Download and extract the neutral body model from the
[SMPL release](https://smpl.is.tue.mpg.de/), then install it with the command below.
Replace `/path/to/SMPL_NEUTRAL.pkl` with your downloaded model's path:

```bash
install -Dm644 /path/to/SMPL_NEUTRAL.pkl assets/body_models/smpl/SMPL_NEUTRAL.pkl
```

Download the fitting priors:

```bash
mkdir -p assets/smplify
curl -fL https://raw.githubusercontent.com/Mael-zys/T2M-GPT/main/visualize/joints2smpl/smpl_models/gmm_08.pkl -o assets/smplify/gmm_08.pkl
curl -fL https://raw.githubusercontent.com/Mael-zys/T2M-GPT/main/visualize/joints2smpl/smpl_models/neutral_smpl_mean_params.h5 -o assets/smplify/neutral_smpl_mean_params.h5
```

Download the motion checkpoints:

```bash
python tools/download_checkpoints.py
```

The commands above install the assets in the layout expected by the default
configuration:

```text
assets/
├── objathor/
│   ├── 2023_09_23/
│   │   ├── assets/
│   │   ├── annotations.json.gz
│   │   └── features/
│   └── holodeck/2023_09_23/
├── procthor/                         # Converted <assetId>.glb furniture
├── doors/                            # doorway_frame/double/door/handle_*.glb
├── windows/                          # Lowercase window asset IDs: <id>.glb
├── pretrained/                       # Downloaded motion checkpoints
├── body_models/smpl/SMPL_NEUTRAL.pkl
└── smplify/
    ├── gmm_08.pkl
    └── neutral_smpl_mean_params.h5
```

Normalization arrays, body-part mappings, and RF assignments are in `resources/`.
Check all asset paths:

```bash
python tools/check_assets.py
```

## 🚀 Generation

Provide `OPENAI_API_KEY` in your environment, then run:

```bash
python generate.py --query "a spacious dance studio" --output outputs/scene
```

On headless Linux machines, prefix this command with `xvfb-run -a`.

The pipeline automatically creates the room, tasks, paths, motions, and fitted
meshes. Settings are in [configs/scene.json](configs/scene.json).

To regenerate from a previous run, reuse its `scene.json` and `tasks.json`:

```bash
python generate.py --scene /path/to/scene.json --tasks /path/to/tasks.json --output outputs/scene
```

Use a new output directory per run, or add `--resume` to continue a matching run.
Use `--stop-after motion` to generate joints without fitting meshes.

## 🎬 Visualization

Preview generated motion:

```bash
python visualize.py --input outputs/scene
```

Open `outputs/scene/visualizations/index.html`. Add `--mesh` for fitted bodies,
or `--format gif` / `--format mp4` to export an animation. MP4 requires `ffmpeg`.

After mesh fitting, view motion inside the furnished room:

```bash
python render_scenes.py --input outputs/scene --output outputs/preview
python -m http.server 8000 --bind 127.0.0.1 --directory outputs/preview
```

Open `http://localhost:8000/`. On a remote machine, forward port `8000` to your
computer. Select a scene to play its motion and inspect it in 3D.

For automatically playing clips on the overview page (requires EGL and `ffmpeg`):

```bash
python -m pip install -r requirements-rendering.txt -c requirements.txt -c requirements-holodeck.txt
PYOPENGL_PLATFORM=egl python tools/render_scene_media.py --input outputs/scene --output outputs/preview --previews
python render_scenes.py --input outputs/scene --output outputs/preview --gallery-only
```

## 📡 Outputs and WaveVerse-Sim export

Generated rooms are saved in `sionna_scene/`; joints, fitted meshes, and body-part
correspondences are in `motion/`. RF materials come from the included assignment
table. Configure material overrides under `radio_materials` in `configs/scene.json`.

Export a completed run for WaveVerse-Sim:

```bash
python tools/export_sim_dataset.py --input outputs/scene --output outputs/scene-sim --report outputs/scene-sim-validation.json
```

The export includes a loading script and this scene layout:

```text
data/<scene>/
├── <scene>.xml                       # RF materials and relative mesh paths
├── scene.json
├── meshes/
└── person/motion_000000/
    ├── description.txt
    └── 000000/                      # One folder per frame
        ├── pose.ply
        └── correspondence.pkl
```

Only final meshes are included. Path arrays, intermediate meshes, and internal
logs are omitted; the validation report stays outside the exported dataset.

## 📥 Generated dataset downloads

Download the 100 static scenes, then optionally add one human motion per scene.

| Archive | Size | Contents |
| --- | --- | --- |
| [WaveVerse-Scenes.zip](https://drive.google.com/file/d/1QqPLo1CwTLFauaTHbrCcet-ZTdJq0EDM/view?usp=sharing) | 347 MB | Static rooms, RF materials, and the scene loader |
| [WaveVerse-Motions.zip](https://drive.google.com/file/d/1u8nVbbKWwqA5yFNNaQyJZ55fkZwsfoGr/view?usp=sharing) | 3.46 GB | Optional human motion meshes, descriptions, and body-part mappings |

Extract the scenes first; extract the motions into the same directory if needed:

```bash
unzip WaveVerse-Scenes.zip -d WaveVerse-Scenes
unzip WaveVerse-Motions.zip -d WaveVerse-Scenes
```


## 📜 Citation

If you use WaveVerse-Scene in your research, please cite:

```bibtex
@inproceedings{zheng2026scalable,
  title={Scalable RF Simulation in Generative 4D Worlds},
  author={Zhiwei Zheng and Dongyin Hu and Mingmin Zhao},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
}
```
