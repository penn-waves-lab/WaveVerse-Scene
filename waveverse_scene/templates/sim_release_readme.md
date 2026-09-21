# WaveVerse Scenes

__SCENES__ scenes with __MOTIONS__ human motion sequences and __FRAMES__ frames
at 20 FPS.

## Layout

```text
data/<scene>/
├── <scene>.xml
├── scene.json
├── meshes/
└── person/
    └── motion_000000/
        ├── description.txt
        └── 000000/                  # One folder per frame
            ├── pose.ply
            └── correspondence.pkl
```

Meshes use Z-up world coordinates in metres. RF materials are embedded in the
scene XML. Human frames are already placed in the room; no additional alignment
is needed. Only final meshes are included.

## Load

Use the WaveVerse-Sim environment. From this directory:

```python
from load_scene import load_scene

scene = load_scene("data/<scene>")
# Load a human motion frame:
scene_with_human = load_scene("data/<scene>", motion=0, frame=0)
```

Advance `frame` through the saved sequence to animate the human. Each frame's
`correspondence.pkl` supplies the body-part mapping for grouped scattering.
