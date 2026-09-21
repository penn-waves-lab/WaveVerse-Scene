import os
import pickle
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import trimesh


def generate_human_dict(refine_folder, segment_folder, seg_data):
    if not segment_folder.exists():
        os.mkdir(segment_folder)
    keys_list = list(seg_data.keys())
    keys_list = sorted(keys_list)
    all_mesh_files = list(sorted(list(refine_folder.glob("*.ply"))))
    for i, mesh_file in enumerate(all_mesh_files):
        mesh = trimesh.load(mesh_file, process=False)
        mesh_file = Path(mesh_file)
        total_mesh = 0
        correspondence_log = {}
        assigned_faces = set()
        meshindex2group = {}
        group2vertices = {}
        vertices2meshindex = {}
        face_indices = np.arange(len(mesh.faces))
        for part_name in keys_list:
            part_vertices = np.array(seg_data[part_name])
            mask = np.isin(mesh.faces, part_vertices).any(axis=1)
            if assigned_faces:
                assigned_array = np.array(list(assigned_faces))
                mask &= ~np.isin(face_indices, assigned_array)
            selected_faces = np.nonzero(mask)[0].tolist()
            for index in selected_faces:
                meshindex2group[index] = part_name
            assigned_faces.update(selected_faces)
            selected_vertices = mesh.vertices[part_vertices]
            if not selected_faces:
                print(f"Warning: No faces found for {part_name}.")
                continue
            group2vertices[part_name] = selected_vertices
            total_mesh += len(selected_faces)
        vertex_coords = [tuple(v) for v in mesh.vertices]
        vertex_to_faces = defaultdict(list)
        for face_idx, face in enumerate(mesh.faces):
            for vertex_idx in face:
                vertex_to_faces[vertex_coords[vertex_idx]].append(face_idx)
        for vertex_coord, face_list in vertex_to_faces.items():
            if face_list:
                vertices2meshindex[vertex_coord] = random.choice(face_list)
        correspondence_log["meshindex2group"] = meshindex2group
        correspondence_log["group2vertices"] = group2vertices
        correspondence_log["vertices2meshindex"] = vertices2meshindex
        with open(segment_folder / f"{mesh_file.stem}.pkl", "wb") as f:
            pickle.dump(correspondence_log, f)
