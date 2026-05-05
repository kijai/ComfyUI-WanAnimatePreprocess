import os
import copy
import math
from collections import defaultdict

import random
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import logging
import traceback
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from ...models.onnx_models import ViTPose, Yolo
from ...pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from ...utils import get_face_bboxes, padding_resize, resize_by_area, resize_to_bounds
from ...pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new, draw_aaface_by_meta
from ...retarget_pose import get_retarget_pose
from ...pose_data_editor_alone_automatic import PoseDataEditorAloneAutomaticChatyNode


BODY_GROUPS = {
    "ALL": list(range(20)),
    "TORSO": [1, 2, 5, 8, 11],
    "SHOULDERS": [2, 5],
    "ARMS": [2, 3, 4, 5, 6, 7],
    "LEGS": [8, 9, 10, 11, 12, 13],
    "FEET": [10, 13, 18, 19],
    "HEAD": [0, 14, 15, 16, 17],
    "HIP_WIDTH": [8, 11],
    "KNEE_WIDTH": [9, 12],
}

HAND_GROUPS = {
    "LEFT_HAND": "left",
    "RIGHT_HAND": "right",
    "HANDS": "both",
}

FACE_GROUP = {
    "FACE": True,
}

TARGET_OPTIONS = [
    "ALL",
    "BODY",
    "TORSO",
    "SHOULDERS",
    "ARMS",
    "LEGS",
    "FEET",
    "HEAD",
    "HIP_WIDTH",
    "KNEE_WIDTH",
    "HANDS",
    "LEFT_HAND",
    "RIGHT_HAND",
    "FACE",
]

TORSO_LENGTH_PAIRS = [
    (1, 2),  # neck to right shoulder
    (1, 5),  # neck to left shoulder
    (1, 8),  # neck to right hip
    (1, 11),  # neck to left hip
    (8, 11),  # hip width
]

FULL_BODY_LENGTH_PAIRS = TORSO_LENGTH_PAIRS + [
    (2, 3),  # right shoulder to right elbow
    (3, 4),  # right elbow to right wrist
    (5, 6),  # left shoulder to left elbow
    (6, 7),  # left elbow to left wrist
    (8, 9),  # right hip to right knee
    (9, 10),  # right knee to right ankle
    (11, 12),  # left hip to left knee
    (12, 13),  # left knee to left ankle
]



class NLFPhysicalScalerV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "STRING")
    RETURN_NAMES = ("nlf_data_physically_scaled", "nlf_render_config_neutral", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Scaling"
    DESCRIPTION = "Übersetzt Kamera-Scaling (von der Scaler Node) in echte physikalische 3D-Knochengrößen."

    def process(self, video_nlf_data, nlf_render_config):
        import copy
        import numpy as np
        import json
        import torch

        log_messages = ["=== NLF PHYSICAL SCALER V1 (3D BONE SCALE) ==="]
        
        # 1. Config Parsen
        try:
            config = json.loads(nlf_render_config)
            scale_y = float(config.get("anchor_scale", 1.0))
            scale_x = float(config.get("scale_x_factor", scale_y))
            log_messages.append(f"Skalierung erkannt: Y={scale_y:.4f}, X/Z={scale_x:.4f}")
        except Exception as e:
            log_messages.append(f"Fehler beim Parsen der Config: {e}")
            return (video_nlf_data, nlf_render_config, "\n".join(log_messages))

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_scaled = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_scaled.get('joints3d_nonparam', [nlf_data_scaled])[0]
        else:
            raw_poses = nlf_data_scaled

        # --- HILFSFUNKTIONEN ---
        # Der identische Baum wie im Retargeter für Konsistenz
        tree = {0:[1,2,3], 1:[4], 4:[7], 7:[10], 2:[5], 5:[8], 8:[11], 3:[6], 6:[9], 9:[12,13,14], 12:[15], 13:[16], 16:[18], 18:[20], 20:[22], 14:[17], 17:[19], 19:[21], 21:[23]}
        
        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child); desc.extend(get_all_descendants(child, tree_map))
            return desc

        # --- VERARBEITUNG ALLER FRAMES ---
        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0: continue
            
            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())
            if pts.ndim == 3: pts = pts[0]

            # Wir wenden die Skalierung Vektor für Vektor an, beginnend beim Pelvis (Root 0)
            # Das sorgt dafür, dass die Proportionen im 3D-Raum stabil bleiben.
            
            # Alle direkten Verbindungen vom Root aus skalieren
            if 0 in tree:
                for child in tree[0]:
                    def scale_recursive(p, c):
                        # Vektor berechnen
                        vec = pts[c] - pts[p]
                        # Vektor skalieren (X/Z mit scale_x, Y mit scale_y)
                        vec[0] *= scale_x
                        vec[1] *= scale_y
                        vec[2] *= scale_x
                        
                        # Neue Position setzen
                        new_pos = pts[p] + vec
                        delta = new_pos - pts[c]
                        pts[c] = new_pos
                        
                        # Alle Nachfahren mitverschieben (Translations-Erhaltung)
                        for d in get_all_descendants(c, tree):
                            if d < len(pts) and np.linalg.norm(pts[d]) > 1e-5:
                                pts[d] += delta
                                
                        # Rekursiv weiter im Baum
                        if c in tree:
                            for grand_child in tree[c]:
                                scale_recursive(c, grand_child)
                    
                    scale_recursive(0, child)

            # GROUND ANCHOR: Damit die Person nach dem Skalieren nicht im Boden versinkt
            # (Wir halten den höchsten Fuß-Punkt auf der originalen Y-Höhe)
            v_orig_feet = [frame_data[0][idx][1].item() if is_tensor and frame_data.dim() == 3 else frame_data[idx][1] 
                           for idx in [7,8,10,11,4,5] if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
            
            v_new_feet = [pts[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
            
            if v_orig_feet and v_new_feet:
                shift = max(v_orig_feet) - max(v_new_feet)
                for j in range(len(pts)):
                    if np.linalg.norm(pts[j]) > 1e-5: pts[j][1] += shift

            # Output-Formatierung
            if is_tensor:
                if frame_data.dim() == 3: raw_poses[frame_idx][0] = torch.from_numpy(pts).to(frame_data.device)
                else: raw_poses[frame_idx] = torch.from_numpy(pts).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts.tolist()

        # --- CONFIG NEUTRALISIEREN ---
        config["anchor_scale"] = 1.0
        config["scale_x_factor"] = 1.0
        neutral_config_str = json.dumps(config)
        log_messages.append("-> Kamera-Config wurde neutralisiert (Skalierung ist nun physisch im 3D-Skelett).")

        return (nlf_data_scaled, neutral_config_str, "\n".join(log_messages))


