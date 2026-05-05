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



class RenderNLFPosesOrthographicMimic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die 3D NLF Daten (retargeted)"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Dicke der Knochen"}),
                "point_radius": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Größe der Gelenke"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet das Gesicht"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet die Hände"}),
                "hand_face_alpha": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Deckkraft für 2D Hände und Gesicht"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht als Fallback"}),
                "pose_data_fallback": ("POSEDATA", {"tooltip": "Pose Data für Hände/Füße"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("image", "mask", "log_output", "nlf_poses", "node_mappings")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Mimic"
    DESCRIPTION = "Rendert NLF Poses mit Weak Perspective (Orthographic) Projektion für perfekte Proportionen ohne Fisheye-Verzerrung."

    def process(self, nlf_poses, width, height, line_thickness, point_radius, draw_face, draw_hands, hand_face_alpha, dw_poses_fallback=None, pose_data_fallback=None):
        log_messages = ["=== RENDER NLF ORTHOGRAPHIC MIMIC ==="]
        
        try:
            # Pose-Input normalisieren
            pose_input = nlf_poses['joints3d_nonparam'][0] if isinstance(nlf_poses, dict) else nlf_poses
            
            # Kameramatrix aufbauen
            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            cx = intrinsic_matrix[0, 2]
            cy = intrinsic_matrix[1, 2]

            frames_np_rgba = []
            
            # OpenPose / DWPose Farb-Mapping
            joint_colors_rgb = [
                [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
                [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
                [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
                [255, 0, 170], [255, 0, 85]
            ]
            
            limb_seq = [
                [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
            ]
            
            for i, frame_data in enumerate(pose_input):
                frame_img = np.zeros((height, width, 3), dtype=np.uint8)
                
                if frame_data is None or len(frame_data) == 0:
                    frames_np_rgba.append(np.zeros((height, width, 4), dtype=np.uint8))
                    continue
                
                if isinstance(frame_data, torch.Tensor):
                    pts_3d = frame_data.cpu().numpy()
                else:
                    pts_3d = np.array(frame_data)
                
                # Geht davon aus, dass pts_3d[0] die primäre Person ist (sonst über Personen iterieren)
                if pts_3d.ndim == 3:
                    pts_3d = pts_3d[0] 
                
                # ==========================================================
                # DER KERN DES FIXES: WEAK PERSPECTIVE PROJECTION
                # ==========================================================
                
                # Ermittlung des Pelvis (Becken-Anker) als globale Z-Referenz
                # In SMPL/NLF ist der Pelvis meist an Index 0
                z_pelvis = pts_3d[0][2] if len(pts_3d) > 0 else 1.0
                if z_pelvis <= 0.01:
                    z_pelvis = 1.0  # Fallback bei fehlerhaften Koordinaten
                
                joints_to_draw = []
                pts_2d = []
                
                for j_idx, pt in enumerate(pts_3d):
                    if np.linalg.norm(pt) > 1e-5:
                        X, Y, Z_real = pt[0], pt[1], pt[2]
                        
                        # 1. ORTHOGRAPHISCHE PROJEKTION: 
                        # Alle Gelenke durch z_pelvis teilen, um den Fisheye-Effekt zu deaktivieren!
                        u = (fx * X / z_pelvis) + cx
                        v = (fy * Y / z_pelvis) + cy
                        pts_2d.append((u, v))
                        
                        # 2. Z-SORTING SPEICHERN:
                        # Die originalen Z-Werte (Z_real) bleiben erhalten für das Z-Sorting!
                        color_rgba = joint_colors_rgb[j_idx % len(joint_colors_rgb)]
                        joints_to_draw.append({
                            'pt': (u, v),
                            'z': Z_real,  
                            'color': color_rgba,
                            'idx': j_idx
                        })
                    else:
                        pts_2d.append(None)
                
                # --- Knochen zeichnen (Z-Sorted) ---
                limbs_to_draw = []
                for limb in limb_seq:
                    idx1, idx2 = limb[0], limb[1]
                    if idx1 < len(pts_2d) and idx2 < len(pts_2d):
                        pt1, pt2 = pts_2d[idx1], pts_2d[idx2]
                        if pt1 is not None and pt2 is not None:
                            # Nimm die echte, durchschnittliche Z-Tiefe des Knochens
                            z_avg = (pts_3d[idx1][2] + pts_3d[idx2][2]) / 2.0
                            color = joint_colors_rgb[idx2 % len(joint_colors_rgb)]
                            limbs_to_draw.append({
                                'pt1': pt1, 'pt2': pt2,
                                'z': z_avg, 'color': color
                            })
                            
                # Sortierung von hinten (hohes Z) nach vorne (kleines Z)
                limbs_to_draw.sort(key=lambda l: l['z'], reverse=True)
                for limb in limbs_to_draw:
                    x1, y1 = limb['pt1']
                    x2, y2 = limb['pt2']
                    length = math.hypot(x1 - x2, y1 - y2)
                    if length > 0.1:
                        mX, mY = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
                        polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), line_thickness), int(angle), 0, 360, 1)
                        cv2.fillConvexPoly(frame_img, polygon, limb['color'], lineType=cv2.LINE_AA)
                
                # --- Gelenke zeichnen (Z-Sorted) ---
                joints_to_draw.sort(key=lambda j: j['z'], reverse=True)
                for joint in joints_to_draw:
                    x, y = joint['pt']
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame_img, (int(x), int(y)), point_radius, joint['color'], thickness=-1, lineType=cv2.LINE_AA)
                
                # Alpha-Kanal hinzufügen
                alpha_channel = np.where(np.any(frame_img > 0, axis=-1), 255, 0).astype(np.uint8)
                frames_np_rgba.append(np.dstack((frame_img, alpha_channel)))
                
            # Konvertierung zu ComfyUI Tensor (B, H, W, C)
            frames_tensor = torch.from_numpy(np.stack(frames_np_rgba, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            node_mappings = json.dumps({"node_name": "RenderNLFPosesOrthographicMimic", "status": "success", "frames": len(pose_input)})
            
            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), nlf_poses, node_mappings)

        except Exception as e:
            log_messages.append(traceback.format_exc())
            return (torch.zeros((1, height, width, 3)), torch.zeros((1, height, width)), "\n".join(log_messages), nlf_poses, "{}")


