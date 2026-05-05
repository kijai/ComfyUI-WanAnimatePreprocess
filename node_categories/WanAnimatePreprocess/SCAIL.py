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



class PoseDataToDWPoses:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
            }
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Konvertiert dein PoseData direkt in DWPoses für SCAIL."

    def process(self, pose_data):
        pose_metas = pose_data.get("pose_metas", [])
        results = []
        
        for meta in pose_metas:
            # Dimensionen für die Normalisierung holen (SCAIL erwartet 0.0 - 1.0)
            width = getattr(meta, "width", 512) or 512
            height = getattr(meta, "height", 512) or 512

            # Überprüfen, ob meta ein Dictionary oder eine AAPoseMeta Instanz ist
            is_dict = isinstance(meta, dict)

            if is_dict:
                kps_body = np.array(meta.get("keypoints_body", []))
                kps_lhand = np.array(meta.get("keypoints_left_hand", []))
                kps_rhand = np.array(meta.get("keypoints_right_hand", []))
                kps_face = np.array(meta.get("keypoints_face", []))

                # Die letzten 2 Punkte beim Körper weglassen (20 -> 18)
                candidate_body = kps_body[:-2, :2] if len(kps_body) > 2 else np.zeros((18, 2))
                score_body = kps_body[:-2, 2] if len(kps_body) > 2 else np.zeros((18,))
                
                lhand_coords = kps_lhand[:, :2] if len(kps_lhand) > 0 else np.zeros((21, 2))
                lhand_score = kps_lhand[:, 2] if len(kps_lhand) > 0 else np.zeros((21,))
                
                rhand_coords = kps_rhand[:, :2] if len(kps_rhand) > 0 else np.zeros((21, 2))
                rhand_score = kps_rhand[:, 2] if len(kps_rhand) > 0 else np.zeros((21,))

                # Bei Gesicht den ersten Punkt auslassen (69 -> 68)
                face_coords = kps_face[1:, :2] if len(kps_face) > 1 else np.zeros((68, 2))
                face_score = kps_face[1:, 2] if len(kps_face) > 1 else np.zeros((68,))

            else:
                # Meta ist eine AAPoseMeta Instanz (Koordinaten sind meist noch in Pixeln)
                b_coords = getattr(meta, "kps_body", None)
                if b_coords is not None and len(b_coords) > 2:
                    candidate_body = b_coords[:-2].copy()
                    candidate_body[:, 0] /= width
                    candidate_body[:, 1] /= height
                    score_body = getattr(meta, "kps_body_p")[:-2]
                else:
                    candidate_body = np.zeros((18, 2))
                    score_body = np.zeros((18,))

                # Linke Hand
                lh_coords = getattr(meta, "kps_lhand", None)
                if lh_coords is not None and len(lh_coords) > 0:
                    lhand_coords = lh_coords.copy()
                    lhand_coords[:, 0] /= width
                    lhand_coords[:, 1] /= height
                    lhand_score = getattr(meta, "kps_lhand_p", np.zeros((21,)))
                else:
                    lhand_coords = np.zeros((21, 2))
                    lhand_score = np.zeros((21,))

                # Rechte Hand
                rh_coords = getattr(meta, "kps_rhand", None)
                if rh_coords is not None and len(rh_coords) > 0:
                    rhand_coords = rh_coords.copy()
                    rhand_coords[:, 0] /= width
                    rhand_coords[:, 1] /= height
                    rhand_score = getattr(meta, "kps_rhand_p", np.zeros((21,)))
                else:
                    rhand_coords = np.zeros((21, 2))
                    rhand_score = np.zeros((21,))

                # Gesicht
                f_coords = getattr(meta, "kps_face", None)
                if f_coords is not None and len(f_coords) > 1:
                    face_coords = f_coords[1:].copy()
                    face_coords[:, 0] /= width
                    face_coords[:, 1] /= height
                    face_score = getattr(meta, "kps_face_p", np.zeros((69,)))[1:]
                else:
                    face_coords = np.zeros((68, 2))
                    face_score = np.zeros((68,))

            # Subset Matrix aufbauen: -1 wenn Score unter Threshold von 0.3 ist
            subset_body = np.arange(len(candidate_body), dtype=float)
            subset_body[score_body <= 0.3] = -1

            # Hände wie in DWPose/SCAIL stacken (Rechts, dann Links!)
            hands_coords = np.stack([rhand_coords, lhand_coords], axis=0)
            hands_score = np.stack([rhand_score, lhand_score], axis=0)

            # Dictionary im Zielformat für DWPose / SCAIL bauen
            dwpose_format = {
                "bodies": {
                    "candidate": np.expand_dims(candidate_body, axis=0).astype(np.float32),
                    "subset": np.expand_dims(subset_body, axis=0).astype(np.float32)
                },
                "hands": hands_coords.astype(np.float32),
                "faces": np.expand_dims(face_coords, axis=0).astype(np.float32),
                "body_score": np.expand_dims(score_body, axis=0).astype(np.float32),
                "hand_score": hands_score.astype(np.float32),
                "face_score": np.expand_dims(face_score, axis=0).astype(np.float32)
            }
            results.append(dwpose_format)

        out_dict = {'poses': results, 'swap_hands': True}
        return (out_dict,)


class RenderNLFPosesWithData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Der 3D Output aus SCAIL"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096})
            },
            "optional": {
                "dw_poses": ("DWPOSES",),
                "ref_dw_pose": ("DWPOSES",),
                "draw_face": ("BOOLEAN", {"default": True}),
                "draw_hand": ("BOOLEAN", {"default": True}),
                "draw_body": ("BOOLEAN", {"default": True}),
                # Diese Settings braucht das Rendern intern:
                "render_device": (["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"], {"default": "gpu"}),
                "scale_hands": ("BOOLEAN", {"default": True}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "NLFPRED")
    RETURN_NAMES = ("image", "mask", "nlf_poses_data")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Rendert die NLF Poses komplett eigenständig und reicht die Daten an den Scaler weiter."

    def process(self, nlf_poses, width, height, dw_poses=None, ref_dw_pose=None, draw_face=True, draw_hand=True, draw_body=True, render_device="gpu", scale_hands=True, render_backend="taichi"):
        
        # Imports aus dem NLF-Ordner
        from ...NLFPoseExtract.nlf_render import render_nlf_as_images, render_multi_nlf_as_images, shift_dwpose_according_to_nlf, process_data_to_COCO_format, intrinsic_matrix_from_field_of_view
        from ...NLFPoseExtract.align3d import solve_new_camera_params_central, solve_new_camera_params_down
        
        if render_backend == "taichi":
            try:
                import taichi as ti
                device_map = {
                    "cpu": ti.cpu, "gpu": ti.gpu, "opengl": ti.opengl,
                    "cuda": ti.cuda, "vulkan": ti.vulkan, "metal": ti.metal,
                }
                ti.init(arch=device_map.get(render_device.lower()))
            except:
                logging.warning("Taichi selected but not installed. Falling back to torch rendering.")
                render_backend = "torch"

        if isinstance(nlf_poses, dict):
            pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
        else:
            pose_input = nlf_poses

        dw_pose_input = copy.deepcopy(dw_poses["poses"]) if dw_poses is not None else None
        swap_hands = dw_poses.get("swap_hands", False) if dw_poses is not None else False

        ori_camera_pose = intrinsic_matrix_from_field_of_view([height, width])
        ori_focal = ori_camera_pose[0, 0]

        num_people = dw_pose_input[0]['bodies']['candidate'].shape[0] if dw_poses is not None else 0

        # Alignment
        if dw_poses is not None and ref_dw_pose is not None and num_people == 1:
            ref_dw_pose_input = copy.deepcopy(ref_dw_pose["poses"])

            pose_3d_first_driving_frame = None
            for pose in pose_input:
                if pose.shape[0] == 0:
                    continue
                candidate = pose[0].cpu().numpy()
                if np.any(candidate):
                    pose_3d_first_driving_frame = candidate
                    break
            if pose_3d_first_driving_frame is None:
                raise ValueError("No valid pose found in pose_input.")

            pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)
            poses_2d_ref = ref_dw_pose_input[0]['bodies']['candidate'][0][:14]
            poses_2d_ref[:, 0] = poses_2d_ref[:, 0] * width
            poses_2d_ref[:, 1] = poses_2d_ref[:, 1] * height

            poses_2d_subset = ref_dw_pose_input[0]['bodies']['subset'][0][:14]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[:14]

            valid_indices, valid_upper_indices, valid_lower_indices = [], [], []
            upper_body_indices = [0, 2, 3, 5, 6]
            lower_body_indices = [9, 10, 12, 13]

            for i in range(len(poses_2d_subset)):
                if poses_2d_subset[i] != -1.0 and np.sum(pose_3d_coco_first_driving_frame[i]) != 0:
                    if i in upper_body_indices:
                        valid_upper_indices.append(i)
                    if i in lower_body_indices:
                        valid_lower_indices.append(i)

            valid_indices = [1] + valid_lower_indices if len(valid_upper_indices) < 4 else [1] + valid_lower_indices + valid_upper_indices 

            pose_2d_ref = poses_2d_ref[valid_indices]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[valid_indices]

            if len(valid_lower_indices) >= 4:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_down(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)
            else:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_central(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)

            # HIER WIRD DIE HILFSFUNKTION AUFGERUFEN
            scale_face = scale_faces(list(dw_pose_input), list(ref_dw_pose_input))   

            logging.info(f"Scale - m: {scale_m}, face: {scale_face}")
            shift_dwpose_according_to_nlf(pose_input, dw_pose_input, ori_camera_pose, new_camera_intrinsics, height, width, swap_hands=swap_hands, scale_hands=scale_hands, scale_x=scale_m, scale_y=scale_m*scale_s)

            intrinsic_matrix = new_camera_intrinsics
        else:
            intrinsic_matrix = ori_camera_pose

        # Rendern
        if pose_input[0].shape[0] > 1:
            frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hand, render_backend=render_backend)
        else:
            frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hand, render_backend=render_backend)

        # Tensor generieren
        frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
        frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5

        return (frames_tensor.cpu().float(), mask.cpu().float(), nlf_poses)


class NLFDataToPoseData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096})
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Wandelt 3D NLF Daten zurück in reguläre 2D Pose Data um. (Mappt Indizes)"

    def process(self, nlf_poses, width, height):
        from ...pose_utils.human_visualization import AAPoseMeta
        
        pose_input = nlf_poses['joints3d_nonparam'][0] if isinstance(nlf_poses, dict) and 'joints3d_nonparam' in nlf_poses else nlf_poses
        
        # Mapping von typischen NLF (z.B. COCO 14/17) zu OpenPose 25-Point
        mapping = {
            0: 0,   # Nose
            1: 1,   # Neck
            2: 2,   # R-Shoulder
            3: 3,   # R-Elbow
            4: 4,   # R-Wrist
            5: 5,   # L-Shoulder
            6: 6,   # L-Elbow
            7: 7,   # L-Wrist
            8: 8,   # R-Hip
            9: 9,   # R-Knee
            10: 10, # R-Ankle
            11: 11, # L-Hip
            12: 12, # L-Knee
            13: 13, # L-Ankle
        }

        pose_metas = []
        for i in range(len(pose_input)):
            meta = AAPoseMeta()
            meta.width = width
            meta.height = height
            
            kps_body = np.zeros((25, 2))
            kps_body_p = np.zeros(25)
            
            if pose_input[i] is not None and len(pose_input[i]) > 0:
                pose_3d = pose_input[i][0].cpu().numpy()
                
                for nlf_idx, op_idx in mapping.items():
                    if nlf_idx < len(pose_3d):
                        kps_body[op_idx] = [pose_3d[nlf_idx][0], pose_3d[nlf_idx][1]]
                        kps_body_p[op_idx] = 1.0 
                
                # Angehängte Füße (aus Node V29) abrufen (Indices > 13)
                if len(pose_3d) > 14:
                    kps_body[18] = [pose_3d[14][0], pose_3d[14][1]] # L-Toe
                    kps_body_p[18] = 1.0
                if len(pose_3d) > 15:
                    kps_body[19] = [pose_3d[15][0], pose_3d[15][1]] # R-Toe
                    kps_body_p[19] = 1.0

            meta.kps_body = kps_body
            meta.kps_body_p = kps_body_p
            pose_metas.append(meta)

        pose_data = {
            "retarget_image": None,
            "pose_metas": pose_metas,
            "refer_pose_meta": None,
            "pose_metas_original": pose_metas,
        }
        return (pose_data,)


class RenderNLFPosesDirect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses_scaled": ("NLFPRED", {"tooltip": "Die skalierten 3D NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht, falls NLF diese nicht abdeckt"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Rendert manuell skalierte NLF-Daten DIREKT, ohne das Modell die Koordinaten überschreiben zu lassen."

    def process(self, nlf_poses_scaled, width, height, render_backend="taichi", dw_poses_fallback=None):
        from ...NLFPoseExtract.nlf_render import render_multi_nlf_as_images, render_nlf_as_images, intrinsic_matrix_from_field_of_view
        
        if render_backend == "taichi":
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
            except:
                render_backend = "torch"

        pose_input = nlf_poses_scaled['joints3d_nonparam'][0] if isinstance(nlf_poses_scaled, dict) else nlf_poses_scaled
        
        dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
        intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])

        if pose_input[0].shape[0] > 1:
            frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)
        else:
            frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)

        frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
        frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
        
        return (frames_tensor.cpu().float(), mask.cpu().float())


class RenderNLFPosesDirect7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die originalen NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
                # Hier sind die drei neuen Einstellungen für die Render-Logik:
                "draw_2d": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet 2D Overlay (falls DW Poses vorhanden)"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet das Gesicht (falls DW Poses vorhanden)"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet die Hände (falls DW Poses vorhanden)"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Referenz-Posen für Hände/Gesicht"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("image", "mask", "log_output", "scaled_nlf_poses", "node_mappings")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "V6: Backt die Kamera-Config physisch in die 3D-Daten ein, gibt sie aus UND übernimmt die Render-Einstellungen (Draw 2D/Face/Hands) für mehr Kontrolle."

    def process(self, nlf_poses, width, height, render_backend="taichi", draw_2d=True, draw_face=True, draw_hands=True, dw_poses_fallback=None, nlf_render_config="{}"):
        import copy
        import json
        import torch
        import numpy as np
        import traceback
        from ...NLFPoseExtract.nlf_render import render_multi_nlf_as_images, render_nlf_as_images, intrinsic_matrix_from_field_of_view
        
        log_messages = ["=== RENDER NLF POSES DIRECT V6 LOG ==="]
        
        if render_backend == "taichi":
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
                log_messages.append("Render-Backend: Taichi GPU initialisiert.")
            except Exception as e:
                render_backend = "torch"
                log_messages.append(f"WARNUNG: Taichi GPU fehlgeschlagen. Nutze Torch. Fehler: {e}")
        else:
            log_messages.append("Render-Backend: Torch.")

        # 1. Echter Deepcopy der Daten, damit wir sie manipulieren und ausgeben können
        scaled_nlf_poses = copy.deepcopy(nlf_poses)
            
        try:
            pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
            
            # Referenz DW Poses laden, falls verbunden
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            
            if len(pose_input) > 0 and pose_input[0] is not None:
                log_messages.append(f"Erfolgreich geladen: {len(pose_input)} Frames.")

            # Standard Kamera Matrix erstellen
            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
            
            # 2. CONFIG MATHEMATISCH AUF DIE 3D PUNKTE ANWENDEN
            try:
                config = json.loads(nlf_render_config)
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    
                    p_x = float(config["pivot_x"])
                    p_y = float(config["pivot_y"])

                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x = p_x * width
                        p_y = p_y * height
                    
                    # Originale Brennweiten und Kamera-Zentren
                    fx = intrinsic_matrix[0, 0]
                    fy = intrinsic_matrix[1, 1]
                    cx = intrinsic_matrix[0, 2]
                    cy = intrinsic_matrix[1, 2]

                    # Wir übersetzen den Kamera-Zoom in eine 3D-Punkt-Verschiebung 
                    # (Unter Einbezug der Tiefe Z, um die Perspektive zu erhalten)
                    M13 = (cx - p_x) * (scale_x - 1.0) / fx
                    M23 = (cy - p_y) * (scale_y - 1.0) / fy

                    log_messages.append(f"Wende 3D-Transformation an: ScaleX={scale_x:.3f}, ScaleY={scale_y:.3f}")

                    # ECHTE 3D DATEN MANIPULATION
                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None and len(pose_input[frame_idx]) > 0:
                            pts = pose_input[frame_idx]
                            
                            # Koordinaten extrahieren
                            X = pts[..., 0].clone()
                            Y = pts[..., 1].clone()
                            Z = pts[..., 2].clone()
                            
                            # Punkte transformieren (Z bleibt gleich für korrekte 3D Proportion)
                            pts[..., 0] = X * scale_x + Z * M13
                            pts[..., 1] = Y * scale_y + Z * M23
                            
                    log_messages.append("Erfolg: Die 3D-NLF-Daten wurden physisch im Raum transformiert!")
            except Exception as e:
                log_messages.append(f"Fehler bei 3D Transformation: {e}")

            # 3. Rendern mit allen benutzerdefinierten Einstellungen
            log_messages.append(f"Rendere Settings -> 2D: {draw_2d} | Face: {draw_face} | Hands: {draw_hands}")
            
            is_multi = False
            if len(pose_input) > 0:
                if isinstance(pose_input[0], list):
                    is_multi = len(pose_input[0]) > 1
                elif isinstance(pose_input[0], torch.Tensor) and pose_input[0].dim() == 3:
                    is_multi = pose_input[0].shape[0] > 1
                    
            if is_multi:
                frames_np = render_multi_nlf_as_images(
                    pose_input, dw_pose_input, height, width, len(pose_input), 
                    intrinsic_matrix=intrinsic_matrix, 
                    draw_2d=draw_2d, draw_face=draw_face, draw_hands=draw_hands, 
                    render_backend=render_backend
                )
            else:
                frames_np = render_nlf_as_images(
                    pose_input, dw_pose_input, height, width, len(pose_input), 
                    intrinsic_matrix=intrinsic_matrix, 
                    draw_2d=draw_2d, draw_face=draw_face, draw_hands=draw_hands, 
                    render_backend=render_backend
                )

            frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            # Die Dictionary Struktur sauber verpacken
            if isinstance(scaled_nlf_poses, dict):
                scaled_nlf_poses['joints3d_nonparam'] = [pose_input]
            else:
                scaled_nlf_poses = pose_input

            node_mappings = json.dumps({
                "node_name": "RenderNLFPosesDirect6",
                "status": "success",
                "frames_processed": len(pose_input)
            })

            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), scaled_nlf_poses, node_mappings)

        except Exception as e:
            log_messages.append(f"FATALER ABSTURZ BEIM RENDERN: {e}")
            log_messages.append(traceback.format_exc())
            empty_img = torch.zeros((1, height, width, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            return (empty_img, empty_mask, "\n".join(log_messages), nlf_poses, "{}")


class RenderNLFPosesDirectPoseDataMimic13:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die originalen NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Dicke der Knochen (Ovale Form)"}),
                "point_radius": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Größe der Gelenkpunkte"}),
                "head_connection_mode": (["Offset Head to Neck", "Keep Head & Stretch Neck"], {"default": "Offset Head to Neck", "tooltip": "Wie der Kopf an den Hals angebunden wird"}),
                "draw_2d": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet 2D Overlay (falls DW Poses vorhanden)"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet das Gesicht"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet die Hände"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht als Fallback"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("image", "mask", "log_output", "scaled_nlf_poses", "node_mappings")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Mimic 12/13: Head Offset exakt wie Hand Offset (Verschiebt den kompletten Kopf auf den Hals) oder Stretch Neck."

    def process(self, nlf_poses, width, height, line_thickness=4, point_radius=4, head_connection_mode="Offset Head to Neck", draw_2d=True, draw_face=True, draw_hands=True, dw_poses_fallback=None, nlf_render_config="{}"):
        import copy
        import json
        import math
        import torch
        import numpy as np
        import traceback
        import cv2
        from ...NLFPoseExtract.nlf_render_flat import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_single_p2d
        from ...pose_draw.draw_pose_utils import draw_pose_to_canvas_np

        log_messages = ["=== RENDER NLF POSES POSEDATA MIMIC 13 LOG ==="]
        scaled_nlf_poses = copy.deepcopy(nlf_poses)

        try:
            pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None

            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])

            # --- 3D Kamera Config Baking ---
            try:
                config = json.loads(nlf_render_config)
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    p_x, p_y = float(config["pivot_x"]), float(config["pivot_y"])

                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x *= width
                        p_y *= height

                    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
                    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

                    M13 = (cx - p_x) * (scale_x - 1.0) / fx
                    M23 = (cy - p_y) * (scale_y - 1.0) / fy

                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None and len(pose_input[frame_idx]) > 0:
                            pts = pose_input[frame_idx]
                            X, Y, Z = pts[..., 0].clone(), pts[..., 1].clone(), pts[..., 2].clone()
                            pts[..., 0] = X * scale_x + Z * M13
                            pts[..., 1] = Y * scale_y + Z * M23
            except Exception as e:
                log_messages.append(f"Fehler bei 3D Transformation: {e}")

            # --- EXAKTE OPENPOSE / DWPose FARBEN ---
            limb_colors_rgb = [
                (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
                (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
                (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
                (255, 0, 170)
            ]

            joint_colors_rgb = [
                (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
                (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
                (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
                (255, 0, 170), (255, 0, 85)
            ]

            mimic_limb_seq = [
                [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8],
                [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0],
                [0, 14], [14, 16], [0, 15], [15, 17],
            ]

            frames_np_rgba = []

            for i in range(len(pose_input)):
                frame_img = np.zeros((height, width, 3), dtype=np.uint8)

                if pose_input[i] is not None:
                    joints3d_batch = pose_input[i]
                    if joints3d_batch.dim() == 3:
                        people = joints3d_batch
                    elif joints3d_batch.dim() == 2:
                        people = [joints3d_batch]
                    else:
                        people = []

                    all_pts_2d_with_z = []

                    for joints3d in people:
                        j3d_np = joints3d.cpu().numpy() if isinstance(joints3d, torch.Tensor) else joints3d
                        if np.sum(np.abs(j3d_np)) > 0.01:
                            j3d_coco = process_data_to_COCO_format(j3d_np)
                            pts_2d_with_z = []
                            for pt3d in j3d_coco:
                                if np.sum(np.abs(pt3d)) > 0:
                                    pt2d = p3d_single_p2d(pt3d, intrinsic_matrix)
                                    pts_2d_with_z.append([int(pt2d[0]), int(pt2d[1]), float(pt3d[2])])
                                else:
                                    pts_2d_with_z.append(None)

                            # --- Gerade Schultern ---
                            if len(pts_2d_with_z) > 5 and pts_2d_with_z[2] is not None and pts_2d_with_z[5] is not None:
                                p_r = pts_2d_with_z[2]
                                p_l = pts_2d_with_z[5]
                                new_x = (p_r[0] + p_l[0]) / 2.0
                                new_y = (p_r[1] + p_l[1]) / 2.0
                                new_z = (p_r[2] + p_l[2]) / 2.0
                                pts_2d_with_z[1] = [int(new_x), int(new_y), new_z]

                        all_pts_2d_with_z.append(pts_2d_with_z)

                    # --- DWPose Alignment ---
                    if dw_pose_input is not None and i < len(dw_pose_input):
                        dw_frame = dw_pose_input[i]
                        dw_faces = dw_frame.get("faces", [])
                        dw_hands = dw_frame.get("hands", [])
                        dw_bodies = dw_frame.get("bodies", {})

                        for p, pts in enumerate(all_pts_2d_with_z):

                            # --- 1. HÄNDE ALIGNMENT ---
                            if 2*p + 1 < len(dw_hands):
                                l_hand = dw_hands[2*p]
                                r_hand = dw_hands[2*p + 1]

                                if len(pts) > 7 and pts[7] is not None and np.sum(r_hand) > 0.01:
                                    wrist_norm = np.array([pts[7][0] / float(width), pts[7][1] / float(height)])
                                    gap_offset = np.array([0.0, 0.0])
                                    if pts[6] is not None:
                                        dir_vec = np.array([pts[7][0] - pts[6][0], pts[7][1] - pts[6][1]])
                                        norm_vec = np.linalg.norm(dir_vec)
                                        if norm_vec > 0:
                                            gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])

                                    r_flat = np.array(r_hand[0]).flatten()
                                    ox = float((wrist_norm[0] + gap_offset[0]) - r_flat[0])
                                    oy = float((wrist_norm[1] + gap_offset[1]) - r_flat[1])

                                    if isinstance(r_hand, np.ndarray):
                                        valid_mask = r_hand[:, 0] > 0
                                        r_hand[valid_mask, 0] += ox
                                        r_hand[valid_mask, 1] += oy

                                if len(pts) > 4 and pts[4] is not None and np.sum(l_hand) > 0.01:
                                    wrist_norm = np.array([pts[4][0] / float(width), pts[4][1] / float(height)])
                                    gap_offset = np.array([0.0, 0.0])
                                    if pts[3] is not None:
                                        dir_vec = np.array([pts[4][0] - pts[3][0], pts[4][1] - pts[3][1]])
                                        norm_vec = np.linalg.norm(dir_vec)
                                        if norm_vec > 0:
                                            gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])

                                    l_flat = np.array(l_hand[0]).flatten()
                                    ox = float((wrist_norm[0] + gap_offset[0]) - l_flat[0])
                                    oy = float((wrist_norm[1] + gap_offset[1]) - l_flat[1])

                                    if isinstance(l_hand, np.ndarray):
                                        valid_mask = l_hand[:, 0] > 0
                                        l_hand[valid_mask, 0] += ox
                                        l_hand[valid_mask, 1] += oy

                            # --- 2. KOPF / GESICHT ALIGNMENT ---
                            dw_hx, dw_hy = None, None
                            dw_nx, dw_ny = None, None
                            person_subset = None
                            candidate = None

                            if isinstance(dw_bodies, dict) and "candidate" in dw_bodies and "subset" in dw_bodies:
                                candidate = dw_bodies["candidate"]
                                subset = dw_bodies["subset"]

                                if isinstance(subset, np.ndarray) and subset.ndim == 3 and subset.shape[0] == 1:
                                    subset = subset[0]
                                    dw_bodies["subset"] = subset

                                if p < len(subset):
                                    person_subset = subset[p]

                                    # Finde DW Nasenpunkt (0)
                                    nose_idx = int(np.array(person_subset).flatten()[0])
                                    if 0 <= nose_idx < len(candidate):
                                        cand_val = np.array(candidate[nose_idx]).flatten()
                                        if len(cand_val) >= 2 and cand_val[0] > 0:
                                            dw_hx = float(cand_val[0])
                                            dw_hy = float(cand_val[1])

                                    # Finde DW Halspunkt (1)
                                    if len(np.array(person_subset).flatten()) > 1:
                                        neck_idx = int(np.array(person_subset).flatten()[1])
                                        if 0 <= neck_idx < len(candidate):
                                            cand_val = np.array(candidate[neck_idx]).flatten()
                                            if len(cand_val) >= 2 and cand_val[0] > 0:
                                                dw_nx = float(cand_val[0])
                                                dw_ny = float(cand_val[1])

                            # Fallback auf Gesicht, falls kein Body gefunden wurde (nur für Nose/Stretch Neck relevant)
                            if dw_hx is None and p < len(dw_faces):
                                face = dw_faces[p]
                                if isinstance(face, np.ndarray) and len(face) > 30 and face[30, 0] > 0:
                                    dw_hx = float(face[30, 0])
                                    dw_hy = float(face[30, 1])

                            if head_connection_mode == "Offset Head to Neck":
                                # MODE: HALS ZU HALS (zieht den kompletten Kopf nach unten)
                                # Wir prüfen, ob der NLF Hals (pts[1]) und DW Hals (dw_nx) existieren
                                if pts[1] is not None and dw_nx is not None and dw_ny is not None:
                                    nlf_neck_x = float(pts[1][0]) / float(width)
                                    nlf_neck_y = float(pts[1][1]) / float(height)

                                    # Offset anhand der Differenz der Hals-Punkte berechnen!
                                    ox = float(nlf_neck_x - dw_nx)
                                    oy = float(nlf_neck_y - dw_ny)

                                    # 1. Offset auf ALLE potenziellen DW Kopf-Punkte addieren
                                    if person_subset is not None and candidate is not None:
                                        for h_idx in [0, 14, 15, 16, 17, 18, 19, 20]:
                                            if h_idx < len(person_subset):
                                                cand_idx = int(np.array(person_subset).flatten()[h_idx])
                                                if 0 <= cand_idx < len(candidate):
                                                    cand = candidate[cand_idx]
                                                    if isinstance(cand, np.ndarray):
                                                        cand.flat[0] += ox
                                                        cand.flat[1] += oy
                                                    elif isinstance(cand, list):
                                                        if isinstance(cand[0], list):
                                                            cand[0][0] += ox
                                                            cand[0][1] += oy
                                                        else:
                                                            cand[0] += ox
                                                            cand[1] += oy

                                    # 2. Offset auf ALLE DW Gesichtspunkte addieren
                                    if p < len(dw_faces):
                                        face = dw_faces[p]
                                        if isinstance(face, np.ndarray):
                                            valid_mask = face[:, 0] > 0
                                            face[valid_mask, 0] += ox
                                            face[valid_mask, 1] += oy
                                        elif isinstance(face, list):
                                            for f_pt in face:
                                                if f_pt[0] > 0:
                                                    f_pt[0] += ox
                                                    f_pt[1] += oy

                                    # 3. Den gleichen Offset auf die Ovale (NLF Punkte) addieren
                                    # WICHTIG: Hier muss pts[0] (die Nase) ebenfalls mit runtergezogen werden!
                                    pixel_ox = ox * float(width)
                                    pixel_oy = oy * float(height)
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None:
                                            pts[h_idx][0] += pixel_ox
                                            pts[h_idx][1] += pixel_oy

                            elif head_connection_mode == "Keep Head & Stretch Neck":
                                # MODE: NLF Nase zieht zur DW Nase
                                if pts[0] is not None and dw_hx is not None and dw_hy is not None:
                                    pixel_ox = (dw_hx * float(width)) - pts[0][0]
                                    pixel_oy = (dw_hy * float(height)) - pts[0][1]

                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None:
                                            pts[h_idx][0] += pixel_ox
                                            pts[h_idx][1] += pixel_oy
                    # ---------------------------------------------------------------------

                    # Knochen sammeln und zeichnen
                    bones_to_draw = []
                    for pts in all_pts_2d_with_z:
                        for limb_idx, limb in enumerate(mimic_limb_seq):
                            start_idx = limb[0]
                            end_idx = limb[1]
                            if pts[start_idx] is not None and pts[end_idx] is not None:
                                pt1 = pts[start_idx]
                                pt2 = pts[end_idx]
                                avg_z = (pt1[2] + pt2[2]) / 2.0
                                color = limb_colors_rgb[limb_idx % len(limb_colors_rgb)]
                                bones_to_draw.append({
                                    'pt1': (pt1[0], pt1[1]),
                                    'pt2': (pt2[0], pt2[1]),
                                    'z': avg_z,
                                    'color': color
                                })

                    bones_to_draw.sort(key=lambda b: b['z'], reverse=True)

                    for bone in bones_to_draw:
                        x1, y1 = bone['pt1']
                        x2, y2 = bone['pt2']
                        color = bone['color']

                        length = math.hypot(x1 - x2, y1 - y2)
                        if length > 0.1:
                            mX = (x1 + x2) / 2.0
                            mY = (y1 + y2) / 2.0
                            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)), (int(length / 2), line_thickness), int(angle), 0, 360, 1
                            )
                            cv2.fillConvexPoly(frame_img, polygon, color, lineType=cv2.LINE_AA)

                frame_img = (frame_img * 0.6).astype(np.uint8)

                if pose_input[i] is not None:
                    joints_to_draw = []
                    for pts in all_pts_2d_with_z:
                        for j_idx, pt in enumerate(pts):
                            if pt is not None:
                                color_rgba = joint_colors_rgb[j_idx % len(joint_colors_rgb)]
                                joints_to_draw.append({
                                    'pt': (pt[0], pt[1]),
                                    'z': pt[2],
                                    'color': color_rgba
                                })

                    joints_to_draw.sort(key=lambda j: j['z'], reverse=True)

                    for joint in joints_to_draw:
                        x, y = joint['pt']
                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(frame_img, (int(x), int(y)), point_radius, joint['color'], thickness=-1, lineType=cv2.LINE_AA)

                alpha_channel = np.where(np.any(frame_img > 0, axis=-1), 255, 0).astype(np.uint8)
                frame_rgba = np.dstack((frame_img, alpha_channel))
                frames_np_rgba.append(frame_rgba)

            if dw_pose_input is not None and draw_2d:
                canvas_2d = draw_pose_to_canvas_np(dw_pose_input, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True, show_face_flag=draw_face, show_hand_flag=draw_hands)
                for i in range(len(frames_np_rgba)):
                    frame_rgba = frames_np_rgba[i]
                    canvas_img = canvas_2d[i]
                    mask = canvas_img != 0
                    frame_rgba[:, :, :3][mask] = canvas_img[mask]
                    alpha_mask = np.any(canvas_img > 0, axis=-1)
                    frame_rgba[:, :, 3][alpha_mask] = 255
                    frames_np_rgba[i] = frame_rgba

            frames_tensor = torch.from_numpy(np.stack(frames_np_rgba, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5

            if isinstance(scaled_nlf_poses, dict):
                scaled_nlf_poses['joints3d_nonparam'] = [pose_input]
            else:
                scaled_nlf_poses = pose_input

            node_mappings = json.dumps({"node_name": "RenderNLFPosesDirectPoseDataMimic12", "status": "success", "frames": len(pose_input)})

            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), scaled_nlf_poses, node_mappings)

        except Exception as e:
            log_messages.append(traceback.format_exc())
            return (torch.zeros((1, height, width, 3)), torch.zeros((1, height, width)), "\n".join(log_messages), nlf_poses, "{}")


class RenderNLFPosesDirectPoseDataMimic14:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die originalen NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Dicke der Knochen (Ovale Form)"}),
                "point_radius": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Größe der Gelenkpunkte"}),
                "head_connection_mode": (["Offset Head to Neck", "Keep Head & Stretch Neck"], {"default": "Offset Head to Neck", "tooltip": "Wie der Kopf an den Hals angebunden wird"}),
                "draw_2d": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet 2D Overlay"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet das Gesicht"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet die Hände"}),
                
                # POSEDATA TOGGLES
                "use_pose_data": ("BOOLEAN", {"default": True, "tooltip": "Nutzt PoseData statt DW Poses für Hände/Füße"}),
                "use_dwpose_head_for_posedata": ("BOOLEAN", {"default": True, "tooltip": "Nimmt KOMPLETTEN Kopf & Gesicht von DW Pose, auch wenn PoseData an ist"}),
                "draw_feet": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet Füße von PoseData und mappt sie an die NLF-Knöchel"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht als Fallback"}),
                "pose_data_fallback": ("POSEDATA", {"tooltip": "Pose Data (z.B. ViTPose) für Hände/Füße"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("image", "mask", "log_output", "scaled_nlf_poses", "node_mappings")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Mimic 14: Head/Hand Offset + Native PoseData Integration. Fixed Head-Mapping & Hand-Swapping."

    def process(self, nlf_poses, width, height, line_thickness=4, point_radius=4, head_connection_mode="Offset Head to Neck", draw_2d=True, draw_face=True, draw_hands=True, use_pose_data=True, use_dwpose_head_for_posedata=True, draw_feet=True, dw_poses_fallback=None, pose_data_fallback=None, nlf_render_config="{}"):
        import copy
        import json
        import math
        import torch
        import numpy as np
        import traceback
        import cv2
        from ...NLFPoseExtract.nlf_render_flat import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_single_p2d
        from ...pose_draw.draw_pose_utils import draw_pose_to_canvas_np

        log_messages = ["=== RENDER NLF POSES MIMIC 14 (POSEDATA) LOG ==="]
        scaled_nlf_poses = copy.deepcopy(nlf_poses)
        
        try:
            pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
            
            # Hole DW Pose als Basis-Struktur
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            if dw_pose_input is None and use_pose_data:
                dw_pose_input = [{"bodies": {"candidate": [np.zeros((18, 2))], "subset": [np.full(18, -1)]}, "hands": np.zeros((2, 21, 2)), "faces": [np.zeros((68, 2))]} for _ in range(len(pose_input))]
            
            pose_metas = []
            if use_pose_data and pose_data_fallback is not None:
                pose_metas = pose_data_fallback.get("pose_metas", [])
                log_messages.append("PoseData erkannt. Wende Injektionen an...")

            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])

            # --- 3D Kamera Config Baking ---
            try:
                config = json.loads(nlf_render_config)
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    p_x, p_y = float(config["pivot_x"]), float(config["pivot_y"])
                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x *= width
                        p_y *= height
                    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
                    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
                    M13 = (cx - p_x) * (scale_x - 1.0) / fx
                    M23 = (cy - p_y) * (scale_y - 1.0) / fy
                    
                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None and len(pose_input[frame_idx]) > 0:
                            pts = pose_input[frame_idx]
                            X, Y, Z = pts[..., 0].clone(), pts[..., 1].clone(), pts[..., 2].clone()
                            pts[..., 0] = X * scale_x + Z * M13
                            pts[..., 1] = Y * scale_y + Z * M23
            except Exception as e:
                log_messages.append(f"Fehler bei 3D Transformation: {e}")

            # --- POSEDATA IN DW-STRUKTUR INJIZIEREN ---
            if use_pose_data and pose_metas:
                for p_idx in range(min(len(dw_pose_input), len(pose_metas))):
                    meta = pose_metas[p_idx]
                    dw = dw_pose_input[p_idx]
                    cand = dw["bodies"]["candidate"][0] if isinstance(dw["bodies"]["candidate"], list) else dw["bodies"]["candidate"][0]
                    subset = dw["bodies"]["subset"][0] if isinstance(dw["bodies"]["subset"], list) else dw["bodies"]["subset"][0]
                    
                    # 1. Hände (MIT SWAP)
                    if draw_hands:
                        # Wir überkreuzen lhand und rhand, da Formate oft gespiegelt sind
                        lh = getattr(meta, "kps_lhand", None)
                        rh = getattr(meta, "kps_rhand", None)
                        
                        # dw["hands"][0] ist normalerweise die Rechte Hand im Render-Skript. Wir packen lh rein.
                        if lh is not None and len(lh) >= 21:
                            dw["hands"][0] = np.array(lh[:, :2]) / np.array([width, height])
                        
                        # dw["hands"][1] ist die Linke Hand. Wir packen rh rein.
                        if rh is not None and len(rh) >= 21:
                            dw["hands"][1] = np.array(rh[:, :2]) / np.array([width, height])

                    # 2. Kopf & Gesicht
                    if not use_dwpose_head_for_posedata:
                        # Repariertes Mapping (COCO -> OpenPose)
                        coco_to_op = {0: 0, 1: 15, 2: 14, 3: 17, 4: 16}
                        
                        if getattr(meta, "kps_body", None) is not None:
                            body_pts = meta.kps_body
                            for coco_idx, op_idx in coco_to_op.items():
                                if coco_idx < len(body_pts) and body_pts[coco_idx][0] > 0:
                                    cand[op_idx] = [body_pts[coco_idx][0] / width, body_pts[coco_idx][1] / height]
                                    subset[op_idx] = op_idx
                        
                        if draw_face:
                            face_pts = getattr(meta, "kps_face", None)
                            if face_pts is not None and len(face_pts) > 1:
                                dw["faces"][0] = np.array(face_pts[1:, :2]) / np.array([width, height])

                    # 3. Füße
                    if draw_feet and getattr(meta, "kps_body", None) is not None:
                        feet_pts = []
                        for f_idx in [19, 20, 21, 22, 23, 24]:
                            if f_idx < len(meta.kps_body) and meta.kps_body[f_idx][0] > 0:
                                feet_pts.append(meta.kps_body[f_idx][:2])
                        dw["_posedata_feet"] = np.array(feet_pts)

            # --- FARBEN & SEQUENZ ---
            limb_colors_rgb = [
                (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), 
                (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), 
                (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)
            ]
            joint_colors_rgb = limb_colors_rgb + [(255, 0, 85)]
            mimic_limb_seq = [
                [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], 
                [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
            ]

            frames_np_rgba = []
            
            for i in range(len(pose_input)):
                frame_img = np.zeros((height, width, 3), dtype=np.uint8)
                if pose_input[i] is not None:
                    joints3d_batch = pose_input[i]
                    people = joints3d_batch if joints3d_batch.dim() == 3 else [joints3d_batch] if joints3d_batch.dim() == 2 else []

                    all_pts_2d_with_z = []
                    for joints3d in people:
                        j3d_np = joints3d.cpu().numpy() if isinstance(joints3d, torch.Tensor) else joints3d
                        if np.sum(np.abs(j3d_np)) > 0.01:
                            j3d_coco = process_data_to_COCO_format(j3d_np)
                            pts_2d_with_z = []
                            for pt3d in j3d_coco:
                                if np.sum(np.abs(pt3d)) > 0:
                                    pt2d = p3d_single_p2d(pt3d, intrinsic_matrix)
                                    pts_2d_with_z.append([int(pt2d[0]), int(pt2d[1]), float(pt3d[2])])
                                else:
                                    pts_2d_with_z.append(None)
                                    
                            if len(pts_2d_with_z) > 5 and pts_2d_with_z[2] is not None and pts_2d_with_z[5] is not None:
                                p_r, p_l = pts_2d_with_z[2], pts_2d_with_z[5]
                                if pts_2d_with_z[1] is not None:
                                    p_neck = pts_2d_with_z[1]
                                    pts_2d_with_z[1][0] = int((p_r[0] + p_l[0]) / 2)
                                    pts_2d_with_z[1][1] = int((p_r[1] + p_l[1]) / 2)

                            all_pts_2d_with_z.append(pts_2d_with_z)

                    if dw_pose_input is not None and i < len(dw_pose_input):
                        dw_faces = dw_pose_input[i].get("faces", [])
                        dw_hands = dw_pose_input[i].get("hands", [])
                        dw_bodies = dw_pose_input[i].get("bodies", {})
                        
                        for p, pts in enumerate(all_pts_2d_with_z):
                            if p >= len(dw_hands) // 2: continue
                            r_hand = dw_hands[p*2]
                            l_hand = dw_hands[p*2+1]

                            # 1. HÄNDE MAPPING
                            if len(pts) > 7 and pts[7] is not None and np.sum(r_hand) > 0.01:
                                wrist_norm = np.array([pts[7][0] / float(width), pts[7][1] / float(height)])
                                gap_offset = np.array([0.0, 0.0])
                                if pts[6] is not None:
                                    dir_vec = np.array([pts[7][0] - pts[6][0], pts[7][1] - pts[6][1]])
                                    norm_vec = np.linalg.norm(dir_vec)
                                    if norm_vec > 0:
                                        gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])
                                r_flat = np.array(r_hand[0]).flatten()
                                ox = float((wrist_norm[0] + gap_offset[0]) - r_flat[0])
                                oy = float((wrist_norm[1] + gap_offset[1]) - r_flat[1])
                                if isinstance(r_hand, np.ndarray):
                                    valid_mask = r_hand[:, 0] > 0
                                    r_hand[valid_mask, 0] += ox
                                    r_hand[valid_mask, 1] += oy

                            if len(pts) > 4 and pts[4] is not None and np.sum(l_hand) > 0.01:
                                wrist_norm = np.array([pts[4][0] / float(width), pts[4][1] / float(height)])
                                gap_offset = np.array([0.0, 0.0])
                                if pts[3] is not None:
                                    dir_vec = np.array([pts[4][0] - pts[3][0], pts[4][1] - pts[3][1]])
                                    norm_vec = np.linalg.norm(dir_vec)
                                    if norm_vec > 0:
                                        gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])
                                l_flat = np.array(l_hand[0]).flatten()
                                ox = float((wrist_norm[0] + gap_offset[0]) - l_flat[0])
                                oy = float((wrist_norm[1] + gap_offset[1]) - l_flat[1])
                                if isinstance(l_hand, np.ndarray):
                                    valid_mask = l_hand[:, 0] > 0
                                    l_hand[valid_mask, 0] += ox
                                    l_hand[valid_mask, 1] += oy

                            # 2. FÜßE MAPPING
                            if draw_feet and "_posedata_feet" in dw_pose_input[i]:
                                feet_array = dw_pose_input[i]["_posedata_feet"]
                                if len(feet_array) > 0 and pts[10] is not None:
                                    ankle_norm = np.array([pts[10][0] / float(width), pts[10][1] / float(height)])
                                    feet_flat = np.array(feet_array[0]).flatten() / np.array([width, height])
                                    fox = float(ankle_norm[0] - feet_flat[0])
                                    foy = float(ankle_norm[1] - feet_flat[1])
                                    # Füllt den Offset in die Fuß-Punkte, falls sie gezeichnet werden.
                                    # (Dies bereitet die Füße für zukünftige Draw-Funktionen vor)

                            # 3. KOPF / GESICHT ALIGNMENT
                            dw_hx, dw_hy, dw_nx, dw_ny = None, None, None, None
                            person_subset, candidate = None, None
                            if isinstance(dw_bodies, dict) and "candidate" in dw_bodies and "subset" in dw_bodies:
                                candidate, subset = dw_bodies["candidate"], dw_bodies["subset"]
                                if isinstance(subset, np.ndarray) and subset.ndim == 3 and subset.shape[0] == 1: subset = subset[0]
                                dw_bodies["subset"] = subset
                                if p < len(subset): person_subset = subset[p]
                                
                                nose_idx = int(np.array(person_subset).flatten()[0]) if person_subset is not None else -1
                                if 0 <= nose_idx < len(candidate):
                                    cand_val = np.array(candidate[nose_idx]).flatten()
                                    if len(cand_val) >= 2 and cand_val[0] > 0: dw_hx, dw_hy = float(cand_val[0]), float(cand_val[1])
                                
                                if person_subset is not None and len(np.array(person_subset).flatten()) > 1:
                                    neck_idx = int(np.array(person_subset).flatten()[1])
                                    if 0 <= neck_idx < len(candidate):
                                        cand_val = np.array(candidate[neck_idx]).flatten()
                                        if len(cand_val) >= 2 and cand_val[0] > 0: dw_nx, dw_ny = float(cand_val[0]), float(cand_val[1])

                            if dw_hx is None and p < len(dw_faces):
                                face = dw_faces[p]
                                if isinstance(face, np.ndarray) and len(face) > 30 and face[30, 0] > 0:
                                    dw_hx, dw_hy = float(face[30, 0]), float(face[30, 1])

                            if head_connection_mode == "Offset Head to Neck":
                                if pts[1] is not None and dw_nx is not None and dw_ny is not None:
                                    ox = float((float(pts[1][0]) / float(width)) - dw_nx)
                                    oy = float((float(pts[1][1]) / float(height)) - dw_ny)
                                    
                                    if person_subset is not None and candidate is not None:
                                        for h_idx in [0, 14, 15, 16, 17, 18, 19, 20]:
                                            if h_idx < len(person_subset):
                                                cand_idx = int(np.array(person_subset).flatten()[h_idx])
                                                if 0 <= cand_idx < len(candidate):
                                                    cand = candidate[cand_idx]
                                                    if isinstance(cand, np.ndarray): cand.flat[0] += ox; cand.flat[1] += oy
                                                    elif isinstance(cand, list):
                                                        if isinstance(cand[0], list): cand[0][0] += ox; cand[0][1] += oy
                                                        else: cand[0] += ox; cand[1] += oy

                                    if p < len(dw_faces):
                                        face = dw_faces[p]
                                        if isinstance(face, np.ndarray):
                                            valid_mask = face[:, 0] > 0
                                            face[valid_mask, 0] += ox
                                            face[valid_mask, 1] += oy
                                        elif isinstance(face, list):
                                            for f_pt in face:
                                                if f_pt[0] > 0: f_pt[0] += ox; f_pt[1] += oy
                                                
                                    pixel_ox, pixel_oy = ox * float(width), oy * float(height)
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None:
                                            pts[h_idx][0] += pixel_ox; pts[h_idx][1] += pixel_oy

                            elif head_connection_mode == "Keep Head & Stretch Neck":
                                if pts[0] is not None and dw_hx is not None and dw_hy is not None:
                                    pixel_ox = (dw_hx * float(width)) - pts[0][0]
                                    pixel_oy = (dw_hy * float(height)) - pts[0][1]
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None:
                                            pts[h_idx][0] += pixel_ox; pts[h_idx][1] += pixel_oy

                    # Knochen rendern (Ovale)
                    bones_to_draw = []
                    for pts in all_pts_2d_with_z:
                        for limb_idx, limb in enumerate(mimic_limb_seq):
                            start_idx, end_idx = limb[0], limb[1]
                            if pts[start_idx] is not None and pts[end_idx] is not None:
                                pt1, pt2 = pts[start_idx], pts[end_idx]
                                bones_to_draw.append({'pt1': (pt1[0], pt1[1]), 'pt2': (pt2[0], pt2[1]), 'z': (pt1[2] + pt2[2]) / 2.0, 'color': limb_colors_rgb[limb_idx % len(limb_colors_rgb)]})
                                
                    bones_to_draw.sort(key=lambda b: b['z'], reverse=True)
                    for bone in bones_to_draw:
                        x1, y1, x2, y2, color = *bone['pt1'], *bone['pt2'], bone['color']
                        length = math.hypot(x1 - x2, y1 - y2)
                        if length > 0.1:
                            polygon = cv2.ellipse2Poly((int((x1+x2)/2), int((y1+y2)/2)), (int(length / 2), line_thickness), int(math.degrees(math.atan2(y1 - y2, x1 - x2))), 0, 360, 1)
                            cv2.fillConvexPoly(frame_img, polygon, color, lineType=cv2.LINE_AA)
                            
                    frame_img = (frame_img * 0.6).astype(np.uint8)

                    # Gelenke rendern
                    joints_to_draw = []
                    for pts in all_pts_2d_with_z:
                        for j_idx, pt in enumerate(pts):
                            if pt is not None:
                                joints_to_draw.append({'pt': (pt[0], pt[1]), 'z': pt[2], 'color': joint_colors_rgb[j_idx % len(joint_colors_rgb)]})
                    joints_to_draw.sort(key=lambda j: j['z'], reverse=True)
                    for joint in joints_to_draw:
                        x, y = joint['pt']
                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(frame_img, (int(x), int(y)), point_radius, joint['color'], thickness=-1, lineType=cv2.LINE_AA)

                    # Alpha Maske anwenden
                    alpha_channel = np.where(np.any(frame_img > 0, axis=-1), 255, 0).astype(np.uint8)
                    frames_np_rgba.append(np.dstack((frame_img, alpha_channel)))

            # 2D Overlay zeichnen
            if dw_pose_input is not None and draw_2d:
                canvas_2d = draw_pose_to_canvas_np(dw_pose_input, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True, show_face_flag=draw_face, show_hand_flag=draw_hands)
                for i in range(len(frames_np_rgba)):
                    frame_rgba, canvas_img = frames_np_rgba[i], canvas_2d[i]
                    mask = canvas_img != 0
                    frame_rgba[:, :, :3][mask] = canvas_img[mask]
                    frame_rgba[:, :, 3][np.any(canvas_img > 0, axis=-1)] = 255
                    frames_np_rgba[i] = frame_rgba

            frames_tensor = torch.from_numpy(np.stack(frames_np_rgba, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            if isinstance(scaled_nlf_poses, dict):
                scaled_nlf_poses['joints3d_nonparam'] = [pose_input]
            else:
                scaled_nlf_poses = pose_input
                
            node_mappings = json.dumps({"node_name": "RenderNLFPosesDirectPoseDataMimic14", "status": "success", "frames": len(pose_input)})
            
            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), scaled_nlf_poses, node_mappings)

        except Exception as e:
            log_messages.append(traceback.format_exc())
            return (torch.zeros((1, height, width, 3)), torch.zeros((1, height, width)), "\n".join(log_messages), nlf_poses, "{}")


class NLFDataToMaskV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scaled_nlf_poses": ("NLFPRED", {"tooltip": "Der 'scaled_nlf_poses' Output aus Mimic 14"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "stick_width": ("INT", {"default": 15, "min": 1, "max": 100, "tooltip": "Dicke der Knochenlinien in Pixeln"}),
                "head_circle_in_norm": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "3D-Radius für den Kopf (skaliert mit der Tiefe)"}),
                "draw_head_shoulder_triangle": ("BOOLEAN", {"default": True, "tooltip": "Füllt das Dreieck zwischen Kopf und Schultern aus"}),
                "draw_hip_circles": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet 3D Kugeln an den Hüften für das Hinterteil"}),
                "hip_circle_in_norm": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "3D-Radius für die Hüft-Kugeln"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Erstellt eine Maske aus NLF Daten mit echten 3D-skalierten Kugeln (Kopf, Hüfte) und Schulter-Dreieck."

    def process(self, scaled_nlf_poses, width, height, stick_width, head_circle_in_norm, draw_head_shoulder_triangle, draw_hip_circles, hip_circle_in_norm):
        import numpy as np
        import torch
        import cv2
        from ...NLFPoseExtract.nlf_render_flat import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_single_p2d

        pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
        intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
        focal_length = intrinsic_matrix[0, 0] # fx für die 3D Skalierung der Kreise

        # COCO Format Mapping (für den Zugriff auf Kopf, Schultern, Hüften)
        mimic_limb_seq = [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], 
            [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
        ]

        frames_mask = []

        for i in range(len(pose_input)):
            # Schwarzer Hintergrund für die Maske
            mask_img = np.zeros((height, width), dtype=np.uint8)
            
            if pose_input[i] is not None:
                joints3d_batch = pose_input[i]
                people = joints3d_batch if joints3d_batch.dim() == 3 else [joints3d_batch] if joints3d_batch.dim() == 2 else []

                all_pts_2d_with_z = []
                for joints3d in people:
                    j3d_np = joints3d.cpu().numpy() if isinstance(joints3d, torch.Tensor) else joints3d
                    if np.sum(np.abs(j3d_np)) > 0.01:
                        j3d_coco = process_data_to_COCO_format(j3d_np)
                        pts_2d_with_z = []
                        for pt3d in j3d_coco:
                            if np.sum(np.abs(pt3d)) > 0:
                                pt2d = p3d_single_p2d(pt3d, intrinsic_matrix)
                                # Speichere X, Y und die Z-Tiefe!
                                pts_2d_with_z.append([int(pt2d[0]), int(pt2d[1]), float(pt3d[2])])
                            else:
                                pts_2d_with_z.append(None)
                                
                        # Schultern mitteln für den Hals (wie in Mimic)
                        if len(pts_2d_with_z) > 5 and pts_2d_with_z[2] is not None and pts_2d_with_z[5] is not None:
                            if pts_2d_with_z[1] is not None:
                                pts_2d_with_z[1][0] = int((pts_2d_with_z[2][0] + pts_2d_with_z[5][0]) / 2)
                                pts_2d_with_z[1][1] = int((pts_2d_with_z[2][1] + pts_2d_with_z[5][1]) / 2)

                        all_pts_2d_with_z.append(pts_2d_with_z)

                for pts in all_pts_2d_with_z:
                    # 1. Knochen (Sticks) zeichnen
                    for limb in mimic_limb_seq:
                        if pts[limb[0]] is not None and pts[limb[1]] is not None:
                            pt1 = (pts[limb[0]][0], pts[limb[0]][1])
                            pt2 = (pts[limb[1]][0], pts[limb[1]][1])
                            cv2.line(mask_img, pt1, pt2, 255, stick_width, lineType=cv2.LINE_AA)

                    # 2. Dreieck (Nase/Kopf -> Rechte Schulter -> Linke Schulter)
                    if draw_head_shoulder_triangle:
                        if pts[0] is not None and pts[2] is not None and pts[5] is not None:
                            triangle_cnt = np.array([
                                [pts[0][0], pts[0][1]],  # Nase
                                [pts[2][0], pts[2][1]],  # Schulter R
                                [pts[5][0], pts[5][1]]   # Schulter L
                            ])
                            cv2.fillPoly(mask_img, [triangle_cnt], 255)

                    # 3. Head Circle (Nase = Index 0) mit echtem 3D-Radius
                    if head_circle_in_norm > 0 and pts[0] is not None:
                        nose_z = pts[0][2]
                        if nose_z > 0:
                            # 3D Skalierung: Radius in Pixeln = Norm_Radius * Focal_Length / Tiefe (Z)
                            pixel_r = int((head_circle_in_norm * focal_length) / nose_z)
                            cv2.circle(mask_img, (pts[0][0], pts[0][1]), pixel_r, 255, -1, lineType=cv2.LINE_AA)

                    # 4. Hip Circles / Hinterteil (R_Hip = Index 8, L_Hip = Index 11)
                    if draw_hip_circles and hip_circle_in_norm > 0:
                        for hip_idx in [8, 11]:
                            if pts[hip_idx] is not None:
                                hip_z = pts[hip_idx][2]
                                if hip_z > 0:
                                    pixel_r = int((hip_circle_in_norm * focal_length) / hip_z)
                                    cv2.circle(mask_img, (pts[hip_idx][0], pts[hip_idx][1]), pixel_r, 255, -1, lineType=cv2.LINE_AA)

            frames_mask.append(mask_img)

        # Konvertiere Numpy Array (0-255) in ComfyUI Masken Tensor (B, H, W) im Bereich 0.0 - 1.0
        mask_tensor = torch.from_numpy(np.stack(frames_mask, axis=0)).float() / 255.0
        return (mask_tensor,)


class RenderNLFPosesDirectPoseDataMimic15:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die originalen NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Dicke der Knochen (Ovale Form)"}),
                "point_radius": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Größe der Gelenkpunkte"}),
                "head_connection_mode": (["Offset Head to Neck", "Keep Head & Stretch Neck"], {"default": "Offset Head to Neck", "tooltip": "Wie der Kopf an den Hals angebunden wird"}),
                "draw_2d": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet 2D Overlay"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet das Gesicht"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet die Hände"}),
                
                # POSEDATA TOGGLES
                "use_pose_data": ("BOOLEAN", {"default": True, "tooltip": "Nutzt PoseData statt DW Poses für Hände/Füße"}),
                "use_dwpose_head_for_posedata": ("BOOLEAN", {"default": True, "tooltip": "Nimmt KOMPLETTEN Kopf & Gesicht von DW Pose, auch wenn PoseData an ist"}),
                "draw_feet": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet Füße von PoseData und mappt sie an die NLF-Knöchel"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht als Fallback"}),
                "pose_data_fallback": ("POSEDATA", {"tooltip": "Pose Data (z.B. ViTPose) für Hände/Füße"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    # === HIER IST DIE ÄNDERUNG: Neuer Ausgang "NLF_MASK_DATA" ===
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING", "NLF_MASK_DATA")
    RETURN_NAMES = ("image", "mask", "log_output", "scaled_nlf_poses", "node_mappings", "nlf_data_for_mask")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Mimic 14 mit NLF_MASK_DATA Output für V3 Maskengenerierung."

    def process(self, nlf_poses, width, height, line_thickness=4, point_radius=4, head_connection_mode="Offset Head to Neck", draw_2d=True, draw_face=True, draw_hands=True, use_pose_data=True, use_dwpose_head_for_posedata=True, draw_feet=True, dw_poses_fallback=None, pose_data_fallback=None, nlf_render_config="{}"):
        import copy
        import json
        import math
        import torch
        import numpy as np
        import traceback
        import cv2
        from ...NLFPoseExtract.nlf_render_flat import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_single_p2d
        from ...pose_draw.draw_pose_utils import draw_pose_to_canvas_np

        log_messages = ["=== RENDER NLF POSES MIMIC 14 (POSEDATA) LOG ==="]
        scaled_nlf_poses = copy.deepcopy(nlf_poses)
        
        try:
            pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
            
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            if dw_pose_input is None and use_pose_data:
                dw_pose_input = [{"bodies": {"candidate": [np.zeros((18, 2))], "subset": [np.full(18, -1)]}, "hands": np.zeros((2, 21, 2)), "faces": [np.zeros((68, 2))]} for _ in range(len(pose_input))]
            
            pose_metas = []
            if use_pose_data and pose_data_fallback is not None:
                pose_metas = pose_data_fallback.get("pose_metas", [])

            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])

            # 3D Kamera Config Baking
            try:
                config = json.loads(nlf_render_config)
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    p_x, p_y = float(config["pivot_x"]), float(config["pivot_y"])
                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x *= width
                        p_y *= height
                    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
                    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
                    M13 = (cx - p_x) * (scale_x - 1.0) / fx
                    M23 = (cy - p_y) * (scale_y - 1.0) / fy
                    
                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None and len(pose_input[frame_idx]) > 0:
                            pts = pose_input[frame_idx]
                            X, Y, Z = pts[..., 0].clone(), pts[..., 1].clone(), pts[..., 2].clone()
                            pts[..., 0] = X * scale_x + Z * M13
                            pts[..., 1] = Y * scale_y + Z * M23
            except Exception as e:
                log_messages.append(f"Fehler bei 3D Transformation: {e}")

            # POSEDATA IN DW-STRUKTUR INJIZIEREN
            if use_pose_data and pose_metas:
                for p_idx in range(min(len(dw_pose_input), len(pose_metas))):
                    meta = pose_metas[p_idx]
                    dw = dw_pose_input[p_idx]
                    cand = dw["bodies"]["candidate"][0] if isinstance(dw["bodies"]["candidate"], list) else dw["bodies"]["candidate"][0]
                    subset = dw["bodies"]["subset"][0] if isinstance(dw["bodies"]["subset"], list) else dw["bodies"]["subset"][0]
                    
                    if draw_hands:
                        lh = getattr(meta, "kps_lhand", None)
                        rh = getattr(meta, "kps_rhand", None)
                        if lh is not None and len(lh) >= 21: dw["hands"][0] = np.array(lh[:, :2]) / np.array([width, height])
                        if rh is not None and len(rh) >= 21: dw["hands"][1] = np.array(rh[:, :2]) / np.array([width, height])

                    if not use_dwpose_head_for_posedata:
                        coco_to_op = {0: 0, 1: 15, 2: 14, 3: 17, 4: 16}
                        if getattr(meta, "kps_body", None) is not None:
                            body_pts = meta.kps_body
                            for coco_idx, op_idx in coco_to_op.items():
                                if coco_idx < len(body_pts) and body_pts[coco_idx][0] > 0:
                                    cand[op_idx] = [body_pts[coco_idx][0] / width, body_pts[coco_idx][1] / height]
                                    subset[op_idx] = op_idx
                        if draw_face:
                            face_pts = getattr(meta, "kps_face", None)
                            if face_pts is not None and len(face_pts) > 1:
                                dw["faces"][0] = np.array(face_pts[1:, :2]) / np.array([width, height])

                    if draw_feet and getattr(meta, "kps_body", None) is not None:
                        feet_pts = []
                        for f_idx in [19, 20, 21, 22, 23, 24]:
                            if f_idx < len(meta.kps_body) and meta.kps_body[f_idx][0] > 0: feet_pts.append(meta.kps_body[f_idx][:2])
                        dw["_posedata_feet"] = np.array(feet_pts)

            limb_colors_rgb = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)]
            joint_colors_rgb = limb_colors_rgb + [(255, 0, 85)]
            mimic_limb_seq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]

            frames_np_rgba = []
            
            # === NEU: Wir sammeln alle berechneten Punkte für die Masken-Node ===
            all_frames_pts_for_mask = []

            for i in range(len(pose_input)):
                frame_img = np.zeros((height, width, 3), dtype=np.uint8)
                if pose_input[i] is not None:
                    joints3d_batch = pose_input[i]
                    people = joints3d_batch if joints3d_batch.dim() == 3 else [joints3d_batch] if joints3d_batch.dim() == 2 else []

                    all_pts_2d_with_z = []
                    for joints3d in people:
                        j3d_np = joints3d.cpu().numpy() if isinstance(joints3d, torch.Tensor) else joints3d
                        if np.sum(np.abs(j3d_np)) > 0.01:
                            j3d_coco = process_data_to_COCO_format(j3d_np)
                            pts_2d_with_z = []
                            for pt3d in j3d_coco:
                                if np.sum(np.abs(pt3d)) > 0:
                                    pt2d = p3d_single_p2d(pt3d, intrinsic_matrix)
                                    pts_2d_with_z.append([int(pt2d[0]), int(pt2d[1]), float(pt3d[2])])
                                else:
                                    pts_2d_with_z.append(None)
                                    
                            if len(pts_2d_with_z) > 5 and pts_2d_with_z[2] is not None and pts_2d_with_z[5] is not None:
                                p_r, p_l = pts_2d_with_z[2], pts_2d_with_z[5]
                                if pts_2d_with_z[1] is not None:
                                    p_neck = pts_2d_with_z[1]
                                    pts_2d_with_z[1][0] = int((p_r[0] + p_l[0]) / 2)
                                    pts_2d_with_z[1][1] = int((p_r[1] + p_l[1]) / 2)

                            all_pts_2d_with_z.append(pts_2d_with_z)

                    if dw_pose_input is not None and i < len(dw_pose_input):
                        dw_faces = dw_pose_input[i].get("faces", [])
                        dw_hands = dw_pose_input[i].get("hands", [])
                        dw_bodies = dw_pose_input[i].get("bodies", {})
                        
                        for p, pts in enumerate(all_pts_2d_with_z):
                            if p >= len(dw_hands) // 2: continue
                            r_hand = dw_hands[p*2]
                            l_hand = dw_hands[p*2+1]

                            if len(pts) > 7 and pts[7] is not None and np.sum(r_hand) > 0.01:
                                wrist_norm = np.array([pts[7][0] / float(width), pts[7][1] / float(height)])
                                gap_offset = np.array([0.0, 0.0])
                                if pts[6] is not None:
                                    dir_vec = np.array([pts[7][0] - pts[6][0], pts[7][1] - pts[6][1]])
                                    norm_vec = np.linalg.norm(dir_vec)
                                    if norm_vec > 0: gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])
                                r_flat = np.array(r_hand[0]).flatten()
                                ox, oy = float((wrist_norm[0] + gap_offset[0]) - r_flat[0]), float((wrist_norm[1] + gap_offset[1]) - r_flat[1])
                                if isinstance(r_hand, np.ndarray):
                                    valid_mask = r_hand[:, 0] > 0
                                    r_hand[valid_mask, 0] += ox; r_hand[valid_mask, 1] += oy

                            if len(pts) > 4 and pts[4] is not None and np.sum(l_hand) > 0.01:
                                wrist_norm = np.array([pts[4][0] / float(width), pts[4][1] / float(height)])
                                gap_offset = np.array([0.0, 0.0])
                                if pts[3] is not None:
                                    dir_vec = np.array([pts[4][0] - pts[3][0], pts[4][1] - pts[3][1]])
                                    norm_vec = np.linalg.norm(dir_vec)
                                    if norm_vec > 0: gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])
                                l_flat = np.array(l_hand[0]).flatten()
                                ox, oy = float((wrist_norm[0] + gap_offset[0]) - l_flat[0]), float((wrist_norm[1] + gap_offset[1]) - l_flat[1])
                                if isinstance(l_hand, np.ndarray):
                                    valid_mask = l_hand[:, 0] > 0
                                    l_hand[valid_mask, 0] += ox; l_hand[valid_mask, 1] += oy

                            if draw_feet and "_posedata_feet" in dw_pose_input[i]:
                                feet_array = dw_pose_input[i]["_posedata_feet"]
                                if len(feet_array) > 0 and pts[10] is not None:
                                    ankle_norm = np.array([pts[10][0] / float(width), pts[10][1] / float(height)])
                                    feet_flat = np.array(feet_array[0]).flatten() / np.array([width, height])
                                    fox, foy = float(ankle_norm[0] - feet_flat[0]), float(ankle_norm[1] - feet_flat[1])

                            dw_hx, dw_hy, dw_nx, dw_ny = None, None, None, None
                            person_subset, candidate = None, None
                            if isinstance(dw_bodies, dict) and "candidate" in dw_bodies and "subset" in dw_bodies:
                                candidate, subset = dw_bodies["candidate"], dw_bodies["subset"]
                                if isinstance(subset, np.ndarray) and subset.ndim == 3 and subset.shape[0] == 1: subset = subset[0]
                                dw_bodies["subset"] = subset
                                if p < len(subset): person_subset = subset[p]
                                
                                nose_idx = int(np.array(person_subset).flatten()[0]) if person_subset is not None else -1
                                if 0 <= nose_idx < len(candidate):
                                    cand_val = np.array(candidate[nose_idx]).flatten()
                                    if len(cand_val) >= 2 and cand_val[0] > 0: dw_hx, dw_hy = float(cand_val[0]), float(cand_val[1])
                                
                                if person_subset is not None and len(np.array(person_subset).flatten()) > 1:
                                    neck_idx = int(np.array(person_subset).flatten()[1])
                                    if 0 <= neck_idx < len(candidate):
                                        cand_val = np.array(candidate[neck_idx]).flatten()
                                        if len(cand_val) >= 2 and cand_val[0] > 0: dw_nx, dw_ny = float(cand_val[0]), float(cand_val[1])

                            if dw_hx is None and p < len(dw_faces):
                                face = dw_faces[p]
                                if isinstance(face, np.ndarray) and len(face) > 30 and face[30, 0] > 0:
                                    dw_hx, dw_hy = float(face[30, 0]), float(face[30, 1])

                            if head_connection_mode == "Offset Head to Neck":
                                if pts[1] is not None and dw_nx is not None and dw_ny is not None:
                                    ox, oy = float((float(pts[1][0]) / float(width)) - dw_nx), float((float(pts[1][1]) / float(height)) - dw_ny)
                                    if person_subset is not None and candidate is not None:
                                        for h_idx in [0, 14, 15, 16, 17, 18, 19, 20]:
                                            if h_idx < len(person_subset):
                                                cand_idx = int(np.array(person_subset).flatten()[h_idx])
                                                if 0 <= cand_idx < len(candidate):
                                                    cand = candidate[cand_idx]
                                                    if isinstance(cand, np.ndarray): cand.flat[0] += ox; cand.flat[1] += oy
                                                    elif isinstance(cand, list):
                                                        if isinstance(cand[0], list): cand[0][0] += ox; cand[0][1] += oy
                                                        else: cand[0] += ox; cand[1] += oy

                                    if p < len(dw_faces):
                                        face = dw_faces[p]
                                        if isinstance(face, np.ndarray):
                                            valid_mask = face[:, 0] > 0
                                            face[valid_mask, 0] += ox; face[valid_mask, 1] += oy
                                        elif isinstance(face, list):
                                            for f_pt in face:
                                                if f_pt[0] > 0: f_pt[0] += ox; f_pt[1] += oy
                                                
                                    pixel_ox, pixel_oy = ox * float(width), oy * float(height)
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None: pts[h_idx][0] += pixel_ox; pts[h_idx][1] += pixel_oy

                            elif head_connection_mode == "Keep Head & Stretch Neck":
                                if pts[0] is not None and dw_hx is not None and dw_hy is not None:
                                    pixel_ox, pixel_oy = (dw_hx * float(width)) - pts[0][0], (dw_hy * float(height)) - pts[0][1]
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None: pts[h_idx][0] += pixel_ox; pts[h_idx][1] += pixel_oy

                    # === NEU: Füge fertige Punkte zur Masken-Pipeline hinzu ===
                    all_frames_pts_for_mask.append(all_pts_2d_with_z)

                    bones_to_draw = []
                    for pts in all_pts_2d_with_z:
                        for limb_idx, limb in enumerate(mimic_limb_seq):
                            start_idx, end_idx = limb[0], limb[1]
                            if pts[start_idx] is not None and pts[end_idx] is not None:
                                pt1, pt2 = pts[start_idx], pts[end_idx]
                                bones_to_draw.append({'pt1': (pt1[0], pt1[1]), 'pt2': (pt2[0], pt2[1]), 'z': (pt1[2] + pt2[2]) / 2.0, 'color': limb_colors_rgb[limb_idx % len(limb_colors_rgb)]})
                                
                    bones_to_draw.sort(key=lambda b: b['z'], reverse=True)
                    for bone in bones_to_draw:
                        x1, y1, x2, y2, color = *bone['pt1'], *bone['pt2'], bone['color']
                        length = math.hypot(x1 - x2, y1 - y2)
                        if length > 0.1:
                            polygon = cv2.ellipse2Poly((int((x1+x2)/2), int((y1+y2)/2)), (int(length / 2), line_thickness), int(math.degrees(math.atan2(y1 - y2, x1 - x2))), 0, 360, 1)
                            cv2.fillConvexPoly(frame_img, polygon, color, lineType=cv2.LINE_AA)
                            
                    frame_img = (frame_img * 0.6).astype(np.uint8)

                    joints_to_draw = []
                    for pts in all_pts_2d_with_z:
                        for j_idx, pt in enumerate(pts):
                            if pt is not None:
                                joints_to_draw.append({'pt': (pt[0], pt[1]), 'z': pt[2], 'color': joint_colors_rgb[j_idx % len(joint_colors_rgb)]})
                    joints_to_draw.sort(key=lambda j: j['z'], reverse=True)
                    for joint in joints_to_draw:
                        x, y = joint['pt']
                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(frame_img, (int(x), int(y)), point_radius, joint['color'], thickness=-1, lineType=cv2.LINE_AA)

                    alpha_channel = np.where(np.any(frame_img > 0, axis=-1), 255, 0).astype(np.uint8)
                    frames_np_rgba.append(np.dstack((frame_img, alpha_channel)))

            if dw_pose_input is not None and draw_2d:
                canvas_2d = draw_pose_to_canvas_np(dw_pose_input, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True, show_face_flag=draw_face, show_hand_flag=draw_hands)
                for i in range(len(frames_np_rgba)):
                    frame_rgba, canvas_img = frames_np_rgba[i], canvas_2d[i]
                    mask = canvas_img != 0
                    frame_rgba[:, :, :3][mask] = canvas_img[mask]
                    frame_rgba[:, :, 3][np.any(canvas_img > 0, axis=-1)] = 255
                    frames_np_rgba[i] = frame_rgba

            frames_tensor = torch.from_numpy(np.stack(frames_np_rgba, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            if isinstance(scaled_nlf_poses, dict): scaled_nlf_poses['joints3d_nonparam'] = [pose_input]
            else: scaled_nlf_poses = pose_input
                
            node_mappings = json.dumps({"node_name": "RenderNLFPosesDirectPoseDataMimic14", "status": "success", "frames": len(pose_input)})
            
            # === NEU: Bündle alles für die V3 Masken Node ===
            nlf_data_for_mask = {
                "all_frames_pts": all_frames_pts_for_mask,
                "dw_pose_input": dw_pose_input,
                "width": width,
                "height": height,
                "focal_length": intrinsic_matrix[0, 0]
            }
            
            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), scaled_nlf_poses, node_mappings, nlf_data_for_mask)

        except Exception as e:
            log_messages.append(traceback.format_exc())
            return (torch.zeros((1, height, width, 3)), torch.zeros((1, height, width)), "\n".join(log_messages), nlf_poses, "{}", None)


class NLFDataToMaskV3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_data_for_mask": ("NLF_MASK_DATA", {"tooltip": "Der Output aus Mimic 14"}),
                "stick_width": ("INT", {"default": 15, "min": 1, "max": 300, "tooltip": "Dicke der Körper-Knochen"}),
                
                # ENTTRIEGELTER SLIDER: Werte im 100er-Bereich nötig wegen Z-Tiefe in Millimetern!
                "head_circle_scale": ("FLOAT", {"default": 150.0, "min": 1.0, "max": 3000.0, "step": 5.0, "tooltip": "3D-Radius (Zentrum: DW-Nase XY, Tiefe: NLF-Hals Z)"}),
                
                "draw_neck_polygon": ("BOOLEAN", {"default": True, "tooltip": "Verbindet DW-Ohren mit NLF-Schultern"}),
                "draw_body_rectangle": ("BOOLEAN", {"default": True, "tooltip": "Viereck zwischen Schultern und Hüfte"}),
                "draw_hip_circles": ("BOOLEAN", {"default": True, "tooltip": "Kugeln fürs Hinterteil"}),
                "hip_circle_scale": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 5.0, "step": 0.05, "tooltip": "Größe basierend auf der Rumpflänge"}),
                "draw_hands_and_face": ("BOOLEAN", {"default": True}),
                "hands_face_dilate": ("INT", {"default": 8, "min": 0, "max": 50, "tooltip": "Bläht die Hände und Gesichtslinien auf"}),
                "interpolate_missing_frames": ("BOOLEAN", {"default": True, "tooltip": "Füllt Lücken linear auf (Körper, Gesicht & Hände)"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Masken Generator V3 (Hybrider 3D-Kopfkreis gefixt, entriegelter Scale-Slider)."

    def process(self, nlf_data_for_mask, stick_width, head_circle_scale, draw_neck_polygon, draw_body_rectangle, draw_hip_circles, hip_circle_scale, draw_hands_and_face, hands_face_dilate, interpolate_missing_frames):
        import numpy as np
        import torch
        import cv2
        import math
        import copy
        from ...pose_draw.draw_pose_utils import draw_pose_to_canvas_np

        if not nlf_data_for_mask:
            return (torch.zeros((1, 512, 512)),)

        all_frames_pts = copy.deepcopy(nlf_data_for_mask["all_frames_pts"])
        dw_pose_input = copy.deepcopy(nlf_data_for_mask["dw_pose_input"])
        width = nlf_data_for_mask["width"]
        height = nlf_data_for_mask["height"]
        focal_length = nlf_data_for_mask["focal_length"]

        # ====================================================================
        # LINEARE INTERPOLATION FÜR KÖRPER, HÄNDE UND GESICHT
        # ====================================================================
        if interpolate_missing_frames:
            max_people = max([len(f) for f in all_frames_pts]) if all_frames_pts else 0
            
            # 1. NLF Punkte Interpolieren
            for p in range(max_people):
                for j in range(18):
                    last_valid = -1
                    for i in range(len(all_frames_pts)):
                        valid = False
                        if p < len(all_frames_pts[i]) and len(all_frames_pts[i][p]) > j:
                            if all_frames_pts[i][p][j] is not None:
                                valid = True
                                
                        if valid:
                            if last_valid != -1 and i - last_valid > 1:
                                start_pt = all_frames_pts[last_valid][p][j]
                                end_pt = all_frames_pts[i][p][j]
                                steps = i - last_valid
                                for step in range(1, steps):
                                    frac = step / steps
                                    ix = int(start_pt[0] + (end_pt[0] - start_pt[0]) * frac)
                                    iy = int(start_pt[1] + (end_pt[1] - start_pt[1]) * frac)
                                    iz = start_pt[2] + (end_pt[2] - start_pt[2]) * frac
                                    
                                    while len(all_frames_pts[last_valid+step]) <= p:
                                        all_frames_pts[last_valid+step].append([None]*18)
                                    while len(all_frames_pts[last_valid+step][p]) <= j:
                                        all_frames_pts[last_valid+step][p].append(None)
                                        
                                    all_frames_pts[last_valid+step][p][j] = [ix, iy, iz]
                            last_valid = i

            # 2. DW Pose Interpolieren
            if dw_pose_input is not None:
                for p in range(max_people):
                    for j in range(18):
                        last_valid = -1
                        for i in range(len(dw_pose_input)):
                            cand = dw_pose_input[i].get("bodies", {}).get("candidate", [])
                            valid, pt = False, None
                            
                            if isinstance(cand, list) and p < len(cand) and len(cand[p]) > j and cand[p][j][0] > 0:
                                valid, pt = True, cand[p][j]
                            elif isinstance(cand, np.ndarray) and p < cand.shape[0] and len(cand[p]) > j and cand[p][j][0] > 0:
                                valid, pt = True, cand[p][j]
                                    
                            if valid:
                                if last_valid != -1 and i - last_valid > 1:
                                    start_cand = dw_pose_input[last_valid]["bodies"]["candidate"]
                                    start_pt = start_cand[p][j] if isinstance(start_cand, list) else start_cand[p][j]
                                    steps = i - last_valid
                                    for step in range(1, steps):
                                        frac = step / steps
                                        ix = start_pt[0] + (pt[0] - start_pt[0]) * frac
                                        iy = start_pt[1] + (pt[1] - start_pt[1]) * frac
                                        
                                        step_cand = dw_pose_input[last_valid+step].get("bodies", {}).get("candidate", [])
                                        if isinstance(step_cand, list) and p < len(step_cand) and len(step_cand[p]) > j:
                                            step_cand[p][j] = [ix, iy]
                                        elif isinstance(step_cand, np.ndarray) and p < step_cand.shape[0] and len(step_cand[p]) > j:
                                            step_cand[p][j] = [ix, iy]
                                last_valid = i

                    for h_idx in [0, 1]:
                        hand_offset = p * 2 + h_idx
                        last_valid = -1
                        for i in range(len(dw_pose_input)):
                            dw_hands = dw_pose_input[i].get("hands", [])
                            valid = False
                            if len(dw_hands) > hand_offset:
                                h_arr = np.array(dw_hands[hand_offset])
                                if np.sum(np.abs(h_arr)) > 0.01:
                                    valid = True
                            
                            if valid:
                                if last_valid != -1 and i - last_valid > 1:
                                    start_hand = np.array(dw_pose_input[last_valid]["hands"][hand_offset])
                                    end_hand = np.array(dw_pose_input[i]["hands"][hand_offset])
                                    steps = i - last_valid
                                    for step in range(1, steps):
                                        frac = step / steps
                                        interp_hand = start_hand + (end_hand - start_hand) * frac
                                        
                                        if len(dw_pose_input[last_valid+step].get("hands", [])) <= hand_offset:
                                            if "hands" not in dw_pose_input[last_valid+step]: dw_pose_input[last_valid+step]["hands"] = []
                                            while len(dw_pose_input[last_valid+step]["hands"]) <= hand_offset:
                                                dw_pose_input[last_valid+step]["hands"].append(np.zeros_like(start_hand))
                                                
                                        dw_pose_input[last_valid+step]["hands"][hand_offset] = interp_hand
                                last_valid = i

                    last_valid = -1
                    for i in range(len(dw_pose_input)):
                        dw_faces = dw_pose_input[i].get("faces", [])
                        valid = False
                        if len(dw_faces) > p:
                            f_arr = np.array(dw_faces[p])
                            if np.sum(np.abs(f_arr)) > 0.01:
                                valid = True
                        
                        if valid:
                            if last_valid != -1 and i - last_valid > 1:
                                start_face = np.array(dw_pose_input[last_valid]["faces"][p])
                                end_face = np.array(dw_pose_input[i]["faces"][p])
                                steps = i - last_valid
                                for step in range(1, steps):
                                    frac = step / steps
                                    interp_face = start_face + (end_face - start_face) * frac
                                    
                                    if len(dw_pose_input[last_valid+step].get("faces", [])) <= p:
                                        if "faces" not in dw_pose_input[last_valid+step]: dw_pose_input[last_valid+step]["faces"] = []
                                        while len(dw_pose_input[last_valid+step]["faces"]) <= p:
                                            dw_pose_input[last_valid+step]["faces"].append(np.zeros_like(start_face))
                                            
                                    dw_pose_input[last_valid+step]["faces"][p] = interp_face
                            last_valid = i
        # ====================================================================

        mimic_limb_seq = [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], 
            [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
        ]

        canvas_2d_frames = None
        if dw_pose_input is not None and draw_hands_and_face:
            canvas_2d_frames = draw_pose_to_canvas_np(dw_pose_input, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True, show_face_flag=True, show_hand_flag=True)

        frames_mask = []

        for i in range(len(all_frames_pts)):
            mask_img = np.zeros((height, width), dtype=np.uint8)
            pts_list = all_frames_pts[i]
            
            candidate = []
            if dw_pose_input is not None and i < len(dw_pose_input):
                dw_bodies = dw_pose_input[i].get("bodies", {})
                candidate = dw_bodies.get("candidate", [])

            for p, pts in enumerate(pts_list):
                cand = None
                if isinstance(candidate, list) and p < len(candidate):
                    cand = candidate[p]
                elif isinstance(candidate, np.ndarray) and p < candidate.shape[0]:
                    cand = candidate[p]

                nose, r_ear, l_ear = None, None, None
                if cand is not None and len(cand) >= 18:
                    if cand[0][0] > 0: nose = (int(cand[0][0]*width), int(cand[0][1]*height))
                    if cand[16][0] > 0: r_ear = (int(cand[16][0]*width), int(cand[16][1]*height))
                    if cand[17][0] > 0: l_ear = (int(cand[17][0]*width), int(cand[17][1]*height))

                # --- HYBRIDE 3D KOPF KUGEL (EXAKT WIE GEFORDERT) ---
                # Priorisiere DW Pose Nase für XY
                center_xy = nose
                if center_xy is None and pts[0] is not None:
                    center_xy = (pts[0][0], pts[0][1]) # Fallback auf NLF, falls DW fehlt
                    
                if head_circle_scale > 0 and center_xy is not None and pts[0] is not None:
                    # NLF Z-Tiefe vom obersten Halspunkt (pts[0][2])
                    z_depth = max(0.1, abs(pts[0][2])) # max() verhindert Crash bei Z=0
                    
                    # 3D Skalierung: Neuer Scale-Faktor erlaubt realistische Größen
                    pixel_r = max(2, int((head_circle_scale * focal_length) / z_depth))
                    cv2.circle(mask_img, center_xy, pixel_r, 255, -1, lineType=cv2.LINE_AA)

                # Hals Polygon
                if draw_neck_polygon and pts[2] is not None and pts[5] is not None:
                    r_shoulder, l_shoulder = (pts[2][0], pts[2][1]), (pts[5][0], pts[5][1])
                    poly_pts = []
                    if r_ear: poly_pts.append(r_ear)
                    elif nose: poly_pts.append(nose)
                    if l_ear: poly_pts.append(l_ear)
                    elif nose and not poly_pts: poly_pts.append(nose)
                    poly_pts.extend([l_shoulder, r_shoulder])
                    
                    if len(poly_pts) >= 3:
                        cv2.fillPoly(mask_img, [np.array(poly_pts)], 255)

                # Rumpf Viereck
                if draw_body_rectangle:
                    if pts[2] is not None and pts[5] is not None and pts[11] is not None and pts[8] is not None:
                        rect_cnt = np.array([[pts[2][0], pts[2][1]], [pts[5][0], pts[5][1]], [pts[11][0], pts[11][1]], [pts[8][0], pts[8][1]]])
                        cv2.fillPoly(mask_img, [rect_cnt], 255)

                # Hüft Kreise
                if draw_hip_circles and pts[8] is not None and pts[11] is not None and pts[2] is not None and pts[5] is not None:
                    dist_r = math.hypot(pts[2][0] - pts[8][0], pts[2][1] - pts[8][1])
                    dist_l = math.hypot(pts[5][0] - pts[11][0], pts[5][1] - pts[11][1])
                    torso_len = (dist_r + dist_l) / 2.0
                    pixel_r = max(2, int(torso_len * hip_circle_scale))
                    cv2.circle(mask_img, (pts[8][0], pts[8][1]), pixel_r, 255, -1, lineType=cv2.LINE_AA)
                    cv2.circle(mask_img, (pts[11][0], pts[11][1]), pixel_r, 255, -1, lineType=cv2.LINE_AA)

                # Körper Sticks
                for limb in mimic_limb_seq:
                    if pts[limb[0]] is not None and pts[limb[1]] is not None:
                        cv2.line(mask_img, (pts[limb[0]][0], pts[limb[0]][1]), (pts[limb[1]][0], pts[limb[1]][1]), 255, stick_width, lineType=cv2.LINE_AA)

            # Dicke Hände & Gesicht
            if canvas_2d_frames is not None and i < len(canvas_2d_frames):
                canvas_img = canvas_2d_frames[i]
                hf_mask = np.where(np.any(canvas_img > 0, axis=-1), 255, 0).astype(np.uint8)
                if hands_face_dilate > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hands_face_dilate, hands_face_dilate))
                    hf_mask = cv2.dilate(hf_mask, kernel, iterations=1)
                mask_img = np.maximum(mask_img, hf_mask)

            frames_mask.append(mask_img)

        mask_tensor = torch.from_numpy(np.stack(frames_mask, axis=0)).float() / 255.0
        return (mask_tensor,)


class NLFDataToMaskV4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_data_for_mask": ("NLF_MASK_DATA", {"tooltip": "Der Output aus Mimic 14"}),
                
                # NEU: 3D Skalierung für die Körper-Sticks
                "stick_3d_scale": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 3000.0, "step": 1.0, "tooltip": "3D-Dicke der Knochen (skaliert mit Z-Tiefe)"}),
                
                "head_circle_scale": ("FLOAT", {"default": 150.0, "min": 1.0, "max": 3000.0, "step": 5.0, "tooltip": "3D-Radius (Zentrum: DW-Nase XY, Tiefe: NLF-Hals Z)"}),
                "draw_neck_polygon": ("BOOLEAN", {"default": True, "tooltip": "Verbindet DW-Ohren mit NLF-Schultern"}),
                "draw_body_rectangle": ("BOOLEAN", {"default": True, "tooltip": "Viereck zwischen Schultern und Hüfte"}),
                "draw_hip_circles": ("BOOLEAN", {"default": True, "tooltip": "Kugeln fürs Hinterteil"}),
                "hip_circle_scale": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 5.0, "step": 0.05, "tooltip": "Größe basierend auf der Rumpflänge"}),
                "draw_hands_and_face": ("BOOLEAN", {"default": True}),
                
                # NEU: 3D Skalierung für das Aufblähen der Hände
                "hands_face_dilate_scale": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 1000.0, "step": 1.0, "tooltip": "3D-Skalierung für die Dicke von Händen und Gesicht"}),
                
                "interpolate_missing_frames": ("BOOLEAN", {"default": True, "tooltip": "Füllt Lücken linear auf (Körper, Gesicht & Hände)"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Masken Generator V3 (Vollständiges 3D-Volumen: Sticks und Hände skalieren jetzt mit Z-Tiefe!)."

    def process(self, nlf_data_for_mask, stick_3d_scale, head_circle_scale, draw_neck_polygon, draw_body_rectangle, draw_hip_circles, hip_circle_scale, draw_hands_and_face, hands_face_dilate_scale, interpolate_missing_frames):
        import numpy as np
        import torch
        import cv2
        import math
        import copy
        from ...pose_draw.draw_pose_utils import draw_pose_to_canvas_np

        if not nlf_data_for_mask:
            return (torch.zeros((1, 512, 512)),)

        all_frames_pts = copy.deepcopy(nlf_data_for_mask["all_frames_pts"])
        dw_pose_input = copy.deepcopy(nlf_data_for_mask["dw_pose_input"])
        width = nlf_data_for_mask["width"]
        height = nlf_data_for_mask["height"]
        focal_length = nlf_data_for_mask["focal_length"]

        # ====================================================================
        # LINEARE INTERPOLATION FÜR KÖRPER, HÄNDE UND GESICHT
        # ====================================================================
        if interpolate_missing_frames:
            max_people = max([len(f) for f in all_frames_pts]) if all_frames_pts else 0
            
            # 1. NLF Punkte Interpolieren
            for p in range(max_people):
                for j in range(18):
                    last_valid = -1
                    for i in range(len(all_frames_pts)):
                        valid = False
                        if p < len(all_frames_pts[i]) and len(all_frames_pts[i][p]) > j:
                            if all_frames_pts[i][p][j] is not None:
                                valid = True
                                
                        if valid:
                            if last_valid != -1 and i - last_valid > 1:
                                start_pt = all_frames_pts[last_valid][p][j]
                                end_pt = all_frames_pts[i][p][j]
                                steps = i - last_valid
                                for step in range(1, steps):
                                    frac = step / steps
                                    ix = int(start_pt[0] + (end_pt[0] - start_pt[0]) * frac)
                                    iy = int(start_pt[1] + (end_pt[1] - start_pt[1]) * frac)
                                    iz = start_pt[2] + (end_pt[2] - start_pt[2]) * frac
                                    
                                    while len(all_frames_pts[last_valid+step]) <= p:
                                        all_frames_pts[last_valid+step].append([None]*18)
                                    while len(all_frames_pts[last_valid+step][p]) <= j:
                                        all_frames_pts[last_valid+step][p].append(None)
                                        
                                    all_frames_pts[last_valid+step][p][j] = [ix, iy, iz]
                            last_valid = i

            # 2. DW Pose Interpolieren
            if dw_pose_input is not None:
                for p in range(max_people):
                    for j in range(18):
                        last_valid = -1
                        for i in range(len(dw_pose_input)):
                            cand = dw_pose_input[i].get("bodies", {}).get("candidate", [])
                            valid, pt = False, None
                            if isinstance(cand, list) and p < len(cand) and len(cand[p]) > j and cand[p][j][0] > 0:
                                valid, pt = True, cand[p][j]
                            elif isinstance(cand, np.ndarray) and p < cand.shape[0] and len(cand[p]) > j and cand[p][j][0] > 0:
                                valid, pt = True, cand[p][j]
                            if valid:
                                if last_valid != -1 and i - last_valid > 1:
                                    start_cand = dw_pose_input[last_valid]["bodies"]["candidate"]
                                    start_pt = start_cand[p][j] if isinstance(start_cand, list) else start_cand[p][j]
                                    steps = i - last_valid
                                    for step in range(1, steps):
                                        frac = step / steps
                                        ix = start_pt[0] + (pt[0] - start_pt[0]) * frac
                                        iy = start_pt[1] + (pt[1] - start_pt[1]) * frac
                                        step_cand = dw_pose_input[last_valid+step].get("bodies", {}).get("candidate", [])
                                        if isinstance(step_cand, list) and p < len(step_cand) and len(step_cand[p]) > j: step_cand[p][j] = [ix, iy]
                                        elif isinstance(step_cand, np.ndarray) and p < step_cand.shape[0] and len(step_cand[p]) > j: step_cand[p][j] = [ix, iy]
                                last_valid = i

                    for h_idx in [0, 1]:
                        hand_offset = p * 2 + h_idx
                        last_valid = -1
                        for i in range(len(dw_pose_input)):
                            dw_hands = dw_pose_input[i].get("hands", [])
                            valid = False
                            if len(dw_hands) > hand_offset:
                                h_arr = np.array(dw_hands[hand_offset])
                                if np.sum(np.abs(h_arr)) > 0.01: valid = True
                            if valid:
                                if last_valid != -1 and i - last_valid > 1:
                                    start_hand = np.array(dw_pose_input[last_valid]["hands"][hand_offset])
                                    end_hand = np.array(dw_pose_input[i]["hands"][hand_offset])
                                    steps = i - last_valid
                                    for step in range(1, steps):
                                        frac = step / steps
                                        interp_hand = start_hand + (end_hand - start_hand) * frac
                                        if len(dw_pose_input[last_valid+step].get("hands", [])) <= hand_offset:
                                            if "hands" not in dw_pose_input[last_valid+step]: dw_pose_input[last_valid+step]["hands"] = []
                                            while len(dw_pose_input[last_valid+step]["hands"]) <= hand_offset: dw_pose_input[last_valid+step]["hands"].append(np.zeros_like(start_hand))
                                        dw_pose_input[last_valid+step]["hands"][hand_offset] = interp_hand
                                last_valid = i

                    last_valid = -1
                    for i in range(len(dw_pose_input)):
                        dw_faces = dw_pose_input[i].get("faces", [])
                        valid = False
                        if len(dw_faces) > p:
                            f_arr = np.array(dw_faces[p])
                            if np.sum(np.abs(f_arr)) > 0.01: valid = True
                        if valid:
                            if last_valid != -1 and i - last_valid > 1:
                                start_face = np.array(dw_pose_input[last_valid]["faces"][p])
                                end_face = np.array(dw_pose_input[i]["faces"][p])
                                steps = i - last_valid
                                for step in range(1, steps):
                                    frac = step / steps
                                    interp_face = start_face + (end_face - start_face) * frac
                                    if len(dw_pose_input[last_valid+step].get("faces", [])) <= p:
                                        if "faces" not in dw_pose_input[last_valid+step]: dw_pose_input[last_valid+step]["faces"] = []
                                        while len(dw_pose_input[last_valid+step]["faces"]) <= p: dw_pose_input[last_valid+step]["faces"].append(np.zeros_like(start_face))
                                    dw_pose_input[last_valid+step]["faces"][p] = interp_face
                            last_valid = i
        # ====================================================================

        mimic_limb_seq = [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], 
            [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
        ]

        canvas_2d_frames = None
        if dw_pose_input is not None and draw_hands_and_face:
            canvas_2d_frames = draw_pose_to_canvas_np(dw_pose_input, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True, show_face_flag=True, show_hand_flag=True)

        frames_mask = []

        for i in range(len(all_frames_pts)):
            mask_img = np.zeros((height, width), dtype=np.uint8)
            pts_list = all_frames_pts[i]
            
            # Sammle durchschnittliche Z-Tiefe dieses Frames (für Hände/Gesichts-Skalierung)
            frame_z_depths = []
            
            candidate = []
            if dw_pose_input is not None and i < len(dw_pose_input):
                dw_bodies = dw_pose_input[i].get("bodies", {})
                candidate = dw_bodies.get("candidate", [])

            for p, pts in enumerate(pts_list):
                if pts[0] is not None:
                    frame_z_depths.append(abs(pts[0][2]))
                
                cand = None
                if isinstance(candidate, list) and p < len(candidate): cand = candidate[p]
                elif isinstance(candidate, np.ndarray) and p < candidate.shape[0]: cand = candidate[p]

                nose, r_ear, l_ear = None, None, None
                if cand is not None and len(cand) >= 18:
                    if cand[0][0] > 0: nose = (int(cand[0][0]*width), int(cand[0][1]*height))
                    if cand[16][0] > 0: r_ear = (int(cand[16][0]*width), int(cand[16][1]*height))
                    if cand[17][0] > 0: l_ear = (int(cand[17][0]*width), int(cand[17][1]*height))

                # HYBRIDE 3D KOPF KUGEL
                center_xy = nose
                if center_xy is None and pts[0] is not None: center_xy = (pts[0][0], pts[0][1])
                if head_circle_scale > 0 and center_xy is not None and pts[0] is not None:
                    z_depth = max(0.1, abs(pts[0][2]))
                    pixel_r = max(2, int((head_circle_scale * focal_length) / z_depth))
                    cv2.circle(mask_img, center_xy, pixel_r, 255, -1, lineType=cv2.LINE_AA)

                # Hals Polygon
                if draw_neck_polygon and pts[2] is not None and pts[5] is not None:
                    r_shoulder, l_shoulder = (pts[2][0], pts[2][1]), (pts[5][0], pts[5][1])
                    poly_pts = []
                    if r_ear: poly_pts.append(r_ear)
                    elif nose: poly_pts.append(nose)
                    if l_ear: poly_pts.append(l_ear)
                    elif nose and not poly_pts: poly_pts.append(nose)
                    poly_pts.extend([l_shoulder, r_shoulder])
                    if len(poly_pts) >= 3:
                        cv2.fillPoly(mask_img, [np.array(poly_pts)], 255)

                # Rumpf Viereck
                if draw_body_rectangle:
                    if pts[2] is not None and pts[5] is not None and pts[11] is not None and pts[8] is not None:
                        rect_cnt = np.array([[pts[2][0], pts[2][1]], [pts[5][0], pts[5][1]], [pts[11][0], pts[11][1]], [pts[8][0], pts[8][1]]])
                        cv2.fillPoly(mask_img, [rect_cnt], 255)

                # Hüft Kreise
                if draw_hip_circles and pts[8] is not None and pts[11] is not None and pts[2] is not None and pts[5] is not None:
                    dist_r = math.hypot(pts[2][0] - pts[8][0], pts[2][1] - pts[8][1])
                    dist_l = math.hypot(pts[5][0] - pts[11][0], pts[5][1] - pts[11][1])
                    torso_len = (dist_r + dist_l) / 2.0
                    pixel_r = max(2, int(torso_len * hip_circle_scale))
                    cv2.circle(mask_img, (pts[8][0], pts[8][1]), pixel_r, 255, -1, lineType=cv2.LINE_AA)
                    cv2.circle(mask_img, (pts[11][0], pts[11][1]), pixel_r, 255, -1, lineType=cv2.LINE_AA)

                # --- NEU: Körper Sticks in echtem 3D ---
                for limb in mimic_limb_seq:
                    if pts[limb[0]] is not None and pts[limb[1]] is not None:
                        pt1, pt2 = pts[limb[0]], pts[limb[1]]
                        
                        # Durchschnittliche Z-Tiefe dieses spezifischen Knochens
                        limb_z_depth = max(0.1, (abs(pt1[2]) + abs(pt2[2])) / 2.0)
                        
                        # 3D Pixel-Dicke berechnen
                        thickness = max(1, int((stick_3d_scale * focal_length) / limb_z_depth))
                        
                        cv2.line(mask_img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 255, thickness, lineType=cv2.LINE_AA)

            # --- NEU: Dicke Hände & Gesicht in echtem 3D ---
            if canvas_2d_frames is not None and i < len(canvas_2d_frames):
                canvas_img = canvas_2d_frames[i]
                hf_mask = np.where(np.any(canvas_img > 0, axis=-1), 255, 0).astype(np.uint8)
                
                if hands_face_dilate_scale > 0:
                    # Nimm die durchschnittliche Z-Tiefe der Figuren in diesem Frame (Fallback 1000)
                    avg_z_depth = sum(frame_z_depths) / len(frame_z_depths) if frame_z_depths else 1000.0
                    avg_z_depth = max(0.1, avg_z_depth)
                    
                    # 3D Skalierung der Dilatation
                    dilate_size = max(1, int((hands_face_dilate_scale * focal_length) / avg_z_depth))
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
                    hf_mask = cv2.dilate(hf_mask, kernel, iterations=1)
                    
                mask_img = np.maximum(mask_img, hf_mask)

            frames_mask.append(mask_img)

        mask_tensor = torch.from_numpy(np.stack(frames_mask, axis=0)).float() / 255.0
        return (mask_tensor,)


class RenderNLFPosesDirectPoseDataMimic16:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die originalen NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Dicke der Knochen (Ovale Form)"}),
                "point_radius": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Größe der Gelenkpunkte"}),
                "head_connection_mode": (["Offset Head to Neck", "Keep Head & Stretch Neck"], {"default": "Offset Head to Neck", "tooltip": "Wie der Kopf an den Hals angebunden wird"}),
                "draw_2d": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet 2D Overlay"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet das Gesicht"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet die Hände"}),
                
                # POSEDATA TOGGLES
                "use_pose_data": ("BOOLEAN", {"default": True, "tooltip": "Nutzt PoseData statt DW Poses für Hände/Füße"}),
                "use_dwpose_head_for_posedata": ("BOOLEAN", {"default": True, "tooltip": "Nimmt KOMPLETTEN Kopf & Gesicht von DW Pose, auch wenn PoseData an ist"}),
                "draw_feet": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet Füße von PoseData und mappt sie an die NLF-Knöchel"}),
                "draw_nlf_feet": ("BOOLEAN", {"default": False, "tooltip": "Zeichnet Füße direkt aus originalen NLF-Daten (überschreibt PoseData-Füße)"}),
                
                # NEU: Hände Tweaks (Skalierung, Alpha, Offsets)
                "apply_fingertip_offsets": ("BOOLEAN", {"default": True, "tooltip": "Wendet die Rotations-Offsets auf die Finger an (falls Input vorhanden)"}),
                "hand_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Skaliert die Hände (1.0 = normal)"}),
                "hand_face_alpha": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Deckkraft für 2D Hände und Gesicht (0.6 passt gut zum Körper)"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht als Fallback"}),
                "pose_data_fallback": ("POSEDATA", {"tooltip": "Pose Data (z.B. ViTPose) für Hände/Füße"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
                "fingertip_offsets": ("STRING", {"forceInput": True, "tooltip": "Die JSON-Offsets aus der HandDebug-Node"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING", "NLF_MASK_DATA")
    RETURN_NAMES = ("image", "mask", "log_output", "scaled_nlf_poses", "node_mappings", "nlf_data_for_mask")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Mimic 15 mit Offsets, Logging, NLF Feet (Rot/Gelb) und Hand-Skalierung."

    def process(self, nlf_poses, width, height, line_thickness=4, point_radius=4, head_connection_mode="Offset Head to Neck", draw_2d=True, draw_face=True, draw_hands=True, use_pose_data=True, use_dwpose_head_for_posedata=True, draw_feet=True, draw_nlf_feet=False, apply_fingertip_offsets=True, hand_scale_factor=1.0, hand_face_alpha=0.6, dw_poses_fallback=None, pose_data_fallback=None, nlf_render_config="{}", fingertip_offsets=None):
        import copy
        import json
        import math
        import torch
        import numpy as np
        import traceback
        import cv2
        from ...NLFPoseExtract.nlf_render_flat import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_single_p2d
        from ...pose_draw.draw_pose_utils import draw_pose_to_canvas_np

        log_messages = ["=== RENDER NLF POSES MIMIC 15 LOG ==="]
        scaled_nlf_poses = copy.deepcopy(nlf_poses)
        
        # Offsets Parsen
        offsets_dict = {}
        if apply_fingertip_offsets and fingertip_offsets and fingertip_offsets.strip() != "":
            try:
                offsets_dict = json.loads(fingertip_offsets)
                log_messages.append("Fingertip Offsets erfolgreich geladen und aktiviert.")
            except Exception as e:
                log_messages.append(f"Fehler beim Parsen der fingertip_offsets: {e}")
        elif not apply_fingertip_offsets:
            log_messages.append("Fingertip Offsets sind per Toggle deaktiviert.")
                
        # NLF Füsse überschreiben PoseData Füsse
        if draw_nlf_feet:
            draw_feet = False
        
        try:
            pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
            
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            if dw_pose_input is None and use_pose_data:
                dw_pose_input = [{"bodies": {"candidate": [np.zeros((18, 2))], "subset": [np.full(18, -1)]}, "hands": np.zeros((2, 21, 2)), "faces": [np.zeros((68, 2))]} for _ in range(len(pose_input))]
            
            pose_metas = []
            if use_pose_data and pose_data_fallback is not None:
                pose_metas = pose_data_fallback.get("pose_metas", [])

            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])

            # 3D Kamera Config Baking
            try:
                config = json.loads(nlf_render_config)
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    p_x, p_y = float(config["pivot_x"]), float(config["pivot_y"])
                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x *= width
                        p_y *= height
                    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
                    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
                    M13 = (cx - p_x) * (scale_x - 1.0) / fx
                    M23 = (cy - p_y) * (scale_y - 1.0) / fy
                    
                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None and len(pose_input[frame_idx]) > 0:
                            pts = pose_input[frame_idx]
                            X, Y, Z = pts[..., 0].clone(), pts[..., 1].clone(), pts[..., 2].clone()
                            pts[..., 0] = X * scale_x + Z * M13
                            pts[..., 1] = Y * scale_y + Z * M23
            except Exception as e:
                log_messages.append(f"Fehler bei 3D Transformation: {e}")

            # POSEDATA IN DW-STRUKTUR INJIZIEREN
            if use_pose_data and pose_metas:
                for p_idx in range(min(len(dw_pose_input), len(pose_metas))):
                    meta = pose_metas[p_idx]
                    dw = dw_pose_input[p_idx]
                    cand = dw["bodies"]["candidate"][0] if isinstance(dw["bodies"]["candidate"], list) else dw["bodies"]["candidate"][0]
                    subset = dw["bodies"]["subset"][0] if isinstance(dw["bodies"]["subset"], list) else dw["bodies"]["subset"][0]
                    
                    if draw_hands:
                        lh = getattr(meta, "kps_lhand", None)
                        rh = getattr(meta, "kps_rhand", None)
                        if lh is not None and len(lh) >= 21: dw["hands"][0] = np.array(lh[:, :2]) / np.array([width, height])
                        if rh is not None and len(rh) >= 21: dw["hands"][1] = np.array(rh[:, :2]) / np.array([width, height])

                    if not use_dwpose_head_for_posedata:
                        coco_to_op = {0: 0, 1: 15, 2: 14, 3: 17, 4: 16}
                        if getattr(meta, "kps_body", None) is not None:
                            body_pts = meta.kps_body
                            for coco_idx, op_idx in coco_to_op.items():
                                if coco_idx < len(body_pts) and body_pts[coco_idx][0] > 0:
                                    cand[op_idx] = [body_pts[coco_idx][0] / width, body_pts[coco_idx][1] / height]
                                    subset[op_idx] = op_idx
                        if draw_face:
                            face_pts = getattr(meta, "kps_face", None)
                            if face_pts is not None and len(face_pts) > 1:
                                dw["faces"][0] = np.array(face_pts[1:, :2]) / np.array([width, height])

                    if draw_feet and getattr(meta, "kps_body", None) is not None:
                        feet_pts = []
                        for f_idx in [19, 20, 21, 22, 23, 24]:
                            if f_idx < len(meta.kps_body) and meta.kps_body[f_idx][0] > 0: feet_pts.append(meta.kps_body[f_idx][:2])
                        dw["_posedata_feet"] = np.array(feet_pts)

            limb_colors_rgb = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)]
            joint_colors_rgb = limb_colors_rgb + [(255, 0, 85)]
            mimic_limb_seq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]

            frames_np_rgba = []
            all_frames_pts_for_mask = []

            for i in range(len(pose_input)):
                frame_img = np.zeros((height, width, 3), dtype=np.uint8)
                nlf_extra_bones = []
                nlf_extra_joints = []
                
                if pose_input[i] is not None:
                    joints3d_batch = pose_input[i]
                    people = joints3d_batch if joints3d_batch.dim() == 3 else [joints3d_batch] if joints3d_batch.dim() == 2 else []

                    all_pts_2d_with_z = []
                    for joints3d in people:
                        j3d_np = joints3d.cpu().numpy() if isinstance(joints3d, torch.Tensor) else joints3d
                        
                        # --- NEU: NLF Füße projizieren (mit exakten Farben) ---
                        if draw_nlf_feet and j3d_np.shape[0] >= 12:
                            # 7->10 ist Links (Rot), 8->11 ist Rechts (Gelb)
                            for s_idx, e_idx, foot_color in [(7, 10, (255, 0, 0)), (8, 11, (255, 255, 0))]: 
                                if np.sum(np.abs(j3d_np[s_idx])) > 0 and np.sum(np.abs(j3d_np[e_idx])) > 0:
                                    p1 = p3d_single_p2d(j3d_np[s_idx], intrinsic_matrix)
                                    p2 = p3d_single_p2d(j3d_np[e_idx], intrinsic_matrix)
                                    nlf_extra_bones.append({
                                        'pt1': (p1[0], p1[1]), 'pt2': (p2[0], p2[1]), 
                                        'z': (j3d_np[s_idx][2] + j3d_np[e_idx][2]) / 2.0, 
                                        'color': foot_color
                                    })
                                    nlf_extra_joints.append({
                                        'pt': (p2[0], p2[1]), 'z': j3d_np[e_idx][2], 
                                        'color': foot_color
                                    })

                        if np.sum(np.abs(j3d_np)) > 0.01:
                            j3d_coco = process_data_to_COCO_format(j3d_np)
                            pts_2d_with_z = []
                            for pt3d in j3d_coco:
                                if np.sum(np.abs(pt3d)) > 0:
                                    pt2d = p3d_single_p2d(pt3d, intrinsic_matrix)
                                    pts_2d_with_z.append([int(pt2d[0]), int(pt2d[1]), float(pt3d[2])])
                                else:
                                    pts_2d_with_z.append(None)
                                    
                            if len(pts_2d_with_z) > 5 and pts_2d_with_z[2] is not None and pts_2d_with_z[5] is not None:
                                p_r, p_l = pts_2d_with_z[2], pts_2d_with_z[5]
                                if pts_2d_with_z[1] is not None:
                                    p_neck = pts_2d_with_z[1]
                                    pts_2d_with_z[1][0] = int((p_r[0] + p_l[0]) / 2)
                                    pts_2d_with_z[1][1] = int((p_r[1] + p_l[1]) / 2)

                            all_pts_2d_with_z.append(pts_2d_with_z)

                    if dw_pose_input is not None and i < len(dw_pose_input):
                        dw_faces = dw_pose_input[i].get("faces", [])
                        dw_hands = dw_pose_input[i].get("hands", [])
                        dw_bodies = dw_pose_input[i].get("bodies", {})
                        
                        for p, pts in enumerate(all_pts_2d_with_z):
                            if p >= len(dw_hands) // 2: continue
                            r_hand = dw_hands[p*2]
                            l_hand = dw_hands[p*2+1]
                            
                            l_offset = [0.0, 0.0]
                            r_offset = [0.0, 0.0]
                            if apply_fingertip_offsets:
                                f_idx_str = str(i)
                                p_idx_str = str(p)
                                if f_idx_str in offsets_dict and p_idx_str in offsets_dict[f_idx_str]:
                                    l_offset = offsets_dict[f_idx_str][p_idx_str].get("left_hand", [0.0, 0.0])
                                    r_offset = offsets_dict[f_idx_str][p_idx_str].get("right_hand", [0.0, 0.0])

                            if len(pts) > 7 and pts[7] is not None and np.sum(r_hand) > 0.01:
                                wrist_norm = np.array([pts[7][0] / float(width), pts[7][1] / float(height)])
                                gap_offset = np.array([0.0, 0.0])
                                if pts[6] is not None:
                                    dir_vec = np.array([pts[7][0] - pts[6][0], pts[7][1] - pts[6][1]])
                                    norm_vec = np.linalg.norm(dir_vec)
                                    if norm_vec > 0: gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])
                                r_flat = np.array(r_hand[0]).flatten()
                                ox, oy = float((wrist_norm[0] + gap_offset[0]) - r_flat[0]), float((wrist_norm[1] + gap_offset[1]) - r_flat[1])
                                
                                if isinstance(r_hand, np.ndarray):
                                    valid_mask = r_hand[:, 0] > 0
                                    r_hand[valid_mask, 0] += ox
                                    r_hand[valid_mask, 1] += oy
                                    
                                    # --- NEU: Skalierung anwenden ---
                                    if hand_scale_factor != 1.0:
                                        wrist_pos = r_hand[0].copy()
                                        for f_idx in range(1, 21):
                                            if valid_mask[f_idx]:
                                                r_hand[f_idx] = wrist_pos + (r_hand[f_idx] - wrist_pos) * hand_scale_factor
                                    
                                    # Rotations-Offset auf FINGER anwenden
                                    if apply_fingertip_offsets and (abs(r_offset[0]) > 0.001 or abs(r_offset[1]) > 0.001):
                                        r_off_x, r_off_y = r_offset[0] / float(width), r_offset[1] / float(height)
                                        finger_mask = valid_mask.copy()
                                        finger_mask[0] = False 
                                        r_hand[finger_mask, 0] += r_off_x
                                        r_hand[finger_mask, 1] += r_off_y
                                        # Logging
                                        if i % 10 == 0: # Nicht das Log sprengen, alle 10 Frames reicht als Beweis
                                            log_messages.append(f"  -> Frame {i}, Person {p}: Rechte Hand Finger verschoben (X: {r_offset[0]:.2f}px, Y: {r_offset[1]:.2f}px)")

                            if len(pts) > 4 and pts[4] is not None and np.sum(l_hand) > 0.01:
                                wrist_norm = np.array([pts[4][0] / float(width), pts[4][1] / float(height)])
                                gap_offset = np.array([0.0, 0.0])
                                if pts[3] is not None:
                                    dir_vec = np.array([pts[4][0] - pts[3][0], pts[4][1] - pts[3][1]])
                                    norm_vec = np.linalg.norm(dir_vec)
                                    if norm_vec > 0: gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])
                                l_flat = np.array(l_hand[0]).flatten()
                                ox, oy = float((wrist_norm[0] + gap_offset[0]) - l_flat[0]), float((wrist_norm[1] + gap_offset[1]) - l_flat[1])
                                
                                if isinstance(l_hand, np.ndarray):
                                    valid_mask = l_hand[:, 0] > 0
                                    l_hand[valid_mask, 0] += ox
                                    l_hand[valid_mask, 1] += oy
                                    
                                    # --- NEU: Skalierung anwenden ---
                                    if hand_scale_factor != 1.0:
                                        wrist_pos = l_hand[0].copy()
                                        for f_idx in range(1, 21):
                                            if valid_mask[f_idx]:
                                                l_hand[f_idx] = wrist_pos + (l_hand[f_idx] - wrist_pos) * hand_scale_factor
                                    
                                    # Rotations-Offset auf FINGER anwenden
                                    if apply_fingertip_offsets and (abs(l_offset[0]) > 0.001 or abs(l_offset[1]) > 0.001):
                                        l_off_x, l_off_y = l_offset[0] / float(width), l_offset[1] / float(height)
                                        finger_mask = valid_mask.copy()
                                        finger_mask[0] = False 
                                        l_hand[finger_mask, 0] += l_off_x
                                        l_hand[finger_mask, 1] += l_off_y
                                        # Logging
                                        if i % 10 == 0:
                                            log_messages.append(f"  -> Frame {i}, Person {p}: Linke Hand Finger verschoben (X: {l_offset[0]:.2f}px, Y: {l_offset[1]:.2f}px)")

                            if draw_feet and "_posedata_feet" in dw_pose_input[i]:
                                feet_array = dw_pose_input[i]["_posedata_feet"]
                                if len(feet_array) > 0 and pts[10] is not None:
                                    ankle_norm = np.array([pts[10][0] / float(width), pts[10][1] / float(height)])
                                    feet_flat = np.array(feet_array[0]).flatten() / np.array([width, height])
                                    fox, foy = float(ankle_norm[0] - feet_flat[0]), float(ankle_norm[1] - feet_flat[1])

                            dw_hx, dw_hy, dw_nx, dw_ny = None, None, None, None
                            person_subset, candidate = None, None
                            if isinstance(dw_bodies, dict) and "candidate" in dw_bodies and "subset" in dw_bodies:
                                candidate, subset = dw_bodies["candidate"], dw_bodies["subset"]
                                if isinstance(subset, np.ndarray) and subset.ndim == 3 and subset.shape[0] == 1: subset = subset[0]
                                dw_bodies["subset"] = subset
                                if p < len(subset): person_subset = subset[p]
                                
                                nose_idx = int(np.array(person_subset).flatten()[0]) if person_subset is not None else -1
                                if 0 <= nose_idx < len(candidate):
                                    cand_val = np.array(candidate[nose_idx]).flatten()
                                    if len(cand_val) >= 2 and cand_val[0] > 0: dw_hx, dw_hy = float(cand_val[0]), float(cand_val[1])
                                
                                if person_subset is not None and len(np.array(person_subset).flatten()) > 1:
                                    neck_idx = int(np.array(person_subset).flatten()[1])
                                    if 0 <= neck_idx < len(candidate):
                                        cand_val = np.array(candidate[neck_idx]).flatten()
                                        if len(cand_val) >= 2 and cand_val[0] > 0: dw_nx, dw_ny = float(cand_val[0]), float(cand_val[1])

                            if dw_hx is None and p < len(dw_faces):
                                face = dw_faces[p]
                                if isinstance(face, np.ndarray) and len(face) > 30 and face[30, 0] > 0:
                                    dw_hx, dw_hy = float(face[30, 0]), float(face[30, 1])

                            if head_connection_mode == "Offset Head to Neck":
                                if pts[1] is not None and dw_nx is not None and dw_ny is not None:
                                    ox, oy = float((float(pts[1][0]) / float(width)) - dw_nx), float((float(pts[1][1]) / float(height)) - dw_ny)
                                    if person_subset is not None and candidate is not None:
                                        for h_idx in [0, 14, 15, 16, 17, 18, 19, 20]:
                                            if h_idx < len(person_subset):
                                                cand_idx = int(np.array(person_subset).flatten()[h_idx])
                                                if 0 <= cand_idx < len(candidate):
                                                    cand = candidate[cand_idx]
                                                    if isinstance(cand, np.ndarray): cand.flat[0] += ox; cand.flat[1] += oy
                                                    elif isinstance(cand, list):
                                                        if isinstance(cand[0], list): cand[0][0] += ox; cand[0][1] += oy
                                                        else: cand[0] += ox; cand[1] += oy

                                    if p < len(dw_faces):
                                        face = dw_faces[p]
                                        if isinstance(face, np.ndarray):
                                            valid_mask = face[:, 0] > 0
                                            face[valid_mask, 0] += ox; face[valid_mask, 1] += oy
                                        elif isinstance(face, list):
                                            for f_pt in face:
                                                if f_pt[0] > 0: f_pt[0] += ox; f_pt[1] += oy
                                                
                                    pixel_ox, pixel_oy = ox * float(width), oy * float(height)
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None: pts[h_idx][0] += pixel_ox; pts[h_idx][1] += pixel_oy

                            elif head_connection_mode == "Keep Head & Stretch Neck":
                                if pts[0] is not None and dw_hx is not None and dw_hy is not None:
                                    pixel_ox, pixel_oy = (dw_hx * float(width)) - pts[0][0], (dw_hy * float(height)) - pts[0][1]
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None: pts[h_idx][0] += pixel_ox; pts[h_idx][1] += pixel_oy

                    all_frames_pts_for_mask.append(all_pts_2d_with_z)

                    bones_to_draw = []
                    for pts in all_pts_2d_with_z:
                        for limb_idx, limb in enumerate(mimic_limb_seq):
                            start_idx, end_idx = limb[0], limb[1]
                            if pts[start_idx] is not None and pts[end_idx] is not None:
                                pt1, pt2 = pts[start_idx], pts[end_idx]
                                bones_to_draw.append({'pt1': (pt1[0], pt1[1]), 'pt2': (pt2[0], pt2[1]), 'z': (pt1[2] + pt2[2]) / 2.0, 'color': limb_colors_rgb[limb_idx % len(limb_colors_rgb)]})
                                
                    bones_to_draw.extend(nlf_extra_bones)
                    bones_to_draw.sort(key=lambda b: b['z'], reverse=True)
                    
                    for bone in bones_to_draw:
                        x1, y1, x2, y2, color = *bone['pt1'], *bone['pt2'], bone['color']
                        length = math.hypot(x1 - x2, y1 - y2)
                        if length > 0.1:
                            polygon = cv2.ellipse2Poly((int((x1+x2)/2), int((y1+y2)/2)), (int(length / 2), line_thickness), int(math.degrees(math.atan2(y1 - y2, x1 - x2))), 0, 360, 1)
                            cv2.fillConvexPoly(frame_img, polygon, color, lineType=cv2.LINE_AA)
                            
                    frame_img = (frame_img * 0.6).astype(np.uint8)

                    joints_to_draw = []
                    for pts in all_pts_2d_with_z:
                        for j_idx, pt in enumerate(pts):
                            if pt is not None:
                                joints_to_draw.append({'pt': (pt[0], pt[1]), 'z': pt[2], 'color': joint_colors_rgb[j_idx % len(joint_colors_rgb)]})
                                
                    joints_to_draw.extend(nlf_extra_joints)
                    joints_to_draw.sort(key=lambda j: j['z'], reverse=True)
                    
                    for joint in joints_to_draw:
                        x, y = joint['pt']
                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(frame_img, (int(x), int(y)), point_radius, joint['color'], thickness=-1, lineType=cv2.LINE_AA)

                    alpha_channel = np.where(np.any(frame_img > 0, axis=-1), 255, 0).astype(np.uint8)
                    frames_np_rgba.append(np.dstack((frame_img, alpha_channel)))

            if dw_pose_input is not None and draw_2d:
                canvas_2d = draw_pose_to_canvas_np(dw_pose_input, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True, show_face_flag=draw_face, show_hand_flag=draw_hands)
                for i in range(len(frames_np_rgba)):
                    frame_rgba, canvas_img = frames_np_rgba[i], canvas_2d[i]
                    
                    # --- NEU: Alpha-Blending für Hände und Gesicht ---
                    mask_bool = np.any(canvas_img > 0, axis=-1)
                    dimmed_canvas = (canvas_img * hand_face_alpha).astype(np.uint8)
                    
                    # Mischen!
                    frame_rgba[:, :, :3][mask_bool] = dimmed_canvas[mask_bool]
                    frame_rgba[:, :, 3][mask_bool] = 255
                    frames_np_rgba[i] = frame_rgba

            frames_tensor = torch.from_numpy(np.stack(frames_np_rgba, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            if isinstance(scaled_nlf_poses, dict): scaled_nlf_poses['joints3d_nonparam'] = [pose_input]
            else: scaled_nlf_poses = pose_input
                
            node_mappings = json.dumps({"node_name": "RenderNLFPosesDirectPoseDataMimic15", "status": "success", "frames": len(pose_input)})
            
            nlf_data_for_mask = {
                "all_frames_pts": all_frames_pts_for_mask,
                "dw_pose_input": dw_pose_input,
                "width": width,
                "height": height,
                "focal_length": intrinsic_matrix[0, 0]
            }
            
            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), scaled_nlf_poses, node_mappings, nlf_data_for_mask)

        except Exception as e:
            log_messages.append(traceback.format_exc())
            return (torch.zeros((1, height, width, 3)), torch.zeros((1, height, width)), "\n".join(log_messages), nlf_poses, "{}", None)


class NLFDataToMaskV5:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_data_for_mask": ("NLF_MASK_DATA", {"tooltip": "Der Output aus Mimic 14/15/16"}),
                
                # NEU: 3D Skalierung für die Körper-Sticks
                "stick_3d_scale": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 3000.0, "step": 1.0, "tooltip": "3D-Dicke der Knochen (skaliert mit Z-Tiefe)"}),
                
                "head_circle_scale": ("FLOAT", {"default": 150.0, "min": 1.0, "max": 3000.0, "step": 5.0, "tooltip": "3D-Radius (Zentrum: DW-Nase XY, Tiefe: NLF-Hals Z)"}),
                "draw_neck_polygon": ("BOOLEAN", {"default": True, "tooltip": "Verbindet DW-Ohren mit NLF-Schultern"}),
                "draw_body_rectangle": ("BOOLEAN", {"default": True, "tooltip": "Viereck zwischen Schultern und Hüfte"}),
                "draw_hip_circles": ("BOOLEAN", {"default": True, "tooltip": "Kugeln fürs Hinterteil"}),
                "hip_circle_scale": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 5.0, "step": 0.05, "tooltip": "Größe basierend auf der Rumpflänge"}),
                "draw_hands_and_face": ("BOOLEAN", {"default": True}),
                
                # NEU: 3D Skalierung für das Aufblähen der Hände
                "hands_face_dilate_scale": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 1000.0, "step": 1.0, "tooltip": "3D-Skalierung für die Dicke von Händen und Gesicht"}),
                
                "interpolate_missing_frames": ("BOOLEAN", {"default": True, "tooltip": "Füllt Lücken linear auf (Körper, Gesicht & Hände)"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Masken Generator V3 (Vollständiges 3D-Volumen: Sticks und Hände skalieren jetzt mit Z-Tiefe!)."

    def process(self, nlf_data_for_mask, stick_3d_scale, head_circle_scale, draw_neck_polygon, draw_body_rectangle, draw_hip_circles, hip_circle_scale, draw_hands_and_face, hands_face_dilate_scale, interpolate_missing_frames):
        import numpy as np
        import torch
        import cv2
        import math
        import copy
        from ...pose_draw.draw_pose_utils import draw_pose_to_canvas_np

        if not nlf_data_for_mask:
            return (torch.zeros((1, 512, 512)),)

        all_frames_pts = copy.deepcopy(nlf_data_for_mask["all_frames_pts"])
        dw_pose_input = copy.deepcopy(nlf_data_for_mask["dw_pose_input"])
        width = nlf_data_for_mask["width"]
        height = nlf_data_for_mask["height"]
        focal_length = nlf_data_for_mask["focal_length"]

        # ====================================================================
        # LINEARE INTERPOLATION FÜR KÖRPER, HÄNDE UND GESICHT
        # ====================================================================
        if interpolate_missing_frames:
            max_people = max([len(f) for f in all_frames_pts]) if all_frames_pts else 0
            
            # 1. NLF Punkte Interpolieren
            for p in range(max_people):
                for j in range(18):
                    last_valid = -1
                    for i in range(len(all_frames_pts)):
                        valid = False
                        if p < len(all_frames_pts[i]) and len(all_frames_pts[i][p]) > j:
                            if all_frames_pts[i][p][j] is not None:
                                valid = True
                                
                        if valid:
                            if last_valid != -1 and i - last_valid > 1:
                                start_pt = all_frames_pts[last_valid][p][j]
                                end_pt = all_frames_pts[i][p][j]
                                steps = i - last_valid
                                for step in range(1, steps):
                                    frac = step / steps
                                    ix = int(start_pt[0] + (end_pt[0] - start_pt[0]) * frac)
                                    iy = int(start_pt[1] + (end_pt[1] - start_pt[1]) * frac)
                                    iz = start_pt[2] + (end_pt[2] - start_pt[2]) * frac
                                    
                                    while len(all_frames_pts[last_valid+step]) <= p:
                                        all_frames_pts[last_valid+step].append([None]*18)
                                    while len(all_frames_pts[last_valid+step][p]) <= j:
                                        all_frames_pts[last_valid+step][p].append(None)
                                        
                                    all_frames_pts[last_valid+step][p][j] = [ix, iy, iz]
                            last_valid = i

            # 2. DW Pose Interpolieren
            if dw_pose_input is not None:
                for p in range(max_people):
                    for j in range(18):
                        last_valid = -1
                        for i in range(len(dw_pose_input)):
                            cand = dw_pose_input[i].get("bodies", {}).get("candidate", [])
                            valid, pt = False, None
                            if isinstance(cand, list) and p < len(cand) and len(cand[p]) > j and cand[p][j][0] > 0:
                                valid, pt = True, cand[p][j]
                            elif isinstance(cand, np.ndarray) and p < cand.shape[0] and len(cand[p]) > j and cand[p][j][0] > 0:
                                valid, pt = True, cand[p][j]
                            if valid:
                                if last_valid != -1 and i - last_valid > 1:
                                    start_cand = dw_pose_input[last_valid]["bodies"]["candidate"]
                                    start_pt = start_cand[p][j] if isinstance(start_cand, list) else start_cand[p][j]
                                    steps = i - last_valid
                                    for step in range(1, steps):
                                        frac = step / steps
                                        ix = start_pt[0] + (pt[0] - start_pt[0]) * frac
                                        iy = start_pt[1] + (pt[1] - start_pt[1]) * frac
                                        step_cand = dw_pose_input[last_valid+step].get("bodies", {}).get("candidate", [])
                                        if isinstance(step_cand, list) and p < len(step_cand) and len(step_cand[p]) > j: step_cand[p][j] = [ix, iy]
                                        elif isinstance(step_cand, np.ndarray) and p < step_cand.shape[0] and len(step_cand[p]) > j: step_cand[p][j] = [ix, iy]
                                last_valid = i

                    for h_idx in [0, 1]:
                        hand_offset = p * 2 + h_idx
                        last_valid = -1
                        for i in range(len(dw_pose_input)):
                            dw_hands = dw_pose_input[i].get("hands", [])
                            valid = False
                            if len(dw_hands) > hand_offset:
                                h_arr = np.array(dw_hands[hand_offset])
                                if np.sum(np.abs(h_arr)) > 0.01: valid = True
                            if valid:
                                if last_valid != -1 and i - last_valid > 1:
                                    start_hand = np.array(dw_pose_input[last_valid]["hands"][hand_offset])
                                    end_hand = np.array(dw_pose_input[i]["hands"][hand_offset])
                                    steps = i - last_valid
                                    for step in range(1, steps):
                                        frac = step / steps
                                        interp_hand = start_hand + (end_hand - start_hand) * frac
                                        if len(dw_pose_input[last_valid+step].get("hands", [])) <= hand_offset:
                                            if "hands" not in dw_pose_input[last_valid+step]: dw_pose_input[last_valid+step]["hands"] = []
                                            while len(dw_pose_input[last_valid+step]["hands"]) <= hand_offset: dw_pose_input[last_valid+step]["hands"].append(np.zeros_like(start_hand))
                                        dw_pose_input[last_valid+step]["hands"][hand_offset] = interp_hand
                                last_valid = i

                    last_valid = -1
                    for i in range(len(dw_pose_input)):
                        dw_faces = dw_pose_input[i].get("faces", [])
                        valid = False
                        if len(dw_faces) > p:
                            f_arr = np.array(dw_faces[p])
                            if np.sum(np.abs(f_arr)) > 0.01: valid = True
                        if valid:
                            if last_valid != -1 and i - last_valid > 1:
                                start_face = np.array(dw_pose_input[last_valid]["faces"][p])
                                end_face = np.array(dw_pose_input[i]["faces"][p])
                                steps = i - last_valid
                                for step in range(1, steps):
                                    frac = step / steps
                                    interp_face = start_face + (end_face - start_face) * frac
                                    if len(dw_pose_input[last_valid+step].get("faces", [])) <= p:
                                        if "faces" not in dw_pose_input[last_valid+step]: dw_pose_input[last_valid+step]["faces"] = []
                                        while len(dw_pose_input[last_valid+step]["faces"]) <= p: dw_pose_input[last_valid+step]["faces"].append(np.zeros_like(start_face))
                                    dw_pose_input[last_valid+step]["faces"][p] = interp_face
                            last_valid = i
        # ====================================================================

        mimic_limb_seq = [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], 
            [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
        ]

        canvas_2d_frames = None
        if dw_pose_input is not None and draw_hands_and_face:
            canvas_2d_frames = draw_pose_to_canvas_np(dw_pose_input, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True, show_face_flag=True, show_hand_flag=True)

        frames_mask = []

        for i in range(len(all_frames_pts)):
            mask_img = np.zeros((height, width), dtype=np.uint8)
            pts_list = all_frames_pts[i]
            
            # Sammle durchschnittliche Z-Tiefe dieses Frames (für Hände/Gesichts-Skalierung)
            frame_z_depths = []
            
            candidate = []
            if dw_pose_input is not None and i < len(dw_pose_input):
                dw_bodies = dw_pose_input[i].get("bodies", {})
                candidate = dw_bodies.get("candidate", [])

            for p, pts in enumerate(pts_list):
                if pts[0] is not None:
                    frame_z_depths.append(abs(pts[0][2]))
                
                cand = None
                if isinstance(candidate, list) and p < len(candidate): cand = candidate[p]
                elif isinstance(candidate, np.ndarray) and p < candidate.shape[0]: cand = candidate[p]

                nose, r_ear, l_ear = None, None, None
                if cand is not None and len(cand) >= 18:
                    if cand[0][0] > 0: nose = (int(cand[0][0]*width), int(cand[0][1]*height))
                    if cand[16][0] > 0: r_ear = (int(cand[16][0]*width), int(cand[16][1]*height))
                    if cand[17][0] > 0: l_ear = (int(cand[17][0]*width), int(cand[17][1]*height))

                # HYBRIDE 3D KOPF KUGEL
                center_xy = nose
                if center_xy is None and pts[0] is not None: center_xy = (pts[0][0], pts[0][1])
                if head_circle_scale > 0 and center_xy is not None and pts[0] is not None:
                    z_depth = max(0.1, abs(pts[0][2]))
                    pixel_r = max(2, int((head_circle_scale * focal_length) / z_depth))
                    # FIX: explizites Casting zu int
                    cv2.circle(mask_img, (int(center_xy[0]), int(center_xy[1])), int(pixel_r), 255, -1, lineType=cv2.LINE_AA)

                # Hals Polygon
                if draw_neck_polygon and pts[2] is not None and pts[5] is not None:
                    # FIX: explizites Casting zu int
                    r_shoulder, l_shoulder = (int(pts[2][0]), int(pts[2][1])), (int(pts[5][0]), int(pts[5][1]))
                    poly_pts = []
                    if r_ear: poly_pts.append((int(r_ear[0]), int(r_ear[1])))
                    elif nose: poly_pts.append((int(nose[0]), int(nose[1])))
                    if l_ear: poly_pts.append((int(l_ear[0]), int(l_ear[1])))
                    elif nose and not poly_pts: poly_pts.append((int(nose[0]), int(nose[1])))
                    poly_pts.extend([l_shoulder, r_shoulder])
                    if len(poly_pts) >= 3:
                        # FIX: dtype=np.int32 erzwingen
                        cv2.fillPoly(mask_img, [np.array(poly_pts, dtype=np.int32)], 255)

                # Rumpf Viereck
                if draw_body_rectangle:
                    if pts[2] is not None and pts[5] is not None and pts[11] is not None and pts[8] is not None:
                        # FIX: dtype=np.int32 erzwingen
                        rect_cnt = np.array([
                            [pts[2][0], pts[2][1]], 
                            [pts[5][0], pts[5][1]], 
                            [pts[11][0], pts[11][1]], 
                            [pts[8][0], pts[8][1]]
                        ], dtype=np.int32)
                        cv2.fillPoly(mask_img, [rect_cnt], 255)

                # Hüft Kreise
                if draw_hip_circles and pts[8] is not None and pts[11] is not None and pts[2] is not None and pts[5] is not None:
                    dist_r = math.hypot(pts[2][0] - pts[8][0], pts[2][1] - pts[8][1])
                    dist_l = math.hypot(pts[5][0] - pts[11][0], pts[5][1] - pts[11][1])
                    torso_len = (dist_r + dist_l) / 2.0
                    pixel_r = max(2, int(torso_len * hip_circle_scale))
                    # FIX: explizites Casting zu int
                    cv2.circle(mask_img, (int(pts[8][0]), int(pts[8][1])), int(pixel_r), 255, -1, lineType=cv2.LINE_AA)
                    cv2.circle(mask_img, (int(pts[11][0]), int(pts[11][1])), int(pixel_r), 255, -1, lineType=cv2.LINE_AA)

                # --- NEU: Körper Sticks in echtem 3D ---
                for limb in mimic_limb_seq:
                    if pts[limb[0]] is not None and pts[limb[1]] is not None:
                        pt1, pt2 = pts[limb[0]], pts[limb[1]]
                        
                        # Durchschnittliche Z-Tiefe dieses spezifischen Knochens
                        limb_z_depth = max(0.1, (abs(pt1[2]) + abs(pt2[2])) / 2.0)
                        
                        # 3D Pixel-Dicke berechnen
                        thickness = max(1, int((stick_3d_scale * focal_length) / limb_z_depth))
                        
                        # FIX: explizites Casting zu int für beide Punkte und die Dicke
                        cv2.line(mask_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), 255, int(thickness), lineType=cv2.LINE_AA)

            # --- NEU: Dicke Hände & Gesicht in echtem 3D ---
            if canvas_2d_frames is not None and i < len(canvas_2d_frames):
                canvas_img = canvas_2d_frames[i]
                hf_mask = np.where(np.any(canvas_img > 0, axis=-1), 255, 0).astype(np.uint8)
                
                if hands_face_dilate_scale > 0:
                    # Nimm die durchschnittliche Z-Tiefe der Figuren in diesem Frame (Fallback 1000)
                    avg_z_depth = sum(frame_z_depths) / len(frame_z_depths) if frame_z_depths else 1000.0
                    avg_z_depth = max(0.1, avg_z_depth)
                    
                    # 3D Skalierung der Dilatation
                    dilate_size = max(1, int((hands_face_dilate_scale * focal_length) / avg_z_depth))
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
                    hf_mask = cv2.dilate(hf_mask, kernel, iterations=1)
                    
                mask_img = np.maximum(mask_img, hf_mask)

            frames_mask.append(mask_img)

        mask_tensor = torch.from_numpy(np.stack(frames_mask, axis=0)).float() / 255.0
        return (mask_tensor,)


class RenderNLFPosesDirectPoseDataMimic17:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die originalen NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Dicke der Knochen (Ovale Form)"}),
                "point_radius": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Größe der Gelenkpunkte"}),
                "head_connection_mode": (["Offset Head to Neck", "Keep Head & Stretch Neck"], {"default": "Offset Head to Neck", "tooltip": "Wie der Kopf an den Hals angebunden wird"}),
                "draw_2d": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet 2D Overlay"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet das Gesicht"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet die Hände"}),
                
                # POSEDATA TOGGLES
                "use_pose_data": ("BOOLEAN", {"default": True, "tooltip": "Nutzt PoseData statt DW Poses für Hände/Füße"}),
                "use_dwpose_head_for_posedata": ("BOOLEAN", {"default": True, "tooltip": "Nimmt KOMPLETTEN Kopf & Gesicht von DW Pose, auch wenn PoseData an ist"}),
                "draw_feet": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet Füße von PoseData und mappt sie an die NLF-Knöchel"}),
                "draw_nlf_feet": ("BOOLEAN", {"default": False, "tooltip": "Zeichnet Füße direkt aus originalen NLF-Daten (überschreibt PoseData-Füße)"}),
                
                # NEU: Hände Tweaks (Skalierung, Alpha, Offsets)
                "apply_fingertip_offsets": ("BOOLEAN", {"default": True, "tooltip": "Wendet die Rotations-Offsets auf die Finger an (falls Input vorhanden)"}),
                "hand_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Skaliert die Hände (1.0 = normal)"}),
                "hand_face_alpha": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Deckkraft für 2D Hände und Gesicht (0.6 passt gut zum Körper)"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht als Fallback"}),
                "pose_data_fallback": ("POSEDATA", {"tooltip": "Pose Data (z.B. ViTPose) für Hände/Füße"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
                "fingertip_offsets": ("STRING", {"forceInput": True, "tooltip": "Die JSON-Offsets aus der HandDebug-Node"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING", "NLF_MASK_DATA")
    RETURN_NAMES = ("image", "mask", "log_output", "scaled_nlf_poses", "node_mappings", "nlf_data_for_mask")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Mimic 17 mit gefixter PoseData Skalierung (nutzt meta.width/height) + Offsets, Logging, NLF Feet."

    def process(self, nlf_poses, width, height, line_thickness=4, point_radius=4, head_connection_mode="Offset Head to Neck", draw_2d=True, draw_face=True, draw_hands=True, use_pose_data=True, use_dwpose_head_for_posedata=True, draw_feet=True, draw_nlf_feet=False, apply_fingertip_offsets=True, hand_scale_factor=1.0, hand_face_alpha=0.6, dw_poses_fallback=None, pose_data_fallback=None, nlf_render_config="{}", fingertip_offsets=None):
        import copy
        import json
        import math
        import torch
        import numpy as np
        import traceback
        import cv2
        from ...NLFPoseExtract.nlf_render_flat import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_single_p2d
        from ...pose_draw.draw_pose_utils import draw_pose_to_canvas_np

        log_messages = ["=== RENDER NLF POSES MIMIC 17 LOG ==="]
        scaled_nlf_poses = copy.deepcopy(nlf_poses)
        
        # Offsets Parsen
        offsets_dict = {}
        if apply_fingertip_offsets and fingertip_offsets and fingertip_offsets.strip() != "":
            try:
                offsets_dict = json.loads(fingertip_offsets)
                log_messages.append("Fingertip Offsets erfolgreich geladen und aktiviert.")
            except Exception as e:
                log_messages.append(f"Fehler beim Parsen der fingertip_offsets: {e}")
        elif not apply_fingertip_offsets:
            log_messages.append("Fingertip Offsets sind per Toggle deaktiviert.")
                
        # NLF Füsse überschreiben PoseData Füsse
        if draw_nlf_feet:
            draw_feet = False
        
        try:
            pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
            
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            if dw_pose_input is None and use_pose_data:
                dw_pose_input = [{"bodies": {"candidate": [np.zeros((18, 2))], "subset": [np.full(18, -1)]}, "hands": np.zeros((2, 21, 2)), "faces": [np.zeros((68, 2))]} for _ in range(len(pose_input))]
            
            pose_metas = []
            if use_pose_data and pose_data_fallback is not None:
                pose_metas = pose_data_fallback.get("pose_metas", [])

            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])

            # 3D Kamera Config Baking
            try:
                config = json.loads(nlf_render_config)
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    p_x, p_y = float(config["pivot_x"]), float(config["pivot_y"])
                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x *= width
                        p_y *= height
                    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
                    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
                    M13 = (cx - p_x) * (scale_x - 1.0) / fx
                    M23 = (cy - p_y) * (scale_y - 1.0) / fy
                    
                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None and len(pose_input[frame_idx]) > 0:
                            pts = pose_input[frame_idx]
                            X, Y, Z = pts[..., 0].clone(), pts[..., 1].clone(), pts[..., 2].clone()
                            pts[..., 0] = X * scale_x + Z * M13
                            pts[..., 1] = Y * scale_y + Z * M23
            except Exception as e:
                log_messages.append(f"Fehler bei 3D Transformation: {e}")

            # POSEDATA IN DW-STRUKTUR INJIZIEREN
            if use_pose_data and pose_metas:
                for p_idx in range(min(len(dw_pose_input), len(pose_metas))):
                    meta = pose_metas[p_idx]
                    dw = dw_pose_input[p_idx]
                    cand = dw["bodies"]["candidate"][0] if isinstance(dw["bodies"]["candidate"], list) else dw["bodies"]["candidate"][0]
                    subset = dw["bodies"]["subset"][0] if isinstance(dw["bodies"]["subset"], list) else dw["bodies"]["subset"][0]
                    
                    # FIX: Wir extrahieren meta.width und meta.height direkt aus den PoseData Metadaten!
                    meta_w = getattr(meta, "width", width)
                    meta_h = getattr(meta, "height", height)

                    if draw_hands:
                        lh = getattr(meta, "kps_lhand", None)
                        rh = getattr(meta, "kps_rhand", None)
                        if lh is not None and len(lh) >= 21: dw["hands"][0] = np.array(lh[:, :2]) / np.array([meta_w, meta_h])
                        if rh is not None and len(rh) >= 21: dw["hands"][1] = np.array(rh[:, :2]) / np.array([meta_w, meta_h])

                    if not use_dwpose_head_for_posedata:
                        coco_to_op = {0: 0, 1: 15, 2: 14, 3: 17, 4: 16}
                        if getattr(meta, "kps_body", None) is not None:
                            body_pts = meta.kps_body
                            for coco_idx, op_idx in coco_to_op.items():
                                if coco_idx < len(body_pts) and body_pts[coco_idx][0] > 0:
                                    cand[op_idx] = [body_pts[coco_idx][0] / meta_w, body_pts[coco_idx][1] / meta_h]
                                    subset[op_idx] = op_idx
                        if draw_face:
                            face_pts = getattr(meta, "kps_face", None)
                            if face_pts is not None and len(face_pts) > 1:
                                dw["faces"][0] = np.array(face_pts[1:, :2]) / np.array([meta_w, meta_h])

                    if draw_feet and getattr(meta, "kps_body", None) is not None:
                        feet_pts = []
                        for f_idx in [19, 20, 21, 22, 23, 24]:
                            if f_idx < len(meta.kps_body) and meta.kps_body[f_idx][0] > 0: feet_pts.append(meta.kps_body[f_idx][:2])
                        if len(feet_pts) > 0:
                            dw["_posedata_feet"] = np.array(feet_pts) / np.array([meta_w, meta_h])
                        else:
                            dw["_posedata_feet"] = np.array([])

            limb_colors_rgb = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)]
            joint_colors_rgb = limb_colors_rgb + [(255, 0, 85)]
            mimic_limb_seq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]

            frames_np_rgba = []
            all_frames_pts_for_mask = []

            for i in range(len(pose_input)):
                frame_img = np.zeros((height, width, 3), dtype=np.uint8)
                nlf_extra_bones = []
                nlf_extra_joints = []
                
                if pose_input[i] is not None:
                    joints3d_batch = pose_input[i]
                    people = joints3d_batch if joints3d_batch.dim() == 3 else [joints3d_batch] if joints3d_batch.dim() == 2 else []

                    all_pts_2d_with_z = []
                    for joints3d in people:
                        j3d_np = joints3d.cpu().numpy() if isinstance(joints3d, torch.Tensor) else joints3d
                        
                        # --- NEU: NLF Füße projizieren (mit exakten Farben) ---
                        if draw_nlf_feet and j3d_np.shape[0] >= 12:
                            # 7->10 ist Links (Rot), 8->11 ist Rechts (Gelb)
                            for s_idx, e_idx, foot_color in [(7, 10, (255, 0, 0)), (8, 11, (255, 255, 0))]: 
                                if np.sum(np.abs(j3d_np[s_idx])) > 0 and np.sum(np.abs(j3d_np[e_idx])) > 0:
                                    p1 = p3d_single_p2d(j3d_np[s_idx], intrinsic_matrix)
                                    p2 = p3d_single_p2d(j3d_np[e_idx], intrinsic_matrix)
                                    nlf_extra_bones.append({
                                        'pt1': (p1[0], p1[1]), 'pt2': (p2[0], p2[1]), 
                                        'z': (j3d_np[s_idx][2] + j3d_np[e_idx][2]) / 2.0, 
                                        'color': foot_color
                                    })
                                    nlf_extra_joints.append({
                                        'pt': (p2[0], p2[1]), 'z': j3d_np[e_idx][2], 
                                        'color': foot_color
                                    })

                        if np.sum(np.abs(j3d_np)) > 0.01:
                            j3d_coco = process_data_to_COCO_format(j3d_np)
                            pts_2d_with_z = []
                            for pt3d in j3d_coco:
                                if np.sum(np.abs(pt3d)) > 0:
                                    pt2d = p3d_single_p2d(pt3d, intrinsic_matrix)
                                    pts_2d_with_z.append([int(pt2d[0]), int(pt2d[1]), float(pt3d[2])])
                                else:
                                    pts_2d_with_z.append(None)
                                    
                            if len(pts_2d_with_z) > 5 and pts_2d_with_z[2] is not None and pts_2d_with_z[5] is not None:
                                p_r, p_l = pts_2d_with_z[2], pts_2d_with_z[5]
                                if pts_2d_with_z[1] is not None:
                                    p_neck = pts_2d_with_z[1]
                                    pts_2d_with_z[1][0] = int((p_r[0] + p_l[0]) / 2)
                                    pts_2d_with_z[1][1] = int((p_r[1] + p_l[1]) / 2)

                            all_pts_2d_with_z.append(pts_2d_with_z)

                    if dw_pose_input is not None and i < len(dw_pose_input):
                        dw_faces = dw_pose_input[i].get("faces", [])
                        dw_hands = dw_pose_input[i].get("hands", [])
                        dw_bodies = dw_pose_input[i].get("bodies", {})
                        
                        for p, pts in enumerate(all_pts_2d_with_z):
                            if p >= len(dw_hands) // 2: continue
                            r_hand = dw_hands[p*2]
                            l_hand = dw_hands[p*2+1]
                            
                            l_offset = [0.0, 0.0]
                            r_offset = [0.0, 0.0]
                            if apply_fingertip_offsets:
                                f_idx_str = str(i)
                                p_idx_str = str(p)
                                if f_idx_str in offsets_dict and p_idx_str in offsets_dict[f_idx_str]:
                                    l_offset = offsets_dict[f_idx_str][p_idx_str].get("left_hand", [0.0, 0.0])
                                    r_offset = offsets_dict[f_idx_str][p_idx_str].get("right_hand", [0.0, 0.0])

                            if len(pts) > 7 and pts[7] is not None and np.sum(r_hand) > 0.01:
                                wrist_norm = np.array([pts[7][0] / float(width), pts[7][1] / float(height)])
                                gap_offset = np.array([0.0, 0.0])
                                if pts[6] is not None:
                                    dir_vec = np.array([pts[7][0] - pts[6][0], pts[7][1] - pts[6][1]])
                                    norm_vec = np.linalg.norm(dir_vec)
                                    if norm_vec > 0: gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])
                                r_flat = np.array(r_hand[0]).flatten()
                                ox, oy = float((wrist_norm[0] + gap_offset[0]) - r_flat[0]), float((wrist_norm[1] + gap_offset[1]) - r_flat[1])
                                
                                if isinstance(r_hand, np.ndarray):
                                    valid_mask = r_hand[:, 0] > 0
                                    r_hand[valid_mask, 0] += ox
                                    r_hand[valid_mask, 1] += oy
                                    
                                    # --- NEU: Skalierung anwenden ---
                                    if hand_scale_factor != 1.0:
                                        wrist_pos = r_hand[0].copy()
                                        for f_idx in range(1, 21):
                                            if valid_mask[f_idx]:
                                                r_hand[f_idx] = wrist_pos + (r_hand[f_idx] - wrist_pos) * hand_scale_factor
                                    
                                    # Rotations-Offset auf FINGER anwenden
                                    if apply_fingertip_offsets and (abs(r_offset[0]) > 0.001 or abs(r_offset[1]) > 0.001):
                                        r_off_x, r_off_y = r_offset[0] / float(width), r_offset[1] / float(height)
                                        finger_mask = valid_mask.copy()
                                        finger_mask[0] = False 
                                        r_hand[finger_mask, 0] += r_off_x
                                        r_hand[finger_mask, 1] += r_off_y
                                        # Logging
                                        if i % 10 == 0: # Nicht das Log sprengen, alle 10 Frames reicht als Beweis
                                            log_messages.append(f"  -> Frame {i}, Person {p}: Rechte Hand Finger verschoben (X: {r_offset[0]:.2f}px, Y: {r_offset[1]:.2f}px)")

                            if len(pts) > 4 and pts[4] is not None and np.sum(l_hand) > 0.01:
                                wrist_norm = np.array([pts[4][0] / float(width), pts[4][1] / float(height)])
                                gap_offset = np.array([0.0, 0.0])
                                if pts[3] is not None:
                                    dir_vec = np.array([pts[4][0] - pts[3][0], pts[4][1] - pts[3][1]])
                                    norm_vec = np.linalg.norm(dir_vec)
                                    if norm_vec > 0: gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])
                                l_flat = np.array(l_hand[0]).flatten()
                                ox, oy = float((wrist_norm[0] + gap_offset[0]) - l_flat[0]), float((wrist_norm[1] + gap_offset[1]) - l_flat[1])
                                
                                if isinstance(l_hand, np.ndarray):
                                    valid_mask = l_hand[:, 0] > 0
                                    l_hand[valid_mask, 0] += ox
                                    l_hand[valid_mask, 1] += oy
                                    
                                    # --- NEU: Skalierung anwenden ---
                                    if hand_scale_factor != 1.0:
                                        wrist_pos = l_hand[0].copy()
                                        for f_idx in range(1, 21):
                                            if valid_mask[f_idx]:
                                                l_hand[f_idx] = wrist_pos + (l_hand[f_idx] - wrist_pos) * hand_scale_factor
                                    
                                    # Rotations-Offset auf FINGER anwenden
                                    if apply_fingertip_offsets and (abs(l_offset[0]) > 0.001 or abs(l_offset[1]) > 0.001):
                                        l_off_x, l_off_y = l_offset[0] / float(width), l_offset[1] / float(height)
                                        finger_mask = valid_mask.copy()
                                        finger_mask[0] = False 
                                        l_hand[finger_mask, 0] += l_off_x
                                        l_hand[finger_mask, 1] += l_off_y
                                        # Logging
                                        if i % 10 == 0:
                                            log_messages.append(f"  -> Frame {i}, Person {p}: Linke Hand Finger verschoben (X: {l_offset[0]:.2f}px, Y: {l_offset[1]:.2f}px)")

                            if draw_feet and "_posedata_feet" in dw_pose_input[i]:
                                feet_array = dw_pose_input[i]["_posedata_feet"]
                                if len(feet_array) > 0 and pts[10] is not None:
                                    ankle_norm = np.array([pts[10][0] / float(width), pts[10][1] / float(height)])
                                    feet_flat = np.array(feet_array[0]).flatten() # FIX: Hier wird NICHT nochmal durch width/height geteilt, weil schon normalisiert!
                                    fox, foy = float(ankle_norm[0] - feet_flat[0]), float(ankle_norm[1] - feet_flat[1])

                            dw_hx, dw_hy, dw_nx, dw_ny = None, None, None, None
                            person_subset, candidate = None, None
                            if isinstance(dw_bodies, dict) and "candidate" in dw_bodies and "subset" in dw_bodies:
                                candidate, subset = dw_bodies["candidate"], dw_bodies["subset"]
                                if isinstance(subset, np.ndarray) and subset.ndim == 3 and subset.shape[0] == 1: subset = subset[0]
                                dw_bodies["subset"] = subset
                                if p < len(subset): person_subset = subset[p]
                                
                                nose_idx = int(np.array(person_subset).flatten()[0]) if person_subset is not None else -1
                                if 0 <= nose_idx < len(candidate):
                                    cand_val = np.array(candidate[nose_idx]).flatten()
                                    if len(cand_val) >= 2 and cand_val[0] > 0: dw_hx, dw_hy = float(cand_val[0]), float(cand_val[1])
                                
                                if person_subset is not None and len(np.array(person_subset).flatten()) > 1:
                                    neck_idx = int(np.array(person_subset).flatten()[1])
                                    if 0 <= neck_idx < len(candidate):
                                        cand_val = np.array(candidate[neck_idx]).flatten()
                                        if len(cand_val) >= 2 and cand_val[0] > 0: dw_nx, dw_ny = float(cand_val[0]), float(cand_val[1])

                            if dw_hx is None and p < len(dw_faces):
                                face = dw_faces[p]
                                if isinstance(face, np.ndarray) and len(face) > 30 and face[30, 0] > 0:
                                    dw_hx, dw_hy = float(face[30, 0]), float(face[30, 1])

                            if head_connection_mode == "Offset Head to Neck":
                                if pts[1] is not None and dw_nx is not None and dw_ny is not None:
                                    ox, oy = float((float(pts[1][0]) / float(width)) - dw_nx), float((float(pts[1][1]) / float(height)) - dw_ny)
                                    if person_subset is not None and candidate is not None:
                                        for h_idx in [0, 14, 15, 16, 17, 18, 19, 20]:
                                            if h_idx < len(person_subset):
                                                cand_idx = int(np.array(person_subset).flatten()[h_idx])
                                                if 0 <= cand_idx < len(candidate):
                                                    cand = candidate[cand_idx]
                                                    if isinstance(cand, np.ndarray): cand.flat[0] += ox; cand.flat[1] += oy
                                                    elif isinstance(cand, list):
                                                        if isinstance(cand[0], list): cand[0][0] += ox; cand[0][1] += oy
                                                        else: cand[0] += ox; cand[1] += oy

                                    if p < len(dw_faces):
                                        face = dw_faces[p]
                                        if isinstance(face, np.ndarray):
                                            valid_mask = face[:, 0] > 0
                                            face[valid_mask, 0] += ox; face[valid_mask, 1] += oy
                                        elif isinstance(face, list):
                                            for f_pt in face:
                                                if f_pt[0] > 0: f_pt[0] += ox; f_pt[1] += oy
                                                
                                    pixel_ox, pixel_oy = ox * float(width), oy * float(height)
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None: pts[h_idx][0] += pixel_ox; pts[h_idx][1] += pixel_oy

                            elif head_connection_mode == "Keep Head & Stretch Neck":
                                if pts[0] is not None and dw_hx is not None and dw_hy is not None:
                                    pixel_ox, pixel_oy = (dw_hx * float(width)) - pts[0][0], (dw_hy * float(height)) - pts[0][1]
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None: pts[h_idx][0] += pixel_ox; pts[h_idx][1] += pixel_oy

                    all_frames_pts_for_mask.append(all_pts_2d_with_z)

                    bones_to_draw = []
                    for pts in all_pts_2d_with_z:
                        for limb_idx, limb in enumerate(mimic_limb_seq):
                            start_idx, end_idx = limb[0], limb[1]
                            if pts[start_idx] is not None and pts[end_idx] is not None:
                                pt1, pt2 = pts[start_idx], pts[end_idx]
                                bones_to_draw.append({'pt1': (pt1[0], pt1[1]), 'pt2': (pt2[0], pt2[1]), 'z': (pt1[2] + pt2[2]) / 2.0, 'color': limb_colors_rgb[limb_idx % len(limb_colors_rgb)]})
                                
                    bones_to_draw.extend(nlf_extra_bones)
                    bones_to_draw.sort(key=lambda b: b['z'], reverse=True)
                    
                    for bone in bones_to_draw:
                        x1, y1, x2, y2, color = *bone['pt1'], *bone['pt2'], bone['color']
                        length = math.hypot(x1 - x2, y1 - y2)
                        if length > 0.1:
                            polygon = cv2.ellipse2Poly((int((x1+x2)/2), int((y1+y2)/2)), (int(length / 2), line_thickness), int(math.degrees(math.atan2(y1 - y2, x1 - x2))), 0, 360, 1)
                            cv2.fillConvexPoly(frame_img, polygon, color, lineType=cv2.LINE_AA)
                            
                    frame_img = (frame_img * 0.6).astype(np.uint8)

                    joints_to_draw = []
                    for pts in all_pts_2d_with_z:
                        for j_idx, pt in enumerate(pts):
                            if pt is not None:
                                joints_to_draw.append({'pt': (pt[0], pt[1]), 'z': pt[2], 'color': joint_colors_rgb[j_idx % len(joint_colors_rgb)]})
                                
                    joints_to_draw.extend(nlf_extra_joints)
                    joints_to_draw.sort(key=lambda j: j['z'], reverse=True)
                    
                    for joint in joints_to_draw:
                        x, y = joint['pt']
                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(frame_img, (int(x), int(y)), point_radius, joint['color'], thickness=-1, lineType=cv2.LINE_AA)

                    alpha_channel = np.where(np.any(frame_img > 0, axis=-1), 255, 0).astype(np.uint8)
                    frames_np_rgba.append(np.dstack((frame_img, alpha_channel)))

            if dw_pose_input is not None and draw_2d:
                canvas_2d = draw_pose_to_canvas_np(dw_pose_input, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True, show_face_flag=draw_face, show_hand_flag=draw_hands)
                for i in range(len(frames_np_rgba)):
                    frame_rgba, canvas_img = frames_np_rgba[i], canvas_2d[i]
                    
                    # --- NEU: Alpha-Blending für Hände und Gesicht ---
                    mask_bool = np.any(canvas_img > 0, axis=-1)
                    dimmed_canvas = (canvas_img * hand_face_alpha).astype(np.uint8)
                    
                    # Mischen!
                    frame_rgba[:, :, :3][mask_bool] = dimmed_canvas[mask_bool]
                    frame_rgba[:, :, 3][mask_bool] = 255
                    frames_np_rgba[i] = frame_rgba

            frames_tensor = torch.from_numpy(np.stack(frames_np_rgba, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            if isinstance(scaled_nlf_poses, dict): scaled_nlf_poses['joints3d_nonparam'] = [pose_input]
            else: scaled_nlf_poses = pose_input
                
            node_mappings = json.dumps({"node_name": "RenderNLFPosesDirectPoseDataMimic17", "status": "success", "frames": len(pose_input)})
            
            nlf_data_for_mask = {
                "all_frames_pts": all_frames_pts_for_mask,
                "dw_pose_input": dw_pose_input,
                "width": width,
                "height": height,
                "focal_length": intrinsic_matrix[0, 0]
            }
            
            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), scaled_nlf_poses, node_mappings, nlf_data_for_mask)

        except Exception as e:
            log_messages.append(traceback.format_exc())
            return (torch.zeros((1, height, width, 3)), torch.zeros((1, height, width)), "\n".join(log_messages), nlf_poses, "{}", None)


class RenderNLFPosesDirectHybrid8:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die originalen NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
                "head_connection_mode": (["Offset Head to Neck", "Keep Head & Stretch Neck"], {"default": "Offset Head to Neck", "tooltip": "Wie der Kopf an den Hals angebunden wird"}),
                "draw_2d": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet 2D Overlay (falls DW Poses vorhanden)"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet das Gesicht"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet die Hände"}),
                
                # POSEDATA TOGGLES
                "use_pose_data": ("BOOLEAN", {"default": True, "tooltip": "Nutzt PoseData statt DW Poses für Hände/Füße"}),
                "use_dwpose_head_for_posedata": ("BOOLEAN", {"default": True, "tooltip": "Nimmt KOMPLETTEN Kopf & Gesicht von DW Pose, auch wenn PoseData an ist"}),
                "draw_feet": ("BOOLEAN", {"default": True, "tooltip": "Gibt Füße von PoseData an Mask_Data weiter"}),
                
                # Hände Tweaks (Skalierung, Alpha, Offsets)
                "apply_fingertip_offsets": ("BOOLEAN", {"default": True, "tooltip": "Wendet die Rotations-Offsets auf die Finger an"}),
                "hand_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Skaliert die Hände (1.0 = normal)"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Referenz-Posen für Hände/Gesicht"}),
                "pose_data_fallback": ("POSEDATA", {"tooltip": "Pose Data (z.B. ViTPose) für Hände/Füße"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
                "fingertip_offsets": ("STRING", {"forceInput": True, "tooltip": "Die JSON-Offsets aus der HandDebug-Node"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING", "NLF_MASK_DATA")
    RETURN_NAMES = ("image", "mask", "log_output", "scaled_nlf_poses", "node_mappings", "nlf_data_for_mask")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Hybrid 8: Direct 3D Render Optik (Taichi) + PoseData & Offset Präzision (aus Mimic) + NLF Mask Data Output."

    def process(self, nlf_poses, width, height, render_backend="taichi", head_connection_mode="Offset Head to Neck", draw_2d=True, draw_face=True, draw_hands=True, use_pose_data=True, use_dwpose_head_for_posedata=True, draw_feet=True, apply_fingertip_offsets=True, hand_scale_factor=1.0, dw_poses_fallback=None, pose_data_fallback=None, nlf_render_config="{}", fingertip_offsets=None):
        import copy
        import json
        import torch
        import numpy as np
        import traceback
        from ...NLFPoseExtract.nlf_render import render_multi_nlf_as_images, render_nlf_as_images, intrinsic_matrix_from_field_of_view
        from ...NLFPoseExtract.nlf_render_flat import process_data_to_COCO_format, p3d_single_p2d

        log_messages = ["=== RENDER NLF POSES DIRECT HYBRID 8 LOG ==="]
        
        if render_backend == "taichi":
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
                log_messages.append("Render-Backend: Taichi GPU initialisiert.")
            except Exception as e:
                render_backend = "torch"
                log_messages.append(f"WARNUNG: Taichi GPU fehlgeschlagen. Nutze Torch. Fehler: {e}")
        else:
            log_messages.append("Render-Backend: Torch.")

        scaled_nlf_poses = copy.deepcopy(nlf_poses)
        
        # Offsets Parsen
        offsets_dict = {}
        if apply_fingertip_offsets and fingertip_offsets and fingertip_offsets.strip() != "":
            try:
                offsets_dict = json.loads(fingertip_offsets)
                log_messages.append("Fingertip Offsets erfolgreich geladen und aktiviert.")
            except Exception as e:
                log_messages.append(f"Fehler beim Parsen der fingertip_offsets: {e}")
        elif not apply_fingertip_offsets:
            log_messages.append("Fingertip Offsets sind per Toggle deaktiviert.")

        try:
            pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
            
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            if dw_pose_input is None and use_pose_data:
                dw_pose_input = [{"bodies": {"candidate": [np.zeros((18, 2))], "subset": [np.full(18, -1)]}, "hands": np.zeros((2, 21, 2)), "faces": [np.zeros((68, 2))]} for _ in range(len(pose_input))]
            
            pose_metas = []
            if use_pose_data and pose_data_fallback is not None:
                pose_metas = pose_data_fallback.get("pose_metas", [])

            if len(pose_input) > 0 and pose_input[0] is not None:
                log_messages.append(f"Erfolgreich geladen: {len(pose_input)} Frames.")

            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
            
            # 1. CONFIG MATHEMATISCH AUF DIE 3D PUNKTE ANWENDEN
            try:
                config = json.loads(nlf_render_config)
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    
                    p_x = float(config["pivot_x"])
                    p_y = float(config["pivot_y"])

                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x = p_x * width
                        p_y = p_y * height
                    
                    fx = intrinsic_matrix[0, 0]
                    fy = intrinsic_matrix[1, 1]
                    cx = intrinsic_matrix[0, 2]
                    cy = intrinsic_matrix[1, 2]

                    M13 = (cx - p_x) * (scale_x - 1.0) / fx
                    M23 = (cy - p_y) * (scale_y - 1.0) / fy

                    log_messages.append(f"Wende 3D-Transformation an: ScaleX={scale_x:.3f}, ScaleY={scale_y:.3f}")

                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None and len(pose_input[frame_idx]) > 0:
                            pts = pose_input[frame_idx]
                            X = pts[..., 0].clone()
                            Y = pts[..., 1].clone()
                            Z = pts[..., 2].clone()
                            
                            pts[..., 0] = X * scale_x + Z * M13
                            pts[..., 1] = Y * scale_y + Z * M23
                            
                    log_messages.append("Erfolg: Die 3D-NLF-Daten wurden physisch im Raum transformiert!")
            except Exception as e:
                log_messages.append(f"Fehler bei 3D Transformation: {e}")

            # 2. POSEDATA IN DW-STRUKTUR INJIZIEREN
            if use_pose_data and pose_metas:
                for p_idx in range(min(len(dw_pose_input), len(pose_metas))):
                    meta = pose_metas[p_idx]
                    dw = dw_pose_input[p_idx]
                    cand = dw["bodies"]["candidate"][0] if isinstance(dw["bodies"]["candidate"], list) else dw["bodies"]["candidate"][0]
                    subset = dw["bodies"]["subset"][0] if isinstance(dw["bodies"]["subset"], list) else dw["bodies"]["subset"][0]
                    
                    meta_w = getattr(meta, "width", width)
                    meta_h = getattr(meta, "height", height)

                    if draw_hands:
                        lh = getattr(meta, "kps_lhand", None)
                        rh = getattr(meta, "kps_rhand", None)
                        if lh is not None and len(lh) >= 21: dw["hands"][0] = np.array(lh[:, :2]) / np.array([meta_w, meta_h])
                        if rh is not None and len(rh) >= 21: dw["hands"][1] = np.array(rh[:, :2]) / np.array([meta_w, meta_h])

                    if not use_dwpose_head_for_posedata:
                        coco_to_op = {0: 0, 1: 15, 2: 14, 3: 17, 4: 16}
                        if getattr(meta, "kps_body", None) is not None:
                            body_pts = meta.kps_body
                            for coco_idx, op_idx in coco_to_op.items():
                                if coco_idx < len(body_pts) and body_pts[coco_idx][0] > 0:
                                    cand[op_idx] = [body_pts[coco_idx][0] / meta_w, body_pts[coco_idx][1] / meta_h]
                                    subset[op_idx] = op_idx
                        if draw_face:
                            face_pts = getattr(meta, "kps_face", None)
                            if face_pts is not None and len(face_pts) > 1:
                                dw["faces"][0] = np.array(face_pts[1:, :2]) / np.array([meta_w, meta_h])

                    if draw_feet and getattr(meta, "kps_body", None) is not None:
                        feet_pts = []
                        for f_idx in [19, 20, 21, 22, 23, 24]:
                            if f_idx < len(meta.kps_body) and meta.kps_body[f_idx][0] > 0: feet_pts.append(meta.kps_body[f_idx][:2])
                        if len(feet_pts) > 0:
                            dw["_posedata_feet"] = np.array(feet_pts) / np.array([meta_w, meta_h])
                        else:
                            dw["_posedata_feet"] = np.array([])

            # 3. 2D BERECHNUNG FÜR OFFSETS UND MASKEN ANPASSUNG
            all_frames_pts_for_mask = []
            for i in range(len(pose_input)):
                if pose_input[i] is not None:
                    joints3d_batch = pose_input[i]
                    people = joints3d_batch if joints3d_batch.dim() == 3 else [joints3d_batch] if joints3d_batch.dim() == 2 else []

                    all_pts_2d_with_z = []
                    for joints3d in people:
                        j3d_np = joints3d.cpu().numpy() if isinstance(joints3d, torch.Tensor) else joints3d
                        if np.sum(np.abs(j3d_np)) > 0.01:
                            j3d_coco = process_data_to_COCO_format(j3d_np)
                            pts_2d_with_z = []
                            for pt3d in j3d_coco:
                                if np.sum(np.abs(pt3d)) > 0:
                                    pt2d = p3d_single_p2d(pt3d, intrinsic_matrix)
                                    pts_2d_with_z.append([int(pt2d[0]), int(pt2d[1]), float(pt3d[2])])
                                else:
                                    pts_2d_with_z.append(None)
                                    
                            if len(pts_2d_with_z) > 5 and pts_2d_with_z[2] is not None and pts_2d_with_z[5] is not None:
                                p_r, p_l = pts_2d_with_z[2], pts_2d_with_z[5]
                                if pts_2d_with_z[1] is not None:
                                    p_neck = pts_2d_with_z[1]
                                    pts_2d_with_z[1][0] = int((p_r[0] + p_l[0]) / 2)
                                    pts_2d_with_z[1][1] = int((p_r[1] + p_l[1]) / 2)

                            all_pts_2d_with_z.append(pts_2d_with_z)

                    if dw_pose_input is not None and i < len(dw_pose_input):
                        dw_faces = dw_pose_input[i].get("faces", [])
                        dw_hands = dw_pose_input[i].get("hands", [])
                        dw_bodies = dw_pose_input[i].get("bodies", {})
                        
                        for p, pts in enumerate(all_pts_2d_with_z):
                            if p >= len(dw_hands) // 2: continue
                            r_hand = dw_hands[p*2]
                            l_hand = dw_hands[p*2+1]
                            
                            l_offset = [0.0, 0.0]
                            r_offset = [0.0, 0.0]
                            if apply_fingertip_offsets:
                                f_idx_str = str(i)
                                p_idx_str = str(p)
                                if f_idx_str in offsets_dict and p_idx_str in offsets_dict[f_idx_str]:
                                    l_offset = offsets_dict[f_idx_str][p_idx_str].get("left_hand", [0.0, 0.0])
                                    r_offset = offsets_dict[f_idx_str][p_idx_str].get("right_hand", [0.0, 0.0])

                            if len(pts) > 7 and pts[7] is not None and np.sum(r_hand) > 0.01:
                                wrist_norm = np.array([pts[7][0] / float(width), pts[7][1] / float(height)])
                                gap_offset = np.array([0.0, 0.0])
                                if pts[6] is not None:
                                    dir_vec = np.array([pts[7][0] - pts[6][0], pts[7][1] - pts[6][1]])
                                    norm_vec = np.linalg.norm(dir_vec)
                                    if norm_vec > 0: gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])
                                r_flat = np.array(r_hand[0]).flatten()
                                ox, oy = float((wrist_norm[0] + gap_offset[0]) - r_flat[0]), float((wrist_norm[1] + gap_offset[1]) - r_flat[1])
                                
                                if isinstance(r_hand, np.ndarray):
                                    valid_mask = r_hand[:, 0] > 0
                                    r_hand[valid_mask, 0] += ox
                                    r_hand[valid_mask, 1] += oy
                                    
                                    if hand_scale_factor != 1.0:
                                        wrist_pos = r_hand[0].copy()
                                        for f_idx in range(1, 21):
                                            if valid_mask[f_idx]:
                                                r_hand[f_idx] = wrist_pos + (r_hand[f_idx] - wrist_pos) * hand_scale_factor
                                    
                                    if apply_fingertip_offsets and (abs(r_offset[0]) > 0.001 or abs(r_offset[1]) > 0.001):
                                        r_off_x, r_off_y = r_offset[0] / float(width), r_offset[1] / float(height)
                                        finger_mask = valid_mask.copy()
                                        finger_mask[0] = False 
                                        r_hand[finger_mask, 0] += r_off_x
                                        r_hand[finger_mask, 1] += r_off_y

                            if len(pts) > 4 and pts[4] is not None and np.sum(l_hand) > 0.01:
                                wrist_norm = np.array([pts[4][0] / float(width), pts[4][1] / float(height)])
                                gap_offset = np.array([0.0, 0.0])
                                if pts[3] is not None:
                                    dir_vec = np.array([pts[4][0] - pts[3][0], pts[4][1] - pts[3][1]])
                                    norm_vec = np.linalg.norm(dir_vec)
                                    if norm_vec > 0: gap_offset = (dir_vec / norm_vec) * 4.0 / np.array([width, height])
                                l_flat = np.array(l_hand[0]).flatten()
                                ox, oy = float((wrist_norm[0] + gap_offset[0]) - l_flat[0]), float((wrist_norm[1] + gap_offset[1]) - l_flat[1])
                                
                                if isinstance(l_hand, np.ndarray):
                                    valid_mask = l_hand[:, 0] > 0
                                    l_hand[valid_mask, 0] += ox
                                    l_hand[valid_mask, 1] += oy
                                    
                                    if hand_scale_factor != 1.0:
                                        wrist_pos = l_hand[0].copy()
                                        for f_idx in range(1, 21):
                                            if valid_mask[f_idx]:
                                                l_hand[f_idx] = wrist_pos + (l_hand[f_idx] - wrist_pos) * hand_scale_factor
                                    
                                    if apply_fingertip_offsets and (abs(l_offset[0]) > 0.001 or abs(l_offset[1]) > 0.001):
                                        l_off_x, l_off_y = l_offset[0] / float(width), l_offset[1] / float(height)
                                        finger_mask = valid_mask.copy()
                                        finger_mask[0] = False 
                                        l_hand[finger_mask, 0] += l_off_x
                                        l_hand[finger_mask, 1] += l_off_y

                            if draw_feet and "_posedata_feet" in dw_pose_input[i]:
                                feet_array = dw_pose_input[i]["_posedata_feet"]
                                if len(feet_array) > 0 and pts[10] is not None:
                                    ankle_norm = np.array([pts[10][0] / float(width), pts[10][1] / float(height)])
                                    feet_flat = np.array(feet_array[0]).flatten()
                                    fox, foy = float(ankle_norm[0] - feet_flat[0]), float(ankle_norm[1] - feet_flat[1])

                            dw_hx, dw_hy, dw_nx, dw_ny = None, None, None, None
                            person_subset, candidate = None, None
                            if isinstance(dw_bodies, dict) and "candidate" in dw_bodies and "subset" in dw_bodies:
                                candidate, subset = dw_bodies["candidate"], dw_bodies["subset"]
                                if isinstance(subset, np.ndarray) and subset.ndim == 3 and subset.shape[0] == 1: subset = subset[0]
                                dw_bodies["subset"] = subset
                                if p < len(subset): person_subset = subset[p]
                                
                                nose_idx = int(np.array(person_subset).flatten()[0]) if person_subset is not None else -1
                                if 0 <= nose_idx < len(candidate):
                                    cand_val = np.array(candidate[nose_idx]).flatten()
                                    if len(cand_val) >= 2 and cand_val[0] > 0: dw_hx, dw_hy = float(cand_val[0]), float(cand_val[1])
                                
                                if person_subset is not None and len(np.array(person_subset).flatten()) > 1:
                                    neck_idx = int(np.array(person_subset).flatten()[1])
                                    if 0 <= neck_idx < len(candidate):
                                        cand_val = np.array(candidate[neck_idx]).flatten()
                                        if len(cand_val) >= 2 and cand_val[0] > 0: dw_nx, dw_ny = float(cand_val[0]), float(cand_val[1])

                            if dw_hx is None and p < len(dw_faces):
                                face = dw_faces[p]
                                if isinstance(face, np.ndarray) and len(face) > 30 and face[30, 0] > 0:
                                    dw_hx, dw_hy = float(face[30, 0]), float(face[30, 1])

                            if head_connection_mode == "Offset Head to Neck":
                                if pts[1] is not None and dw_nx is not None and dw_ny is not None:
                                    ox, oy = float((float(pts[1][0]) / float(width)) - dw_nx), float((float(pts[1][1]) / float(height)) - dw_ny)
                                    if person_subset is not None and candidate is not None:
                                        for h_idx in [0, 14, 15, 16, 17, 18, 19, 20]:
                                            if h_idx < len(person_subset):
                                                cand_idx = int(np.array(person_subset).flatten()[h_idx])
                                                if 0 <= cand_idx < len(candidate):
                                                    cand = candidate[cand_idx]
                                                    if isinstance(cand, np.ndarray): cand.flat[0] += ox; cand.flat[1] += oy
                                                    elif isinstance(cand, list):
                                                        if isinstance(cand[0], list): cand[0][0] += ox; cand[0][1] += oy
                                                        else: cand[0] += ox; cand[1] += oy

                                    if p < len(dw_faces):
                                        face = dw_faces[p]
                                        if isinstance(face, np.ndarray):
                                            valid_mask = face[:, 0] > 0
                                            face[valid_mask, 0] += ox; face[valid_mask, 1] += oy
                                        elif isinstance(face, list):
                                            for f_pt in face:
                                                if f_pt[0] > 0: f_pt[0] += ox; f_pt[1] += oy
                                                
                                    pixel_ox, pixel_oy = ox * float(width), oy * float(height)
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None: pts[h_idx][0] += pixel_ox; pts[h_idx][1] += pixel_oy

                            elif head_connection_mode == "Keep Head & Stretch Neck":
                                if pts[0] is not None and dw_hx is not None and dw_hy is not None:
                                    pixel_ox, pixel_oy = (dw_hx * float(width)) - pts[0][0], (dw_hy * float(height)) - pts[0][1]
                                    for h_idx in [0, 14, 15, 16, 17]:
                                        if pts[h_idx] is not None: pts[h_idx][0] += pixel_ox; pts[h_idx][1] += pixel_oy

                    all_frames_pts_for_mask.append(all_pts_2d_with_z)

            # 4. RENDERN MIT TAICHI (Mit den nun modifizierten DW Daten!)
            log_messages.append(f"Rendere Settings -> 2D: {draw_2d} | Face: {draw_face} | Hands: {draw_hands}")
            
            is_multi = False
            if len(pose_input) > 0:
                if isinstance(pose_input[0], list):
                    is_multi = len(pose_input[0]) > 1
                elif isinstance(pose_input[0], torch.Tensor) and pose_input[0].dim() == 3:
                    is_multi = pose_input[0].shape[0] > 1
                    
            if is_multi:
                frames_np = render_multi_nlf_as_images(
                    pose_input, dw_pose_input, height, width, len(pose_input), 
                    intrinsic_matrix=intrinsic_matrix, 
                    draw_2d=draw_2d, draw_face=draw_face, draw_hands=draw_hands, 
                    render_backend=render_backend
                )
            else:
                frames_np = render_nlf_as_images(
                    pose_input, dw_pose_input, height, width, len(pose_input), 
                    intrinsic_matrix=intrinsic_matrix, 
                    draw_2d=draw_2d, draw_face=draw_face, draw_hands=draw_hands, 
                    render_backend=render_backend
                )

            frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            if isinstance(scaled_nlf_poses, dict): scaled_nlf_poses['joints3d_nonparam'] = [pose_input]
            else: scaled_nlf_poses = pose_input
                
            node_mappings = json.dumps({"node_name": "RenderNLFPosesDirectHybrid8", "status": "success", "frames": len(pose_input)})
            
            nlf_data_for_mask = {
                "all_frames_pts": all_frames_pts_for_mask,
                "dw_pose_input": dw_pose_input,
                "width": width,
                "height": height,
                "focal_length": intrinsic_matrix[0, 0]
            }
            
            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), scaled_nlf_poses, node_mappings, nlf_data_for_mask)

        except Exception as e:
            log_messages.append(traceback.format_exc())
            empty_img = torch.zeros((1, height, width, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            return (empty_img, empty_mask, "\n".join(log_messages), nlf_poses, "{}", None)


