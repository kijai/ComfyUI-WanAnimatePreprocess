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



class SavePoseCalibration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Daten aus dem Retarget Pose Calibrator"}),
                "folder_name": ("STRING", {"default": "pose_calibrations", "tooltip": "Unterordner im ComfyUI/output Verzeichnis"}),
                "filename_prefix": ("STRING", {"default": "my_retarget_profile", "tooltip": "Name der Datei"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True 
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "Speichert das Kalibrierungs-Profil im Output-Ordner ab."

    def save(self, calibration_data, folder_name, filename_prefix):
        # 1. Zielordner im "output" Verzeichnis erstellen
        output_dir = folder_paths.get_output_directory()
        save_dir = os.path.join(output_dir, folder_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 2. Freien Dateinamen finden (z.B. profil_0001.json)
        counter = 1
        file_path = os.path.join(save_dir, f"{filename_prefix}_{counter:04d}.json")
        while os.path.exists(file_path):
            counter += 1
            file_path = os.path.join(save_dir, f"{filename_prefix}_{counter:04d}.json")

        # 3. Abspeichern
        with open(file_path, 'w') as f:
            json.dump(calibration_data, f, indent=4)
            
        print(f"[SavePoseCalibration] Gespeichert in: {file_path}")
        return ()


class LoadPoseCalibration:
    @classmethod
    def INPUT_TYPES(cls):
        # Wir definieren den Standard-Input-Ordner
        input_dir = folder_paths.get_input_directory()
        default_load_dir = os.path.join(input_dir, "pose_calibrations")
        
        # Ordner erstellen, falls er noch nicht existiert, damit ComfyUI nicht crasht
        if not os.path.exists(default_load_dir):
            os.makedirs(default_load_dir)
            
        # Dateien für das Dropdown-Menü sammeln
        files = [f for f in os.listdir(default_load_dir) if f.endswith('.json')]
        if not files:
            files = ["Keine_Dateien_gefunden.json"]

        return {
            "required": {
                "folder_name": ("STRING", {"default": "pose_calibrations", "tooltip": "Unterordner im ComfyUI/input Verzeichnis"}),
                "filename": (files, {"tooltip": "Wähle dein Profil (Muss im Input-Ordner liegen!)"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION",)
    RETURN_NAMES = ("calibration_data",)
    FUNCTION = "load"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "Lädt ein Profil aus dem Input-Ordner."

    def load(self, folder_name, filename):
        if filename == "Keine_Dateien_gefunden.json":
            print("[LoadPoseCalibration] Warnung: Keine Datei ausgewählt!")
            return ({},)
            
        # Pfad zusammenbauen
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, folder_name, filename)
        
        if not os.path.exists(file_path):
            print(f"[LoadPoseCalibration] Fehler: Datei nicht gefunden: {file_path}")
            return ({},)

        # JSON Laden
        with open(file_path, 'r') as f:
            calibration_data = json.load(f)
            
        print(f"[LoadPoseCalibration] Profil '{filename}' geladen!")
        return (calibration_data,)


class PoseLocalBoneRetargeterV10:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scaled_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_nlf_data": ("NLFPRED",),
            }
        }

    # HIER NEU: Wir fügen "STRING" als zweiten Output hinzu!
    RETURN_TYPES = ("POSEDATA", "STRING",)
    RETURN_NAMES = ("final_pose_data", "log_output",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V10B: Nutzt 3D-Längen zum Skalieren und gibt ein Log aus, wenn NLF fehlt."

    def process(self, scaled_pose_data, calibration_data, video_nlf_data):
        pose_data_copy = copy.deepcopy(scaled_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        
        log_messages = [] # Hier sammeln wir alle Fehlermeldungen für die Anzeige
        
        target_3d_bones = calibration_data.get("true_3d_bones", {})
        if not target_3d_bones:
            msg = "[V10B] FEHLER: Keine 3D Bone Längen in Calibration Data. Überspringe lokales Scaling."
            print(msg)
            return (pose_data_copy, msg) # Text über den zweiten Output ausgeben
            
        pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) else video_nlf_data

        def dist_3d(p1, p2):
            return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

        for i, meta in enumerate(pose_metas):
            if i >= len(pose_input_3d): 
                log_messages.append(f"Frame {i}: NLF-Daten enden hier (Video länger als 3D-Daten).")
                break
                
            kps_2d = getattr(meta, "kps_body", None)
            
            # --- LOGGING: Prüfen ob NLF existiert ---
            if pose_input_3d[i] is None or len(pose_input_3d[i]) == 0: 
                log_messages.append(f"Frame {i}: NLF hat keine Person erkannt (Leeres Array).")
                continue
                
            pose_3d = pose_input_3d[i][0]
            
            # --- LOGGING: Prüfen ob genug Keypoints da sind ---
            if kps_2d is None or len(kps_2d) == 0:
                log_messages.append(f"Frame {i}: Keine 2D-Keypoints im Originalvideo gefunden.")
                continue
            if len(pose_3d) < 14:
                log_messages.append(f"Frame {i}: NLF 3D-Skelett ist unvollständig (< 14 Keypoints).")
                continue 
            
            # (Die eigentliche Skalierungs-Mathematik bleibt komplett gleich)
            src_3d_bones = {
                "torso": dist_3d(pose_3d[1], pose_3d[8]),
                "r_thigh": dist_3d(pose_3d[8], pose_3d[9]),
                "r_calf": dist_3d(pose_3d[9], pose_3d[10]),
                "l_thigh": dist_3d(pose_3d[11], pose_3d[12]),
                "l_calf": dist_3d(pose_3d[12], pose_3d[13])
            }
            
            scales = {k: (target_3d_bones[k] / src_3d_bones[k] if src_3d_bones[k] > 0 else 1.0) for k in src_3d_bones}
            
            hip_center = [(kps_2d[8][0]+kps_2d[11][0])/2, (kps_2d[8][1]+kps_2d[11][1])/2] if kps_2d[8][1] > 0 and kps_2d[11][1] > 0 else None
            
            if hip_center and "torso" in scales:
                s_torso = scales["torso"]
                upper_indices = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]
                for idx in upper_indices:
                    if idx < len(kps_2d) and kps_2d[idx][1] > 0:
                        kps_2d[idx][0] = hip_center[0] + (kps_2d[idx][0] - hip_center[0]) * s_torso
                        kps_2d[idx][1] = hip_center[1] + (kps_2d[idx][1] - hip_center[1]) * s_torso
                        
                for attr_name in ["kps_lhand", "kps_rhand", "kps_face"]:
                    arr = getattr(meta, attr_name, None)
                    if arr is not None and len(arr) > 0: 
                        for j in range(len(arr)):
                            if len(arr[j]) >= 2 and arr[j][1] > 0:
                                arr[j][0] = hip_center[0] + (arr[j][0] - hip_center[0]) * s_torso
                                arr[j][1] = hip_center[1] + (arr[j][1] - hip_center[1]) * s_torso

            def scale_bone(start_idx, end_idx, scale_factor):
                if kps_2d[start_idx][1] > 0 and kps_2d[end_idx][1] > 0:
                    vec_x = kps_2d[end_idx][0] - kps_2d[start_idx][0]
                    vec_y = kps_2d[end_idx][1] - kps_2d[start_idx][1]
                    kps_2d[end_idx][0] = kps_2d[start_idx][0] + (vec_x * scale_factor)
                    kps_2d[end_idx][1] = kps_2d[start_idx][1] + (vec_y * scale_factor)

            scale_bone(8, 9, scales["r_thigh"])
            scale_bone(9, 10, scales["r_calf"])
            scale_bone(11, 12, scales["l_thigh"])
            scale_bone(12, 13, scales["l_calf"])

        # --- LOGGING: Abschluss-Bericht erstellen ---
        if len(log_messages) == 0:
            final_log = "ERFOLG: Alle Frames wurden fehlerfrei mit 3D-NLF-Daten skaliert."
        else:
            final_log = f"WARNUNGEN ({len(log_messages)} Fehler gefunden):\n" + "\n".join(log_messages)

        # Gibt jetzt ZWEI Sachen zurück: Die Posen-Daten und den Text!
        return (pose_data_copy, final_log)


class PoseGlobalPerspectiveScalerV30:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data_scaled")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"

    def process(self, video_pose_data, calibration_data, video_depth_map, include_head, anchor_window, min_confidence, frontal_method="3D_NLF", frontal_2d_threshold=0.65, frontal_3d_angle_tolerance=20.0, video_nlf_data=None):
        import copy
        import numpy as np
        import math
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V30 GLOBAL SCALER LOG ==="]

        if not pose_metas:
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data)

        # --- V30 Lese Logik für reparierte 3D-Längen ---
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if "calf" in true_3d_bones:
            log_messages.append(f"V30 hat die reparierten 3D-Längen aus V18 erfolgreich verstanden.")
            log_messages.append(f"Soll-Wade: {true_3d_bones['calf']:.3f} (Faktor: {true_3d_bones.get('factor_nah_fern', 1.0):.3f}x)")
        else:
            log_messages.append("Keine reparierten 3D-Längen im Calibration Data gefunden.")

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        slope = calibration_data.get("perspective_slope", 0.0)
        intercept = calibration_data.get("perspective_intercept", 1.0)
        is_inverted = calibration_data.get("is_depth_inverted", False)
        norm_method = calibration_data.get("norm_method", "Dynamic Full-Body")
        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        echte_groesse = calibration_data.get("echte_groesse", 1.75)
        fx = calibration_data.get("focal_length_fx", 500.0)

        # --- 2D FRAME AUSWAHL UND SKALIERUNG ---
        def is_valid_point(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else (float(pt[2]) if len(pt)>=3 else 1.0)
            if c < min_confidence: return False
            return True

        def is_frontal_2d(meta):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            if is_valid_point(kps, confs, 2) and is_valid_point(kps, confs, 5) and is_valid_point(kps, confs, 8) and is_valid_point(kps, confs, 11):
                shoulder_w = abs(kps[2][0] - kps[5][0])
                hip_w = abs(kps[8][0] - kps[11][0])
                if hip_w > 0:
                    ratio = shoulder_w / hip_w
                    return ratio > frontal_2d_threshold
            return False

        def is_frontal_3d(frame_idx):
            if video_nlf_data is None: return False
            pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0]
            if pose_input_3d is None or len(pose_input_3d) <= frame_idx or pose_input_3d[frame_idx] is None or len(pose_input_3d[frame_idx]) == 0:
                return False
            pose_3d = pose_input_3d[frame_idx][0]
            if len(pose_3d) > 11:
                l_hip = pose_3d[11]
                r_hip = pose_3d[8]
                dz = abs(l_hip[2] - r_hip[2])
                dx = abs(l_hip[0] - r_hip[0])
                angle = math.degrees(math.atan2(dz, dx)) if dx > 0 else 90.0
                return angle < frontal_3d_angle_tolerance
            return False

        best_idx = 0
        best_area = 0.0

        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            
            if not valid_y or not valid_x: continue
            
            is_front = False
            if frontal_method == "3D_NLF" and video_nlf_data is not None:
                is_front = is_frontal_3d(i)
            else:
                is_front = is_frontal_2d(meta)
                
            if not is_front: continue
            
            area = (max(valid_x) - min(valid_x)) * (max(valid_y) - min(valid_y))
            if area > best_area:
                best_area = area
                best_idx = i

        start_idx = max(0, best_idx - anchor_window)
        end_idx = min(len(pose_metas), best_idx + anchor_window + 1)

        sum_norm = 0.0
        sum_depth = 0.0
        valid_frames_in_window = 0

        for i in range(start_idx, end_idx):
            meta = pose_metas[i]
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            
            norm_val = 0.0
            if norm_method == "Torso (Neck-Hip)":
                if is_valid_point(kps, confs, 1) and is_valid_point(kps, confs, 8) and is_valid_point(kps, confs, 11):
                    mid_x = (kps[8][0] + kps[11][0]) / 2.0
                    mid_y = (kps[8][1] + kps[11][1]) / 2.0
                    norm_val = math.sqrt((kps[1][0] - mid_x)**2 + (kps[1][1] - mid_y)**2)
            else:
                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
                valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
                if valid_y and valid_x:
                    norm_val = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)
            
            if norm_val <= 0: continue

            v_idx = min(i, depth_np.shape[0]-1)
            valid_x_d = [kps[idx][0] * W for idx in [1,8,11] if is_valid_point(kps, confs, idx)]
            valid_y_d = [kps[idx][1] * H for idx in [1,8,11] if is_valid_point(kps, confs, idx)]
            
            depth_vals = []
            for px, py in zip(valid_x_d, valid_y_d):
                ix, iy = int(px), int(py)
                if 0 <= ix < W and 0 <= iy < H:
                    depth_vals.append(depth_np[v_idx, iy, ix])
                    
            frame_depth = float(np.mean(depth_vals)) if depth_vals else 0.5
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            sum_norm += norm_val
            sum_depth += frame_depth
            valid_frames_in_window += 1

        if valid_frames_in_window == 0:
            return (pose_data_copy, "Fehler: Im Anchor-Window konnte keine Tiefe/Norm ermittelt werden.", video_nlf_data)

        avg_anchor_norm = sum_norm / valid_frames_in_window
        avg_anchor_depth = sum_depth / valid_frames_in_window

        if use_pinhole_math and echte_groesse > 0.0:
            expected_norm = (echte_groesse * fx) / avg_anchor_depth
        else:
            expected_norm = (avg_anchor_depth * slope) + intercept

        anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0

        log_messages.append(f"\nSkalierungs-Faktor = {expected_norm:.1f} / {avg_anchor_norm:.1f} = {anchor_scale:.3f}x")

        # 1. 2D Daten Skalieren
        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            
            if not valid_y or not valid_x: continue
            
            pivot_y, pivot_x = max(valid_y), np.mean(valid_x)
            
            for attr_name in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr_name, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = pivot_x + (arr[j][0] - pivot_x) * anchor_scale
                            arr[j][1] = pivot_y + (arr[j][1] - pivot_y) * anchor_scale

        # --- 3D NLF DATEN ROBUST SKALIEREN & FÜSSE ANKNÜPFEN ---
        nlf_data_scaled = None
        if video_nlf_data is not None:
            try:
                # Nutze deepcopy statt clone(), da es eine reine Python-Liste sein kann
                nlf_data_scaled = copy.deepcopy(video_nlf_data)
                is_dict = isinstance(nlf_data_scaled, dict)
                pose_input_3d = nlf_data_scaled.get('joints3d_nonparam', [nlf_data_scaled])[0] if is_dict else nlf_data_scaled
                
                for frame_idx in range(len(pose_input_3d)):
                    if pose_input_3d[frame_idx] is None or len(pose_input_3d[frame_idx]) == 0:
                        continue
                    
                    # 1. NLF mit Anchor Scale multiplizieren (Liste oder Tensor robust handhaben)
                    if isinstance(pose_input_3d[frame_idx], list):
                        for i in range(len(pose_input_3d[frame_idx])):
                            pose_input_3d[frame_idx][i] *= anchor_scale
                        frame_data_ref = pose_input_3d[frame_idx][0]
                    else:
                        pose_input_3d[frame_idx] *= anchor_scale
                        frame_data_ref = pose_input_3d[frame_idx][0] if len(pose_input_3d[frame_idx]) > 0 else pose_input_3d[frame_idx]

                    # Daten für Tensorerstellung ermitteln (damit es nicht abstürzt)
                    device_ref = frame_data_ref.device if hasattr(frame_data_ref, 'device') else torch.device('cpu')
                    dtype_ref = frame_data_ref.dtype if hasattr(frame_data_ref, 'dtype') else torch.float32
                    
                    # 2. Füße aus den 2D Daten holen
                    if frame_idx < len(pose_metas):
                        meta = pose_metas[frame_idx]
                        kps_2d = getattr(meta, "kps_body", [])
                        
                        extra_feet = []
                        if len(kps_2d) > 20: 
                            # Linker Zeh (Index 18)
                            if (kps_2d[18][2] if len(kps_2d[18])>2 else 1) > min_confidence:
                                z_left_ankle = frame_data_ref[13][2] if len(frame_data_ref) > 13 else 0.0
                                extra_feet.append([kps_2d[18][0], kps_2d[18][1], float(z_left_ankle)])
                            
                            # Rechter Zeh (Index 19)
                            if (kps_2d[19][2] if len(kps_2d[19])>2 else 1) > min_confidence:
                                z_right_ankle = frame_data_ref[10][2] if len(frame_data_ref) > 10 else 0.0
                                extra_feet.append([kps_2d[19][0], kps_2d[19][1], float(z_right_ankle)])
                        
                        # Neue Füße ans 3D Array dranhängen
                        if extra_feet:
                            feet_tensor = torch.tensor(extra_feet, dtype=dtype_ref, device=device_ref).unsqueeze(0)
                            if isinstance(pose_input_3d[frame_idx], list):
                                pose_input_3d[frame_idx][0] = torch.cat((pose_input_3d[frame_idx][0], feet_tensor), dim=1)
                            else:
                                pose_input_3d[frame_idx] = torch.cat((pose_input_3d[frame_idx], feet_tensor), dim=1)

                log_messages.append(f"Erfolgreich: 3D NLF Daten fehlerfrei skaliert und Füße angeknüpft.")
            except Exception as e:
                log_messages.append(f"Fehler bei der NLF 3D-Skalierung: {e}")
                nlf_data_scaled = video_nlf_data

        return (pose_data_copy, "\n".join(log_messages), nlf_data_scaled)


class PoseGlobalPerspectiveScalerV38:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
            }
        }

    # NEU: Wir geben einen neuen STRING namens nlf_render_config aus!
    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"

    def process(self, video_pose_data, calibration_data, video_depth_map, include_head, anchor_window, min_confidence, frontal_method="3D_NLF", frontal_2d_threshold=0.65, frontal_3d_angle_tolerance=20.0, scale_2d_axes="X and Y (Uniform)", video_nlf_data=None):
        import copy
        import numpy as np
        import math
        import json

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V38 GLOBAL SCALER LOG ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        slope = calibration_data.get("perspective_slope", 0.0)
        intercept = calibration_data.get("perspective_intercept", 1.0)
        is_inverted = calibration_data.get("is_depth_inverted", False)
        norm_method = calibration_data.get("norm_method", "Dynamic Full-Body")
        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        echte_groesse = calibration_data.get("echte_groesse", 1.75)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        
        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        def is_valid_point(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else (float(pt[2]) if len(pt)>=3 else 1.0)
            return c >= min_confidence

        def is_frontal_3d(frame_idx):
            if video_nlf_data is None: return False
            pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0]
            if pose_input_3d is None or len(pose_input_3d) <= frame_idx or pose_input_3d[frame_idx] is None or len(pose_input_3d[frame_idx]) == 0: return False
            pose_3d = pose_input_3d[frame_idx][0]
            if len(pose_3d) > 11:
                l_hip, r_hip = pose_3d[11], pose_3d[8]
                dz, dx = abs(l_hip[2] - r_hip[2]), abs(l_hip[0] - r_hip[0])
                return (math.degrees(math.atan2(dz, dx)) if dx > 0 else 90.0) < frontal_3d_angle_tolerance
            return False

        # --- Frame Evaluierung ---
        best_idx, best_area = 0, 0.0
        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            if not valid_y or not valid_x: continue
            if frontal_method == "3D_NLF" and video_nlf_data is not None and not is_frontal_3d(i): continue
            
            area = (max(valid_x) - min(valid_x)) * (max(valid_y) - min(valid_y))
            if area > best_area:
                best_area, best_idx = area, i

        start_idx = max(0, best_idx - anchor_window)
        end_idx = min(len(pose_metas), best_idx + anchor_window + 1)
        sum_norm, sum_depth, valid_frames_in_window = 0.0, 0.0, 0

        for i in range(start_idx, end_idx):
            meta = pose_metas[i]
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            if not valid_y or not valid_x: continue
            norm_val = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)
            
            v_idx = min(i, depth_np.shape[0]-1)
            valid_x_d = [kps[idx][0] * W for idx in [1,8,11] if is_valid_point(kps, confs, idx)]
            valid_y_d = [kps[idx][1] * H for idx in [1,8,11] if is_valid_point(kps, confs, idx)]
            depth_vals = [depth_np[v_idx, int(py), int(px)] for px, py in zip(valid_x_d, valid_y_d) if 0 <= int(px) < W and 0 <= int(py) < H]
            
            frame_depth = float(np.mean(depth_vals)) if depth_vals else 0.5
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            sum_norm += norm_val
            sum_depth += frame_depth
            valid_frames_in_window += 1

        if valid_frames_in_window == 0: 
            return (pose_data_copy, "Fehler: Anchor-Window ungültig.", video_nlf_data, "{}")

        avg_anchor_norm = sum_norm / valid_frames_in_window
        avg_anchor_depth = sum_depth / valid_frames_in_window

        if use_pinhole_math and echte_groesse > 0.0:
            expected_norm = (echte_groesse * fx_calib) / avg_anchor_depth
        else:
            expected_norm = (avg_anchor_depth * slope) + intercept

        anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0
        scale_x_factor = anchor_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0
        log_messages.append(f"Skalierungs-Faktor = {anchor_scale:.3f}x")

        # --- Wir berechnen den GLOBALEN KAMERA PIVOT aus dem Anchor Frame ---
        global_pivot_x, global_pivot_y = 0.5, 0.5
        if best_idx < len(pose_metas):
            kps_best = getattr(pose_metas[best_idx], "kps_body", [])
            c_best = getattr(pose_metas[best_idx], "kps_body_p", None)
            val_y = [kps_best[idx][1] for idx in range(len(kps_best)) if is_valid_point(kps_best, c_best, idx)]
            val_x = [kps_best[idx][0] for idx in range(len(kps_best)) if is_valid_point(kps_best, c_best, idx)]
            if val_y and val_x:
                global_pivot_x = np.mean(val_x)
                global_pivot_y = max(val_y) # Füße im Anchor Frame

        # --- 1. 2D Daten Skalieren (relativ zum globalen Kamera Pivot) ---
        for i, meta in enumerate(pose_metas):
            for attr_name in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr_name, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x_factor
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * anchor_scale

        # --- 2. CONFIGURATION DATA BAUEN ---
        nlf_render_config = {
            "anchor_scale": float(anchor_scale),
            "scale_x_factor": float(scale_x_factor),
            "pivot_x": float(global_pivot_x),
            "pivot_y": float(global_pivot_y)
        }
        config_str = json.dumps(nlf_render_config)
        
        log_messages.append("\n=== NLF 3D DATA DELEGATION LOG ===")
        log_messages.append("Genialer Plan aktiv: 3D-Daten bleiben UNVERÄNDERT. Die Kamera-Anweisungen wurden in nlf_render_config verpackt!")

        # Wir reichen die originalen, unbeschädigten NLF-Daten weiter!
        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseCalibrationV20:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_nah": ("IMAGE",),
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_fern": ("IMAGE",),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "intrinsics_json": ("STRING", {"forceInput": True, "tooltip": "Intrinsics JSON"}),
                "nlf_data_nah": ("NLFPRED", {"tooltip": "3D NLF Daten Nah"}),
                "nlf_data_fern": ("NLFPRED", {"tooltip": "3D NLF Daten Fern"}),
                "config_data": ("STRING", {"default": "{}", "tooltip": "JSON Config"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V20: Extrahiert ALLE 3D-Knochenlängen und repariert beide Waden über das Fern-Bild."

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, intrinsics_json=None, nlf_data_nah=None, nlf_data_fern=None, config_data="{}"):
        import json
        import math
        import numpy as np

        log_messages = ["=== V20 CALIBRATION LOG (FULL SKELETON + SMART CALF) ==="]
        try: config = json.loads(config_data)
        except: config = {}

        # --- 2D & DEPTH LOGIK (Unverändert, weil sie gut funktioniert) ---
        def get_body_metrics(pose_s, pose_u, depth_map):
            meta_s = pose_s.get("pose_metas", [])[0]
            meta_u = pose_u.get("pose_metas", [])[0]
            kps_s = getattr(meta_s, "kps_body", None)
            confs_s = getattr(meta_s, "kps_body_p", None)
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]

            def is_val(kps, confs, idx):
                if kps is None or idx >= len(kps): return False
                pt = kps[idx]
                if pt is None or len(pt) < 2: return False
                c = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
                return c >= min_confidence

            norm = 100.0 
            if norm_method == "Torso (Neck-Hip)":
                if is_val(kps_s, confs_s, 1) and is_val(kps_s, confs_s, 8) and is_val(kps_s, confs_s, 11):
                    mid_x = (kps_s[8][0] + kps_s[11][0]) / 2.0
                    mid_y = (kps_s[8][1] + kps_s[11][1]) / 2.0
                    norm = math.sqrt((kps_s[1][0] - mid_x)**2 + (kps_s[1][1] - mid_y)**2)
            else:
                valid_y = [kps_s[i][1] for i in range(len(kps_s)) if is_val(kps_s, confs_s, i)]
                valid_x = [kps_s[i][0] for i in range(len(kps_s)) if is_val(kps_s, confs_s, i)]
                if valid_y and valid_x:
                    norm = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)

            depth = 0.5 
            valid_u_x = [kps_u[idx][0] for idx in [1, 8, 11] if is_val(kps_u, confs_u, idx)]
            valid_u_y = [kps_u[idx][1] for idx in [1, 8, 11] if is_val(kps_u, confs_u, idx)]
            
            if valid_u_x and valid_u_y:
                min_x = int(max(0, min(valid_u_x) * W))
                max_x = int(min(W-1, max(valid_u_x) * W))
                min_y = int(max(0, min(valid_u_y) * H))
                max_y = int(min(H-1, max(valid_u_y) * H))
                if max_x > min_x and max_y > min_y:
                    depth = float(np.mean(depth_np[0, min_y:max_y, min_x:max_x]))
            return {"norm": norm, "depth": depth}

        data_nah = get_body_metrics(pose_nah_scaled, pose_nah_unscaled, depth_nah)
        data_fern = get_body_metrics(pose_fern_scaled, pose_fern_unscaled, depth_fern)
        
        norm_nah, norm_fern = data_nah['norm'], data_fern['norm']
        depth_c, depth_f = data_nah['depth'], data_fern['depth']

        if invert_depth:
            depth_c = 1.0 / max(depth_c, 0.0001)
            depth_f = 1.0 / max(depth_f, 0.0001)

        slope, intercept = 0.0, 1.0
        depth_diff = abs(depth_f - depth_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_f - depth_c)
            intercept = norm_nah - (slope * depth_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_c)

        fx, echte_groesse = 500.0, 0.0
        if use_pinhole_math:
            delta_z = abs(data_fern['depth'] - data_nah['depth'])
            if delta_z > 0.001 and abs(norm_nah - norm_fern) > 0.1:
                echte_groesse = (norm_nah * norm_fern * delta_z) / (fx * abs(norm_nah - norm_fern))
            else:
                echte_groesse = (norm_nah * data_nah['depth']) / fx

        # --- 3D LÄNGEN-EXTRAKTION (ALLE KNOCHEN + WADEN REPARATUR) ---
        true_3d_bones = {}
        total_3d_height = 0.0

        def extract_all_3d_bones(nlf_data):
            if nlf_data is None: return None
            try:
                pose_input_3d = nlf_data.get('joints3d_nonparam', [nlf_data])[0]
                if pose_input_3d is not None and len(pose_input_3d) > 0 and len(pose_input_3d[0]) > 0:
                    pose_3d = pose_input_3d[0][0]
                    def dist_3d(p1, p2): return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))
                    
                    # Mid-Hip berechnen für korrekte Torso-Länge
                    mid_hip = [(pose_3d[8][i] + pose_3d[11][i]) / 2.0 for i in range(3)]
                    
                    return {
                        "torso": dist_3d(pose_3d[1], mid_hip),
                        "shoulder_width": dist_3d(pose_3d[2], pose_3d[5]),
                        "hip_width": dist_3d(pose_3d[8], pose_3d[11]),
                        "r_arm": dist_3d(pose_3d[2], pose_3d[3]),
                        "r_forearm": dist_3d(pose_3d[3], pose_3d[4]),
                        "l_arm": dist_3d(pose_3d[5], pose_3d[6]),
                        "l_forearm": dist_3d(pose_3d[6], pose_3d[7]),
                        "r_thigh": dist_3d(pose_3d[8], pose_3d[9]),
                        "r_calf": dist_3d(pose_3d[9], pose_3d[10]),
                        "l_thigh": dist_3d(pose_3d[11], pose_3d[12]),
                        "l_calf": dist_3d(pose_3d[12], pose_3d[13])
                    }
            except: pass
            return None

        bones_nah = extract_all_3d_bones(nlf_data_nah)
        bones_fern = extract_all_3d_bones(nlf_data_fern)

        if bones_nah and bones_fern:
            # Den Skalierungsfaktor zwischen Nah und Fern ermitteln (anhand des stabilen Torsos)
            master_3d_factor = bones_nah["torso"] / bones_fern["torso"] if bones_fern["torso"] > 0 else 1.0
            
            true_3d_bones = bones_nah.copy()

            # BEIDE Waden reparieren (weil die im Nah-Bild oft abgeschnitten/halluziniert sind)
            # Wir nehmen die echte Länge aus dem Fern-Bild und skalieren sie auf das Nah-Bild hoch
            r_calf_estimated = bones_fern["r_calf"] * master_3d_factor
            l_calf_estimated = bones_fern["l_calf"] * master_3d_factor
            
            true_3d_bones["r_calf"] = r_calf_estimated
            true_3d_bones["l_calf"] = l_calf_estimated
            true_3d_bones["factor_nah_fern"] = master_3d_factor

            # Durchschnittliche Beinlänge für die Total-Height
            avg_thigh = (true_3d_bones["r_thigh"] + true_3d_bones["l_thigh"]) / 2.0
            avg_calf = (r_calf_estimated + l_calf_estimated) / 2.0
            
            total_3d_height = true_3d_bones["torso"] + avg_thigh + avg_calf + config.get("head_allowance_3d", 0.15) 
            log_messages.append(f"Waden repariert! Neu R: {r_calf_estimated:.3f} | L: {l_calf_estimated:.3f}")
            log_messages.append(f"Alle Knochen für den Retargeter extrahiert!")
        
        calib_data = {
            "perspective_slope": slope, "perspective_intercept": intercept, "is_depth_inverted": invert_depth,
            "norm_method": norm_method, "use_pinhole_math": use_pinhole_math, "focal_length_fx": fx,
            "echte_groesse": echte_groesse, "true_3d_bones": true_3d_bones, "total_3d_height": total_3d_height,
            "config": config
        }
        return (calib_data, "\n".join(log_messages))


class PoseCalibrationV22:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_nah": ("IMAGE",),
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_fern": ("IMAGE",),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
                "normalize_bones_to_100": ("BOOLEAN", {"default": True, "tooltip": "Setzt Torso auf 100 und berechnet den Rest in Prozent"}),
            },
            "optional": {
                "intrinsics_json": ("STRING", {"forceInput": True, "tooltip": "Intrinsics JSON"}),
                "nlf_data_nah": ("NLFPRED", {"tooltip": "Wird ignoriert (nur für Kabel-Kompatibilität im Workflow)"}),
                "nlf_data_fern": ("NLFPRED", {"tooltip": "Wird ignoriert (nur für Kabel-Kompatibilität im Workflow)"}),
                "config_data": ("STRING", {"default": "{}", "tooltip": "JSON Config"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, normalize_bones_to_100=True, intrinsics_json=None, nlf_data_nah=None, nlf_data_fern=None, config_data="{}"):
        log_messages = ["=== V22 CALIBRATION LOG (PURE 2D PROPORTIONS & SYMMETRY) ==="]
        
        try:
            config = json.loads(config_data)
        except:
            config = {}

        # ---------------------------------------------------------
        # 1. BERECHNUNG VON NORM & TIEFE (Unverändert für Slope etc.)
        # ---------------------------------------------------------
        def get_body_metrics(pose_s, pose_u, depth_map):
            meta_s = pose_s.get("pose_metas", [])[0]
            meta_u = pose_u.get("pose_metas", [])[0]
            kps_s = getattr(meta_s, "kps_body", None)
            confs_s = getattr(meta_s, "kps_body_p", None)
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]

            def is_val(kps, confs, idx):
                if kps is None or idx >= len(kps): return False
                pt = kps[idx]
                if pt is None or len(pt) < 2: return False
                c = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
                return c >= min_confidence

            norm = 100.0
            if norm_method == "Torso (Neck-Hip)":
                if is_val(kps_s, confs_s, 1) and is_val(kps_s, confs_s, 8) and is_val(kps_s, confs_s, 11):
                    mid_x = (kps_s[8][0] + kps_s[11][0]) / 2.0
                    mid_y = (kps_s[8][1] + kps_s[11][1]) / 2.0
                    norm = math.sqrt((kps_s[1][0] - mid_x)**2 + (kps_s[1][1] - mid_y)**2)
            else:
                valid_y = [kps_s[i][1] for i in range(len(kps_s)) if is_val(kps_s, confs_s, i)]
                valid_x = [kps_s[i][0] for i in range(len(kps_s)) if is_val(kps_s, confs_s, i)]
                if valid_y and valid_x:
                    norm = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)
            
            depth = 0.5
            valid_u_x = [kps_u[idx][0] for idx in [1, 8, 11] if is_val(kps_u, confs_u, idx)]
            valid_u_y = [kps_u[idx][1] for idx in [1, 8, 11] if is_val(kps_u, confs_u, idx)]
            if valid_u_x and valid_u_y:
                min_x = int(max(0, min(valid_u_x) * W))
                max_x = int(min(W-1, max(valid_u_x) * W))
                min_y = int(max(0, min(valid_u_y) * H))
                max_y = int(min(H-1, max(valid_u_y) * H))
                if max_x > min_x and max_y > min_y:
                    depth = float(np.mean(depth_np[0, min_y:max_y, min_x:max_x]))
                    
            return {"norm": norm, "depth": depth}

        data_nah = get_body_metrics(pose_nah_scaled, pose_nah_unscaled, depth_nah)
        data_fern = get_body_metrics(pose_fern_scaled, pose_fern_unscaled, depth_fern)
        norm_nah, norm_fern = data_nah['norm'], data_fern['norm']
        depth_c, depth_f = data_nah['depth'], data_fern['depth']

        if invert_depth:
            depth_c = 1.0 / max(depth_c, 0.0001)
            depth_f = 1.0 / max(depth_f, 0.0001)

        slope, intercept = 0.0, 1.0
        depth_diff = abs(depth_f - depth_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_f - depth_c)
            intercept = norm_nah - (slope * depth_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_c)

        fx, echte_groesse = 500.0, 0.0
        if use_pinhole_math:
            delta_z = abs(data_fern['depth'] - data_nah['depth'])
            if delta_z > 0.001 and abs(norm_nah - norm_fern) > 0.1:
                echte_groesse = (norm_nah * norm_fern * delta_z) / (fx * abs(norm_nah - norm_fern))
            else:
                echte_groesse = (norm_nah * data_nah['depth']) / fx

        # ---------------------------------------------------------
        # 2. NEU V22: PURE 2D BONES BERECHNEN (KEIN DEPTH-GLITCH MEHR!)
        # ---------------------------------------------------------
        def extract_2d_bones(pose_data):
            try:
                meta = pose_data.get("pose_metas", [])[0]
                kps = getattr(meta, "kps_body", None)
                if kps is None or len(kps) < 14: 
                    return None
                
                def dist_2d(idx1, idx2):
                    if idx1 >= len(kps) or idx2 >= len(kps): return 0.0
                    p1, p2 = kps[idx1], kps[idx2]
                    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2: return 0.0
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                # Torso-Länge (Hals bis Mitte Hüfte)
                mid_hip_x = (kps[8][0] + kps[11][0]) / 2.0
                mid_hip_y = (kps[8][1] + kps[11][1]) / 2.0
                torso_len = math.sqrt((kps[1][0] - mid_hip_x)**2 + (kps[1][1] - mid_hip_y)**2)
                
                if torso_len <= 0: return None

                # Originale 2D Pixel-Längen (COCO Format)
                raw_bones = {
                    "torso": torso_len,
                    "shoulder_width": dist_2d(2, 5),
                    "hip_width": dist_2d(8, 11),
                    "r_arm": dist_2d(2, 3),
                    "r_forearm": dist_2d(3, 4),
                    "l_arm": dist_2d(5, 6),
                    "l_forearm": dist_2d(6, 7),
                    "r_thigh": dist_2d(8, 9),
                    "r_calf": dist_2d(9, 10),
                    "l_thigh": dist_2d(11, 12),
                    "l_calf": dist_2d(12, 13)
                }
                
                # SYMMETRIE ERZWINGEN
                sym_bones = {
                    "torso": raw_bones["torso"],
                    "shoulder_width": raw_bones["shoulder_width"],
                    "hip_width": raw_bones["hip_width"],
                    "r_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "l_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "r_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "l_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "r_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "l_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "r_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0,
                    "l_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0
                }
                
                # OPTIONAL: Torso auf 100 setzen (Prozentuale Verhältnisse)
                if normalize_bones_to_100:
                    norm_bones = {}
                    for k, v in sym_bones.items():
                        norm_bones[k] = (v / sym_bones["torso"]) * 100.0
                    return norm_bones
                
                return sym_bones
                
            except Exception as e:
                log_messages.append(f"Fehler bei 2D Bone Extraktion: {e}")
                return None

        # Wir nehmen standardmäßig das FERN-Bild, da man dort meist in sauberer T-Pose steht
        true_3d_bones = extract_2d_bones(pose_fern_scaled)
        
        # Fallback auf NAH-Bild, falls FERN kaputt ist
        if not true_3d_bones:
            true_3d_bones = extract_2d_bones(pose_nah_scaled)
            log_messages.append("Nutze Nah-Bild für Bones (Fern fehlgeschlagen).")
        else:
            log_messages.append("Perfekte 2D-Proportionen aus Fern-Bild extrahiert.")

        total_3d_height = 0.0
        if true_3d_bones:
            total_3d_height = true_3d_bones["torso"] + true_3d_bones["r_thigh"] + true_3d_bones["r_calf"]
            head_allowance = config.get("head_allowance_3d", 0.15)
            total_3d_height += head_allowance

        calib_data = {
            "perspective_slope": slope,
            "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth,
            "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math,
            "focal_length_fx": fx,
            "echte_groesse": echte_groesse,
            "true_3d_bones": true_3d_bones or {},
            "total_3d_height": total_3d_height,
            "config": config
        }

        return (calib_data, "\n".join(log_messages))


class PoseCalibrationV15:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_nah": ("IMAGE",),
                
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_fern": ("IMAGE",),
                
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "intrinsics_json": ("STRING", {"forceInput": True, "tooltip": "Intrinsics JSON aus DA3"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, 
                  pose_fern_scaled, pose_fern_unscaled, depth_fern, 
                  norm_method, min_confidence, invert_depth, use_pinhole_math=True, intrinsics_json=None):
        import numpy as np
        import math
        import json
        
        log_messages = ["=== V15 CALIBRATION LOG (UNSCALED DEPTH + PINHOLE) ==="]
        log_messages.append(f"Methode: {norm_method}")

        def get_body_metrics(pose_s, pose_u, depth_map):
            meta_s = pose_s.get("pose_metas", [])[0]
            meta_u = pose_u.get("pose_metas", [])[0]
            
            # Scaled für Pixel-Abmessungen
            kps_s = getattr(meta_s, "kps_body", None)
            confs_s = getattr(meta_s, "kps_body_p", None)
            
            # Unscaled für Depth-Map Sampling
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]
            
            # Validierung für Scaled
            def get_c_s(idx):
                if confs_s is not None and idx < len(confs_s): return float(confs_s[idx])
                if len(kps_s[idx]) >= 3: return float(kps_s[idx][2])
                return 1.0

            def is_val_s(idx):
                if kps_s is None or idx >= len(kps_s): return False
                pt = kps_s[idx]
                if pt is None or len(pt) < 2: return False
                if get_c_s(idx) < min_confidence: return False
                return True # Begrenzung wird bei Scaled ignoriert, da es über den Canvas hinausragen kann

            # Validierung für Unscaled (Depth)
            def get_c_u(idx):
                if confs_u is not None and idx < len(confs_u): return float(confs_u[idx])
                if len(kps_u[idx]) >= 3: return float(kps_u[idx][2])
                return 1.0

            def is_val_u(idx):
                if kps_u is None or idx >= len(kps_u): return False
                pt = kps_u[idx]
                if pt is None or len(pt) < 2: return False
                if get_c_u(idx) < min_confidence: return False
                return (0 <= pt[0] < W) and (0 <= pt[1] < H)

            if kps_s is None or len(kps_s) < 14: 
                return {'torso': 0.0, 'max_len': 0.0, 'depth': 0.5, 'parts': 'None'}

            # PIXELGRÖSSE BERECHNEN (MIT SCALED)
            torso_len = 0.0
            if is_val_s(1) and is_val_s(8) and is_val_s(11):
                torso_len = math.sqrt((kps_s[1][0] - (kps_s[8][0]+kps_s[11][0])/2.0)**2 + (kps_s[1][1] - (kps_s[8][1]+kps_s[11][1])/2.0)**2)

            top_y, bottom_y = None, None
            if is_val_s(0): top_y = kps_s[0][1]
            elif is_val_s(1): top_y = kps_s[1][1]

            parts_str = "Torso"
            heels = [idx for idx in [10, 13, 21, 24] if is_val_s(idx)]
            knees = [idx for idx in [9, 12] if is_val_s(idx)]
            hips = [idx for idx in [8, 11] if is_val_s(idx)]

            if heels:
                bottom_y = max([kps_s[idx][1] for idx in heels])
                parts_str = "Ganzkörper (mit Füßen)"
            elif knees:
                bottom_y = max([kps_s[idx][1] for idx in knees])
                parts_str = "Bis Knie"
            elif hips:
                bottom_y = max([kps_s[idx][1] for idx in hips])
                parts_str = "Nur Torso"

            max_len = torso_len 
            if top_y is not None and bottom_y is not None and bottom_y > top_y:
                max_len = bottom_y - top_y

            # TIEFE AUSLESEN (MIT UNSCALED)
            torso_x_u, torso_y_u = [], []
            if is_val_u(1) and is_val_u(8) and is_val_u(11):
                torso_x_u.extend([kps_u[1][0], kps_u[8][0], kps_u[11][0]])
                torso_y_u.extend([kps_u[1][1], kps_u[8][1], kps_u[11][1]])

            depth_vals = []
            for px, py in zip(torso_x_u, torso_y_u):
                ix, iy = int(px), int(py)
                if 0 <= ix < W and 0 <= iy < H:
                    depth_vals.append(depth_np[0, iy, ix])
            
            d_val = float(np.mean(depth_vals)) if depth_vals else 0.5
            if invert_depth: d_val = 1.0 / max(d_val, 0.0001)
            
            return {'torso': torso_len, 'max_len': max_len, 'depth': d_val, 'parts': parts_str}

        data_nah = get_body_metrics(pose_nah_scaled, pose_nah_unscaled, depth_nah)
        data_fern = get_body_metrics(pose_fern_scaled, pose_fern_unscaled, depth_fern)

        if data_nah['torso'] == 0.0 or data_fern['torso'] == 0.0:
            return ({}, "Fehler: Torso konnte nicht in beiden Posen erkannt werden.")

        norm_nah = data_nah['torso']
        norm_fern = data_fern['torso']

        log_messages.append("\n--- EXTRAPOLATION RECHENVORGANG ---")
        if norm_method == "Dynamic Full-Body":
            ratio_nah = data_nah['max_len'] / data_nah['torso']
            ratio_fern = data_fern['max_len'] / data_fern['torso']
            torso_scale_factor = data_nah['torso'] / data_fern['torso']

            log_messages.append(f"Torso-Faktor = Torso Nah ({data_nah['torso']:.1f}) / Torso Fern ({data_fern['torso']:.1f}) = {torso_scale_factor:.3f}")

            if ratio_fern > ratio_nah:
                norm_fern = data_fern['max_len']
                norm_nah = data_fern['max_len'] * torso_scale_factor
                log_messages.append(f"'Fern' zeigt mehr vom Körper ({data_fern['parts']}).")
                log_messages.append(f"Extrapoliere 'Nah' = Max_Len_Fern ({data_fern['max_len']:.1f}) * Torso-Faktor ({torso_scale_factor:.3f}) = {norm_nah:.1f} px")
            else:
                norm_nah = data_nah['max_len']
                norm_fern = data_nah['max_len'] / torso_scale_factor
                if ratio_nah > ratio_fern:
                    log_messages.append(f"'Nah' zeigt mehr vom Körper ({data_nah['parts']}).")
                    log_messages.append(f"Extrapoliere 'Fern' = Max_Len_Nah ({data_nah['max_len']:.1f}) / Torso-Faktor ({torso_scale_factor:.3f}) = {norm_fern:.1f} px")
                else:
                    log_messages.append(f"Beide Frames zeigen proportional gleich viel Körper ({data_nah['parts']}). Keine Extrapolation nötig.")
        else:
            log_messages.append("Methode 'Torso' gewählt. Keine Extrapolation angewandt.")

        # --- LINEARE BERECHNUNG (ALS FALLBACK/VERGLEICH) ---
        depth_diff = data_fern['depth'] - data_nah['depth']
        slope = 0.0 if abs(depth_diff) < 0.001 else (norm_fern - norm_nah) / depth_diff
        intercept = norm_nah - (slope * data_nah['depth'])

        # --- NEU: PINHOLE BERECHNUNG ---
        fx = 512.0
        if intrinsics_json:
            try:
                int_data = json.loads(intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx = float(matrix[0][0])
                        log_messages.append(f"\nBrennweite (fx) aus DA3 geladen: {fx:.2f}")
            except Exception as e:
                log_messages.append(f"\nWarnung: Intrinsics JSON Fehler ({e}). Nutze Fallback fx={fx}.")

        log_messages.append("\n--- PINHOLE DELTA RECHNUNG ---")
        delta_z = data_fern['depth'] - data_nah['depth']
        log_messages.append(f"Gemessene metrische Differenz (Delta Z): {delta_z:.3f}m")

        echte_groesse = 0.0
        if delta_z > 0 and (norm_nah - norm_fern) > 0:
            echte_groesse = (delta_z * norm_nah * norm_fern) / (fx * (norm_nah - norm_fern))
            log_messages.append(f"Echte physikalische Größe berechnet: {echte_groesse:.3f}m")
        else:
            echte_groesse = (norm_nah * data_nah['depth']) / fx
            log_messages.append(f"Warnung: Delta Z negativ oder Norm-Fehler. Nutze Absolutwert-Fallback: {echte_groesse:.3f}m")

        log_messages.append("\n=== ERGEBNIS ===")
        log_messages.append(f"Finale Norm Nah: {norm_nah:.1f} px | Tiefe Nah (Unscaled): {data_nah['depth']:.4f}")
        log_messages.append(f"Finale Norm Fern: {norm_fern:.1f} px | Tiefe Fern (Unscaled): {data_fern['depth']:.4f}")
        log_messages.append(f"Echte Größe: {echte_groesse:.3f}m | fx: {fx:.2f}")

        calib_data = {
            "perspective_slope": slope,
            "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth,
            "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math,
            "focal_length_fx": fx,
            "echte_groesse": echte_groesse
        }
        return (calib_data, "\n".join(log_messages))


class PoseGlobalPerspectiveScalerV28:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING",)
    RETURN_NAMES = ("scaled_pose_data", "log_output",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"

    def process(self, video_pose_data, calibration_data, video_depth_map, include_head, anchor_window, min_confidence, frontal_method="3D_NLF", frontal_2d_threshold=0.65, frontal_3d_angle_tolerance=20.0, video_nlf_data=None):
        import copy
        import numpy as np
        import math
        
        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V28 GLOBAL SCALER LOG (PINHOLE MATH) ==="]

        if not pose_metas: return (pose_data_copy, "Fehler: Keine Pose-Daten.")
        
        # --- CALIBRATION DATEN ABRUFEN ---
        slope = calibration_data.get("perspective_slope", 0.0)
        intercept = calibration_data.get("perspective_intercept", 1.0)
        is_inverted = calibration_data.get("is_depth_inverted", False)
        norm_method = calibration_data.get("norm_method", "Dynamic Full-Body")
        
        # NEU: Pinhole Daten abrufen
        use_pinhole_math = calibration_data.get("use_pinhole_math", False)
        fx = calibration_data.get("focal_length_fx", 512.0)
        echte_groesse = calibration_data.get("echte_groesse", 0.0)
        
        log_messages.append(f"Genutzte Skalierungs-Norm: {norm_method}\n")
        
        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]
        
        def get_conf(kps, confs, idx):
            if confs is not None and idx < len(confs): return float(confs[idx])
            if len(kps[idx]) >= 3: return float(kps[idx][2])
            return 1.0

        def is_valid_point(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            if get_conf(kps, confs, idx) < min_confidence: return False
            return (0 <= pt[0] < W) and (0 <= pt[1] < H)

        pose_input_3d = None
        if video_nlf_data is not None:
            if isinstance(video_nlf_data, dict): pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0]
            else: pose_input_3d = video_nlf_data

        body_lengths = []
        valid_body_indices = [1, 2, 5, 8, 11, 9, 12, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] 
        if include_head: valid_body_indices.append(0)

        frame_data = []
        has_premium_frontal = False
        
        for i, meta in enumerate(pose_metas):
            kps_2d = getattr(meta, "kps_body", None)
            confs = getattr(meta, "kps_body_p", None)
            
            t_norm = 0.0
            if is_valid_point(kps_2d, confs, 1) and is_valid_point(kps_2d, confs, 8) and is_valid_point(kps_2d, confs, 11):
                t_norm = math.sqrt((kps_2d[1][0] - (kps_2d[8][0]+kps_2d[11][0])/2.0)**2 + (kps_2d[1][1] - (kps_2d[8][1]+kps_2d[11][1])/2.0)**2)
                
            if kps_2d is None or len(kps_2d) < 14 or t_norm == 0.0:
                frame_data.append({'valid': False})
                body_lengths.append(0.0)
                continue
                
            valid_y = [kps_2d[idx][1] for idx in valid_body_indices if is_valid_point(kps_2d, confs, idx)]
            length = max(valid_y) - min(valid_y) if valid_y else 0.0
            body_lengths.append(length)

            has_feet = any(is_valid_point(kps_2d, confs, idx) for idx in [18, 19, 20, 21, 22, 23])
            has_ankles = any(is_valid_point(kps_2d, confs, idx) for idx in [10, 13])
            has_knees = any(is_valid_point(kps_2d, confs, idx) for idx in [9, 12])
            has_lower_body = has_feet or has_ankles or has_knees
            
            is_frontal = False
            frontal_pts = 0.0
            used_method_log = ""
            
            frontal_ratio_2d = 0.0
            if is_valid_point(kps_2d, confs, 2) and is_valid_point(kps_2d, confs, 5):
                shoulder_width = math.sqrt((kps_2d[2][0] - kps_2d[5][0])**2 + (kps_2d[2][1] - kps_2d[5][1])**2)
                frontal_ratio_2d = shoulder_width / t_norm if t_norm > 0 else 0.0

            if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                pose_3d_frame = pose_input_3d[i]
                if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                    person_3d = pose_3d_frame[0]
                    num_joints = len(person_3d)
                    
                    idx_r, idx_l = 2, 5
                    format_name = "OpenPose"
                    if num_joints == 17:
                        idx_r, idx_l = 11, 14
                        format_name = "H36M"
                    elif num_joints in [24, 45, 68]:
                        idx_r, idx_l = 16, 17
                        format_name = "SMPL"
                    
                    if num_joints > max(idx_r, idx_l):
                        x_r, z_r = float(person_3d[idx_r][0]), float(person_3d[idx_r][2])
                        x_l, z_l = float(person_3d[idx_l][0]), float(person_3d[idx_l][2])
                        dx, dz = abs(x_l - x_r), abs(z_l - z_r)
                        angle_deg = math.degrees(math.atan2(dz, dx)) if (dx != 0 or dz != 0) else 0.0
                        
                        is_frontal = angle_deg <= frontal_3d_angle_tolerance
                        frontal_pts = 500.0 * max(0.0, 1.0 - (angle_deg / 45.0))
                        used_method_log = f"3D({format_name}, {angle_deg:.1f}°)"
                    else:
                        is_frontal = frontal_ratio_2d >= frontal_2d_threshold
                        frontal_pts = min(500.0, (frontal_ratio_2d / 0.8) * 500.0)
                        used_method_log = f"2D-Fallback({frontal_ratio_2d:.2f})"
                else:
                    is_frontal = frontal_ratio_2d >= frontal_2d_threshold
                    frontal_pts = min(500.0, (frontal_ratio_2d / 0.8) * 500.0)
                    used_method_log = f"2D-Fallback({frontal_ratio_2d:.2f})"
            else:
                is_frontal = frontal_ratio_2d >= frontal_2d_threshold
                frontal_pts = min(500.0, (frontal_ratio_2d / 0.8) * 500.0)
                used_method_log = f"2D-Ratio({frontal_ratio_2d:.2f})"
                
            if is_frontal and has_lower_body:
                has_premium_frontal = True
                
            frame_data.append({
                'valid': True, 'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees,
                'is_frontal': is_frontal, 'frontal_pts': frontal_pts, 'used_method_log': used_method_log, 'length': length
            })

        max_body_length = max(body_lengths) if body_lengths else 1.0
        if max_body_length <= 0: max_body_length = 1.0

        if has_premium_frontal:
            log_messages.append(f">> PASS-FILTER AKTIV: Frontale Frames gefunden! Seitliche Frames fliegen raus.")

        frame_scores = []
        for i, data in enumerate(frame_data):
            if not data['valid'] or (has_premium_frontal and not data['is_frontal']):
                frame_scores.append(-1.0)
                continue

            waden_pts = 1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0
            schenkel_pts = 500.0 if (waden_pts == 0 and data['has_knees']) else 0.0
            bein_pts = max(waden_pts, schenkel_pts)
            fuss_bonus_pts = 500.0 if (data['has_feet'] and data['is_frontal']) else 0.0
            
            total = bein_pts + fuss_bonus_pts + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            frame_scores.append(total)

        best_idx = int(np.argmax(frame_scores))
        if frame_scores[best_idx] < 0: return (pose_data_copy, "Fehler: Kein gültiger Frame gefunden.")

        # --- SCALING ---
        start_idx = max(0, best_idx - anchor_window)
        end_idx = min(len(pose_metas) - 1, best_idx + anchor_window)
        sum_norm, sum_depth, valid_frames_in_window = 0.0, 0.0, 0
        
        for i in range(start_idx, end_idx + 1):
            kps = getattr(pose_metas[i], "kps_body", None)
            confs = getattr(pose_metas[i], "kps_body_p", None)
            
            norm_val = 0.0
            valid_x, valid_y = [], []

            if norm_method == "Torso (Neck-Hip)":
                if is_valid_point(kps, confs, 1) and is_valid_point(kps, confs, 8) and is_valid_point(kps, confs, 11):
                    norm_val = math.sqrt((kps[1][0] - (kps[8][0]+kps[11][0])/2.0)**2 + (kps[1][1] - (kps[8][1]+kps[11][1])/2.0)**2)
                    valid_x.extend([kps[1][0], kps[8][0], kps[11][0]])
                    valid_y.extend([kps[1][1], kps[8][1], kps[11][1]])
            else:
                top_y, bottom_y = None, None
                if is_valid_point(kps, confs, 0): top_y = kps[0][1]; valid_x.append(kps[0][0]); valid_y.append(kps[0][1])
                elif is_valid_point(kps, confs, 1): top_y = kps[1][1]; valid_x.append(kps[1][0]); valid_y.append(kps[1][1])

                heels = [idx for idx in [10, 13, 21, 24] if is_valid_point(kps, confs, idx)]
                knees = [idx for idx in [9, 12] if is_valid_point(kps, confs, idx)]
                hips = [idx for idx in [8, 11] if is_valid_point(kps, confs, idx)]

                if heels:
                    bottom_y = max([kps[idx][1] for idx in heels])
                    for idx in heels: valid_x.append(kps[idx][0]); valid_y.append(kps[idx][1])
                elif knees:
                    bottom_y = max([kps[idx][1] for idx in knees])
                    for idx in knees: valid_x.append(kps[idx][0]); valid_y.append(kps[idx][1])
                elif hips:
                    bottom_y = max([kps[idx][1] for idx in hips])
                    for idx in hips: valid_x.append(kps[idx][0]); valid_y.append(kps[idx][1])

                if top_y is not None and bottom_y is not None and bottom_y > top_y:
                    norm_val = bottom_y - top_y

            if norm_val == 0.0: continue

            depth_vals = []
            v_idx = min(i, depth_np.shape[0] - 1)
            for px, py in zip(valid_x, valid_y):
                ix, iy = int(px), int(py)
                if 0 <= ix < W and 0 <= iy < H:
                    depth_vals.append(depth_np[v_idx, iy, ix])
                    
            frame_depth = float(np.mean(depth_vals)) if depth_vals else 0.5
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)
            
            sum_norm += norm_val
            sum_depth += frame_depth
            valid_frames_in_window += 1

        if valid_frames_in_window == 0:
            return (pose_data_copy, "Fehler: Im Anchor-Window konnte keine Tiefe/Norm ermittelt werden.")

        avg_anchor_norm = sum_norm / valid_frames_in_window
        avg_anchor_depth = sum_depth / valid_frames_in_window
        
        log_messages.append(f"\n--- SKALIERUNG RECHENVORGANG ---")
        log_messages.append(f"Gewinner Frame: {best_idx}")
        log_messages.append(f"Ist-Norm (Video): {avg_anchor_norm:.1f} px")
        log_messages.append(f"Ist-Tiefe (Video): {avg_anchor_depth:.3f} m")

        # --- DIE NEUE PINHOLE ODER LINEARE RECHNUNG ---
        if use_pinhole_math and echte_groesse > 0.0:
            log_messages.append("\n--> PINHOLE SKALIERUNG AKTIV")
            expected_norm = (echte_groesse * fx) / avg_anchor_depth
            log_messages.append(f"Soll-Norm = ({echte_groesse:.3f}m * {fx:.2f}) / {avg_anchor_depth:.3f}m = {expected_norm:.1f} px")
        else:
            log_messages.append("\n--> LINEARE SKALIERUNG AKTIV (FALLBACK)")
            expected_norm = (avg_anchor_depth * slope) + intercept
            log_messages.append(f"Soll-Norm = ({avg_anchor_depth:.3f} * {slope:.2f}) + {intercept:.2f} = {expected_norm:.1f} px")

        anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0
        
        log_messages.append(f"\nFaktor = Soll-Norm / Ist-Norm")
        log_messages.append(f"Faktor = {expected_norm:.1f} / {avg_anchor_norm:.1f} = {anchor_scale:.3f}x")
        
        log_messages.append(f"\n=== ERGEBNIS ===")
        log_messages.append(f"FESTER SKALIERUNGSFAKTOR FÜR GANZES VIDEO: {anchor_scale:.3f}x\n")

        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
                    
            if not valid_y or not valid_x: continue
            pivot_y, pivot_x = max(valid_y), np.mean(valid_x)
            
            for attr_name in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr_name, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = pivot_x + (arr[j][0] - pivot_x) * anchor_scale
                            arr[j][1] = pivot_y + (arr[j][1] - pivot_y) * anchor_scale
                            
        log_messages.append("Erfolgreich: Skalierung angewandt.")
        return (pose_data_copy, "\n".join(log_messages))


class PoseCalibrationV23:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_nah": ("IMAGE",),
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_fern": ("IMAGE",),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
                "normalize_bones_to_100": ("BOOLEAN", {"default": True, "tooltip": "Setzt Torso auf 100 und berechnet den Rest in Prozent"}),
            },
            "optional": {
                "intrinsics_json": ("STRING", {"forceInput": True, "tooltip": "Intrinsics JSON"}),
                "nlf_data_nah": ("NLFPRED", {"tooltip": "3D NLF Daten Nah (abgeschnitten)"}),
                "nlf_data_fern": ("NLFPRED", {"tooltip": "3D NLF Daten Fern (ganzer Körper)"}),
                "config_data": ("STRING", {"default": "{}", "tooltip": "JSON Config"}),
            }
        }
    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V23: 3D-Längen aus NLF berechnen, an 2D-Gesamthöhe skalieren und in bone_length_for_scaler speichern."

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, normalize_bones_to_100=True, intrinsics_json=None, nlf_data_nah=None, nlf_data_fern=None, config_data="{}"):
        import json
        import math
        import numpy as np
        
        log_messages = ["=== V23 CALIBRATION LOG (3D TO 2D SCALER) ==="]
        try:
            config = json.loads(config_data)
        except:
            config = {}

        # --- 2D & DEPTH LOGIK (Wie in V22) ---
        def get_body_metrics(pose_s, pose_u, depth_map):
            meta_s = pose_s.get("pose_metas", [])[0]
            meta_u = pose_u.get("pose_metas", [])[0]
            kps_s = getattr(meta_s, "kps_body", None)
            confs_s = getattr(meta_s, "kps_body_p", None)
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]
            
            def is_val(kps, confs, idx):
                if kps is None or idx >= len(kps): return False
                pt = kps[idx]
                if pt is None or len(pt) < 2: return False
                c = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
                return c >= min_confidence
                
            norm = 100.0
            total_2d_height = 100.0 # Neu für V23
            
            valid_y = [kps_s[i][1] for i in range(len(kps_s)) if is_val(kps_s, confs_s, i)]
            valid_x = [kps_s[i][0] for i in range(len(kps_s)) if is_val(kps_s, confs_s, i)]
            
            if valid_y and valid_x:
                total_2d_height = max(valid_y) - min(valid_y)
                
            if norm_method == "Torso (Neck-Hip)":
                if is_val(kps_s, confs_s, 1) and is_val(kps_s, confs_s, 8) and is_val(kps_s, confs_s, 11):
                    mid_x = (kps_s[8][0] + kps_s[11][0]) / 2.0
                    mid_y = (kps_s[8][1] + kps_s[11][1]) / 2.0
                    norm = math.sqrt((kps_s[1][0] - mid_x)**2 + (kps_s[1][1] - mid_y)**2)
            else:
                if valid_y and valid_x:
                    norm = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)
                    
            depth = 0.5 
            valid_u_x = [kps_u[idx][0] for idx in [1, 8, 11] if is_val(kps_u, confs_u, idx)]
            valid_u_y = [kps_u[idx][1] for idx in [1, 8, 11] if is_val(kps_u, confs_u, idx)]
            
            if valid_u_x and valid_u_y:
                min_x = int(max(0, min(valid_u_x) * W))
                max_x = int(min(W-1, max(valid_u_x) * W))
                min_y = int(max(0, min(valid_u_y) * H))
                max_y = int(min(H-1, max(valid_u_y) * H))
                if max_x > min_x and max_y > min_y:
                    depth = float(np.mean(depth_np[0, min_y:max_y, min_x:max_x]))
            return {"norm": norm, "depth": depth, "total_2d_height": total_2d_height}

        data_nah = get_body_metrics(pose_nah_scaled, pose_nah_unscaled, depth_nah)
        data_fern = get_body_metrics(pose_fern_scaled, pose_fern_unscaled, depth_fern)
        
        norm_nah, norm_fern = data_nah['norm'], data_fern['norm']
        depth_c, depth_f = data_nah['depth'], data_fern['depth']
        
        if invert_depth:
            depth_c = 1.0 / max(depth_c, 0.0001)
            depth_f = 1.0 / max(depth_f, 0.0001)
            
        slope, intercept = 0.0, 1.0
        depth_diff = abs(depth_f - depth_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_f - depth_c)
            intercept = norm_nah - (slope * depth_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_c)
            
        fx, echte_groesse = 500.0, 0.0
        if use_pinhole_math:
            delta_z = abs(data_fern['depth'] - data_nah['depth'])
            if delta_z > 0.001 and abs(norm_nah - norm_fern) > 0.1:
                echte_groesse = (norm_nah * norm_fern * delta_z) / (fx * abs(norm_nah - norm_fern))
            else:
                echte_groesse = (norm_nah * data_nah['depth']) / fx

        # --- 3D LÄNGEN-EXTRAKTION (WADE REPARIEREN) ---
        bone_length_for_scaler = {}
        total_3d_height = 0.0
        
        def extract_3d_bones(nlf_data):
            if nlf_data is None: return None
            try:
                pose_input_3d = nlf_data.get('joints3d_nonparam', [nlf_data])[0]
                if pose_input_3d is not None and len(pose_input_3d) > 0 and len(pose_input_3d[0]) > 0:
                    pose_3d = pose_input_3d[0][0]
                    def dist_3d(p1, p2):
                        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                    
                    # Wichtige Bones für V18 Logik
                    torso = dist_3d(pose_3d[1], pose_3d[8]) if len(pose_3d) > 8 else 0
                    thigh = dist_3d(pose_3d[8], pose_3d[9]) if len(pose_3d) > 9 else 0
                    calf = dist_3d(pose_3d[9], pose_3d[10]) if len(pose_3d) > 10 else 0
                    return {"torso": torso, "thigh": thigh, "calf": calf}
            except: pass
            return None

        bones_nah = extract_3d_bones(nlf_data_nah)
        bones_fern = extract_3d_bones(nlf_data_fern)

        if bones_nah and bones_fern:
            factor_torso = bones_nah["torso"] / bones_fern["torso"] if bones_fern["torso"] > 0 else 1.0
            factor_thigh = bones_nah["thigh"] / bones_fern["thigh"] if bones_fern["thigh"] > 0 else 1.0
            master_3d_factor = (factor_torso + factor_thigh) / 2.0
            
            calf_nah_estimated = bones_fern["calf"] * master_3d_factor
            
            raw_3d_bones = {
                "torso": bones_nah["torso"],
                "thigh": bones_nah["thigh"],
                "calf": calf_nah_estimated
            }
            
            head_allowance = config.get("head_allowance_3d", 0.15)
            total_3d_height = bones_nah["torso"] + bones_nah["thigh"] + calf_nah_estimated + head_allowance
            
            log_messages.append(f"Faktor Nah/Fern (aus Oberschenkel/Torso): {master_3d_factor:.3f}x")
            log_messages.append(f"Wade repariert! Alt: {bones_nah['calf']:.3f} -> Neu: {calf_nah_estimated:.3f}")
            
            # === V23 SCALING: 3D auf 2D Größe (aus dem genauen Frontal-Kalibrator) ===
            true_2d_height = data_fern['total_2d_height'] # Wir nehmen das ferne Bild, da dort alles drauf ist
            scale_3d_to_2d = true_2d_height / total_3d_height if total_3d_height > 0 else 1.0
            
            log_messages.append(f"Skaliere 3D Längen auf echte 2D Pose Größe: {true_2d_height:.1f}px (Faktor {scale_3d_to_2d:.3f})")
            
            for k, v in raw_3d_bones.items():
                bone_length_for_scaler[k] = v * scale_3d_to_2d
                log_messages.append(f" -> Skalierter {k}: {bone_length_for_scaler[k]:.2f}")
                
        else:
            log_messages.append("Fehler/Warnung: Konnte nicht beide NLF-Frames extrahieren. bone_length_for_scaler ist leer.")

        calib_data = {
            "perspective_slope": slope,
            "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth,
            "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math,
            "focal_length_fx": fx,
            "echte_groesse": echte_groesse,
            "bone_length_for_scaler": bone_length_for_scaler, # NEUER KEY für V39 / Retargeter
            "total_3d_height": total_3d_height,
            "config": config
        }
        
        return (calib_data, "\n".join(log_messages))


class PoseGlobalPerspectiveScalerV39:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "scale_calculation_method": (["2D Pose Data", "NLF Data"], {"default": "2D Pose Data"}), # NEU!
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V39: Skaliert 2D-Posen wahlweise anhand 2D-Tiefenmap oder 3D-NLF Daten. Schreibt Config für NLF Renderer."

    def process(self, video_pose_data, calibration_data, video_depth_map, include_head, anchor_window, min_confidence, scale_calculation_method="2D Pose Data", frontal_2d_threshold=0.65, scale_2d_axes="X and Y (Uniform)", video_nlf_data=None):
        import copy
        import numpy as np
        import math
        import json
        
        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V39 GLOBAL SCALER LOG ({scale_calculation_method}) ==="]
        
        if not pose_metas:
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        def is_valid_point(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else (float(pt[2]) if len(pt)>=3 else 1.0)
            return c >= min_confidence

        anchor_scale = 1.0
        global_pivot_x, global_pivot_y = 0.5, 0.5

        # =========================================================
        # METHODE A: NLF DATA (Frame-Logik aus dem Retargeter V5)
        # =========================================================
        if scale_calculation_method == "NLF Data" and video_nlf_data is not None:
            target_bones = calibration_data.get("bone_length_for_scaler", {})
            if not target_bones or "torso" not in target_bones:
                log_messages.append("FEHLER: 'bone_length_for_scaler' fehlt in Calibration. Fallback auf 1.0.")
            else:
                pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) else video_nlf_data
                
                # 1. Frontalsten Frame finden (Retargeter Logik: kleinste Z-Differenz bei Hüften)
                best_idx = 0
                best_frontal_score = float('inf')
                for i in range(len(pose_metas)):
                    if pose_input_3d is None or i >= len(pose_input_3d) or pose_input_3d[i] is None or len(pose_input_3d[i]) == 0:
                        continue
                    pose_3d = pose_input_3d[i][0]
                    if len(pose_3d) > 11:
                        # Berechne Z-Differenz zwischen rechter und linker Hüfte
                        z_diff = abs(pose_3d[11][2] - pose_3d[8][2]) 
                        if z_diff < best_frontal_score:
                            best_frontal_score = z_diff
                            best_idx = i
                            
                log_messages.append(f"Frontalster Frame (NLF Logik): {best_idx} (Z-Differenz: {best_frontal_score:.4f})")

                # 2. Skalierungsfaktor im Anchor-Frame berechnen (Vergleich Torso-Länge)
                anchor_3d = pose_input_3d[best_idx][0]
                def dist_3d(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                
                current_torso_3d = dist_3d(anchor_3d[1], anchor_3d[8]) if len(anchor_3d) > 8 else 1.0
                target_torso_3d = target_bones["torso"]
                
                anchor_scale = target_torso_3d / current_torso_3d if current_torso_3d > 0 else 1.0
                log_messages.append(f"NLF Skalierungs-Faktor: {target_torso_3d:.2f} / {current_torso_3d:.2f} = {anchor_scale:.3f}x")

                # Pivot setzen
                kps_best = getattr(pose_metas[best_idx], "kps_body", [])
                val_y = [kps_best[idx][1] for idx in range(len(kps_best)) if is_valid_point(kps_best, None, idx)]
                val_x = [kps_best[idx][0] for idx in range(len(kps_best)) if is_valid_point(kps_best, None, idx)]
                if val_y and val_x:
                    global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        # =========================================================
        # METHODE B: 2D POSE DATA (Wie vorher in V38)
        # =========================================================
        else:
            if scale_calculation_method == "NLF Data":
                log_messages.append("WARNUNG: NLF Data ausgewählt, aber kein 'video_nlf_data' verbunden! Fallback auf 2D.")
            
            slope = calibration_data.get("perspective_slope", 0.0)
            intercept = calibration_data.get("perspective_intercept", 1.0)
            is_inverted = calibration_data.get("is_depth_inverted", False)
            echte_groesse = calibration_data.get("echte_groesse", 1.75)
            fx_calib = calibration_data.get("focal_length_fx", 500.0)
            use_pinhole_math = calibration_data.get("use_pinhole_math", True)
            
            depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]

            # Finde Frontal-Frame anhand 2D Schulter/Hüft Ratio
            best_idx, best_area = 0, 0.0
            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
                valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
                if not valid_y or not valid_x: continue
                
                is_front = False
                if is_valid_point(kps, confs, 2) and is_valid_point(kps, confs, 5) and is_valid_point(kps, confs, 8) and is_valid_point(kps, confs, 11):
                    shoulder_w, hip_w = abs(kps[2][0] - kps[5][0]), abs(kps[8][0] - kps[11][0])
                    if hip_w > 0 and (shoulder_w / hip_w) > frontal_2d_threshold: is_front = True
                
                if not is_front: continue
                area = (max(valid_x) - min(valid_x)) * (max(valid_y) - min(valid_y))
                if area > best_area:
                    best_area, best_idx = area, i
            
            log_messages.append(f"Frontalster Frame (2D Ratio Logik): {best_idx}")

            # Window Scaling
            start_idx = max(0, best_idx - anchor_window)
            end_idx = min(len(pose_metas), best_idx + anchor_window + 1)
            sum_norm, sum_depth, valid_frames_in_window = 0.0, 0.0, 0
            
            for i in range(start_idx, end_idx):
                meta = pose_metas[i]
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
                valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
                if not valid_y or not valid_x: continue
                norm_val = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)
                
                v_idx = min(i, depth_np.shape[0]-1)
                valid_x_d = [kps[idx][0] * W for idx in [1,8,11] if is_valid_point(kps, confs, idx)]
                valid_y_d = [kps[idx][1] * H for idx in [1,8,11] if is_valid_point(kps, confs, idx)]
                depth_vals = [depth_np[v_idx, int(py), int(px)] for px, py in zip(valid_x_d, valid_y_d) if 0 <= int(px) < W and 0 <= int(py) < H]
                frame_depth = float(np.mean(depth_vals)) if depth_vals else 0.5
                if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)
                
                sum_norm += norm_val
                sum_depth += frame_depth
                valid_frames_in_window += 1
                
            if valid_frames_in_window == 0:
                return (pose_data_copy, "Fehler: Anchor-Window ungültig.", video_nlf_data, "{}")
                
            avg_anchor_norm = sum_norm / valid_frames_in_window
            avg_anchor_depth = sum_depth / valid_frames_in_window
            
            if use_pinhole_math and echte_groesse > 0.0:
                expected_norm = (echte_groesse * fx_calib) / avg_anchor_depth
            else:
                expected_norm = (avg_anchor_depth * slope) + intercept
                
            anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0
            log_messages.append(f"2D Skalierungs-Faktor = {anchor_scale:.3f}x")

            kps_best = getattr(pose_metas[best_idx], "kps_body", [])
            c_best = getattr(pose_metas[best_idx], "kps_body_p", None)
            val_y = [kps_best[idx][1] for idx in range(len(kps_best)) if is_valid_point(kps_best, c_best, idx)]
            val_x = [kps_best[idx][0] for idx in range(len(kps_best)) if is_valid_point(kps_best, c_best, idx)]
            if val_y and val_x:
                global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)


        # --- GEMEINSAME ANWENDUNG AUF 2D POSEN ---
        scale_x_factor = anchor_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0
        
        for i, meta in enumerate(pose_metas):
            for attr_name in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr_name, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x_factor
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * anchor_scale

        # --- NLF RENDER CONFIG BAUEN (NLF DATEN BLEIBEN UNBERÜHRT!) ---
        nlf_render_config = {
            "anchor_scale": float(anchor_scale),
            "scale_x_factor": float(scale_x_factor),
            "pivot_x": float(global_pivot_x),
            "pivot_y": float(global_pivot_y)
        }
        config_str = json.dumps(nlf_render_config)
        
        log_messages.append("\n=== NLF 3D DATA DELEGATION LOG ===")
        log_messages.append("Die echten NLF 3D-Daten bleiben UNVERÄNDERT.")
        log_messages.append("Die Skalierungsfaktoren wurden erfolgreich in 'nlf_render_config' geschrieben.")

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseGlobalPerspectiveScalerV40:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V40: Kombiniert V28 (Scoring-System mit 3D NLF) und V38 (Pinhole Math & nlf_render_config)."

    def process(self, video_pose_data, calibration_data, video_depth_map, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_nlf_data=None):
        import copy
        import numpy as np
        import math
        import json

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V40 GLOBAL SCALER LOG (V28 Scoring + V38 Output) ==="]

        if not pose_metas:
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        # Kalibrierungsdaten extrahieren
        slope = calibration_data.get("perspective_slope", 0.0)
        intercept = calibration_data.get("perspective_intercept", 1.0)
        is_inverted = calibration_data.get("is_depth_inverted", False)
        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        echte_groesse = calibration_data.get("echte_groesse", 0.0)
        
        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        def is_valid_point(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else (float(pt[2]) if len(pt)>=3 else 1.0)
            return c >= min_confidence

        # --- SCHRITT 1: V28 SCORING SYSTEM (Mit 3D NLF Check) ---
        log_messages.append("\n--- SUCHE NACH DEM BESTEN FRAME (V28 Punktesystem) ---")
        frame_scores = []
        frame_details = []
        
        pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) else video_nlf_data

        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            
            # Wichtige Punkte checken
            has_ankles = is_valid_point(kps, confs, 10) or is_valid_point(kps, confs, 13)
            has_knees = is_valid_point(kps, confs, 9) or is_valid_point(kps, confs, 12)
            has_feet = any(is_valid_point(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

            # Länge berechnen
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            
            top_y = min(valid_y) if valid_y else None
            bottom_y = max(valid_y) if valid_y else None
            if not include_head and valid_y:
                if is_valid_point(kps, confs, 1):
                    top_y = kps[1][1]
            length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

            # Frontalität prüfen (Mit NLF Toleranz)
            is_frontal = False
            frontal_pts = 0.0
            
            if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                pose_3d = pose_input_3d[i][0] if len(pose_input_3d[i]) > 0 else []
                if len(pose_3d) > 11:
                    # Hüft-Winkel im 3D Raum
                    dx = pose_3d[11][0] - pose_3d[8][0]
                    dz = pose_3d[11][2] - pose_3d[8][2]
                    angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                    if angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)
            elif frontal_method == "2D_Ratio":
                if valid_y and valid_x:
                    w = max(valid_x) - min(valid_x)
                    ratio = w / length if length > 0 else 0.0
                    if ratio >= frontal_2d_threshold:
                        is_frontal = True
                        frontal_pts = ratio * 100.0

            data = {
                'has_feet': has_feet,
                'has_ankles': has_ankles,
                'has_knees': has_knees,
                'is_frontal': is_frontal,
                'length': length,
                'frontal_pts': frontal_pts
            }
            frame_details.append(data)

        # Die von dir genannte originale V28 Punktevergabe
        max_body_length = max([d['length'] for d in frame_details]) if frame_details else 1.0
        if max_body_length == 0: max_body_length = 1.0

        for i, data in enumerate(frame_details):
            waden_pts = 1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0
            schenkel_pts = 500.0 if (waden_pts == 0 and data['has_knees']) else 0.0
            bein_pts = max(waden_pts, schenkel_pts)
            fuss_bonus_pts = 500.0 if (data['has_feet'] and data['is_frontal']) else 0.0
            
            total = bein_pts + fuss_bonus_pts + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            frame_scores.append(total)

        if not frame_scores:
            return (pose_data_copy, "Fehler: Konnte keine Scores berechnen.", video_nlf_data, "{}")

        best_idx = int(np.argmax(frame_scores))
        best_score = frame_scores[best_idx]
        log_messages.append(f"-> Gewinner Frame: {best_idx} (V28 Score: {best_score:.1f})")

        # --- SCHRITT 2: ANCHOR WINDOW UND DURCHSCHNITT BERECHNEN ---
        start_idx = max(0, best_idx - anchor_window)
        end_idx = min(len(pose_metas) - 1, best_idx + anchor_window)
        
        sum_norm = 0.0
        sum_depth = 0.0
        valid_frames_in_window = 0

        for i in range(start_idx, end_idx + 1):
            kps = getattr(pose_metas[i], "kps_body", [])
            confs = getattr(pose_metas[i], "kps_body_p", None)
            
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            
            top_y = min(valid_y) if valid_y else None
            bottom_y = max(valid_y) if valid_y else None
            if not include_head and valid_y:
                if is_valid_point(kps, confs, 1):
                    top_y = kps[1][1]
            
            if top_y is not None and bottom_y is not None and bottom_y > top_y:
                norm_val = bottom_y - top_y
                
                depth_vals = []
                v_idx = min(i, depth_np.shape[0] - 1)
                for px, py in zip(valid_x, valid_y):
                    ix, iy = int(px), int(py)
                    if 0 <= ix < W and 0 <= iy < H:
                        depth_vals.append(depth_np[v_idx, iy, ix])
                
                frame_depth = float(np.mean(depth_vals)) if depth_vals else 0.5
                if is_inverted:
                    frame_depth = 1.0 / max(frame_depth, 0.0001)
                
                sum_norm += norm_val
                sum_depth += frame_depth
                valid_frames_in_window += 1

        if valid_frames_in_window == 0:
            return (pose_data_copy, "Fehler: Im Anchor-Window konnte keine Tiefe/Norm ermittelt werden.", video_nlf_data, "{}")

        avg_anchor_norm = sum_norm / valid_frames_in_window
        avg_anchor_depth = sum_depth / valid_frames_in_window

        log_messages.append(f"\n--- SKALIERUNG RECHENVORGANG ---")
        log_messages.append(f"Ist-Norm (Video): {avg_anchor_norm:.1f} px")
        log_messages.append(f"Ist-Tiefe (Video): {avg_anchor_depth:.3f} m")

        # --- SCHRITT 3: V38 BERECHNUNG & CONFIG-OUTPUT ---
        if use_pinhole_math and echte_groesse > 0.0:
            log_messages.append("\n--> PINHOLE SKALIERUNG AKTIV")
            expected_norm = (echte_groesse * fx_calib) / avg_anchor_depth
            log_messages.append(f"Soll-Norm = ({echte_groesse:.3f}m * {fx_calib:.2f}) / {avg_anchor_depth:.3f}m = {expected_norm:.1f} px")
        else:
            log_messages.append("\n--> LINEARE SKALIERUNG AKTIV (FALLBACK)")
            expected_norm = (avg_anchor_depth * slope) + intercept
            log_messages.append(f"Soll-Norm = ({avg_anchor_depth:.3f} * {slope:.2f}) + {intercept:.2f} = {expected_norm:.1f} px")

        anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0
        scale_x_factor = anchor_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0
        
        log_messages.append(f"\nFaktor = Soll-Norm / Ist-Norm = {anchor_scale:.3f}x")

        # Global Kamera Pivot finden (vom best_idx Frame)
        global_pivot_x, global_pivot_y = 0.5, 0.5
        if best_idx < len(pose_metas):
            kps_best = getattr(pose_metas[best_idx], "kps_body", [])
            c_best = getattr(pose_metas[best_idx], "kps_body_p", None)
            val_y = [kps_best[idx][1] for idx in range(len(kps_best)) if is_valid_point(kps_best, c_best, idx)]
            val_x = [kps_best[idx][0] for idx in range(len(kps_best)) if is_valid_point(kps_best, c_best, idx)]
            if val_y and val_x:
                global_pivot_x = np.mean(val_x)
                global_pivot_y = max(val_y)  # Tiefster Punkt als Anker (meistens die Füße)

        # 2D Posen updaten
        for i, meta in enumerate(pose_metas):
            for attr_name in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr_name, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x_factor
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * anchor_scale

        # NLF Config bauen (V38 Prinzip)
        nlf_render_config = {
            "anchor_scale": float(anchor_scale),
            "scale_x_factor": float(scale_x_factor),
            "pivot_x": float(global_pivot_x),
            "pivot_y": float(global_pivot_y)
        }
        config_str = json.dumps(nlf_render_config)

        log_messages.append("\n=== NLF 3D DATA DELEGATION LOG ===")
        log_messages.append("Die echten NLF 3D-Daten bleiben UNVERÄNDERT.")
        log_messages.append("Die Skalierungs- und Pivot-Werte wurden erfolgreich als 'nlf_render_config' an den Renderer gesendet.")

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseGlobalPerspectiveScalerV38:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V38: Nutzt die Pinhole-Skalierung und reicht 3D-Daten UNVERÄNDERT an den Renderer weiter."

    def process(self, video_pose_data, calibration_data, video_depth_map, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, scale_2d_axes, video_nlf_data=None):
        import copy
        import numpy as np
        import math
        import json

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V38 GLOBAL SCALER LOG (DEBUG EDITION) ==="]

        if not pose_metas:
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        slope = calibration_data.get("perspective_slope", 0.0)
        intercept = calibration_data.get("perspective_intercept", 1.0)
        is_inverted = calibration_data.get("is_depth_inverted", False)
        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        echte_groesse = calibration_data.get("echte_groesse", 0.0)

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]
        
        log_messages.append(f"Image Dimensions (H x W): {H} x {W}")
        log_messages.append(f"Kalibrierung: fx={fx_calib}, echte_groesse={echte_groesse}")

        def is_valid_point(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else (float(pt[2]) if len(pt)>=3 else 1.0)
            return c >= min_confidence

        best_idx = 0
        best_area = 0.0
        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            if valid_y and valid_x:
                area = (max(valid_x) - min(valid_x)) * (max(valid_y) - min(valid_y))
                if area > best_area:
                    best_area = area
                    best_idx = i

        start_idx = max(0, best_idx - anchor_window)
        end_idx = min(len(pose_metas) - 1, best_idx + anchor_window)

        log_messages.append(f"Suche abgeschlossen. Bester Frame (best_idx): {best_idx}")
        log_messages.append(f"Anchor Window: Frame {start_idx} bis {end_idx}")

        sum_norm = 0.0
        sum_depth = 0.0
        valid_frames_in_window = 0

        for i in range(start_idx, end_idx + 1):
            meta = pose_metas[i]
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            
            if not valid_y or not valid_x:
                log_messages.append(f"  Frame {i}: Übersprungen (Keine validen Punkte).")
                continue
                
            norm_val = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)
            
            v_idx = min(i, depth_np.shape[0] - 1)
            
            # --- Hier liest V38 die Punkte für die Tiefe aus ---
            valid_x_d = [kps[idx][0] * W for idx in [1,8,11] if is_valid_point(kps, confs, idx)]
            valid_y_d = [kps[idx][1] * H for idx in [1,8,11] if is_valid_point(kps, confs, idx)]
            
            log_messages.append(f"\n  Frame {i} Depth-Analyse:")
            log_messages.append(f"    Gefundene Norm: {norm_val:.1f} px")
            log_messages.append(f"    Punkte zum Tiefe auslesen (Neck/Hip): X={valid_x_d}, Y={valid_y_d}")
            
            depth_vals = [depth_np[v_idx, int(py), int(px)] for px, py in zip(valid_x_d, valid_y_d) if 0 <= int(px) < W and 0 <= int(py) < H]
            
            log_messages.append(f"    Gefundene depth_vals in DepthMap: {depth_vals}")
            
            frame_depth = float(np.mean(depth_vals)) if depth_vals else 0.5
            log_messages.append(f"    Resultierende Tiefe (Mittelwert oder 0.5 Fallback): {frame_depth:.4f} m")

            if is_inverted:
                frame_depth = 1.0 / max(frame_depth, 0.0001)
                
            sum_norm += norm_val
            sum_depth += frame_depth
            valid_frames_in_window += 1

        if valid_frames_in_window == 0:
            return (pose_data_copy, "Fehler: Anchor-Window ungültig.", video_nlf_data, "{}")

        avg_anchor_norm = sum_norm / valid_frames_in_window
        avg_anchor_depth = sum_depth / valid_frames_in_window

        log_messages.append(f"\n--- SKALIERUNG RECHENVORGANG ---")
        log_messages.append(f"Durchschnittliche Ist-Norm: {avg_anchor_norm:.1f} px")
        log_messages.append(f"Durchschnittliche Ist-Tiefe: {avg_anchor_depth:.4f} m")

        if use_pinhole_math and echte_groesse > 0.0:
            expected_norm = (echte_groesse * fx_calib) / avg_anchor_depth
            log_messages.append(f"Soll-Norm = ({echte_groesse:.3f}m * {fx_calib:.2f}) / {avg_anchor_depth:.4f}m = {expected_norm:.1f} px")
        else:
            expected_norm = (avg_anchor_depth * slope) + intercept
            log_messages.append(f"Soll-Norm = ({avg_anchor_depth:.4f} * {slope:.2f}) + {intercept:.2f} = {expected_norm:.1f} px")

        anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0
        scale_x_factor = anchor_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0
        
        log_messages.append(f"Skalierungs-Faktor = Soll-Norm / Ist-Norm = {anchor_scale:.3f}x")

        # --- Wir berechnen den GLOBALEN KAMERA PIVOT aus dem Anchor Frame ---
        global_pivot_x, global_pivot_y = 0.5, 0.5
        if best_idx < len(pose_metas):
            kps_best = getattr(pose_metas[best_idx], "kps_body", [])
            c_best = getattr(pose_metas[best_idx], "kps_body_p", None)
            val_y = [kps_best[idx][1] for idx in range(len(kps_best)) if is_valid_point(kps_best, c_best, idx)]
            val_x = [kps_best[idx][0] for idx in range(len(kps_best)) if is_valid_point(kps_best, c_best, idx)]
            if val_y and val_x:
                global_pivot_x = np.mean(val_x)
                global_pivot_y = max(val_y)  # Füße im Anchor Frame

        log_messages.append(f"Kamera Pivot X/Y: {global_pivot_x:.1f}, {global_pivot_y:.1f}")

        # --- 1. 2D Daten Skalieren (relativ zum globalen Kamera Pivot) ---
        for i, meta in enumerate(pose_metas):
            for attr_name in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr_name, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x_factor
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * anchor_scale

        # --- 2. CONFIGURATION DATA BAUEN ---
        nlf_render_config = {
            "anchor_scale": float(anchor_scale),
            "scale_x_factor": float(scale_x_factor),
            "pivot_x": float(global_pivot_x),
            "pivot_y": float(global_pivot_y)
        }
        config_str = json.dumps(nlf_render_config)

        log_messages.append("\n=== NLF 3D DATA DELEGATION LOG ===")
        log_messages.append("Genialer Plan aktiv: 3D-Daten bleiben UNVERÄNDERT. Die Kamera-Anweisungen wurden in nlf_render_config verpackt!")

        # Wir reichen die originalen, unbeschädigten NLF-Daten weiter!
        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseCalibrationV24:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_nah": ("IMAGE",),
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_fern": ("IMAGE",),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
                "normalize_bones_to_100": ("BOOLEAN", {"default": True, "tooltip": "Setzt Torso auf 100 und berechnet den Rest in Prozent (für V5 Retargeter)"}),
            },
            "optional": {
                "intrinsics_json": ("STRING", {"forceInput": True, "tooltip": "Intrinsics JSON aus DA3"}),
                "nlf_data_nah": ("NLFPRED",),
                "nlf_data_fern": ("NLFPRED",),
                "config_data": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V24: Vereint V15 Pinhole Math + V22 True 3D Bones + V23 Scaler Bones."

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, normalize_bones_to_100=True, intrinsics_json=None, nlf_data_nah=None, nlf_data_fern=None, config_data="{}"):
        import json
        import math
        import numpy as np

        log_messages = ["=== V24 CALIBRATION LOG (V15 Pinhole + V22/23 Bones) ==="]

        try:
            config = json.loads(config_data)
        except:
            config = {}

        def get_body_metrics(pose_s, pose_u, depth_map):
            meta_s = pose_s.get("pose_metas", [])[0]
            meta_u = pose_u.get("pose_metas", [])[0]
            kps_s = getattr(meta_s, "kps_body", None)
            confs_s = getattr(meta_s, "kps_body_p", None)
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]

            def is_val(kps, confs, idx):
                if kps is None or idx >= len(kps): return False
                pt = kps[idx]
                if pt is None or len(pt) < 2: return False
                c = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
                return c >= min_confidence

            # Norm
            norm = 100.0
            if norm_method == "Torso (Neck-Hip)":
                if is_val(kps_s, confs_s, 1) and is_val(kps_s, confs_s, 8) and is_val(kps_s, confs_s, 11):
                    mid_x = (kps_s[8][0] + kps_s[11][0]) / 2.0
                    mid_y = (kps_s[8][1] + kps_s[11][1]) / 2.0
                    norm = math.sqrt((kps_s[1][0] - mid_x)**2 + (kps_s[1][1] - mid_y)**2)
            else:
                valid_y = [kps_s[i][1] for i in range(len(kps_s)) if is_val(kps_s, confs_s, i)]
                valid_x = [kps_s[i][0] for i in range(len(kps_s)) if is_val(kps_s, confs_s, i)]
                if valid_y and valid_x:
                    norm = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)

            # Depth (from unscaled)
            depth = 0.5
            valid_u_x = [kps_u[idx][0] for idx in [1, 8, 11] if is_val(kps_u, confs_u, idx)]
            valid_u_y = [kps_u[idx][1] for idx in [1, 8, 11] if is_val(kps_u, confs_u, idx)]
            
            if valid_u_x and valid_u_y:
                min_x = int(max(0, min(valid_u_x) * W))
                max_x = int(min(W-1, max(valid_u_x) * W))
                min_y = int(max(0, min(valid_u_y) * H))
                max_y = int(min(H-1, max(valid_u_y) * H))
                if max_x > min_x and max_y > min_y:
                    depth = float(np.mean(depth_np[0, min_y:max_y, min_x:max_x]))
            
            return {"norm": norm, "depth": depth}

        data_nah = get_body_metrics(pose_nah_scaled, pose_nah_unscaled, depth_nah)
        data_fern = get_body_metrics(pose_fern_scaled, pose_fern_unscaled, depth_fern)

        norm_nah, norm_fern = data_nah['norm'], data_fern['norm']
        depth_c, depth_f = data_nah['depth'], data_fern['depth']

        if invert_depth:
            depth_c = 1.0 / max(depth_c, 0.0001)
            depth_f = 1.0 / max(depth_f, 0.0001)

        slope, intercept = 0.0, 1.0
        depth_diff = abs(depth_f - depth_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_f - depth_c)
            intercept = norm_nah - (slope * depth_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_c)

        # --- V15 PINHOLE DELTA RECHNUNG ---
        fx = 500.0
        if intrinsics_json:
            try:
                int_data = json.loads(intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx = float(matrix[0][0])
                        log_messages.append(f"Brennweite (fx) aus DA3 geladen: {fx:.2f}")
            except Exception as e:
                log_messages.append(f"Warnung: Intrinsics JSON Fehler ({e}). Nutze Fallback fx={fx}.")

        log_messages.append("\n--- PINHOLE DELTA RECHNUNG ---")
        # Hier ist der V15 Fix: delta_z genau wie früher berechnen
        delta_z = depth_f - depth_c
        log_messages.append(f"Gemessene metrische Differenz (Delta Z): {delta_z:.3f}m")

        echte_groesse = 0.0
        if use_pinhole_math:
            if delta_z > 0 and (norm_nah - norm_fern) > 0:
                echte_groesse = (delta_z * norm_nah * norm_fern) / (fx * (norm_nah - norm_fern))
                log_messages.append(f"Echte physikalische Größe berechnet: {echte_groesse:.3f}m")
            else:
                echte_groesse = (norm_nah * depth_c) / fx
                log_messages.append(f"Warnung: Delta Z negativ oder Norm-Fehler. Fallback: {echte_groesse:.3f}m")

        log_messages.append(f"\nFinale Norm Nah: {norm_nah:.1f} px | Tiefe: {depth_c:.4f}m")
        log_messages.append(f"Finale Norm Fern: {norm_fern:.1f} px | Tiefe: {depth_f:.4f}m")

        # --- V22/V23 2D BONES BERECHNUNG ---
        log_messages.append("\n--- 2D BONE EXTRAKTION ---")

        def extract_2d_bones(pose_data):
            try:
                meta = pose_data.get("pose_metas", [])[0]
                kps = getattr(meta, "kps_body", None)
                if kps is None or len(kps) < 14: return None, None

                def dist_2d(idx1, idx2):
                    if idx1 >= len(kps) or idx2 >= len(kps): return 0.0
                    p1, p2 = kps[idx1], kps[idx2]
                    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2: return 0.0
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                mid_hip_x = (kps[8][0] + kps[11][0]) / 2.0
                mid_hip_y = (kps[8][1] + kps[11][1]) / 2.0
                torso_len = math.sqrt((kps[1][0] - mid_hip_x)**2 + (kps[1][1] - mid_hip_y)**2)
                
                if torso_len <= 0: return None, None

                raw_bones = {
                    "torso": torso_len,
                    "shoulder_width": dist_2d(2, 5),
                    "hip_width": dist_2d(8, 11),
                    "r_arm": dist_2d(2, 3), "r_forearm": dist_2d(3, 4),
                    "l_arm": dist_2d(5, 6), "l_forearm": dist_2d(6, 7),
                    "r_thigh": dist_2d(8, 9), "r_calf": dist_2d(9, 10),
                    "l_thigh": dist_2d(11, 12), "l_calf": dist_2d(12, 13)
                }

                # Symmetrie erzwingen (Aus V22)
                sym_bones = {
                    "torso": raw_bones["torso"],
                    "shoulder_width": raw_bones["shoulder_width"],
                    "hip_width": raw_bones["hip_width"],
                    "r_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "l_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "r_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "l_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "r_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "l_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "r_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0,
                    "l_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0
                }

                # Optional: Torso auf 100 setzen (Aus V22)
                norm_bones = {}
                if normalize_bones_to_100:
                    for k, v in sym_bones.items():
                        norm_bones[k] = (v / sym_bones["torso"]) * 100.0
                else:
                    norm_bones = sym_bones.copy()

                # Gebe BEIDES zurück (Unskaliert für V23 Scaler, Normalisiert für V22 Retargeter)
                return sym_bones, norm_bones
            except Exception as e:
                log_messages.append(f"Fehler bei Bone-Extraktion: {e}")
                return None, None

        # Priorisiere Fern-Bild für saubere Knochen (wie immer)
        unscaled_bones, true_3d_bones = extract_2d_bones(pose_fern_scaled)
        if not unscaled_bones:
            unscaled_bones, true_3d_bones = extract_2d_bones(pose_nah_scaled)
            log_messages.append("Nutze Nah-Bild für Bones (Fern fehlgeschlagen).")
        else:
            log_messages.append("2D-Proportionen aus Fern-Bild extrahiert.")

        total_3d_height = 0.0
        bone_length_for_scaler = {}
        
        if unscaled_bones:
            # Das ist das V23 Feature, das dem Scaler gefehlt hat
            bone_length_for_scaler = {
                "torso": unscaled_bones["torso"],
                "thigh": unscaled_bones["r_thigh"],
                "calf": unscaled_bones["r_calf"]
            }
            log_messages.append(f"-> Unscaled Torso: {bone_length_for_scaler['torso']:.2f}")
            log_messages.append(f"-> Unscaled Thigh: {bone_length_for_scaler['thigh']:.2f}")
            log_messages.append(f"-> Unscaled Calf: {bone_length_for_scaler['calf']:.2f}")
            
            total_3d_height = unscaled_bones["torso"] + unscaled_bones["r_thigh"] + unscaled_bones["r_calf"] + config.get("head_allowance_3d", 150.0)

        calib_data = {
            "perspective_slope": slope,
            "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth,
            "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math,
            "focal_length_fx": fx,
            "echte_groesse": echte_groesse,
            "true_3d_bones": true_3d_bones or {},
            "bone_length_for_scaler": bone_length_for_scaler or {},
            "total_3d_height": total_3d_height,
            "config": config
        }

        return (calib_data, "\n".join(log_messages))


class PoseCalibrationV25:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_nah": ("IMAGE",),
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_fern": ("IMAGE",),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
                "normalize_bones_to_100": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "intrinsics_json": ("STRING", {"forceInput": True}),
                "nlf_data_nah": ("NLFPRED",),
                "nlf_data_fern": ("NLFPRED",),
                "config_data": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V28: Nutzt das exakte Unscaled-Skelett als Maske für perfekte Tiefe."

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, normalize_bones_to_100=True, intrinsics_json=None, nlf_data_nah=None, nlf_data_fern=None, config_data="{}"):
        import json
        import math
        import numpy as np

        log_messages = ["=== V28 CALIBRATION LOG (SKELETON MASK DEPTH) ==="]

        try:
            config = json.loads(config_data)
        except:
            config = {}

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
            return c >= min_confidence

        # --- DIE NEUE SKELETT-MASKE ---
        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [
                (0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),
                (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)
            ]
            depth_vals = []
            
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    
                    # Berechne die Pixel auf der Linie zwischen den Gelenken
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px = int(x1 + t * (x2 - x1))
                            py = int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            
            # Falls gar keine Knochen gefunden wurden (Fallback auf Einzelpunkte)
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            
            if depth_vals:
                return float(np.mean(depth_vals))
            return 0.5

        def get_body_metrics(pose_s, pose_u, depth_map):
            meta_s = pose_s.get("pose_metas", [])[0] if pose_s.get("pose_metas") else None
            meta_u = pose_u.get("pose_metas", [])[0] if pose_u.get("pose_metas") else None
            if not meta_s or not meta_u: return {"norm": 100.0, "depth": 0.5}

            kps_s = getattr(meta_s, "kps_body", None)
            confs_s = getattr(meta_s, "kps_body_p", None)
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]

            norm = 100.0
            if norm_method == "Torso (Neck-Hip)":
                if is_val(kps_s, confs_s, 1) and is_val(kps_s, confs_s, 8) and is_val(kps_s, confs_s, 11):
                    mid_x = (kps_s[8][0] + kps_s[11][0]) / 2.0
                    mid_y = (kps_s[8][1] + kps_s[11][1]) / 2.0
                    norm = math.sqrt((kps_s[1][0] - mid_x)**2 + (kps_s[1][1] - mid_y)**2)
            else:
                valid_y = [kps_s[i][1] for i in range(len(kps_s)) if is_val(kps_s, confs_s, i)]
                valid_x = [kps_s[i][0] for i in range(len(kps_s)) if is_val(kps_s, confs_s, i)]
                if valid_y and valid_x:
                    norm = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)

            # Benutze die UNSCALED Pose für die Skelett-Maske
            depth = get_skeleton_depth(kps_u, confs_u, depth_np, 0, W, H)
            
            return {"norm": norm, "depth": depth}

        def extract_2d_bones(pose_data):
            try:
                meta = pose_data.get("pose_metas", [])[0]
                kps = getattr(meta, "kps_body", None)
                if kps is None or len(kps) < 14: return None, None
                def dist_2d(idx1, idx2):
                    if idx1 >= len(kps) or idx2 >= len(kps): return 0.0
                    p1, p2 = kps[idx1], kps[idx2]
                    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2: return 0.0
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                mid_hip_x = (kps[8][0] + kps[11][0]) / 2.0
                mid_hip_y = (kps[8][1] + kps[11][1]) / 2.0
                
                raw_bones = {
                    "head": dist_2d(0, 1),
                    "torso": math.sqrt((kps[1][0] - mid_hip_x)**2 + (kps[1][1] - mid_hip_y)**2),
                    "shoulder_width": dist_2d(2, 5), "hip_width": dist_2d(8, 11),
                    "r_arm": dist_2d(2, 3), "r_forearm": dist_2d(3, 4),
                    "l_arm": dist_2d(5, 6), "l_forearm": dist_2d(6, 7),
                    "r_thigh": dist_2d(8, 9), "r_calf": dist_2d(9, 10),
                    "l_thigh": dist_2d(11, 12), "l_calf": dist_2d(12, 13)
                }

                sym_bones = {
                    "head": raw_bones["head"], "torso": raw_bones["torso"],
                    "shoulder_width": raw_bones["shoulder_width"], "hip_width": raw_bones["hip_width"],
                    "r_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "l_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "r_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "l_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "r_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "l_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "r_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0,
                    "l_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0
                }

                norm_bones = {}
                if normalize_bones_to_100:
                    for k, v in sym_bones.items():
                        norm_bones[k] = (v / sym_bones["torso"]) * 100.0 if sym_bones["torso"] > 0 else 0
                else: norm_bones = sym_bones.copy()

                return sym_bones, norm_bones
            except Exception as e: return None, None

        log_messages.append(f"Methode: {norm_method}")

        data_nah = get_body_metrics(pose_nah_scaled, pose_nah_unscaled, depth_nah)
        data_fern = get_body_metrics(pose_fern_scaled, pose_fern_unscaled, depth_fern)
        norm_nah, norm_fern = data_nah['norm'], data_fern['norm']
        depth_c, depth_f = data_nah['depth'], data_fern['depth']

        if invert_depth:
            depth_c = 1.0 / max(depth_c, 0.0001)
            depth_f = 1.0 / max(depth_f, 0.0001)

        unscaled_bones_nah, _ = extract_2d_bones(pose_nah_scaled)
        unscaled_bones_fern, true_3d_bones = extract_2d_bones(pose_fern_scaled)

        if unscaled_bones_nah and unscaled_bones_fern:
            torso_nah = unscaled_bones_nah["torso"]
            torso_fern = unscaled_bones_fern["torso"]
            if torso_fern > 0:
                torso_faktor = torso_nah / torso_fern
                extrapolated_nah = norm_fern * torso_faktor
                norm_nah = extrapolated_nah
        else:
            if not unscaled_bones_fern: unscaled_bones_fern, true_3d_bones = unscaled_bones_nah, _

        slope, intercept = 0.0, 1.0
        depth_diff = abs(depth_f - depth_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_f - depth_c)
            intercept = norm_nah - (slope * depth_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_c)

        fx = 500.0
        if intrinsics_json:
            try:
                int_data = json.loads(intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx = float(matrix[0][0])
                        log_messages.append(f"\nBrennweite (fx) aus DA3 geladen: {fx:.2f}")
            except: pass

        delta_z = depth_f - depth_c
        log_messages.append("\n--- PINHOLE DELTA RECHNUNG ---")
        log_messages.append(f"Skelett-Masken Tiefe Nah: {depth_c:.4f}m")
        log_messages.append(f"Skelett-Masken Tiefe Fern: {depth_f:.4f}m")
        log_messages.append(f"Gemessene metrische Differenz (Delta Z): {delta_z:.3f}m")

        echte_groesse = 0.0
        if use_pinhole_math and delta_z > 0 and (norm_nah - norm_fern) > 0:
            echte_groesse = (delta_z * norm_nah * norm_fern) / (fx * (norm_nah - norm_fern))
            log_messages.append(f"Echte physikalische Größe berechnet: {echte_groesse:.3f}m")
        else:
            echte_groesse = (norm_nah * depth_c) / fx
            log_messages.append(f"Warnung: Delta Z negativ. Fallback Größe: {echte_groesse:.3f}m")

        log_messages.append(f"\n=== ERGEBNIS ===")
        log_messages.append(f"Finale Norm Nah: {norm_nah:.1f} px | Tiefe Nah: {depth_c:.4f}m")
        log_messages.append(f"Finale Norm Fern: {norm_fern:.1f} px | Tiefe Fern: {depth_f:.4f}m")

        bone_length_for_scaler = {}
        bone_lengths_in_meters = {}

        if unscaled_bones_fern:
            bone_length_for_scaler = {
                "head": unscaled_bones_fern["head"],
                "torso": unscaled_bones_fern["torso"],
                "thigh": unscaled_bones_fern["r_thigh"],
                "calf": unscaled_bones_fern["r_calf"]
            }
            total_px = sum(bone_length_for_scaler.values())
            if total_px > 0 and echte_groesse > 0:
                for k, px_val in bone_length_for_scaler.items():
                    bone_lengths_in_meters[k] = (px_val / total_px) * echte_groesse

        calib_data = {
            "perspective_slope": slope, "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth, "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math, "focal_length_fx": fx,
            "echte_groesse": echte_groesse,
            "true_3d_bones": true_3d_bones or {},
            "bone_length_for_scaler": bone_length_for_scaler or {},
            "bone_lengths_in_meters": bone_lengths_in_meters or {},
            "total_3d_height": sum(bone_length_for_scaler.values()) if bone_length_for_scaler else 0.0,
            "config": config
        }

        return (calib_data, "\n".join(log_messages))


class PoseGlobalPerspectiveScalerV41:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V41: V40 Frontal-Scoring kombiniert mit Dynamic Fullbody Knochen-Skalierung."

    def process(self, video_pose_data, calibration_data, video_depth_map, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_nlf_data=None):
        import copy
        import numpy as np
        import math
        import json

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V41 GLOBAL SCALER LOG (V40 SCORING + DYNAMIC FULLBODY) ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        bone_m = calibration_data.get("bone_lengths_in_meters", {})
        is_inverted = calibration_data.get("is_depth_inverted", False)

        if not bone_m:
            return (pose_data_copy, "Fehler: Calibration V25 Daten fehlen (bone_lengths_in_meters).", video_nlf_data, "{}")

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        def is_valid_point(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        # --- SCHRITT 1: V40/V28 SCORING SYSTEM WIEDERHERGESTELLT ---
        log_messages.append("\n--- SUCHE NACH DEM BESTEN FRAME (V40 Punktesystem inkl. 3D NLF) ---")
        frame_scores = []
        frame_details = []
        
        pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) else video_nlf_data

        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            
            has_ankles = is_valid_point(kps, confs, 10) or is_valid_point(kps, confs, 13)
            has_knees = is_valid_point(kps, confs, 9) or is_valid_point(kps, confs, 12)
            has_feet = any(is_valid_point(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            
            top_y = min(valid_y) if valid_y else None
            bottom_y = max(valid_y) if valid_y else None
            if not include_head and valid_y:
                if is_valid_point(kps, confs, 1):
                    top_y = kps[1][1]
            length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

            is_frontal = False
            frontal_pts = 0.0
            
            if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                pose_3d = pose_input_3d[i][0] if len(pose_input_3d[i]) > 0 else []
                if len(pose_3d) > 11:
                    dx = pose_3d[11][0] - pose_3d[8][0]
                    dz = pose_3d[11][2] - pose_3d[8][2]
                    angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                    if angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)
            elif frontal_method == "2D_Ratio":
                if valid_y and valid_x:
                    w = max(valid_x) - min(valid_x)
                    ratio = w / length if length > 0 else 0.0
                    if ratio >= frontal_2d_threshold:
                        is_frontal = True
                        frontal_pts = ratio * 100.0

            data = {
                'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees,
                'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts
            }
            frame_details.append(data)

        max_body_length = max([d['length'] for d in frame_details]) if frame_details else 1.0
        if max_body_length == 0: max_body_length = 1.0

        for i, data in enumerate(frame_details):
            waden_pts = 1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0
            schenkel_pts = 500.0 if (waden_pts == 0 and data['has_knees']) else 0.0
            bein_pts = max(waden_pts, schenkel_pts)
            fuss_bonus_pts = 500.0 if (data['has_feet'] and data['is_frontal']) else 0.0
            total = bein_pts + fuss_bonus_pts + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            frame_scores.append(total)

        best_idx = int(np.argmax(frame_scores))
        start_idx = max(0, best_idx - anchor_window)
        end_idx = min(len(pose_metas) - 1, best_idx + anchor_window)
        log_messages.append(f"-> Gewinner Frame gefunden: {best_idx} (Score: {frame_scores[best_idx]:.1f})")

        # --- SCHRITT 2: DYNAMIC FULLBODY RECHNUNG IM ANCHOR WINDOW ---
        sum_scale_factors = 0.0
        valid_frames = 0

        for i in range(start_idx, end_idx + 1):
            kps = getattr(pose_metas[i], "kps_body", [])
            confs = getattr(pose_metas[i], "kps_body_p", None)
            
            frame_ist_px = 0.0
            frame_soll_m = 0.0
            visible_parts = []

            # Kopf nur addieren, wenn include_head aktiv ist
            if include_head and is_valid_point(kps, confs, 0) and is_valid_point(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1)
                frame_soll_m += bone_m.get("head", 0)
                visible_parts.append("Kopf")
                
            if is_valid_point(kps, confs, 1) and is_valid_point(kps, confs, 8) and is_valid_point(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0)
                visible_parts.append("Torso")
                
            if is_valid_point(kps, confs, 8) and is_valid_point(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9)
                frame_soll_m += bone_m.get("thigh", 0)
                visible_parts.append("Oberschenkel")
                
            if is_valid_point(kps, confs, 9) and is_valid_point(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10)
                frame_soll_m += bone_m.get("calf", 0)
                visible_parts.append("Wade")

            if frame_ist_px == 0 or frame_soll_m == 0: continue

            # Tiefe auslesen
            v_idx = min(i, depth_np.shape[0] - 1)
            depth_vals = []
            for idx in [1, 8, 11]:
                if is_valid_point(kps, confs, idx):
                    ix, iy = int(kps[idx][0]), int(kps[idx][1])
                    if 0 <= ix < W and 0 <= iy < H:
                        depth_vals.append(depth_np[v_idx, iy, ix])
            
            frame_depth = float(np.mean(depth_vals)) if depth_vals else 0.5
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            expected_px = (frame_soll_m * fx_calib) / frame_depth
            scale_factor = expected_px / frame_ist_px
            
            sum_scale_factors += scale_factor
            valid_frames += 1
            
            log_messages.append(f"\n  Frame {i} Analyse:")
            log_messages.append(f"    Sichtbar: {', '.join(visible_parts)}")
            log_messages.append(f"    Ist-Pixel (Addiert): {frame_ist_px:.1f} px")
            log_messages.append(f"    Ist-Meter (Addiert): {frame_soll_m:.3f} m")
            log_messages.append(f"    Tiefe: {frame_depth:.3f} m -> Soll-Pixel: {expected_px:.1f} px")
            log_messages.append(f"    Lokaler Faktor: {scale_factor:.3f}x")

        if valid_frames == 0:
            return (pose_data_copy, "Fehler: Keine validen Körperteile im Anchor-Window gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor: {final_scale:.3f}x")

        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b = getattr(pose_metas[best_idx], "kps_body", [])
        c_b = getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_valid_point(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_valid_point(kps_b, c_b, idx)]
        if val_y and val_x:
            global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        config_str = json.dumps({
            "anchor_scale": float(final_scale), "scale_x_factor": float(scale_x),
            "pivot_x": float(global_pivot_x), "pivot_y": float(global_pivot_y)
        })

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseCalibrationManipulator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "calibration_data": ("POSE_CALIBRATION",),
                "echte_groesse_override": ("FLOAT", {"default": 2.10, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip": "Erzwingt eine neue echte Größe in Metern."}),
                "enable_override": ("BOOLEAN", {"default": False, "tooltip": "Wenn False, werden die Originaldaten durchgeleitet."})
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING")
    RETURN_NAMES = ("modified_calibration", "log_output")
    FUNCTION = "manipulate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "Manipuliert die echte_groesse nachträglich und passt alle abhängigen Meter-Werte proportional an."

    def manipulate(self, calibration_data, echte_groesse_override, enable_override):
        import copy
        import json
        
        # Tiefkopie, um das Original nicht zu zerstören
        calib = copy.deepcopy(calibration_data)
        log_messages = ["=== CALIBRATION MANIPULATOR LOG ==="]

        if not enable_override:
            log_messages.append("Bypass aktiv: Originaldaten werden unverändert weitergeleitet.")
            return (calib, "\n".join(log_messages))

        alte_groesse = calib.get("echte_groesse", 1.0)
        
        if alte_groesse <= 0:
            log_messages.append("Fehler: Originale echte_groesse ist <= 0. Manipulation abgebrochen.")
            return (calib, "\n".join(log_messages))

        # Skalierungsfaktor berechnen
        faktor = echte_groesse_override / alte_groesse
        
        log_messages.append(f"Originale Größe: {alte_groesse:.3f}m")
        log_messages.append(f"Neue Ziel-Größe: {echte_groesse_override:.3f}m")
        log_messages.append(f"Manipulations-Faktor: {faktor:.4f}x")

        # 1. Hauptgröße überschreiben
        calib["echte_groesse"] = echte_groesse_override

        # 2. Metrische Knochen proportional anpassen
        bone_m = calib.get("bone_lengths_in_meters", {})
        if bone_m:
            log_messages.append("\n--- NEUE METRISCHE KNOCHEN ---")
            for key, val in bone_m.items():
                neuer_wert = val * faktor
                bone_m[key] = neuer_wert
                log_messages.append(f"{key.capitalize()}: {val:.3f}m -> {neuer_wert:.3f}m")
            calib["bone_lengths_in_meters"] = bone_m
        
        # 3. Knochen-Längen für den Scaler (Pixel und 3D Height) bleiben unangetastet!
        # Warum? Die Pixel im Video bleiben gleich groß. Wir ändern nur die Interpretation, 
        # wie viele Meter diese Pixel in der realen Welt darstellen.
        log_messages.append("\nPixel-Werte (bone_length_for_scaler) bleiben unangetastet.")
        log_messages.append("Proportionen (true_3d_bones) bleiben unangetastet.")

        return (calib, "\n".join(log_messages))


class PoseCalibrationV29:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_nah": ("IMAGE",),
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "depth_fern": ("IMAGE",),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
                "normalize_bones_to_100": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "intrinsics_json": ("STRING", {"forceInput": True}),
                "nlf_data_nah": ("NLFPRED",),
                "nlf_data_fern": ("NLFPRED",),
                "config_data": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V29: Berechnet die Ist-Norm aus der echten Knochen-Summe (Keine Diagonale mehr!)."

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, normalize_bones_to_100=True, intrinsics_json=None, nlf_data_nah=None, nlf_data_fern=None, config_data="{}"):
        import json
        import math
        import numpy as np

        log_messages = ["=== V29 CALIBRATION LOG (PURE BONE-SUM NORM & SKELETON DEPTH) ==="]

        try:
            config = json.loads(config_data)
        except:
            config = {}

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
            return c >= min_confidence

        # --- 1. SKELETT-MASKE (TIEFENAUSLESUNG WIE IN V28) ---
        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [
                (0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),
                (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)
            ]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px = int(x1 + t * (x2 - x1))
                            py = int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        def get_depth_for_pose(pose_u, depth_img):
            meta_u = pose_u.get("pose_metas", [])[0] if pose_u.get("pose_metas") else None
            if not meta_u: return 0.5
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            depth_np = depth_img.cpu().numpy() if hasattr(depth_img, 'cpu') else depth_img
            H, W = depth_np.shape[1], depth_np.shape[2]
            return get_skeleton_depth(kps_u, confs_u, depth_np, 0, W, H)

        # --- 2. KNOCHEN EXTRAHIEREN ---
        def extract_2d_bones(pose_data):
            try:
                meta = pose_data.get("pose_metas", [])[0]
                kps = getattr(meta, "kps_body", None)
                if kps is None or len(kps) < 14: return None, None
                def dist_2d(idx1, idx2):
                    if idx1 >= len(kps) or idx2 >= len(kps): return 0.0
                    p1, p2 = kps[idx1], kps[idx2]
                    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2: return 0.0
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                mid_hip_x = (kps[8][0] + kps[11][0]) / 2.0
                mid_hip_y = (kps[8][1] + kps[11][1]) / 2.0
                
                raw_bones = {
                    "head": dist_2d(0, 1),
                    "torso": math.sqrt((kps[1][0] - mid_hip_x)**2 + (kps[1][1] - mid_hip_y)**2),
                    "shoulder_width": dist_2d(2, 5), "hip_width": dist_2d(8, 11),
                    "r_arm": dist_2d(2, 3), "r_forearm": dist_2d(3, 4),
                    "l_arm": dist_2d(5, 6), "l_forearm": dist_2d(6, 7),
                    "r_thigh": dist_2d(8, 9), "r_calf": dist_2d(9, 10),
                    "l_thigh": dist_2d(11, 12), "l_calf": dist_2d(12, 13)
                }

                sym_bones = {
                    "head": raw_bones["head"], "torso": raw_bones["torso"],
                    "shoulder_width": raw_bones["shoulder_width"], "hip_width": raw_bones["hip_width"],
                    "r_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "l_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "r_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "l_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "r_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "l_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "r_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0,
                    "l_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0
                }

                norm_bones = {}
                if normalize_bones_to_100:
                    for k, v in sym_bones.items():
                        norm_bones[k] = (v / sym_bones["torso"]) * 100.0 if sym_bones["torso"] > 0 else 0
                else: norm_bones = sym_bones.copy()

                return sym_bones, norm_bones
            except Exception as e: return None, None

        log_messages.append(f"Methode: {norm_method}")
        log_messages.append("WARNUNG AN BOUNDING-BOX: Du bist gefeuert! Es zählen ab sofort nur noch echte Knochenlängen.")

        unscaled_bones_nah, _ = extract_2d_bones(pose_nah_scaled)
        unscaled_bones_fern, true_3d_bones = extract_2d_bones(pose_fern_scaled)

        # --- 3. NORM BERECHNUNG (NEU: REINE KNOCHEN-SUMME) ---
        def calc_norm_from_bones(bones, method):
            if not bones: return 100.0
            if method == "Torso (Neck-Hip)": return bones["torso"]
            return bones["head"] + bones["torso"] + bones["r_thigh"] + bones["r_calf"]

        norm_nah_raw = calc_norm_from_bones(unscaled_bones_nah, norm_method)
        norm_fern = calc_norm_from_bones(unscaled_bones_fern, norm_method)
        norm_nah = norm_nah_raw

        depth_c = get_depth_for_pose(pose_nah_unscaled, depth_nah)
        depth_f = get_depth_for_pose(pose_fern_unscaled, depth_fern)

        if invert_depth:
            depth_c = 1.0 / max(depth_c, 0.0001)
            depth_f = 1.0 / max(depth_f, 0.0001)

        # --- 4. EXTRAPOLATION ---
        if unscaled_bones_nah and unscaled_bones_fern:
            torso_nah = unscaled_bones_nah["torso"]
            torso_fern = unscaled_bones_fern["torso"]
            if torso_fern > 0:
                torso_faktor = torso_nah / torso_fern
                extrapolated_nah = norm_fern * torso_faktor
                log_messages.append("\n--- EXTRAPOLATION RECHENVORGANG ---")
                log_messages.append(f"Torso-Faktor = Torso Nah ({torso_nah:.1f}) / Torso Fern ({torso_fern:.1f}) = {torso_faktor:.3f}")
                log_messages.append(f"Extrapoliere Knochen-Summe 'Nah' = Fern-Norm ({norm_fern:.1f}) * Torso-Faktor ({torso_faktor:.3f}) = {extrapolated_nah:.1f} px")
                norm_nah = extrapolated_nah
        else:
            if not unscaled_bones_fern: unscaled_bones_fern, true_3d_bones = unscaled_bones_nah, _

        slope, intercept = 0.0, 1.0
        depth_diff = abs(depth_f - depth_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_f - depth_c)
            intercept = norm_nah - (slope * depth_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_c)

        fx = 500.0
        if intrinsics_json:
            try:
                int_data = json.loads(intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx = float(matrix[0][0])
                        log_messages.append(f"\nBrennweite (fx) aus DA3 geladen: {fx:.2f}")
            except: pass

        # --- 5. PINHOLE DELTA RECHNUNG ---
        delta_z = depth_f - depth_c
        log_messages.append("\n--- PINHOLE DELTA RECHNUNG ---")
        log_messages.append(f"Skelett-Masken Tiefe Nah: {depth_c:.4f}m")
        log_messages.append(f"Skelett-Masken Tiefe Fern: {depth_f:.4f}m")
        log_messages.append(f"Gemessene metrische Differenz (Delta Z): {delta_z:.3f}m")

        echte_groesse = 0.0
        if use_pinhole_math and delta_z > 0 and (norm_nah - norm_fern) > 0:
            echte_groesse = (delta_z * norm_nah * norm_fern) / (fx * (norm_nah - norm_fern))
            log_messages.append(f"Echte physikalische Knochen-Größe berechnet: {echte_groesse:.3f}m")
        else:
            echte_groesse = (norm_nah * depth_c) / fx
            log_messages.append(f"Warnung: Delta Z negativ. Fallback Größe: {echte_groesse:.3f}m")

        log_messages.append(f"\n=== ERGEBNIS ===")
        log_messages.append(f"Finale Norm Nah (Extrapoliert): {norm_nah:.1f} px | Tiefe Nah: {depth_c:.4f}m")
        log_messages.append(f"Finale Norm Fern (Knochensumme): {norm_fern:.1f} px | Tiefe Fern: {depth_f:.4f}m")

        bone_length_for_scaler = {}
        bone_lengths_in_meters = {}

        if unscaled_bones_fern:
            bone_length_for_scaler = {
                "head": unscaled_bones_fern["head"],
                "torso": unscaled_bones_fern["torso"],
                "thigh": unscaled_bones_fern["r_thigh"],
                "calf": unscaled_bones_fern["r_calf"]
            }
            total_px = sum(bone_length_for_scaler.values())
            if total_px > 0 and echte_groesse > 0:
                log_messages.append("\n--- METRISCHE KNOCHEN VERTEILUNG ---")
                for k, px_val in bone_length_for_scaler.items():
                    meter_val = (px_val / total_px) * echte_groesse
                    bone_lengths_in_meters[k] = meter_val
                    log_messages.append(f"{k.capitalize()}: {px_val:.1f} px  =>  {meter_val:.3f} m")

        calib_data = {
            "perspective_slope": slope, "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth, "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math, "focal_length_fx": fx,
            "echte_groesse": echte_groesse,
            "true_3d_bones": true_3d_bones or {},
            "bone_length_for_scaler": bone_length_for_scaler or {},
            "bone_lengths_in_meters": bone_lengths_in_meters or {},
            "total_3d_height": sum(bone_length_for_scaler.values()) if bone_length_for_scaler else 0.0,
            "config": config
        }

        return (calib_data, "\n".join(log_messages))


class PoseGlobalPerspectiveScalerV43:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V45: V28 Perfect 3D-Format-Detector + Skelett-Masken Tiefe + Dynamic Fullbody."

    def process(self, video_pose_data, calibration_data, video_depth_map, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_nlf_data=None):
        import copy
        import numpy as np
        import math
        import json

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V45 GLOBAL SCALER LOG (PERFECT FORMAT DETECTOR) ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        bone_m = calibration_data.get("bone_lengths_in_meters", {})
        is_inverted = calibration_data.get("is_depth_inverted", False)

        if not bone_m:
            return (pose_data_copy, "Fehler: Calibration Daten (bone_lengths_in_meters) fehlen.", video_nlf_data, "{}")

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        # --- SKELETT-MASKE ---
        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [
                (0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),
                (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)
            ]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px = int(x1 + t * (x2 - x1))
                            py = int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        # --- STUFE 1: ALLE FRAMES ANALYSIEREN UND FRONTALE FILTERN ---
        all_frames_data = []
        frontal_indices = []
        pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) else video_nlf_data

        log_messages.append("\n--- WINKEL-RADAR (3D NLF Format-Check wie in V28) ---")

        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            
            has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
            has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
            has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
            top_y = min(valid_y) if valid_y else None
            bottom_y = max(valid_y) if valid_y else None
            if not include_head and valid_y:
                if is_val(kps, confs, 1): top_y = kps[1][1]
            length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

            is_frontal = False
            frontal_pts = 0.0
            
            if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                pose_3d_frame = pose_input_3d[i]
                if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                    person_3d = pose_3d_frame[0]
                    num_joints = len(person_3d)
                    
                    # === DIE PERFEKTE V28 LOGIK IST ZURÜCK ===
                    idx_r, idx_l = 2, 5
                    format_name = "OpenPose"
                    if num_joints == 17:
                        idx_r, idx_l = 11, 14
                        format_name = "H36M"
                    elif num_joints in [24, 45, 68]:
                        idx_r, idx_l = 16, 17
                        format_name = "SMPL"
                        
                    if num_joints > max(idx_r, idx_l):
                        x_r, z_r = float(person_3d[idx_r][0]), float(person_3d[idx_r][2])
                        x_l, z_l = float(person_3d[idx_l][0]), float(person_3d[idx_l][2])
                        dx = x_r - x_l
                        dz = z_r - z_l
                        angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                        
                        if angle <= frontal_3d_angle_tolerance:
                            is_frontal = True
                            frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                        if i <= 6 or is_frontal:
                            status = "FRONTAL" if is_frontal else "SEITLICH"
                            log_messages.append(f"Frame {i}: Winkel {angle:.1f}° ({format_name} Schultern) -> {status}")

            frame_data = {'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts}
            all_frames_data.append(frame_data)
            
            if is_frontal:
                frontal_indices.append(i)

        log_messages.append("\n--- PASS-FILTER (DER TÜRSTEHER) ---")
        if len(frontal_indices) > 0:
            log_messages.append(f">> PASS-FILTER AKTIV: {len(frontal_indices)} echte frontale Frames gefunden! Alle anderen fliegen raus.")
            candidates = frontal_indices
        else:
            log_messages.append(f">> PASS-FILTER INAKTIV: Kein einziger Frame ist unter {frontal_3d_angle_tolerance}°. Nutze alle Frames als Fallback.")
            candidates = list(range(len(pose_metas)))

        # --- STUFE 2: SCORING NUR FÜR KANDIDATEN ---
        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx = candidates[0]
        best_score = -1.0

        for idx in candidates:
            data = all_frames_data[idx]
            waden_pts = 1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0
            schenkel_pts = 500.0 if (waden_pts == 0 and data['has_knees']) else 0.0
            bein_pts = max(waden_pts, schenkel_pts)
            fuss_bonus_pts = 500.0 if (data['has_feet'] and data['is_frontal']) else 0.0
            
            total_score = bein_pts + fuss_bonus_pts + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        log_messages.append(f"\n-> Gewinner Frame: {best_idx} (Score: {best_score:.1f})")

        # --- STUFE 3: DYNAMIC FULLBODY RECHNUNG MIT SKELETT-MASKE ---
        start_idx = max(0, best_idx - anchor_window)
        end_idx = min(len(pose_metas) - 1, best_idx + anchor_window)
        
        sum_scale_factors = 0.0
        valid_frames = 0

        for i in range(start_idx, end_idx + 1):
            kps = getattr(pose_metas[i], "kps_body", [])
            confs = getattr(pose_metas[i], "kps_body_p", None)
            
            frame_ist_px, frame_soll_m = 0.0, 0.0

            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1); frame_soll_m += bone_m.get("head", 0)
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0)
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9); frame_soll_m += bone_m.get("thigh", 0)
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10); frame_soll_m += bone_m.get("calf", 0)

            if frame_ist_px == 0 or frame_soll_m == 0: continue

            v_idx = min(i, depth_np.shape[0] - 1)
            frame_depth = get_skeleton_depth(kps, confs, depth_np, v_idx, W, H)
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            expected_px = (frame_soll_m * fx_calib) / frame_depth
            scale_factor = expected_px / frame_ist_px
            sum_scale_factors += scale_factor
            valid_frames += 1

        if valid_frames == 0:
            return (pose_data_copy, "Fehler: Keine validen Körperteile gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor: {final_scale:.3f}x")

        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b = getattr(pose_metas[best_idx], "kps_body", [])
        c_b = getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x:
            global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        config_str = json.dumps({
            "anchor_scale": float(final_scale), "scale_x_factor": float(scale_x),
            "pivot_x": float(global_pivot_x), "pivot_y": float(global_pivot_y)
        })

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseGlobalPerspectiveScalerV46:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "PoseData (2D)"}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V46: Wählbare Best-Frame-Quelle (PoseData vs. NLF) + NLF Render Config Output."

    def process(self, video_pose_data, calibration_data, video_depth_map, best_frame_source, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_nlf_data=None):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V46 GLOBAL SCALER LOG (QUELLE: {best_frame_source}) ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        bone_m = calibration_data.get("bone_lengths_in_meters", {})
        is_inverted = calibration_data.get("is_depth_inverted", False)

        if not bone_m:
            return (pose_data_copy, "Fehler: Calibration Daten (bone_lengths_in_meters) fehlen.", video_nlf_data, "{}")

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        # --- SKELETT-MASKE (Für die spätere Tiefenauslesung) ---
        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [
                (0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),
                (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)
            ]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px = int(x1 + t * (x2 - x1))
                            py = int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        # --- STUFE 1: FRAME ANALYSE (SWITCH ZWISCHEN 2D UND 3D NLF) ---
        all_frames_data = []
        frontal_indices = []
        
        # Sicherheits-Check
        if best_frame_source == "NLF (3D SMPL)" and video_nlf_data is None:
            log_messages.append("WARNUNG: 'NLF (3D SMPL)' gewählt, aber keine NLF-Daten vorhanden! Fallback auf PoseData.")
            best_frame_source = "PoseData (2D)"

        if best_frame_source == "NLF (3D SMPL)":
            log_messages.append("\n--- WINKEL-RADAR & SCORING (Quelle: NLF 3D SMPL) ---")
            is_dict = isinstance(video_nlf_data, dict)
            raw_poses = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if is_dict else video_nlf_data

            for i in range(len(pose_metas)):
                if i >= len(raw_poses) or raw_poses[i] is None or len(raw_poses[i]) == 0:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                frame_data = raw_poses[i]
                is_tensor = isinstance(frame_data, torch.Tensor)
                if is_tensor:
                    pts = frame_data[0].cpu().numpy() if frame_data.dim() == 3 else frame_data.cpu().numpy()
                else:
                    arr = np.array(frame_data)
                    pts = arr[0] if arr.ndim == 3 else arr

                def is_val_nlf(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

                has_knees = is_val_nlf(4) or is_val_nlf(5)
                has_ankles = is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles # NLF (SMPL) Näherung

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

                is_frontal = False
                frontal_pts = 0.0
                angle_h, angle_s, max_angle = 90.0, 90.0, 90.0

                if len(pts) >= 18:
                    dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                    angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))

                    dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                    angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))

                    max_angle = max(angle_h, angle_s)

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                if i <= 6 or is_frontal:
                    status = "FRONTAL" if is_frontal else "SEITLICH"
                    log_messages.append(f"Frame {i}: Max-Winkel {max_angle:.1f}° (Hüfte: {angle_h:.1f}°, Schultern: {angle_s:.1f}°) -> {status}")

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        else:
            # POSEDATA (2D) LOGIK (wie in V45)
            log_messages.append("\n--- WINKEL-RADAR & SCORING (Quelle: PoseData 2D + Format Detector) ---")
            pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) and video_nlf_data else None

            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                
                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
                top_y = min(valid_y) if valid_y else None
                bottom_y = max(valid_y) if valid_y else None
                if not include_head and valid_y:
                    if is_val(kps, confs, 1): top_y = kps[1][1]
                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal = False
                frontal_pts = 0.0
                
                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pose_3d_frame = pose_input_3d[i]
                    if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                        person_3d = pose_3d_frame[0]
                        num_joints = len(person_3d)
                        
                        idx_r, idx_l = 2, 5
                        format_name = "OpenPose"
                        if num_joints == 17:
                            idx_r, idx_l = 11, 14
                            format_name = "H36M"
                        elif num_joints in [24, 45, 68]:
                            idx_r, idx_l = 16, 17
                            format_name = "SMPL"
                            
                        if num_joints > max(idx_r, idx_l):
                            x_r, z_r = float(person_3d[idx_r][0]), float(person_3d[idx_r][2])
                            x_l, z_l = float(person_3d[idx_l][0]), float(person_3d[idx_l][2])
                            dx, dz = x_r - x_l, z_r - z_l
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                            
                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                            if i <= 6 or is_frontal:
                                status = "FRONTAL" if is_frontal else "SEITLICH"
                                log_messages.append(f"Frame {i}: Winkel {angle:.1f}° ({format_name} Schultern) -> {status}")

                frame_data = {'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts}
                all_frames_data.append(frame_data)
                
                if is_frontal: frontal_indices.append(i)

        # --- STUFE 2: PASS-FILTER UND GEWINNER ---
        log_messages.append("\n--- PASS-FILTER (DER TÜRSTEHER) ---")
        if len(frontal_indices) > 0:
            log_messages.append(f">> PASS-FILTER AKTIV: {len(frontal_indices)} echte frontale Frames gefunden! Alle anderen fliegen raus.")
            candidates = frontal_indices
        else:
            log_messages.append(f">> PASS-FILTER INAKTIV: Kein einziger Frame ist unter {frontal_3d_angle_tolerance}°. Nutze alle Frames als Fallback.")
            candidates = list(range(len(pose_metas)))

        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx = candidates[0]
        best_score = -1.0

        for idx in candidates:
            data = all_frames_data[idx]
            waden_pts = 1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0
            schenkel_pts = 500.0 if (waden_pts == 0 and data['has_knees']) else 0.0
            bein_pts = max(waden_pts, schenkel_pts)
            fuss_bonus_pts = 500.0 if (data['has_feet'] and data['is_frontal']) else 0.0
            
            total_score = bein_pts + fuss_bonus_pts + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        log_messages.append(f"\n-> Gewinner Frame (Anchor): {best_idx} (Score: {best_score:.1f})")

        # --- STUFE 3: DYNAMIC FULLBODY RECHNUNG MIT SKELETT-MASKE (Nutzt immer PoseData) ---
        start_idx = max(0, best_idx - anchor_window)
        end_idx = min(len(pose_metas) - 1, best_idx + anchor_window)
        
        sum_scale_factors = 0.0
        valid_frames = 0

        for i in range(start_idx, end_idx + 1):
            kps = getattr(pose_metas[i], "kps_body", [])
            confs = getattr(pose_metas[i], "kps_body_p", None)
            
            frame_ist_px, frame_soll_m = 0.0, 0.0

            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1); frame_soll_m += bone_m.get("head", 0)
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0)
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9); frame_soll_m += bone_m.get("thigh", 0)
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10); frame_soll_m += bone_m.get("calf", 0)

            if frame_ist_px == 0 or frame_soll_m == 0: continue

            v_idx = min(i, depth_np.shape[0] - 1)
            frame_depth = get_skeleton_depth(kps, confs, depth_np, v_idx, W, H)
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            expected_px = (frame_soll_m * fx_calib) / frame_depth
            scale_factor = expected_px / frame_ist_px
            sum_scale_factors += scale_factor
            valid_frames += 1

        if valid_frames == 0:
            return (pose_data_copy, "Fehler: Keine validen Körperteile gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor berechnet: {final_scale:.3f}x")

        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b = getattr(pose_metas[best_idx], "kps_body", [])
        c_b = getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x:
            global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        # SKALIERUNG ANWENDEN (Multiplikation der PoseData)
        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        log_messages.append(">> PoseData wurde erfolgreich mit dem Faktor multipliziert.")

        # OUTPUT FÜR NLF (Config String)
        config_str = json.dumps({
            "anchor_scale": float(final_scale), 
            "scale_x_factor": float(scale_x),
            "pivot_x": float(global_pivot_x), 
            "pivot_y": float(global_pivot_y)
        })
        
        log_messages.append(f">> Skalierung ({final_scale:.3f}x) in nlf_render_config geschrieben.")

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseGlobalPerspectiveScalerV47:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "PoseData (2D)"}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V47: V46 + Head-Compensation (Addiert halben Kopf in Pixeln im NLF-Modus)."

    def process(self, video_pose_data, calibration_data, video_depth_map, best_frame_source, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_nlf_data=None):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V47 GLOBAL SCALER LOG (QUELLE: {best_frame_source}) ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        bone_m = calibration_data.get("bone_lengths_in_meters", {})
        is_inverted = calibration_data.get("is_depth_inverted", False)

        if not bone_m:
            return (pose_data_copy, "Fehler: Calibration Daten fehlen.", video_nlf_data, "{}")

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        # Die Pixel-Länge des Kopfes aus V29 abrufen (Für NLF Kompensation)
        head_px_calib = calibration_data.get("bone_length_for_scaler", {}).get("head", 0.0)

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [
                (0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),
                (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)
            ]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px = int(x1 + t * (x2 - x1))
                            py = int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        all_frames_data = []
        frontal_indices = []
        
        if best_frame_source == "NLF (3D SMPL)" and video_nlf_data is None:
            log_messages.append("WARNUNG: 'NLF (3D SMPL)' gewählt, aber keine NLF-Daten vorhanden! Fallback auf PoseData.")
            best_frame_source = "PoseData (2D)"

        if best_frame_source == "NLF (3D SMPL)":
            log_messages.append("\n--- WINKEL-RADAR & SCORING (Quelle: NLF 3D SMPL) ---")
            is_dict = isinstance(video_nlf_data, dict)
            raw_poses = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if is_dict else video_nlf_data

            for i in range(len(pose_metas)):
                if i >= len(raw_poses) or raw_poses[i] is None or len(raw_poses[i]) == 0:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                frame_data = raw_poses[i]
                is_tensor = isinstance(frame_data, torch.Tensor)
                if is_tensor:
                    pts = frame_data[0].cpu().numpy() if frame_data.dim() == 3 else frame_data.cpu().numpy()
                else:
                    arr = np.array(frame_data)
                    pts = arr[0] if arr.ndim == 3 else arr

                def is_val_nlf(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

                has_knees = is_val_nlf(4) or is_val_nlf(5)
                has_ankles = is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0
                
                # DEIN FIX: Wir addieren die halbe Pixel-Kopflänge auf die NLF-Messung drauf!
                if include_head and length > 0:
                    length += (head_px_calib / 2.0)

                is_frontal = False
                frontal_pts = 0.0
                
                if len(pts) >= 18:
                    dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                    angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))

                    dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                    angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))

                    max_angle = max(angle_h, angle_s)

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        else:
            log_messages.append("\n--- WINKEL-RADAR & SCORING (Quelle: PoseData 2D) ---")
            pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) and video_nlf_data else None

            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                
                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
                top_y = min(valid_y) if valid_y else None
                bottom_y = max(valid_y) if valid_y else None
                if not include_head and valid_y:
                    if is_val(kps, confs, 1): top_y = kps[1][1]
                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal = False
                frontal_pts = 0.0
                
                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pose_3d_frame = pose_input_3d[i]
                    if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                        person_3d = pose_3d_frame[0]
                        num_joints = len(person_3d)
                        
                        idx_r, idx_l = 2, 5
                        if num_joints == 17: idx_r, idx_l = 11, 14
                        elif num_joints in [24, 45, 68]: idx_r, idx_l = 16, 17
                            
                        if num_joints > max(idx_r, idx_l):
                            x_r, z_r = float(person_3d[idx_r][0]), float(person_3d[idx_r][2])
                            x_l, z_l = float(person_3d[idx_l][0]), float(person_3d[idx_l][2])
                            dx, dz = x_r - x_l, z_r - z_l
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                            
                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        log_messages.append("\n--- PASS-FILTER (DER TÜRSTEHER) ---")
        if len(frontal_indices) > 0:
            candidates = frontal_indices
        else:
            candidates = list(range(len(pose_metas)))

        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx = candidates[0]
        best_score = -1.0

        for idx in candidates:
            data = all_frames_data[idx]
            waden_pts = 1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0
            schenkel_pts = 500.0 if (waden_pts == 0 and data['has_knees']) else 0.0
            bein_pts = max(waden_pts, schenkel_pts)
            fuss_bonus_pts = 500.0 if (data['has_feet'] and data['is_frontal']) else 0.0
            
            total_score = bein_pts + fuss_bonus_pts + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        log_messages.append(f"\n-> Gewinner Frame (Anchor): {best_idx} (Score: {best_score:.1f})")

        start_idx = max(0, best_idx - anchor_window)
        end_idx = min(len(pose_metas) - 1, best_idx + anchor_window)
        
        sum_scale_factors = 0.0
        valid_frames = 0

        for i in range(start_idx, end_idx + 1):
            kps = getattr(pose_metas[i], "kps_body", [])
            confs = getattr(pose_metas[i], "kps_body_p", None)
            
            frame_ist_px, frame_soll_m = 0.0, 0.0

            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1); frame_soll_m += bone_m.get("head", 0)
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0)
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9); frame_soll_m += bone_m.get("thigh", 0)
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10); frame_soll_m += bone_m.get("calf", 0)

            if frame_ist_px == 0 or frame_soll_m == 0: continue

            v_idx = min(i, depth_np.shape[0] - 1)
            frame_depth = get_skeleton_depth(kps, confs, depth_np, v_idx, W, H)
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            expected_px = (frame_soll_m * fx_calib) / frame_depth
            scale_factor = expected_px / frame_ist_px
            sum_scale_factors += scale_factor
            valid_frames += 1

        if valid_frames == 0:
            return (pose_data_copy, "Fehler: Keine validen Körperteile gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor berechnet: {final_scale:.3f}x")

        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b = getattr(pose_metas[best_idx], "kps_body", [])
        c_b = getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x:
            global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        config_str = json.dumps({
            "anchor_scale": float(final_scale), "scale_x_factor": float(scale_x),
            "pivot_x": float(global_pivot_x), "pivot_y": float(global_pivot_y)
        })

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseGlobalPerspectiveScalerV48:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "PoseData (2D)"}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
                "video_intrinsics_json": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V48: Löst das Kamera-Problem. Akzeptiert Video-Brennweite unabhängig von der Kalibrierung."

    def process(self, video_pose_data, calibration_data, video_depth_map, best_frame_source, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_nlf_data=None, video_intrinsics_json=None):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V48 GLOBAL SCALER LOG (QUELLE: {best_frame_source}) ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        bone_m = calibration_data.get("bone_lengths_in_meters", {})
        is_inverted = calibration_data.get("is_depth_inverted", False)

        if not bone_m:
            return (pose_data_copy, "Fehler: Calibration Daten fehlen.", video_nlf_data, "{}")

        # --- NEU: BRENNWEITE FÜR DAS AKTUELLE VIDEO AUSLESEN ---
        fx_video = fx_calib
        if video_intrinsics_json:
            try:
                int_data = json.loads(video_intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx_video = float(matrix[0][0])
                        if abs(fx_video - fx_calib) > 1.0:
                            log_messages.append(f">> KAMERA-WECHSEL ERKANNT: Nutze neues Objektiv fx={fx_video:.2f} (Kalibrierung war fx={fx_calib:.2f})")
            except: pass

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]
        head_px_calib = calibration_data.get("bone_length_for_scaler", {}).get("head", 0.0)

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        all_frames_data = []
        frontal_indices = []
        
        if best_frame_source == "NLF (3D SMPL)" and video_nlf_data is None:
            best_frame_source = "PoseData (2D)"

        if best_frame_source == "NLF (3D SMPL)":
            is_dict = isinstance(video_nlf_data, dict)
            raw_poses = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if is_dict else video_nlf_data

            for i in range(len(pose_metas)):
                if i >= len(raw_poses) or raw_poses[i] is None or len(raw_poses[i]) == 0:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                frame_data = raw_poses[i]
                is_tensor = isinstance(frame_data, torch.Tensor)
                pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))

                def is_val_nlf(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

                has_knees, has_ankles = is_val_nlf(4) or is_val_nlf(5), is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0
                if include_head and length > 0: length += (head_px_calib / 2.0)

                is_frontal, frontal_pts = False, 0.0
                if len(pts) >= 18:
                    dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                    dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                    max_angle = max(math.degrees(math.atan2(abs(dz_h), abs(dx_h))), math.degrees(math.atan2(abs(dz_s), abs(dx_s))))

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        else:
            pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) and video_nlf_data else None

            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                
                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
                top_y, bottom_y = min(valid_y) if valid_y else None, max(valid_y) if valid_y else None
                if not include_head and valid_y and is_val(kps, confs, 1): top_y = kps[1][1]
                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal, frontal_pts = False, 0.0
                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pose_3d_frame = pose_input_3d[i]
                    if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                        person_3d = pose_3d_frame[0]
                        num_joints = len(person_3d)
                        idx_r, idx_l = 2, 5
                        if num_joints == 17: idx_r, idx_l = 11, 14
                        elif num_joints in [24, 45, 68]: idx_r, idx_l = 16, 17
                            
                        if num_joints > max(idx_r, idx_l):
                            dx, dz = float(person_3d[idx_r][0]) - float(person_3d[idx_l][0]), float(person_3d[idx_r][2]) - float(person_3d[idx_l][2])
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        candidates = frontal_indices if len(frontal_indices) > 0 else list(range(len(pose_metas)))
        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            data = all_frames_data[idx]
            bein_pts = max(1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0, 500.0 if not (data['has_feet'] or data['has_ankles']) and data['has_knees'] else 0.0)
            total_score = bein_pts + (500.0 if data['has_feet'] and data['is_frontal'] else 0.0) + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score: best_score, best_idx = total_score, idx

        start_idx, end_idx = max(0, best_idx - anchor_window), min(len(pose_metas) - 1, best_idx + anchor_window)
        sum_scale_factors, valid_frames = 0.0, 0

        for i in range(start_idx, end_idx + 1):
            kps, confs = getattr(pose_metas[i], "kps_body", []), getattr(pose_metas[i], "kps_body_p", None)
            frame_ist_px, frame_soll_m = 0.0, 0.0

            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1); frame_soll_m += bone_m.get("head", 0)
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0)
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9); frame_soll_m += bone_m.get("thigh", 0)
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10); frame_soll_m += bone_m.get("calf", 0)

            if frame_ist_px == 0 or frame_soll_m == 0: continue

            frame_depth = get_skeleton_depth(kps, confs, depth_np, min(i, depth_np.shape[0] - 1), W, H)
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            # DIE MAGIE: HIER WIRD MIT DER NEUEN VIDEO-KAMERA GERECHNET!
            expected_px = (frame_soll_m * fx_video) / frame_depth
            sum_scale_factors += (expected_px / frame_ist_px)
            valid_frames += 1

        if valid_frames == 0: return (pose_data_copy, "Fehler: Keine validen Körperteile gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        
        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b, c_b = getattr(pose_metas[best_idx], "kps_body", []), getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x: global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        config_str = json.dumps({"anchor_scale": float(final_scale), "scale_x_factor": float(scale_x), "pivot_x": float(global_pivot_x), "pivot_y": float(global_pivot_y)})
        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseGlobalPerspectiveScalerV49:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "PoseData (2D)"}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
                "video_intrinsics_json": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V48: Kamera-Brennweite-Fix + VOLLE DETAILLOGS (Radar & Frame-Analyse)."

    def process(self, video_pose_data, calibration_data, video_depth_map, best_frame_source, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_nlf_data=None, video_intrinsics_json=None):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V48 GLOBAL SCALER LOG (QUELLE: {best_frame_source} | VOLLE LOGS) ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        bone_m = calibration_data.get("bone_lengths_in_meters", {})
        is_inverted = calibration_data.get("is_depth_inverted", False)

        if not bone_m:
            return (pose_data_copy, "Fehler: Calibration Daten fehlen.", video_nlf_data, "{}")

        # --- BRENNWEITE FÜR DAS AKTUELLE VIDEO AUSLESEN ---
        fx_video = fx_calib
        if video_intrinsics_json:
            try:
                int_data = json.loads(video_intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx_video = float(matrix[0][0])
                        if abs(fx_video - fx_calib) > 1.0:
                            log_messages.append(f">> KAMERA-WECHSEL ERKANNT: Nutze neues Objektiv fx={fx_video:.2f} (Kalibrierung war fx={fx_calib:.2f})")
            except: pass

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]
        head_px_calib = calibration_data.get("bone_length_for_scaler", {}).get("head", 0.0)

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        all_frames_data = []
        frontal_indices = []
        
        if best_frame_source == "NLF (3D SMPL)" and video_nlf_data is None:
            best_frame_source = "PoseData (2D)"

        if best_frame_source == "NLF (3D SMPL)":
            log_messages.append("\n--- WINKEL-RADAR (Alle Frames werden protokolliert) ---")
            is_dict = isinstance(video_nlf_data, dict)
            raw_poses = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if is_dict else video_nlf_data

            for i in range(len(pose_metas)):
                if i >= len(raw_poses) or raw_poses[i] is None or len(raw_poses[i]) == 0:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                frame_data = raw_poses[i]
                is_tensor = isinstance(frame_data, torch.Tensor)
                pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))

                def is_val_nlf(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

                has_knees = is_val_nlf(4) or is_val_nlf(5)
                has_ankles = is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0
                
                # Kopf-Kompensation
                if include_head and length > 0: 
                    length += (head_px_calib / 2.0)

                is_frontal, frontal_pts = False, 0.0
                angle_h, angle_s, max_angle = 90.0, 90.0, 90.0
                
                if len(pts) >= 18:
                    dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                    dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                    
                    angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                    angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                    max_angle = max(angle_h, angle_s)

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                status = "FRONTAL (Akzeptiert)" if is_frontal else "SEITLICH (Abgelehnt)"
                log_messages.append(f"Frame {i}: Max-Winkel {max_angle:.1f}° (Hüfte: {angle_h:.1f}°, Schultern: {angle_s:.1f}°) -> {status}")

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        else:
            log_messages.append("\n--- WINKEL-RADAR (Alle Frames werden protokolliert) ---")
            pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) and video_nlf_data else None

            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                
                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
                top_y, bottom_y = min(valid_y) if valid_y else None, max(valid_y) if valid_y else None
                if not include_head and valid_y and is_val(kps, confs, 1): top_y = kps[1][1]
                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal, frontal_pts = False, 0.0
                angle = 90.0
                
                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pose_3d_frame = pose_input_3d[i]
                    if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                        person_3d = pose_3d_frame[0]
                        num_joints = len(person_3d)
                        idx_r, idx_l = 2, 5
                        if num_joints == 17: idx_r, idx_l = 11, 14
                        elif num_joints in [24, 45, 68]: idx_r, idx_l = 16, 17
                            
                        if num_joints > max(idx_r, idx_l):
                            dx, dz = float(person_3d[idx_r][0]) - float(person_3d[idx_l][0]), float(person_3d[idx_r][2]) - float(person_3d[idx_l][2])
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                status = "FRONTAL (Akzeptiert)" if is_frontal else "SEITLICH (Abgelehnt)"
                log_messages.append(f"Frame {i}: Winkel {angle:.1f}° -> {status}")

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        log_messages.append("\n--- PASS-FILTER (DER TÜRSTEHER) ---")
        if len(frontal_indices) > 0:
            log_messages.append(f">> PASS-FILTER AKTIV: {len(frontal_indices)} echte frontale Frames gefunden! Alle anderen fliegen raus.")
            candidates = frontal_indices
        else:
            log_messages.append(f">> PASS-FILTER INAKTIV: Kein Frame erfüllt die Toleranz. Nutze alle Frames.")
            candidates = list(range(len(pose_metas)))

        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            data = all_frames_data[idx]
            bein_pts = max(1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0, 500.0 if not (data['has_feet'] or data['has_ankles']) and data['has_knees'] else 0.0)
            total_score = bein_pts + (500.0 if data['has_feet'] and data['is_frontal'] else 0.0) + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score: best_score, best_idx = total_score, idx

        log_messages.append(f"\n-> Gewinner Frame (Anchor): {best_idx} (Score: {best_score:.1f})")

        start_idx, end_idx = max(0, best_idx - anchor_window), min(len(pose_metas) - 1, best_idx + anchor_window)
        sum_scale_factors, valid_frames = 0.0, 0

        for i in range(start_idx, end_idx + 1):
            kps, confs = getattr(pose_metas[i], "kps_body", []), getattr(pose_metas[i], "kps_body_p", None)
            frame_ist_px, frame_soll_m = 0.0, 0.0
            visible_parts = []

            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1); frame_soll_m += bone_m.get("head", 0)
                visible_parts.append("Kopf")
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0)
                visible_parts.append("Torso")
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9); frame_soll_m += bone_m.get("thigh", 0)
                visible_parts.append("Oberschenkel")
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10); frame_soll_m += bone_m.get("calf", 0)
                visible_parts.append("Wade")

            if frame_ist_px == 0 or frame_soll_m == 0: continue

            frame_depth = get_skeleton_depth(kps, confs, depth_np, min(i, depth_np.shape[0] - 1), W, H)
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            # DIE MAGIE: HIER WIRD MIT DER NEUEN VIDEO-KAMERA GERECHNET!
            expected_px = (frame_soll_m * fx_video) / frame_depth
            scale_factor = expected_px / frame_ist_px
            sum_scale_factors += scale_factor
            valid_frames += 1

            # --- VOLLES LOGGING WIEDER DA! ---
            log_messages.append(f"\n  Frame {i} Analyse:")
            log_messages.append(f"    Sichtbar: {', '.join(visible_parts)}")
            log_messages.append(f"    Ist-Pixel (Knochensumme): {frame_ist_px:.1f} px")
            log_messages.append(f"    Ist-Meter (Knochensumme): {frame_soll_m:.3f} m")
            log_messages.append(f"    Skelett-Tiefe: {frame_depth:.3f} m -> Soll-Pixel: {expected_px:.1f} px")
            log_messages.append(f"    Lokaler Faktor: {scale_factor:.3f}x")

        if valid_frames == 0: return (pose_data_copy, "Fehler: Keine validen Körperteile gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor berechnet: {final_scale:.3f}x")
        
        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b, c_b = getattr(pose_metas[best_idx], "kps_body", []), getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x: global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        log_messages.append(">> PoseData wurde erfolgreich mit dem Faktor multipliziert.")

        config_str = json.dumps({"anchor_scale": float(final_scale), "scale_x_factor": float(scale_x), "pivot_x": float(global_pivot_x), "pivot_y": float(global_pivot_y)})
        log_messages.append(f">> Skalierung ({final_scale:.3f}x) in nlf_render_config geschrieben.")

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseCalibrationManipulator2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "calibration_data": ("POSE_CALIBRATION",),
                "echte_groesse_override": ("FLOAT", {"default": 2.10, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip": "Erzwingt eine neue echte Größe in Metern."}),
                "enable_override": ("BOOLEAN", {"default": False, "tooltip": "Wenn False, werden die Originaldaten durchgeleitet."})
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING")
    RETURN_NAMES = ("modified_calibration", "log_output")
    FUNCTION = "manipulate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "Manipuliert die echte_groesse nachträglich und passt alle abhängigen Meter- und Scaler-Werte proportional an."

    def manipulate(self, calibration_data, echte_groesse_override, enable_override):
        import copy
        import json
        
        # Tiefkopie, um das Original nicht zu zerstören
        calib = copy.deepcopy(calibration_data)
        log_messages = ["=== CALIBRATION MANIPULATOR LOG ==="]

        if not enable_override:
            log_messages.append("Bypass aktiv: Originaldaten werden unverändert weitergeleitet.")
            return (calib, "\n".join(log_messages))

        alte_groesse = calib.get("echte_groesse", 1.0)
        
        if alte_groesse <= 0:
            log_messages.append("Fehler: Originale echte_groesse ist <= 0. Manipulation abgebrochen.")
            return (calib, "\n".join(log_messages))

        # Skalierungsfaktor berechnen
        faktor = echte_groesse_override / alte_groesse
        
        log_messages.append(f"Originale Größe: {alte_groesse:.3f}m")
        log_messages.append(f"Neue Ziel-Größe: {echte_groesse_override:.3f}m")
        log_messages.append(f"Manipulations-Faktor: {faktor:.4f}x")

        # 1. Hauptgröße überschreiben
        calib["echte_groesse"] = echte_groesse_override

        # 2. Metrische Knochen proportional anpassen
        bone_m = calib.get("bone_lengths_in_meters", {})
        if bone_m:
            log_messages.append("\n--- NEUE METRISCHE KNOCHEN (bone_lengths_in_meters) ---")
            for key, val in bone_m.items():
                neuer_wert = val * faktor
                calib["bone_lengths_in_meters"][key] = neuer_wert
                log_messages.append(f"{key.capitalize()}: {val:.3f}m -> {neuer_wert:.3f}m")
        
        # 3. Scaler-Knochen proportional anpassen (Wie gewünscht)
        bone_s = calib.get("bone_length_for_scaler", {})
        if bone_s:
            log_messages.append("\n--- NEUE SCALER KNOCHEN (bone_length_for_scaler) ---")
            for key, val in bone_s.items():
                neuer_wert = val * faktor
                calib["bone_length_for_scaler"][key] = neuer_wert
                log_messages.append(f"{key.capitalize()}: {val:.3f} -> {neuer_wert:.3f}")

        log_messages.append("\nProportionen (true_3d_bones) bleiben unangetastet (Das erledigt der Bones2-Scaler).")

        return (calib, "\n".join(log_messages))


class PoseGlobalPerspectiveScalerV50:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "PoseData (2D)"}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
                "video_intrinsics_json": ("STRING", {"forceInput": True}),
                "valid_depth_indices": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V50: Frame-Subsampler Integration für reduzierte Depth-Maps (Verhindert OOM-Errors)."

    def process(self, video_pose_data, calibration_data, video_depth_map, best_frame_source, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_nlf_data=None, video_intrinsics_json=None, valid_depth_indices=None):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V50 GLOBAL SCALER LOG (QUELLE: {best_frame_source} | VOLLE LOGS) ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        bone_m = calibration_data.get("bone_lengths_in_meters", {})
        is_inverted = calibration_data.get("is_depth_inverted", False)

        if not bone_m:
            return (pose_data_copy, "Fehler: Calibration Daten fehlen.", video_nlf_data, "{}")

        # --- DEPTH INDICES PARSEN ---
        valid_depth_frames = None
        if valid_depth_indices:
            try:
                valid_depth_frames = [int(x.strip()) for x in valid_depth_indices.split(",") if x.strip().isdigit()]
                log_messages.append(f">> DEPTH FPS OPTIMIERUNG AKTIV: {len(valid_depth_frames)} verknüpfte Depth-Maps erkannt.")
            except Exception as e:
                log_messages.append(f">> WARNUNG: Konnte valid_depth_indices nicht parsen. Nutze klassisches Mapping. Fehler: {e}")

        # --- BRENNWEITE FÜR DAS AKTUELLE VIDEO AUSLESEN ---
        fx_video = fx_calib
        if video_intrinsics_json:
            try:
                int_data = json.loads(video_intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx_video = float(matrix[0][0])
                        if abs(fx_video - fx_calib) > 1.0:
                            log_messages.append(f">> KAMERA-WECHSEL ERKANNT: Nutze neues Objektiv fx={fx_video:.2f} (Kalibrierung war fx={fx_calib:.2f})")
            except: pass

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]
        head_px_calib = calibration_data.get("bone_length_for_scaler", {}).get("head", 0.0)

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        all_frames_data = []
        frontal_indices = []
        
        if best_frame_source == "NLF (3D SMPL)" and video_nlf_data is None:
            best_frame_source = "PoseData (2D)"

        if best_frame_source == "NLF (3D SMPL)":
            log_messages.append("\n--- WINKEL-RADAR (Alle Frames werden protokolliert) ---")
            is_dict = isinstance(video_nlf_data, dict)
            raw_poses = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if is_dict else video_nlf_data

            for i in range(len(pose_metas)):
                if i >= len(raw_poses) or raw_poses[i] is None or len(raw_poses[i]) == 0:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                frame_data = raw_poses[i]
                is_tensor = isinstance(frame_data, torch.Tensor)
                pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))

                def is_val_nlf(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

                has_knees = is_val_nlf(4) or is_val_nlf(5)
                has_ankles = is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0
                
                # Kopf-Kompensation
                if include_head and length > 0: 
                    length += (head_px_calib / 2.0)

                is_frontal, frontal_pts = False, 0.0
                angle_h, angle_s, max_angle = 90.0, 90.0, 90.0
                
                if len(pts) >= 18:
                    dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                    dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                    
                    angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                    angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                    max_angle = max(angle_h, angle_s)

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                status = "FRONTAL (Akzeptiert)" if is_frontal else "SEITLICH (Abgelehnt)"
                log_messages.append(f"Frame {i}: Max-Winkel {max_angle:.1f}° (Hüfte: {angle_h:.1f}°, Schultern: {angle_s:.1f}°) -> {status}")

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        else:
            log_messages.append("\n--- WINKEL-RADAR (Alle Frames werden protokolliert) ---")
            pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) and video_nlf_data else None

            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                
                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
                top_y, bottom_y = min(valid_y) if valid_y else None, max(valid_y) if valid_y else None
                if not include_head and valid_y and is_val(kps, confs, 1): top_y = kps[1][1]
                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal, frontal_pts = False, 0.0
                angle = 90.0
                
                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pose_3d_frame = pose_input_3d[i]
                    if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                        person_3d = pose_3d_frame[0]
                        num_joints = len(person_3d)
                        idx_r, idx_l = 2, 5
                        if num_joints == 17: idx_r, idx_l = 11, 14
                        elif num_joints in [24, 45, 68]: idx_r, idx_l = 16, 17
                            
                        if num_joints > max(idx_r, idx_l):
                            dx, dz = float(person_3d[idx_r][0]) - float(person_3d[idx_l][0]), float(person_3d[idx_r][2]) - float(person_3d[idx_l][2])
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                status = "FRONTAL (Akzeptiert)" if is_frontal else "SEITLICH (Abgelehnt)"
                log_messages.append(f"Frame {i}: Winkel {angle:.1f}° -> {status}")

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        log_messages.append("\n--- PASS-FILTER (DER TÜRSTEHER) ---")
        if len(frontal_indices) > 0:
            log_messages.append(f">> PASS-FILTER AKTIV: {len(frontal_indices)} echte frontale Frames gefunden! Alle anderen fliegen raus.")
            candidates = frontal_indices
        else:
            log_messages.append(f">> PASS-FILTER INAKTIV: Kein Frame erfüllt die Toleranz. Nutze alle Frames.")
            candidates = list(range(len(pose_metas)))

        # WICHTIG: Kandidaten filtern, die KEINE Depth Map haben
        if valid_depth_frames is not None:
            filtered_candidates = [idx for idx in candidates if idx in valid_depth_frames]
            if not filtered_candidates:
                log_messages.append(">> WARNUNG: Kein Kandidaten-Frame hat eine Depth Map! Falle auf valide Depth-Frames zurück.")
                candidates = [idx for idx in valid_depth_frames if idx < len(all_frames_data)]
                if not candidates: candidates = [0]
            else:
                candidates = filtered_candidates

        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            data = all_frames_data[idx]
            bein_pts = max(1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0, 500.0 if not (data['has_feet'] or data['has_ankles']) and data['has_knees'] else 0.0)
            total_score = bein_pts + (500.0 if data['has_feet'] and data['is_frontal'] else 0.0) + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score: best_score, best_idx = total_score, idx

        log_messages.append(f"\n-> Gewinner Frame (Anchor): {best_idx} (Score: {best_score:.1f})")

        start_idx, end_idx = max(0, best_idx - anchor_window), min(len(pose_metas) - 1, best_idx + anchor_window)
        sum_scale_factors, valid_frames = 0.0, 0

        for i in range(start_idx, end_idx + 1):
            # Prüfen ob der Frame eine gültige Depth-Map hat
            if valid_depth_frames is not None:
                if i in valid_depth_frames:
                    depth_v_idx = valid_depth_frames.index(i)
                else:
                    log_messages.append(f"  Frame {i} übersprungen (Besitzt keine Subsample Depth-Map).")
                    continue
            else:
                depth_v_idx = min(i, depth_np.shape[0] - 1)

            kps, confs = getattr(pose_metas[i], "kps_body", []), getattr(pose_metas[i], "kps_body_p", None)
            frame_ist_px, frame_soll_m = 0.0, 0.0
            visible_parts = []

            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1); frame_soll_m += bone_m.get("head", 0)
                visible_parts.append("Kopf")
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0)
                visible_parts.append("Torso")
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9); frame_soll_m += bone_m.get("thigh", 0)
                visible_parts.append("Oberschenkel")
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10); frame_soll_m += bone_m.get("calf", 0)
                visible_parts.append("Wade")

            if frame_ist_px == 0 or frame_soll_m == 0: continue

            # HIER wird nun der korrekt gemappte depth_v_idx genutzt!
            frame_depth = get_skeleton_depth(kps, confs, depth_np, depth_v_idx, W, H)
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            # DIE MAGIE: HIER WIRD MIT DER NEUEN VIDEO-KAMERA GERECHNET!
            expected_px = (frame_soll_m * fx_video) / frame_depth
            scale_factor = expected_px / frame_ist_px
            sum_scale_factors += scale_factor
            valid_frames += 1

            # --- VOLLES LOGGING WIEDER DA! ---
            log_messages.append(f"\n  Frame {i} Analyse:")
            log_messages.append(f"    Depth-Array-Index: {depth_v_idx}")
            log_messages.append(f"    Sichtbar: {', '.join(visible_parts)}")
            log_messages.append(f"    Ist-Pixel (Knochensumme): {frame_ist_px:.1f} px")
            log_messages.append(f"    Ist-Meter (Knochensumme): {frame_soll_m:.3f} m")
            log_messages.append(f"    Skelett-Tiefe: {frame_depth:.3f} m -> Soll-Pixel: {expected_px:.1f} px")
            log_messages.append(f"    Lokaler Faktor: {scale_factor:.3f}x")

        if valid_frames == 0: return (pose_data_copy, "Fehler: Keine validen Körperteile in Frames mit Depth-Map gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor berechnet: {final_scale:.3f}x")
        
        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b, c_b = getattr(pose_metas[best_idx], "kps_body", []), getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x: global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        log_messages.append(">> PoseData wurde erfolgreich mit dem Faktor multipliziert.")

        config_str = json.dumps({"anchor_scale": float(final_scale), "scale_x_factor": float(scale_x), "pivot_x": float(global_pivot_x), "pivot_y": float(global_pivot_y)})
        log_messages.append(f">> Skalierung ({final_scale:.3f}x) in nlf_render_config geschrieben.")

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseGlobalPerspectiveScalerV51:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "video_depth_map": ("IMAGE",),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "PoseData (2D)"}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_nlf_data": ("NLFPRED",),
                "video_intrinsics_json": ("STRING", {"forceInput": True}),
                "valid_depth_indices": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V50: Frame-Subsampler Integration mit Nearest-Neighbor Fallback & intelligentem Radar."

    def process(self, video_pose_data, calibration_data, video_depth_map, best_frame_source, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_nlf_data=None, video_intrinsics_json=None, valid_depth_indices=None):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V50 GLOBAL SCALER LOG (QUELLE: {best_frame_source} | VOLLE LOGS) ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        bone_m = calibration_data.get("bone_lengths_in_meters", {})
        is_inverted = calibration_data.get("is_depth_inverted", False)

        if not bone_m:
            return (pose_data_copy, "Fehler: Calibration Daten fehlen.", video_nlf_data, "{}")

        # --- DEPTH INDICES PARSEN ---
        valid_depth_frames = None
        if valid_depth_indices:
            try:
                valid_depth_frames = [int(x.strip()) for x in valid_depth_indices.split(",") if x.strip().isdigit()]
                log_messages.append(f">> DEPTH FPS OPTIMIERUNG AKTIV: {len(valid_depth_frames)} verknüpfte Depth-Maps erkannt.")
            except Exception as e:
                log_messages.append(f">> WARNUNG: Konnte valid_depth_indices nicht parsen. Nutze klassisches Mapping. Fehler: {e}")

        # --- Hilfsfunktion: Nächsten validen Depth-Frame finden ---
        def get_nearest_depth_idx(target_idx, valid_list):
            if not valid_list: 
                return target_idx, target_idx
            nearest_val = min(valid_list, key=lambda x: abs(x - target_idx))
            return valid_list.index(nearest_val), nearest_val

        # --- BRENNWEITE FÜR DAS AKTUELLE VIDEO AUSLESEN ---
        fx_video = fx_calib
        if video_intrinsics_json:
            try:
                int_data = json.loads(video_intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx_video = float(matrix[0][0])
                        if abs(fx_video - fx_calib) > 1.0:
                            log_messages.append(f">> KAMERA-WECHSEL ERKANNT: Nutze neues Objektiv fx={fx_video:.2f} (Kalibrierung war fx={fx_calib:.2f})")
            except: pass

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]
        head_px_calib = calibration_data.get("bone_length_for_scaler", {}).get("head", 0.0)

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        all_frames_data = []
        frontal_indices = []
        
        if best_frame_source == "NLF (3D SMPL)" and video_nlf_data is None:
            best_frame_source = "PoseData (2D)"

        if best_frame_source == "NLF (3D SMPL)":
            log_messages.append("\n--- WINKEL-RADAR (Alle Frames werden protokolliert) ---")
            is_dict = isinstance(video_nlf_data, dict)
            raw_poses = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if is_dict else video_nlf_data

            for i in range(len(pose_metas)):
                if i >= len(raw_poses) or raw_poses[i] is None or len(raw_poses[i]) == 0:
                    log_messages.append(f"Frame {i}: Keine 3D-Daten vorhanden -> ÜBERSPRUNGEN (Abgelehnt)")
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                frame_data = raw_poses[i]
                is_tensor = isinstance(frame_data, torch.Tensor)
                pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))

                def is_val_nlf(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

                has_knees = is_val_nlf(4) or is_val_nlf(5)
                has_ankles = is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0
                
                # Kopf-Kompensation
                if include_head and length > 0: 
                    length += (head_px_calib / 2.0)

                is_frontal, frontal_pts = False, 0.0
                angle_h, angle_s, max_angle = 90.0, 90.0, 90.0
                
                if len(pts) >= 18:
                    dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                    dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                    
                    angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                    angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                    max_angle = max(angle_h, angle_s)

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                status = "FRONTAL (Akzeptiert)" if is_frontal else "SEITLICH (Abgelehnt)"
                log_messages.append(f"Frame {i}: Max-Winkel {max_angle:.1f}° (Hüfte: {angle_h:.1f}°, Schultern: {angle_s:.1f}°) -> {status}")

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        else:
            log_messages.append("\n--- WINKEL-RADAR (Alle Frames werden protokolliert) ---")
            pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if isinstance(video_nlf_data, dict) and video_nlf_data else None

            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                
                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
                if not valid_y:
                    log_messages.append(f"Frame {i}: Keine 2D-Pose-Daten vorhanden -> ÜBERSPRUNGEN (Abgelehnt)")
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                top_y, bottom_y = min(valid_y), max(valid_y)
                if not include_head and is_val(kps, confs, 1): top_y = kps[1][1]
                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal, frontal_pts = False, 0.0
                angle = 90.0
                
                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pose_3d_frame = pose_input_3d[i]
                    if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                        person_3d = pose_3d_frame[0]
                        num_joints = len(person_3d)
                        idx_r, idx_l = 2, 5
                        if num_joints == 17: idx_r, idx_l = 11, 14
                        elif num_joints in [24, 45, 68]: idx_r, idx_l = 16, 17
                            
                        if num_joints > max(idx_r, idx_l):
                            dx, dz = float(person_3d[idx_r][0]) - float(person_3d[idx_l][0]), float(person_3d[idx_r][2]) - float(person_3d[idx_l][2])
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                status = "FRONTAL (Akzeptiert)" if is_frontal else "SEITLICH (Abgelehnt)"
                log_messages.append(f"Frame {i}: Winkel {angle:.1f}° -> {status}")

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        log_messages.append("\n--- PASS-FILTER (DER TÜRSTEHER) ---")
        if len(frontal_indices) > 0:
            log_messages.append(f">> PASS-FILTER AKTIV: {len(frontal_indices)} echte frontale Frames gefunden! Alle anderen fliegen raus.")
            candidates = frontal_indices
        else:
            log_messages.append(f">> PASS-FILTER INAKTIV: Kein Frame erfüllt die Toleranz. Nutze alle Frames.")
            candidates = list(range(len(pose_metas)))

        # Da wir Depth-Maps jetzt "ausleihen" können, werfen wir hier niemanden mehr grundlos raus.
        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            data = all_frames_data[idx]
            bein_pts = max(1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0, 500.0 if not (data['has_feet'] or data['has_ankles']) and data['has_knees'] else 0.0)
            total_score = bein_pts + (500.0 if data['has_feet'] and data['is_frontal'] else 0.0) + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score: best_score, best_idx = total_score, idx

        log_messages.append(f"\n-> Gewinner Frame (Anchor): {best_idx} (Score: {best_score:.1f})")

        start_idx, end_idx = max(0, best_idx - anchor_window), min(len(pose_metas) - 1, best_idx + anchor_window)
        sum_scale_factors, valid_frames = 0.0, 0

        for i in range(start_idx, end_idx + 1):
            
            # --- Depth-Map Beschaffung (mit Nearest-Neighbor Ausleihe) ---
            if valid_depth_frames is not None:
                depth_v_idx, borrowed_from = get_nearest_depth_idx(i, valid_depth_frames)
                borrow_str = f" (Geliehen von Frame {borrowed_from})" if borrowed_from != i else ""
            else:
                depth_v_idx = min(i, depth_np.shape[0] - 1)
                borrow_str = ""

            kps, confs = getattr(pose_metas[i], "kps_body", []), getattr(pose_metas[i], "kps_body_p", None)
            frame_ist_px, frame_soll_m = 0.0, 0.0
            visible_parts = []

            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1); frame_soll_m += bone_m.get("head", 0)
                visible_parts.append("Kopf")
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0)
                visible_parts.append("Torso")
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9); frame_soll_m += bone_m.get("thigh", 0)
                visible_parts.append("Oberschenkel")
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10); frame_soll_m += bone_m.get("calf", 0)
                visible_parts.append("Wade")

            if frame_ist_px == 0 or frame_soll_m == 0: 
                log_messages.append(f"  Frame {i} übersprungen (Pose-Punkte fehlen).")
                continue

            frame_depth = get_skeleton_depth(kps, confs, depth_np, depth_v_idx, W, H)
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            expected_px = (frame_soll_m * fx_video) / frame_depth
            scale_factor = expected_px / frame_ist_px
            sum_scale_factors += scale_factor
            valid_frames += 1

            # --- VOLLES LOGGING WIEDER DA! ---
            log_messages.append(f"\n  Frame {i} Analyse:")
            log_messages.append(f"    Depth-Array-Index: {depth_v_idx}{borrow_str}")
            log_messages.append(f"    Sichtbar: {', '.join(visible_parts)}")
            log_messages.append(f"    Ist-Pixel (Knochensumme): {frame_ist_px:.1f} px")
            log_messages.append(f"    Ist-Meter (Knochensumme): {frame_soll_m:.3f} m")
            log_messages.append(f"    Skelett-Tiefe: {frame_depth:.3f} m -> Soll-Pixel: {expected_px:.1f} px")
            log_messages.append(f"    Lokaler Faktor: {scale_factor:.3f}x")

        if valid_frames == 0: return (pose_data_copy, "Fehler: Keine validen Körperteile in Frames gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor berechnet: {final_scale:.3f}x")
        
        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b, c_b = getattr(pose_metas[best_idx], "kps_body", []), getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x: global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        log_messages.append(">> PoseData wurde erfolgreich mit dem Faktor multipliziert.")

        config_str = json.dumps({"anchor_scale": float(final_scale), "scale_x_factor": float(scale_x), "pivot_x": float(global_pivot_x), "pivot_y": float(global_pivot_y)})
        log_messages.append(f">> Skalierung ({final_scale:.3f}x) in nlf_render_config geschrieben.")

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseCalibrationV30:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
                "normalize_bones_to_100": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "depth_nah": ("IMAGE",),
                "depth_fern": ("IMAGE",),
                "nlf_data_nah": ("NLFPRED",),
                "nlf_data_fern": ("NLFPRED",),
                "intrinsics_json": ("STRING", {"forceInput": True}),
                "config_data": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V30: Universal Data Hub - Speichert Depth-Map & native NLF-3D-Tiefen für flexiblen Scaler."

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, pose_fern_scaled, pose_fern_unscaled, norm_method, min_confidence, invert_depth, use_pinhole_math=True, normalize_bones_to_100=True, depth_nah=None, depth_fern=None, nlf_data_nah=None, nlf_data_fern=None, intrinsics_json=None, config_data="{}"):
        import json
        import math
        import numpy as np

        log_messages = ["=== V30 CALIBRATION LOG (UNIVERSAL DATA HUB) ==="]

        try:
            config = json.loads(config_data)
        except:
            config = {}

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
            return c >= min_confidence

        # --- 1. DEPTH MAP LOGIK (WIE V29) ---
        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [
                (0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),
                (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)
            ]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px = int(x1 + t * (x2 - x1))
                            py = int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        def get_depth_for_pose(pose_u, depth_img):
            if depth_img is None: return 0.0
            meta_u = pose_u.get("pose_metas", [])[0] if pose_u.get("pose_metas") else None
            if not meta_u: return 0.5
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            depth_np = depth_img.cpu().numpy() if hasattr(depth_img, 'cpu') else depth_img
            H, W = depth_np.shape[1], depth_np.shape[2]
            return get_skeleton_depth(kps_u, confs_u, depth_np, 0, W, H)

        # --- 1.b NLF 3D TIEFEN-LOGIK (NEU!) ---
        def get_nlf_torso_z(nlf_data):
            if nlf_data is None: return 0.0
            is_dict = isinstance(nlf_data, dict)
            raw_poses = nlf_data.get('joints3d_nonparam', [nlf_data])[0] if is_dict else nlf_data
            if not raw_poses or len(raw_poses) == 0 or raw_poses[0] is None: return 0.0
            
            frame_data = raw_poses[0]
            is_tensor = hasattr(frame_data, 'dim') # Check if it's a torch tensor
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
            
            if len(pts) < 17: return 0.0
            
            # Torso/Brustkorb Fokus (ignoriere wackelige Arme/Beine). Indizes basierend auf SMPL Format.
            # Normalerweise: 0=Pelvis, 3=Spine1, 6=Spine2, 9=Spine3/Chest, 12=Neck. 
            torso_indices = [0, 3, 6, 9, 12]
            valid_z = []
            for idx in torso_indices:
                if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5:
                    # Index 2 ist die Z-Achse in den meisten 3D-Skeletten
                    valid_z.append(float(pts[idx][2]))
            
            if valid_z: return float(np.mean(valid_z))
            return 0.0

        # --- 2. KNOCHEN EXTRAHIEREN ---
        def extract_2d_bones(pose_data):
            try:
                meta = pose_data.get("pose_metas", [])[0]
                kps = getattr(meta, "kps_body", None)
                if kps is None or len(kps) < 14: return None, None
                def dist_2d(idx1, idx2):
                    if idx1 >= len(kps) or idx2 >= len(kps): return 0.0
                    p1, p2 = kps[idx1], kps[idx2]
                    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2: return 0.0
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                mid_hip_x = (kps[8][0] + kps[11][0]) / 2.0
                mid_hip_y = (kps[8][1] + kps[11][1]) / 2.0
                
                raw_bones = {
                    "head": dist_2d(0, 1),
                    "torso": math.sqrt((kps[1][0] - mid_hip_x)**2 + (kps[1][1] - mid_hip_y)**2),
                    "shoulder_width": dist_2d(2, 5), "hip_width": dist_2d(8, 11),
                    "r_arm": dist_2d(2, 3), "r_forearm": dist_2d(3, 4),
                    "l_arm": dist_2d(5, 6), "l_forearm": dist_2d(6, 7),
                    "r_thigh": dist_2d(8, 9), "r_calf": dist_2d(9, 10),
                    "l_thigh": dist_2d(11, 12), "l_calf": dist_2d(12, 13)
                }

                sym_bones = {
                    "head": raw_bones["head"], "torso": raw_bones["torso"],
                    "shoulder_width": raw_bones["shoulder_width"], "hip_width": raw_bones["hip_width"],
                    "r_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "l_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "r_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "l_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "r_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "l_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "r_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0,
                    "l_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0
                }

                norm_bones = {}
                if normalize_bones_to_100:
                    for k, v in sym_bones.items():
                        norm_bones[k] = (v / sym_bones["torso"]) * 100.0 if sym_bones["torso"] > 0 else 0
                else: norm_bones = sym_bones.copy()

                return sym_bones, norm_bones
            except Exception as e: return None, None

        log_messages.append(f"Norm-Methode: {norm_method}")

        unscaled_bones_nah, _ = extract_2d_bones(pose_nah_scaled)
        unscaled_bones_fern, true_3d_bones = extract_2d_bones(pose_fern_scaled)

        # --- 3. NORM BERECHNUNG ---
        def calc_norm_from_bones(bones, method):
            if not bones: return 100.0
            if method == "Torso (Neck-Hip)": return bones["torso"]
            return bones["head"] + bones["torso"] + bones["r_thigh"] + bones["r_calf"]

        norm_nah_raw = calc_norm_from_bones(unscaled_bones_nah, norm_method)
        norm_fern = calc_norm_from_bones(unscaled_bones_fern, norm_method)
        norm_nah = norm_nah_raw

        # TIEFEN-AUSLESUNG: Depth-Map und NLF parallel
        depth_map_c = get_depth_for_pose(pose_nah_unscaled, depth_nah)
        depth_map_f = get_depth_for_pose(pose_fern_unscaled, depth_fern)
        
        nlf_z_c = get_nlf_torso_z(nlf_data_nah)
        nlf_z_f = get_nlf_torso_z(nlf_data_fern)

        if invert_depth:
            depth_map_c = 1.0 / max(depth_map_c, 0.0001) if depth_map_c > 0 else 0.0
            depth_map_f = 1.0 / max(depth_map_f, 0.0001) if depth_map_f > 0 else 0.0
            # NLF Z-Werte invertieren wir normalerweise nicht, da sie native 3D-Metriken sind, 
            # aber falls gewollt, würden wir das hier tun. Ich lasse NLF unangetastet, da es absolute Koordinaten sind.

        # --- 4. EXTRAPOLATION ---
        if unscaled_bones_nah and unscaled_bones_fern:
            torso_nah = unscaled_bones_nah["torso"]
            torso_fern = unscaled_bones_fern["torso"]
            if torso_fern > 0:
                torso_faktor = torso_nah / torso_fern
                extrapolated_nah = norm_fern * torso_faktor
                norm_nah = extrapolated_nah
        else:
            if not unscaled_bones_fern: unscaled_bones_fern, true_3d_bones = unscaled_bones_nah, _

        # Intrinsics abfragen
        fx = 500.0
        if intrinsics_json:
            try:
                int_data = json.loads(intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx = float(matrix[0][0])
            except: pass

        # --- 5. BERECHNUNG ECHTE GRÖSSE (BEIDE METHODEN) ---
        echte_groesse_depthmap = 0.0
        delta_z_depth = depth_map_f - depth_map_c
        if depth_map_f > 0 and depth_map_c > 0:
            log_messages.append("\n--- DEPTH MAP BERECHNUNG ---")
            log_messages.append(f"Tiefe Nah: {depth_map_c:.4f} | Tiefe Fern: {depth_map_f:.4f} | Delta Z: {delta_z_depth:.4f}")
            if use_pinhole_math and delta_z_depth > 0 and (norm_nah - norm_fern) > 0:
                echte_groesse_depthmap = (delta_z_depth * norm_nah * norm_fern) / (fx * (norm_nah - norm_fern))
            else:
                echte_groesse_depthmap = (norm_nah * depth_map_c) / fx
            log_messages.append(f"Errechnete Körpergröße (Depth Map): {echte_groesse_depthmap:.3f}m")

        echte_groesse_nlf = 0.0
        delta_z_nlf = abs(nlf_z_f - nlf_z_c) # Absolut, falls die Kameraachse andersrum ist
        if nlf_z_f != 0.0 and nlf_z_c != 0.0:
            log_messages.append("\n--- NLF 3D DATA BERECHNUNG ---")
            log_messages.append(f"Torso Z Nah: {nlf_z_c:.4f} | Torso Z Fern: {nlf_z_f:.4f} | Delta Z: {delta_z_nlf:.4f}")
            if use_pinhole_math and delta_z_nlf > 0 and (norm_nah - norm_fern) > 0:
                echte_groesse_nlf = (delta_z_nlf * norm_nah * norm_fern) / (fx * (norm_nah - norm_fern))
            else:
                echte_groesse_nlf = (norm_nah * abs(nlf_z_c)) / fx
            log_messages.append(f"Errechnete Körpergröße (NLF Data): {echte_groesse_nlf:.3f}m")

        bone_length_for_scaler = {}
        bone_lengths_in_meters_depthmap = {}
        bone_lengths_in_meters_nlf = {}

        if unscaled_bones_fern:
            bone_length_for_scaler = {
                "head": unscaled_bones_fern["head"],
                "torso": unscaled_bones_fern["torso"],
                "thigh": unscaled_bones_fern["r_thigh"],
                "calf": unscaled_bones_fern["r_calf"]
            }
            total_px = sum(bone_length_for_scaler.values())
            if total_px > 0:
                for k, px_val in bone_length_for_scaler.items():
                    if echte_groesse_depthmap > 0:
                        bone_lengths_in_meters_depthmap[k] = (px_val / total_px) * echte_groesse_depthmap
                    if echte_groesse_nlf > 0:
                        bone_lengths_in_meters_nlf[k] = (px_val / total_px) * echte_groesse_nlf

        # ALLES wird im Dictionary gespeichert, der Scaler sucht sich dann aus, was er braucht!
        calib_data = {
            "is_depth_inverted": invert_depth, "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math, "focal_length_fx": fx,
            
            # Depth Map Spezifisch
            "depth_c": depth_map_c, "depth_f": depth_map_f,
            "echte_groesse_depthmap": echte_groesse_depthmap,
            "bone_lengths_in_meters_depthmap": bone_lengths_in_meters_depthmap,
            
            # NLF Spezifisch
            "nlf_z_c": nlf_z_c, "nlf_z_f": nlf_z_f,
            "echte_groesse_nlf": echte_groesse_nlf,
            "bone_lengths_in_meters_nlf": bone_lengths_in_meters_nlf,

            # Allgemein
            "true_3d_bones": true_3d_bones or {},
            "bone_length_for_scaler": bone_length_for_scaler or {},
            "total_3d_height": sum(bone_length_for_scaler.values()) if bone_length_for_scaler else 0.0,
            "config": config
        }

        log_messages.append("\n>> Alle Kalibrierungsdaten wurden erfolgreich in den Hub geladen.")

        return (calib_data, "\n".join(log_messages))


class PoseGlobalPerspectiveScalerV53:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "depth_source": (["Depth Map (2D Pixel)", "NLF 3D Model (Z-Axis)"], {"default": "Depth Map (2D Pixel)"}),
                "nlf_smoothing_window": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1, "tooltip": "Glättet die Z-Werte über X Frames, um das NLF-Zittern zu stoppen."}),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "PoseData (2D)"}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_depth_map": ("IMAGE",),
                "video_nlf_data": ("NLFPRED",),
                "intrinsics_source": (["Use DA3/Depth JSON", "Use NLF Default (5000)"], {"default": "Use DA3/Depth JSON"}),
                "video_intrinsics_json": ("STRING", {"forceInput": True}),
                "valid_depth_indices": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V53: NLF 2D-Overlay auf Depth Map hinzugefügt. Flexibler Toggle zwischen Z-Achse und Depth Map."

    def process(self, video_pose_data, calibration_data, depth_source, nlf_smoothing_window, best_frame_source, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_depth_map=None, video_nlf_data=None, intrinsics_source="Use DA3/Depth JSON", video_intrinsics_json=None, valid_depth_indices=None):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V53 GLOBAL SCALER LOG (SOURCE: {depth_source}) ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        is_inverted = calibration_data.get("is_depth_inverted", False)

        # Wähle die richtigen Calibration-Metriken basierend auf dem Dropdown
        if depth_source == "NLF 3D Model (Z-Axis)":
            if not calibration_data.get("bone_lengths_in_meters_nlf"):
                return (pose_data_copy, "Fehler: Calibration Hub hat keine NLF-Daten. Bitte NLF in die Calibration Node stecken!", video_nlf_data, "{}")
            bone_m = calibration_data.get("bone_lengths_in_meters_nlf", {})
            log_messages.append(">> NLF-MODUS AKTIV: Lade dedizierte NLF-Knochenmaße aus Calibration Hub.")
        else:
            bone_m = calibration_data.get("bone_lengths_in_meters_depthmap", {})
            if not bone_m: # Fallback falls alter Workflow
                bone_m = calibration_data.get("bone_lengths_in_meters", {})
            log_messages.append(">> DEPTH-MAP-MODUS AKTIV: Lade metrische Knochenmaße aus Calibration Hub.")

        # --- BRENNWEITE ---
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        fx_video = fx_calib
        if intrinsics_source == "Use NLF Default (5000)":
            fx_video = 5000.0
            log_messages.append(f">> Nutze NLF Default Brennweite: fx={fx_video}")
        elif video_intrinsics_json:
            try:
                int_data = json.loads(video_intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx_video = float(matrix[0][0])
                        log_messages.append(f">> Kamera-Intrinsics geladen: fx={fx_video:.2f}")
            except: pass

        # --- DEPTH INDICES PARSEN (nur für Depth Map wichtig) ---
        valid_depth_frames = None
        if valid_depth_indices:
            try:
                valid_depth_frames = [int(x.strip()) for x in valid_depth_indices.split(",") if x.strip().isdigit()]
            except: pass

        def get_nearest_depth_idx(target_idx, valid_list):
            if not valid_list: return target_idx, target_idx
            nearest_val = min(valid_list, key=lambda x: abs(x - target_idx))
            return valid_list.index(nearest_val), nearest_val

        depth_np = None
        if video_depth_map is not None:
            depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        
        H, W = (depth_np.shape[1], depth_np.shape[2]) if depth_np is not None else (1024, 1024)
        head_px_calib = calibration_data.get("bone_length_for_scaler", {}).get("head", 0.0)

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        # Alte 2D Pose Maske
        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            if depth_img is None: return 0.5
            skeleton_connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        # NEU V53: NLF 2D Overlay Maske für Depth Map
        def get_nlf_2d_depth(pts, depth_img, v_idx, W, H):
            if depth_img is None: return 0.5
            depth_vals = []
            for idx in range(len(pts)):
                if np.linalg.norm(pts[idx]) > 1e-5: # Nur valide NLF Punkte
                    px, py = int(pts[idx][0]), int(pts[idx][1])
                    if 0 <= px < W and 0 <= py < H:
                        val = depth_img[v_idx, py, px]
                        depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        # RADAR LOGIK
        all_frames_data = []
        frontal_indices = []
        
        if best_frame_source == "NLF (3D SMPL)" and video_nlf_data is None:
            best_frame_source = "PoseData (2D)"

        is_dict = isinstance(video_nlf_data, dict)
        raw_poses = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if is_dict and video_nlf_data else video_nlf_data

        if best_frame_source == "NLF (3D SMPL)":
            for i in range(len(pose_metas)):
                if raw_poses is None or i >= len(raw_poses) or raw_poses[i] is None or len(raw_poses[i]) == 0:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                frame_data = raw_poses[i]
                is_tensor = hasattr(frame_data, 'dim')
                pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))

                def is_val_nlf(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

                has_knees = is_val_nlf(4) or is_val_nlf(5)
                has_ankles = is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0
                if include_head and length > 0: length += (head_px_calib / 2.0)

                is_frontal, frontal_pts, max_angle = False, 0.0, 90.0
                if len(pts) >= 18:
                    dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                    dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                    angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                    angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                    max_angle = max(angle_h, angle_s)

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        else: # 2D RADAR
            pose_input_3d = raw_poses
            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                
                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
                if not valid_y:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                top_y, bottom_y = min(valid_y), max(valid_y)
                if not include_head and is_val(kps, confs, 1): top_y = kps[1][1]
                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal, frontal_pts = False, 0.0
                
                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pose_3d_frame = pose_input_3d[i]
                    if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                        person_3d = pose_3d_frame[0]
                        num_joints = len(person_3d)
                        idx_r, idx_l = 2, 5
                        if num_joints == 17: idx_r, idx_l = 11, 14
                        elif num_joints in [24, 45, 68]: idx_r, idx_l = 16, 17
                            
                        if num_joints > max(idx_r, idx_l):
                            dx, dz = float(person_3d[idx_r][0]) - float(person_3d[idx_l][0]), float(person_3d[idx_r][2]) - float(person_3d[idx_l][2])
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        candidates = frontal_indices if len(frontal_indices) > 0 else list(range(len(pose_metas)))
        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            data = all_frames_data[idx]
            bein_pts = max(1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0, 500.0 if not (data['has_feet'] or data['has_ankles']) and data['has_knees'] else 0.0)
            total_score = bein_pts + (500.0 if data['has_feet'] and data['is_frontal'] else 0.0) + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score: best_score, best_idx = total_score, idx

        log_messages.append(f"\n-> Gewinner Frame (Anchor): {best_idx}")

        start_idx, end_idx = max(0, best_idx - anchor_window), min(len(pose_metas) - 1, best_idx + anchor_window)
        
        # --- NLF TEMPORAL SMOOTHING VORBEREITUNG ---
        nlf_raw_z_values = {}
        if depth_source == "NLF 3D Model (Z-Axis)" and raw_poses is not None:
            log_messages.append(f"\n--- BERECHNE NLF Z-TIEFE & SMOOTHING (Window: {nlf_smoothing_window}) ---")
            smooth_start = max(0, start_idx - nlf_smoothing_window)
            smooth_end = min(len(pose_metas) - 1, end_idx + nlf_smoothing_window)
            
            for i in range(smooth_start, smooth_end + 1):
                if i < len(raw_poses) and raw_poses[i] is not None and len(raw_poses[i]) > 0:
                    frame_data = raw_poses[i]
                    is_tensor = hasattr(frame_data, 'dim')
                    pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
                    
                    torso_indices = [0, 3, 6, 9, 12]
                    valid_z = [float(pts[idx][2]) for idx in torso_indices if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
                    if valid_z:
                        nlf_raw_z_values[i] = float(np.mean(valid_z))
                    else:
                        nlf_raw_z_values[i] = 0.5
                else:
                    nlf_raw_z_values[i] = 0.5

        sum_scale_factors, valid_frames = 0.0, 0

        for i in range(start_idx, end_idx + 1):
            kps, confs = getattr(pose_metas[i], "kps_body", []), getattr(pose_metas[i], "kps_body_p", None)
            frame_ist_px, frame_soll_m = 0.0, 0.0
            visible_parts = []

            # Knochenlängen für den Ist-Soll-Vergleich (Die 2D-Strecken)
            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1); frame_soll_m += bone_m.get("head", 0)
                visible_parts.append("Kopf")
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0)
                visible_parts.append("Torso")
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9); frame_soll_m += bone_m.get("thigh", 0)
                visible_parts.append("Oberschenkel")
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10); frame_soll_m += bone_m.get("calf", 0)
                visible_parts.append("Wade")

            if frame_ist_px == 0 or frame_soll_m == 0: continue

            # --- TIEFEN-BERECHNUNG WEICHE (V53 NEU) ---
            frame_depth = 0.5
            if depth_source == "NLF 3D Model (Z-Axis)" and i in nlf_raw_z_values:
                # 1. Option: Native NLF Z-Tiefe mit Smoothing
                z_window = []
                for w in range(max(0, i - nlf_smoothing_window), min(len(pose_metas), i + nlf_smoothing_window + 1)):
                    if w in nlf_raw_z_values: z_window.append(nlf_raw_z_values[w])
                
                raw_z = nlf_raw_z_values[i]
                frame_depth = float(np.mean(z_window)) if z_window else raw_z
                frame_depth = abs(frame_depth)
                if frame_depth < 0.01: frame_depth = 0.01
                
                log_messages.append(f"\n  Frame {i} NLF-Z-Tiefe: Raw Z={raw_z:.3f}m -> Smoothed Z={frame_depth:.3f}m")

            else:
                # Depth Map Logik
                if valid_depth_frames is not None:
                    depth_v_idx, borrowed_from = get_nearest_depth_idx(i, valid_depth_frames)
                else:
                    depth_v_idx = min(i, (depth_np.shape[0] - 1) if depth_np is not None else 0)

                # 2. Option: NLF-Punkte über Depth-Map legen (Dein Wunsch!)
                if best_frame_source == "NLF (3D SMPL)" and raw_poses is not None and i < len(raw_poses) and raw_poses[i] is not None:
                    frame_data = raw_poses[i]
                    is_tensor = hasattr(frame_data, 'dim')
                    pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
                    
                    frame_depth = get_nlf_2d_depth(pts, depth_np, depth_v_idx, W, H)
                    log_messages.append(f"\n  Frame {i} Depth Map via NLF 2D Overlay Maske abgelesen.")
                
                # 3. Option: Alte 2D Pose über Depth Map legen
                else:
                    frame_depth = get_skeleton_depth(kps, confs, depth_np, depth_v_idx, W, H)
                    log_messages.append(f"\n  Frame {i} Depth Map via PoseData 2D Maske abgelesen.")
                
                if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            expected_px = (frame_soll_m * fx_video) / frame_depth
            scale_factor = expected_px / frame_ist_px
            sum_scale_factors += scale_factor
            valid_frames += 1

            log_messages.append(f"    Ist-Pixel: {frame_ist_px:.1f} px | Ist-Meter: {frame_soll_m:.3f} m")
            log_messages.append(f"    Finale Tiefe: {frame_depth:.3f} m -> Soll-Pixel: {expected_px:.1f} px")
            log_messages.append(f"    Lokaler Faktor: {scale_factor:.3f}x")

        if valid_frames == 0: return (pose_data_copy, "Fehler: Keine validen Körperteile gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor ({valid_frames} Frames): {final_scale:.3f}x")
        
        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b, c_b = getattr(pose_metas[best_idx], "kps_body", []), getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x: global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        config_str = json.dumps({"anchor_scale": float(final_scale), "scale_x_factor": float(scale_x), "pivot_x": float(global_pivot_x), "pivot_y": float(global_pivot_y)})
        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseCalibrationV31:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
                "normalize_bones_to_100": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "depth_nah": ("IMAGE",),
                "depth_fern": ("IMAGE",),
                "nlf_data_nah": ("NLFPRED",),
                "nlf_data_fern": ("NLFPRED",),
                "intrinsics_json": ("STRING", {"forceInput": True}),
                "config_data": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V31: Universal Data Hub - Berechnet automatisch BEIDE Welten (DepthMap + NLF) parallel."

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, pose_fern_scaled, pose_fern_unscaled, norm_method, min_confidence, invert_depth, use_pinhole_math=True, normalize_bones_to_100=True, depth_nah=None, depth_fern=None, nlf_data_nah=None, nlf_data_fern=None, intrinsics_json=None, config_data="{}"):
        import json
        import math
        import numpy as np

        log_messages = ["=== V31 CALIBRATION LOG (UNIVERSAL DUAL HUB) ==="]

        try:
            config = json.loads(config_data)
        except:
            config = {}

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
            return c >= min_confidence

        # --- 1. DEPTH MAP LOGIK ---
        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [
                (0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),
                (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)
            ]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px = int(x1 + t * (x2 - x1))
                            py = int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        def get_depth_for_pose(pose_u, depth_img):
            if depth_img is None: return 0.0
            meta_u = pose_u.get("pose_metas", [])[0] if pose_u.get("pose_metas") else None
            if not meta_u: return 0.5
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            depth_np = depth_img.cpu().numpy() if hasattr(depth_img, 'cpu') else depth_img
            H, W = depth_np.shape[1], depth_np.shape[2]
            return get_skeleton_depth(kps_u, confs_u, depth_np, 0, W, H)

        # --- 1.b NLF 3D TIEFEN-LOGIK ---
        def get_nlf_torso_z(nlf_data):
            if nlf_data is None: return 0.0
            is_dict = isinstance(nlf_data, dict)
            raw_poses = nlf_data.get('joints3d_nonparam', [nlf_data])[0] if is_dict else nlf_data
            if not raw_poses or len(raw_poses) == 0 or raw_poses[0] is None: return 0.0
            
            frame_data = raw_poses[0]
            is_tensor = hasattr(frame_data, 'dim')
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
            
            if len(pts) < 17: return 0.0
            
            torso_indices = [0, 3, 6, 9, 12]
            valid_z = []
            for idx in torso_indices:
                if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5:
                    valid_z.append(float(pts[idx][2]))
            
            if valid_z: return float(np.mean(valid_z))
            return 0.0

        # --- 2. KNOCHEN EXTRAHIEREN ---
        def extract_2d_bones(pose_data):
            try:
                meta = pose_data.get("pose_metas", [])[0]
                kps = getattr(meta, "kps_body", None)
                if kps is None or len(kps) < 14: return None, None
                def dist_2d(idx1, idx2):
                    if idx1 >= len(kps) or idx2 >= len(kps): return 0.0
                    p1, p2 = kps[idx1], kps[idx2]
                    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2: return 0.0
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                mid_hip_x = (kps[8][0] + kps[11][0]) / 2.0
                mid_hip_y = (kps[8][1] + kps[11][1]) / 2.0
                
                raw_bones = {
                    "head": dist_2d(0, 1),
                    "torso": math.sqrt((kps[1][0] - mid_hip_x)**2 + (kps[1][1] - mid_hip_y)**2),
                    "shoulder_width": dist_2d(2, 5), "hip_width": dist_2d(8, 11),
                    "r_arm": dist_2d(2, 3), "r_forearm": dist_2d(3, 4),
                    "l_arm": dist_2d(5, 6), "l_forearm": dist_2d(6, 7),
                    "r_thigh": dist_2d(8, 9), "r_calf": dist_2d(9, 10),
                    "l_thigh": dist_2d(11, 12), "l_calf": dist_2d(12, 13)
                }

                sym_bones = {
                    "head": raw_bones["head"], "torso": raw_bones["torso"],
                    "shoulder_width": raw_bones["shoulder_width"], "hip_width": raw_bones["hip_width"],
                    "r_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "l_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "r_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "l_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "r_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "l_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "r_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0,
                    "l_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0
                }

                norm_bones = {}
                if normalize_bones_to_100:
                    for k, v in sym_bones.items():
                        norm_bones[k] = (v / sym_bones["torso"]) * 100.0 if sym_bones["torso"] > 0 else 0
                else: norm_bones = sym_bones.copy()

                return sym_bones, norm_bones
            except Exception as e: return None, None

        log_messages.append(f"Norm-Methode: {norm_method}")

        unscaled_bones_nah, _ = extract_2d_bones(pose_nah_scaled)
        unscaled_bones_fern, true_3d_bones = extract_2d_bones(pose_fern_scaled)

        # --- 3. NORM BERECHNUNG ---
        def calc_norm_from_bones(bones, method):
            if not bones: return 100.0
            if method == "Torso (Neck-Hip)": return bones["torso"]
            return bones["head"] + bones["torso"] + bones["r_thigh"] + bones["r_calf"]

        norm_nah_raw = calc_norm_from_bones(unscaled_bones_nah, norm_method)
        norm_fern = calc_norm_from_bones(unscaled_bones_fern, norm_method)
        norm_nah = norm_nah_raw

        # TIEFEN-AUSLESUNG: Depth-Map und NLF parallel
        depth_map_c = get_depth_for_pose(pose_nah_unscaled, depth_nah)
        depth_map_f = get_depth_for_pose(pose_fern_unscaled, depth_fern)
        
        nlf_z_c = get_nlf_torso_z(nlf_data_nah)
        nlf_z_f = get_nlf_torso_z(nlf_data_fern)

        if invert_depth:
            depth_map_c = 1.0 / max(depth_map_c, 0.0001) if depth_map_c > 0 else 0.0
            depth_map_f = 1.0 / max(depth_map_f, 0.0001) if depth_map_f > 0 else 0.0

        # --- 4. EXTRAPOLATION ---
        if unscaled_bones_nah and unscaled_bones_fern:
            torso_nah = unscaled_bones_nah["torso"]
            torso_fern = unscaled_bones_fern["torso"]
            if torso_fern > 0:
                torso_faktor = torso_nah / torso_fern
                extrapolated_nah = norm_fern * torso_faktor
                norm_nah = extrapolated_nah
        else:
            if not unscaled_bones_fern: unscaled_bones_fern, true_3d_bones = unscaled_bones_nah, _

        fx = 500.0
        if intrinsics_json:
            try:
                int_data = json.loads(intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx = float(matrix[0][0])
            except: pass

        # --- 5. BERECHNUNG ECHTE GRÖSSE (BEIDE METHODEN) ---
        echte_groesse_depthmap = 0.0
        delta_z_depth = depth_map_f - depth_map_c
        if depth_map_f > 0 and depth_map_c > 0:
            log_messages.append("\n--- DEPTH MAP BERECHNUNG ---")
            log_messages.append(f"Tiefe Nah: {depth_map_c:.4f} | Tiefe Fern: {depth_map_f:.4f} | Delta Z: {delta_z_depth:.4f}")
            if use_pinhole_math and delta_z_depth > 0 and (norm_nah - norm_fern) > 0:
                echte_groesse_depthmap = (delta_z_depth * norm_nah * norm_fern) / (fx * (norm_nah - norm_fern))
            else:
                echte_groesse_depthmap = (norm_nah * depth_map_c) / fx
            log_messages.append(f"Errechnete Körpergröße (Depth Map): {echte_groesse_depthmap:.3f}m")

        echte_groesse_nlf = 0.0
        delta_z_nlf = abs(nlf_z_f - nlf_z_c)
        if nlf_z_f != 0.0 and nlf_z_c != 0.0:
            log_messages.append("\n--- NLF 3D DATA BERECHNUNG ---")
            log_messages.append(f"Torso Z Nah: {nlf_z_c:.4f} | Torso Z Fern: {nlf_z_f:.4f} | Delta Z: {delta_z_nlf:.4f}")
            if use_pinhole_math and delta_z_nlf > 0 and (norm_nah - norm_fern) > 0:
                echte_groesse_nlf = (delta_z_nlf * norm_nah * norm_fern) / (fx * (norm_nah - norm_fern))
            else:
                echte_groesse_nlf = (norm_nah * abs(nlf_z_c)) / fx
            log_messages.append(f"Errechnete Körpergröße (NLF Data): {echte_groesse_nlf:.3f}m")

        bone_length_for_scaler = {}
        bone_lengths_in_meters_depthmap = {}
        bone_lengths_in_meters_nlf = {}

        if unscaled_bones_fern:
            bone_length_for_scaler = {
                "head": unscaled_bones_fern["head"],
                "torso": unscaled_bones_fern["torso"],
                "thigh": unscaled_bones_fern["r_thigh"],
                "calf": unscaled_bones_fern["r_calf"]
            }
            total_px = sum(bone_length_for_scaler.values())
            if total_px > 0:
                for k, px_val in bone_length_for_scaler.items():
                    if echte_groesse_depthmap > 0:
                        bone_lengths_in_meters_depthmap[k] = (px_val / total_px) * echte_groesse_depthmap
                    if echte_groesse_nlf > 0:
                        bone_lengths_in_meters_nlf[k] = (px_val / total_px) * echte_groesse_nlf

        calib_data = {
            "is_depth_inverted": invert_depth, "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math, "focal_length_fx": fx,
            
            "depth_c": depth_map_c, "depth_f": depth_map_f,
            "echte_groesse_depthmap": echte_groesse_depthmap,
            "bone_lengths_in_meters_depthmap": bone_lengths_in_meters_depthmap,
            
            "nlf_z_c": nlf_z_c, "nlf_z_f": nlf_z_f,
            "echte_groesse_nlf": echte_groesse_nlf,
            "bone_lengths_in_meters_nlf": bone_lengths_in_meters_nlf,

            "true_3d_bones": true_3d_bones or {},
            "bone_length_for_scaler": bone_length_for_scaler or {},
            "total_3d_height": sum(bone_length_for_scaler.values()) if bone_length_for_scaler else 0.0,
            "config": config
        }

        log_messages.append("\n>> Alle Kalibrierungsdaten wurden erfolgreich in den Hub geladen.")

        return (calib_data, "\n".join(log_messages))


class PoseGlobalPerspectiveScalerV54:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "scaling_mode": (["1. PoseData 2D + Depth Map", "2. NLF 3D Z-Axis (Smoothed)", "3. NLF 2D Overlay + Depth Map"], {"default": "1. PoseData 2D + Depth Map"}),
                "nlf_smoothing_window": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1, "tooltip": "Glättet die Z-Werte über X Frames, um das NLF-Zittern zu stoppen."}),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "PoseData (2D)"}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_depth_map": ("IMAGE",),
                "video_nlf_data": ("NLFPRED",),
                "intrinsics_source": (["Use DA3/Depth JSON", "Use NLF Default (5000)"], {"default": "Use DA3/Depth JSON"}),
                "video_intrinsics_json": ("STRING", {"forceInput": True}),
                "valid_depth_indices": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V54: Schreibt Körpergröße in JSON. Wählt smarte Metriken aus dem Calibration Hub."

    def process(self, video_pose_data, calibration_data, scaling_mode, nlf_smoothing_window, best_frame_source, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_depth_map=None, video_nlf_data=None, intrinsics_source="Use DA3/Depth JSON", video_intrinsics_json=None, valid_depth_indices=None):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V54 GLOBAL SCALER LOG (MODE: {scaling_mode}) ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        is_inverted = calibration_data.get("is_depth_inverted", False)

        # --- LOGIK: WELCHE KALIBRIERUNG NUTZEN WIR? ---
        used_echte_groesse = 0.0
        if scaling_mode.startswith("1"):
            bone_m = calibration_data.get("bone_lengths_in_meters_depthmap", {})
            used_echte_groesse = calibration_data.get("echte_groesse_depthmap", 0.0)
            if not bone_m: bone_m = calibration_data.get("bone_lengths_in_meters", {})
            log_messages.append(">> MODUS 1: Lade klassische Depth-Map Kalibrierung.")
        else:
            # Modi 2 und 3 nutzen die NLF-Kalibrierung aus dem Hub
            bone_m = calibration_data.get("bone_lengths_in_meters_nlf", {})
            used_echte_groesse = calibration_data.get("echte_groesse_nlf", 0.0)
            if not bone_m:
                return (pose_data_copy, "Fehler: Calibration Hub hat keine NLF-Daten. Bitte NLF in die Calibration Node stecken!", video_nlf_data, "{}")
            log_messages.append(f">> MODUS {scaling_mode[0]}: Lade dedizierte NLF Kalibrierung aus Hub.")

        # --- BRENNWEITE ---
        fx_calib = calibration_data.get("focal_length_fx", 500.0)
        fx_video = fx_calib
        if intrinsics_source == "Use NLF Default (5000)":
            fx_video = 5000.0
            log_messages.append(f">> Nutze NLF Default Brennweite: fx={fx_video}")
        elif video_intrinsics_json:
            try:
                int_data = json.loads(video_intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None:
                        fx_video = float(matrix[0][0])
                        log_messages.append(f">> Kamera-Intrinsics geladen: fx={fx_video:.2f}")
            except: pass

        valid_depth_frames = None
        if valid_depth_indices:
            try:
                valid_depth_frames = [int(x.strip()) for x in valid_depth_indices.split(",") if x.strip().isdigit()]
            except: pass

        def get_nearest_depth_idx(target_idx, valid_list):
            if not valid_list: return target_idx, target_idx
            nearest_val = min(valid_list, key=lambda x: abs(x - target_idx))
            return valid_list.index(nearest_val), nearest_val

        depth_np = None
        if video_depth_map is not None:
            depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        
        H, W = (depth_np.shape[1], depth_np.shape[2]) if depth_np is not None else (1024, 1024)
        head_px_calib = calibration_data.get("bone_length_for_scaler", {}).get("head", 0.0)

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            if depth_img is None: return 0.5
            skeleton_connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        def get_nlf_2d_depth(pts, depth_img, v_idx, W, H):
            if depth_img is None: return 0.5
            depth_vals = []
            for idx in range(len(pts)):
                if np.linalg.norm(pts[idx]) > 1e-5:
                    px, py = int(pts[idx][0]), int(pts[idx][1])
                    if 0 <= px < W and 0 <= py < H:
                        val = depth_img[v_idx, py, px]
                        depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        # --- RADAR ---
        all_frames_data = []
        frontal_indices = []
        
        if best_frame_source == "NLF (3D SMPL)" and video_nlf_data is None:
            best_frame_source = "PoseData (2D)"

        is_dict = isinstance(video_nlf_data, dict)
        raw_poses = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if is_dict and video_nlf_data else video_nlf_data

        if best_frame_source == "NLF (3D SMPL)":
            for i in range(len(pose_metas)):
                if raw_poses is None or i >= len(raw_poses) or raw_poses[i] is None or len(raw_poses[i]) == 0:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                frame_data = raw_poses[i]
                is_tensor = hasattr(frame_data, 'dim')
                pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))

                def is_val_nlf(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5
                has_knees = is_val_nlf(4) or is_val_nlf(5)
                has_ankles = is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0
                if include_head and length > 0: length += (head_px_calib / 2.0)

                is_frontal, frontal_pts, max_angle = False, 0.0, 90.0
                if len(pts) >= 18:
                    dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                    dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                    angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                    angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                    max_angle = max(angle_h, angle_s)

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        else: # 2D RADAR
            pose_input_3d = raw_poses
            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                
                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
                if not valid_y:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                top_y, bottom_y = min(valid_y), max(valid_y)
                if not include_head and is_val(kps, confs, 1): top_y = kps[1][1]
                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal, frontal_pts = False, 0.0
                
                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pose_3d_frame = pose_input_3d[i]
                    if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                        person_3d = pose_3d_frame[0]
                        num_joints = len(person_3d)
                        idx_r, idx_l = 2, 5
                        if num_joints == 17: idx_r, idx_l = 11, 14
                        elif num_joints in [24, 45, 68]: idx_r, idx_l = 16, 17
                            
                        if num_joints > max(idx_r, idx_l):
                            dx, dz = float(person_3d[idx_r][0]) - float(person_3d[idx_l][0]), float(person_3d[idx_r][2]) - float(person_3d[idx_l][2])
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        candidates = frontal_indices if len(frontal_indices) > 0 else list(range(len(pose_metas)))
        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            data = all_frames_data[idx]
            bein_pts = max(1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0, 500.0 if not (data['has_feet'] or data['has_ankles']) and data['has_knees'] else 0.0)
            total_score = bein_pts + (500.0 if data['has_feet'] and data['is_frontal'] else 0.0) + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score: best_score, best_idx = total_score, idx

        log_messages.append(f"\n-> Gewinner Frame (Anchor): {best_idx}")

        start_idx, end_idx = max(0, best_idx - anchor_window), min(len(pose_metas) - 1, best_idx + anchor_window)
        
        # --- NLF TEMPORAL SMOOTHING VORBEREITUNG (Nur wenn Modus 2 aktiv ist) ---
        nlf_raw_z_values = {}
        if scaling_mode.startswith("2") and raw_poses is not None:
            log_messages.append(f"\n--- BERECHNE NLF Z-TIEFE & SMOOTHING (Window: {nlf_smoothing_window}) ---")
            smooth_start = max(0, start_idx - nlf_smoothing_window)
            smooth_end = min(len(pose_metas) - 1, end_idx + nlf_smoothing_window)
            
            for i in range(smooth_start, smooth_end + 1):
                if i < len(raw_poses) and raw_poses[i] is not None and len(raw_poses[i]) > 0:
                    frame_data = raw_poses[i]
                    is_tensor = hasattr(frame_data, 'dim')
                    pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
                    
                    torso_indices = [0, 3, 6, 9, 12]
                    valid_z = [float(pts[idx][2]) for idx in torso_indices if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
                    if valid_z:
                        nlf_raw_z_values[i] = float(np.mean(valid_z))
                    else:
                        nlf_raw_z_values[i] = 0.5
                else:
                    nlf_raw_z_values[i] = 0.5

        sum_scale_factors, valid_frames = 0.0, 0
        total_camera_distance = 0.0 # Für das finale JSON

        for i in range(start_idx, end_idx + 1):
            kps, confs = getattr(pose_metas[i], "kps_body", []), getattr(pose_metas[i], "kps_body_p", None)
            frame_ist_px, frame_soll_m = 0.0, 0.0

            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1); frame_soll_m += bone_m.get("head", 0)
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0)
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9); frame_soll_m += bone_m.get("thigh", 0)
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10); frame_soll_m += bone_m.get("calf", 0)

            if frame_ist_px == 0 or frame_soll_m == 0: continue

            # --- TIEFEN-BERECHNUNG WEICHE (V54 NEU: 3 MODI) ---
            frame_depth = 0.5
            
            # MODUS 2: NLF Z-Axis
            if scaling_mode.startswith("2") and i in nlf_raw_z_values:
                z_window = []
                for w in range(max(0, i - nlf_smoothing_window), min(len(pose_metas), i + nlf_smoothing_window + 1)):
                    if w in nlf_raw_z_values: z_window.append(nlf_raw_z_values[w])
                
                raw_z = nlf_raw_z_values[i]
                frame_depth = float(np.mean(z_window)) if z_window else raw_z
                frame_depth = abs(frame_depth)
                if frame_depth < 0.01: frame_depth = 0.01
                log_messages.append(f"\n  Frame {i} | NLF-Z-Tiefe: Smoothed Z={frame_depth:.3f}m")

            # MODUS 1 oder 3: Depth Map (mit unterschiedlichen Masken)
            else:
                if valid_depth_frames is not None:
                    depth_v_idx, borrowed_from = get_nearest_depth_idx(i, valid_depth_frames)
                else:
                    depth_v_idx = min(i, (depth_np.shape[0] - 1) if depth_np is not None else 0)

                # MODUS 3: NLF 2D Overlay
                if scaling_mode.startswith("3") and raw_poses is not None and i < len(raw_poses) and raw_poses[i] is not None:
                    frame_data = raw_poses[i]
                    is_tensor = hasattr(frame_data, 'dim')
                    pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
                    
                    frame_depth = get_nlf_2d_depth(pts, depth_np, depth_v_idx, W, H)
                    log_messages.append(f"\n  Frame {i} | Depth Map via NLF 2D Overlay abgelesen.")
                
                # MODUS 1: PoseData 2D Maske
                else:
                    frame_depth = get_skeleton_depth(kps, confs, depth_np, depth_v_idx, W, H)
                    log_messages.append(f"\n  Frame {i} | Depth Map via PoseData 2D Maske abgelesen.")
                
                if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            total_camera_distance += frame_depth
            expected_px = (frame_soll_m * fx_video) / frame_depth
            scale_factor = expected_px / frame_ist_px
            sum_scale_factors += scale_factor
            valid_frames += 1

            log_messages.append(f"    Ist-Pixel: {frame_ist_px:.1f} px | Ist-Meter: {frame_soll_m:.3f} m")
            log_messages.append(f"    Finale Tiefe: {frame_depth:.3f} m -> Soll-Pixel: {expected_px:.1f} px")
            log_messages.append(f"    Lokaler Faktor: {scale_factor:.3f}x")

        if valid_frames == 0: return (pose_data_copy, "Fehler: Keine validen Körperteile gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        avg_camera_dist = total_camera_distance / valid_frames
        
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor ({valid_frames} Frames): {final_scale:.3f}x")
        
        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b, c_b = getattr(pose_metas[best_idx], "kps_body", []), getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x: global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        # --- NEU V54: JSON CONFIG OUTPUT ERWEITERT ---
        config_dict = {
            "anchor_scale": float(final_scale), 
            "scale_x_factor": float(scale_x), 
            "pivot_x": float(global_pivot_x), 
            "pivot_y": float(global_pivot_y),
            "echte_groesse_m": float(used_echte_groesse),
            "camera_distance_m": float(avg_camera_dist)
        }
        config_str = json.dumps(config_dict)
        log_messages.append(f">> Schreibe Metriken in Config: {config_dict}")

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseCalibrationV32:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
                "normalize_bones_to_100": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "depth_nah": ("IMAGE",),
                "depth_fern": ("IMAGE",),
                "nlf_data_nah": ("NLFPRED",),
                "nlf_data_fern": ("NLFPRED",),
                "intrinsics_json": ("STRING", {"forceInput": True}),
                "config_data": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V32: Hub mit reparierter Slope/Intercept Logik & Z-Werten für 3D-Direct-Compare."

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, pose_fern_scaled, pose_fern_unscaled, norm_method, min_confidence, invert_depth, use_pinhole_math=True, normalize_bones_to_100=True, depth_nah=None, depth_fern=None, nlf_data_nah=None, nlf_data_fern=None, intrinsics_json=None, config_data="{}"):
        import json
        import math
        import numpy as np

        log_messages = ["=== V32 CALIBRATION LOG (UNIVERSAL DUAL HUB) ==="]

        try:
            config = json.loads(config_data)
        except:
            config = {}

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
            return c >= min_confidence

        # --- 1. DEPTH MAP LOGIK ---
        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px = int(x1 + t * (x2 - x1))
                            py = int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        def get_depth_for_pose(pose_u, depth_img):
            if depth_img is None: return 0.0
            meta_u = pose_u.get("pose_metas", [])[0] if pose_u.get("pose_metas") else None
            if not meta_u: return 0.5
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            depth_np = depth_img.cpu().numpy() if hasattr(depth_img, 'cpu') else depth_img
            H, W = depth_np.shape[1], depth_np.shape[2]
            return get_skeleton_depth(kps_u, confs_u, depth_np, 0, W, H)

        # --- 1.b NLF 3D TIEFEN-LOGIK ---
        def get_nlf_torso_z(nlf_data):
            if nlf_data is None: return 0.0
            is_dict = isinstance(nlf_data, dict)
            raw_poses = nlf_data.get('joints3d_nonparam', [nlf_data])[0] if is_dict else nlf_data
            if not raw_poses or len(raw_poses) == 0 or raw_poses[0] is None: return 0.0
            
            frame_data = raw_poses[0]
            is_tensor = hasattr(frame_data, 'dim')
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
            
            if len(pts) < 17: return 0.0
            torso_indices = [0, 3, 6, 9, 12]
            valid_z = [float(pts[idx][2]) for idx in torso_indices if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
            if valid_z: return float(np.mean(valid_z))
            return 0.0

        # --- 2. KNOCHEN EXTRAHIEREN ---
        def extract_2d_bones(pose_data):
            try:
                meta = pose_data.get("pose_metas", [])[0]
                kps = getattr(meta, "kps_body", None)
                if kps is None or len(kps) < 14: return None, None
                def dist_2d(idx1, idx2):
                    if idx1 >= len(kps) or idx2 >= len(kps): return 0.0
                    p1, p2 = kps[idx1], kps[idx2]
                    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2: return 0.0
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                mid_hip_x = (kps[8][0] + kps[11][0]) / 2.0
                mid_hip_y = (kps[8][1] + kps[11][1]) / 2.0
                
                raw_bones = {
                    "head": dist_2d(0, 1),
                    "torso": math.sqrt((kps[1][0] - mid_hip_x)**2 + (kps[1][1] - mid_hip_y)**2),
                    "shoulder_width": dist_2d(2, 5), "hip_width": dist_2d(8, 11),
                    "r_arm": dist_2d(2, 3), "r_forearm": dist_2d(3, 4),
                    "l_arm": dist_2d(5, 6), "l_forearm": dist_2d(6, 7),
                    "r_thigh": dist_2d(8, 9), "r_calf": dist_2d(9, 10),
                    "l_thigh": dist_2d(11, 12), "l_calf": dist_2d(12, 13)
                }

                sym_bones = {
                    "head": raw_bones["head"], "torso": raw_bones["torso"],
                    "shoulder_width": raw_bones["shoulder_width"], "hip_width": raw_bones["hip_width"],
                    "r_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0, "l_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "r_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0, "l_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "r_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0, "l_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "r_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0, "l_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0
                }

                norm_bones = {}
                if normalize_bones_to_100:
                    for k, v in sym_bones.items():
                        norm_bones[k] = (v / sym_bones["torso"]) * 100.0 if sym_bones["torso"] > 0 else 0
                else: norm_bones = sym_bones.copy()
                return sym_bones, norm_bones
            except Exception as e: return None, None

        log_messages.append(f"Norm-Methode: {norm_method}")

        unscaled_bones_nah, _ = extract_2d_bones(pose_nah_scaled)
        unscaled_bones_fern, true_3d_bones = extract_2d_bones(pose_fern_scaled)

        # --- 3. NORM BERECHNUNG ---
        def calc_norm_from_bones(bones, method):
            if not bones: return 100.0
            if method == "Torso (Neck-Hip)": return bones["torso"]
            return bones["head"] + bones["torso"] + bones["r_thigh"] + bones["r_calf"]

        norm_nah_raw = calc_norm_from_bones(unscaled_bones_nah, norm_method)
        norm_fern = calc_norm_from_bones(unscaled_bones_fern, norm_method)
        norm_nah = norm_nah_raw

        depth_map_c = get_depth_for_pose(pose_nah_unscaled, depth_nah)
        depth_map_f = get_depth_for_pose(pose_fern_unscaled, depth_fern)
        
        nlf_z_c = get_nlf_torso_z(nlf_data_nah)
        nlf_z_f = get_nlf_torso_z(nlf_data_fern)

        if invert_depth:
            depth_map_c = 1.0 / max(depth_map_c, 0.0001) if depth_map_c > 0 else 0.0
            depth_map_f = 1.0 / max(depth_map_f, 0.0001) if depth_map_f > 0 else 0.0

        # --- 4. EXTRAPOLATION & SLOPE (REPARIERT) ---
        if unscaled_bones_nah and unscaled_bones_fern:
            torso_nah = unscaled_bones_nah["torso"]
            torso_fern = unscaled_bones_fern["torso"]
            if torso_fern > 0:
                torso_faktor = torso_nah / torso_fern
                norm_nah = norm_fern * torso_faktor
        else:
            if not unscaled_bones_fern: unscaled_bones_fern, true_3d_bones = unscaled_bones_nah, _

        # FEHLENDE SLOPE LOGIK WIEDER EINGEBAUT!
        slope, intercept = 0.0, 1.0
        depth_diff = abs(depth_map_f - depth_map_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_map_f - depth_map_c)
            intercept = norm_nah - (slope * depth_map_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_map_c)

        fx = 500.0
        if intrinsics_json:
            try:
                int_data = json.loads(intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None: fx = float(matrix[0][0])
            except: pass

        # --- 5. BERECHNUNG ECHTE GRÖSSE ---
        echte_groesse_depthmap = 0.0
        if depth_map_f > 0 and depth_map_c > 0:
            delta_z_depth = depth_map_f - depth_map_c
            if use_pinhole_math and delta_z_depth > 0 and (norm_nah - norm_fern) > 0:
                echte_groesse_depthmap = (delta_z_depth * norm_nah * norm_fern) / (fx * (norm_nah - norm_fern))
            else:
                echte_groesse_depthmap = (norm_nah * depth_map_c) / fx

        bone_length_for_scaler = {}
        bone_lengths_in_meters_depthmap = {}

        if unscaled_bones_fern:
            bone_length_for_scaler = {
                "head": unscaled_bones_fern["head"], "torso": unscaled_bones_fern["torso"],
                "thigh": unscaled_bones_fern["r_thigh"], "calf": unscaled_bones_fern["r_calf"]
            }
            total_px = sum(bone_length_for_scaler.values())
            if total_px > 0 and echte_groesse_depthmap > 0:
                for k, px_val in bone_length_for_scaler.items():
                    bone_lengths_in_meters_depthmap[k] = (px_val / total_px) * echte_groesse_depthmap

        calib_data = {
            "perspective_slope": slope, "perspective_intercept": intercept, # REPARIERT!
            "is_depth_inverted": invert_depth, "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math, "focal_length_fx": fx,
            
            "norm_fern": norm_fern, # WICHTIG FÜR MODUS 3
            "depth_c": depth_map_c, "depth_f": depth_map_f,
            "echte_groesse_depthmap": echte_groesse_depthmap,
            "bone_lengths_in_meters_depthmap": bone_lengths_in_meters_depthmap,
            
            "nlf_z_c": nlf_z_c, "nlf_z_f": nlf_z_f,

            "true_3d_bones": true_3d_bones or {},
            "bone_length_for_scaler": bone_length_for_scaler or {},
            "total_3d_height": sum(bone_length_for_scaler.values()) if bone_length_for_scaler else 0.0,
            "config": config
        }

        log_messages.append("\n>> Hub V32: Slope, Intercept und alle 3D-Metriken erfolgreich gespeichert.")
        return (calib_data, "\n".join(log_messages))


class PoseGlobalPerspectiveScalerV55:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "scaling_mode": (["1. Classic 2D + Depth Map", "2. NLF 2D Overlay + Depth Map", "3. Pure NLF 3D Z-Depth (Direct Compare)"], {"default": "1. Classic 2D + Depth Map"}),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "PoseData (2D)"}),
                "nlf_smoothing_window": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1, "tooltip": "Nur für Modus 3: Glättet die Z-Werte über X Frames."}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_depth_map": ("IMAGE",),
                "video_nlf_data": ("NLFPRED",),
                "video_intrinsics_json": ("STRING", {"forceInput": True}),
                "valid_depth_indices": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V55: Drei saubere Modus-Säulen. Modus 3 ignoriert Kameras komplett und rechnet rein relativ im 3D-Raum."

    def process(self, video_pose_data, calibration_data, scaling_mode, best_frame_source, nlf_smoothing_window, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_depth_map=None, video_nlf_data=None, video_intrinsics_json=None, valid_depth_indices=None):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V55 GLOBAL SCALER LOG (MODE: {scaling_mode}) ==="]

        if not pose_metas: return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        is_inverted = calibration_data.get("is_depth_inverted", False)

        # --- MODUS SETUP ---
        mode_id = scaling_mode[0] # "1", "2" oder "3"
        bone_m = {}
        fx_video = 500.0
        baseline_nlf_z = 0.0
        baseline_pixels_fern = 0.0
        bone_length_for_scaler = calibration_data.get("bone_length_for_scaler", {})

        if mode_id in ["1", "2"]:
            # Klassische Pinhole-Mathe mit Depth Map
            bone_m = calibration_data.get("bone_lengths_in_meters_depthmap", {})
            if not bone_m: bone_m = calibration_data.get("bone_lengths_in_meters", {})
            
            fx_video = calibration_data.get("focal_length_fx", 500.0)
            if video_intrinsics_json:
                try:
                    int_data = json.loads(video_intrinsics_json)
                    if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                        matrix = int_data["intrinsics"][0].get("image_0", None)
                        if matrix is not None: fx_video = float(matrix[0][0])
                except: pass
            log_messages.append(f">> Lade Pinhole-Metriken (fx={fx_video:.2f}) für Depth Map.")
        
        elif mode_id == "3":
            # Pure 3D Direct Compare!
            baseline_nlf_z = calibration_data.get("nlf_z_f", 0.0)
            baseline_pixels_fern = calibration_data.get("norm_fern", 0.0)
            if baseline_nlf_z == 0.0:
                return (pose_data_copy, "Fehler: Calibration Hub hat keine NLF-Z Daten gespeichert. NLF in Calibration anstecken!", video_nlf_data, "{}")
            log_messages.append(f">> Lade pure 3D-Metrik: Baseline Z = {baseline_nlf_z:.4f} | Kameras werden ignoriert.")

        valid_depth_frames = None
        if valid_depth_indices:
            try:
                valid_depth_frames = [int(x.strip()) for x in valid_depth_indices.split(",") if x.strip().isdigit()]
            except: pass

        def get_nearest_depth_idx(target_idx, valid_list):
            if not valid_list: return target_idx, target_idx
            nearest_val = min(valid_list, key=lambda x: abs(x - target_idx))
            return valid_list.index(nearest_val), nearest_val

        depth_np = None
        if video_depth_map is not None:
            depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = (depth_np.shape[1], depth_np.shape[2]) if depth_np is not None else (1024, 1024)

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            if depth_img is None: return 0.5
            skeleton_connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        def get_nlf_2d_depth(pts, depth_img, v_idx, W, H):
            if depth_img is None: return 0.5
            depth_vals = []
            for idx in range(len(pts)):
                if np.linalg.norm(pts[idx]) > 1e-5:
                    px, py = int(pts[idx][0]), int(pts[idx][1])
                    if 0 <= px < W and 0 <= py < H:
                        val = depth_img[v_idx, py, px]
                        depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        # --- RADAR ---
        all_frames_data = []
        frontal_indices = []
        is_dict = isinstance(video_nlf_data, dict)
        raw_poses = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if is_dict and video_nlf_data else video_nlf_data

        if best_frame_source == "NLF (3D SMPL)":
            for i in range(len(pose_metas)):
                if raw_poses is None or i >= len(raw_poses) or raw_poses[i] is None or len(raw_poses[i]) == 0:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                frame_data = raw_poses[i]
                is_tensor = hasattr(frame_data, 'dim')
                pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))

                def is_val_nlf(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5
                has_knees = is_val_nlf(4) or is_val_nlf(5)
                has_ankles = is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

                is_frontal, frontal_pts, max_angle = False, 0.0, 90.0
                if len(pts) >= 18:
                    dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                    dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                    angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                    angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                    max_angle = max(angle_h, angle_s)

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        else: # 2D RADAR
            pose_input_3d = raw_poses
            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                
                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
                if not valid_y:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                top_y, bottom_y = min(valid_y), max(valid_y)
                if not include_head and is_val(kps, confs, 1): top_y = kps[1][1]
                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal, frontal_pts = False, 0.0
                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pose_3d_frame = pose_input_3d[i]
                    if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                        person_3d = pose_3d_frame[0]
                        num_joints = len(person_3d)
                        idx_r, idx_l = 2, 5
                        if num_joints == 17: idx_r, idx_l = 11, 14
                        elif num_joints in [24, 45, 68]: idx_r, idx_l = 16, 17
                        if num_joints > max(idx_r, idx_l):
                            dx, dz = float(person_3d[idx_r][0]) - float(person_3d[idx_l][0]), float(person_3d[idx_r][2]) - float(person_3d[idx_l][2])
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        candidates = frontal_indices if len(frontal_indices) > 0 else list(range(len(pose_metas)))
        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            data = all_frames_data[idx]
            bein_pts = max(1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0, 500.0 if not (data['has_feet'] or data['has_ankles']) and data['has_knees'] else 0.0)
            total_score = bein_pts + (500.0 if data['has_feet'] and data['is_frontal'] else 0.0) + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score: best_score, best_idx = total_score, idx

        log_messages.append(f"\n-> Gewinner Frame (Anchor): {best_idx}")
        start_idx, end_idx = max(0, best_idx - anchor_window), min(len(pose_metas) - 1, best_idx + anchor_window)
        
        # --- NLF TEMPORAL SMOOTHING (Nur für Modus 3) ---
        nlf_raw_z_values = {}
        if mode_id == "3" and raw_poses is not None:
            smooth_start = max(0, start_idx - nlf_smoothing_window)
            smooth_end = min(len(pose_metas) - 1, end_idx + nlf_smoothing_window)
            for i in range(smooth_start, smooth_end + 1):
                if i < len(raw_poses) and raw_poses[i] is not None and len(raw_poses[i]) > 0:
                    frame_data = raw_poses[i]
                    is_tensor = hasattr(frame_data, 'dim')
                    pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
                    
                    torso_indices = [0, 3, 6, 9, 12]
                    valid_z = [float(pts[idx][2]) for idx in torso_indices if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
                    nlf_raw_z_values[i] = float(np.mean(valid_z)) if valid_z else 0.5
                else:
                    nlf_raw_z_values[i] = 0.5

        sum_scale_factors, valid_frames = 0.0, 0

        for i in range(start_idx, end_idx + 1):
            kps, confs = getattr(pose_metas[i], "kps_body", []), getattr(pose_metas[i], "kps_body_p", None)
            frame_ist_px = 0.0
            
            # Für Modus 1/2
            frame_soll_m = 0.0
            # Für Modus 3
            frame_soll_px_baseline = 0.0

            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1)
                frame_soll_m += bone_m.get("head", 0); frame_soll_px_baseline += bone_length_for_scaler.get("head", 0)
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2)
                frame_soll_m += bone_m.get("torso", 0); frame_soll_px_baseline += bone_length_for_scaler.get("torso", 0)
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9)
                frame_soll_m += bone_m.get("thigh", 0); frame_soll_px_baseline += bone_length_for_scaler.get("thigh", 0)
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10)
                frame_soll_m += bone_m.get("calf", 0); frame_soll_px_baseline += bone_length_for_scaler.get("calf", 0)

            if frame_ist_px == 0: continue

            expected_px = 0.0

            # --- SÄULE 3: Pure NLF 3D-to-3D Compare ---
            if mode_id == "3" and i in nlf_raw_z_values:
                z_window = []
                for w in range(max(0, i - nlf_smoothing_window), min(len(pose_metas), i + nlf_smoothing_window + 1)):
                    if w in nlf_raw_z_values: z_window.append(nlf_raw_z_values[w])
                
                smoothed_current_z = abs(float(np.mean(z_window))) if z_window else abs(nlf_raw_z_values[i])
                if smoothed_current_z < 0.01: smoothed_current_z = 0.01
                
                # Das geniale: Ziel-Pixel = Baseline-Pixel * (Baseline-Tiefe / Aktuelle Tiefe)
                expected_px = frame_soll_px_baseline * (abs(baseline_nlf_z) / smoothed_current_z)
                
                log_messages.append(f"\n  Frame {i} | 3D-Compare: Aktuelles Z={smoothed_current_z:.3f} (Soll-Px = {expected_px:.1f})")

            # --- SÄULE 1 & 2: Depth Map (Pinhole) ---
            else:
                if frame_soll_m == 0: continue
                if valid_depth_frames is not None:
                    depth_v_idx, borrowed_from = get_nearest_depth_idx(i, valid_depth_frames)
                else:
                    depth_v_idx = min(i, (depth_np.shape[0] - 1) if depth_np is not None else 0)

                if mode_id == "2" and raw_poses is not None and i < len(raw_poses) and raw_poses[i] is not None:
                    frame_data = raw_poses[i]
                    is_tensor = hasattr(frame_data, 'dim')
                    pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
                    frame_depth = get_nlf_2d_depth(pts, depth_np, depth_v_idx, W, H)
                    log_messages.append(f"\n  Frame {i} | Depth Map via NLF 2D Overlay")
                else:
                    frame_depth = get_skeleton_depth(kps, confs, depth_np, depth_v_idx, W, H)
                    log_messages.append(f"\n  Frame {i} | Depth Map via PoseData 2D Maske")
                
                if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)
                expected_px = (frame_soll_m * fx_video) / frame_depth

            scale_factor = expected_px / frame_ist_px
            sum_scale_factors += scale_factor
            valid_frames += 1

            log_messages.append(f"    Ist-Pixel der Maske: {frame_ist_px:.1f} px | Lokaler Skalierungs-Faktor: {scale_factor:.3f}x")

        if valid_frames == 0: return (pose_data_copy, "Fehler: Keine validen Körperteile gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor ({valid_frames} Frames): {final_scale:.3f}x")
        
        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b, c_b = getattr(pose_metas[best_idx], "kps_body", []), getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x: global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        config_dict = {
            "anchor_scale": float(final_scale), 
            "scale_x_factor": float(scale_x), 
            "pivot_x": float(global_pivot_x), 
            "pivot_y": float(global_pivot_y)
        }
        config_str = json.dumps(config_dict)

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseCalibrationV33:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_nah_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "pose_fern_scaled": ("POSEDATA", {"tooltip": "Skalierte Pose für Pixel-Größe"}),
                "pose_fern_unscaled": ("POSEDATA", {"tooltip": "Originale Pose als Maske für Depth-Map"}),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "use_pinhole_math": ("BOOLEAN", {"default": True}),
                "normalize_bones_to_100": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "depth_nah": ("IMAGE",),
                "depth_fern": ("IMAGE",),
                "nlf_data_nah": ("NLFPRED",),
                "nlf_data_fern": ("NLFPRED",),
                "intrinsics_json": ("STRING", {"forceInput": True}),
                "config_data": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V33: Universal Hub. Behält ALLE alten 2D/Pinhole Daten und fügt echte NLF 3D-Knochen-Längen hinzu."

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, pose_fern_scaled, pose_fern_unscaled, norm_method, min_confidence, invert_depth, use_pinhole_math=True, normalize_bones_to_100=True, depth_nah=None, depth_fern=None, nlf_data_nah=None, nlf_data_fern=None, intrinsics_json=None, config_data="{}"):
        import json
        import math
        import numpy as np

        log_messages = ["=== V33 CALIBRATION LOG (DUAL HUB + PURE 3D BONES) ==="]

        try:
            config = json.loads(config_data)
        except:
            config = {}

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
            return c >= min_confidence

        # --- 1. DEPTH MAP LOGIK (Unverändert) ---
        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            skeleton_connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px = int(x1 + t * (x2 - x1))
                            py = int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            
            if not depth_vals:
                for idx in range(len(kps)):
                    if is_val(kps, confs, idx):
                        px, py = int(kps[idx][0]), int(kps[idx][1])
                        if 0 <= px < W and 0 <= py < H:
                            val = depth_img[v_idx, py, px]
                            depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        def get_depth_for_pose(pose_u, depth_img):
            if depth_img is None: return 0.0
            meta_u = pose_u.get("pose_metas", [])[0] if pose_u.get("pose_metas") else None
            if not meta_u: return 0.5
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            depth_np = depth_img.cpu().numpy() if hasattr(depth_img, 'cpu') else depth_img
            H, W = depth_np.shape[1], depth_np.shape[2]
            return get_skeleton_depth(kps_u, confs_u, depth_np, 0, W, H)

        def get_nlf_torso_z(nlf_data):
            if nlf_data is None: return 0.0
            is_dict = isinstance(nlf_data, dict)
            raw_poses = nlf_data.get('joints3d_nonparam', [nlf_data])[0] if is_dict else nlf_data
            if not raw_poses or len(raw_poses) == 0 or raw_poses[0] is None: return 0.0
            frame_data = raw_poses[0]
            is_tensor = hasattr(frame_data, 'dim')
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
            if len(pts) < 17: return 0.0
            torso_indices = [0, 3, 6, 9, 12]
            valid_z = [float(pts[idx][2]) for idx in torso_indices if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
            if valid_z: return float(np.mean(valid_z))
            return 0.0

        # --- NEU V33: PURE 3D KNOCHEN EXTRAKTION (Für Scaler Modus 3) ---
        def extract_pure_nlf_3d_bones(nlf_data):
            if nlf_data is None: return {}
            is_dict = isinstance(nlf_data, dict)
            raw_poses = nlf_data.get('joints3d_nonparam', [nlf_data])[0] if is_dict else nlf_data
            if not raw_poses or len(raw_poses) == 0 or raw_poses[0] is None: return {}
            frame_data = raw_poses[0]
            is_tensor = hasattr(frame_data, 'dim')
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
            
            if len(pts) < 16: return {}
            
            def dist3d(idx1, idx2):
                if idx1 >= len(pts) or idx2 >= len(pts): return 0.0
                p1, p2 = pts[idx1], pts[idx2]
                if np.linalg.norm(p1) < 1e-5 or np.linalg.norm(p2) < 1e-5: return 0.0
                return float(np.linalg.norm(p1 - p2))

            # SMPL Format Indizes: 0:Pelvis, 1:L_Hip, 2:R_Hip, 4:L_Knee, 5:R_Knee, 7:L_Ankle, 8:R_Ankle, 12:Neck, 15:Head
            head = dist3d(12, 15)
            torso = dist3d(0, 12) # Pelvis zu Neck
            l_thigh, r_thigh = dist3d(1, 4), dist3d(2, 5)
            l_calf, r_calf = dist3d(4, 7), dist3d(5, 8)

            thigh_avg = (l_thigh + r_thigh) / 2.0 if (l_thigh > 0 and r_thigh > 0) else max(l_thigh, r_thigh)
            calf_avg = (l_calf + r_calf) / 2.0 if (l_calf > 0 and r_calf > 0) else max(l_calf, r_calf)

            return {
                "head": head,
                "torso": torso,
                "thigh": thigh_avg,
                "calf": calf_avg
            }

        # --- 2. 2D KNOCHEN EXTRAHIEREN (Die Pixel, unverändert) ---
        def extract_2d_bones(pose_data):
            try:
                meta = pose_data.get("pose_metas", [])[0]
                kps = getattr(meta, "kps_body", None)
                if kps is None or len(kps) < 14: return None, None
                def dist_2d(idx1, idx2):
                    if idx1 >= len(kps) or idx2 >= len(kps): return 0.0
                    p1, p2 = kps[idx1], kps[idx2]
                    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2: return 0.0
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                mid_hip_x = (kps[8][0] + kps[11][0]) / 2.0
                mid_hip_y = (kps[8][1] + kps[11][1]) / 2.0
                
                raw_bones = {
                    "head": dist_2d(0, 1),
                    "torso": math.sqrt((kps[1][0] - mid_hip_x)**2 + (kps[1][1] - mid_hip_y)**2),
                    "shoulder_width": dist_2d(2, 5), "hip_width": dist_2d(8, 11),
                    "r_arm": dist_2d(2, 3), "r_forearm": dist_2d(3, 4),
                    "l_arm": dist_2d(5, 6), "l_forearm": dist_2d(6, 7),
                    "r_thigh": dist_2d(8, 9), "r_calf": dist_2d(9, 10),
                    "l_thigh": dist_2d(11, 12), "l_calf": dist_2d(12, 13)
                }

                sym_bones = {
                    "head": raw_bones["head"], "torso": raw_bones["torso"],
                    "shoulder_width": raw_bones["shoulder_width"], "hip_width": raw_bones["hip_width"],
                    "r_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0, "l_arm": (raw_bones["r_arm"] + raw_bones["l_arm"]) / 2.0,
                    "r_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0, "l_forearm": (raw_bones["r_forearm"] + raw_bones["l_forearm"]) / 2.0,
                    "r_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0, "l_thigh": (raw_bones["r_thigh"] + raw_bones["l_thigh"]) / 2.0,
                    "r_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0, "l_calf": (raw_bones["r_calf"] + raw_bones["l_calf"]) / 2.0
                }

                norm_bones = {}
                if normalize_bones_to_100:
                    for k, v in sym_bones.items():
                        norm_bones[k] = (v / sym_bones["torso"]) * 100.0 if sym_bones["torso"] > 0 else 0
                else: norm_bones = sym_bones.copy()
                return sym_bones, norm_bones
            except Exception as e: return None, None

        log_messages.append(f"Norm-Methode: {norm_method}")

        unscaled_bones_nah, _ = extract_2d_bones(pose_nah_scaled)
        unscaled_bones_fern, true_3d_bones = extract_2d_bones(pose_fern_scaled)

        # Die neuen echten 3D-Metriken abrufen!
        bone_lengths_nlf_3d = extract_pure_nlf_3d_bones(nlf_data_fern)

        # --- 3. NORM BERECHNUNG ---
        def calc_norm_from_bones(bones, method):
            if not bones: return 100.0
            if method == "Torso (Neck-Hip)": return bones["torso"]
            return bones["head"] + bones["torso"] + bones["r_thigh"] + bones["r_calf"]

        norm_nah_raw = calc_norm_from_bones(unscaled_bones_nah, norm_method)
        norm_fern = calc_norm_from_bones(unscaled_bones_fern, norm_method)
        norm_nah = norm_nah_raw

        depth_map_c = get_depth_for_pose(pose_nah_unscaled, depth_nah)
        depth_map_f = get_depth_for_pose(pose_fern_unscaled, depth_fern)
        
        nlf_z_c = get_nlf_torso_z(nlf_data_nah)
        nlf_z_f = get_nlf_torso_z(nlf_data_fern)

        if invert_depth:
            depth_map_c = 1.0 / max(depth_map_c, 0.0001) if depth_map_c > 0 else 0.0
            depth_map_f = 1.0 / max(depth_map_f, 0.0001) if depth_map_f > 0 else 0.0

        # --- 4. EXTRAPOLATION & SLOPE ---
        if unscaled_bones_nah and unscaled_bones_fern:
            torso_nah = unscaled_bones_nah["torso"]
            torso_fern = unscaled_bones_fern["torso"]
            if torso_fern > 0:
                torso_faktor = torso_nah / torso_fern
                norm_nah = norm_fern * torso_faktor
        else:
            if not unscaled_bones_fern: unscaled_bones_fern, true_3d_bones = unscaled_bones_nah, _

        slope, intercept = 0.0, 1.0
        depth_diff = abs(depth_map_f - depth_map_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_map_f - depth_map_c)
            intercept = norm_nah - (slope * depth_map_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_map_c)

        fx = 500.0
        if intrinsics_json:
            try:
                int_data = json.loads(intrinsics_json)
                if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                    matrix = int_data["intrinsics"][0].get("image_0", None)
                    if matrix is not None: fx = float(matrix[0][0])
            except: pass

        # --- 5. BERECHNUNG ECHTE GRÖSSE ---
        echte_groesse_depthmap = 0.0
        if depth_map_f > 0 and depth_map_c > 0:
            delta_z_depth = depth_map_f - depth_map_c
            if use_pinhole_math and delta_z_depth > 0 and (norm_nah - norm_fern) > 0:
                echte_groesse_depthmap = (delta_z_depth * norm_nah * norm_fern) / (fx * (norm_nah - norm_fern))
            else:
                echte_groesse_depthmap = (norm_nah * depth_map_c) / fx

        bone_length_for_scaler = {}
        bone_lengths_in_meters_depthmap = {}

        if unscaled_bones_fern:
            bone_length_for_scaler = {
                "head": unscaled_bones_fern["head"], "torso": unscaled_bones_fern["torso"],
                "thigh": unscaled_bones_fern["r_thigh"], "calf": unscaled_bones_fern["r_calf"]
            }
            total_px = sum(bone_length_for_scaler.values())
            if total_px > 0 and echte_groesse_depthmap > 0:
                for k, px_val in bone_length_for_scaler.items():
                    bone_lengths_in_meters_depthmap[k] = (px_val / total_px) * echte_groesse_depthmap

        calib_data = {
            # ALLE ALTEN WERTE BLEIBEN ERHALTEN:
            "perspective_slope": slope, "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth, "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math, "focal_length_fx": fx,
            "norm_fern": norm_fern,
            "depth_c": depth_map_c, "depth_f": depth_map_f,
            "echte_groesse_depthmap": echte_groesse_depthmap,
            "bone_lengths_in_meters_depthmap": bone_lengths_in_meters_depthmap,
            "nlf_z_c": nlf_z_c, "nlf_z_f": nlf_z_f,
            "true_3d_bones": true_3d_bones or {},
            "bone_length_for_scaler": bone_length_for_scaler or {},
            "total_3d_height": sum(bone_length_for_scaler.values()) if bone_length_for_scaler else 0.0,
            
            # DER NEUE WERT FÜR MODUS 3 (Pure 3D):
            "bone_lengths_nlf_3d": bone_lengths_nlf_3d,
            "config": config
        }

        if bone_lengths_nlf_3d:
            log_messages.append(f">> Extrahiere echte NLF 3D-Metriken (Pythagoras): {bone_lengths_nlf_3d}")

        return (calib_data, "\n".join(log_messages))


class PoseGlobalPerspectiveScalerV56:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "scaling_mode": (["1. Classic 2D + Depth Map", "2. NLF 2D Overlay + Depth Map", "3. Pure NLF 3D Compare (No Pinhole)"], {"default": "1. Classic 2D + Depth Map"}),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "PoseData (2D)"}),
                "nlf_smoothing_window": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1, "tooltip": "Nur für Modus 3: Glättet die 3D-Knochenlängen über X Frames."}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),
            },
            "optional": {
                "video_depth_map": ("IMAGE",),
                "video_nlf_data": ("NLFPRED",),
                "video_intrinsics_json": ("STRING", {"forceInput": True}),
                "valid_depth_indices": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V56: Modus 3 skaliert Pixel rein über das Verhältnis der physikalischen 3D-Knochenlängen."

    def process(self, video_pose_data, calibration_data, scaling_mode, best_frame_source, nlf_smoothing_window, include_head, anchor_window, min_confidence, frontal_method, frontal_2d_threshold, frontal_3d_angle_tolerance, scale_2d_axes, video_depth_map=None, video_nlf_data=None, video_intrinsics_json=None, valid_depth_indices=None):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V56 GLOBAL SCALER LOG (MODE: {scaling_mode}) ==="]

        if not pose_metas: return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        is_inverted = calibration_data.get("is_depth_inverted", False)

        # --- MODUS SETUP ---
        mode_id = scaling_mode[0] # "1", "2" oder "3"
        bone_m = {}
        fx_video = 500.0
        
        # Für Modus 3: Echte 3D Vektoren aus dem Hub
        calib_nlf_3d = calibration_data.get("bone_lengths_nlf_3d", {})

        if mode_id in ["1", "2"]:
            bone_m = calibration_data.get("bone_lengths_in_meters_depthmap", {})
            if not bone_m: bone_m = calibration_data.get("bone_lengths_in_meters", {})
            
            fx_video = calibration_data.get("focal_length_fx", 500.0)
            if video_intrinsics_json:
                try:
                    int_data = json.loads(video_intrinsics_json)
                    if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                        matrix = int_data["intrinsics"][0].get("image_0", None)
                        if matrix is not None: fx_video = float(matrix[0][0])
                except: pass
            log_messages.append(f">> Lade Pinhole-Metriken (fx={fx_video:.2f}) für Depth Map.")
        
        elif mode_id == "3":
            if not calib_nlf_3d:
                return (pose_data_copy, "Fehler: Hub hat keine bone_lengths_nlf_3d! Bitte NLF in die V33 Node stecken.", video_nlf_data, "{}")
            log_messages.append(f">> Lade pure 3D-Metrik: {calib_nlf_3d} | Pinhole und Z-Tiefe werden komplett ignoriert!")

        valid_depth_frames = None
        if valid_depth_indices:
            try:
                valid_depth_frames = [int(x.strip()) for x in valid_depth_indices.split(",") if x.strip().isdigit()]
            except: pass

        def get_nearest_depth_idx(target_idx, valid_list):
            if not valid_list: return target_idx, target_idx
            nearest_val = min(valid_list, key=lambda x: abs(x - target_idx))
            return valid_list.index(nearest_val), nearest_val

        depth_np = None
        if video_depth_map is not None:
            depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = (depth_np.shape[1], depth_np.shape[2]) if depth_np is not None else (1024, 1024)

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0])**2 + (kps[i1][1] - kps[i2][1])**2)

        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            if depth_img is None: return 0.5
            skeleton_connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (8,11)]
            depth_vals = []
            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))
                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        def get_nlf_2d_depth(pts, depth_img, v_idx, W, H):
            if depth_img is None: return 0.5
            depth_vals = []
            for idx in range(len(pts)):
                if np.linalg.norm(pts[idx]) > 1e-5:
                    px, py = int(pts[idx][0]), int(pts[idx][1])
                    if 0 <= px < W and 0 <= py < H:
                        val = depth_img[v_idx, py, px]
                        depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))
            if depth_vals: return float(np.mean(depth_vals))
            return 0.5

        # Hilfsfunktion für Modus 3: 3D Knochen aus einem bestimmten Video-Frame ziehen
        def get_nlf_3d_bones_for_frame(raw_poses, f_idx):
            if raw_poses is None or f_idx >= len(raw_poses) or raw_poses[f_idx] is None or len(raw_poses[f_idx]) == 0:
                return {}
            frame_data = raw_poses[f_idx]
            is_tensor = hasattr(frame_data, 'dim')
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
            
            if len(pts) < 16: return {}
            def dist3d(idx1, idx2):
                if idx1 >= len(pts) or idx2 >= len(pts): return 0.0
                p1, p2 = pts[idx1], pts[idx2]
                if np.linalg.norm(p1) < 1e-5 or np.linalg.norm(p2) < 1e-5: return 0.0
                return float(np.linalg.norm(p1 - p2))

            h = dist3d(12, 15)
            t = dist3d(0, 12)
            th_avg = (dist3d(1, 4) + dist3d(2, 5)) / 2.0 if (dist3d(1, 4)>0 and dist3d(2, 5)>0) else max(dist3d(1, 4), dist3d(2, 5))
            c_avg = (dist3d(4, 7) + dist3d(5, 8)) / 2.0 if (dist3d(4, 7)>0 and dist3d(5, 8)>0) else max(dist3d(4, 7), dist3d(5, 8))
            return {"head": h, "torso": t, "thigh": th_avg, "calf": c_avg}

        # --- RADAR ---
        all_frames_data = []
        frontal_indices = []
        is_dict = isinstance(video_nlf_data, dict)
        raw_poses = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0] if is_dict and video_nlf_data else video_nlf_data

        if best_frame_source == "NLF (3D SMPL)":
            for i in range(len(pose_metas)):
                if raw_poses is None or i >= len(raw_poses) or raw_poses[i] is None or len(raw_poses[i]) == 0:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                frame_data = raw_poses[i]
                is_tensor = hasattr(frame_data, 'dim')
                pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))

                def is_val_nlf(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5
                has_knees = is_val_nlf(4) or is_val_nlf(5)
                has_ankles = is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

                is_frontal, frontal_pts, max_angle = False, 0.0, 90.0
                if len(pts) >= 18:
                    dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                    dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                    angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                    angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                    max_angle = max(angle_h, angle_s)

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        else: # 2D RADAR
            pose_input_3d = raw_poses
            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)
                
                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]
                if not valid_y:
                    all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                    continue

                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                top_y, bottom_y = min(valid_y), max(valid_y)
                if not include_head and is_val(kps, confs, 1): top_y = kps[1][1]
                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal, frontal_pts = False, 0.0
                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pose_3d_frame = pose_input_3d[i]
                    if pose_3d_frame is not None and len(pose_3d_frame) > 0:
                        person_3d = pose_3d_frame[0]
                        num_joints = len(person_3d)
                        idx_r, idx_l = 2, 5
                        if num_joints == 17: idx_r, idx_l = 11, 14
                        elif num_joints in [24, 45, 68]: idx_r, idx_l = 16, 17
                        if num_joints > max(idx_r, idx_l):
                            dx, dz = float(person_3d[idx_r][0]) - float(person_3d[idx_l][0]), float(person_3d[idx_r][2]) - float(person_3d[idx_l][2])
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))
                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
                if is_frontal: frontal_indices.append(i)

        candidates = frontal_indices if len(frontal_indices) > 0 else list(range(len(pose_metas)))
        max_body_length = max([all_frames_data[idx]['length'] for idx in candidates]) if candidates else 1.0
        if max_body_length == 0: max_body_length = 1.0

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            data = all_frames_data[idx]
            bein_pts = max(1000.0 if (data['has_feet'] or data['has_ankles']) else 0.0, 500.0 if not (data['has_feet'] or data['has_ankles']) and data['has_knees'] else 0.0)
            total_score = bein_pts + (500.0 if data['has_feet'] and data['is_frontal'] else 0.0) + data['frontal_pts'] + ((data['length'] / max_body_length) * 100.0)
            if total_score > best_score: best_score, best_idx = total_score, idx

        log_messages.append(f"\n-> Gewinner Frame (Anchor): {best_idx}")
        start_idx, end_idx = max(0, best_idx - anchor_window), min(len(pose_metas) - 1, best_idx + anchor_window)
        
        sum_scale_factors, valid_frames = 0.0, 0

        for i in range(start_idx, end_idx + 1):
            kps, confs = getattr(pose_metas[i], "kps_body", []), getattr(pose_metas[i], "kps_body_p", None)
            
            # 1. Sammeln: Was ist im 2D-Bild überhaupt sichtbar?
            visible_parts_keys = []
            frame_ist_px = 0.0
            
            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1); visible_parts_keys.append("head")
            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x, mid_y = (kps[8][0]+kps[11][0])/2, (kps[8][1]+kps[11][1])/2
                frame_ist_px += math.sqrt((kps[1][0]-mid_x)**2 + (kps[1][1]-mid_y)**2); visible_parts_keys.append("torso")
            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9); visible_parts_keys.append("thigh")
            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10); visible_parts_keys.append("calf")

            if frame_ist_px == 0 or not visible_parts_keys: continue

            scale_factor = 1.0

            # --- SÄULE 3: Pure NLF 3D-to-3D Compare ---
            if mode_id == "3":
                # Soll 3D-Summe der SICHTBAREN Teile aus der Calibration
                soll_3d_sum = sum(calib_nlf_3d.get(k, 0) for k in visible_parts_keys)
                if soll_3d_sum <= 0: continue

                # Ist 3D-Summe (Geglättet über Window)
                ist_3d_window_vals = []
                for w in range(max(0, i - nlf_smoothing_window), min(len(pose_metas), i + nlf_smoothing_window + 1)):
                    w_bones = get_nlf_3d_bones_for_frame(raw_poses, w)
                    if w_bones:
                        w_sum = sum(w_bones.get(k, 0) for k in visible_parts_keys)
                        if w_sum > 0: ist_3d_window_vals.append(w_sum)
                
                smoothed_ist_3d = float(np.mean(ist_3d_window_vals)) if ist_3d_window_vals else sum(get_nlf_3d_bones_for_frame(raw_poses, i).get(k, 0) for k in visible_parts_keys)
                if smoothed_ist_3d <= 0.01: smoothed_ist_3d = 0.01

                # Magischer Faktor: Verhältnis der 3D-Knochen zueinander!
                scale_factor = soll_3d_sum / smoothed_ist_3d
                
                log_messages.append(f"\n  Frame {i} | 3D-Compare: Sichtbar={visible_parts_keys}")
                log_messages.append(f"    Soll-3D: {soll_3d_sum:.4f} | Smoothed-Ist-3D: {smoothed_ist_3d:.4f}")
                log_messages.append(f"    Lokaler Faktor: {scale_factor:.3f}x")

            # --- SÄULE 1 & 2: Depth Map (Pinhole) ---
            else:
                frame_soll_m = sum(bone_m.get(k, 0) for k in visible_parts_keys)
                if frame_soll_m == 0: continue
                
                if valid_depth_frames is not None:
                    depth_v_idx, borrowed_from = get_nearest_depth_idx(i, valid_depth_frames)
                else:
                    depth_v_idx = min(i, (depth_np.shape[0] - 1) if depth_np is not None else 0)

                if mode_id == "2" and raw_poses is not None and i < len(raw_poses) and raw_poses[i] is not None:
                    frame_data = raw_poses[i]
                    is_tensor = hasattr(frame_data, 'dim')
                    pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else (np.array(frame_data)[0] if np.array(frame_data).ndim == 3 else np.array(frame_data)))
                    frame_depth = get_nlf_2d_depth(pts, depth_np, depth_v_idx, W, H)
                    log_messages.append(f"\n  Frame {i} | Depth Map via NLF 2D Overlay")
                else:
                    frame_depth = get_skeleton_depth(kps, confs, depth_np, depth_v_idx, W, H)
                    log_messages.append(f"\n  Frame {i} | Depth Map via PoseData 2D Maske")
                
                if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)
                
                expected_px = (frame_soll_m * fx_video) / frame_depth
                scale_factor = expected_px / frame_ist_px

                log_messages.append(f"    Ist-Px: {frame_ist_px:.1f} | Ist-Meter: {frame_soll_m:.3f}m | Tiefe: {frame_depth:.3f}m")
                log_messages.append(f"    Soll-Px: {expected_px:.1f} | Lokaler Faktor: {scale_factor:.3f}x")

            sum_scale_factors += scale_factor
            valid_frames += 1

        if valid_frames == 0: return (pose_data_copy, "Fehler: Keine validen Körperteile gefunden.", video_nlf_data, "{}")

        final_scale = sum_scale_factors / valid_frames
        
        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Gemittelter Skalierungsfaktor ({valid_frames} Frames): {final_scale:.3f}x")
        
        global_pivot_x, global_pivot_y = 0.5, 0.5
        kps_b, c_b = getattr(pose_metas[best_idx], "kps_body", []), getattr(pose_metas[best_idx], "kps_body_p", None)
        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        if val_y and val_x: global_pivot_x, global_pivot_y = np.mean(val_x), max(val_y)

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        config_dict = {
            "anchor_scale": float(final_scale), 
            "scale_x_factor": float(scale_x), 
            "pivot_x": float(global_pivot_x), 
            "pivot_y": float(global_pivot_y)
        }
        config_str = json.dumps(config_dict)

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


class PoseCalibrationManipulator3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "calibration_data": ("POSE_CALIBRATION",),
                "echte_groesse_override": ("FLOAT", {"default": 2.10, "min": 0.1, "max": 5.0, "step": 0.01, "tooltip": "Erzwingt eine neue echte Größe in Metern (für Depth Map/2D)."}),
                "nlf_3d_scale_factor": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 10.0, "step": 0.01, "tooltip": "Multiplikator für die echten 3D-Knochenlängen aus NLF (bone_lengths_nlf_3d)."}),
                "enable_override": ("BOOLEAN", {"default": False, "tooltip": "Wenn False, werden die Originaldaten durchgeleitet."})
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING")
    RETURN_NAMES = ("modified_calibration", "log_output")
    FUNCTION = "manipulate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V3: Manipuliert echte_groesse und bietet einen präzisen Multiplikator für NLF 3D-Knochen."

    def manipulate(self, calibration_data, echte_groesse_override, nlf_3d_scale_factor, enable_override):
        import copy
        
        # Tiefkopie, um das Original nicht zu zerstören
        calib = copy.deepcopy(calibration_data)
        log_messages = ["=== CALIBRATION MANIPULATOR LOG (V3) ==="]

        if not enable_override:
            log_messages.append("Bypass aktiv: Originaldaten werden unverändert weitergeleitet.")
            return (calib, "\n".join(log_messages))

        # --- 1. ECHTE GRÖSSE (DEPTH MAP / 2D) MANIPULATION ---
        # Fallback auf echte_groesse_depthmap, da wir das in V33 umbenannt haben
        alte_groesse = calib.get("echte_groesse_depthmap", calib.get("echte_groesse", 1.0))
        
        if alte_groesse > 0:
            faktor = echte_groesse_override / alte_groesse
            
            log_messages.append(f"Originale 2D/Depth-Größe: {alte_groesse:.3f}m")
            log_messages.append(f"Neue 2D/Depth-Größe: {echte_groesse_override:.3f}m")
            log_messages.append(f"Manipulations-Faktor (2D): {faktor:.4f}x")

            # Hauptgrößen überschreiben
            calib["echte_groesse"] = echte_groesse_override
            if "echte_groesse_depthmap" in calib:
                calib["echte_groesse_depthmap"] = echte_groesse_override

            # Metrische Knochen proportional anpassen
            bone_m = calib.get("bone_lengths_in_meters", {})
            if not bone_m:
                bone_m = calib.get("bone_lengths_in_meters_depthmap", {})
                
            if bone_m:
                log_messages.append("\n--- NEUE METRISCHE KNOCHEN (2D/DepthMap) ---")
                for key, val in bone_m.items():
                    neuer_wert = val * faktor
                    bone_m[key] = neuer_wert
                    log_messages.append(f"{key.capitalize()}: {val:.3f}m -> {neuer_wert:.3f}m")
            
            # Scaler-Knochen proportional anpassen
            bone_s = calib.get("bone_length_for_scaler", {})
            if bone_s:
                log_messages.append("\n--- NEUE SCALER KNOCHEN (2D/DepthMap) ---")
                for key, val in bone_s.items():
                    neuer_wert = val * faktor
                    calib["bone_length_for_scaler"][key] = neuer_wert
                    log_messages.append(f"{key.capitalize()}: {val:.3f} -> {neuer_wert:.3f}")
        else:
            log_messages.append("Warnung: Originale echte_groesse ist <= 0. Überspringe 2D/Depth-Manipulation.")

        # --- 2. NEU: NLF 3D KNOCHEN MANIPULATION ---
        bone_nlf_3d = calib.get("bone_lengths_nlf_3d", {})
        if bone_nlf_3d:
            if nlf_3d_scale_factor != 1.0:
                log_messages.append("\n--- NEUE NLF 3D KNOCHEN (bone_lengths_nlf_3d) ---")
                log_messages.append(f"Wende 3D-Skalierungsfaktor an: {nlf_3d_scale_factor:.2f}x")
                for key, val in bone_nlf_3d.items():
                    neuer_wert = val * nlf_3d_scale_factor
                    calib["bone_lengths_nlf_3d"][key] = neuer_wert
                    log_messages.append(f"{key.capitalize()}: {val:.3f} -> {neuer_wert:.3f}")
            else:
                log_messages.append("\n--- NLF 3D KNOCHEN (bone_lengths_nlf_3d) ---")
                log_messages.append("Faktor ist 1.00x, 3D-Werte bleiben unverändert.")
        else:
            log_messages.append("\nKeine 'bone_lengths_nlf_3d' im Hub gefunden. Bitte V33 nutzen!")

        log_messages.append("\nProportionen (true_3d_bones) bleiben unangetastet.")

        return (calib, "\n".join(log_messages))


class PoseGlobalPerspectiveScalerV57:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_pose_data": ("POSEDATA",),
                "calibration_data": ("POSE_CALIBRATION",),
                "scaling_mode": (
                    [
                        "1. Classic 2D + Depth Map",
                        "2. NLF 2D Overlay + Depth Map",
                        "3. Pure NLF 3D Compare (Legacy)",
                        "4. Robust NLF 3D Ratio"
                    ],
                    {"default": "4. Robust NLF 3D Ratio"}
                ),
                "best_frame_source": (["PoseData (2D)", "NLF (3D SMPL)"], {"default": "NLF (3D SMPL)"}),
                "nlf_smoothing_window": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1, "tooltip": "Für NLF-Modi: Glättet die 3D-Knochenlängen über X Frames."}),
                "include_head": ("BOOLEAN", {"default": True}),
                "anchor_window": ("INT", {"default": 2, "min": 0, "max": 15, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frontal_method": (["3D_NLF", "2D_Ratio"], {"default": "3D_NLF"}),
                "frontal_2d_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0}),
                "scale_2d_axes": (["X and Y (Uniform)", "Only Y (Height)"], {"default": "X and Y (Uniform)"}),

                "nlf_top_frame_count": ("INT", {"default": 12, "min": 1, "max": 80, "step": 1, "tooltip": "Nur Modus 4: Anzahl der besten NLF-Frames, aus denen der finale Scale robust berechnet wird."}),
                "nlf_ratio_aggregation": (["median", "trimmed_mean", "weighted_mean"], {"default": "median", "tooltip": "Nur Modus 4: Wie die Top-Frame-Ratios zu einem einzigen Scale kombiniert werden."}),
                "nlf_outlier_threshold": ("FLOAT", {"default": 0.18, "min": 0.02, "max": 1.00, "step": 0.01, "tooltip": "Nur Modus 4: Relative Ausreißer-Schwelle. 0.18 = Frames über ca. 18 Prozent vom Median werden verworfen."}),
                "nlf_min_bone_count": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "Nur Modus 4: Mindestanzahl gültiger Bone-Ratios pro Frame."}),
                "nlf_scale_basis": (
                    ["visible_parts", "torso_legs_balanced", "full_body", "upper_body_priority", "legs_priority"],
                    {"default": "torso_legs_balanced", "tooltip": "Nur Modus 4: Welche Bone-Gruppen für den einen finalen NLF-Scale bevorzugt werden."}
                ),
            },
            "optional": {
                "video_depth_map": ("IMAGE",),
                "video_nlf_data": ("NLFPRED",),
                "video_intrinsics_json": ("STRING", {"forceInput": True}),
                "valid_depth_indices": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data", "nlf_render_config")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V57: Fügt Robust NLF 3D Ratio als Modus 4 hinzu. Gibt weiterhin nur einen finalen Scale aus."

    def process(
        self,
        video_pose_data,
        calibration_data,
        scaling_mode,
        best_frame_source,
        nlf_smoothing_window,
        include_head,
        anchor_window,
        min_confidence,
        frontal_method,
        frontal_2d_threshold,
        frontal_3d_angle_tolerance,
        scale_2d_axes,
        nlf_top_frame_count,
        nlf_ratio_aggregation,
        nlf_outlier_threshold,
        nlf_min_bone_count,
        nlf_scale_basis,
        video_depth_map=None,
        video_nlf_data=None,
        video_intrinsics_json=None,
        valid_depth_indices=None
    ):
        import copy
        import numpy as np
        import math
        import json
        import torch

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = [f"=== V57 GLOBAL SCALER LOG (MODE: {scaling_mode}) ==="]

        if not pose_metas:
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data, "{}")

        is_inverted = calibration_data.get("is_depth_inverted", False)

        mode_id = scaling_mode[0]
        bone_m = {}
        fx_video = 500.0

        calib_nlf_3d = calibration_data.get("bone_lengths_nlf_3d", {})

        if mode_id in ["1", "2"]:
            bone_m = calibration_data.get("bone_lengths_in_meters_depthmap", {})
            if not bone_m:
                bone_m = calibration_data.get("bone_lengths_in_meters", {})

            fx_video = calibration_data.get("focal_length_fx", 500.0)

            if video_intrinsics_json:
                try:
                    int_data = json.loads(video_intrinsics_json)
                    if "intrinsics" in int_data and len(int_data["intrinsics"]) > 0:
                        matrix = int_data["intrinsics"][0].get("image_0", None)
                        if matrix is not None:
                            fx_video = float(matrix[0][0])
                except Exception:
                    pass

            log_messages.append(f">> Lade Pinhole-Metriken (fx={fx_video:.2f}) für Depth Map.")

        elif mode_id in ["3", "4"]:
            if not calib_nlf_3d:
                return (
                    pose_data_copy,
                    "Fehler: Hub hat keine bone_lengths_nlf_3d! Bitte NLF in die Calibration/Hub Node stecken.",
                    video_nlf_data,
                    "{}"
                )

            if mode_id == "3":
                log_messages.append(f">> Lade pure 3D-Metrik LEGACY: {calib_nlf_3d} | Pinhole und Z-Tiefe werden ignoriert.")
            else:
                log_messages.append(f">> Lade ROBUST NLF 3D Ratio: {calib_nlf_3d} | Pinhole und Z-Tiefe werden ignoriert.")

        valid_depth_frames = None
        if valid_depth_indices:
            try:
                valid_depth_frames = [int(x.strip()) for x in valid_depth_indices.split(",") if x.strip().isdigit()]
            except Exception:
                valid_depth_frames = None

        def get_nearest_depth_idx(target_idx, valid_list):
            if not valid_list:
                return target_idx, target_idx
            nearest_val = min(valid_list, key=lambda x: abs(x - target_idx))
            return valid_list.index(nearest_val), nearest_val

        depth_np = None
        if video_depth_map is not None:
            depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, "cpu") else video_depth_map

        H, W = (depth_np.shape[1], depth_np.shape[2]) if depth_np is not None else (1024, 1024)

        def is_val(kps, confs, idx):
            if kps is None or idx >= len(kps):
                return False
            pt = kps[idx]
            if pt is None or len(pt) < 2:
                return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else 1.0
            return c >= min_confidence

        def dist_2d(kps, i1, i2):
            return math.sqrt((kps[i1][0] - kps[i2][0]) ** 2 + (kps[i1][1] - kps[i2][1]) ** 2)

        def get_skeleton_depth(kps, confs, depth_img, v_idx, W, H):
            if depth_img is None:
                return 0.5

            skeleton_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (1, 5), (5, 6), (6, 7),
                (1, 8), (8, 9), (9, 10),
                (1, 11), (11, 12), (12, 13),
                (8, 11)
            ]

            depth_vals = []

            for p1, p2 in skeleton_connections:
                if is_val(kps, confs, p1) and is_val(kps, confs, p2):
                    x1, y1 = int(kps[p1][0]), int(kps[p1][1])
                    x2, y2 = int(kps[p2][0]), int(kps[p2][1])
                    dist = max(abs(x2 - x1), abs(y2 - y1))

                    if dist > 0:
                        for step in range(dist + 1):
                            t = step / dist
                            px = int(x1 + t * (x2 - x1))
                            py = int(y1 + t * (y2 - y1))

                            if 0 <= px < W and 0 <= py < H:
                                val = depth_img[v_idx, py, px]
                                depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))

            if depth_vals:
                return float(np.mean(depth_vals))

            return 0.5

        def get_nlf_2d_depth(pts, depth_img, v_idx, W, H):
            if depth_img is None:
                return 0.5

            depth_vals = []

            for idx in range(len(pts)):
                if np.linalg.norm(pts[idx]) > 1e-5:
                    px, py = int(pts[idx][0]), int(pts[idx][1])

                    if 0 <= px < W and 0 <= py < H:
                        val = depth_img[v_idx, py, px]
                        depth_vals.append(float(val[0] if isinstance(val, (np.ndarray, list)) else val))

            if depth_vals:
                return float(np.mean(depth_vals))

            return 0.5

        is_dict = isinstance(video_nlf_data, dict)
        raw_poses = video_nlf_data.get("joints3d_nonparam", [video_nlf_data])[0] if is_dict and video_nlf_data else video_nlf_data

        def extract_nlf_points(raw_poses_local, f_idx):
            if raw_poses_local is None:
                return None
            if f_idx < 0 or f_idx >= len(raw_poses_local):
                return None
            if raw_poses_local[f_idx] is None or len(raw_poses_local[f_idx]) == 0:
                return None

            frame_data = raw_poses_local[f_idx]
            is_tensor = hasattr(frame_data, "dim")

            if is_tensor and frame_data.dim() == 3:
                pts = frame_data[0].cpu().numpy()
            elif is_tensor:
                pts = frame_data.cpu().numpy()
            else:
                arr = np.array(frame_data)
                pts = arr[0] if arr.ndim == 3 else arr

            if pts is None or len(pts) == 0:
                return None

            return pts

        def get_nlf_3d_bones_for_frame(raw_poses_local, f_idx):
            pts = extract_nlf_points(raw_poses_local, f_idx)
            if pts is None or len(pts) < 16:
                return {}

            def valid(idx):
                return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            def dist3d(idx1, idx2):
                if not valid(idx1) or not valid(idx2):
                    return 0.0
                return float(np.linalg.norm(pts[idx1] - pts[idx2]))

            head = dist3d(12, 15)
            torso = dist3d(0, 12)

            l_thigh = dist3d(1, 4)
            r_thigh = dist3d(2, 5)
            l_calf = dist3d(4, 7)
            r_calf = dist3d(5, 8)

            thigh_vals = [v for v in [l_thigh, r_thigh] if v > 1e-5]
            calf_vals = [v for v in [l_calf, r_calf] if v > 1e-5]

            thigh = float(np.mean(thigh_vals)) if thigh_vals else 0.0
            calf = float(np.mean(calf_vals)) if calf_vals else 0.0

            shoulder_width = dist3d(16, 17)
            hip_width = dist3d(1, 2)

            return {
                "head": head,
                "torso": torso,
                "l_thigh": l_thigh,
                "r_thigh": r_thigh,
                "thigh": thigh,
                "l_calf": l_calf,
                "r_calf": r_calf,
                "calf": calf,
                "shoulder_width": shoulder_width,
                "hip_width": hip_width
            }

        def get_smoothed_nlf_3d_bones(raw_poses_local, f_idx, window):
            collected = []

            start = max(0, f_idx - window)
            end = min(len(raw_poses_local) - 1, f_idx + window) if raw_poses_local is not None else -1

            for w in range(start, end + 1):
                bones = get_nlf_3d_bones_for_frame(raw_poses_local, w)
                if bones:
                    collected.append(bones)

            if not collected:
                return get_nlf_3d_bones_for_frame(raw_poses_local, f_idx)

            keys = [
                "head", "torso",
                "l_thigh", "r_thigh", "thigh",
                "l_calf", "r_calf", "calf",
                "shoulder_width", "hip_width"
            ]

            out = {}

            for key in keys:
                vals = [float(b.get(key, 0.0)) for b in collected if float(b.get(key, 0.0)) > 1e-5]
                out[key] = float(np.median(vals)) if vals else 0.0

            return out

        def get_calib_target_value(key):
            if key in calib_nlf_3d and float(calib_nlf_3d.get(key, 0.0)) > 1e-5:
                return float(calib_nlf_3d.get(key, 0.0))

            if key == "l_thigh" or key == "r_thigh":
                if "thigh" in calib_nlf_3d:
                    return float(calib_nlf_3d.get("thigh", 0.0))

            if key == "l_calf" or key == "r_calf":
                if "calf" in calib_nlf_3d:
                    return float(calib_nlf_3d.get("calf", 0.0))

            if key == "thigh":
                vals = [
                    float(calib_nlf_3d.get("l_thigh", 0.0)),
                    float(calib_nlf_3d.get("r_thigh", 0.0))
                ]
                vals = [v for v in vals if v > 1e-5]
                if vals:
                    return float(np.mean(vals))

            if key == "calf":
                vals = [
                    float(calib_nlf_3d.get("l_calf", 0.0)),
                    float(calib_nlf_3d.get("r_calf", 0.0))
                ]
                vals = [v for v in vals if v > 1e-5]
                if vals:
                    return float(np.mean(vals))

            return 0.0

        def get_visible_parts_for_frame(frame_idx):
            if frame_idx < 0 or frame_idx >= len(pose_metas):
                return [], 0.0

            kps = getattr(pose_metas[frame_idx], "kps_body", [])
            confs = getattr(pose_metas[frame_idx], "kps_body_p", None)

            visible_parts_keys = []
            frame_ist_px = 0.0

            if include_head and is_val(kps, confs, 0) and is_val(kps, confs, 1):
                frame_ist_px += dist_2d(kps, 0, 1)
                visible_parts_keys.append("head")

            if is_val(kps, confs, 1) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                mid_x = (kps[8][0] + kps[11][0]) / 2.0
                mid_y = (kps[8][1] + kps[11][1]) / 2.0
                frame_ist_px += math.sqrt((kps[1][0] - mid_x) ** 2 + (kps[1][1] - mid_y) ** 2)
                visible_parts_keys.append("torso")

            if is_val(kps, confs, 8) and is_val(kps, confs, 9):
                frame_ist_px += dist_2d(kps, 8, 9)
                visible_parts_keys.append("thigh")

            if is_val(kps, confs, 9) and is_val(kps, confs, 10):
                frame_ist_px += dist_2d(kps, 9, 10)
                visible_parts_keys.append("calf")

            return visible_parts_keys, frame_ist_px

        def get_robust_basis_keys(visible_parts_keys):
            if nlf_scale_basis == "visible_parts":
                keys = list(visible_parts_keys)
                if not keys:
                    keys = ["torso", "thigh", "calf"]
                return keys

            if nlf_scale_basis == "full_body":
                return ["head", "torso", "thigh", "calf"]

            if nlf_scale_basis == "upper_body_priority":
                return ["head", "torso"]

            if nlf_scale_basis == "legs_priority":
                return ["thigh", "calf"]

            return ["torso", "thigh", "calf"]

        def get_bone_weight(key):
            if nlf_scale_basis == "upper_body_priority":
                weights = {
                    "head": 0.65,
                    "torso": 1.50,
                    "thigh": 0.50,
                    "calf": 0.50,
                    "shoulder_width": 0.40,
                    "hip_width": 0.30
                }
            elif nlf_scale_basis == "legs_priority":
                weights = {
                    "head": 0.25,
                    "torso": 0.80,
                    "thigh": 1.35,
                    "calf": 1.35,
                    "shoulder_width": 0.25,
                    "hip_width": 0.40
                }
            elif nlf_scale_basis == "full_body":
                weights = {
                    "head": 0.55,
                    "torso": 1.20,
                    "thigh": 1.00,
                    "calf": 1.00,
                    "shoulder_width": 0.35,
                    "hip_width": 0.35
                }
            else:
                weights = {
                    "head": 0.35,
                    "torso": 1.25,
                    "thigh": 1.00,
                    "calf": 1.00,
                    "shoulder_width": 0.25,
                    "hip_width": 0.25
                }

            return float(weights.get(key, 1.0))

        def weighted_median(values, weights):
            if not values:
                return 1.0

            pairs = sorted(zip(values, weights), key=lambda x: x[0])
            total_weight = sum(w for _, w in pairs)

            if total_weight <= 1e-8:
                return float(np.median(values))

            acc = 0.0
            half = total_weight * 0.5

            for value, weight in pairs:
                acc += weight
                if acc >= half:
                    return float(value)

            return float(pairs[-1][0])

        def trimmed_mean(values, trim_fraction=0.20):
            if not values:
                return 1.0

            vals = sorted([float(v) for v in values])
            if len(vals) <= 2:
                return float(np.mean(vals))

            trim_count = int(len(vals) * trim_fraction)

            if trim_count <= 0:
                return float(np.mean(vals))

            trimmed = vals[trim_count:-trim_count]
            if not trimmed:
                trimmed = vals

            return float(np.mean(trimmed))

        def compute_frame_robust_nlf_ratio(frame_idx, max_body_length):
            visible_parts_keys, frame_ist_px = get_visible_parts_for_frame(frame_idx)
            source_bones = get_smoothed_nlf_3d_bones(raw_poses, frame_idx, nlf_smoothing_window)

            if not source_bones:
                return None

            basis_keys = get_robust_basis_keys(visible_parts_keys)

            ratios = {}
            ratio_values = []
            ratio_weights = []
            used_keys = []
            rejected_keys = []

            for key in basis_keys:
                source_value = float(source_bones.get(key, 0.0))
                target_value = get_calib_target_value(key)

                if source_value <= 1e-5 or target_value <= 1e-5:
                    rejected_keys.append(f"{key}:missing")
                    continue

                ratio = target_value / source_value

                if ratio <= 0.05 or ratio >= 10.0:
                    rejected_keys.append(f"{key}:hard_outlier_{ratio:.3f}")
                    continue

                ratios[key] = float(ratio)
                ratio_values.append(float(ratio))
                ratio_weights.append(get_bone_weight(key))
                used_keys.append(key)

            if len(ratio_values) < nlf_min_bone_count:
                return None

            frame_ratio = weighted_median(ratio_values, ratio_weights)

            ratio_mean = float(np.mean(ratio_values)) if ratio_values else frame_ratio
            ratio_std = float(np.std(ratio_values)) if len(ratio_values) > 1 else 0.0
            ratio_consistency = 1.0 / (1.0 + (ratio_std / max(abs(ratio_mean), 1e-5)))

            radar = all_frames_data[frame_idx] if frame_idx < len(all_frames_data) else {
                "has_feet": False,
                "has_ankles": False,
                "has_knees": False,
                "is_frontal": False,
                "length": 0.0,
                "frontal_pts": 0.0
            }

            length_score = float(np.clip(radar.get("length", 0.0) / max(max_body_length, 1e-5), 0.0, 1.0))
            frontal_score = 1.0 if radar.get("is_frontal", False) else 0.0
            feet_score = 1.0 if (radar.get("has_feet", False) or radar.get("has_ankles", False)) else (0.50 if radar.get("has_knees", False) else 0.0)
            bone_count_score = float(np.clip(len(ratio_values) / max(len(basis_keys), 1), 0.0, 1.0))

            score = (
                length_score * 0.25
                + frontal_score * 0.25
                + feet_score * 0.20
                + bone_count_score * 0.15
                + ratio_consistency * 0.15
            )

            if "torso" in used_keys:
                score += 0.05

            if ("thigh" in used_keys or "calf" in used_keys) and (radar.get("has_feet", False) or radar.get("has_ankles", False) or radar.get("has_knees", False)):
                score += 0.05

            score = float(np.clip(score, 0.0, 1.0))

            return {
                "frame": int(frame_idx),
                "ratio": float(frame_ratio),
                "score": score,
                "visible_parts": visible_parts_keys,
                "used_keys": used_keys,
                "rejected_keys": rejected_keys,
                "ratios": ratios,
                "ratio_std": ratio_std,
                "ratio_consistency": ratio_consistency,
                "length_score": length_score,
                "frontal_score": frontal_score,
                "feet_score": feet_score,
                "bone_count_score": bone_count_score,
                "frame_ist_px": frame_ist_px
            }

        def aggregate_robust_records(records):
            if not records:
                return 1.0, [], [], 0.0

            sorted_records = sorted(records, key=lambda r: r["score"], reverse=True)

            pre_count = min(len(sorted_records), max(nlf_top_frame_count * 3, nlf_top_frame_count))
            preselected = sorted_records[:pre_count]

            base_median = float(np.median([r["ratio"] for r in preselected]))

            accepted = []
            rejected = []

            for record in preselected:
                rel_err = abs(record["ratio"] - base_median) / max(abs(base_median), 1e-5)

                if rel_err <= nlf_outlier_threshold:
                    accepted.append(record)
                else:
                    record_copy = dict(record)
                    record_copy["reject_reason"] = f"ratio_outlier_{rel_err * 100.0:.2f}%"
                    rejected.append(record_copy)

            if not accepted:
                accepted = preselected[:max(1, nlf_top_frame_count)]
                rejected = []

            accepted = sorted(accepted, key=lambda r: r["score"], reverse=True)[:nlf_top_frame_count]

            ratios = [float(r["ratio"]) for r in accepted]
            weights = [float(r["score"]) for r in accepted]

            if nlf_ratio_aggregation == "weighted_mean":
                weight_sum = sum(weights)
                if weight_sum > 1e-8:
                    final = float(sum(r * w for r, w in zip(ratios, weights)) / weight_sum)
                else:
                    final = float(np.mean(ratios))
            elif nlf_ratio_aggregation == "trimmed_mean":
                final = trimmed_mean(ratios, trim_fraction=0.20)
            else:
                final = float(np.median(ratios))

            avg_score = float(np.mean(weights)) if weights else 0.0
            ratio_std = float(np.std(ratios)) if len(ratios) > 1 else 0.0
            ratio_mean = float(np.mean(ratios)) if ratios else final
            consistency = 1.0 / (1.0 + (ratio_std / max(abs(ratio_mean), 1e-5)))
            count_factor = float(np.clip(len(accepted) / max(nlf_top_frame_count, 1), 0.0, 1.0))
            confidence = float(np.clip((avg_score * 0.55) + (consistency * 0.30) + (count_factor * 0.15), 0.0, 1.0))

            return final, accepted, rejected, confidence

        # --- RADAR ---
        all_frames_data = []
        frontal_indices = []

        if best_frame_source == "NLF (3D SMPL)":
            for i in range(len(pose_metas)):
                pts = extract_nlf_points(raw_poses, i)

                if pts is None:
                    all_frames_data.append({
                        "has_feet": False,
                        "has_ankles": False,
                        "has_knees": False,
                        "is_frontal": False,
                        "length": 0.0,
                        "frontal_pts": 0.0
                    })
                    continue

                def is_val_nlf(idx):
                    return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

                has_knees = is_val_nlf(4) or is_val_nlf(5)
                has_ankles = is_val_nlf(7) or is_val_nlf(8)
                has_feet = has_ankles

                valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_nlf(idx)]
                length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

                is_frontal = False
                frontal_pts = 0.0
                max_angle = 90.0

                if len(pts) >= 18 and is_val_nlf(1) and is_val_nlf(2) and is_val_nlf(16) and is_val_nlf(17):
                    dx_h = pts[2][0] - pts[1][0]
                    dz_h = pts[2][2] - pts[1][2]
                    dx_s = pts[17][0] - pts[16][0]
                    dz_s = pts[17][2] - pts[16][2]

                    angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                    angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                    max_angle = max(angle_h, angle_s)

                    if max_angle <= frontal_3d_angle_tolerance:
                        is_frontal = True
                        frontal_pts = max(0.0, (frontal_3d_angle_tolerance - max_angle) * 10.0)

                all_frames_data.append({
                    "has_feet": has_feet,
                    "has_ankles": has_ankles,
                    "has_knees": has_knees,
                    "is_frontal": is_frontal,
                    "length": length,
                    "frontal_pts": frontal_pts
                })

                if is_frontal:
                    frontal_indices.append(i)

        else:
            pose_input_3d = raw_poses

            for i, meta in enumerate(pose_metas):
                kps = getattr(meta, "kps_body", [])
                confs = getattr(meta, "kps_body_p", None)

                valid_y = [kps[idx][1] for idx in range(len(kps)) if is_val(kps, confs, idx)]

                if not valid_y:
                    all_frames_data.append({
                        "has_feet": False,
                        "has_ankles": False,
                        "has_knees": False,
                        "is_frontal": False,
                        "length": 0.0,
                        "frontal_pts": 0.0
                    })
                    continue

                has_ankles = is_val(kps, confs, 10) or is_val(kps, confs, 13)
                has_knees = is_val(kps, confs, 9) or is_val(kps, confs, 12)
                has_feet = any(is_val(kps, confs, x) for x in [18, 19, 20, 21, 22, 23, 24])

                top_y = min(valid_y)
                bottom_y = max(valid_y)

                if not include_head and is_val(kps, confs, 1):
                    top_y = kps[1][1]

                length = (bottom_y - top_y) if top_y is not None and bottom_y is not None else 0.0

                is_frontal = False
                frontal_pts = 0.0

                if frontal_method == "3D_NLF" and pose_input_3d is not None and i < len(pose_input_3d):
                    pts_3d = extract_nlf_points(pose_input_3d, i)

                    if pts_3d is not None and len(pts_3d) > 0:
                        num_joints = len(pts_3d)
                        idx_r, idx_l = 2, 5

                        if num_joints == 17:
                            idx_r, idx_l = 11, 14
                        elif num_joints in [24, 45, 68]:
                            idx_r, idx_l = 16, 17

                        if num_joints > max(idx_r, idx_l):
                            dx = float(pts_3d[idx_r][0]) - float(pts_3d[idx_l][0])
                            dz = float(pts_3d[idx_r][2]) - float(pts_3d[idx_l][2])
                            angle = math.degrees(math.atan2(abs(dz), abs(dx)))

                            if angle <= frontal_3d_angle_tolerance:
                                is_frontal = True
                                frontal_pts = max(0.0, (frontal_3d_angle_tolerance - angle) * 10.0)

                elif frontal_method == "2D_Ratio":
                    try:
                        if is_val(kps, confs, 2) and is_val(kps, confs, 5) and is_val(kps, confs, 8) and is_val(kps, confs, 11):
                            shoulder_w = abs(kps[2][0] - kps[5][0])
                            hip_w = abs(kps[8][0] - kps[11][0])
                            if shoulder_w > 1e-5:
                                ratio_2d = hip_w / shoulder_w
                                if ratio_2d >= frontal_2d_threshold:
                                    is_frontal = True
                                    frontal_pts = max(0.0, ratio_2d * 100.0)
                    except Exception:
                        pass

                all_frames_data.append({
                    "has_feet": has_feet,
                    "has_ankles": has_ankles,
                    "has_knees": has_knees,
                    "is_frontal": is_frontal,
                    "length": length,
                    "frontal_pts": frontal_pts
                })

                if is_frontal:
                    frontal_indices.append(i)

        candidates = frontal_indices if len(frontal_indices) > 0 else list(range(len(pose_metas)))
        max_body_length = max([all_frames_data[idx]["length"] for idx in candidates]) if candidates else 1.0

        if max_body_length == 0:
            max_body_length = 1.0

        best_idx = candidates[0]
        best_score = -1.0

        for idx in candidates:
            data = all_frames_data[idx]

            bein_pts = max(
                1000.0 if (data["has_feet"] or data["has_ankles"]) else 0.0,
                500.0 if not (data["has_feet"] or data["has_ankles"]) and data["has_knees"] else 0.0
            )

            total_score = (
                bein_pts
                + (500.0 if data["has_feet"] and data["is_frontal"] else 0.0)
                + data["frontal_pts"]
                + ((data["length"] / max_body_length) * 100.0)
            )

            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        log_messages.append(f"\n-> Gewinner Frame (Anchor): {best_idx}")

        start_idx = max(0, best_idx - anchor_window)
        end_idx = min(len(pose_metas) - 1, best_idx + anchor_window)

        sum_scale_factors = 0.0
        valid_frames = 0
        final_scale = 1.0

        robust_used_records = []
        robust_rejected_records = []
        robust_confidence = 0.0

        # --- MODUS 4: ROBUST NLF 3D RATIO ---
        if mode_id == "4":
            log_messages.append("\n--- ROBUST NLF 3D RATIO ENGINE ---")
            log_messages.append(
                f"TopFrames={nlf_top_frame_count} | Aggregation={nlf_ratio_aggregation} | "
                f"OutlierThreshold={nlf_outlier_threshold * 100.0:.1f}% | MinBones={nlf_min_bone_count} | Basis={nlf_scale_basis}"
            )

            robust_records = []

            for i in range(len(pose_metas)):
                record = compute_frame_robust_nlf_ratio(i, max_body_length)
                if record is not None:
                    robust_records.append(record)

            if not robust_records:
                return (
                    pose_data_copy,
                    "Fehler: Robust NLF 3D Ratio konnte keine validen Frame-Ratios berechnen.",
                    video_nlf_data,
                    "{}"
                )

            final_scale, robust_used_records, robust_rejected_records, robust_confidence = aggregate_robust_records(robust_records)
            valid_frames = len(robust_used_records)

            log_messages.append(f"Valide Robust-Frame-Kandidaten: {len(robust_records)}")
            log_messages.append(f"Verwendete Top-Frames nach Outlier-Filter: {len(robust_used_records)}")
            log_messages.append(f"Verworfene Outlier im Preselect: {len(robust_rejected_records)}")
            log_messages.append(f"Robust Final Scale: {final_scale:.6f}x | Confidence: {robust_confidence:.3f}")

            log_messages.append("\nTop verwendete Robust-NLF-Frames:")
            for rec in robust_used_records[:30]:
                ratio_text = ", ".join([f"{k}:{v:.3f}" for k, v in rec["ratios"].items()])
                log_messages.append(
                    f"  Frame {str(rec['frame']).rjust(4)} | Scale {rec['ratio']:.6f}x | Score {rec['score']:.3f} | "
                    f"Bones={rec['used_keys']} | Visible={rec['visible_parts']} | Ratios[{ratio_text}] | "
                    f"Std={rec['ratio_std']:.4f} | Consistency={rec['ratio_consistency']:.3f}"
                )

            if len(robust_used_records) > 30:
                log_messages.append(f"  ... weitere {len(robust_used_records) - 30} verwendete Frames ausgelassen.")

            if robust_rejected_records:
                log_messages.append("\nVerworfene Robust-NLF-Outlier:")
                for rec in robust_rejected_records[:20]:
                    ratio_text = ", ".join([f"{k}:{v:.3f}" for k, v in rec["ratios"].items()])
                    log_messages.append(
                        f"  Frame {str(rec['frame']).rjust(4)} | Scale {rec['ratio']:.6f}x | Score {rec['score']:.3f} | "
                        f"Reason={rec.get('reject_reason', 'unknown')} | Ratios[{ratio_text}]"
                    )

                if len(robust_rejected_records) > 20:
                    log_messages.append(f"  ... weitere {len(robust_rejected_records) - 20} verworfene Frames ausgelassen.")

        # --- MODUS 1, 2, 3: LEGACY-BERECHNUNG ---
        else:
            for i in range(start_idx, end_idx + 1):
                kps = getattr(pose_metas[i], "kps_body", [])
                confs = getattr(pose_metas[i], "kps_body_p", None)

                visible_parts_keys, frame_ist_px = get_visible_parts_for_frame(i)

                if frame_ist_px == 0 or not visible_parts_keys:
                    continue

                scale_factor = 1.0

                if mode_id == "3":
                    soll_3d_sum = sum(calib_nlf_3d.get(k, 0) for k in visible_parts_keys)

                    if soll_3d_sum <= 0:
                        continue

                    ist_3d_window_vals = []

                    for w in range(max(0, i - nlf_smoothing_window), min(len(pose_metas), i + nlf_smoothing_window + 1)):
                        w_bones = get_nlf_3d_bones_for_frame(raw_poses, w)
                        if w_bones:
                            w_sum = sum(w_bones.get(k, 0) for k in visible_parts_keys)
                            if w_sum > 0:
                                ist_3d_window_vals.append(w_sum)

                    if ist_3d_window_vals:
                        smoothed_ist_3d = float(np.mean(ist_3d_window_vals))
                    else:
                        current_bones = get_nlf_3d_bones_for_frame(raw_poses, i)
                        smoothed_ist_3d = sum(current_bones.get(k, 0) for k in visible_parts_keys)

                    if smoothed_ist_3d <= 0.01:
                        smoothed_ist_3d = 0.01

                    scale_factor = soll_3d_sum / smoothed_ist_3d

                    log_messages.append(f"\n  Frame {i} | 3D-Compare LEGACY: Sichtbar={visible_parts_keys}")
                    log_messages.append(f"    Soll-3D: {soll_3d_sum:.4f} | Smoothed-Ist-3D: {smoothed_ist_3d:.4f}")
                    log_messages.append(f"    Lokaler Faktor: {scale_factor:.3f}x")

                else:
                    frame_soll_m = sum(bone_m.get(k, 0) for k in visible_parts_keys)

                    if frame_soll_m == 0:
                        continue

                    if valid_depth_frames is not None:
                        depth_v_idx, borrowed_from = get_nearest_depth_idx(i, valid_depth_frames)
                    else:
                        depth_v_idx = min(i, (depth_np.shape[0] - 1) if depth_np is not None else 0)

                    if mode_id == "2" and raw_poses is not None and i < len(raw_poses) and raw_poses[i] is not None:
                        pts = extract_nlf_points(raw_poses, i)
                        if pts is not None:
                            frame_depth = get_nlf_2d_depth(pts, depth_np, depth_v_idx, W, H)
                        else:
                            frame_depth = get_skeleton_depth(kps, confs, depth_np, depth_v_idx, W, H)

                        log_messages.append(f"\n  Frame {i} | Depth Map via NLF 2D Overlay")
                    else:
                        frame_depth = get_skeleton_depth(kps, confs, depth_np, depth_v_idx, W, H)
                        log_messages.append(f"\n  Frame {i} | Depth Map via PoseData 2D Maske")

                    if is_inverted:
                        frame_depth = 1.0 / max(frame_depth, 0.0001)

                    expected_px = (frame_soll_m * fx_video) / frame_depth
                    scale_factor = expected_px / frame_ist_px

                    log_messages.append(f"    Ist-Px: {frame_ist_px:.1f} | Ist-Meter: {frame_soll_m:.3f}m | Tiefe: {frame_depth:.3f}m")
                    log_messages.append(f"    Soll-Px: {expected_px:.1f} | Lokaler Faktor: {scale_factor:.3f}x")

                sum_scale_factors += scale_factor
                valid_frames += 1

            if valid_frames == 0:
                return (pose_data_copy, "Fehler: Keine validen Körperteile gefunden.", video_nlf_data, "{}")

            final_scale = sum_scale_factors / valid_frames

        log_messages.append(f"\n=== FINALES ERGEBNIS ===")
        log_messages.append(f"Finaler Skalierungsfaktor ({valid_frames} Frames): {final_scale:.6f}x")

        global_pivot_x = 0.5
        global_pivot_y = 0.5

        kps_b = getattr(pose_metas[best_idx], "kps_body", [])
        c_b = getattr(pose_metas[best_idx], "kps_body_p", None)

        val_y = [kps_b[idx][1] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]
        val_x = [kps_b[idx][0] for idx in range(len(kps_b)) if is_val(kps_b, c_b, idx)]

        if val_y and val_x:
            global_pivot_x = float(np.mean(val_x))
            global_pivot_y = float(max(val_y))

        scale_x = final_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0

        for meta in pose_metas:
            for attr in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr, None)

                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = global_pivot_x + (arr[j][0] - global_pivot_x) * scale_x
                            arr[j][1] = global_pivot_y + (arr[j][1] - global_pivot_y) * final_scale

        config_dict = {
            "anchor_scale": float(final_scale),
            "scale_x_factor": float(scale_x),
            "pivot_x": float(global_pivot_x),
            "pivot_y": float(global_pivot_y)
        }

        if mode_id == "4":
            config_dict["nlf_scale_mode"] = "robust_nlf_3d_ratio"
            config_dict["nlf_anchor_frame"] = int(best_idx)
            config_dict["nlf_used_frames"] = [int(r["frame"]) for r in robust_used_records]
            config_dict["nlf_rejected_frames"] = [int(r["frame"]) for r in robust_rejected_records]
            config_dict["nlf_scale_confidence"] = float(robust_confidence)
            config_dict["nlf_final_aggregation"] = str(nlf_ratio_aggregation)
            config_dict["nlf_scale_basis"] = str(nlf_scale_basis)
            config_dict["nlf_top_frame_count"] = int(nlf_top_frame_count)
            config_dict["nlf_outlier_threshold"] = float(nlf_outlier_threshold)
        elif mode_id == "3":
            config_dict["nlf_scale_mode"] = "pure_nlf_3d_compare_legacy"
            config_dict["nlf_anchor_frame"] = int(best_idx)
        else:
            config_dict["nlf_scale_mode"] = "depth_pinhole"
            config_dict["nlf_anchor_frame"] = int(best_idx)

        config_str = json.dumps(config_dict)

        return (pose_data_copy, "\n".join(log_messages), video_nlf_data, config_str)


