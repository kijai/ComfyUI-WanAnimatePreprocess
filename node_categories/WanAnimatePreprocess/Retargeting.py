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



class NLFProportionalRetargeterV5:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"

    def process(self, video_nlf_data, calibration_data, frontal_3d_angle_tolerance):
        import copy
        import numpy as np
        import math
        import torch

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V5 (MIT DETAILLIERTEM LOGGING) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages))

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        if is_normalized:
            log_messages.append("Modus: NORMALISIERT V22 (Prozentual basierend auf Torso=100%).\n")
        else:
            log_messages.append("Modus: ABSOLUT V20 (Absolute Werte).\n")

        # --- SMPL 3D KINEMATISCHER BAUM ---
        tree = {
            0: [1, 2, 3],
            1: [4], 4: [7], 7: [10],           # Linkes Bein
            2: [5], 5: [8], 8: [11],           # Rechtes Bein
            3: [6], 6: [9], 9: [12, 13, 14],   # Wirbelsäule
            12: [15],                          # Kopf
            13: [16], 16: [18], 18: [20], 20: [22], # Linker Arm
            14: [17], 17: [19], 19: [21], 21: [23]  # Rechter Arm
        }

        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child)
                    desc.extend(get_all_descendants(child, tree_map))
            return desc

        frames_processed = 0
        detailed_log_done = False # Damit wir das Log nicht mit 100 Frames überfluten

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0:
                continue
                
            is_tensor = isinstance(frame_data, torch.Tensor)
            if is_tensor:
                pts = frame_data[0].cpu().numpy().copy() if frame_data.dim() == 3 else frame_data.cpu().numpy().copy()
            else:
                arr = np.array(frame_data)
                pts = arr[0].copy() if arr.ndim == 3 else arr.copy()
            
            pts_new = pts.copy()

            # 1. Torso-Länge messen
            vec_torso = pts[12] - pts[0]
            current_torso_length = np.linalg.norm(vec_torso)
            
            if current_torso_length < 1e-5:
                continue

            if not detailed_log_done:
                log_messages.append(f"--- DETAILLIERTER BERICHT FÜR FRAME {frame_idx} ---")
                log_messages.append(f"Gemessene Torso-Länge (Idx 12 -> 0): {current_torso_length:.4f} Einheiten\n")

            # 2. Ziel-Längen berechnen
            targets = {}
            for k, v in true_3d_bones.items():
                targets[k] = (v / 100.0) * current_torso_length if is_normalized else v

            operations = [
                ('shoulder_width', 12, 17), # Hals -> R Schulter
                ('shoulder_width', 12, 16), # Hals -> L Schulter
                ('hip_width', 0, 2),        # Becken -> R Hüfte
                ('hip_width', 0, 1),        # Becken -> L Hüfte
                ('r_arm', 17, 19),          # R Schulter -> R Ellbogen
                ('r_forearm', 19, 21),      # R Ellbogen -> R Handgelenk
                ('l_arm', 16, 18),          # L Schulter -> L Ellbogen
                ('l_forearm', 18, 20),      # L Ellbogen -> L Handgelenk
                ('r_thigh', 2, 5),          # R Hüfte -> R Knie
                ('r_calf', 5, 8),           # R Knie -> R Knöchel
                ('l_thigh', 1, 4),          # L Hüfte -> L Knie
                ('l_calf', 4, 7)            # L Knie -> L Knöchel
            ]

            for bone_key, p_idx, c_idx in operations:
                if bone_key not in targets: continue
                if c_idx >= len(pts_new) or p_idx >= len(pts_new): continue
                
                target_len = targets[bone_key]
                if bone_key in ['shoulder_width', 'hip_width']:
                    target_len = target_len / 2.0
                
                p_pos = pts_new[p_idx]
                c_pos = pts_new[c_idx]
                
                if np.linalg.norm(p_pos) < 1e-5 or np.linalg.norm(c_pos) < 1e-5:
                    continue

                vec = c_pos - p_pos
                curr_len = np.linalg.norm(vec)
                
                if curr_len < 1e-5: continue
                
                # --- LOGGING FÜR DIESEN KNOCHEN ---
                if not detailed_log_done:
                    scale_factor = target_len / curr_len
                    log_messages.append(f"Knochen: {bone_key} (Idx {p_idx} -> {c_idx})")
                    log_messages.append(f"  Länge aktuell: {curr_len:.4f}")
                    log_messages.append(f"  Länge Ziel:    {target_len:.4f}")
                    log_messages.append(f"  Faktor:        {scale_factor:.4f}x")
                    log_messages.append("") # Leerzeile für Lesbarkeit
                
                dir_vec = vec / curr_len
                new_c_pos = p_pos + (dir_vec * target_len)
                
                delta = new_c_pos - c_pos
                
                pts_new[c_idx] += delta
                
                descendants = get_all_descendants(c_idx, tree)
                for d in descendants:
                    if d < len(pts_new):
                        if np.linalg.norm(pts_new[d]) > 1e-5:
                            pts_new[d] += delta

            detailed_log_done = True # Nach dem ersten Frame aufhören so detailliert zu loggen

            if is_tensor:
                if frame_data.dim() == 3:
                    raw_poses[frame_idx][0] = torch.from_numpy(pts_new).to(frame_data.device)
                else:
                    raw_poses[frame_idx] = torch.from_numpy(pts_new).to(frame_data.device)
            else:
                arr_new = np.array(frame_data)
                if arr_new.ndim == 3:
                    arr_new[0] = pts_new
                    raw_poses[frame_idx] = arr_new.tolist()
                else:
                    raw_poses[frame_idx] = pts_new.tolist()
                
            frames_processed += 1

        log_messages.append(f"--- ZUSAMMENFASSUNG ---")
        log_messages.append(f"Erfolgreich skaliert: {frames_processed} Frames.")
        
        return (nlf_data_retargeted, "\n".join(log_messages))


class NLFConfigScaler3DBones:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die originalen Konfigurationsdaten (aus V22 JSON)"}),
                "torso_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "1.0 = keine Änderung"}),
                "shoulder_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "hip_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "arm_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Oberarme (Links & Rechts gekoppelt)"}),
                "forearm_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Unterarme (Links & Rechts gekoppelt)"}),
                "thigh_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Oberschenkel (Links & Rechts gekoppelt)"}),
                "calf_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Waden/Unterschenkel (Links & Rechts gekoppelt)"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION",)
    RETURN_NAMES = ("scaled_calibration_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"

    def process(self, calibration_data, torso_scale, shoulder_scale, hip_scale, arm_scale, forearm_scale, thigh_scale, calf_scale):
        import copy
        import math
        
        # Tiefe Kopie, damit wir die originalen Daten nicht überschreiben
        data = copy.deepcopy(calibration_data)
        bones = data.get("true_3d_bones", {})
        
        if not bones:
            return (data,)
            
        # Prüfen, ob wir mit den 100% Prozentdaten (V22) arbeiten
        is_normalized = math.isclose(bones.get("torso", 0.0), 100.0, abs_tol=1e-3)

        # 1. Gekoppeltes Skalieren der einzelnen "Pakete"
        if "shoulder_width" in bones: bones["shoulder_width"] *= shoulder_scale
        if "hip_width" in bones: bones["hip_width"] *= hip_scale
        
        if "r_arm" in bones: bones["r_arm"] *= arm_scale
        if "l_arm" in bones: bones["l_arm"] *= arm_scale
        
        if "r_forearm" in bones: bones["r_forearm"] *= forearm_scale
        if "l_forearm" in bones: bones["l_forearm"] *= forearm_scale
        
        if "r_thigh" in bones: bones["r_thigh"] *= thigh_scale
        if "l_thigh" in bones: bones["l_thigh"] *= thigh_scale
        
        if "r_calf" in bones: bones["r_calf"] *= calf_scale
        if "l_calf" in bones: bones["l_calf"] *= calf_scale

        # 2. Smart Torso Scale
        # Wenn der Torso der 100% Anker ist, skalieren wir alle anderen Knochen invers, 
        # anstatt den Torso-Wert zu verändern. Das hält die Mathematik sauber!
        if torso_scale != 1.0:
            if is_normalized:
                for k in bones.keys():
                    if k != "torso":
                        bones[k] /= torso_scale
            else:
                # Fallback für unsaubere/alte V20 Daten
                bones["torso"] *= torso_scale

        data["true_3d_bones"] = bones
        
        return (data,)


class NLFProportionalRetargeterV6:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V6: Integriert den strengen V45 Doppel-Winkel-Türsteher für die perfekte Basis-Pose."

    def process(self, video_nlf_data, calibration_data, frontal_3d_angle_tolerance):
        import copy
        import numpy as np
        import math
        import torch

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V6 (MIT V45 TÜRSTEHER) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages))

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        if is_normalized:
            log_messages.append("Modus: NORMALISIERT V22 (Prozentual basierend auf Torso=100%).\n")
        else:
            log_messages.append("Modus: ABSOLUT V20 (Absolute Werte).\n")

        # --- STUFE 1: V45 TÜRSTEHER AUF NLF ANGEPASST ---
        log_messages.append("--- WINKEL-RADAR (3D NLF SMPL Check) ---")
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            if is_tensor:
                pts = frame_data[0].cpu().numpy() if frame_data.dim() == 3 else frame_data.cpu().numpy()
            else:
                arr = np.array(frame_data)
                pts = arr[0] if arr.ndim == 3 else arr

            def is_val(idx):
                return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            # SMPL Indizes: Knie (4,5), Knöchel (7,8)
            has_knees = is_val(4) or is_val(5)
            has_ankles = is_val(7) or is_val(8)
            has_feet = has_ankles # Als Näherung für Füße in 24-Joint SMPL

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

            is_frontal = False
            frontal_pts = 0.0
            angle_h, angle_s, max_angle = 90.0, 90.0, 90.0

            if len(pts) >= 18:
                # NLF SMPL Hüfte: L=1, R=2
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))

                # NLF SMPL Schultern: L=16, R=17
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
            if is_frontal:
                frontal_indices.append(i)

        log_messages.append("\n--- PASS-FILTER (DER TÜRSTEHER) ---")
        if len(frontal_indices) > 0:
            log_messages.append(f">> PASS-FILTER AKTIV: {len(frontal_indices)} echte frontale Frames gefunden! Alle anderen fliegen raus.")
            candidates = frontal_indices
        else:
            log_messages.append(f">> PASS-FILTER INAKTIV: Kein Frame ist unter {frontal_3d_angle_tolerance}°. Nutze alle Frames als Fallback.")
            candidates = list(range(len(raw_poses)))

        # --- STUFE 2: GEWINNER ERMITTELN ---
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

        log_messages.append(f"\n-> Gewinner Frame für Basis-Torso: {best_idx} (Score: {best_score:.1f})")

        # --- DIE MAGIE: WIR MESSEN DEN REFERENZ-TORSO IM BESTEN FRAME ---
        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        if is_t:
            ref_pts = ref_frame_data[0].cpu().numpy() if ref_frame_data.dim() == 3 else ref_frame_data.cpu().numpy()
        else:
            ref_arr = np.array(ref_frame_data)
            ref_pts = ref_arr[0] if ref_arr.ndim == 3 else ref_arr

        # Torso-Länge im besten Frame (Becken=0, Hals=12)
        reference_torso_length = np.linalg.norm(ref_pts[12] - ref_pts[0])
        log_messages.append(f"Fester Referenz Torso (wird für ALLE Frames genutzt): {reference_torso_length:.4f} Einheiten\n")

        # --- STUFE 3: SMPL 3D KINEMATISCHER BAUM RETARGETING ---
        tree = {
            0: [1, 2, 3],
            1: [4], 4: [7], 7: [10],           # Linkes Bein
            2: [5], 5: [8], 8: [11],           # Rechtes Bein
            3: [6], 6: [9], 9: [12, 13, 14],   # Wirbelsäule
            12: [15],                          # Kopf
            13: [16], 16: [18], 18: [20], 20: [22], # Linker Arm
            14: [17], 17: [19], 19: [21], 21: [23]  # Rechter Arm
        }

        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child)
                    desc.extend(get_all_descendants(child, tree_map))
            return desc

        frames_processed = 0
        detailed_log_done = False

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0:
                continue
                
            is_tensor = isinstance(frame_data, torch.Tensor)
            if is_tensor:
                pts = frame_data[0].cpu().numpy().copy() if frame_data.dim() == 3 else frame_data.cpu().numpy().copy()
            else:
                arr = np.array(frame_data)
                pts = arr[0].copy() if arr.ndim == 3 else arr.copy()
            
            pts_new = pts.copy()

            if not detailed_log_done:
                log_messages.append(f"--- DETAILLIERTER BERICHT FÜR ERSTEN FRAME ---")

            # Ziel-Längen basieren jetzt auf dem PERFEKTEN reference_torso_length, nicht mehr auf dem aktuellen!
            targets = {}
            for k, v in true_3d_bones.items():
                targets[k] = (v / 100.0) * reference_torso_length if is_normalized else v

            operations = [
                ('shoulder_width', 12, 17), # Hals -> R Schulter
                ('shoulder_width', 12, 16), # Hals -> L Schulter
                ('hip_width', 0, 2),        # Becken -> R Hüfte
                ('hip_width', 0, 1),        # Becken -> L Hüfte
                ('r_arm', 17, 19),          # R Schulter -> R Ellbogen
                ('r_forearm', 19, 21),      # R Ellbogen -> R Handgelenk
                ('l_arm', 16, 18),          # L Schulter -> L Ellbogen
                ('l_forearm', 18, 20),      # L Ellbogen -> L Handgelenk
                ('r_thigh', 2, 5),          # R Hüfte -> R Knie
                ('r_calf', 5, 8),           # R Knie -> R Knöchel
                ('l_thigh', 1, 4),          # L Hüfte -> L Knie
                ('l_calf', 4, 7)            # L Knie -> L Knöchel
            ]

            for bone_key, p_idx, c_idx in operations:
                if bone_key not in targets: continue
                if c_idx >= len(pts_new) or p_idx >= len(pts_new): continue
                
                target_len = targets[bone_key]
                if bone_key in ['shoulder_width', 'hip_width']:
                    target_len = target_len / 2.0
                
                p_pos = pts_new[p_idx]
                c_pos = pts_new[c_idx]
                
                if np.linalg.norm(p_pos) < 1e-5 or np.linalg.norm(c_pos) < 1e-5:
                    continue

                vec = c_pos - p_pos
                curr_len = np.linalg.norm(vec)
                
                if curr_len < 1e-5: continue
                
                if not detailed_log_done:
                    scale_factor = target_len / curr_len
                    log_messages.append(f"Knochen: {bone_key} (Idx {p_idx} -> {c_idx})")
                    log_messages.append(f"  Länge aktuell: {curr_len:.4f}")
                    log_messages.append(f"  Länge Ziel:    {target_len:.4f}")
                    log_messages.append(f"  Faktor:        {scale_factor:.4f}x\n")
                
                dir_vec = vec / curr_len
                new_c_pos = p_pos + (dir_vec * target_len)
                
                delta = new_c_pos - c_pos
                pts_new[c_idx] += delta
                
                descendants = get_all_descendants(c_idx, tree)
                for d in descendants:
                    if d < len(pts_new):
                        if np.linalg.norm(pts_new[d]) > 1e-5:
                            pts_new[d] += delta

            detailed_log_done = True 

            if is_tensor:
                if frame_data.dim() == 3:
                    raw_poses[frame_idx][0] = torch.from_numpy(pts_new).to(frame_data.device)
                else:
                    raw_poses[frame_idx] = torch.from_numpy(pts_new).to(frame_data.device)
            else:
                arr_new = np.array(frame_data)
                if arr_new.ndim == 3:
                    arr_new[0] = pts_new
                    raw_poses[frame_idx] = arr_new.tolist()
                else:
                    raw_poses[frame_idx] = pts_new.tolist()
                
            frames_processed += 1

        log_messages.append(f"--- ZUSAMMENFASSUNG ---")
        log_messages.append(f"Erfolgreich skaliert: {frames_processed} Frames.")
        
        return (nlf_data_retargeted, "\n".join(log_messages))


class NLFProportionalRetargeterV7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V7: V6 + Head-Compensation (Addiert halben Kopf zum kurzen NLF-Torso)."

    def process(self, video_nlf_data, calibration_data, frontal_3d_angle_tolerance):
        import copy
        import numpy as np
        import math
        import torch

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V7 (MIT KOPF-KOMPENSATION) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages))

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        if is_normalized:
            log_messages.append("Modus: NORMALISIERT V22 (Prozentual basierend auf Torso=100%).\n")
        else:
            log_messages.append("Modus: ABSOLUT V20 (Absolute Werte).\n")

        # --- STUFE 1: V45 TÜRSTEHER AUF NLF ANGEPASST ---
        log_messages.append("--- WINKEL-RADAR (3D NLF SMPL Check) ---")
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            if is_tensor:
                pts = frame_data[0].cpu().numpy() if frame_data.dim() == 3 else frame_data.cpu().numpy()
            else:
                arr = np.array(frame_data)
                pts = arr[0] if arr.ndim == 3 else arr

            def is_val(idx):
                return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            has_knees = is_val(4) or is_val(5)
            has_ankles = is_val(7) or is_val(8)
            has_feet = has_ankles 

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
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

            all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
            if is_frontal:
                frontal_indices.append(i)

        log_messages.append("\n--- PASS-FILTER (DER TÜRSTEHER) ---")
        if len(frontal_indices) > 0:
            candidates = frontal_indices
        else:
            candidates = list(range(len(raw_poses)))

        # --- STUFE 2: GEWINNER ERMITTELN ---
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

        log_messages.append(f"\n-> Gewinner Frame für Basis-Torso: {best_idx} (Score: {best_score:.1f})")

        # --- DIE MAGIE: WIR MESSEN DEN REFERENZ-TORSO UND KOMPENSIEREN DEN KOPF ---
        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        if is_t:
            ref_pts = ref_frame_data[0].cpu().numpy() if ref_frame_data.dim() == 3 else ref_frame_data.cpu().numpy()
        else:
            ref_arr = np.array(ref_frame_data)
            ref_pts = ref_arr[0] if ref_arr.ndim == 3 else ref_arr

        # Rohe Torso-Länge (NLF Hals zu Becken)
        raw_torso = np.linalg.norm(ref_pts[12] - ref_pts[0])
        
        # DEIN FIX: Kopflänge auslesen und zur Hälfte addieren
        head_val = true_3d_bones.get("head", 0.0)
        if is_normalized:
            missing_neck = raw_torso * (head_val / 100.0) / 2.0
        else:
            missing_neck = head_val / 2.0
            
        reference_torso_length = raw_torso + missing_neck

        log_messages.append(f"Gemessener NLF Torso (zu kurz): {raw_torso:.4f}")
        log_messages.append(f"Kompensation (Halber Kopf addiert): +{missing_neck:.4f}")
        log_messages.append(f"-> Fester, echter Referenz Torso: {reference_torso_length:.4f} Einheiten\n")

        # --- STUFE 3: SMPL 3D KINEMATISCHER BAUM RETARGETING ---
        tree = {
            0: [1, 2, 3],
            1: [4], 4: [7], 7: [10],           
            2: [5], 5: [8], 8: [11],           
            3: [6], 6: [9], 9: [12, 13, 14],   
            12: [15],                          
            13: [16], 16: [18], 18: [20], 20: [22], 
            14: [17], 17: [19], 19: [21], 21: [23]  
        }

        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child)
                    desc.extend(get_all_descendants(child, tree_map))
            return desc

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0:
                continue
                
            is_tensor = isinstance(frame_data, torch.Tensor)
            if is_tensor:
                pts = frame_data[0].cpu().numpy().copy() if frame_data.dim() == 3 else frame_data.cpu().numpy().copy()
            else:
                arr = np.array(frame_data)
                pts = arr[0].copy() if arr.ndim == 3 else arr.copy()
            
            pts_new = pts.copy()

            targets = {}
            for k, v in true_3d_bones.items():
                targets[k] = (v / 100.0) * reference_torso_length if is_normalized else v

            operations = [
                ('shoulder_width', 12, 17), ('shoulder_width', 12, 16),
                ('hip_width', 0, 2),        ('hip_width', 0, 1),        
                ('r_arm', 17, 19),          ('r_forearm', 19, 21),      
                ('l_arm', 16, 18),          ('l_forearm', 18, 20),      
                ('r_thigh', 2, 5),          ('r_calf', 5, 8),           
                ('l_thigh', 1, 4),          ('l_calf', 4, 7)            
            ]

            for bone_key, p_idx, c_idx in operations:
                if bone_key not in targets: continue
                if c_idx >= len(pts_new) or p_idx >= len(pts_new): continue
                
                target_len = targets[bone_key]
                if bone_key in ['shoulder_width', 'hip_width']:
                    target_len = target_len / 2.0
                
                p_pos = pts_new[p_idx]
                c_pos = pts_new[c_idx]
                
                if np.linalg.norm(p_pos) < 1e-5 or np.linalg.norm(c_pos) < 1e-5: continue

                vec = c_pos - p_pos
                curr_len = np.linalg.norm(vec)
                
                if curr_len < 1e-5: continue
                
                dir_vec = vec / curr_len
                new_c_pos = p_pos + (dir_vec * target_len)
                
                delta = new_c_pos - c_pos
                pts_new[c_idx] += delta
                
                descendants = get_all_descendants(c_idx, tree)
                for d in descendants:
                    if d < len(pts_new) and np.linalg.norm(pts_new[d]) > 1e-5:
                        pts_new[d] += delta

            if is_tensor:
                if frame_data.dim() == 3: raw_poses[frame_idx][0] = torch.from_numpy(pts_new).to(frame_data.device)
                else: raw_poses[frame_idx] = torch.from_numpy(pts_new).to(frame_data.device)
            else:
                arr_new = np.array(frame_data)
                if arr_new.ndim == 3:
                    arr_new[0] = pts_new
                    raw_poses[frame_idx] = arr_new.tolist()
                else: raw_poses[frame_idx] = pts_new.tolist()

        return (nlf_data_retargeted, "\n".join(log_messages))


class NLFConfigScaler3DBones2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die originalen Konfigurationsdaten (aus V22 JSON)"}),
                "torso_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "1.0 = keine Änderung"}),
                "shoulder_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "shoulder_config_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Skaliert NUR das Schultergelenk (Arme bleiben stehen)"}),
                "hip_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "hip_config_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Skaliert NUR das Hüftgelenk (Knie bleiben stehen)"}),
                "arm_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Oberarme (Links & Rechts gekoppelt)"}),
                "forearm_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Unterarme (Links & Rechts gekoppelt)"}),
                "thigh_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Oberschenkel (Links & Rechts gekoppelt)"}),
                "calf_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Waden/Unterschenkel (Links & Rechts gekoppelt)"}),
            }
        }

    # Output um den String-Log erweitert, damit du volle Kontrolle hast
    RETURN_TYPES = ("POSE_CALIBRATION", "STRING")
    RETURN_NAMES = ("scaled_calibration_data", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"

    def process(self, calibration_data, torso_scale, shoulder_scale, shoulder_config_scale, hip_scale, hip_config_scale, arm_scale, forearm_scale, thigh_scale, calf_scale):
        import copy
        import math
        
        # Tiefe Kopie, damit wir die originalen Daten nicht überschreiben
        data = copy.deepcopy(calibration_data)
        bones = data.get("true_3d_bones", {})
        log_messages = ["=== NLF CONFIG SCALER 3D BONES ==="]
        
        if not bones:
            return (data, "Fehler: Keine 3D Bones gefunden.")
            
        # Prüfen, ob wir mit den 100% Prozentdaten (V22) arbeiten
        is_normalized = math.isclose(bones.get("torso", 0.0), 100.0, abs_tol=1e-3)

        # 1. Gekoppeltes Skalieren inkl. DOPPEL-EINTRAG (Die Kinematik-Magie)
        if "shoulder_width" in bones: 
            bones["shoulder_width"] *= shoulder_scale
            # config_scale wird ZUSÄTZLICH für das Gelenk angewandt
            bones["calibration_shoulder_width"] = bones["shoulder_width"] * shoulder_config_scale
            log_messages.append(f"Shoulder Scale: {shoulder_scale}x | Config Scale (nur Gelenk): {shoulder_config_scale}x")

        if "hip_width" in bones: 
            bones["hip_width"] *= hip_scale
            # config_scale wird ZUSÄTZLICH für das Gelenk angewandt
            bones["calibration_hip_width"] = bones["hip_width"] * hip_config_scale
            log_messages.append(f"Hip Scale: {hip_scale}x | Config Scale (nur Gelenk): {hip_config_scale}x")
        
        if "r_arm" in bones: bones["r_arm"] *= arm_scale
        if "l_arm" in bones: bones["l_arm"] *= arm_scale
        
        if "r_forearm" in bones: bones["r_forearm"] *= forearm_scale
        if "l_forearm" in bones: bones["l_forearm"] *= forearm_scale
        
        if "r_thigh" in bones: bones["r_thigh"] *= thigh_scale
        if "l_thigh" in bones: bones["l_thigh"] *= thigh_scale
        
        if "r_calf" in bones: bones["r_calf"] *= calf_scale
        if "l_calf" in bones: bones["l_calf"] *= calf_scale

        # 2. Smart Torso Scale
        # Wenn der Torso der 100% Anker ist, skalieren wir alle anderen Knochen invers, 
        # anstatt den Torso-Wert zu verändern. Das hält die Mathematik sauber!
        if torso_scale != 1.0:
            log_messages.append(f"\n-> Smart Torso Scale ({torso_scale}x) wird berechnet...")
            if is_normalized:
                # WICHTIG: list() hinzugefügt, da wir oben neue Keys generiert haben
                for k in list(bones.keys()): 
                    if k != "torso":
                        bones[k] /= torso_scale
                log_messages.append("   Alle Knochen (inkl. Calibration) wurden invers skaliert, um Torso=100% zu erhalten.")
            else:
                # Fallback für unsaubere/alte V20 Daten
                bones["torso"] *= torso_scale
                log_messages.append("   Absoluter Torso-Wert wurde multipliziert.")

        data["true_3d_bones"] = bones
        
        return (data, "\n".join(log_messages))


class NLFProportionalRetargeterV9:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V9: Doppelter Kinematik-Eintrag (calibration_hip_width) für natürliche Knie-Verschiebung."

    def process(self, video_nlf_data, calibration_data, frontal_3d_angle_tolerance):
        import copy
        import numpy as np
        import math
        import torch

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V9 (DOUBLE ENTRY KINEMATICS & FULL LOGS) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages))

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        if is_normalized:
            log_messages.append("Modus: NORMALISIERT V22 (Prozentual basierend auf Torso=100%).\n")
        else:
            log_messages.append("Modus: ABSOLUT V20 (Absolute Werte).\n")

        # --- STUFE 1: V45 TÜRSTEHER AUF NLF ANGEPASST ---
        log_messages.append("--- WINKEL-RADAR (Alle Frames werden protokolliert) ---")
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'has_feet': False, 'has_ankles': False, 'has_knees': False, 'is_frontal': False, 'length': 0.0, 'frontal_pts': 0.0})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            if is_tensor:
                pts = frame_data[0].cpu().numpy() if frame_data.dim() == 3 else frame_data.cpu().numpy()
            else:
                arr = np.array(frame_data)
                pts = arr[0] if arr.ndim == 3 else arr

            def is_val(idx):
                return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            has_knees = is_val(4) or is_val(5)
            has_ankles = is_val(7) or is_val(8)
            has_feet = has_ankles 

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
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

            status = "FRONTAL (Akzeptiert)" if is_frontal else "SEITLICH (Abgelehnt)"
            log_messages.append(f"Frame {i}: Max-Winkel {max_angle:.1f}° (Hüfte: {angle_h:.1f}°, Schultern: {angle_s:.1f}°) -> {status}")

            all_frames_data.append({'has_feet': has_feet, 'has_ankles': has_ankles, 'has_knees': has_knees, 'is_frontal': is_frontal, 'length': length, 'frontal_pts': frontal_pts})
            if is_frontal:
                frontal_indices.append(i)

        log_messages.append("\n--- PASS-FILTER (DER TÜRSTEHER) ---")
        if len(frontal_indices) > 0:
            log_messages.append(f">> PASS-FILTER AKTIV: {len(frontal_indices)} frontale Frames haben es geschafft.")
            candidates = frontal_indices
        else:
            log_messages.append(f">> PASS-FILTER INAKTIV: Kein Frame erfüllt die Toleranz. Nutze alle Frames.")
            candidates = list(range(len(raw_poses)))

        # --- STUFE 2: GEWINNER ERMITTELN ---
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

        log_messages.append(f"\n-> Gewinner Frame für Basis-Torso: {best_idx} (Score: {best_score:.1f})")

        # --- KOPF-KOMPENSATION FÜR NLF TORSO ---
        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        if is_t:
            ref_pts = ref_frame_data[0].cpu().numpy() if ref_frame_data.dim() == 3 else ref_frame_data.cpu().numpy()
        else:
            ref_arr = np.array(ref_frame_data)
            ref_pts = ref_arr[0] if ref_arr.ndim == 3 else ref_arr

        raw_torso = np.linalg.norm(ref_pts[12] - ref_pts[0])
        head_val = true_3d_bones.get("head", 0.0)
        
        if is_normalized:
            missing_neck = raw_torso * (head_val / 100.0) / 2.0
        else:
            missing_neck = head_val / 2.0
            
        reference_torso_length = raw_torso + missing_neck

        log_messages.append(f"Gemessener NLF Torso (Hals zu Becken): {raw_torso:.4f}")
        log_messages.append(f"Kompensation (Halber Kopf addiert): +{missing_neck:.4f}")
        log_messages.append(f"-> Fester, echter Referenz Torso: {reference_torso_length:.4f} Einheiten\n")

        # --- STUFE 3: SMPL 3D KINEMATISCHER BAUM RETARGETING ---
        tree = {
            0: [1, 2, 3],
            1: [4], 4: [7], 7: [10],           
            2: [5], 5: [8], 8: [11],           
            3: [6], 6: [9], 9: [12, 13, 14],   
            12: [15],                          
            13: [16], 16: [18], 18: [20], 20: [22], 
            14: [17], 17: [19], 19: [21], 21: [23]  
        }

        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child)
                    desc.extend(get_all_descendants(child, tree_map))
            return desc

        detailed_log_done = False
        frames_processed = 0

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0:
                continue
                
            is_tensor = isinstance(frame_data, torch.Tensor)
            if is_tensor:
                pts = frame_data[0].cpu().numpy().copy() if frame_data.dim() == 3 else frame_data.cpu().numpy().copy()
            else:
                arr = np.array(frame_data)
                pts = arr[0].copy() if arr.ndim == 3 else arr.copy()
            
            pts_new = pts.copy()

            targets = {}
            for k, v in true_3d_bones.items():
                targets[k] = (v / 100.0) * reference_torso_length if is_normalized else v

            operations = [
                ('shoulder_width', 12, 17), ('shoulder_width', 12, 16),
                ('hip_width', 0, 2),        ('hip_width', 0, 1),        
                ('r_arm', 17, 19),          ('r_forearm', 19, 21),      
                ('l_arm', 16, 18),          ('l_forearm', 18, 20),      
                ('r_thigh', 2, 5),          ('r_calf', 5, 8),           
                ('l_thigh', 1, 4),          ('l_calf', 4, 7)            
            ]

            do_log_bones = (frame_idx == best_idx and not detailed_log_done)
            if do_log_bones:
                log_messages.append(f"--- BONE SCALING DETAILS (Gemessen am Anker-Frame {best_idx}) ---")

            for bone_key, p_idx, c_idx in operations:
                if bone_key not in targets: continue
                if c_idx >= len(pts_new) or p_idx >= len(pts_new): continue
                
                # DIE MAGIE: DOPPELTER EINTRAG (NORMAL VS CALIBRATION)
                target_len_normal = targets[bone_key]
                calib_key = "calibration_" + bone_key
                target_len_calib = targets.get(calib_key, target_len_normal)

                if bone_key in ['shoulder_width', 'hip_width']:
                    target_len_normal = target_len_normal / 2.0
                    target_len_calib = target_len_calib / 2.0
                
                p_pos = pts_new[p_idx]
                c_pos = pts_new[c_idx]
                
                if np.linalg.norm(p_pos) < 1e-5 or np.linalg.norm(c_pos) < 1e-5: continue

                vec = c_pos - p_pos
                curr_len = np.linalg.norm(vec)
                
                if curr_len < 1e-5: continue
                
                scale_normal = target_len_normal / curr_len
                scale_calib = target_len_calib / curr_len
                
                if do_log_bones:
                    if scale_normal != scale_calib:
                        log_messages.append(f"Knochen: {bone_key.ljust(15)} | Idx {str(p_idx).rjust(2)} -> {str(c_idx).rjust(2)} | Gelenk-Faktor: {scale_calib:.4f}x | Kinder-Faktor: {scale_normal:.4f}x")
                    else:
                        log_messages.append(f"Knochen: {bone_key.ljust(15)} | Idx {str(p_idx).rjust(2)} -> {str(c_idx).rjust(2)} | Ist: {curr_len:.4f} | Soll: {target_len_normal:.4f} | Faktor: {scale_normal:.4f}x")

                dir_vec = vec / curr_len
                
                # Delta für die Kinder (Knie, Füße -> Normaler Scale)
                new_c_pos_normal = p_pos + (dir_vec * target_len_normal)
                delta_normal = new_c_pos_normal - c_pos

                # Delta für das Gelenk selbst (Hüfte, Schulter -> Calibration Scale)
                new_c_pos_calib = p_pos + (dir_vec * target_len_calib)
                delta_calib = new_c_pos_calib - c_pos
                
                # 1. Gelenk verschieben
                pts_new[c_idx] += delta_calib
                
                # 2. Alle Kinder verschieben (mit dem "normalen", schwächeren Delta)
                descendants = get_all_descendants(c_idx, tree)
                for d in descendants:
                    if d < len(pts_new) and np.linalg.norm(pts_new[d]) > 1e-5:
                        pts_new[d] += delta_normal

            if do_log_bones:
                detailed_log_done = True
                log_messages.append("------------------------------------------------------------------\n")

            if is_tensor:
                if frame_data.dim() == 3: raw_poses[frame_idx][0] = torch.from_numpy(pts_new).to(frame_data.device)
                else: raw_poses[frame_idx] = torch.from_numpy(pts_new).to(frame_data.device)
            else:
                arr_new = np.array(frame_data)
                if arr_new.ndim == 3:
                    arr_new[0] = pts_new
                    raw_poses[frame_idx] = arr_new.tolist()
                else: raw_poses[frame_idx] = pts_new.tolist()
            
            frames_processed += 1

        log_messages.append(f"--- ZUSAMMENFASSUNG ---")
        log_messages.append(f"Erfolgreich skaliert: {frames_processed} Frames.")
        
        return (nlf_data_retargeted, "\n".join(log_messages))


class NLFProportionalRetargeterV13:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V18: Exaktes Mapping! Base-Width = Ganzes Bein (Stance), Calibration-Width = Nur Gelenk (Config)."

    def process(self, video_nlf_data, calibration_data, frontal_3d_angle_tolerance):
        import copy
        import numpy as np
        import math
        import torch

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V18 (BASE STANCE -> CALIBRATION JOINT) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages))

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)

        # --- STUFE 1: TÜRSTEHER (WINKEL-RADAR) ---
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'length': 0.0, 'is_frontal': False, 'has_feet': False})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else np.array(frame_data))
            if pts.ndim == 3: pts = pts[0]

            def is_val(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

            is_frontal = False
            if len(pts) >= 18:
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                if max(angle_h, angle_s) <= frontal_3d_angle_tolerance:
                    is_frontal = True

            all_frames_data.append({'length': length, 'is_frontal': is_frontal, 'has_feet': is_val(7) or is_val(8)})
            if is_frontal: frontal_indices.append(i)

        # --- STUFE 2: ANCHOR-FRAME BESTIMMEN ---
        candidates = frontal_indices if frontal_indices else list(range(len(raw_poses)))
        max_len = max([d['length'] for d in all_frames_data]) if all_frames_data else 1.0
        
        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            d = all_frames_data[idx]
            score = (1000.0 if d['has_feet'] else 0.0) + (500.0 if d['is_frontal'] else 0.0) + ((d['length'] / max_len) * 100.0)
            if score > best_score:
                best_score, best_idx = score, idx

        log_messages.append(f"-> Referenz-Frame für Messung: {best_idx}")

        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        ref_pts = ref_frame_data[0].cpu().numpy() if is_t and ref_frame_data.dim() == 3 else (ref_frame_data.cpu().numpy() if is_t else np.array(ref_frame_data))
        if ref_pts.ndim == 3: ref_pts = ref_pts[0]

        orig_torso_ref = np.linalg.norm(ref_pts[12] - ref_pts[0]) if np.linalg.norm(ref_pts[12]) > 1e-5 else 0.0
        head_val = true_3d_bones.get("head", 0.0)
        missing_neck = orig_torso_ref * (head_val / 100.0) / 2.0 if is_normalized else head_val / 2.0
        reference_torso_length = orig_torso_ref + missing_neck

        # --- HILFSFUNKTIONEN ---
        tree = {0:[1,2,3], 1:[4], 4:[7], 7:[10], 2:[5], 5:[8], 8:[11], 3:[6], 6:[9], 9:[12,13,14], 12:[15], 13:[16], 16:[18], 18:[20], 20:[22], 14:[17], 17:[19], 19:[21], 21:[23]}
        
        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child); desc.extend(get_all_descendants(child, tree_map))
            return desc

        def get_height_stable(p_array):
            if 12 < len(p_array) and np.linalg.norm(p_array[12]) > 1e-5:
                top_y = p_array[12][1]
            else: return 0.0
            feet_y = [p_array[idx][1] for idx in [7,8,10,11,4,5] if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5]
            return (max(feet_y) - top_y) if feet_y else 0.0

        # --- VERARBEITUNG ---
        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0: continue
            
            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())
            if pts.ndim == 3: pts = pts[0]

            targets = {k: (v / 100.0 * reference_torso_length if is_normalized else v) for k, v in true_3d_bones.items()}

            def build_and_log(pts_source, factor, final_mode=False):
                pts_b = pts_source.copy()
                do_log = (final_mode and frame_idx == best_idx)
                
                # 1. Torso Skalierung
                cv = pts_b[12] - pts_b[0]; cl = np.linalg.norm(cv)
                if cl > 1e-5:
                    t_len = targets.get("torso", cl)
                    if final_mode: t_len *= factor
                    if do_log: log_messages.append(f"Knochen: Torso          | Ist: {cl:.4f} -> Soll: {t_len:.4f}")
                    
                    f_node = t_len / cl
                    for p, c in [(0,3), (3,6), (6,9), (9,12)]:
                        vec = pts_b[c] - pts_b[p]; new_c = pts_b[p] + vec * f_node
                        delta = new_c - pts_b[c]; pts_b[c] += delta
                        for d in get_all_descendants(c, tree):
                            if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5: pts_b[d] += delta

                # 2. Kopf
                if 15 < len(pts_b):
                    cv = pts_b[15] - pts_b[12]; cl = np.linalg.norm(cv)
                    if cl > 1e-5:
                        t_len = targets.get("head", cl * 2.0) / 2.0
                        if do_log: log_messages.append(f"Knochen: Kopf (NLF-Map)  | Ist: {cl:.4f} -> Soll: {t_len:.4f}")
                        f_node = t_len / cl
                        delta = (pts_b[12] + (cv / cl * t_len)) - pts_b[15]
                        pts_b[15] += delta

                # 3. Operations (Gliedmaßen & Breiten)
                ops = [('shoulder_width',12,17), ('shoulder_width',12,16), ('hip_width',0,2), ('hip_width',0,1),
                       ('r_arm',17,19), ('r_forearm',19,21), ('l_arm',16,18), ('l_forearm',18,20),
                       ('r_thigh',2,5), ('r_calf',5,8), ('l_thigh',1,4), ('l_calf',4,7)]

                for key, p_idx, c_idx in ops:
                    cv = pts_b[c_idx] - pts_b[p_idx]; cl = np.linalg.norm(cv)
                    if cl < 1e-5: continue

                    if key in ['shoulder_width', 'hip_width']:
                        # --- V18 LOGIK: KLARE TRENNUNG BASIS vs CALIBRATION ---
                        
                        # 1. Stance Target (Ganzes Bein wandert) -> Nimmt den Standard Config Wert (z.B. 43.97)
                        stance_target = targets.get(key, cl * 2.0) / 2.0
                        
                        # 2. Bone Target (Nur Gelenk wandert) -> Nimmt den Calibration Wert (z.B. 77.84)
                        calib_key = f"calibration_{key}"
                        bone_target = targets.get(calib_key, stance_target * 2.0) / 2.0

                        # SCHRITT A: STANCE SCALE ANWENDEN (Bewegt Gelenk UND Beine!)
                        scale_xz_stance = stance_target / cl
                        pos_stance = pts_b[p_idx].copy()
                        pos_stance[0] += cv[0] * scale_xz_stance
                        pos_stance[1] += cv[1] # Y-Lock
                        pos_stance[2] += cv[2] * scale_xz_stance
                        
                        delta_stance = pos_stance - pts_b[c_idx]
                        pts_b[c_idx] += delta_stance # Gelenk verschieben
                        for d in get_all_descendants(c_idx, tree):
                            if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                                pts_b[d] += delta_stance # Beine mitverschieben

                        # SCHRITT B: BONE CONFIG ANWENDEN (Bewegt NUR das Gelenk nach!)
                        scale_xz_config = bone_target / cl
                        pos_config = pts_b[p_idx].copy()
                        pos_config[0] += cv[0] * scale_xz_config
                        pos_config[1] += cv[1] # Y-Lock
                        pos_config[2] += cv[2] * scale_xz_config
                        
                        # Wir berechnen die Differenz von der Stance-Position zur neuen Bone-Position
                        delta_config = pos_config - pts_b[c_idx]
                        pts_b[c_idx] += delta_config # NUR das Gelenk verschieben, Beine bleiben stehen!
                                
                        # Log-Fix: Wird nur für die rechte Seite (Index 2 bei Hip, 17 bei Schulter) ausgedruckt, damit es übersichtlich bleibt
                        if do_log and (c_idx == 2 or c_idx == 17): 
                            log_messages.append(f"Knochen: {key.ljust(15)} | Ist: {cl:.4f} -> Scale (ganzes Bein): {stance_target:.4f} -> Config (nur Gelenk): {bone_target:.4f}")

                    else:
                        # Normale Arme / Beine
                        if key not in targets: continue
                        t_len_normal = targets[key]
                        cal_k = "calibration_" + key
                        t_len_final = targets.get(cal_k, t_len_normal)
                        
                        if final_mode: t_len_final *= factor
                        dir_vec = cv / cl
                        new_c_pos = pts_b[p_idx] + (dir_vec * t_len_final)

                        # Log-Fix: Nur rechts loggen
                        if do_log and key.startswith('r_'):
                            log_messages.append(f"Knochen: {key.ljust(15)} | Ist: {cl:.4f} -> Soll: {t_len_final:.4f}")

                        delta_shift = new_c_pos - pts_b[c_idx]
                        pts_b[c_idx] = new_c_pos

                        for d in get_all_descendants(c_idx, tree):
                            if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                                pts_b[d] += delta_shift
                            
                return pts_b

            # PHASE 1: Messung
            orig_h = get_height_stable(pts)
            pts_dry = build_and_log(pts, 1.0, final_mode=False)
            dry_h = get_height_stable(pts_dry)
            
            f_scale = orig_h / dry_h if (orig_h > 1e-5 and dry_h > 1e-5) else 1.0

            if frame_idx == best_idx:
                log_messages.append(f"\n--- SKALIERUNGS-LOG (Frame {frame_idx}) ---")
                log_messages.append(f"Gemessene Ziel-Skalierung: {f_scale:.4f}x")

            # PHASE 2: Finaler Build
            pts_final = build_and_log(pts, f_scale, final_mode=True)

            # GROUND ANCHOR
            v_orig_feet = [pts[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
            v_new_feet = [pts_final[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts_final) and np.linalg.norm(pts_final[idx]) > 1e-5]
            if v_orig_feet and v_new_feet:
                shift = max(v_orig_feet) - max(v_new_feet)
                for j in range(len(pts_final)):
                    if np.linalg.norm(pts_final[j]) > 1e-5: pts_final[j][1] += shift

            # Output-Formatierung
            if is_tensor:
                if frame_data.dim() == 3: raw_poses[frame_idx][0] = torch.from_numpy(pts_final).to(frame_data.device)
                else: raw_poses[frame_idx] = torch.from_numpy(pts_final).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts_final.tolist()

        return (nlf_data_retargeted, "\n".join(log_messages))


class NLFProportionalRetargeterV14:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
                "scale_stance_and_head": ("BOOLEAN", {"default": False, "tooltip": "Wendet den Höhen-Korrekturfaktor auch auf Schulter-/Hüftbreite und den Kopf an."}),
            },
            "optional": {
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus dem Scaler (wird bereinigt)"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output", "nlf_render_config_clean")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V14: Perfekter NLF-Loop zur Höhenstabilisierung. Bereinigt zusätzlich die Kamera-Config für Mimic."

    def process(self, video_nlf_data, calibration_data, frontal_3d_angle_tolerance, scale_stance_and_head, nlf_render_config="{}"):
        import copy
        import numpy as np
        import math
        import torch
        import json

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V14 (EXAKTER 3D-LOOP & CAMERA FIX) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            # Gebe bei Fehler eine leere Config zurück
            return (video_nlf_data, "\n".join(log_messages), "{}")

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)

        # --- STUFE 1: TÜRSTEHER (WINKEL-RADAR) ---
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'length': 0.0, 'is_frontal': False, 'has_feet': False})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else np.array(frame_data))
            if pts.ndim == 3: pts = pts[0]

            def is_val(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

            is_frontal = False
            if len(pts) >= 18:
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                if max(angle_h, angle_s) <= frontal_3d_angle_tolerance:
                    is_frontal = True

            all_frames_data.append({'length': length, 'is_frontal': is_frontal, 'has_feet': is_val(7) or is_val(8)})
            if is_frontal: frontal_indices.append(i)

        # --- STUFE 2: ANCHOR-FRAME BESTIMMEN ---
        candidates = frontal_indices if frontal_indices else list(range(len(raw_poses)))
        max_len = max([d['length'] for d in all_frames_data]) if all_frames_data else 1.0
        
        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            d = all_frames_data[idx]
            score = (1000.0 if d['has_feet'] else 0.0) + (500.0 if d['is_frontal'] else 0.0) + ((d['length'] / max_len) * 100.0)
            if score > best_score:
                best_score, best_idx = score, idx

        log_messages.append(f"-> Referenz-Frame (Anchor) für globalen Scale-Loop: {best_idx}")

        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        ref_pts = ref_frame_data[0].cpu().numpy() if is_t and ref_frame_data.dim() == 3 else (ref_frame_data.cpu().numpy() if is_t else np.array(ref_frame_data))
        if ref_pts.ndim == 3: ref_pts = ref_pts[0]

        orig_torso_ref = np.linalg.norm(ref_pts[12] - ref_pts[0]) if np.linalg.norm(ref_pts[12]) > 1e-5 else 0.0
        head_val = true_3d_bones.get("head", 0.0)
        missing_neck = orig_torso_ref * (head_val / 100.0) / 2.0 if is_normalized else head_val / 2.0
        reference_torso_length = orig_torso_ref + missing_neck

        targets = {k: (v / 100.0 * reference_torso_length if is_normalized else v) for k, v in true_3d_bones.items()}

        # --- HILFSFUNKTIONEN ---
        tree = {0:[1,2,3], 1:[4], 4:[7], 7:[10], 2:[5], 5:[8], 8:[11], 3:[6], 6:[9], 9:[12,13,14], 12:[15], 13:[16], 16:[18], 18:[20], 20:[22], 14:[17], 17:[19], 19:[21], 21:[23]}
        
        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child); desc.extend(get_all_descendants(child, tree_map))
            return desc

        def get_height_stable(p_array):
            if 12 < len(p_array) and np.linalg.norm(p_array[12]) > 1e-5:
                top_y = p_array[12][1]
            else: return 0.0
            feet_y = [p_array[idx][1] for idx in [7,8,10,11,4,5] if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5]
            return (max(feet_y) - top_y) if feet_y else 0.0

        def build_and_log(pts_source, factor, final_mode=False, do_log=False):
            pts_b = pts_source.copy()
            
            # 1. Torso Skalierung
            cv = pts_b[12] - pts_b[0]; cl = np.linalg.norm(cv)
            if cl > 1e-5:
                t_len = targets.get("torso", cl)
                if final_mode: t_len *= factor
                if do_log: log_messages.append(f"Knochen: Torso          | Ist: {cl:.4f} -> Soll: {t_len:.4f}")
                
                f_node = t_len / cl
                for p, c in [(0,3), (3,6), (6,9), (9,12)]:
                    vec = pts_b[c] - pts_b[p]; new_c = pts_b[p] + vec * f_node
                    delta = new_c - pts_b[c]; pts_b[c] += delta
                    for d in get_all_descendants(c, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5: pts_b[d] += delta

            # 2. Kopf
            if 15 < len(pts_b):
                cv = pts_b[15] - pts_b[12]; cl = np.linalg.norm(cv)
                if cl > 1e-5:
                    t_len = targets.get("head", cl * 2.0) / 2.0
                    if final_mode and scale_stance_and_head: 
                        t_len *= factor
                    
                    if do_log: log_messages.append(f"Knochen: Kopf (NLF)      | Ist: {cl:.4f} -> Soll: {t_len:.4f}")
                    f_node = t_len / cl
                    delta = (pts_b[12] + (cv / cl * t_len)) - pts_b[15]
                    pts_b[15] += delta

            # 3. Operations (Gliedmaßen & Breiten)
            ops = [('shoulder_width',12,17), ('shoulder_width',12,16), ('hip_width',0,2), ('hip_width',0,1),
                   ('r_arm',17,19), ('r_forearm',19,21), ('l_arm',16,18), ('l_forearm',18,20),
                   ('r_thigh',2,5), ('r_calf',5,8), ('l_thigh',1,4), ('l_calf',4,7)]

            for key, p_idx, c_idx in ops:
                cv = pts_b[c_idx] - pts_b[p_idx]; cl = np.linalg.norm(cv)
                if cl < 1e-5: continue

                if key in ['shoulder_width', 'hip_width']:
                    stance_target = targets.get(key, cl * 2.0) / 2.0
                    calib_key = f"calibration_{key}"
                    bone_target = targets.get(calib_key, stance_target * 2.0) / 2.0

                    if final_mode and scale_stance_and_head:
                        stance_target *= factor
                        bone_target *= factor

                    scale_xz_stance = stance_target / cl
                    pos_stance = pts_b[p_idx].copy()
                    pos_stance[0] += cv[0] * scale_xz_stance
                    pos_stance[1] += cv[1] # Y-Lock
                    pos_stance[2] += cv[2] * scale_xz_stance
                    
                    delta_stance = pos_stance - pts_b[c_idx]
                    pts_b[c_idx] += delta_stance
                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_stance

                    scale_xz_config = bone_target / cl
                    pos_config = pts_b[p_idx].copy()
                    pos_config[0] += cv[0] * scale_xz_config
                    pos_config[1] += cv[1] # Y-Lock
                    pos_config[2] += cv[2] * scale_xz_config
                    
                    delta_config = pos_config - pts_b[c_idx]
                    pts_b[c_idx] += delta_config
                            
                    if do_log and (c_idx == 2 or c_idx == 17): 
                        log_messages.append(f"Knochen: {key.ljust(15)} | Ist: {cl:.4f} -> Scale: {stance_target:.4f} -> Config: {bone_target:.4f}")

                else:
                    if key not in targets: continue
                    t_len_normal = targets[key]
                    cal_k = "calibration_" + key
                    t_len_final = targets.get(cal_k, t_len_normal)
                    
                    if final_mode: t_len_final *= factor
                    
                    dir_vec = cv / cl
                    new_c_pos = pts_b[p_idx] + (dir_vec * t_len_final)

                    if do_log and key.startswith('r_'):
                        log_messages.append(f"Knochen: {key.ljust(15)} | Ist: {cl:.4f} -> Soll: {t_len_final:.4f}")

                    delta_shift = new_c_pos - pts_b[c_idx]
                    pts_b[c_idx] = new_c_pos

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_shift
                        
            return pts_b


        # --- PHASE 1: ITERATIVER LOOP AM ANCHOR-FRAME (Für NLF-Raum Perfektion) ---
        orig_h_global = get_height_stable(ref_pts)
        global_f_scale = 1.0
        
        log_messages.append(f"\n--- NLF-LOOP SKALIERUNG (Anchor Frame {best_idx}) ---")
        log_messages.append(f"Ziel-Originalhöhe (NLF-Raum): {orig_h_global:.4f}")
        
        if orig_h_global > 1e-5:
            # Maximal 10 Iterationen, um den perfekten Faktor zu finden
            for iteration in range(10):
                pts_test = build_and_log(ref_pts, global_f_scale, final_mode=True, do_log=False)
                test_h = get_height_stable(pts_test)
                
                if test_h < 1e-5: break
                
                diff = abs(orig_h_global - test_h)
                if diff < 0.1: # Toleranz: 0.1 NLF-Einheiten Differenz
                    log_messages.append(f"Loop beendet in Iteration {iteration+1}. Differenz < 0.1")
                    break
                    
                ratio = orig_h_global / test_h
                global_f_scale *= ratio

        log_messages.append(f"Erreichte Test-Höhe im NLF-Raum: {test_h:.4f}")
        log_messages.append(f"-> Angewandter Globaler Faktor für alle Frames: {global_f_scale:.6f}x")


        # --- PHASE 2: VERARBEITUNG ALLER FRAMES ---
        log_messages.append("\n--- LOG FINALE KNochenlängen (Anchor Frame) ---")
        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0: continue
            
            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())
            if pts.ndim == 3: pts = pts[0]

            is_anchor = (frame_idx == best_idx)

            # Wende den finalen Scale auf alle Frames an
            pts_final = build_and_log(pts, global_f_scale, final_mode=True, do_log=is_anchor)

            # GROUND ANCHOR (Person wieder auf den Boden stellen)
            v_orig_feet = [pts[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
            v_new_feet = [pts_final[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts_final) and np.linalg.norm(pts_final[idx]) > 1e-5]
            if v_orig_feet and v_new_feet:
                shift = max(v_orig_feet) - max(v_new_feet)
                for j in range(len(pts_final)):
                    if np.linalg.norm(pts_final[j]) > 1e-5: pts_final[j][1] += shift

            # Output-Formatierung
            if is_tensor:
                if frame_data.dim() == 3: raw_poses[frame_idx][0] = torch.from_numpy(pts_final).to(frame_data.device)
                else: raw_poses[frame_idx] = torch.from_numpy(pts_final).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts_final.tolist()


        # --- PHASE 3: KAMERA-CONFIG BEREINIGEN ---
        # Da wir die Höhe jetzt physikalisch in 3D korrigiert haben, darf Mimic16 die Kamera NICHT mehr verzerren!
        try:
            config_dict = json.loads(nlf_render_config) if isinstance(nlf_render_config, str) and nlf_render_config.strip() else {}
        except Exception as e:
            config_dict = {}
            log_messages.append(f"Konnte nlf_render_config nicht parsen: {e}")
            
        # Wir zwingen die störenden Scale-Faktoren hart auf 1.0
        config_dict["anchor_scale"] = 1.0
        config_dict["scale_x_factor"] = 1.0
        
        # Den Rest (wie pivot_x, pivot_y) lassen wir in Ruhe, das braucht Mimic16 vielleicht noch.
        clean_config_str = json.dumps(config_dict)
        log_messages.append("\n-> Kamera-Config wurde erfolgreich für Mimic16 bereinigt (Scale = 1.0)")


        return (nlf_data_retargeted, "\n".join(log_messages), clean_config_str)


class NLFProportionalRetargeterV16:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
                "scale_stance_and_head": ("BOOLEAN", {"default": False, "tooltip": "Wendet den Höhen-Korrekturfaktor auch auf Schulter-/Hüftbreite und den Kopf an."}),
            },
            "optional": {
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus dem Scaler (wird bereinigt)"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output", "nlf_render_config_clean")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V16: Total-Height-Enforcer mit detailliertem 3D-Knochen-Log (Alle 10 Frames) + Camera Fix."

    def process(self, video_nlf_data, calibration_data, frontal_3d_angle_tolerance, scale_stance_and_head, nlf_render_config="{}"):
        import copy
        import numpy as np
        import math
        import torch
        import json

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V16 (TOTAL HEIGHT ENFORCER & LOGGING) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        head_val = true_3d_bones.get("head", 0.0)

        # --- STUFE 1: TÜRSTEHER (WINKEL-RADAR) ---
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'length': 0.0, 'is_frontal': False, 'has_feet': False})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else np.array(frame_data))
            if pts.ndim == 3: pts = pts[0]

            def is_val(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

            is_frontal = False
            if len(pts) >= 18:
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                if max(angle_h, angle_s) <= frontal_3d_angle_tolerance:
                    is_frontal = True

            all_frames_data.append({'length': length, 'is_frontal': is_frontal, 'has_feet': is_val(7) or is_val(8)})
            if is_frontal: frontal_indices.append(i)

        # --- STUFE 2: ANCHOR-FRAME BESTIMMEN ---
        candidates = frontal_indices if frontal_indices else list(range(len(raw_poses)))
        max_len = max([d['length'] for d in all_frames_data]) if all_frames_data else 1.0
        
        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            d = all_frames_data[idx]
            score = (1000.0 if d['has_feet'] else 0.0) + (500.0 if d['is_frontal'] else 0.0) + ((d['length'] / max_len) * 100.0)
            if score > best_score:
                best_score, best_idx = score, idx

        log_messages.append(f"-> Referenz-Frame (Anchor) für globalen Ratio-Loop: {best_idx}")

        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        ref_pts = ref_frame_data[0].cpu().numpy() if is_t and ref_frame_data.dim() == 3 else (ref_frame_data.cpu().numpy() if is_t else np.array(ref_frame_data))
        if ref_pts.ndim == 3: ref_pts = ref_pts[0]

        # --- HILFSFUNKTIONEN ---
        tree = {0:[1,2,3], 1:[4], 4:[7], 7:[10], 2:[5], 5:[8], 8:[11], 3:[6], 6:[9], 9:[12,13,14], 12:[15], 13:[16], 16:[18], 18:[20], 20:[22], 14:[17], 17:[19], 19:[21], 21:[23]}
        
        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child); desc.extend(get_all_descendants(child, tree_map))
            return desc

        def get_height_stable(p_array):
            if 12 < len(p_array) and np.linalg.norm(p_array[12]) > 1e-5:
                top_y = p_array[12][1]
            else: return 0.0
            feet_y = [p_array[idx][1] for idx in [7,8,10,11,4,5] if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5]
            return (max(feet_y) - top_y) if feet_y else 0.0

        def get_bone_lengths(pts_array):
            # Berechnet die exakten Vektorlängen in 3D
            def dist(p1, p2):
                if p1 < len(pts_array) and p2 < len(pts_array) and np.linalg.norm(pts_array[p1]) > 1e-5 and np.linalg.norm(pts_array[p2]) > 1e-5:
                    return float(np.linalg.norm(pts_array[p2] - pts_array[p1]))
                return 0.0

            return {
                "Torso": dist(0, 12),
                "Kopf": dist(12, 15),
                "R_Oberschenkel": dist(2, 5),
                "R_Wade": dist(5, 8),
                "L_Oberschenkel": dist(1, 4),
                "L_Wade": dist(4, 7),
                "R_Arm": dist(17, 19),
                "R_Unterarm": dist(19, 21),
                "L_Arm": dist(16, 18),
                "L_Unterarm": dist(18, 20),
                "Schulterbreite": dist(16, 17), # Von Schulter zu Schulter
                "Hueftbreite": dist(1, 2)       # Von Huefte zu Huefte
            }

        def build_and_log(pts_source, factor, final_mode=False):
            pts_b = pts_source.copy()
            
            # Dynamische Targets pro Frame (Erhält Perspektive)
            orig_torso_curr = np.linalg.norm(pts_b[12] - pts_b[0]) if np.linalg.norm(pts_b[12]) > 1e-5 else 0.0
            missing_neck_curr = orig_torso_curr * (head_val / 100.0) / 2.0 if is_normalized else head_val / 2.0
            frame_ref_torso = orig_torso_curr + missing_neck_curr

            targets = {k: (v / 100.0 * frame_ref_torso if is_normalized else v) for k, v in true_3d_bones.items()}
            
            # 1. Torso Skalierung
            cv = pts_b[12] - pts_b[0]; cl = np.linalg.norm(cv)
            if cl > 1e-5:
                t_len = targets.get("torso", cl)
                if final_mode: t_len *= factor # Total-Height-Enforcer staucht den Torso
                
                f_node = t_len / cl
                for p, c in [(0,3), (3,6), (6,9), (9,12)]:
                    vec = pts_b[c] - pts_b[p]; new_c = pts_b[p] + vec * f_node
                    delta = new_c - pts_b[c]; pts_b[c] += delta
                    for d in get_all_descendants(c, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5: pts_b[d] += delta

            # 2. Kopf
            if 15 < len(pts_b):
                cv = pts_b[15] - pts_b[12]; cl = np.linalg.norm(cv)
                if cl > 1e-5:
                    t_len = targets.get("head", cl * 2.0) / 2.0
                    if final_mode and scale_stance_and_head: 
                        t_len *= factor
                    
                    f_node = t_len / cl
                    delta = (pts_b[12] + (cv / cl * t_len)) - pts_b[15]
                    pts_b[15] += delta

            # 3. Operations (Gliedmaßen & Breiten)
            ops = [('shoulder_width',12,17), ('shoulder_width',12,16), ('hip_width',0,2), ('hip_width',0,1),
                   ('r_arm',17,19), ('r_forearm',19,21), ('l_arm',16,18), ('l_forearm',18,20),
                   ('r_thigh',2,5), ('r_calf',5,8), ('l_thigh',1,4), ('l_calf',4,7)]

            for key, p_idx, c_idx in ops:
                cv = pts_b[c_idx] - pts_b[p_idx]; cl = np.linalg.norm(cv)
                if cl < 1e-5: continue

                if key in ['shoulder_width', 'hip_width']:
                    stance_target = targets.get(key, cl * 2.0) / 2.0
                    calib_key = f"calibration_{key}"
                    bone_target = targets.get(calib_key, stance_target * 2.0) / 2.0

                    if final_mode and scale_stance_and_head:
                        stance_target *= factor
                        bone_target *= factor

                    scale_xz_stance = stance_target / cl
                    pos_stance = pts_b[p_idx].copy()
                    pos_stance[0] += cv[0] * scale_xz_stance
                    pos_stance[1] += cv[1] # Y-Lock
                    pos_stance[2] += cv[2] * scale_xz_stance
                    
                    delta_stance = pos_stance - pts_b[c_idx]
                    pts_b[c_idx] += delta_stance
                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_stance

                    scale_xz_config = bone_target / cl
                    pos_config = pts_b[p_idx].copy()
                    pos_config[0] += cv[0] * scale_xz_config
                    pos_config[1] += cv[1] # Y-Lock
                    pos_config[2] += cv[2] * scale_xz_config
                    
                    delta_config = pos_config - pts_b[c_idx]
                    pts_b[c_idx] += delta_config

                else:
                    if key not in targets: continue
                    t_len_normal = targets[key]
                    cal_k = "calibration_" + key
                    t_len_final = targets.get(cal_k, t_len_normal)
                    
                    if final_mode: t_len_final *= factor # Total-Height-Enforcer staucht Arme/Beine
                    
                    dir_vec = cv / cl
                    new_c_pos = pts_b[p_idx] + (dir_vec * t_len_final)
                    delta_shift = new_c_pos - pts_b[c_idx]
                    pts_b[c_idx] = new_c_pos

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_shift
                        
            return pts_b


        # --- PHASE 1: RATIO-LOOP AM ANCHOR-FRAME ---
        orig_h_global = get_height_stable(ref_pts)
        global_f_scale = 1.0
        
        log_messages.append(f"\n--- TOTAL-HEIGHT-ENFORCER ---")
        if orig_h_global > 1e-5:
            for iteration in range(10):
                pts_test = build_and_log(ref_pts, global_f_scale, final_mode=True)
                test_h = get_height_stable(pts_test)
                
                if test_h < 1e-5: break
                
                diff = abs(orig_h_global - test_h)
                if diff < 0.1: break
                    
                ratio = orig_h_global / test_h
                global_f_scale *= ratio

        log_messages.append(f"Berechneter universeller Kompressions-Faktor: {global_f_scale:.6f}x")


        # --- PHASE 2: VERARBEITUNG ALLER FRAMES (mit Logging alle 10 Frames) ---
        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0: continue
            
            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())
            if pts.ndim == 3: pts = pts[0]

            # 1. MESSUNG VORHER (Nur alle 10 Frames)
            log_this_frame = (frame_idx % 10 == 0)
            if log_this_frame:
                bones_before = get_bone_lengths(pts)
                h_before = get_height_stable(pts)

            # 2. Wende den finalen Scale auf den Frame an
            pts_final = build_and_log(pts, global_f_scale, final_mode=True)

            # 3. MESSUNG NACHHER & LOG-AUSGABE
            if log_this_frame:
                bones_after = get_bone_lengths(pts_final)
                h_after = get_height_stable(pts_final)
                
                log_messages.append(f"\n--- FRAME {frame_idx} VERGLEICH (Physische 3D-Knochenlängen in NLF) ---")
                log_messages.append(f"Gesamthöhe (Y-Bounding Box) | Vorher: {h_before:.2f} -> Nachher: {h_after:.2f}")
                log_messages.append("-" * 70)
                for k in bones_before.keys():
                    log_messages.append(f"Knochen: {k.ljust(18)} | Vorher: {bones_before[k]:.2f} -> Nachher: {bones_after[k]:.2f}")

            # GROUND ANCHOR: Person auf den Boden stellen
            v_orig_feet = [pts[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
            v_new_feet = [pts_final[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts_final) and np.linalg.norm(pts_final[idx]) > 1e-5]
            if v_orig_feet and v_new_feet:
                shift = max(v_orig_feet) - max(v_new_feet)
                for j in range(len(pts_final)):
                    if np.linalg.norm(pts_final[j]) > 1e-5: pts_final[j][1] += shift

            # Output-Formatierung
            if is_tensor:
                if frame_data.dim() == 3: raw_poses[frame_idx][0] = torch.from_numpy(pts_final).to(frame_data.device)
                else: raw_poses[frame_idx] = torch.from_numpy(pts_final).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts_final.tolist()


        # --- PHASE 3: KAMERA-CONFIG BEREINIGEN ---
        try:
            config_dict = json.loads(nlf_render_config) if isinstance(nlf_render_config, str) and nlf_render_config.strip() else {}
        except Exception as e:
            config_dict = {}
            
        config_dict["anchor_scale"] = 1.0
        config_dict["scale_x_factor"] = 1.0
        clean_config_str = json.dumps(config_dict)

        return (nlf_data_retargeted, "\n".join(log_messages), clean_config_str)


class NLFProportionalRetargeterV17:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
                "scale_stance_and_head": ("BOOLEAN", {"default": False, "tooltip": "Wendet den Höhen-Korrekturfaktor auch auf Schulter-/Hüftbreite und den Kopf an."}),
                "temporal_smooth_factor": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet die Gelenke (Arme/Beine) gegen Zittern."}),
                "ground_smooth_factor": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet NUR die Auf/Ab-Bewegung (Walk-Cycle Anti-Jitter). Oft höher als Temporal."}),
            },
            "optional": {
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus dem Scaler (wird bereinigt)"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output", "nlf_render_config_clean")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V16: Total-Height-Enforcer + Getrenntes Smoothing für Gelenke & Boden."

    def process(self, video_nlf_data, calibration_data, frontal_3d_angle_tolerance, scale_stance_and_head, temporal_smooth_factor, ground_smooth_factor, nlf_render_config="{}"):
        import copy
        import numpy as np
        import math
        import torch
        import json

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V16 (SEPARATE SMOOTHING CONTROLS) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        head_val = true_3d_bones.get("head", 0.0)

        # --- STUFE 1: TÜRSTEHER (WINKEL-RADAR) ---
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'length': 0.0, 'is_frontal': False, 'has_feet': False})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else np.array(frame_data))
            if pts.ndim == 3: pts = pts[0]

            def is_val(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

            is_frontal = False
            if len(pts) >= 18:
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                if max(angle_h, angle_s) <= frontal_3d_angle_tolerance:
                    is_frontal = True

            all_frames_data.append({'length': length, 'is_frontal': is_frontal, 'has_feet': is_val(7) or is_val(8)})
            if is_frontal: frontal_indices.append(i)

        # --- STUFE 2: ANCHOR-FRAME BESTIMMEN ---
        candidates = frontal_indices if frontal_indices else list(range(len(raw_poses)))
        max_len = max([d['length'] for d in all_frames_data]) if all_frames_data else 1.0
        
        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            d = all_frames_data[idx]
            score = (1000.0 if d['has_feet'] else 0.0) + (500.0 if d['is_frontal'] else 0.0) + ((d['length'] / max_len) * 100.0)
            if score > best_score:
                best_score, best_idx = score, idx

        log_messages.append(f"-> Referenz-Frame (Anchor) für globalen Ratio-Loop: {best_idx}")

        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        ref_pts = ref_frame_data[0].cpu().numpy() if is_t and ref_frame_data.dim() == 3 else (ref_frame_data.cpu().numpy() if is_t else np.array(ref_frame_data))
        if ref_pts.ndim == 3: ref_pts = ref_pts[0]

        # --- HILFSFUNKTIONEN ---
        tree = {0:[1,2,3], 1:[4], 4:[7], 7:[10], 2:[5], 5:[8], 8:[11], 3:[6], 6:[9], 9:[12,13,14], 12:[15], 13:[16], 16:[18], 18:[20], 20:[22], 14:[17], 17:[19], 19:[21], 21:[23]}
        
        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child); desc.extend(get_all_descendants(child, tree_map))
            return desc

        def get_height_stable(p_array):
            if 12 < len(p_array) and np.linalg.norm(p_array[12]) > 1e-5:
                top_y = p_array[12][1]
            else: return 0.0
            feet_y = [p_array[idx][1] for idx in [7,8,10,11,4,5] if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5]
            return (max(feet_y) - top_y) if feet_y else 0.0

        def get_bone_lengths(pts_array):
            def dist(p1, p2):
                if p1 < len(pts_array) and p2 < len(pts_array) and np.linalg.norm(pts_array[p1]) > 1e-5 and np.linalg.norm(pts_array[p2]) > 1e-5:
                    return float(np.linalg.norm(pts_array[p2] - pts_array[p1]))
                return 0.0

            return {
                "Torso": dist(0, 12), "Kopf": dist(12, 15),
                "R_Oberschenkel": dist(2, 5), "R_Wade": dist(5, 8),
                "L_Oberschenkel": dist(1, 4), "L_Wade": dist(4, 7),
                "R_Arm": dist(17, 19), "R_Unterarm": dist(19, 21),
                "L_Arm": dist(16, 18), "L_Unterarm": dist(18, 20),
                "Schulterbreite": dist(16, 17), "Hueftbreite": dist(1, 2)
            }

        def build_and_log(pts_source, factor, final_mode=False):
            pts_b = pts_source.copy()
            
            orig_torso_curr = np.linalg.norm(pts_b[12] - pts_b[0]) if np.linalg.norm(pts_b[12]) > 1e-5 else 0.0
            missing_neck_curr = orig_torso_curr * (head_val / 100.0) / 2.0 if is_normalized else head_val / 2.0
            frame_ref_torso = orig_torso_curr + missing_neck_curr

            targets = {k: (v / 100.0 * frame_ref_torso if is_normalized else v) for k, v in true_3d_bones.items()}
            
            # 1. Torso Skalierung
            cv = pts_b[12] - pts_b[0]; cl = np.linalg.norm(cv)
            if cl > 1e-5:
                t_len = targets.get("torso", cl)
                if final_mode: t_len *= factor
                
                f_node = t_len / cl
                for p, c in [(0,3), (3,6), (6,9), (9,12)]:
                    vec = pts_b[c] - pts_b[p]; new_c = pts_b[p] + vec * f_node
                    delta = new_c - pts_b[c]; pts_b[c] += delta
                    for d in get_all_descendants(c, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5: pts_b[d] += delta

            # 2. Kopf
            if 15 < len(pts_b):
                cv = pts_b[15] - pts_b[12]; cl = np.linalg.norm(cv)
                if cl > 1e-5:
                    t_len = targets.get("head", cl * 2.0) / 2.0
                    if final_mode and scale_stance_and_head: 
                        t_len *= factor
                    
                    f_node = t_len / cl
                    delta = (pts_b[12] + (cv / cl * t_len)) - pts_b[15]
                    pts_b[15] += delta

            # 3. Operations
            ops = [('shoulder_width',12,17), ('shoulder_width',12,16), ('hip_width',0,2), ('hip_width',0,1),
                   ('r_arm',17,19), ('r_forearm',19,21), ('l_arm',16,18), ('l_forearm',18,20),
                   ('r_thigh',2,5), ('r_calf',5,8), ('l_thigh',1,4), ('l_calf',4,7)]

            for key, p_idx, c_idx in ops:
                cv = pts_b[c_idx] - pts_b[p_idx]; cl = np.linalg.norm(cv)
                if cl < 1e-5: continue

                if key in ['shoulder_width', 'hip_width']:
                    stance_target = targets.get(key, cl * 2.0) / 2.0
                    calib_key = f"calibration_{key}"
                    bone_target = targets.get(calib_key, stance_target * 2.0) / 2.0

                    if final_mode and scale_stance_and_head:
                        stance_target *= factor
                        bone_target *= factor

                    scale_xz_stance = stance_target / cl
                    pos_stance = pts_b[p_idx].copy()
                    pos_stance[0] += cv[0] * scale_xz_stance
                    pos_stance[1] += cv[1]
                    pos_stance[2] += cv[2] * scale_xz_stance
                    
                    delta_stance = pos_stance - pts_b[c_idx]
                    pts_b[c_idx] += delta_stance
                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_stance

                    scale_xz_config = bone_target / cl
                    pos_config = pts_b[p_idx].copy()
                    pos_config[0] += cv[0] * scale_xz_config
                    pos_config[1] += cv[1]
                    pos_config[2] += cv[2] * scale_xz_config
                    
                    delta_config = pos_config - pts_b[c_idx]
                    pts_b[c_idx] += delta_config

                else:
                    if key not in targets: continue
                    t_len_normal = targets[key]
                    cal_k = "calibration_" + key
                    t_len_final = targets.get(cal_k, t_len_normal)
                    
                    if final_mode: t_len_final *= factor
                    
                    dir_vec = cv / cl
                    new_c_pos = pts_b[p_idx] + (dir_vec * t_len_final)
                    delta_shift = new_c_pos - pts_b[c_idx]
                    pts_b[c_idx] = new_c_pos

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_shift
                        
            return pts_b


        # --- PHASE 1: RATIO-LOOP AM ANCHOR-FRAME ---
        orig_h_global = get_height_stable(ref_pts)
        global_f_scale = 1.0
        
        log_messages.append(f"\n--- TOTAL-HEIGHT-ENFORCER ---")
        if orig_h_global > 1e-5:
            for iteration in range(10):
                pts_test = build_and_log(ref_pts, global_f_scale, final_mode=True)
                test_h = get_height_stable(pts_test)
                
                if test_h < 1e-5: break
                
                diff = abs(orig_h_global - test_h)
                if diff < 0.1: break
                    
                ratio = orig_h_global / test_h
                global_f_scale *= ratio

        log_messages.append(f"Berechneter universeller Kompressions-Faktor: {global_f_scale:.6f}x")


        # --- PHASE 2: VERARBEITUNG ALLER FRAMES (mit getrenntem Smoothing) ---
        prev_pts = None   # Speicher für den vorherigen Frame (Gelenk-Smoothing)
        prev_shift = None # Speicher für weiche Bodenhaftung (Walk-Cycle-Fix)

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0: continue
            
            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())
            if pts.ndim == 3: pts = pts[0]

            log_this_frame = (frame_idx % 10 == 0)
            if log_this_frame:
                bones_before = get_bone_lengths(pts)
                h_before = get_height_stable(pts)

            pts_final = build_and_log(pts, global_f_scale, final_mode=True)

            # --- GROUND ANCHOR (ANTI-JITTER / WALK-CYCLE FIX mit eigenem Regler) ---
            v_orig_feet = [pts[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
            v_new_feet = [pts_final[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts_final) and np.linalg.norm(pts_final[idx]) > 1e-5]
            if v_orig_feet and v_new_feet:
                raw_shift = max(v_orig_feet) - max(v_new_feet)
                
                # Hier nutzen wir nun den neuen ground_smooth_factor!
                if ground_smooth_factor > 0.0 and prev_shift is not None:
                    shift = (prev_shift * ground_smooth_factor) + (raw_shift * (1.0 - ground_smooth_factor))
                else:
                    shift = raw_shift
                    
                prev_shift = shift # Merken für den nächsten Frame

                for j in range(len(pts_final)):
                    if np.linalg.norm(pts_final[j]) > 1e-5: pts_final[j][1] += shift


            # --- TEMPORAL SMOOTHING DER GELENKE (mit temporal_smooth_factor) ---
            if temporal_smooth_factor > 0.0:
                if prev_pts is None:
                    prev_pts = pts_final.copy()
                else:
                    for j in range(len(pts_final)):
                        if np.linalg.norm(pts_final[j]) > 1e-5 and np.linalg.norm(prev_pts[j]) > 1e-5:
                            pts_final[j] = (prev_pts[j] * temporal_smooth_factor) + (pts_final[j] * (1.0 - temporal_smooth_factor))
                        prev_pts[j] = pts_final[j].copy()


            # --- MESSUNG NACHHER & LOGGING ---
            if log_this_frame:
                bones_after = get_bone_lengths(pts_final)
                h_after = get_height_stable(pts_final)
                
                log_messages.append(f"\n--- FRAME {frame_idx} VERGLEICH (Physische 3D-Knochenlängen in NLF) ---")
                log_messages.append(f"Gesamthöhe (Y-Bounding Box) | Vorher: {h_before:.2f} -> Nachher: {h_after:.2f}")
                log_messages.append("-" * 70)
                for k in bones_before.keys():
                    log_messages.append(f"Knochen: {k.ljust(18)} | Vorher: {bones_before[k]:.2f} -> Nachher: {bones_after[k]:.2f}")

            # Output-Formatierung
            if is_tensor:
                if frame_data.dim() == 3: raw_poses[frame_idx][0] = torch.from_numpy(pts_final).to(frame_data.device)
                else: raw_poses[frame_idx] = torch.from_numpy(pts_final).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts_final.tolist()


        # --- PHASE 3: KAMERA-CONFIG BEREINIGEN ---
        try:
            config_dict = json.loads(nlf_render_config) if isinstance(nlf_render_config, str) and nlf_render_config.strip() else {}
        except Exception as e:
            config_dict = {}
            
        config_dict["anchor_scale"] = 1.0
        config_dict["scale_x_factor"] = 1.0
        clean_config_str = json.dumps(config_dict)

        return (nlf_data_retargeted, "\n".join(log_messages), clean_config_str)


class NLFProportionalRetargeterV17ex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
                "scale_stance_and_head": ("BOOLEAN", {"default": False, "tooltip": "Wendet den Höhen-Korrekturfaktor auch auf Schulter-/Hüftbreite und den Kopf an."}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V17: PELVIS-ANKER (Dein Fix!). Keine Boden-Verschiebung mehr. Kopf behält Originalgröße."

    def process(self, video_nlf_data, calibration_data, frontal_3d_angle_tolerance, scale_stance_and_head):
        import copy
        import numpy as np
        import math
        import torch

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V17 (PELVIS ANCHOR & FIXED PROPORTIONS) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages))

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        head_val = true_3d_bones.get("head", 0.0)

        # --- STUFE 1 & 2: Anchor-Frame finden (wie vorher) ---
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'length': 0.0, 'is_frontal': False, 'has_feet': False})
                continue
            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else np.array(frame_data))
            if pts.ndim == 3: pts = pts[0]
            def is_val(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5
            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0
            is_frontal = False
            if len(pts) >= 18:
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                if max(angle_h, angle_s) <= frontal_3d_angle_tolerance: is_frontal = True
            all_frames_data.append({'length': length, 'is_frontal': is_frontal, 'has_feet': is_val(7) or is_val(8)})
            if is_frontal: frontal_indices.append(i)

        candidates = frontal_indices if frontal_indices else list(range(len(raw_poses)))
        max_len = max([d['length'] for d in all_frames_data]) if all_frames_data else 1.0
        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            d = all_frames_data[idx]
            score = (1000.0 if d['has_feet'] else 0.0) + (500.0 if d['is_frontal'] else 0.0) + ((d['length'] / max_len) * 100.0)
            if score > best_score: best_score, best_idx = score, idx

        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        ref_pts = ref_frame_data[0].cpu().numpy() if is_t and ref_frame_data.dim() == 3 else (ref_frame_data.cpu().numpy() if is_t else np.array(ref_frame_data))
        if ref_pts.ndim == 3: ref_pts = ref_pts[0]

        # --- HILFSFUNKTIONEN ---
        tree = {0:[1,2,3], 1:[4], 4:[7], 7:[10], 2:[5], 5:[8], 8:[11], 3:[6], 6:[9], 9:[12,13,14], 12:[15], 13:[16], 16:[18], 18:[20], 20:[22], 14:[17], 17:[19], 19:[21], 21:[23]}
        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child); desc.extend(get_all_descendants(child, tree_map))
            return desc

        def get_height_stable(p_array):
            if 12 < len(p_array) and np.linalg.norm(p_array[12]) > 1e-5: top_y = p_array[12][1]
            else: return 0.0
            feet_y = [p_array[idx][1] for idx in [7,8,10,11,4,5] if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5]
            return (max(feet_y) - top_y) if feet_y else 0.0

        def get_bone_lengths(pts_array):
            def dist(p1, p2):
                if p1 < len(pts_array) and p2 < len(pts_array) and np.linalg.norm(pts_array[p1]) > 1e-5 and np.linalg.norm(pts_array[p2]) > 1e-5:
                    return float(np.linalg.norm(pts_array[p2] - pts_array[p1]))
                return 0.0
            return {
                "Torso": dist(0, 12), "Kopf": dist(12, 15),
                "R_Oberschenkel": dist(2, 5), "R_Wade": dist(5, 8),
                "L_Oberschenkel": dist(1, 4), "L_Wade": dist(4, 7),
                "R_Arm": dist(17, 19), "R_Unterarm": dist(19, 21),
                "L_Arm": dist(16, 18), "L_Unterarm": dist(18, 20),
                "Schulterbreite": dist(16, 17), "Hueftbreite": dist(1, 2)
            }

        def build_and_log(pts_source, factor, final_mode=False):
            pts_b = pts_source.copy()
            
            orig_torso_curr = np.linalg.norm(pts_b[12] - pts_b[0]) if np.linalg.norm(pts_b[12]) > 1e-5 else 0.0
            missing_neck_curr = orig_torso_curr * (head_val / 100.0) / 2.0 if is_normalized else head_val / 2.0
            frame_ref_torso = orig_torso_curr + missing_neck_curr
            targets = {k: (v / 100.0 * frame_ref_torso if is_normalized else v) for k, v in true_3d_bones.items()}
            
            # 1. Torso Skalierung
            cv = pts_b[12] - pts_b[0]; cl = np.linalg.norm(cv)
            if cl > 1e-5:
                t_len = targets.get("torso", cl)
                if final_mode: t_len *= factor
                f_node = t_len / cl
                for p, c in [(0,3), (3,6), (6,9), (9,12)]:
                    vec = pts_b[c] - pts_b[p]; new_c = pts_b[p] + vec * f_node
                    delta = new_c - pts_b[c]; pts_b[c] += delta
                    for d in get_all_descendants(c, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5: pts_b[d] += delta

            # 2. Kopf (USER-FIX: Behält Originalgröße, außer Toggle ist aktiv!)
            if 15 < len(pts_b):
                cv = pts_b[15] - pts_b[12]; cl = np.linalg.norm(cv)
                if cl > 1e-5:
                    t_len = cl # <--- HIER IST DER FIX! Originalgröße statt Config-Überschreibung
                    if final_mode and scale_stance_and_head: t_len *= factor
                    f_node = t_len / cl
                    delta = (pts_b[12] + (cv / cl * t_len)) - pts_b[15]
                    pts_b[15] += delta

            # 3. Operations
            ops = [('shoulder_width',12,17), ('shoulder_width',12,16), ('hip_width',0,2), ('hip_width',0,1),
                   ('r_arm',17,19), ('r_forearm',19,21), ('l_arm',16,18), ('l_forearm',18,20),
                   ('r_thigh',2,5), ('r_calf',5,8), ('l_thigh',1,4), ('l_calf',4,7)]

            for key, p_idx, c_idx in ops:
                cv = pts_b[c_idx] - pts_b[p_idx]; cl = np.linalg.norm(cv)
                if cl < 1e-5: continue

                if key in ['shoulder_width', 'hip_width']:
                    stance_target = targets.get(key, cl * 2.0) / 2.0
                    calib_key = f"calibration_{key}"
                    bone_target = targets.get(calib_key, stance_target * 2.0) / 2.0

                    if final_mode and scale_stance_and_head:
                        stance_target *= factor
                        bone_target *= factor

                    scale_xz_stance = stance_target / cl
                    pos_stance = pts_b[p_idx].copy()
                    pos_stance[0] += cv[0] * scale_xz_stance
                    pos_stance[1] += cv[1]
                    pos_stance[2] += cv[2] * scale_xz_stance
                    delta_stance = pos_stance - pts_b[c_idx]
                    pts_b[c_idx] += delta_stance
                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5: pts_b[d] += delta_stance

                    scale_xz_config = bone_target / cl
                    pos_config = pts_b[p_idx].copy()
                    pos_config[0] += cv[0] * scale_xz_config
                    pos_config[1] += cv[1]
                    pos_config[2] += cv[2] * scale_xz_config
                    delta_config = pos_config - pts_b[c_idx]
                    pts_b[c_idx] += delta_config

                else:
                    # Arme behalten Originalgröße (werden nur geschrumpft), Beine nehmen Target!
                    t_len_normal = targets.get(key, cl) 
                    if 'arm' in key: t_len_normal = cl # Arm-Fix: Keine Verzerrung durch Config!
                    
                    cal_k = "calibration_" + key
                    t_len_final = targets.get(cal_k, t_len_normal)
                    
                    if final_mode: t_len_final *= factor 
                    
                    dir_vec = cv / cl
                    new_c_pos = pts_b[p_idx] + (dir_vec * t_len_final)
                    delta_shift = new_c_pos - pts_b[c_idx]
                    pts_b[c_idx] = new_c_pos

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5: pts_b[d] += delta_shift
                        
            return pts_b


        # --- PHASE 1: RATIO-LOOP AM ANCHOR-FRAME ---
        orig_h_global = get_height_stable(ref_pts)
        global_f_scale = 1.0
        
        log_messages.append(f"\n--- TOTAL-HEIGHT-ENFORCER ---")
        if orig_h_global > 1e-5:
            for iteration in range(10):
                pts_test = build_and_log(ref_pts, global_f_scale, final_mode=True)
                test_h = get_height_stable(pts_test)
                if test_h < 1e-5: break
                diff = abs(orig_h_global - test_h)
                if diff < 0.1: break
                ratio = orig_h_global / test_h
                global_f_scale *= ratio

        log_messages.append(f"Berechneter universeller Kompressions-Faktor: {global_f_scale:.6f}x")

        # --- PHASE 2: VERARBEITUNG ALLER FRAMES ---
        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0: continue
            
            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())
            if pts.ndim == 3: pts = pts[0]

            log_this_frame = (frame_idx % 10 == 0)
            if log_this_frame:
                bones_before = get_bone_lengths(pts)
                h_before = get_height_stable(pts)

            pts_final = build_and_log(pts, global_f_scale, final_mode=True)

            if log_this_frame:
                bones_after = get_bone_lengths(pts_final)
                h_after = get_height_stable(pts_final)
                log_messages.append(f"\n--- FRAME {frame_idx} VERGLEICH ---")
                log_messages.append(f"Gesamthöhe (3D) | Vorher: {h_before:.2f} -> Nachher: {h_after:.2f}")
                log_messages.append("-" * 50)
                for k in bones_before.keys():
                    log_messages.append(f"Knochen: {k.ljust(18)} | Vorher: {bones_before[k]:.2f} -> Nachher: {bones_after[k]:.2f}")

            # PELVIS ANCHOR: DAS IST DER MAGIC FIX!
            # Wir entfernen den gesamten Ground-Anchor-Block. 
            # Das Becken (Joint 0) bleibt exakt an seiner Originalkoordinate im Raum.
            # Die Beine wachsen nach unten, der Torso schrumpft nach oben. 
            # Die Kameraperspektive bleibt absolut fehlerfrei erhalten!

            # Output-Formatierung
            if is_tensor:
                if frame_data.dim() == 3: raw_poses[frame_idx][0] = torch.from_numpy(pts_final).to(frame_data.device)
                else: raw_poses[frame_idx] = torch.from_numpy(pts_final).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts_final.tolist()

        return (nlf_data_retargeted, "\n".join(log_messages))


class NLFProportionalRetargeterV18:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "bypass": ("BOOLEAN", {"default": False, "tooltip": "Ignoriert die Node komplett und gibt Originaldaten zurück."}),
                "scale_torso": ("BOOLEAN", {"default": True, "tooltip": "Skaliert den Torso"}),
                "scale_shoulders": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Schultern"}),
                "scale_hips": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Hüften"}),
                "scale_arms": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Arme"}),
                "scale_legs": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Beine"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
                "scale_stance_and_head": ("BOOLEAN", {"default": False, "tooltip": "Wendet den Höhen-Korrekturfaktor auch auf Schulter-/Hüftbreite und den Kopf an."}),
                "temporal_smooth_factor": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet die Gelenke (Arme/Beine) gegen Zittern."}),
                "ground_smooth_factor": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet NUR die Auf/Ab-Bewegung (Walk-Cycle Anti-Jitter). Oft höher als Temporal."}),
            },
            "optional": {
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus dem Scaler (wird bereinigt)"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output", "nlf_render_config_clean")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V18: Selektive Körperteil-Skalierung (Toggles) für Close-Up-Debugging & Bypass."

    def process(self, video_nlf_data, calibration_data, bypass, scale_torso, scale_shoulders, scale_hips, scale_arms, scale_legs, frontal_3d_angle_tolerance, scale_stance_and_head, temporal_smooth_factor, ground_smooth_factor, nlf_render_config="{}"):
        import copy
        import numpy as np
        import math
        import torch
        import json

        # --- BYPASS ---
        if bypass:
            return (video_nlf_data, "=== NLF PROPORTIONAL RETARGETER V18 ===\nBYPASS AKTIVIERT: Keine Daten verändert.", nlf_render_config if nlf_render_config else "{}")

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V18 (SELECTIVE SCALING) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        head_val = true_3d_bones.get("head", 0.0)

        # Toggles Dictionary für saubere Übergabe an build_and_log
        toggles = {
            "scale_torso": scale_torso,
            "scale_shoulders": scale_shoulders,
            "scale_hips": scale_hips,
            "scale_arms": scale_arms,
            "scale_legs": scale_legs
        }

        # --- STUFE 1: TÜRSTEHER (WINKEL-RADAR) ---
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'length': 0.0, 'is_frontal': False, 'has_feet': False})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else np.array(frame_data))
            if pts.ndim == 3: pts = pts[0]

            def is_val(idx): return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

            is_frontal = False
            if len(pts) >= 18:
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                if max(angle_h, angle_s) <= frontal_3d_angle_tolerance:
                    is_frontal = True

            all_frames_data.append({'length': length, 'is_frontal': is_frontal, 'has_feet': is_val(7) or is_val(8)})
            if is_frontal: frontal_indices.append(i)

        # --- STUFE 2: ANCHOR-FRAME BESTIMMEN ---
        candidates = frontal_indices if frontal_indices else list(range(len(raw_poses)))
        max_len = max([d['length'] for d in all_frames_data]) if all_frames_data else 1.0
        
        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            d = all_frames_data[idx]
            score = (1000.0 if d['has_feet'] else 0.0) + (500.0 if d['is_frontal'] else 0.0) + ((d['length'] / max_len) * 100.0)
            if score > best_score:
                best_score, best_idx = score, idx

        log_messages.append(f"-> Referenz-Frame (Anchor) für globalen Ratio-Loop: {best_idx}")

        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        ref_pts = ref_frame_data[0].cpu().numpy() if is_t and ref_frame_data.dim() == 3 else (ref_frame_data.cpu().numpy() if is_t else np.array(ref_frame_data))
        if ref_pts.ndim == 3: ref_pts = ref_pts[0]

        # --- HILFSFUNKTIONEN ---
        tree = {0:[1,2,3], 1:[4], 4:[7], 7:[10], 2:[5], 5:[8], 8:[11], 3:[6], 6:[9], 9:[12,13,14], 12:[15], 13:[16], 16:[18], 18:[20], 20:[22], 14:[17], 17:[19], 19:[21], 21:[23]}
        
        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child); desc.extend(get_all_descendants(child, tree_map))
            return desc

        def get_height_stable(p_array):
            if 12 < len(p_array) and np.linalg.norm(p_array[12]) > 1e-5:
                top_y = p_array[12][1]
            else: return 0.0
            feet_y = [p_array[idx][1] for idx in [7,8,10,11,4,5] if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5]
            return (max(feet_y) - top_y) if feet_y else 0.0

        def get_bone_lengths(pts_array):
            def dist(p1, p2):
                if p1 < len(pts_array) and p2 < len(pts_array) and np.linalg.norm(pts_array[p1]) > 1e-5 and np.linalg.norm(pts_array[p2]) > 1e-5:
                    return float(np.linalg.norm(pts_array[p2] - pts_array[p1]))
                return 0.0

            return {
                "Torso": dist(0, 12), "Kopf": dist(12, 15),
                "R_Oberschenkel": dist(2, 5), "R_Wade": dist(5, 8),
                "L_Oberschenkel": dist(1, 4), "L_Wade": dist(4, 7),
                "R_Arm": dist(17, 19), "R_Unterarm": dist(19, 21),
                "L_Arm": dist(16, 18), "L_Unterarm": dist(18, 20),
                "Schulterbreite": dist(16, 17), "Hueftbreite": dist(1, 2)
            }

        def build_and_log(pts_source, factor, tgls, final_mode=False, force_all=False):
            pts_b = pts_source.copy()
            
            orig_torso_curr = np.linalg.norm(pts_b[12] - pts_b[0]) if np.linalg.norm(pts_b[12]) > 1e-5 else 0.0
            missing_neck_curr = orig_torso_curr * (head_val / 100.0) / 2.0 if is_normalized else head_val / 2.0
            frame_ref_torso = orig_torso_curr + missing_neck_curr

            targets = {k: (v / 100.0 * frame_ref_torso if is_normalized else v) for k, v in true_3d_bones.items()}
            
            # 1. Torso Skalierung (Mit Toggle / Force Check)
            if force_all or tgls.get("scale_torso", True):
                cv = pts_b[12] - pts_b[0]; cl = np.linalg.norm(cv)
                if cl > 1e-5:
                    t_len = targets.get("torso", cl)
                    if final_mode: t_len *= factor
                    
                    f_node = t_len / cl
                    for p, c in [(0,3), (3,6), (6,9), (9,12)]:
                        vec = pts_b[c] - pts_b[p]; new_c = pts_b[p] + vec * f_node
                        delta = new_c - pts_b[c]; pts_b[c] += delta
                        for d in get_all_descendants(c, tree):
                            if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5: pts_b[d] += delta

            # 2. Kopf (bleibt unverändert, skaliert immer mit falls vorhanden)
            if 15 < len(pts_b):
                cv = pts_b[15] - pts_b[12]; cl = np.linalg.norm(cv)
                if cl > 1e-5:
                    t_len = targets.get("head", cl * 2.0) / 2.0
                    if final_mode and scale_stance_and_head: 
                        t_len *= factor
                    
                    f_node = t_len / cl
                    delta = (pts_b[12] + (cv / cl * t_len)) - pts_b[15]
                    pts_b[15] += delta

            # 3. Operations (Arme, Beine, Schultern, Hüften)
            ops = [('shoulder_width',12,17), ('shoulder_width',12,16), ('hip_width',0,2), ('hip_width',0,1),
                   ('r_arm',17,19), ('r_forearm',19,21), ('l_arm',16,18), ('l_forearm',18,20),
                   ('r_thigh',2,5), ('r_calf',5,8), ('l_thigh',1,4), ('l_calf',4,7)]

            for key, p_idx, c_idx in ops:
                # Toggle Check
                is_allowed = force_all
                if not is_allowed:
                    if 'shoulder' in key and tgls.get('scale_shoulders', True): is_allowed = True
                    elif 'hip' in key and tgls.get('scale_hips', True): is_allowed = True
                    elif 'arm' in key and tgls.get('scale_arms', True): is_allowed = True
                    elif ('thigh' in key or 'calf' in key) and tgls.get('scale_legs', True): is_allowed = True
                
                if not is_allowed:
                    continue

                cv = pts_b[c_idx] - pts_b[p_idx]; cl = np.linalg.norm(cv)
                if cl < 1e-5: continue

                if key in ['shoulder_width', 'hip_width']:
                    stance_target = targets.get(key, cl * 2.0) / 2.0
                    calib_key = f"calibration_{key}"
                    bone_target = targets.get(calib_key, stance_target * 2.0) / 2.0

                    if final_mode and scale_stance_and_head:
                        stance_target *= factor
                        bone_target *= factor

                    scale_xz_stance = stance_target / cl
                    pos_stance = pts_b[p_idx].copy()
                    pos_stance[0] += cv[0] * scale_xz_stance
                    pos_stance[1] += cv[1]
                    pos_stance[2] += cv[2] * scale_xz_stance
                    
                    delta_stance = pos_stance - pts_b[c_idx]
                    pts_b[c_idx] += delta_stance
                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_stance

                    scale_xz_config = bone_target / cl
                    pos_config = pts_b[p_idx].copy()
                    pos_config[0] += cv[0] * scale_xz_config
                    pos_config[1] += cv[1]
                    pos_config[2] += cv[2] * scale_xz_config
                    
                    delta_config = pos_config - pts_b[c_idx]
                    pts_b[c_idx] += delta_config

                else:
                    if key not in targets: continue
                    t_len_normal = targets[key]
                    cal_k = "calibration_" + key
                    t_len_final = targets.get(cal_k, t_len_normal)
                    
                    if final_mode: t_len_final *= factor
                    
                    dir_vec = cv / cl
                    new_c_pos = pts_b[p_idx] + (dir_vec * t_len_final)
                    delta_shift = new_c_pos - pts_b[c_idx]
                    pts_b[c_idx] = new_c_pos

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_shift
                        
            return pts_b


        # --- PHASE 1: RATIO-LOOP AM ANCHOR-FRAME ---
        orig_h_global = get_height_stable(ref_pts)
        global_f_scale = 1.0
        
        log_messages.append(f"\n--- TOTAL-HEIGHT-ENFORCER ---")
        if orig_h_global > 1e-5:
            for iteration in range(10):
                # force_all=True: Hier MÜSSEN wir alles berechnen, um den physischen Ratio-Faktor korrekt zu ermitteln!
                pts_test = build_and_log(ref_pts, global_f_scale, toggles, final_mode=True, force_all=True)
                test_h = get_height_stable(pts_test)
                
                if test_h < 1e-5: break
                
                diff = abs(orig_h_global - test_h)
                if diff < 0.1: break
                    
                ratio = orig_h_global / test_h
                global_f_scale *= ratio

        log_messages.append(f"Berechneter universeller Kompressions-Faktor: {global_f_scale:.6f}x")


        # --- PHASE 2: VERARBEITUNG ALLER FRAMES (mit Toggles und Smoothing) ---
        prev_pts = None   
        prev_shift = None 

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0: continue
            
            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())
            if pts.ndim == 3: pts = pts[0]

            log_this_frame = (frame_idx % 10 == 0)
            if log_this_frame:
                bones_before = get_bone_lengths(pts)
                h_before = get_height_stable(pts)

            # force_all=False: Hier wirken nun deine UI-Toggles (z.B. Beine ignorieren)
            pts_final = build_and_log(pts, global_f_scale, toggles, final_mode=True, force_all=False)

            # --- GROUND ANCHOR (ANTI-JITTER / WALK-CYCLE FIX) ---
            v_orig_feet = [pts[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5]
            v_new_feet = [pts_final[idx][1] for idx in [7,8,10,11,4,5] if idx < len(pts_final) and np.linalg.norm(pts_final[idx]) > 1e-5]
            if v_orig_feet and v_new_feet:
                raw_shift = max(v_orig_feet) - max(v_new_feet)
                
                if ground_smooth_factor > 0.0 and prev_shift is not None:
                    shift = (prev_shift * ground_smooth_factor) + (raw_shift * (1.0 - ground_smooth_factor))
                else:
                    shift = raw_shift
                    
                prev_shift = shift 

                for j in range(len(pts_final)):
                    if np.linalg.norm(pts_final[j]) > 1e-5: pts_final[j][1] += shift

            # --- TEMPORAL SMOOTHING DER GELENKE ---
            if temporal_smooth_factor > 0.0:
                if prev_pts is None:
                    prev_pts = pts_final.copy()
                else:
                    for j in range(len(pts_final)):
                        if np.linalg.norm(pts_final[j]) > 1e-5 and np.linalg.norm(prev_pts[j]) > 1e-5:
                            pts_final[j] = (prev_pts[j] * temporal_smooth_factor) + (pts_final[j] * (1.0 - temporal_smooth_factor))
                        prev_pts[j] = pts_final[j].copy()

            # --- MESSUNG NACHHER & LOGGING ---
            if log_this_frame:
                bones_after = get_bone_lengths(pts_final)
                h_after = get_height_stable(pts_final)
                
                log_messages.append(f"\n--- FRAME {frame_idx} VERGLEICH (Physische 3D-Knochenlängen in NLF) ---")
                log_messages.append(f"Gesamthöhe (Y-Bounding Box) | Vorher: {h_before:.2f} -> Nachher: {h_after:.2f}")
                log_messages.append("-" * 70)
                for k in bones_before.keys():
                    log_messages.append(f"Knochen: {k.ljust(18)} | Vorher: {bones_before[k]:.2f} -> Nachher: {bones_after[k]:.2f}")

            # Output-Formatierung
            if is_tensor:
                if frame_data.dim() == 3: raw_poses[frame_idx][0] = torch.from_numpy(pts_final).to(frame_data.device)
                else: raw_poses[frame_idx] = torch.from_numpy(pts_final).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts_final.tolist()

        # --- PHASE 3: KAMERA-CONFIG BEREINIGEN ---
        try:
            config_dict = json.loads(nlf_render_config) if isinstance(nlf_render_config, str) and nlf_render_config.strip() else {}
        except Exception as e:
            config_dict = {}
            
        config_dict["anchor_scale"] = 1.0
        config_dict["scale_x_factor"] = 1.0
        clean_config_str = json.dumps(config_dict)

        return (nlf_data_retargeted, "\n".join(log_messages), clean_config_str)


class NLFProportionalRetargeterV181:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "bypass": ("BOOLEAN", {"default": False, "tooltip": "Ignoriert die Node komplett und gibt Originaldaten zurück."}),
                "scale_torso": ("BOOLEAN", {"default": True, "tooltip": "Skaliert den Torso"}),
                "scale_shoulders": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Schultern"}),
                "scale_hips": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Hüften"}),
                "scale_arms": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Arme"}),
                "scale_legs": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Beine"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
                "scale_stance_and_head": ("BOOLEAN", {"default": False, "tooltip": "Wendet den Höhen-Korrekturfaktor auch auf Schulter-/Hüftbreite und den Kopf an."}),
                "temporal_smooth_factor": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet die Gelenke (Arme/Beine) gegen Zittern."}),
                "ground_smooth_factor": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet NUR die Auf/Ab-Bewegung (Walk-Cycle Anti-Jitter). Oft höher als Temporal."}),

                "leg_height_guard": ("BOOLEAN", {"default": True, "tooltip": "V18.1 Safety: verhindert, dass Leg Scaling die Gesamthöhe/Schulterhöhe in einzelnen Frames sprengt."}),
                "leg_height_guard_tolerance": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.25, "step": 0.005, "tooltip": "Relative Toleranz. 0.025 = 2.5 Prozent Abweichung zur No-Legs-Baseline erlaubt."}),
                "leg_height_guard_min_factor": ("FLOAT", {"default": 0.70, "min": 0.30, "max": 1.00, "step": 0.01, "tooltip": "Minimaler lokaler Bein-Kompressionsfaktor bei problematischen Frames."}),
                "leg_height_guard_max_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 1.50, "step": 0.01, "tooltip": "Maximaler lokaler Bein-Faktor. Default 1.0 bedeutet: Guard darf nur komprimieren, nicht verlängern."}),
                "leg_height_guard_smooth": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "Glättet den lokalen Bein-Guard-Faktor. Niedrig lassen, damit Ausreißer schnell abgefangen werden."}),
            },
            "optional": {
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus dem Scaler (wird bereinigt)"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output", "nlf_render_config_clean")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V18.1: Selektive Skalierung + Leg Height Guard gegen Bein-bedingte Höhen-Sprünge."

    def process(
        self,
        video_nlf_data,
        calibration_data,
        bypass,
        scale_torso,
        scale_shoulders,
        scale_hips,
        scale_arms,
        scale_legs,
        frontal_3d_angle_tolerance,
        scale_stance_and_head,
        temporal_smooth_factor,
        ground_smooth_factor,
        leg_height_guard,
        leg_height_guard_tolerance,
        leg_height_guard_min_factor,
        leg_height_guard_max_factor,
        leg_height_guard_smooth,
        nlf_render_config="{}"
    ):
        import copy
        import numpy as np
        import math
        import torch
        import json

        # --- BYPASS ---
        if bypass:
            return (
                video_nlf_data,
                "=== NLF PROPORTIONAL RETARGETER V18.1 ===\nBYPASS AKTIVIERT: Keine Daten verändert.",
                nlf_render_config if nlf_render_config else "{}"
            )

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V18.1 (SELECTIVE SCALING + LEG HEIGHT GUARD) ==="]

        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)

        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        head_val = true_3d_bones.get("head", 0.0)

        # Safety: clamp user inputs to sane runtime values.
        leg_height_guard_tolerance = max(0.0, float(leg_height_guard_tolerance))
        leg_height_guard_min_factor = float(np.clip(leg_height_guard_min_factor, 0.30, 1.00))
        leg_height_guard_max_factor = float(np.clip(leg_height_guard_max_factor, 1.00, 1.50))
        leg_height_guard_smooth = float(np.clip(leg_height_guard_smooth, 0.0, 0.95))

        # Toggles Dictionary für saubere Übergabe an build_and_log
        toggles = {
            "scale_torso": scale_torso,
            "scale_shoulders": scale_shoulders,
            "scale_hips": scale_hips,
            "scale_arms": scale_arms,
            "scale_legs": scale_legs
        }

        log_messages.append(
            f"Leg Height Guard: {'AKTIV' if leg_height_guard and scale_legs else 'INAKTIV'} | "
            f"Tolerance: {leg_height_guard_tolerance * 100.0:.2f}% | "
            f"Clamp: [{leg_height_guard_min_factor:.3f}, {leg_height_guard_max_factor:.3f}] | "
            f"Smooth: {leg_height_guard_smooth:.2f}"
        )

        # --- STUFE 1: TÜRSTEHER (WINKEL-RADAR) ---
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'length': 0.0, 'is_frontal': False, 'has_feet': False})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else np.array(frame_data))
            if pts.ndim == 3:
                pts = pts[0]

            def is_val(idx):
                return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

            is_frontal = False
            if len(pts) >= 18:
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                if max(angle_h, angle_s) <= frontal_3d_angle_tolerance:
                    is_frontal = True

            all_frames_data.append({'length': length, 'is_frontal': is_frontal, 'has_feet': is_val(7) or is_val(8)})
            if is_frontal:
                frontal_indices.append(i)

        # --- STUFE 2: ANCHOR-FRAME BESTIMMEN ---
        candidates = frontal_indices if frontal_indices else list(range(len(raw_poses)))
        max_len = max([d['length'] for d in all_frames_data]) if all_frames_data else 1.0

        if not candidates:
            log_messages.append("FEHLER: Keine verwertbaren Frames gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            d = all_frames_data[idx]
            score = (1000.0 if d['has_feet'] else 0.0) + (500.0 if d['is_frontal'] else 0.0) + ((d['length'] / max_len) * 100.0 if max_len > 1e-5 else 0.0)
            if score > best_score:
                best_score, best_idx = score, idx

        log_messages.append(f"-> Referenz-Frame (Anchor) für globalen Ratio-Loop: {best_idx}")

        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        ref_pts = ref_frame_data[0].cpu().numpy() if is_t and ref_frame_data.dim() == 3 else (ref_frame_data.cpu().numpy() if is_t else np.array(ref_frame_data))
        if ref_pts.ndim == 3:
            ref_pts = ref_pts[0]

        # --- HILFSFUNKTIONEN ---
        tree = {
            0: [1, 2, 3],
            1: [4],
            4: [7],
            7: [10],
            2: [5],
            5: [8],
            8: [11],
            3: [6],
            6: [9],
            9: [12, 13, 14],
            12: [15],
            13: [16],
            16: [18],
            18: [20],
            20: [22],
            14: [17],
            17: [19],
            19: [21],
            21: [23]
        }

        def is_valid_point(p_array, idx):
            return idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5

        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child)
                    desc.extend(get_all_descendants(child, tree_map))
            return desc

        def get_height_stable(p_array):
            if 12 < len(p_array) and np.linalg.norm(p_array[12]) > 1e-5:
                top_y = p_array[12][1]
            else:
                return 0.0

            feet_y = [
                p_array[idx][1]
                for idx in [7, 8, 10, 11, 4, 5]
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]

            return (max(feet_y) - top_y) if feet_y else 0.0

        def get_avg_y(p_array, indices):
            vals = [
                p_array[idx][1]
                for idx in indices
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]
            return float(sum(vals) / len(vals)) if vals else 0.0

        def get_foot_anchor_y(p_array):
            vals = [
                p_array[idx][1]
                for idx in [7, 8, 10, 11, 4, 5]
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]
            return float(max(vals)) if vals else 0.0

        def get_bone_lengths(pts_array):
            def dist(p1, p2):
                if (
                    p1 < len(pts_array)
                    and p2 < len(pts_array)
                    and np.linalg.norm(pts_array[p1]) > 1e-5
                    and np.linalg.norm(pts_array[p2]) > 1e-5
                ):
                    return float(np.linalg.norm(pts_array[p2] - pts_array[p1]))
                return 0.0

            return {
                "Torso": dist(0, 12),
                "Kopf": dist(12, 15),
                "R_Oberschenkel": dist(2, 5),
                "R_Wade": dist(5, 8),
                "L_Oberschenkel": dist(1, 4),
                "L_Wade": dist(4, 7),
                "R_Arm": dist(17, 19),
                "R_Unterarm": dist(19, 21),
                "L_Arm": dist(16, 18),
                "L_Unterarm": dist(18, 20),
                "Schulterbreite": dist(16, 17),
                "Hueftbreite": dist(1, 2)
            }

        def build_and_log(pts_source, factor, tgls, final_mode=False, force_all=False, leg_height_factor=1.0):
            pts_b = pts_source.copy()

            if len(pts_b) <= 12 or not is_valid_point(pts_b, 0) or not is_valid_point(pts_b, 12):
                return pts_b

            orig_torso_curr = np.linalg.norm(pts_b[12] - pts_b[0]) if np.linalg.norm(pts_b[12]) > 1e-5 else 0.0
            missing_neck_curr = orig_torso_curr * (head_val / 100.0) / 2.0 if is_normalized else head_val / 2.0
            frame_ref_torso = orig_torso_curr + missing_neck_curr

            targets = {
                k: (v / 100.0 * frame_ref_torso if is_normalized else v)
                for k, v in true_3d_bones.items()
            }

            # 1. Torso Skalierung
            if force_all or tgls.get("scale_torso", True):
                if len(pts_b) > 12 and is_valid_point(pts_b, 0) and is_valid_point(pts_b, 12):
                    cv = pts_b[12] - pts_b[0]
                    cl = np.linalg.norm(cv)

                    if cl > 1e-5:
                        t_len = targets.get("torso", cl)
                        if final_mode:
                            t_len *= factor

                        f_node = t_len / cl

                        for p, c in [(0, 3), (3, 6), (6, 9), (9, 12)]:
                            if p >= len(pts_b) or c >= len(pts_b):
                                continue
                            if not is_valid_point(pts_b, p) or not is_valid_point(pts_b, c):
                                continue

                            vec = pts_b[c] - pts_b[p]
                            new_c = pts_b[p] + vec * f_node
                            delta = new_c - pts_b[c]
                            pts_b[c] += delta

                            for d in get_all_descendants(c, tree):
                                if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                                    pts_b[d] += delta

            # 2. Kopf
            if 15 < len(pts_b) and is_valid_point(pts_b, 12) and is_valid_point(pts_b, 15):
                cv = pts_b[15] - pts_b[12]
                cl = np.linalg.norm(cv)

                if cl > 1e-5:
                    t_len = targets.get("head", cl * 2.0) / 2.0
                    if final_mode and scale_stance_and_head:
                        t_len *= factor

                    delta = (pts_b[12] + (cv / cl * t_len)) - pts_b[15]
                    pts_b[15] += delta

            # 3. Operations: Arme, Beine, Schultern, Hüften
            ops = [
                ('shoulder_width', 12, 17),
                ('shoulder_width', 12, 16),
                ('hip_width', 0, 2),
                ('hip_width', 0, 1),
                ('r_arm', 17, 19),
                ('r_forearm', 19, 21),
                ('l_arm', 16, 18),
                ('l_forearm', 18, 20),
                ('r_thigh', 2, 5),
                ('r_calf', 5, 8),
                ('l_thigh', 1, 4),
                ('l_calf', 4, 7)
            ]

            for key, p_idx, c_idx in ops:
                if p_idx >= len(pts_b) or c_idx >= len(pts_b):
                    continue

                if not is_valid_point(pts_b, p_idx) or not is_valid_point(pts_b, c_idx):
                    continue

                # Toggle Check
                is_allowed = force_all

                if not is_allowed:
                    if 'shoulder' in key and tgls.get('scale_shoulders', True):
                        is_allowed = True
                    elif 'hip' in key and tgls.get('scale_hips', True):
                        is_allowed = True
                    elif 'arm' in key and tgls.get('scale_arms', True):
                        is_allowed = True
                    elif ('thigh' in key or 'calf' in key) and tgls.get('scale_legs', True):
                        is_allowed = True

                if not is_allowed:
                    continue

                cv = pts_b[c_idx] - pts_b[p_idx]
                cl = np.linalg.norm(cv)

                if cl < 1e-5:
                    continue

                if key in ['shoulder_width', 'hip_width']:
                    stance_target = targets.get(key, cl * 2.0) / 2.0
                    calib_key = f"calibration_{key}"
                    bone_target = targets.get(calib_key, stance_target * 2.0) / 2.0

                    if final_mode and scale_stance_and_head:
                        stance_target *= factor
                        bone_target *= factor

                    scale_xz_stance = stance_target / cl
                    pos_stance = pts_b[p_idx].copy()
                    pos_stance[0] += cv[0] * scale_xz_stance
                    pos_stance[1] += cv[1]
                    pos_stance[2] += cv[2] * scale_xz_stance

                    delta_stance = pos_stance - pts_b[c_idx]
                    pts_b[c_idx] += delta_stance

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_stance

                    scale_xz_config = bone_target / cl
                    pos_config = pts_b[p_idx].copy()
                    pos_config[0] += cv[0] * scale_xz_config
                    pos_config[1] += cv[1]
                    pos_config[2] += cv[2] * scale_xz_config

                    delta_config = pos_config - pts_b[c_idx]
                    pts_b[c_idx] += delta_config

                else:
                    if key not in targets:
                        continue

                    t_len_normal = targets[key]
                    cal_k = "calibration_" + key
                    t_len_final = targets.get(cal_k, t_len_normal)

                    if final_mode:
                        t_len_final *= factor

                    # V18.1: lokaler Height-Guard wirkt ausschließlich auf Beinsegmente.
                    if final_mode and ('thigh' in key or 'calf' in key):
                        t_len_final *= leg_height_factor

                    dir_vec = cv / cl
                    new_c_pos = pts_b[p_idx] + (dir_vec * t_len_final)
                    delta_shift = new_c_pos - pts_b[c_idx]
                    pts_b[c_idx] = new_c_pos

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_shift

            return pts_b

        # --- PHASE 1: RATIO-LOOP AM ANCHOR-FRAME ---
        orig_h_global = get_height_stable(ref_pts)
        global_f_scale = 1.0

        log_messages.append(f"\n--- TOTAL-HEIGHT-ENFORCER ---")
        if orig_h_global > 1e-5:
            for iteration in range(10):
                # force_all=True: Hier wird alles berechnet, um den physischen Ratio-Faktor zu ermitteln.
                pts_test = build_and_log(
                    ref_pts,
                    global_f_scale,
                    toggles,
                    final_mode=True,
                    force_all=True,
                    leg_height_factor=1.0
                )
                test_h = get_height_stable(pts_test)

                if test_h < 1e-5:
                    break

                diff = abs(orig_h_global - test_h)
                if diff < 0.1:
                    break

                ratio = orig_h_global / test_h
                global_f_scale *= ratio

        log_messages.append(f"Berechneter universeller Kompressions-Faktor: {global_f_scale:.6f}x")

        # --- PHASE 2: VERARBEITUNG ALLER FRAMES ---
        prev_pts = None
        prev_shift = None
        prev_leg_guard_factor = 1.0
        guard_events = []

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0:
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())
            if pts.ndim == 3:
                pts = pts[0]

            log_this_frame = (frame_idx % 10 == 0)

            bones_before = None
            h_before = get_height_stable(pts)
            shoulder_y_before = get_avg_y(pts, [16, 17])
            hip_y_before = get_avg_y(pts, [1, 2])
            foot_y_before = get_foot_anchor_y(pts)

            if log_this_frame:
                bones_before = get_bone_lengths(pts)

            # --- V18.1 LEG HEIGHT GUARD ---
            guard_used = False
            guard_smoothing_bypassed = False
            guard_iterations = 0
            guard_target_h = 0.0
            guard_raw_h = 0.0
            guard_final_pre_ground_h = 0.0
            guard_unsmoothed_factor = 1.0
            guard_local_factor = 1.0
            guard_raw_rel_error = 0.0
            guard_final_rel_error = 0.0

            if leg_height_guard and scale_legs:
                no_leg_toggles = toggles.copy()
                no_leg_toggles["scale_legs"] = False

                # Baseline: Was würde genau dieser Frame tun, wenn alles außer Leg Scaling aktiv wäre?
                pts_no_legs = build_and_log(
                    pts,
                    global_f_scale,
                    no_leg_toggles,
                    final_mode=True,
                    force_all=False,
                    leg_height_factor=1.0
                )
                guard_target_h = get_height_stable(pts_no_legs)

                # Raw: normales Ergebnis mit voller Bein-Skalierung.
                pts_raw_legs = build_and_log(
                    pts,
                    global_f_scale,
                    toggles,
                    final_mode=True,
                    force_all=False,
                    leg_height_factor=1.0
                )
                guard_raw_h = get_height_stable(pts_raw_legs)

                if guard_target_h <= 1e-5:
                    guard_target_h = h_before

                if guard_target_h > 1e-5 and guard_raw_h > 1e-5:
                    guard_raw_rel_error = (guard_raw_h - guard_target_h) / guard_target_h

                    if abs(guard_raw_rel_error) > leg_height_guard_tolerance:
                        guard_used = True

                        candidate_factor = 1.0
                        pts_candidate = pts_raw_legs
                        h_candidate = guard_raw_h

                        # Kleiner lokaler Ratio-Loop nur für Beinlängen.
                        # Wichtig: Dieser Faktor wirkt NICHT auf Torso/Arme/Kopf.
                        for guard_iterations in range(1, 7):
                            if h_candidate <= 1e-5:
                                break

                            ratio = guard_target_h / h_candidate
                            candidate_factor *= ratio
                            candidate_factor = float(np.clip(candidate_factor, leg_height_guard_min_factor, leg_height_guard_max_factor))

                            pts_candidate = build_and_log(
                                pts,
                                global_f_scale,
                                toggles,
                                final_mode=True,
                                force_all=False,
                                leg_height_factor=candidate_factor
                            )
                            h_candidate = get_height_stable(pts_candidate)

                            rel_err_candidate = abs(h_candidate - guard_target_h) / guard_target_h
                            if rel_err_candidate <= leg_height_guard_tolerance:
                                break

                        guard_unsmoothed_factor = candidate_factor

                        if leg_height_guard_smooth > 0.0:
                            smoothed_factor = (prev_leg_guard_factor * leg_height_guard_smooth) + (guard_unsmoothed_factor * (1.0 - leg_height_guard_smooth))
                            smoothed_factor = float(np.clip(smoothed_factor, leg_height_guard_min_factor, leg_height_guard_max_factor))

                            pts_smoothed_candidate = build_and_log(
                                pts,
                                global_f_scale,
                                toggles,
                                final_mode=True,
                                force_all=False,
                                leg_height_factor=smoothed_factor
                            )
                            h_smoothed_candidate = get_height_stable(pts_smoothed_candidate)

                            # Safety: Wenn Smoothing den Ausreißer nicht schnell genug einfängt,
                            # wird für diesen Frame der ungesmoothete Faktor benutzt.
                            rel_err_smoothed = abs(h_smoothed_candidate - guard_target_h) / guard_target_h
                            hard_limit = max(leg_height_guard_tolerance * 2.0, 0.04)

                            if rel_err_smoothed > hard_limit:
                                guard_smoothing_bypassed = True
                                guard_local_factor = guard_unsmoothed_factor
                                pts_final = pts_candidate
                                guard_final_pre_ground_h = h_candidate
                            else:
                                guard_local_factor = smoothed_factor
                                pts_final = pts_smoothed_candidate
                                guard_final_pre_ground_h = h_smoothed_candidate
                        else:
                            guard_local_factor = guard_unsmoothed_factor
                            pts_final = pts_candidate
                            guard_final_pre_ground_h = h_candidate

                        prev_leg_guard_factor = guard_local_factor
                        guard_final_rel_error = (guard_final_pre_ground_h - guard_target_h) / guard_target_h if guard_target_h > 1e-5 else 0.0

                        guard_events.append({
                            "frame": frame_idx,
                            "target_h": guard_target_h,
                            "raw_h": guard_raw_h,
                            "final_h": guard_final_pre_ground_h,
                            "raw_error": guard_raw_rel_error,
                            "final_error": guard_final_rel_error,
                            "factor": guard_local_factor,
                            "unsmoothed_factor": guard_unsmoothed_factor,
                            "iterations": guard_iterations,
                            "smoothing_bypassed": guard_smoothing_bypassed
                        })

                    else:
                        pts_final = pts_raw_legs

                        if leg_height_guard_smooth > 0.0:
                            prev_leg_guard_factor = (prev_leg_guard_factor * leg_height_guard_smooth) + (1.0 * (1.0 - leg_height_guard_smooth))
                        else:
                            prev_leg_guard_factor = 1.0

                        guard_final_pre_ground_h = guard_raw_h
                else:
                    pts_final = pts_raw_legs
                    guard_final_pre_ground_h = guard_raw_h

            else:
                pts_final = build_and_log(
                    pts,
                    global_f_scale,
                    toggles,
                    final_mode=True,
                    force_all=False,
                    leg_height_factor=1.0
                )
                guard_final_pre_ground_h = get_height_stable(pts_final)

            # --- GROUND ANCHOR (ANTI-JITTER / WALK-CYCLE FIX) ---
            raw_shift = None
            shift = None

            v_orig_feet = [
                pts[idx][1]
                for idx in [7, 8, 10, 11, 4, 5]
                if idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5
            ]
            v_new_feet = [
                pts_final[idx][1]
                for idx in [7, 8, 10, 11, 4, 5]
                if idx < len(pts_final) and np.linalg.norm(pts_final[idx]) > 1e-5
            ]

            if v_orig_feet and v_new_feet:
                raw_shift = max(v_orig_feet) - max(v_new_feet)

                if ground_smooth_factor > 0.0 and prev_shift is not None:
                    shift = (prev_shift * ground_smooth_factor) + (raw_shift * (1.0 - ground_smooth_factor))
                else:
                    shift = raw_shift

                prev_shift = shift

                for j in range(len(pts_final)):
                    if np.linalg.norm(pts_final[j]) > 1e-5:
                        pts_final[j][1] += shift

            # --- TEMPORAL SMOOTHING DER GELENKE ---
            if temporal_smooth_factor > 0.0:
                if prev_pts is None:
                    prev_pts = pts_final.copy()
                else:
                    for j in range(len(pts_final)):
                        if j < len(prev_pts) and np.linalg.norm(pts_final[j]) > 1e-5 and np.linalg.norm(prev_pts[j]) > 1e-5:
                            pts_final[j] = (prev_pts[j] * temporal_smooth_factor) + (pts_final[j] * (1.0 - temporal_smooth_factor))
                        if j < len(prev_pts):
                            prev_pts[j] = pts_final[j].copy()

            # --- MESSUNG NACHHER & LOGGING ---
            if log_this_frame:
                bones_after = get_bone_lengths(pts_final)
                h_after = get_height_stable(pts_final)
                shoulder_y_after = get_avg_y(pts_final, [16, 17])
                hip_y_after = get_avg_y(pts_final, [1, 2])
                foot_y_after = get_foot_anchor_y(pts_final)

                log_messages.append(f"\n--- FRAME {frame_idx} VERGLEICH (Physische 3D-Knochenlängen in NLF) ---")
                log_messages.append(f"Gesamthöhe (Y-Bounding Box) | Vorher: {h_before:.2f} -> Nachher: {h_after:.2f}")
                log_messages.append(f"Schulter-Y Ø             | Vorher: {shoulder_y_before:.2f} -> Nachher: {shoulder_y_after:.2f} | Delta: {(shoulder_y_after - shoulder_y_before):+.2f}")
                log_messages.append(f"Hüft-Y Ø                 | Vorher: {hip_y_before:.2f} -> Nachher: {hip_y_after:.2f} | Delta: {(hip_y_after - hip_y_before):+.2f}")
                log_messages.append(f"Fuß-Anker-Y              | Vorher: {foot_y_before:.2f} -> Nachher: {foot_y_after:.2f} | Delta: {(foot_y_after - foot_y_before):+.2f}")

                if leg_height_guard and scale_legs:
                    log_messages.append(
                        f"Leg Height Guard         | Target(No-Legs): {guard_target_h:.2f} | "
                        f"Raw-Legs: {guard_raw_h:.2f} | FinalPreGround: {guard_final_pre_ground_h:.2f}"
                    )
                    log_messages.append(
                        f"Leg Guard Faktor         | Used: {guard_local_factor:.5f} | "
                        f"Unsmoothed: {guard_unsmoothed_factor:.5f} | "
                        f"Triggered: {guard_used} | Iter: {guard_iterations} | "
                        f"SmoothingBypassed: {guard_smoothing_bypassed}"
                    )
                    log_messages.append(
                        f"Leg Guard Error          | Raw: {guard_raw_rel_error * 100.0:+.2f}% | "
                        f"Final: {guard_final_rel_error * 100.0:+.2f}%"
                    )

                if raw_shift is not None and shift is not None:
                    log_messages.append(f"Ground Anchor            | RawShift: {raw_shift:+.2f} | SmoothedShift: {shift:+.2f}")
                else:
                    log_messages.append("Ground Anchor            | Keine gültigen Fußpunkte für Shift gefunden.")

                log_messages.append("-" * 70)

                for k in bones_before.keys():
                    log_messages.append(f"Knochen: {k.ljust(18)} | Vorher: {bones_before[k]:.2f} -> Nachher: {bones_after[k]:.2f}")

            # Output-Formatierung
            if is_tensor:
                if frame_data.dim() == 3:
                    raw_poses[frame_idx][0] = torch.from_numpy(pts_final).to(frame_data.device)
                else:
                    raw_poses[frame_idx] = torch.from_numpy(pts_final).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts_final.tolist()

        # --- PHASE 2.5: GUARD SUMMARY ---
        log_messages.append(f"\n--- V18.1 LEG HEIGHT GUARD SUMMARY ---")
        log_messages.append(f"Guard Events: {len(guard_events)}")

        if guard_events:
            worst_raw = max(guard_events, key=lambda e: abs(e["raw_error"]))
            worst_final = max(guard_events, key=lambda e: abs(e["final_error"]))

            log_messages.append(
                f"Stärkster Raw-Ausreißer: Frame {worst_raw['frame']} | "
                f"Target: {worst_raw['target_h']:.2f} | Raw: {worst_raw['raw_h']:.2f} | "
                f"RawError: {worst_raw['raw_error'] * 100.0:+.2f}% | "
                f"Faktor: {worst_raw['factor']:.5f}"
            )
            log_messages.append(
                f"Stärkster Final-Ausreißer nach Guard: Frame {worst_final['frame']} | "
                f"Target: {worst_final['target_h']:.2f} | Final: {worst_final['final_h']:.2f} | "
                f"FinalError: {worst_final['final_error'] * 100.0:+.2f}% | "
                f"Faktor: {worst_final['factor']:.5f}"
            )

            log_messages.append("\nErste Guard-Events:")
            for e in guard_events[:25]:
                log_messages.append(
                    f"Frame {str(e['frame']).rjust(4)} | "
                    f"Target {e['target_h']:.2f} | Raw {e['raw_h']:.2f} | Final {e['final_h']:.2f} | "
                    f"RawErr {e['raw_error'] * 100.0:+.2f}% | FinalErr {e['final_error'] * 100.0:+.2f}% | "
                    f"Factor {e['factor']:.5f} | Iter {e['iterations']} | "
                    f"SmoothBypass {e['smoothing_bypassed']}"
                )

            if len(guard_events) > 25:
                log_messages.append(f"... weitere {len(guard_events) - 25} Guard-Events ausgelassen.")
        else:
            log_messages.append("Keine Height-Guard-Eingriffe nötig oder Leg Scaling/Guard war deaktiviert.")

        # --- PHASE 3: KAMERA-CONFIG BEREINIGEN ---
        try:
            config_dict = json.loads(nlf_render_config) if isinstance(nlf_render_config, str) and nlf_render_config.strip() else {}
        except Exception:
            config_dict = {}

        config_dict["anchor_scale"] = 1.0
        config_dict["scale_x_factor"] = 1.0
        clean_config_str = json.dumps(config_dict)

        return (nlf_data_retargeted, "\n".join(log_messages), clean_config_str)


class NLFProportionalRetargeterV19:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "bypass": ("BOOLEAN", {"default": False, "tooltip": "Ignoriert die Node komplett und gibt Originaldaten zurück."}),
                "scale_torso": ("BOOLEAN", {"default": True, "tooltip": "Skaliert den Torso"}),
                "scale_shoulders": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Schultern"}),
                "scale_hips": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Hüften"}),
                "scale_arms": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Arme"}),
                "scale_legs": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Beine"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
                "scale_stance_and_head": ("BOOLEAN", {"default": False, "tooltip": "Wendet den Höhen-Korrekturfaktor auch auf Schulter-/Hüftbreite und den Kopf an."}),
                "temporal_smooth_factor": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet die Gelenke (Arme/Beine) gegen Zittern."}),
                "ground_smooth_factor": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet NUR die Auf/Ab-Bewegung (Walk-Cycle Anti-Jitter). Oft höher als Temporal."}),

                "leg_height_guard": ("BOOLEAN", {"default": True, "tooltip": "V18.1 Safety: verhindert, dass Leg Scaling die Gesamthöhe/Schulterhöhe in einzelnen Frames sprengt."}),
                "leg_height_guard_tolerance": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.25, "step": 0.005, "tooltip": "Relative Toleranz. 0.025 = 2.5 Prozent Abweichung zur No-Legs-Baseline erlaubt."}),
                "leg_height_guard_min_factor": ("FLOAT", {"default": 0.70, "min": 0.30, "max": 1.00, "step": 0.01, "tooltip": "Minimaler lokaler Bein-Kompressionsfaktor bei problematischen Frames."}),
                "leg_height_guard_max_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 1.50, "step": 0.01, "tooltip": "Maximaler lokaler Bein-Faktor. Default 1.0 bedeutet: Guard darf nur komprimieren, nicht verlängern."}),
                "leg_height_guard_smooth": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "Glättet den lokalen Bein-Guard-Faktor. Niedrig lassen, damit Ausreißer schnell abgefangen werden."}),

                "ground_anchor_mode": (["v18_legacy", "conservative", "advanced_trend"], {"default": "conservative", "tooltip": "V19 Ground-Modus: v18_legacy = altes Verhalten, conservative = Clamp + Body Guard, advanced_trend = Lookahead/Trend-Analyse."}),
                "ground_shift_percentile": ("FLOAT", {"default": 80.0, "min": 50.0, "max": 100.0, "step": 1.0, "tooltip": "Robuster Fußanker für V19-Modi. 80 nutzt einen hohen, aber weniger ausreißeranfälligen Fußwert statt hartem max()."}),
                "ground_shift_max_step": ("FLOAT", {"default": 14.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Maximal erlaubte Änderung des Ground-Shifts pro Frame in V19-Modi. 0 deaktiviert diese Begrenzung."}),
                "body_anchor_guard": ("BOOLEAN", {"default": True, "tooltip": "Begrenzt den Ground-Shift, wenn Schulter/Hüfte dadurch zu stark hoch/runter gezogen werden."}),
                "body_anchor_max_delta": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 120.0, "step": 1.0, "tooltip": "Maximale erlaubte Body-Y-Abweichung durch Ground-Shift gegenüber Original/Trend."}),
                "body_anchor_lookahead_radius": ("INT", {"default": 3, "min": 0, "max": 12, "step": 1, "tooltip": "Advanced: Anzahl Frames davor/danach für Trend-Erkennung. 3 = 7-Frame-Fenster."}),
                "body_anchor_trend_tolerance": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 80.0, "step": 1.0, "tooltip": "Advanced: zusätzliche Toleranz für echte kontinuierliche Auf/Ab-Bewegung, z.B. Treppen."}),
            },
            "optional": {
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus dem Scaler (wird bereinigt)"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output", "nlf_render_config_clean")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V19: V18.1 Leg Height Guard + umschaltbarer Body-stabilized Ground Anchor mit Conservative/Advanced-Trend-Modus."

    def process(
        self,
        video_nlf_data,
        calibration_data,
        bypass,
        scale_torso,
        scale_shoulders,
        scale_hips,
        scale_arms,
        scale_legs,
        frontal_3d_angle_tolerance,
        scale_stance_and_head,
        temporal_smooth_factor,
        ground_smooth_factor,
        leg_height_guard,
        leg_height_guard_tolerance,
        leg_height_guard_min_factor,
        leg_height_guard_max_factor,
        leg_height_guard_smooth,
        ground_anchor_mode,
        ground_shift_percentile,
        ground_shift_max_step,
        body_anchor_guard,
        body_anchor_max_delta,
        body_anchor_lookahead_radius,
        body_anchor_trend_tolerance,
        nlf_render_config="{}"
    ):
        import copy
        import numpy as np
        import math
        import torch
        import json

        # --- BYPASS ---
        if bypass:
            return (
                video_nlf_data,
                "=== NLF PROPORTIONAL RETARGETER V19 ===\nBYPASS AKTIVIERT: Keine Daten verändert.",
                nlf_render_config if nlf_render_config else "{}"
            )

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V19 (LEG GUARD + BODY-STABILIZED GROUND ANCHOR) ==="]

        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)

        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        head_val = true_3d_bones.get("head", 0.0)

        # Safety: clamp user inputs to sane runtime values.
        allowed_ground_modes = ["v18_legacy", "conservative", "advanced_trend"]
        if ground_anchor_mode not in allowed_ground_modes:
            ground_anchor_mode = "conservative"

        leg_height_guard_tolerance = max(0.0, float(leg_height_guard_tolerance))
        leg_height_guard_min_factor = float(np.clip(leg_height_guard_min_factor, 0.30, 1.00))
        leg_height_guard_max_factor = float(np.clip(leg_height_guard_max_factor, 1.00, 1.50))
        leg_height_guard_smooth = float(np.clip(leg_height_guard_smooth, 0.0, 0.95))
        ground_shift_percentile = float(np.clip(ground_shift_percentile, 50.0, 100.0))
        ground_shift_max_step = max(0.0, float(ground_shift_max_step))
        body_anchor_max_delta = max(0.0, float(body_anchor_max_delta))
        body_anchor_lookahead_radius = int(max(0, min(12, body_anchor_lookahead_radius)))
        body_anchor_trend_tolerance = max(0.0, float(body_anchor_trend_tolerance))

        toggles = {
            "scale_torso": scale_torso,
            "scale_shoulders": scale_shoulders,
            "scale_hips": scale_hips,
            "scale_arms": scale_arms,
            "scale_legs": scale_legs
        }

        log_messages.append(
            f"Leg Height Guard: {'AKTIV' if leg_height_guard and scale_legs else 'INAKTIV'} | "
            f"Tolerance: {leg_height_guard_tolerance * 100.0:.2f}% | "
            f"Clamp: [{leg_height_guard_min_factor:.3f}, {leg_height_guard_max_factor:.3f}] | "
            f"Smooth: {leg_height_guard_smooth:.2f}"
        )
        log_messages.append(
            f"Ground Anchor Mode: {ground_anchor_mode} | "
            f"GroundSmooth: {ground_smooth_factor:.2f} | "
            f"FootPercentile: {ground_shift_percentile:.1f} | "
            f"MaxStep: {ground_shift_max_step:.2f} | "
            f"BodyGuard: {body_anchor_guard} | "
            f"BodyMaxDelta: {body_anchor_max_delta:.2f} | "
            f"LookaheadRadius: {body_anchor_lookahead_radius} | "
            f"TrendTolerance: {body_anchor_trend_tolerance:.2f}"
        )

        # --- STUFE 1: TÜRSTEHER (WINKEL-RADAR) ---
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({'length': 0.0, 'is_frontal': False, 'has_feet': False})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else np.array(frame_data))
            if pts.ndim == 3:
                pts = pts[0]

            def is_val(idx):
                return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

            is_frontal = False
            if len(pts) >= 18:
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                if max(angle_h, angle_s) <= frontal_3d_angle_tolerance:
                    is_frontal = True

            all_frames_data.append({'length': length, 'is_frontal': is_frontal, 'has_feet': is_val(7) or is_val(8)})
            if is_frontal:
                frontal_indices.append(i)

        # --- STUFE 2: ANCHOR-FRAME BESTIMMEN ---
        candidates = frontal_indices if frontal_indices else list(range(len(raw_poses)))
        max_len = max([d['length'] for d in all_frames_data]) if all_frames_data else 1.0

        if not candidates:
            log_messages.append("FEHLER: Keine verwertbaren Frames gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            d = all_frames_data[idx]
            score = (1000.0 if d['has_feet'] else 0.0) + (500.0 if d['is_frontal'] else 0.0) + ((d['length'] / max_len) * 100.0 if max_len > 1e-5 else 0.0)
            if score > best_score:
                best_score, best_idx = score, idx

        log_messages.append(f"-> Referenz-Frame (Anchor) für globalen Ratio-Loop: {best_idx}")

        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        ref_pts = ref_frame_data[0].cpu().numpy() if is_t and ref_frame_data.dim() == 3 else (ref_frame_data.cpu().numpy() if is_t else np.array(ref_frame_data))
        if ref_pts.ndim == 3:
            ref_pts = ref_pts[0]

        # --- HILFSFUNKTIONEN ---
        tree = {
            0: [1, 2, 3],
            1: [4],
            4: [7],
            7: [10],
            2: [5],
            5: [8],
            8: [11],
            3: [6],
            6: [9],
            9: [12, 13, 14],
            12: [15],
            13: [16],
            16: [18],
            18: [20],
            20: [22],
            14: [17],
            17: [19],
            19: [21],
            21: [23]
        }

        def is_valid_point(p_array, idx):
            return idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5

        def extract_points(frame_data, copy_array=False):
            if frame_data is None or len(frame_data) == 0:
                return None
            is_tensor_local = isinstance(frame_data, torch.Tensor)
            if is_tensor_local and frame_data.dim() == 3:
                arr = frame_data[0].cpu().numpy()
            elif is_tensor_local:
                arr = frame_data.cpu().numpy()
            else:
                arr = np.array(frame_data)
            if arr.ndim == 3:
                arr = arr[0]
            if copy_array:
                arr = arr.copy()
            return arr

        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child)
                    desc.extend(get_all_descendants(child, tree_map))
            return desc

        def get_height_stable(p_array):
            if p_array is None:
                return 0.0
            if 12 < len(p_array) and np.linalg.norm(p_array[12]) > 1e-5:
                top_y = p_array[12][1]
            else:
                return 0.0

            feet_y = [
                p_array[idx][1]
                for idx in [7, 8, 10, 11, 4, 5]
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]

            return (max(feet_y) - top_y) if feet_y else 0.0

        def get_avg_y(p_array, indices):
            if p_array is None:
                return 0.0
            vals = [
                p_array[idx][1]
                for idx in indices
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]
            return float(sum(vals) / len(vals)) if vals else 0.0

        def get_body_anchor_y(p_array):
            # Kombiniert Schulter und Hüfte. Robuster als nur Schulter, aber sensibler als Gesamthöhe.
            shoulder_y = get_avg_y(p_array, [16, 17])
            hip_y = get_avg_y(p_array, [1, 2])

            if shoulder_y != 0.0 and hip_y != 0.0:
                return float((shoulder_y + hip_y) * 0.5)
            if shoulder_y != 0.0:
                return float(shoulder_y)
            if hip_y != 0.0:
                return float(hip_y)
            return 0.0

        def get_foot_values_y(p_array):
            if p_array is None:
                return []
            return [
                float(p_array[idx][1])
                for idx in [7, 8, 10, 11, 4, 5]
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]

        def get_foot_anchor_y(p_array, robust=False, percentile=80.0):
            vals = get_foot_values_y(p_array)
            if not vals:
                return 0.0
            if robust:
                return float(np.percentile(np.array(vals, dtype=np.float32), percentile))
            return float(max(vals))

        def median_filter_value(series, idx, radius):
            start = max(0, idx - radius)
            end = min(len(series), idx + radius + 1)
            vals = [float(v) for v in series[start:end] if v is not None and abs(float(v)) > 1e-5]
            return float(np.median(vals)) if vals else 0.0

        def window_range_value(series, idx, radius):
            start = max(0, idx - radius)
            end = min(len(series), idx + radius + 1)
            vals = [float(v) for v in series[start:end] if v is not None and abs(float(v)) > 1e-5]
            return float(max(vals) - min(vals)) if vals else 0.0

        def get_bone_lengths(pts_array):
            def dist(p1, p2):
                if (
                    p1 < len(pts_array)
                    and p2 < len(pts_array)
                    and np.linalg.norm(pts_array[p1]) > 1e-5
                    and np.linalg.norm(pts_array[p2]) > 1e-5
                ):
                    return float(np.linalg.norm(pts_array[p2] - pts_array[p1]))
                return 0.0

            return {
                "Torso": dist(0, 12),
                "Kopf": dist(12, 15),
                "R_Oberschenkel": dist(2, 5),
                "R_Wade": dist(5, 8),
                "L_Oberschenkel": dist(1, 4),
                "L_Wade": dist(4, 7),
                "R_Arm": dist(17, 19),
                "R_Unterarm": dist(19, 21),
                "L_Arm": dist(16, 18),
                "L_Unterarm": dist(18, 20),
                "Schulterbreite": dist(16, 17),
                "Hueftbreite": dist(1, 2)
            }

        def build_and_log(pts_source, factor, tgls, final_mode=False, force_all=False, leg_height_factor=1.0):
            pts_b = pts_source.copy()

            if len(pts_b) <= 12 or not is_valid_point(pts_b, 0) or not is_valid_point(pts_b, 12):
                return pts_b

            orig_torso_curr = np.linalg.norm(pts_b[12] - pts_b[0]) if np.linalg.norm(pts_b[12]) > 1e-5 else 0.0
            missing_neck_curr = orig_torso_curr * (head_val / 100.0) / 2.0 if is_normalized else head_val / 2.0
            frame_ref_torso = orig_torso_curr + missing_neck_curr

            targets = {
                k: (v / 100.0 * frame_ref_torso if is_normalized else v)
                for k, v in true_3d_bones.items()
            }

            # 1. Torso Skalierung
            if force_all or tgls.get("scale_torso", True):
                if len(pts_b) > 12 and is_valid_point(pts_b, 0) and is_valid_point(pts_b, 12):
                    cv = pts_b[12] - pts_b[0]
                    cl = np.linalg.norm(cv)

                    if cl > 1e-5:
                        t_len = targets.get("torso", cl)
                        if final_mode:
                            t_len *= factor

                        f_node = t_len / cl

                        for p, c in [(0, 3), (3, 6), (6, 9), (9, 12)]:
                            if p >= len(pts_b) or c >= len(pts_b):
                                continue
                            if not is_valid_point(pts_b, p) or not is_valid_point(pts_b, c):
                                continue

                            vec = pts_b[c] - pts_b[p]
                            new_c = pts_b[p] + vec * f_node
                            delta = new_c - pts_b[c]
                            pts_b[c] += delta

                            for d in get_all_descendants(c, tree):
                                if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                                    pts_b[d] += delta

            # 2. Kopf
            if 15 < len(pts_b) and is_valid_point(pts_b, 12) and is_valid_point(pts_b, 15):
                cv = pts_b[15] - pts_b[12]
                cl = np.linalg.norm(cv)

                if cl > 1e-5:
                    t_len = targets.get("head", cl * 2.0) / 2.0
                    if final_mode and scale_stance_and_head:
                        t_len *= factor

                    delta = (pts_b[12] + (cv / cl * t_len)) - pts_b[15]
                    pts_b[15] += delta

            # 3. Operations: Arme, Beine, Schultern, Hüften
            ops = [
                ('shoulder_width', 12, 17),
                ('shoulder_width', 12, 16),
                ('hip_width', 0, 2),
                ('hip_width', 0, 1),
                ('r_arm', 17, 19),
                ('r_forearm', 19, 21),
                ('l_arm', 16, 18),
                ('l_forearm', 18, 20),
                ('r_thigh', 2, 5),
                ('r_calf', 5, 8),
                ('l_thigh', 1, 4),
                ('l_calf', 4, 7)
            ]

            for key, p_idx, c_idx in ops:
                if p_idx >= len(pts_b) or c_idx >= len(pts_b):
                    continue

                if not is_valid_point(pts_b, p_idx) or not is_valid_point(pts_b, c_idx):
                    continue

                is_allowed = force_all

                if not is_allowed:
                    if 'shoulder' in key and tgls.get('scale_shoulders', True):
                        is_allowed = True
                    elif 'hip' in key and tgls.get('scale_hips', True):
                        is_allowed = True
                    elif 'arm' in key and tgls.get('scale_arms', True):
                        is_allowed = True
                    elif ('thigh' in key or 'calf' in key) and tgls.get('scale_legs', True):
                        is_allowed = True

                if not is_allowed:
                    continue

                cv = pts_b[c_idx] - pts_b[p_idx]
                cl = np.linalg.norm(cv)

                if cl < 1e-5:
                    continue

                if key in ['shoulder_width', 'hip_width']:
                    stance_target = targets.get(key, cl * 2.0) / 2.0
                    calib_key = f"calibration_{key}"
                    bone_target = targets.get(calib_key, stance_target * 2.0) / 2.0

                    if final_mode and scale_stance_and_head:
                        stance_target *= factor
                        bone_target *= factor

                    scale_xz_stance = stance_target / cl
                    pos_stance = pts_b[p_idx].copy()
                    pos_stance[0] += cv[0] * scale_xz_stance
                    pos_stance[1] += cv[1]
                    pos_stance[2] += cv[2] * scale_xz_stance

                    delta_stance = pos_stance - pts_b[c_idx]
                    pts_b[c_idx] += delta_stance

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_stance

                    scale_xz_config = bone_target / cl
                    pos_config = pts_b[p_idx].copy()
                    pos_config[0] += cv[0] * scale_xz_config
                    pos_config[1] += cv[1]
                    pos_config[2] += cv[2] * scale_xz_config

                    delta_config = pos_config - pts_b[c_idx]
                    pts_b[c_idx] += delta_config

                else:
                    if key not in targets:
                        continue

                    t_len_normal = targets[key]
                    cal_k = "calibration_" + key
                    t_len_final = targets.get(cal_k, t_len_normal)

                    if final_mode:
                        t_len_final *= factor

                    if final_mode and ('thigh' in key or 'calf' in key):
                        t_len_final *= leg_height_factor

                    dir_vec = cv / cl
                    new_c_pos = pts_b[p_idx] + (dir_vec * t_len_final)
                    delta_shift = new_c_pos - pts_b[c_idx]
                    pts_b[c_idx] = new_c_pos

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_shift

            return pts_b

        # --- PHASE 1: RATIO-LOOP AM ANCHOR-FRAME ---
        orig_h_global = get_height_stable(ref_pts)
        global_f_scale = 1.0

        log_messages.append(f"\n--- TOTAL-HEIGHT-ENFORCER ---")
        if orig_h_global > 1e-5:
            for iteration in range(10):
                pts_test = build_and_log(
                    ref_pts,
                    global_f_scale,
                    toggles,
                    final_mode=True,
                    force_all=True,
                    leg_height_factor=1.0
                )
                test_h = get_height_stable(pts_test)

                if test_h < 1e-5:
                    break

                diff = abs(orig_h_global - test_h)
                if diff < 0.1:
                    break

                ratio = orig_h_global / test_h
                global_f_scale *= ratio

        log_messages.append(f"Berechneter universeller Kompressions-Faktor: {global_f_scale:.6f}x")

        # --- PHASE 1.5: ADVANCED-TREND-VORANALYSE ---
        body_y_series = []
        foot_y_series = []
        height_series = []
        body_y_trend = []
        foot_y_trend = []
        body_window_range = []
        foot_window_range = []

        for frame_data in raw_poses:
            pts_pre = extract_points(frame_data, copy_array=False)
            body_y_series.append(get_body_anchor_y(pts_pre) if pts_pre is not None else 0.0)
            foot_y_series.append(get_foot_anchor_y(pts_pre, robust=True, percentile=ground_shift_percentile) if pts_pre is not None else 0.0)
            height_series.append(get_height_stable(pts_pre) if pts_pre is not None else 0.0)

        for idx in range(len(raw_poses)):
            body_y_trend.append(median_filter_value(body_y_series, idx, body_anchor_lookahead_radius))
            foot_y_trend.append(median_filter_value(foot_y_series, idx, body_anchor_lookahead_radius))
            body_window_range.append(window_range_value(body_y_series, idx, body_anchor_lookahead_radius))
            foot_window_range.append(window_range_value(foot_y_series, idx, body_anchor_lookahead_radius))

        if ground_anchor_mode == "advanced_trend":
            log_messages.append(
                f"Advanced Trend Analyse aktiv: Frames={len(raw_poses)} | "
                f"Medianfenster={body_anchor_lookahead_radius * 2 + 1} | "
                f"BodyRangeMedian={float(np.median([v for v in body_window_range if v > 1e-5])) if any(v > 1e-5 for v in body_window_range) else 0.0:.2f} | "
                f"FootRangeMedian={float(np.median([v for v in foot_window_range if v > 1e-5])) if any(v > 1e-5 for v in foot_window_range) else 0.0:.2f}"
            )

        # --- PHASE 2: VERARBEITUNG ALLER FRAMES ---
        prev_pts = None
        prev_shift = None
        prev_leg_guard_factor = 1.0
        guard_events = []
        ground_guard_events = []

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0:
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())
            if pts.ndim == 3:
                pts = pts[0]

            log_this_frame = (frame_idx % 10 == 0)

            bones_before = None
            h_before = get_height_stable(pts)
            shoulder_y_before = get_avg_y(pts, [16, 17])
            hip_y_before = get_avg_y(pts, [1, 2])
            body_y_before = get_body_anchor_y(pts)
            foot_y_before_legacy = get_foot_anchor_y(pts, robust=False)
            foot_y_before_robust = get_foot_anchor_y(pts, robust=True, percentile=ground_shift_percentile)

            if log_this_frame:
                bones_before = get_bone_lengths(pts)

            # --- V18.1 LEG HEIGHT GUARD ---
            guard_used = False
            guard_smoothing_bypassed = False
            guard_clamped_by_no_expand = False
            guard_iterations = 0
            guard_target_h = 0.0
            guard_raw_h = 0.0
            guard_final_pre_ground_h = 0.0
            guard_unsmoothed_factor = 1.0
            guard_local_factor = 1.0
            guard_raw_rel_error = 0.0
            guard_final_rel_error = 0.0

            if leg_height_guard and scale_legs:
                no_leg_toggles = toggles.copy()
                no_leg_toggles["scale_legs"] = False

                pts_no_legs = build_and_log(
                    pts,
                    global_f_scale,
                    no_leg_toggles,
                    final_mode=True,
                    force_all=False,
                    leg_height_factor=1.0
                )
                guard_target_h = get_height_stable(pts_no_legs)

                pts_raw_legs = build_and_log(
                    pts,
                    global_f_scale,
                    toggles,
                    final_mode=True,
                    force_all=False,
                    leg_height_factor=1.0
                )
                guard_raw_h = get_height_stable(pts_raw_legs)

                if guard_target_h <= 1e-5:
                    guard_target_h = h_before

                if guard_target_h > 1e-5 and guard_raw_h > 1e-5:
                    guard_raw_rel_error = (guard_raw_h - guard_target_h) / guard_target_h

                    if abs(guard_raw_rel_error) > leg_height_guard_tolerance:
                        # Wenn die Beine zu kurz werden, aber max_factor 1.0 ist, darf der Guard bewusst nicht verlängern.
                        # Dann loggen wir es als blockierten Eingriff statt 6 nutzlose Iterationen zu fahren.
                        if guard_raw_rel_error < 0.0 and leg_height_guard_max_factor <= 1.0001:
                            guard_clamped_by_no_expand = True
                            pts_final = pts_raw_legs
                            guard_final_pre_ground_h = guard_raw_h
                            guard_local_factor = 1.0
                            guard_unsmoothed_factor = 1.0
                            guard_final_rel_error = guard_raw_rel_error
                        else:
                            guard_used = True

                            candidate_factor = 1.0
                            pts_candidate = pts_raw_legs
                            h_candidate = guard_raw_h

                            for guard_iterations in range(1, 7):
                                if h_candidate <= 1e-5:
                                    break

                                ratio = guard_target_h / h_candidate
                                candidate_factor *= ratio
                                candidate_factor = float(np.clip(candidate_factor, leg_height_guard_min_factor, leg_height_guard_max_factor))

                                pts_candidate = build_and_log(
                                    pts,
                                    global_f_scale,
                                    toggles,
                                    final_mode=True,
                                    force_all=False,
                                    leg_height_factor=candidate_factor
                                )
                                h_candidate = get_height_stable(pts_candidate)

                                rel_err_candidate = abs(h_candidate - guard_target_h) / guard_target_h
                                if rel_err_candidate <= leg_height_guard_tolerance:
                                    break

                            guard_unsmoothed_factor = candidate_factor

                            if leg_height_guard_smooth > 0.0:
                                smoothed_factor = (prev_leg_guard_factor * leg_height_guard_smooth) + (guard_unsmoothed_factor * (1.0 - leg_height_guard_smooth))
                                smoothed_factor = float(np.clip(smoothed_factor, leg_height_guard_min_factor, leg_height_guard_max_factor))

                                pts_smoothed_candidate = build_and_log(
                                    pts,
                                    global_f_scale,
                                    toggles,
                                    final_mode=True,
                                    force_all=False,
                                    leg_height_factor=smoothed_factor
                                )
                                h_smoothed_candidate = get_height_stable(pts_smoothed_candidate)

                                rel_err_smoothed = abs(h_smoothed_candidate - guard_target_h) / guard_target_h
                                hard_limit = max(leg_height_guard_tolerance * 2.0, 0.04)

                                if rel_err_smoothed > hard_limit:
                                    guard_smoothing_bypassed = True
                                    guard_local_factor = guard_unsmoothed_factor
                                    pts_final = pts_candidate
                                    guard_final_pre_ground_h = h_candidate
                                else:
                                    guard_local_factor = smoothed_factor
                                    pts_final = pts_smoothed_candidate
                                    guard_final_pre_ground_h = h_smoothed_candidate
                            else:
                                guard_local_factor = guard_unsmoothed_factor
                                pts_final = pts_candidate
                                guard_final_pre_ground_h = h_candidate

                            prev_leg_guard_factor = guard_local_factor
                            guard_final_rel_error = (guard_final_pre_ground_h - guard_target_h) / guard_target_h if guard_target_h > 1e-5 else 0.0

                        guard_events.append({
                            "frame": frame_idx,
                            "target_h": guard_target_h,
                            "raw_h": guard_raw_h,
                            "final_h": guard_final_pre_ground_h,
                            "raw_error": guard_raw_rel_error,
                            "final_error": guard_final_rel_error,
                            "factor": guard_local_factor,
                            "unsmoothed_factor": guard_unsmoothed_factor,
                            "iterations": guard_iterations,
                            "smoothing_bypassed": guard_smoothing_bypassed,
                            "blocked_no_expand": guard_clamped_by_no_expand
                        })

                    else:
                        pts_final = pts_raw_legs

                        if leg_height_guard_smooth > 0.0:
                            prev_leg_guard_factor = (prev_leg_guard_factor * leg_height_guard_smooth) + (1.0 * (1.0 - leg_height_guard_smooth))
                        else:
                            prev_leg_guard_factor = 1.0

                        guard_final_pre_ground_h = guard_raw_h
                else:
                    pts_final = pts_raw_legs
                    guard_final_pre_ground_h = guard_raw_h

            else:
                pts_final = build_and_log(
                    pts,
                    global_f_scale,
                    toggles,
                    final_mode=True,
                    force_all=False,
                    leg_height_factor=1.0
                )
                guard_final_pre_ground_h = get_height_stable(pts_final)

            # --- V19 GROUND ANCHOR ---
            ground_debug = {
                "mode": ground_anchor_mode,
                "has_feet": False,
                "foot_ref_source": "none",
                "orig_anchor": 0.0,
                "new_anchor": 0.0,
                "raw_shift": 0.0,
                "smoothed_shift": 0.0,
                "step_clamped_shift": 0.0,
                "body_guard_shift": 0.0,
                "final_shift": 0.0,
                "step_clamped": False,
                "body_guarded": False,
                "body_ref_y": 0.0,
                "body_before_ground": 0.0,
                "body_after_raw": 0.0,
                "body_after_final": 0.0,
                "body_delta_raw": 0.0,
                "body_delta_final": 0.0,
                "allowed_body_delta": body_anchor_max_delta,
                "trend_extra": 0.0,
                "current_body_y": body_y_before,
                "trend_body_y": body_y_trend[frame_idx] if frame_idx < len(body_y_trend) else 0.0,
                "current_foot_y": foot_y_before_robust,
                "trend_foot_y": foot_y_trend[frame_idx] if frame_idx < len(foot_y_trend) else 0.0,
            }

            if ground_anchor_mode == "v18_legacy":
                orig_anchor = get_foot_anchor_y(pts, robust=False)
                new_anchor = get_foot_anchor_y(pts_final, robust=False)
                foot_ref_source = "legacy_max"
            else:
                current_orig_anchor = get_foot_anchor_y(pts, robust=True, percentile=ground_shift_percentile)
                trend_orig_anchor = foot_y_trend[frame_idx] if frame_idx < len(foot_y_trend) else 0.0

                if ground_anchor_mode == "advanced_trend" and trend_orig_anchor > 1e-5:
                    foot_spike_limit = max(body_anchor_trend_tolerance * 2.0, 12.0)
                    if current_orig_anchor > 1e-5 and abs(current_orig_anchor - trend_orig_anchor) > foot_spike_limit:
                        orig_anchor = trend_orig_anchor
                        foot_ref_source = "advanced_trend_median"
                    else:
                        orig_anchor = current_orig_anchor
                        foot_ref_source = "advanced_current_robust"
                else:
                    orig_anchor = current_orig_anchor
                    foot_ref_source = "conservative_robust"

                new_anchor = get_foot_anchor_y(pts_final, robust=True, percentile=ground_shift_percentile)

            if orig_anchor > 1e-5 and new_anchor > 1e-5:
                ground_debug["has_feet"] = True
                ground_debug["foot_ref_source"] = foot_ref_source
                ground_debug["orig_anchor"] = orig_anchor
                ground_debug["new_anchor"] = new_anchor

                raw_shift = orig_anchor - new_anchor
                ground_debug["raw_shift"] = raw_shift

                if ground_smooth_factor > 0.0 and prev_shift is not None:
                    smoothed_shift = (prev_shift * ground_smooth_factor) + (raw_shift * (1.0 - ground_smooth_factor))
                else:
                    smoothed_shift = raw_shift

                ground_debug["smoothed_shift"] = smoothed_shift

                if ground_anchor_mode == "v18_legacy":
                    final_shift = smoothed_shift
                    ground_debug["step_clamped_shift"] = final_shift
                    ground_debug["body_guard_shift"] = final_shift
                else:
                    # Step Clamp: verhindert abrupte Ground-Shift-Sprünge.
                    step_clamped_shift = smoothed_shift
                    if ground_shift_max_step > 0.0 and prev_shift is not None:
                        step_delta = smoothed_shift - prev_shift
                        if abs(step_delta) > ground_shift_max_step:
                            step_clamped_shift = prev_shift + float(np.clip(step_delta, -ground_shift_max_step, ground_shift_max_step))
                            ground_debug["step_clamped"] = True

                    ground_debug["step_clamped_shift"] = step_clamped_shift

                    # Body Guard: Ground darf Schulter/Hüfte nicht hart pumpen lassen.
                    body_guard_shift = step_clamped_shift
                    body_before_ground = get_body_anchor_y(pts_final)
                    current_body_ref = body_y_before
                    trend_body_ref = body_y_trend[frame_idx] if frame_idx < len(body_y_trend) else 0.0

                    if ground_anchor_mode == "advanced_trend" and trend_body_ref > 1e-5:
                        body_ref_y = trend_body_ref
                        local_body_range = body_window_range[frame_idx] if frame_idx < len(body_window_range) else 0.0
                        trend_extra = min(body_anchor_trend_tolerance, max(0.0, local_body_range * 0.50))
                    else:
                        body_ref_y = current_body_ref
                        trend_extra = 0.0

                    allowed_body_delta = body_anchor_max_delta + trend_extra
                    body_after_raw = body_before_ground + step_clamped_shift
                    body_delta_raw = body_after_raw - body_ref_y if body_ref_y > 1e-5 else 0.0

                    if body_anchor_guard and body_ref_y > 1e-5 and body_before_ground > 1e-5 and allowed_body_delta > 0.0:
                        if abs(body_delta_raw) > allowed_body_delta:
                            desired_body_after = body_ref_y + (allowed_body_delta if body_delta_raw > 0.0 else -allowed_body_delta)
                            body_guard_shift = desired_body_after - body_before_ground
                            ground_debug["body_guarded"] = True

                    ground_debug["body_guard_shift"] = body_guard_shift
                    ground_debug["body_ref_y"] = body_ref_y
                    ground_debug["body_before_ground"] = body_before_ground
                    ground_debug["body_after_raw"] = body_after_raw
                    ground_debug["body_delta_raw"] = body_delta_raw
                    ground_debug["allowed_body_delta"] = allowed_body_delta
                    ground_debug["trend_extra"] = trend_extra

                    final_shift = body_guard_shift

                ground_debug["final_shift"] = final_shift

                for j in range(len(pts_final)):
                    if np.linalg.norm(pts_final[j]) > 1e-5:
                        pts_final[j][1] += final_shift

                prev_shift = final_shift
                ground_debug["body_after_final"] = get_body_anchor_y(pts_final)
                if ground_debug["body_ref_y"] > 1e-5:
                    ground_debug["body_delta_final"] = ground_debug["body_after_final"] - ground_debug["body_ref_y"]

                if ground_debug["step_clamped"] or ground_debug["body_guarded"]:
                    ground_guard_events.append({
                        "frame": frame_idx,
                        "raw_shift": ground_debug["raw_shift"],
                        "smoothed_shift": ground_debug["smoothed_shift"],
                        "step_clamped_shift": ground_debug["step_clamped_shift"],
                        "final_shift": ground_debug["final_shift"],
                        "step_clamped": ground_debug["step_clamped"],
                        "body_guarded": ground_debug["body_guarded"],
                        "body_delta_raw": ground_debug["body_delta_raw"],
                        "body_delta_final": ground_debug["body_delta_final"],
                        "allowed_body_delta": ground_debug["allowed_body_delta"],
                        "foot_ref_source": ground_debug["foot_ref_source"]
                    })

            # --- TEMPORAL SMOOTHING DER GELENKE ---
            if temporal_smooth_factor > 0.0:
                if prev_pts is None:
                    prev_pts = pts_final.copy()
                else:
                    for j in range(len(pts_final)):
                        if j < len(prev_pts) and np.linalg.norm(pts_final[j]) > 1e-5 and np.linalg.norm(prev_pts[j]) > 1e-5:
                            pts_final[j] = (prev_pts[j] * temporal_smooth_factor) + (pts_final[j] * (1.0 - temporal_smooth_factor))
                        if j < len(prev_pts):
                            prev_pts[j] = pts_final[j].copy()

            # --- MESSUNG NACHHER & LOGGING ---
            if log_this_frame:
                bones_after = get_bone_lengths(pts_final)
                h_after = get_height_stable(pts_final)
                shoulder_y_after = get_avg_y(pts_final, [16, 17])
                hip_y_after = get_avg_y(pts_final, [1, 2])
                body_y_after = get_body_anchor_y(pts_final)
                foot_y_after_legacy = get_foot_anchor_y(pts_final, robust=False)
                foot_y_after_robust = get_foot_anchor_y(pts_final, robust=True, percentile=ground_shift_percentile)

                log_messages.append(f"\n--- FRAME {frame_idx} VERGLEICH (Physische 3D-Knochenlängen in NLF) ---")
                log_messages.append(f"Gesamthöhe (Y-Bounding Box) | Vorher: {h_before:.2f} -> Nachher: {h_after:.2f}")
                log_messages.append(f"Schulter-Y Ø             | Vorher: {shoulder_y_before:.2f} -> Nachher: {shoulder_y_after:.2f} | Delta: {(shoulder_y_after - shoulder_y_before):+.2f}")
                log_messages.append(f"Hüft-Y Ø                 | Vorher: {hip_y_before:.2f} -> Nachher: {hip_y_after:.2f} | Delta: {(hip_y_after - hip_y_before):+.2f}")
                log_messages.append(f"Body-Y Ø                 | Vorher: {body_y_before:.2f} -> Nachher: {body_y_after:.2f} | Delta: {(body_y_after - body_y_before):+.2f}")
                log_messages.append(f"Fuß-Anker-Y Legacy/Robust | Vorher: {foot_y_before_legacy:.2f}/{foot_y_before_robust:.2f} -> Nachher: {foot_y_after_legacy:.2f}/{foot_y_after_robust:.2f}")

                if leg_height_guard and scale_legs:
                    log_messages.append(
                        f"Leg Height Guard         | Target(No-Legs): {guard_target_h:.2f} | "
                        f"Raw-Legs: {guard_raw_h:.2f} | FinalPreGround: {guard_final_pre_ground_h:.2f}"
                    )
                    log_messages.append(
                        f"Leg Guard Faktor         | Used: {guard_local_factor:.5f} | "
                        f"Unsmoothed: {guard_unsmoothed_factor:.5f} | "
                        f"Triggered: {guard_used} | Iter: {guard_iterations} | "
                        f"SmoothingBypassed: {guard_smoothing_bypassed} | NoExpandBlocked: {guard_clamped_by_no_expand}"
                    )
                    log_messages.append(
                        f"Leg Guard Error          | Raw: {guard_raw_rel_error * 100.0:+.2f}% | "
                        f"Final: {guard_final_rel_error * 100.0:+.2f}%"
                    )

                if ground_debug["has_feet"]:
                    log_messages.append(
                        f"Ground Anchor V19        | Mode: {ground_debug['mode']} | Source: {ground_debug['foot_ref_source']} | "
                        f"OrigAnchor: {ground_debug['orig_anchor']:.2f} | NewAnchor: {ground_debug['new_anchor']:.2f}"
                    )
                    log_messages.append(
                        f"Ground Shift             | Raw: {ground_debug['raw_shift']:+.2f} | "
                        f"Smooth: {ground_debug['smoothed_shift']:+.2f} | "
                        f"StepClamp: {ground_debug['step_clamped_shift']:+.2f} | "
                        f"BodyGuard: {ground_debug['body_guard_shift']:+.2f} | "
                        f"Final: {ground_debug['final_shift']:+.2f}"
                    )
                    log_messages.append(
                        f"Ground Guards            | StepClamped: {ground_debug['step_clamped']} | "
                        f"BodyGuarded: {ground_debug['body_guarded']} | "
                        f"AllowedBodyDelta: {ground_debug['allowed_body_delta']:.2f} | TrendExtra: {ground_debug['trend_extra']:.2f}"
                    )
                    log_messages.append(
                        f"Body Guard Ref           | BodyRef: {ground_debug['body_ref_y']:.2f} | "
                        f"BodyBeforeGround: {ground_debug['body_before_ground']:.2f} | "
                        f"DeltaRaw: {ground_debug['body_delta_raw']:+.2f} | "
                        f"DeltaFinal: {ground_debug['body_delta_final']:+.2f}"
                    )
                    if ground_anchor_mode == "advanced_trend":
                        log_messages.append(
                            f"Advanced Trend           | BodyCurrent/Trend: {ground_debug['current_body_y']:.2f}/{ground_debug['trend_body_y']:.2f} | "
                            f"FootCurrent/Trend: {ground_debug['current_foot_y']:.2f}/{ground_debug['trend_foot_y']:.2f} | "
                            f"BodyRangeWin: {body_window_range[frame_idx]:.2f} | FootRangeWin: {foot_window_range[frame_idx]:.2f}"
                        )
                else:
                    log_messages.append("Ground Anchor V19        | Keine gültigen Fußpunkte für Shift gefunden.")

                log_messages.append("-" * 70)

                for k in bones_before.keys():
                    log_messages.append(f"Knochen: {k.ljust(18)} | Vorher: {bones_before[k]:.2f} -> Nachher: {bones_after[k]:.2f}")

            # Output-Formatierung
            if is_tensor:
                if frame_data.dim() == 3:
                    raw_poses[frame_idx][0] = torch.from_numpy(pts_final).to(frame_data.device)
                else:
                    raw_poses[frame_idx] = torch.from_numpy(pts_final).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts_final.tolist()

        # --- PHASE 2.5: GUARD SUMMARY ---
        log_messages.append(f"\n--- V19 LEG HEIGHT GUARD SUMMARY ---")
        log_messages.append(f"Leg Guard Events: {len(guard_events)}")

        if guard_events:
            worst_raw = max(guard_events, key=lambda e: abs(e["raw_error"]))
            worst_final = max(guard_events, key=lambda e: abs(e["final_error"]))
            blocked_count = sum(1 for e in guard_events if e.get("blocked_no_expand", False))

            log_messages.append(
                f"Stärkster Raw-Ausreißer: Frame {worst_raw['frame']} | "
                f"Target: {worst_raw['target_h']:.2f} | Raw: {worst_raw['raw_h']:.2f} | "
                f"RawError: {worst_raw['raw_error'] * 100.0:+.2f}% | "
                f"Faktor: {worst_raw['factor']:.5f}"
            )
            log_messages.append(
                f"Stärkster Final-Ausreißer nach Guard: Frame {worst_final['frame']} | "
                f"Target: {worst_final['target_h']:.2f} | Final: {worst_final['final_h']:.2f} | "
                f"FinalError: {worst_final['final_error'] * 100.0:+.2f}% | "
                f"Faktor: {worst_final['factor']:.5f}"
            )
            log_messages.append(f"No-Expand-blockierte negative Events: {blocked_count}")

            log_messages.append("\nErste Leg-Guard-Events:")
            for e in guard_events[:25]:
                log_messages.append(
                    f"Frame {str(e['frame']).rjust(4)} | "
                    f"Target {e['target_h']:.2f} | Raw {e['raw_h']:.2f} | Final {e['final_h']:.2f} | "
                    f"RawErr {e['raw_error'] * 100.0:+.2f}% | FinalErr {e['final_error'] * 100.0:+.2f}% | "
                    f"Factor {e['factor']:.5f} | Iter {e['iterations']} | "
                    f"SmoothBypass {e['smoothing_bypassed']} | NoExpandBlocked {e.get('blocked_no_expand', False)}"
                )

            if len(guard_events) > 25:
                log_messages.append(f"... weitere {len(guard_events) - 25} Leg-Guard-Events ausgelassen.")
        else:
            log_messages.append("Keine Height-Guard-Eingriffe nötig oder Leg Scaling/Guard war deaktiviert.")

        log_messages.append(f"\n--- V19 GROUND ANCHOR GUARD SUMMARY ---")
        log_messages.append(f"Ground Guard Events: {len(ground_guard_events)}")
        log_messages.append(f"Ground Mode benutzt: {ground_anchor_mode}")

        if ground_guard_events:
            worst_body_raw = max(ground_guard_events, key=lambda e: abs(e["body_delta_raw"]))
            worst_shift = max(ground_guard_events, key=lambda e: abs(e["raw_shift"] - e["final_shift"]))
            step_count = sum(1 for e in ground_guard_events if e.get("step_clamped", False))
            body_count = sum(1 for e in ground_guard_events if e.get("body_guarded", False))

            log_messages.append(
                f"StepClamp Events: {step_count} | BodyGuard Events: {body_count}"
            )
            log_messages.append(
                f"Stärkster Body-Raw-Ausreißer: Frame {worst_body_raw['frame']} | "
                f"RawShift: {worst_body_raw['raw_shift']:+.2f} | FinalShift: {worst_body_raw['final_shift']:+.2f} | "
                f"BodyDeltaRaw: {worst_body_raw['body_delta_raw']:+.2f} | BodyDeltaFinal: {worst_body_raw['body_delta_final']:+.2f} | "
                f"Allowed: {worst_body_raw['allowed_body_delta']:.2f}"
            )
            log_messages.append(
                f"Stärkste Shift-Korrektur: Frame {worst_shift['frame']} | "
                f"RawShift: {worst_shift['raw_shift']:+.2f} | Smooth: {worst_shift['smoothed_shift']:+.2f} | "
                f"StepClamp: {worst_shift['step_clamped_shift']:+.2f} | Final: {worst_shift['final_shift']:+.2f} | "
                f"Source: {worst_shift['foot_ref_source']}"
            )

            log_messages.append("\nErste Ground-Guard-Events:")
            for e in ground_guard_events[:25]:
                log_messages.append(
                    f"Frame {str(e['frame']).rjust(4)} | "
                    f"RawShift {e['raw_shift']:+.2f} | Smooth {e['smoothed_shift']:+.2f} | "
                    f"Step {e['step_clamped_shift']:+.2f} | Final {e['final_shift']:+.2f} | "
                    f"StepClamp {e['step_clamped']} | BodyGuard {e['body_guarded']} | "
                    f"BodyRaw {e['body_delta_raw']:+.2f} | BodyFinal {e['body_delta_final']:+.2f} | "
                    f"Allowed {e['allowed_body_delta']:.2f} | Source {e['foot_ref_source']}"
                )

            if len(ground_guard_events) > 25:
                log_messages.append(f"... weitere {len(ground_guard_events) - 25} Ground-Guard-Events ausgelassen.")
        else:
            log_messages.append("Keine Ground-StepClamp/BodyGuard-Eingriffe nötig oder v18_legacy ohne Guard benutzt.")

        # --- PHASE 3: KAMERA-CONFIG BEREINIGEN ---
        try:
            config_dict = json.loads(nlf_render_config) if isinstance(nlf_render_config, str) and nlf_render_config.strip() else {}
        except Exception:
            config_dict = {}

        config_dict["anchor_scale"] = 1.0
        config_dict["scale_x_factor"] = 1.0
        clean_config_str = json.dumps(config_dict)

        return (nlf_data_retargeted, "\n".join(log_messages), clean_config_str)


class NLFProportionalRetargeterV20:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "bypass": ("BOOLEAN", {"default": False, "tooltip": "Ignoriert die Node komplett und gibt Originaldaten zurück."}),

                "scale_torso": ("BOOLEAN", {"default": True, "tooltip": "Skaliert den Torso"}),
                "scale_shoulders": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Schultern"}),
                "scale_hips": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Hüften"}),
                "scale_arms": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Arme"}),
                "scale_legs": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Beine"}),

                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
                "scale_stance_and_head": ("BOOLEAN", {"default": False, "tooltip": "Wendet den Höhen-Korrekturfaktor auch auf Schulter-/Hüftbreite und den Kopf an."}),
                "temporal_smooth_factor": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet die Gelenke gegen Zittern."}),
                "ground_smooth_factor": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet die Auf/Ab-Bewegung des Ground Anchors."}),

                "body_solver_mode": (["legacy_v19", "pelvis_ratio", "shoulder_foot_ik"], {"default": "pelvis_ratio", "tooltip": "V20 Solver: legacy_v19 = altes Push-Verhalten, pelvis_ratio = Hüfte proportional zwischen Schulter/Fuß, shoulder_foot_ik = zusätzlich Two-Bone-IK für Beine."}),
                "upper_body_anchor_mode": (["neck_shoulders", "shoulders_only", "neck_only"], {"default": "neck_shoulders", "tooltip": "Oberer visueller Anker für den Shoulder-Foot-Solver."}),
                "pelvis_ratio_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Wie stark die Hüfte zur Zielproportion zwischen Schulter und Boden verschoben wird."}),
                "pelvis_vertical_smooth": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "Glättet die gelöste Hüftposition über die Zeit."}),
                "ik_foot_lock_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Wie stark die Original-Fußposition bei IK gehalten wird."}),
                "ik_knee_bend_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Wie stark die originale Knie-Beugerichtung erhalten bleibt."}),

                "leg_height_guard": ("BOOLEAN", {"default": True, "tooltip": "Airbag: verhindert, dass Leg Scaling die Gesamthöhe in einzelnen Frames sprengt."}),
                "leg_height_guard_tolerance": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.25, "step": 0.005, "tooltip": "Relative Toleranz. 0.025 = 2.5 Prozent Abweichung zur Baseline erlaubt."}),
                "leg_height_guard_min_factor": ("FLOAT", {"default": 0.70, "min": 0.30, "max": 1.00, "step": 0.01, "tooltip": "Minimaler lokaler Bein-Kompressionsfaktor."}),
                "leg_height_guard_max_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 1.50, "step": 0.01, "tooltip": "Maximaler lokaler Bein-Faktor."}),
                "leg_height_guard_smooth": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "Glättet den lokalen Bein-Guard-Faktor."}),

                "ground_anchor_mode": (["v18_legacy", "conservative", "advanced_trend"], {"default": "conservative", "tooltip": "Ground-Modus aus V19."}),
                "ground_shift_percentile": ("FLOAT", {"default": 80.0, "min": 50.0, "max": 100.0, "step": 1.0, "tooltip": "Robuster Fußanker statt hartem max()."}),
                "ground_shift_max_step": ("FLOAT", {"default": 14.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Maximale Änderung des Ground-Shifts pro Frame."}),
                "body_anchor_guard": ("BOOLEAN", {"default": True, "tooltip": "Begrenzt Ground-Shift, wenn Schulter/Hüfte zu stark gepumpt werden."}),
                "body_anchor_max_delta": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 120.0, "step": 1.0, "tooltip": "Maximale erlaubte Body-Y-Abweichung durch Ground-Shift."}),
                "body_anchor_lookahead_radius": ("INT", {"default": 3, "min": 0, "max": 12, "step": 1, "tooltip": "Advanced: Frames davor/danach für Trend-Erkennung."}),
                "body_anchor_trend_tolerance": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 80.0, "step": 1.0, "tooltip": "Advanced: Toleranz für echte kontinuierliche Auf/Ab-Bewegung."}),
            },
            "optional": {
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus dem Scaler wird bereinigt."}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output", "nlf_render_config_clean")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V20: Shoulder-Foot Proportional Solver mit beweglicher Hüfte, Soft-IK-Beinen und V19 Ground Guard."

    def process(
        self,
        video_nlf_data,
        calibration_data,
        bypass,
        scale_torso,
        scale_shoulders,
        scale_hips,
        scale_arms,
        scale_legs,
        frontal_3d_angle_tolerance,
        scale_stance_and_head,
        temporal_smooth_factor,
        ground_smooth_factor,
        body_solver_mode,
        upper_body_anchor_mode,
        pelvis_ratio_strength,
        pelvis_vertical_smooth,
        ik_foot_lock_strength,
        ik_knee_bend_strength,
        leg_height_guard,
        leg_height_guard_tolerance,
        leg_height_guard_min_factor,
        leg_height_guard_max_factor,
        leg_height_guard_smooth,
        ground_anchor_mode,
        ground_shift_percentile,
        ground_shift_max_step,
        body_anchor_guard,
        body_anchor_max_delta,
        body_anchor_lookahead_radius,
        body_anchor_trend_tolerance,
        nlf_render_config="{}"
    ):
        import copy
        import numpy as np
        import math
        import torch
        import json

        if bypass:
            return (
                video_nlf_data,
                "=== NLF PROPORTIONAL RETARGETER V20 ===\nBYPASS AKTIVIERT: Keine Daten verändert.",
                nlf_render_config if nlf_render_config else "{}"
            )

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V20 (SHOULDER-FOOT PROPORTIONAL SOLVER) ==="]

        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)

        if is_dict:
            raw_poses = nlf_data_retargeted.get("joints3d_nonparam", [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        head_val = true_3d_bones.get("head", 0.0)

        allowed_solver_modes = ["legacy_v19", "pelvis_ratio", "shoulder_foot_ik"]
        if body_solver_mode not in allowed_solver_modes:
            body_solver_mode = "pelvis_ratio"

        allowed_ground_modes = ["v18_legacy", "conservative", "advanced_trend"]
        if ground_anchor_mode not in allowed_ground_modes:
            ground_anchor_mode = "conservative"

        pelvis_ratio_strength = float(np.clip(pelvis_ratio_strength, 0.0, 1.0))
        pelvis_vertical_smooth = float(np.clip(pelvis_vertical_smooth, 0.0, 0.95))
        ik_foot_lock_strength = float(np.clip(ik_foot_lock_strength, 0.0, 1.0))
        ik_knee_bend_strength = float(np.clip(ik_knee_bend_strength, 0.0, 1.0))

        leg_height_guard_tolerance = max(0.0, float(leg_height_guard_tolerance))
        leg_height_guard_min_factor = float(np.clip(leg_height_guard_min_factor, 0.30, 1.00))
        leg_height_guard_max_factor = float(np.clip(leg_height_guard_max_factor, 1.00, 1.50))
        leg_height_guard_smooth = float(np.clip(leg_height_guard_smooth, 0.0, 0.95))

        ground_shift_percentile = float(np.clip(ground_shift_percentile, 50.0, 100.0))
        ground_shift_max_step = max(0.0, float(ground_shift_max_step))
        body_anchor_max_delta = max(0.0, float(body_anchor_max_delta))
        body_anchor_lookahead_radius = int(max(0, min(12, body_anchor_lookahead_radius)))
        body_anchor_trend_tolerance = max(0.0, float(body_anchor_trend_tolerance))

        toggles = {
            "scale_torso": scale_torso,
            "scale_shoulders": scale_shoulders,
            "scale_hips": scale_hips,
            "scale_arms": scale_arms,
            "scale_legs": scale_legs
        }

        log_messages.append(
            f"Body Solver Mode: {body_solver_mode} | UpperAnchor: {upper_body_anchor_mode} | "
            f"PelvisStrength: {pelvis_ratio_strength:.2f} | PelvisSmooth: {pelvis_vertical_smooth:.2f} | "
            f"IKFootLock: {ik_foot_lock_strength:.2f} | IKKneeBend: {ik_knee_bend_strength:.2f}"
        )
        log_messages.append(
            f"Leg Height Guard: {'AKTIV' if leg_height_guard and scale_legs else 'INAKTIV'} | "
            f"Tolerance: {leg_height_guard_tolerance * 100.0:.2f}% | "
            f"Clamp: [{leg_height_guard_min_factor:.3f}, {leg_height_guard_max_factor:.3f}] | "
            f"Smooth: {leg_height_guard_smooth:.2f}"
        )
        log_messages.append(
            f"Ground Anchor Mode: {ground_anchor_mode} | GroundSmooth: {ground_smooth_factor:.2f} | "
            f"FootPercentile: {ground_shift_percentile:.1f} | MaxStep: {ground_shift_max_step:.2f} | "
            f"BodyGuard: {body_anchor_guard} | BodyMaxDelta: {body_anchor_max_delta:.2f} | "
            f"LookaheadRadius: {body_anchor_lookahead_radius} | TrendTolerance: {body_anchor_trend_tolerance:.2f}"
        )

        # --- STUFE 1: Anchor-Frame finden ---
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({"length": 0.0, "is_frontal": False, "has_feet": False})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else np.array(frame_data))
            if pts.ndim == 3:
                pts = pts[0]

            def is_val(idx):
                return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

            is_frontal = False
            if len(pts) >= 18:
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                if max(angle_h, angle_s) <= frontal_3d_angle_tolerance:
                    is_frontal = True

            all_frames_data.append({"length": length, "is_frontal": is_frontal, "has_feet": is_val(7) or is_val(8)})
            if is_frontal:
                frontal_indices.append(i)

        candidates = frontal_indices if frontal_indices else list(range(len(raw_poses)))
        max_len = max([d["length"] for d in all_frames_data]) if all_frames_data else 1.0

        if not candidates:
            log_messages.append("FEHLER: Keine verwertbaren Frames gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            d = all_frames_data[idx]
            score = (1000.0 if d["has_feet"] else 0.0) + (500.0 if d["is_frontal"] else 0.0) + ((d["length"] / max_len) * 100.0 if max_len > 1e-5 else 0.0)
            if score > best_score:
                best_score, best_idx = score, idx

        log_messages.append(f"-> Referenz-Frame (Anchor) für globalen Ratio-Loop: {best_idx}")

        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        ref_pts = ref_frame_data[0].cpu().numpy() if is_t and ref_frame_data.dim() == 3 else (ref_frame_data.cpu().numpy() if is_t else np.array(ref_frame_data))
        if ref_pts.ndim == 3:
            ref_pts = ref_pts[0]

        tree = {
            0: [1, 2, 3],
            1: [4],
            4: [7],
            7: [10],
            2: [5],
            5: [8],
            8: [11],
            3: [6],
            6: [9],
            9: [12, 13, 14],
            12: [15],
            13: [16],
            16: [18],
            18: [20],
            20: [22],
            14: [17],
            17: [19],
            19: [21],
            21: [23]
        }

        def is_valid_point(p_array, idx):
            return p_array is not None and idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5

        def extract_points(frame_data, copy_array=False):
            if frame_data is None or len(frame_data) == 0:
                return None
            is_tensor_local = isinstance(frame_data, torch.Tensor)
            if is_tensor_local and frame_data.dim() == 3:
                arr = frame_data[0].cpu().numpy()
            elif is_tensor_local:
                arr = frame_data.cpu().numpy()
            else:
                arr = np.array(frame_data)
            if arr.ndim == 3:
                arr = arr[0]
            if copy_array:
                arr = arr.copy()
            return arr

        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child)
                    desc.extend(get_all_descendants(child, tree_map))
            return desc

        def get_height_stable(p_array):
            if p_array is None:
                return 0.0
            if 12 < len(p_array) and np.linalg.norm(p_array[12]) > 1e-5:
                top_y = p_array[12][1]
            else:
                return 0.0
            feet_y = [
                p_array[idx][1]
                for idx in [7, 8, 10, 11, 4, 5]
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]
            return (max(feet_y) - top_y) if feet_y else 0.0

        def get_avg_y(p_array, indices):
            if p_array is None:
                return 0.0
            vals = [
                p_array[idx][1]
                for idx in indices
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]
            return float(sum(vals) / len(vals)) if vals else 0.0

        def get_center(p_array, indices):
            if p_array is None:
                return None
            vals = [
                p_array[idx]
                for idx in indices
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]
            if not vals:
                return None
            return np.mean(np.stack(vals, axis=0), axis=0)

        def get_upper_body_anchor(p_array):
            if upper_body_anchor_mode == "shoulders_only":
                center = get_center(p_array, [16, 17])
                if center is not None:
                    return center
            elif upper_body_anchor_mode == "neck_only":
                if is_valid_point(p_array, 12):
                    return p_array[12].copy()

            center = get_center(p_array, [12, 16, 17])
            if center is not None:
                return center
            if is_valid_point(p_array, 12):
                return p_array[12].copy()
            center = get_center(p_array, [16, 17])
            if center is not None:
                return center
            return None

        def get_body_anchor_y(p_array):
            shoulder_y = get_avg_y(p_array, [16, 17])
            hip_y = get_avg_y(p_array, [1, 2])
            if shoulder_y != 0.0 and hip_y != 0.0:
                return float((shoulder_y + hip_y) * 0.5)
            if shoulder_y != 0.0:
                return float(shoulder_y)
            if hip_y != 0.0:
                return float(hip_y)
            return 0.0

        def get_foot_values_y(p_array):
            if p_array is None:
                return []
            return [
                float(p_array[idx][1])
                for idx in [7, 8, 10, 11, 4, 5]
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]

        def get_foot_anchor_y(p_array, robust=False, percentile=80.0):
            vals = get_foot_values_y(p_array)
            if not vals:
                return 0.0
            if robust:
                return float(np.percentile(np.array(vals, dtype=np.float32), percentile))
            return float(max(vals))

        def median_filter_value(series, idx, radius):
            start = max(0, idx - radius)
            end = min(len(series), idx + radius + 1)
            vals = [float(v) for v in series[start:end] if v is not None and abs(float(v)) > 1e-5]
            return float(np.median(vals)) if vals else 0.0

        def window_range_value(series, idx, radius):
            start = max(0, idx - radius)
            end = min(len(series), idx + radius + 1)
            vals = [float(v) for v in series[start:end] if v is not None and abs(float(v)) > 1e-5]
            return float(max(vals) - min(vals)) if vals else 0.0

        def get_bone_lengths(pts_array):
            def dist(p1, p2):
                if is_valid_point(pts_array, p1) and is_valid_point(pts_array, p2):
                    return float(np.linalg.norm(pts_array[p2] - pts_array[p1]))
                return 0.0

            return {
                "Torso": dist(0, 12),
                "Kopf": dist(12, 15),
                "R_Oberschenkel": dist(2, 5),
                "R_Wade": dist(5, 8),
                "L_Oberschenkel": dist(1, 4),
                "L_Wade": dist(4, 7),
                "R_Arm": dist(17, 19),
                "R_Unterarm": dist(19, 21),
                "L_Arm": dist(16, 18),
                "L_Unterarm": dist(18, 20),
                "Schulterbreite": dist(16, 17),
                "Hueftbreite": dist(1, 2)
            }

        def get_frame_targets(p_array):
            if p_array is None or len(p_array) <= 12 or not is_valid_point(p_array, 0) or not is_valid_point(p_array, 12):
                return {}
            orig_torso_curr = np.linalg.norm(p_array[12] - p_array[0]) if np.linalg.norm(p_array[12]) > 1e-5 else 0.0
            missing_neck_curr = orig_torso_curr * (head_val / 100.0) / 2.0 if is_normalized else head_val / 2.0
            frame_ref_torso = orig_torso_curr + missing_neck_curr
            return {
                k: (v / 100.0 * frame_ref_torso if is_normalized else v)
                for k, v in true_3d_bones.items()
            }

        def build_and_log(pts_source, factor, tgls, final_mode=False, force_all=False, leg_height_factor=1.0):
            pts_b = pts_source.copy()

            if len(pts_b) <= 12 or not is_valid_point(pts_b, 0) or not is_valid_point(pts_b, 12):
                return pts_b

            targets = get_frame_targets(pts_b)

            if force_all or tgls.get("scale_torso", True):
                if is_valid_point(pts_b, 0) and is_valid_point(pts_b, 12):
                    cv = pts_b[12] - pts_b[0]
                    cl = np.linalg.norm(cv)
                    if cl > 1e-5:
                        t_len = targets.get("torso", cl)
                        if final_mode:
                            t_len *= factor
                        f_node = t_len / cl

                        for p, c in [(0, 3), (3, 6), (6, 9), (9, 12)]:
                            if p >= len(pts_b) or c >= len(pts_b):
                                continue
                            if not is_valid_point(pts_b, p) or not is_valid_point(pts_b, c):
                                continue
                            vec = pts_b[c] - pts_b[p]
                            new_c = pts_b[p] + vec * f_node
                            delta = new_c - pts_b[c]
                            pts_b[c] += delta
                            for d in get_all_descendants(c, tree):
                                if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                                    pts_b[d] += delta

            if 15 < len(pts_b) and is_valid_point(pts_b, 12) and is_valid_point(pts_b, 15):
                cv = pts_b[15] - pts_b[12]
                cl = np.linalg.norm(cv)
                if cl > 1e-5:
                    t_len = targets.get("head", cl * 2.0) / 2.0
                    if final_mode and scale_stance_and_head:
                        t_len *= factor
                    delta = (pts_b[12] + (cv / cl * t_len)) - pts_b[15]
                    pts_b[15] += delta

            ops = [
                ("shoulder_width", 12, 17),
                ("shoulder_width", 12, 16),
                ("hip_width", 0, 2),
                ("hip_width", 0, 1),
                ("r_arm", 17, 19),
                ("r_forearm", 19, 21),
                ("l_arm", 16, 18),
                ("l_forearm", 18, 20),
                ("r_thigh", 2, 5),
                ("r_calf", 5, 8),
                ("l_thigh", 1, 4),
                ("l_calf", 4, 7)
            ]

            for key, p_idx, c_idx in ops:
                if p_idx >= len(pts_b) or c_idx >= len(pts_b):
                    continue
                if not is_valid_point(pts_b, p_idx) or not is_valid_point(pts_b, c_idx):
                    continue

                is_allowed = force_all
                if not is_allowed:
                    if "shoulder" in key and tgls.get("scale_shoulders", True):
                        is_allowed = True
                    elif "hip" in key and tgls.get("scale_hips", True):
                        is_allowed = True
                    elif "arm" in key and tgls.get("scale_arms", True):
                        is_allowed = True
                    elif ("thigh" in key or "calf" in key) and tgls.get("scale_legs", True):
                        is_allowed = True

                if not is_allowed:
                    continue

                cv = pts_b[c_idx] - pts_b[p_idx]
                cl = np.linalg.norm(cv)
                if cl < 1e-5:
                    continue

                if key in ["shoulder_width", "hip_width"]:
                    stance_target = targets.get(key, cl * 2.0) / 2.0
                    calib_key = f"calibration_{key}"
                    bone_target = targets.get(calib_key, stance_target * 2.0) / 2.0

                    if final_mode and scale_stance_and_head:
                        stance_target *= factor
                        bone_target *= factor

                    scale_xz_stance = stance_target / cl
                    pos_stance = pts_b[p_idx].copy()
                    pos_stance[0] += cv[0] * scale_xz_stance
                    pos_stance[1] += cv[1]
                    pos_stance[2] += cv[2] * scale_xz_stance

                    delta_stance = pos_stance - pts_b[c_idx]
                    pts_b[c_idx] += delta_stance

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_stance

                    scale_xz_config = bone_target / cl
                    pos_config = pts_b[p_idx].copy()
                    pos_config[0] += cv[0] * scale_xz_config
                    pos_config[1] += cv[1]
                    pos_config[2] += cv[2] * scale_xz_config

                    delta_config = pos_config - pts_b[c_idx]
                    pts_b[c_idx] += delta_config

                else:
                    if key not in targets:
                        continue
                    t_len_normal = targets[key]
                    cal_k = "calibration_" + key
                    t_len_final = targets.get(cal_k, t_len_normal)
                    if final_mode:
                        t_len_final *= factor
                    if final_mode and ("thigh" in key or "calf" in key):
                        t_len_final *= leg_height_factor

                    dir_vec = cv / cl
                    new_c_pos = pts_b[p_idx] + (dir_vec * t_len_final)
                    delta_shift = new_c_pos - pts_b[c_idx]
                    pts_b[c_idx] = new_c_pos

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_shift

            return pts_b

        def solve_two_bone_ik(hip, knee_orig, foot_target, upper_len, lower_len):
            axis = foot_target - hip
            d = float(np.linalg.norm(axis))
            if d < 1e-5 or upper_len < 1e-5 or lower_len < 1e-5:
                return knee_orig.copy(), foot_target.copy(), False, 0.0

            dir_vec = axis / d
            chain_len = upper_len + lower_len
            min_len = abs(upper_len - lower_len) + 1e-4

            d_solved = float(np.clip(d, min_len, max(chain_len - 1e-4, min_len)))

            a = ((upper_len * upper_len) - (lower_len * lower_len) + (d_solved * d_solved)) / (2.0 * d_solved)
            h_sq = max((upper_len * upper_len) - (a * a), 0.0)
            h = math.sqrt(h_sq)

            proj = hip + dir_vec * np.dot(knee_orig - hip, dir_vec)
            bend_vec = knee_orig - proj
            bend_len = float(np.linalg.norm(bend_vec))

            if bend_len < 1e-5:
                fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                if abs(float(np.dot(fallback, dir_vec))) > 0.95:
                    fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                bend_vec = fallback - dir_vec * np.dot(fallback, dir_vec)
                bend_len = float(np.linalg.norm(bend_vec))

            bend_dir = bend_vec / max(bend_len, 1e-5)

            knee_ik = hip + dir_vec * a + bend_dir * h
            knee_new = (knee_orig * (1.0 - ik_knee_bend_strength)) + (knee_ik * ik_knee_bend_strength)

            reachable_ratio = d / max(chain_len, 1e-5)
            return knee_new, foot_target.copy(), True, reachable_ratio

        def compute_target_pelvis_ratio(targets):
            torso_len = float(targets.get("torso", 0.0))

            r_thigh = float(targets.get("calibration_r_thigh", targets.get("r_thigh", 0.0)))
            r_calf = float(targets.get("calibration_r_calf", targets.get("r_calf", 0.0)))
            l_thigh = float(targets.get("calibration_l_thigh", targets.get("l_thigh", 0.0)))
            l_calf = float(targets.get("calibration_l_calf", targets.get("l_calf", 0.0)))

            right_leg = r_thigh + r_calf
            left_leg = l_thigh + l_calf
            leg_len = 0.0

            if right_leg > 1e-5 and left_leg > 1e-5:
                leg_len = (right_leg + left_leg) * 0.5
            elif right_leg > 1e-5:
                leg_len = right_leg
            elif left_leg > 1e-5:
                leg_len = left_leg

            if torso_len <= 1e-5 or leg_len <= 1e-5:
                return 0.45, torso_len, leg_len

            ratio = torso_len / (torso_len + leg_len)
            return float(np.clip(ratio, 0.18, 0.70)), torso_len, leg_len

        def apply_shoulder_foot_solver(pts_base, pts_orig, frame_idx, global_factor, leg_height_factor, prev_pelvis_y):
            pts_s = pts_base.copy()
            debug = {
                "applied": False,
                "solver_mode": body_solver_mode,
                "upper_y": 0.0,
                "ground_y": 0.0,
                "current_pelvis_y": 0.0,
                "target_pelvis_y": 0.0,
                "applied_pelvis_y": 0.0,
                "pelvis_shift_y": 0.0,
                "target_ratio": 0.0,
                "target_torso": 0.0,
                "target_leg": 0.0,
                "r_ik": False,
                "l_ik": False,
                "r_reach": 0.0,
                "l_reach": 0.0,
                "prev_pelvis_y": prev_pelvis_y if prev_pelvis_y is not None else 0.0
            }

            if body_solver_mode == "legacy_v19" or not scale_legs:
                return pts_s, debug, prev_pelvis_y

            upper_anchor = get_upper_body_anchor(pts_s)
            if upper_anchor is None:
                return pts_s, debug, prev_pelvis_y

            ground_y_current = get_foot_anchor_y(pts_orig, robust=True, percentile=ground_shift_percentile)
            ground_y_trend = foot_y_trend[frame_idx] if frame_idx < len(foot_y_trend) else 0.0

            if ground_anchor_mode == "advanced_trend" and ground_y_trend > 1e-5:
                ground_y = ground_y_trend
            else:
                ground_y = ground_y_current

            pelvis_center = get_center(pts_s, [0, 1, 2])
            if pelvis_center is None or ground_y <= 1e-5:
                return pts_s, debug, prev_pelvis_y

            targets = get_frame_targets(pts_s)
            target_ratio, target_torso, target_leg = compute_target_pelvis_ratio(targets)

            upper_y = float(upper_anchor[1])
            current_pelvis_y = float(pelvis_center[1])

            if abs(ground_y - upper_y) < 1e-5:
                return pts_s, debug, prev_pelvis_y

            target_pelvis_y_raw = upper_y + (ground_y - upper_y) * target_ratio
            target_pelvis_y = current_pelvis_y + (target_pelvis_y_raw - current_pelvis_y) * pelvis_ratio_strength

            if pelvis_vertical_smooth > 0.0 and prev_pelvis_y is not None:
                applied_pelvis_y = (prev_pelvis_y * pelvis_vertical_smooth) + (target_pelvis_y * (1.0 - pelvis_vertical_smooth))
            else:
                applied_pelvis_y = target_pelvis_y

            pelvis_shift_y = applied_pelvis_y - current_pelvis_y

            lower_body_nodes = [0, 1, 2, 4, 5, 7, 8, 10, 11]
            for idx in lower_body_nodes:
                if is_valid_point(pts_s, idx):
                    pts_s[idx][1] += pelvis_shift_y

            # Spine zwischen neuer Hüfte und oberem Bereich neu verteilen.
            if is_valid_point(pts_s, 0) and is_valid_point(pts_s, 12):
                root = pts_s[0].copy()
                chest = pts_s[12].copy()
                for idx, alpha in [(3, 0.25), (6, 0.50), (9, 0.75)]:
                    if is_valid_point(pts_s, idx):
                        old = pts_s[idx].copy()
                        new_pos = root * (1.0 - alpha) + chest * alpha
                        pts_s[idx] = new_pos
                        delta = pts_s[idx] - old
                        # Nur lokale Spine-Kinder, nicht Schultern/Kopf komplett verschieben.
                        if idx == 3 and is_valid_point(pts_s, 6):
                            pass
                        if idx == 6 and is_valid_point(pts_s, 9):
                            pass

            # Hüftbreite nach Pelvis-Shift noch einmal symmetrisch stabilisieren.
            if scale_hips and is_valid_point(pts_s, 0):
                for hip_idx in [1, 2]:
                    if is_valid_point(pts_s, hip_idx):
                        vec = pts_s[hip_idx] - pts_s[0]
                        cl = float(np.linalg.norm(vec))
                        if cl > 1e-5:
                            hip_target = targets.get("hip_width", cl * 2.0) / 2.0
                            calib_hip_target = targets.get("calibration_hip_width", hip_target * 2.0) / 2.0
                            final_hip_target = calib_hip_target
                            new_pos = pts_s[0].copy()
                            scale_xz = final_hip_target / cl
                            new_pos[0] += vec[0] * scale_xz
                            new_pos[1] += vec[1]
                            new_pos[2] += vec[2] * scale_xz
                            delta = new_pos - pts_s[hip_idx]
                            pts_s[hip_idx] = new_pos
                            for d in get_all_descendants(hip_idx, tree):
                                if d < len(pts_s) and np.linalg.norm(pts_s[d]) > 1e-5:
                                    pts_s[d] += delta

            if body_solver_mode == "shoulder_foot_ik":
                leg_specs = [
                    {
                        "side": "L",
                        "hip": 1,
                        "knee": 4,
                        "ankle": 7,
                        "toe": 10,
                        "upper_key": "l_thigh",
                        "lower_key": "l_calf"
                    },
                    {
                        "side": "R",
                        "hip": 2,
                        "knee": 5,
                        "ankle": 8,
                        "toe": 11,
                        "upper_key": "r_thigh",
                        "lower_key": "r_calf"
                    }
                ]

                for spec in leg_specs:
                    hip_idx = spec["hip"]
                    knee_idx = spec["knee"]
                    ankle_idx = spec["ankle"]
                    toe_idx = spec["toe"]

                    if not (is_valid_point(pts_s, hip_idx) and is_valid_point(pts_s, knee_idx) and is_valid_point(pts_s, ankle_idx)):
                        continue

                    upper_key = spec["upper_key"]
                    lower_key = spec["lower_key"]

                    upper_len = targets.get("calibration_" + upper_key, targets.get(upper_key, np.linalg.norm(pts_s[knee_idx] - pts_s[hip_idx])))
                    lower_len = targets.get("calibration_" + lower_key, targets.get(lower_key, np.linalg.norm(pts_s[ankle_idx] - pts_s[knee_idx])))

                    upper_len *= global_factor * leg_height_factor
                    lower_len *= global_factor * leg_height_factor

                    foot_current = pts_s[ankle_idx].copy()
                    foot_orig = pts_orig[ankle_idx].copy() if is_valid_point(pts_orig, ankle_idx) else foot_current
                    foot_target = (foot_current * (1.0 - ik_foot_lock_strength)) + (foot_orig * ik_foot_lock_strength)

                    knee_new, ankle_new, ik_ok, reach_ratio = solve_two_bone_ik(
                        pts_s[hip_idx],
                        pts_s[knee_idx],
                        foot_target,
                        float(upper_len),
                        float(lower_len)
                    )

                    old_ankle = pts_s[ankle_idx].copy()
                    pts_s[knee_idx] = knee_new
                    pts_s[ankle_idx] = ankle_new
                    ankle_delta = pts_s[ankle_idx] - old_ankle

                    if is_valid_point(pts_s, toe_idx):
                        pts_s[toe_idx] += ankle_delta

                    if spec["side"] == "L":
                        debug["l_ik"] = ik_ok
                        debug["l_reach"] = reach_ratio
                    else:
                        debug["r_ik"] = ik_ok
                        debug["r_reach"] = reach_ratio

            debug["applied"] = True
            debug["upper_y"] = upper_y
            debug["ground_y"] = ground_y
            debug["current_pelvis_y"] = current_pelvis_y
            debug["target_pelvis_y"] = target_pelvis_y_raw
            debug["applied_pelvis_y"] = applied_pelvis_y
            debug["pelvis_shift_y"] = pelvis_shift_y
            debug["target_ratio"] = target_ratio
            debug["target_torso"] = target_torso
            debug["target_leg"] = target_leg

            return pts_s, debug, applied_pelvis_y

        # --- PHASE 1: GLOBALER HEIGHT-FACTOR ---
        orig_h_global = get_height_stable(ref_pts)
        global_f_scale = 1.0

        log_messages.append("\n--- TOTAL-HEIGHT-ENFORCER ---")
        if orig_h_global > 1e-5:
            for iteration in range(10):
                pts_test = build_and_log(
                    ref_pts,
                    global_f_scale,
                    toggles,
                    final_mode=True,
                    force_all=True,
                    leg_height_factor=1.0
                )
                test_h = get_height_stable(pts_test)
                if test_h < 1e-5:
                    break
                diff = abs(orig_h_global - test_h)
                if diff < 0.1:
                    break
                ratio = orig_h_global / test_h
                global_f_scale *= ratio

        log_messages.append(f"Berechneter universeller Kompressions-Faktor: {global_f_scale:.6f}x")

        # --- PHASE 1.5: TREND-ANALYSE ---
        body_y_series = []
        foot_y_series = []
        body_y_trend = []
        foot_y_trend = []
        body_window_range = []
        foot_window_range = []

        for frame_data in raw_poses:
            pts_pre = extract_points(frame_data, copy_array=False)
            body_y_series.append(get_body_anchor_y(pts_pre) if pts_pre is not None else 0.0)
            foot_y_series.append(get_foot_anchor_y(pts_pre, robust=True, percentile=ground_shift_percentile) if pts_pre is not None else 0.0)

        for idx in range(len(raw_poses)):
            body_y_trend.append(median_filter_value(body_y_series, idx, body_anchor_lookahead_radius))
            foot_y_trend.append(median_filter_value(foot_y_series, idx, body_anchor_lookahead_radius))
            body_window_range.append(window_range_value(body_y_series, idx, body_anchor_lookahead_radius))
            foot_window_range.append(window_range_value(foot_y_series, idx, body_anchor_lookahead_radius))

        if ground_anchor_mode == "advanced_trend":
            log_messages.append(
                f"Advanced Trend Analyse aktiv: Frames={len(raw_poses)} | "
                f"Medianfenster={body_anchor_lookahead_radius * 2 + 1}"
            )

        # --- PHASE 2: FRAMES VERARBEITEN ---
        prev_pts = None
        prev_shift = None
        prev_leg_guard_factor = 1.0
        prev_pelvis_y = None

        leg_guard_events = []
        ground_guard_events = []
        pelvis_events = []

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0:
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())
            if pts.ndim == 3:
                pts = pts[0]

            log_this_frame = (frame_idx % 10 == 0)

            h_before = get_height_stable(pts)
            shoulder_y_before = get_avg_y(pts, [16, 17])
            hip_y_before = get_avg_y(pts, [1, 2])
            body_y_before = get_body_anchor_y(pts)
            foot_y_before_legacy = get_foot_anchor_y(pts, robust=False)
            foot_y_before_robust = get_foot_anchor_y(pts, robust=True, percentile=ground_shift_percentile)
            bones_before = get_bone_lengths(pts) if log_this_frame else None

            no_leg_toggles = toggles.copy()
            no_leg_toggles["scale_legs"] = False

            if body_solver_mode == "legacy_v19":
                pts_base = build_and_log(
                    pts,
                    global_f_scale,
                    toggles,
                    final_mode=True,
                    force_all=False,
                    leg_height_factor=1.0
                )
                solver_debug = {"applied": False, "solver_mode": "legacy_v19"}
                pts_final = pts_base
            else:
                pts_no_legs = build_and_log(
                    pts,
                    global_f_scale,
                    no_leg_toggles,
                    final_mode=True,
                    force_all=False,
                    leg_height_factor=1.0
                )

                guard_target_h = get_height_stable(pts_no_legs)
                guard_raw_h = 0.0
                guard_final_h = 0.0
                guard_raw_rel_error = 0.0
                guard_final_rel_error = 0.0
                guard_factor = 1.0
                guard_unsmoothed = 1.0
                guard_used = False
                guard_iterations = 0
                guard_smoothing_bypassed = False

                pts_candidate, solver_debug, solved_pelvis_y = apply_shoulder_foot_solver(
                    pts_no_legs,
                    pts,
                    frame_idx,
                    global_f_scale,
                    1.0,
                    prev_pelvis_y
                )
                guard_raw_h = get_height_stable(pts_candidate)

                if leg_height_guard and scale_legs and guard_target_h > 1e-5 and guard_raw_h > 1e-5:
                    guard_raw_rel_error = (guard_raw_h - guard_target_h) / guard_target_h

                    if abs(guard_raw_rel_error) > leg_height_guard_tolerance:
                        if guard_raw_rel_error < 0.0 and leg_height_guard_max_factor <= 1.0001:
                            pts_final = pts_candidate
                            guard_final_h = guard_raw_h
                            guard_factor = 1.0
                            guard_unsmoothed = 1.0
                        else:
                            guard_used = True
                            local_factor = 1.0
                            best_pts = pts_candidate
                            best_h = guard_raw_h
                            best_debug = solver_debug
                            best_pelvis_y = solved_pelvis_y

                            for guard_iterations in range(1, 7):
                                if best_h <= 1e-5:
                                    break
                                ratio = guard_target_h / best_h
                                local_factor *= ratio
                                local_factor = float(np.clip(local_factor, leg_height_guard_min_factor, leg_height_guard_max_factor))

                                test_pts, test_debug, test_pelvis_y = apply_shoulder_foot_solver(
                                    pts_no_legs,
                                    pts,
                                    frame_idx,
                                    global_f_scale,
                                    local_factor,
                                    prev_pelvis_y
                                )
                                test_h = get_height_stable(test_pts)

                                best_pts = test_pts
                                best_h = test_h
                                best_debug = test_debug
                                best_pelvis_y = test_pelvis_y

                                if guard_target_h > 1e-5:
                                    rel_err = abs(test_h - guard_target_h) / guard_target_h
                                    if rel_err <= leg_height_guard_tolerance:
                                        break

                            guard_unsmoothed = local_factor

                            if leg_height_guard_smooth > 0.0:
                                smoothed_factor = (prev_leg_guard_factor * leg_height_guard_smooth) + (guard_unsmoothed * (1.0 - leg_height_guard_smooth))
                                smoothed_factor = float(np.clip(smoothed_factor, leg_height_guard_min_factor, leg_height_guard_max_factor))

                                smooth_pts, smooth_debug, smooth_pelvis_y = apply_shoulder_foot_solver(
                                    pts_no_legs,
                                    pts,
                                    frame_idx,
                                    global_f_scale,
                                    smoothed_factor,
                                    prev_pelvis_y
                                )
                                smooth_h = get_height_stable(smooth_pts)
                                rel_err_smoothed = abs(smooth_h - guard_target_h) / guard_target_h if guard_target_h > 1e-5 else 0.0
                                hard_limit = max(leg_height_guard_tolerance * 2.0, 0.04)

                                if rel_err_smoothed > hard_limit:
                                    guard_smoothing_bypassed = True
                                    pts_final = best_pts
                                    solver_debug = best_debug
                                    solved_pelvis_y = best_pelvis_y
                                    guard_factor = guard_unsmoothed
                                    guard_final_h = best_h
                                else:
                                    pts_final = smooth_pts
                                    solver_debug = smooth_debug
                                    solved_pelvis_y = smooth_pelvis_y
                                    guard_factor = smoothed_factor
                                    guard_final_h = smooth_h
                            else:
                                pts_final = best_pts
                                solver_debug = best_debug
                                solved_pelvis_y = best_pelvis_y
                                guard_factor = guard_unsmoothed
                                guard_final_h = best_h

                            prev_leg_guard_factor = guard_factor
                    else:
                        pts_final = pts_candidate
                        guard_final_h = guard_raw_h
                        guard_factor = 1.0
                        guard_unsmoothed = 1.0
                        if leg_height_guard_smooth > 0.0:
                            prev_leg_guard_factor = (prev_leg_guard_factor * leg_height_guard_smooth) + (1.0 * (1.0 - leg_height_guard_smooth))
                        else:
                            prev_leg_guard_factor = 1.0
                else:
                    pts_final = pts_candidate
                    guard_final_h = guard_raw_h

                if solved_pelvis_y is not None:
                    prev_pelvis_y = solved_pelvis_y

                if leg_height_guard and scale_legs and abs(guard_raw_rel_error) > leg_height_guard_tolerance:
                    guard_final_rel_error = (guard_final_h - guard_target_h) / guard_target_h if guard_target_h > 1e-5 else 0.0
                    leg_guard_events.append({
                        "frame": frame_idx,
                        "target_h": guard_target_h,
                        "raw_h": guard_raw_h,
                        "final_h": guard_final_h,
                        "raw_error": guard_raw_rel_error,
                        "final_error": guard_final_rel_error,
                        "factor": guard_factor,
                        "unsmoothed_factor": guard_unsmoothed,
                        "iterations": guard_iterations,
                        "smoothing_bypassed": guard_smoothing_bypassed,
                        "used": guard_used
                    })

            if solver_debug.get("applied", False):
                pelvis_events.append({
                    "frame": frame_idx,
                    "upper_y": solver_debug.get("upper_y", 0.0),
                    "ground_y": solver_debug.get("ground_y", 0.0),
                    "current_pelvis_y": solver_debug.get("current_pelvis_y", 0.0),
                    "target_pelvis_y": solver_debug.get("target_pelvis_y", 0.0),
                    "applied_pelvis_y": solver_debug.get("applied_pelvis_y", 0.0),
                    "pelvis_shift_y": solver_debug.get("pelvis_shift_y", 0.0),
                    "target_ratio": solver_debug.get("target_ratio", 0.0),
                    "r_ik": solver_debug.get("r_ik", False),
                    "l_ik": solver_debug.get("l_ik", False),
                    "r_reach": solver_debug.get("r_reach", 0.0),
                    "l_reach": solver_debug.get("l_reach", 0.0)
                })

            # --- V19 Ground Anchor Airbag ---
            ground_debug = {
                "has_feet": False,
                "mode": ground_anchor_mode,
                "source": "none",
                "orig_anchor": 0.0,
                "new_anchor": 0.0,
                "raw_shift": 0.0,
                "smoothed_shift": 0.0,
                "step_clamped_shift": 0.0,
                "body_guard_shift": 0.0,
                "final_shift": 0.0,
                "step_clamped": False,
                "body_guarded": False,
                "body_ref_y": 0.0,
                "body_before_ground": 0.0,
                "body_delta_raw": 0.0,
                "body_delta_final": 0.0,
                "allowed_body_delta": body_anchor_max_delta,
                "trend_extra": 0.0
            }

            if ground_anchor_mode == "v18_legacy":
                orig_anchor = get_foot_anchor_y(pts, robust=False)
                new_anchor = get_foot_anchor_y(pts_final, robust=False)
                source = "legacy_max"
            else:
                current_orig_anchor = get_foot_anchor_y(pts, robust=True, percentile=ground_shift_percentile)
                trend_orig_anchor = foot_y_trend[frame_idx] if frame_idx < len(foot_y_trend) else 0.0

                if ground_anchor_mode == "advanced_trend" and trend_orig_anchor > 1e-5:
                    foot_spike_limit = max(body_anchor_trend_tolerance * 2.0, 12.0)
                    if current_orig_anchor > 1e-5 and abs(current_orig_anchor - trend_orig_anchor) > foot_spike_limit:
                        orig_anchor = trend_orig_anchor
                        source = "advanced_trend_median"
                    else:
                        orig_anchor = current_orig_anchor
                        source = "advanced_current_robust"
                else:
                    orig_anchor = current_orig_anchor
                    source = "conservative_robust"

                new_anchor = get_foot_anchor_y(pts_final, robust=True, percentile=ground_shift_percentile)

            if orig_anchor > 1e-5 and new_anchor > 1e-5:
                ground_debug["has_feet"] = True
                ground_debug["source"] = source
                ground_debug["orig_anchor"] = orig_anchor
                ground_debug["new_anchor"] = new_anchor

                raw_shift = orig_anchor - new_anchor
                ground_debug["raw_shift"] = raw_shift

                if ground_smooth_factor > 0.0 and prev_shift is not None:
                    smoothed_shift = (prev_shift * ground_smooth_factor) + (raw_shift * (1.0 - ground_smooth_factor))
                else:
                    smoothed_shift = raw_shift

                ground_debug["smoothed_shift"] = smoothed_shift

                if ground_anchor_mode == "v18_legacy":
                    final_shift = smoothed_shift
                    ground_debug["step_clamped_shift"] = final_shift
                    ground_debug["body_guard_shift"] = final_shift
                else:
                    step_clamped_shift = smoothed_shift
                    if ground_shift_max_step > 0.0 and prev_shift is not None:
                        step_delta = smoothed_shift - prev_shift
                        if abs(step_delta) > ground_shift_max_step:
                            step_clamped_shift = prev_shift + float(np.clip(step_delta, -ground_shift_max_step, ground_shift_max_step))
                            ground_debug["step_clamped"] = True

                    ground_debug["step_clamped_shift"] = step_clamped_shift

                    body_guard_shift = step_clamped_shift
                    body_before_ground = get_body_anchor_y(pts_final)
                    current_body_ref = body_y_before
                    trend_body_ref = body_y_trend[frame_idx] if frame_idx < len(body_y_trend) else 0.0

                    if ground_anchor_mode == "advanced_trend" and trend_body_ref > 1e-5:
                        body_ref_y = trend_body_ref
                        local_body_range = body_window_range[frame_idx] if frame_idx < len(body_window_range) else 0.0
                        trend_extra = min(body_anchor_trend_tolerance, max(0.0, local_body_range * 0.50))
                    else:
                        body_ref_y = current_body_ref
                        trend_extra = 0.0

                    allowed_body_delta = body_anchor_max_delta + trend_extra
                    body_after_raw = body_before_ground + step_clamped_shift
                    body_delta_raw = body_after_raw - body_ref_y if body_ref_y > 1e-5 else 0.0

                    if body_anchor_guard and body_ref_y > 1e-5 and body_before_ground > 1e-5 and allowed_body_delta > 0.0:
                        if abs(body_delta_raw) > allowed_body_delta:
                            desired_body_after = body_ref_y + (allowed_body_delta if body_delta_raw > 0.0 else -allowed_body_delta)
                            body_guard_shift = desired_body_after - body_before_ground
                            ground_debug["body_guarded"] = True

                    final_shift = body_guard_shift

                    ground_debug["body_guard_shift"] = body_guard_shift
                    ground_debug["body_ref_y"] = body_ref_y
                    ground_debug["body_before_ground"] = body_before_ground
                    ground_debug["body_delta_raw"] = body_delta_raw
                    ground_debug["allowed_body_delta"] = allowed_body_delta
                    ground_debug["trend_extra"] = trend_extra

                ground_debug["final_shift"] = final_shift

                for j in range(len(pts_final)):
                    if np.linalg.norm(pts_final[j]) > 1e-5:
                        pts_final[j][1] += final_shift

                prev_shift = final_shift

                if ground_debug["body_ref_y"] > 1e-5:
                    body_after_final = get_body_anchor_y(pts_final)
                    ground_debug["body_delta_final"] = body_after_final - ground_debug["body_ref_y"]

                if ground_debug["step_clamped"] or ground_debug["body_guarded"]:
                    ground_guard_events.append({
                        "frame": frame_idx,
                        "raw_shift": ground_debug["raw_shift"],
                        "smoothed_shift": ground_debug["smoothed_shift"],
                        "step_clamped_shift": ground_debug["step_clamped_shift"],
                        "final_shift": ground_debug["final_shift"],
                        "step_clamped": ground_debug["step_clamped"],
                        "body_guarded": ground_debug["body_guarded"],
                        "body_delta_raw": ground_debug["body_delta_raw"],
                        "body_delta_final": ground_debug["body_delta_final"],
                        "allowed_body_delta": ground_debug["allowed_body_delta"],
                        "source": ground_debug["source"]
                    })

            # --- TEMPORAL SMOOTHING ---
            if temporal_smooth_factor > 0.0:
                if prev_pts is None:
                    prev_pts = pts_final.copy()
                else:
                    for j in range(len(pts_final)):
                        if j < len(prev_pts) and np.linalg.norm(pts_final[j]) > 1e-5 and np.linalg.norm(prev_pts[j]) > 1e-5:
                            pts_final[j] = (prev_pts[j] * temporal_smooth_factor) + (pts_final[j] * (1.0 - temporal_smooth_factor))
                        if j < len(prev_pts):
                            prev_pts[j] = pts_final[j].copy()

            if log_this_frame:
                bones_after = get_bone_lengths(pts_final)
                h_after = get_height_stable(pts_final)
                shoulder_y_after = get_avg_y(pts_final, [16, 17])
                hip_y_after = get_avg_y(pts_final, [1, 2])
                body_y_after = get_body_anchor_y(pts_final)
                foot_y_after_legacy = get_foot_anchor_y(pts_final, robust=False)
                foot_y_after_robust = get_foot_anchor_y(pts_final, robust=True, percentile=ground_shift_percentile)

                log_messages.append(f"\n--- FRAME {frame_idx} VERGLEICH (V20 Shoulder-Foot Solver) ---")
                log_messages.append(f"Gesamthöhe              | Vorher: {h_before:.2f} -> Nachher: {h_after:.2f}")
                log_messages.append(f"Schulter-Y Ø            | Vorher: {shoulder_y_before:.2f} -> Nachher: {shoulder_y_after:.2f} | Delta: {(shoulder_y_after - shoulder_y_before):+.2f}")
                log_messages.append(f"Hüft-Y Ø                | Vorher: {hip_y_before:.2f} -> Nachher: {hip_y_after:.2f} | Delta: {(hip_y_after - hip_y_before):+.2f}")
                log_messages.append(f"Body-Y Ø                | Vorher: {body_y_before:.2f} -> Nachher: {body_y_after:.2f} | Delta: {(body_y_after - body_y_before):+.2f}")
                log_messages.append(f"Fuß-Anker Legacy/Robust | Vorher: {foot_y_before_legacy:.2f}/{foot_y_before_robust:.2f} -> Nachher: {foot_y_after_legacy:.2f}/{foot_y_after_robust:.2f}")

                if solver_debug.get("applied", False):
                    log_messages.append(
                        f"V20 Pelvis Solver       | UpperY: {solver_debug.get('upper_y', 0.0):.2f} | GroundY: {solver_debug.get('ground_y', 0.0):.2f} | "
                        f"Ratio: {solver_debug.get('target_ratio', 0.0):.4f} | Pelvis: {solver_debug.get('current_pelvis_y', 0.0):.2f} -> {solver_debug.get('applied_pelvis_y', 0.0):.2f} | "
                        f"Shift: {solver_debug.get('pelvis_shift_y', 0.0):+.2f}"
                    )
                    log_messages.append(
                        f"V20 IK                  | L_OK: {solver_debug.get('l_ik', False)} Reach: {solver_debug.get('l_reach', 0.0):.3f} | "
                        f"R_OK: {solver_debug.get('r_ik', False)} Reach: {solver_debug.get('r_reach', 0.0):.3f}"
                    )
                else:
                    log_messages.append(f"V20 Pelvis Solver       | Nicht aktiv / Mode: {body_solver_mode}")

                if ground_debug["has_feet"]:
                    log_messages.append(
                        f"Ground Anchor           | Mode: {ground_debug['mode']} | Source: {ground_debug['source']} | "
                        f"Orig: {ground_debug['orig_anchor']:.2f} | New: {ground_debug['new_anchor']:.2f}"
                    )
                    log_messages.append(
                        f"Ground Shift            | Raw: {ground_debug['raw_shift']:+.2f} | Smooth: {ground_debug['smoothed_shift']:+.2f} | "
                        f"Step: {ground_debug['step_clamped_shift']:+.2f} | Body: {ground_debug['body_guard_shift']:+.2f} | Final: {ground_debug['final_shift']:+.2f}"
                    )
                    log_messages.append(
                        f"Ground Guards           | StepClamped: {ground_debug['step_clamped']} | BodyGuarded: {ground_debug['body_guarded']} | "
                        f"BodyRaw: {ground_debug['body_delta_raw']:+.2f} | BodyFinal: {ground_debug['body_delta_final']:+.2f} | Allowed: {ground_debug['allowed_body_delta']:.2f}"
                    )
                else:
                    log_messages.append("Ground Anchor           | Keine gültigen Fußpunkte gefunden.")

                log_messages.append("-" * 70)
                for k in bones_before.keys():
                    log_messages.append(f"Knochen: {k.ljust(18)} | Vorher: {bones_before[k]:.2f} -> Nachher: {bones_after[k]:.2f}")

            if is_tensor:
                if frame_data.dim() == 3:
                    raw_poses[frame_idx][0] = torch.from_numpy(pts_final).to(frame_data.device)
                else:
                    raw_poses[frame_idx] = torch.from_numpy(pts_final).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts_final.tolist()

        # --- SUMMARY ---
        log_messages.append("\n--- V20 PELVIS SOLVER SUMMARY ---")
        log_messages.append(f"Pelvis Solver Events: {len(pelvis_events)} | Mode: {body_solver_mode}")

        if pelvis_events:
            strongest_pelvis = max(pelvis_events, key=lambda e: abs(e["pelvis_shift_y"]))
            log_messages.append(
                f"Stärkster Pelvis-Shift: Frame {strongest_pelvis['frame']} | "
                f"UpperY: {strongest_pelvis['upper_y']:.2f} | GroundY: {strongest_pelvis['ground_y']:.2f} | "
                f"Pelvis: {strongest_pelvis['current_pelvis_y']:.2f} -> {strongest_pelvis['applied_pelvis_y']:.2f} | "
                f"Shift: {strongest_pelvis['pelvis_shift_y']:+.2f} | Ratio: {strongest_pelvis['target_ratio']:.4f}"
            )
            log_messages.append("Erste Pelvis-Events:")
            for e in pelvis_events[:20]:
                log_messages.append(
                    f"Frame {str(e['frame']).rjust(4)} | "
                    f"Upper {e['upper_y']:.2f} | Ground {e['ground_y']:.2f} | "
                    f"Pelvis {e['current_pelvis_y']:.2f}->{e['applied_pelvis_y']:.2f} | "
                    f"Shift {e['pelvis_shift_y']:+.2f} | Ratio {e['target_ratio']:.4f} | "
                    f"L_IK {e['l_ik']} R_IK {e['r_ik']}"
                )
            if len(pelvis_events) > 20:
                log_messages.append(f"... weitere {len(pelvis_events) - 20} Pelvis-Events ausgelassen.")

        log_messages.append("\n--- V20 LEG HEIGHT GUARD SUMMARY ---")
        log_messages.append(f"Leg Guard Events: {len(leg_guard_events)}")
        if leg_guard_events:
            worst_raw = max(leg_guard_events, key=lambda e: abs(e["raw_error"]))
            worst_final = max(leg_guard_events, key=lambda e: abs(e["final_error"]))
            log_messages.append(
                f"Stärkster Raw-Ausreißer: Frame {worst_raw['frame']} | "
                f"Target: {worst_raw['target_h']:.2f} | Raw: {worst_raw['raw_h']:.2f} | "
                f"RawError: {worst_raw['raw_error'] * 100.0:+.2f}% | Faktor: {worst_raw['factor']:.5f}"
            )
            log_messages.append(
                f"Stärkster Final-Ausreißer: Frame {worst_final['frame']} | "
                f"Target: {worst_final['target_h']:.2f} | Final: {worst_final['final_h']:.2f} | "
                f"FinalError: {worst_final['final_error'] * 100.0:+.2f}% | Faktor: {worst_final['factor']:.5f}"
            )

        log_messages.append("\n--- V20 GROUND GUARD SUMMARY ---")
        log_messages.append(f"Ground Guard Events: {len(ground_guard_events)} | Mode: {ground_anchor_mode}")
        if ground_guard_events:
            step_count = sum(1 for e in ground_guard_events if e.get("step_clamped", False))
            body_count = sum(1 for e in ground_guard_events if e.get("body_guarded", False))
            worst_shift = max(ground_guard_events, key=lambda e: abs(e["raw_shift"] - e["final_shift"]))
            log_messages.append(f"StepClamp Events: {step_count} | BodyGuard Events: {body_count}")
            log_messages.append(
                f"Stärkste Shift-Korrektur: Frame {worst_shift['frame']} | "
                f"RawShift: {worst_shift['raw_shift']:+.2f} | FinalShift: {worst_shift['final_shift']:+.2f} | "
                f"Source: {worst_shift['source']}"
            )

        # --- CONFIG CLEAN ---
        try:
            config_dict = json.loads(nlf_render_config) if isinstance(nlf_render_config, str) and nlf_render_config.strip() else {}
        except Exception:
            config_dict = {}

        config_dict["anchor_scale"] = 1.0
        config_dict["scale_x_factor"] = 1.0
        clean_config_str = json.dumps(config_dict)

        return (nlf_data_retargeted, "\n".join(log_messages), clean_config_str)



class NLFProportionalRetargeterV21:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20/V22"}),
                "bypass": ("BOOLEAN", {"default": False, "tooltip": "Ignoriert die Node komplett und gibt Originaldaten zurück."}),

                "scale_torso": ("BOOLEAN", {"default": True, "tooltip": "Skaliert den Torso"}),
                "scale_shoulders": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Schultern"}),
                "scale_hips": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Hüften"}),
                "scale_arms": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Arme"}),
                "scale_legs": ("BOOLEAN", {"default": True, "tooltip": "Skaliert die Beine"}),

                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
                "scale_stance_and_head": ("BOOLEAN", {"default": False, "tooltip": "Wendet den Höhen-Korrekturfaktor auch auf Schulter-/Hüftbreite und den Kopf an."}),
                "temporal_smooth_factor": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet die Gelenke gegen Zittern."}),
                "ground_smooth_factor": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Glättet die Auf/Ab-Bewegung des Ground Anchors."}),

                "body_solver_mode": (["legacy_v19", "pelvis_ratio", "shoulder_foot_ik"], {"default": "shoulder_foot_ik", "tooltip": "V21 Solver: legacy_v19 = altes Verhalten, pelvis_ratio = Hüfte proportional, shoulder_foot_ik = Foot-Locked Two-Bone-IK."}),
                "upper_body_anchor_mode": (["neck_shoulders", "shoulders_only", "neck_only"], {"default": "neck_shoulders", "tooltip": "Oberer visueller Anker für den Shoulder-Foot-Solver."}),
                "pelvis_ratio_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Wie stark die Hüfte zur Zielproportion zwischen Schulter und Boden verschoben wird."}),
                "pelvis_vertical_smooth": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "Glättet die gelöste Hüftposition über die Zeit."}),
                "solver_max_pelvis_shift": ("FLOAT", {"default": 35.0, "min": 0.0, "max": 180.0, "step": 1.0, "tooltip": "Maximaler einzelner Pelvis-Shift. 0 = kein Clamp."}),
                "ik_foot_lock_strength": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Wie stark die Original-Fußposition bei IK gehalten wird."}),
                "ik_knee_bend_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Wie stark die originale Knie-Beugerichtung erhalten bleibt."}),
                "ik_reach_margin": ("FLOAT", {"default": 0.985, "min": 0.50, "max": 1.00, "step": 0.005, "tooltip": "Sicherheitsmarge für Bein-Reichweite. Kleiner = konservativer, verhindert Überstrecken."}),

                "leg_height_guard": ("BOOLEAN", {"default": True, "tooltip": "Airbag: verhindert, dass Leg Scaling die Gesamthöhe in einzelnen Frames sprengt."}),
                "leg_height_guard_tolerance": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.25, "step": 0.005, "tooltip": "Relative Toleranz. 0.025 = 2.5 Prozent Abweichung zur Baseline erlaubt."}),
                "leg_height_guard_min_factor": ("FLOAT", {"default": 0.70, "min": 0.30, "max": 1.00, "step": 0.01, "tooltip": "Minimaler lokaler Bein-Kompressionsfaktor."}),
                "leg_height_guard_max_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 1.50, "step": 0.01, "tooltip": "Maximaler lokaler Bein-Faktor."}),
                "leg_height_guard_smooth": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "Glättet den lokalen Bein-Guard-Faktor."}),

                "ground_anchor_mode": (["v18_legacy", "conservative", "advanced_trend"], {"default": "conservative", "tooltip": "Ground-Modus aus V19/V20."}),
                "ground_shift_percentile": ("FLOAT", {"default": 80.0, "min": 50.0, "max": 100.0, "step": 1.0, "tooltip": "Robuster Fußanker statt hartem max()."}),
                "ground_shift_max_step": ("FLOAT", {"default": 14.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Maximale Änderung des Ground-Shifts pro Frame."}),
                "ground_cancellation_limit": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "V21: verhindert, dass Ground Anchor den Pelvis Solver wieder stark zurückkorrigiert. 0.35 = maximal 35 Prozent Gegenzug."}),
                "body_anchor_guard": ("BOOLEAN", {"default": True, "tooltip": "Begrenzt Ground-Shift, wenn Schulter/Hüfte zu stark gepumpt werden."}),
                "body_anchor_max_delta": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 120.0, "step": 1.0, "tooltip": "Maximale erlaubte Body-Y-Abweichung durch Ground-Shift."}),
                "body_anchor_lookahead_radius": ("INT", {"default": 3, "min": 0, "max": 12, "step": 1, "tooltip": "Advanced: Frames davor/danach für Trend-Erkennung."}),
                "body_anchor_trend_tolerance": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 80.0, "step": 1.0, "tooltip": "Advanced: Toleranz für echte kontinuierliche Auf/Ab-Bewegung."}),
            },
            "optional": {
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus dem Scaler wird bereinigt."}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output", "nlf_render_config_clean")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Retargeting"
    DESCRIPTION = "V21: Foot-Locked Shoulder-Foot IK Solver. Upper bleibt stabil, Füße bleiben stabil, Hüfte wird dazwischen gelöst."

    def process(
        self,
        video_nlf_data,
        calibration_data,
        bypass,
        scale_torso,
        scale_shoulders,
        scale_hips,
        scale_arms,
        scale_legs,
        frontal_3d_angle_tolerance,
        scale_stance_and_head,
        temporal_smooth_factor,
        ground_smooth_factor,
        body_solver_mode,
        upper_body_anchor_mode,
        pelvis_ratio_strength,
        pelvis_vertical_smooth,
        solver_max_pelvis_shift,
        ik_foot_lock_strength,
        ik_knee_bend_strength,
        ik_reach_margin,
        leg_height_guard,
        leg_height_guard_tolerance,
        leg_height_guard_min_factor,
        leg_height_guard_max_factor,
        leg_height_guard_smooth,
        ground_anchor_mode,
        ground_shift_percentile,
        ground_shift_max_step,
        ground_cancellation_limit,
        body_anchor_guard,
        body_anchor_max_delta,
        body_anchor_lookahead_radius,
        body_anchor_trend_tolerance,
        nlf_render_config="{}"
    ):
        import copy
        import numpy as np
        import math
        import torch
        import json

        if bypass:
            return (
                video_nlf_data,
                "=== NLF PROPORTIONAL RETARGETER V21 ===\nBYPASS AKTIVIERT: Keine Daten verändert.",
                nlf_render_config if nlf_render_config else "{}"
            )

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V21 (FOOT-LOCKED SHOULDER-FOOT IK SOLVER) ==="]

        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)

        if is_dict:
            raw_poses = nlf_data_retargeted.get("joints3d_nonparam", [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        if raw_poses is None or len(raw_poses) == 0:
            log_messages.append("FEHLER: Keine raw_poses in video_nlf_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)
        head_val = true_3d_bones.get("head", 0.0)

        allowed_solver_modes = ["legacy_v19", "pelvis_ratio", "shoulder_foot_ik"]
        if body_solver_mode not in allowed_solver_modes:
            body_solver_mode = "shoulder_foot_ik"

        allowed_ground_modes = ["v18_legacy", "conservative", "advanced_trend"]
        if ground_anchor_mode not in allowed_ground_modes:
            ground_anchor_mode = "conservative"

        pelvis_ratio_strength = float(np.clip(pelvis_ratio_strength, 0.0, 1.0))
        pelvis_vertical_smooth = float(np.clip(pelvis_vertical_smooth, 0.0, 0.95))
        solver_max_pelvis_shift = max(0.0, float(solver_max_pelvis_shift))
        ik_foot_lock_strength = float(np.clip(ik_foot_lock_strength, 0.0, 1.0))
        ik_knee_bend_strength = float(np.clip(ik_knee_bend_strength, 0.0, 1.0))
        ik_reach_margin = float(np.clip(ik_reach_margin, 0.50, 1.00))

        leg_height_guard_tolerance = max(0.0, float(leg_height_guard_tolerance))
        leg_height_guard_min_factor = float(np.clip(leg_height_guard_min_factor, 0.30, 1.00))
        leg_height_guard_max_factor = float(np.clip(leg_height_guard_max_factor, 1.00, 1.50))
        leg_height_guard_smooth = float(np.clip(leg_height_guard_smooth, 0.0, 0.95))

        ground_shift_percentile = float(np.clip(ground_shift_percentile, 50.0, 100.0))
        ground_shift_max_step = max(0.0, float(ground_shift_max_step))
        ground_cancellation_limit = float(np.clip(ground_cancellation_limit, 0.0, 1.0))
        body_anchor_max_delta = max(0.0, float(body_anchor_max_delta))
        body_anchor_lookahead_radius = int(max(0, min(12, body_anchor_lookahead_radius)))
        body_anchor_trend_tolerance = max(0.0, float(body_anchor_trend_tolerance))

        toggles = {
            "scale_torso": scale_torso,
            "scale_shoulders": scale_shoulders,
            "scale_hips": scale_hips,
            "scale_arms": scale_arms,
            "scale_legs": scale_legs
        }

        log_messages.append(
            f"Body Solver Mode: {body_solver_mode} | UpperAnchor: {upper_body_anchor_mode} | "
            f"PelvisStrength: {pelvis_ratio_strength:.2f} | PelvisSmooth: {pelvis_vertical_smooth:.2f} | "
            f"MaxPelvisShift: {solver_max_pelvis_shift:.2f} | IKFootLock: {ik_foot_lock_strength:.2f} | "
            f"IKKneeBend: {ik_knee_bend_strength:.2f} | IKReachMargin: {ik_reach_margin:.3f}"
        )
        log_messages.append(
            f"Leg Height Guard: {'AKTIV' if leg_height_guard and scale_legs else 'INAKTIV'} | "
            f"Tolerance: {leg_height_guard_tolerance * 100.0:.2f}% | "
            f"Clamp: [{leg_height_guard_min_factor:.3f}, {leg_height_guard_max_factor:.3f}] | "
            f"Smooth: {leg_height_guard_smooth:.2f}"
        )
        log_messages.append(
            f"Ground Anchor Mode: {ground_anchor_mode} | GroundSmooth: {ground_smooth_factor:.2f} | "
            f"FootPercentile: {ground_shift_percentile:.1f} | MaxStep: {ground_shift_max_step:.2f} | "
            f"CancellationLimit: {ground_cancellation_limit:.2f} | BodyGuard: {body_anchor_guard} | "
            f"BodyMaxDelta: {body_anchor_max_delta:.2f} | LookaheadRadius: {body_anchor_lookahead_radius} | "
            f"TrendTolerance: {body_anchor_trend_tolerance:.2f}"
        )

        # --- STUFE 1: Anchor-Frame finden ---
        all_frames_data = []
        frontal_indices = []

        for i, frame_data in enumerate(raw_poses):
            if frame_data is None or len(frame_data) == 0:
                all_frames_data.append({"length": 0.0, "is_frontal": False, "has_feet": False})
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy() if is_tensor else np.array(frame_data))
            if pts.ndim == 3:
                pts = pts[0]

            def is_val_anchor(idx):
                return idx < len(pts) and np.linalg.norm(pts[idx]) > 1e-5

            valid_y = [pts[idx][1] for idx in range(len(pts)) if is_val_anchor(idx)]
            length = (max(valid_y) - min(valid_y)) if valid_y else 0.0

            is_frontal = False
            if len(pts) >= 18:
                dx_h, dz_h = pts[2][0] - pts[1][0], pts[2][2] - pts[1][2]
                angle_h = math.degrees(math.atan2(abs(dz_h), abs(dx_h)))
                dx_s, dz_s = pts[17][0] - pts[16][0], pts[17][2] - pts[16][2]
                angle_s = math.degrees(math.atan2(abs(dz_s), abs(dx_s)))
                if max(angle_h, angle_s) <= frontal_3d_angle_tolerance:
                    is_frontal = True

            has_feet = is_val_anchor(7) or is_val_anchor(8)
            all_frames_data.append({"length": length, "is_frontal": is_frontal, "has_feet": has_feet})

            if is_frontal:
                frontal_indices.append(i)

        candidates = frontal_indices if frontal_indices else list(range(len(raw_poses)))
        max_len = max([d["length"] for d in all_frames_data]) if all_frames_data else 1.0

        if not candidates:
            log_messages.append("FEHLER: Keine verwertbaren Frames gefunden.")
            return (video_nlf_data, "\n".join(log_messages), "{}")

        best_idx, best_score = candidates[0], -1.0
        for idx in candidates:
            d = all_frames_data[idx]
            score = (
                (1000.0 if d["has_feet"] else 0.0)
                + (500.0 if d["is_frontal"] else 0.0)
                + ((d["length"] / max_len) * 100.0 if max_len > 1e-5 else 0.0)
            )
            if score > best_score:
                best_score, best_idx = score, idx

        log_messages.append(f"-> Referenz-Frame (Anchor) für globalen Ratio-Loop: {best_idx}")

        ref_frame_data = raw_poses[best_idx]
        is_t = isinstance(ref_frame_data, torch.Tensor)
        ref_pts = ref_frame_data[0].cpu().numpy() if is_t and ref_frame_data.dim() == 3 else (ref_frame_data.cpu().numpy() if is_t else np.array(ref_frame_data))
        if ref_pts.ndim == 3:
            ref_pts = ref_pts[0]

        tree = {
            0: [1, 2, 3],
            1: [4],
            4: [7],
            7: [10],
            2: [5],
            5: [8],
            8: [11],
            3: [6],
            6: [9],
            9: [12, 13, 14],
            12: [15],
            13: [16],
            16: [18],
            18: [20],
            20: [22],
            14: [17],
            17: [19],
            19: [21],
            21: [23]
        }

        def is_valid_point(p_array, idx):
            return p_array is not None and idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5

        def extract_points(frame_data, copy_array=False):
            if frame_data is None or len(frame_data) == 0:
                return None

            is_tensor_local = isinstance(frame_data, torch.Tensor)

            if is_tensor_local and frame_data.dim() == 3:
                arr = frame_data[0].cpu().numpy()
            elif is_tensor_local:
                arr = frame_data.cpu().numpy()
            else:
                arr = np.array(frame_data)

            if arr.ndim == 3:
                arr = arr[0]

            if copy_array:
                arr = arr.copy()

            return arr

        def get_all_descendants(node, tree_map):
            desc = []
            if node in tree_map:
                for child in tree_map[node]:
                    desc.append(child)
                    desc.extend(get_all_descendants(child, tree_map))
            return desc

        def get_height_stable(p_array):
            if p_array is None:
                return 0.0

            if 12 < len(p_array) and np.linalg.norm(p_array[12]) > 1e-5:
                top_y = p_array[12][1]
            else:
                return 0.0

            feet_y = [
                p_array[idx][1]
                for idx in [7, 8, 10, 11, 4, 5]
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]

            return (max(feet_y) - top_y) if feet_y else 0.0

        def get_avg_y_valid(p_array, indices):
            if p_array is None:
                return 0.0, False

            vals = [
                float(p_array[idx][1])
                for idx in indices
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]

            if not vals:
                return 0.0, False

            return float(sum(vals) / len(vals)), True

        def get_avg_y(p_array, indices):
            val, valid = get_avg_y_valid(p_array, indices)
            return val if valid else 0.0

        def get_center(p_array, indices):
            if p_array is None:
                return None

            vals = [
                p_array[idx]
                for idx in indices
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]

            if not vals:
                return None

            return np.mean(np.stack(vals, axis=0), axis=0)

        def get_upper_body_anchor(p_array):
            if upper_body_anchor_mode == "shoulders_only":
                center = get_center(p_array, [16, 17])
                if center is not None:
                    return center

            elif upper_body_anchor_mode == "neck_only":
                if is_valid_point(p_array, 12):
                    return p_array[12].copy()

            center = get_center(p_array, [12, 16, 17])
            if center is not None:
                return center

            if is_valid_point(p_array, 12):
                return p_array[12].copy()

            center = get_center(p_array, [16, 17])
            if center is not None:
                return center

            return None

        def get_body_anchor_y_valid(p_array):
            shoulder_y, shoulder_valid = get_avg_y_valid(p_array, [16, 17])
            hip_y, hip_valid = get_avg_y_valid(p_array, [1, 2])

            if shoulder_valid and hip_valid:
                return float((shoulder_y + hip_y) * 0.5), True

            if shoulder_valid:
                return float(shoulder_y), True

            if hip_valid:
                return float(hip_y), True

            return 0.0, False

        def get_body_anchor_y(p_array):
            val, valid = get_body_anchor_y_valid(p_array)
            return val if valid else 0.0

        def get_foot_values_y(p_array):
            if p_array is None:
                return []

            return [
                float(p_array[idx][1])
                for idx in [7, 8, 10, 11, 4, 5]
                if idx < len(p_array) and np.linalg.norm(p_array[idx]) > 1e-5
            ]

        def get_foot_anchor_y_valid(p_array, robust=False, percentile=80.0):
            vals = get_foot_values_y(p_array)

            if not vals:
                return 0.0, False

            if robust:
                return float(np.percentile(np.array(vals, dtype=np.float32), percentile)), True

            return float(max(vals)), True

        def get_foot_anchor_y(p_array, robust=False, percentile=80.0):
            val, valid = get_foot_anchor_y_valid(p_array, robust=robust, percentile=percentile)
            return val if valid else 0.0

        def median_filter_value(series, idx, radius):
            start = max(0, idx - radius)
            end = min(len(series), idx + radius + 1)

            vals = []
            for v in series[start:end]:
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        vals.append(fv)
                except Exception:
                    pass

            return float(np.median(vals)) if vals else None

        def window_range_value(series, idx, radius):
            start = max(0, idx - radius)
            end = min(len(series), idx + radius + 1)

            vals = []
            for v in series[start:end]:
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        vals.append(fv)
                except Exception:
                    pass

            return float(max(vals) - min(vals)) if vals else 0.0

        def get_bone_lengths(pts_array):
            def dist(p1, p2):
                if is_valid_point(pts_array, p1) and is_valid_point(pts_array, p2):
                    return float(np.linalg.norm(pts_array[p2] - pts_array[p1]))
                return 0.0

            return {
                "Torso": dist(0, 12),
                "Kopf": dist(12, 15),
                "R_Oberschenkel": dist(2, 5),
                "R_Wade": dist(5, 8),
                "L_Oberschenkel": dist(1, 4),
                "L_Wade": dist(4, 7),
                "R_Arm": dist(17, 19),
                "R_Unterarm": dist(19, 21),
                "L_Arm": dist(16, 18),
                "L_Unterarm": dist(18, 20),
                "Schulterbreite": dist(16, 17),
                "Hueftbreite": dist(1, 2)
            }

        def get_frame_targets(p_array):
            if p_array is None or len(p_array) <= 12 or not is_valid_point(p_array, 0) or not is_valid_point(p_array, 12):
                return {}

            orig_torso_curr = np.linalg.norm(p_array[12] - p_array[0]) if np.linalg.norm(p_array[12]) > 1e-5 else 0.0
            missing_neck_curr = orig_torso_curr * (head_val / 100.0) / 2.0 if is_normalized else head_val / 2.0
            frame_ref_torso = orig_torso_curr + missing_neck_curr

            return {
                k: (v / 100.0 * frame_ref_torso if is_normalized else v)
                for k, v in true_3d_bones.items()
            }

        def get_target_len(targets, key, fallback_len, factor, leg_height_factor=1.0):
            t_len_normal = targets.get(key, fallback_len)
            cal_key = "calibration_" + key
            t_len_final = targets.get(cal_key, t_len_normal)
            return float(t_len_final * factor * leg_height_factor)

        def build_and_log(pts_source, factor, tgls, final_mode=False, force_all=False, leg_height_factor=1.0):
            pts_b = pts_source.copy()

            if len(pts_b) <= 12 or not is_valid_point(pts_b, 0) or not is_valid_point(pts_b, 12):
                return pts_b

            targets = get_frame_targets(pts_b)

            if force_all or tgls.get("scale_torso", True):
                if is_valid_point(pts_b, 0) and is_valid_point(pts_b, 12):
                    cv = pts_b[12] - pts_b[0]
                    cl = np.linalg.norm(cv)

                    if cl > 1e-5:
                        t_len = targets.get("torso", cl)

                        if final_mode:
                            t_len *= factor

                        f_node = t_len / cl

                        for p, c in [(0, 3), (3, 6), (6, 9), (9, 12)]:
                            if p >= len(pts_b) or c >= len(pts_b):
                                continue

                            if not is_valid_point(pts_b, p) or not is_valid_point(pts_b, c):
                                continue

                            vec = pts_b[c] - pts_b[p]
                            new_c = pts_b[p] + vec * f_node
                            delta = new_c - pts_b[c]
                            pts_b[c] += delta

                            for d in get_all_descendants(c, tree):
                                if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                                    pts_b[d] += delta

            if 15 < len(pts_b) and is_valid_point(pts_b, 12) and is_valid_point(pts_b, 15):
                cv = pts_b[15] - pts_b[12]
                cl = np.linalg.norm(cv)

                if cl > 1e-5:
                    t_len = targets.get("head", cl * 2.0) / 2.0

                    if final_mode and scale_stance_and_head:
                        t_len *= factor

                    delta = (pts_b[12] + (cv / cl * t_len)) - pts_b[15]
                    pts_b[15] += delta

            ops = [
                ("shoulder_width", 12, 17),
                ("shoulder_width", 12, 16),
                ("hip_width", 0, 2),
                ("hip_width", 0, 1),
                ("r_arm", 17, 19),
                ("r_forearm", 19, 21),
                ("l_arm", 16, 18),
                ("l_forearm", 18, 20),
                ("r_thigh", 2, 5),
                ("r_calf", 5, 8),
                ("l_thigh", 1, 4),
                ("l_calf", 4, 7)
            ]

            for key, p_idx, c_idx in ops:
                if p_idx >= len(pts_b) or c_idx >= len(pts_b):
                    continue

                if not is_valid_point(pts_b, p_idx) or not is_valid_point(pts_b, c_idx):
                    continue

                is_allowed = force_all

                if not is_allowed:
                    if "shoulder" in key and tgls.get("scale_shoulders", True):
                        is_allowed = True
                    elif "hip" in key and tgls.get("scale_hips", True):
                        is_allowed = True
                    elif "arm" in key and tgls.get("scale_arms", True):
                        is_allowed = True
                    elif ("thigh" in key or "calf" in key) and tgls.get("scale_legs", True):
                        is_allowed = True

                if not is_allowed:
                    continue

                cv = pts_b[c_idx] - pts_b[p_idx]
                cl = np.linalg.norm(cv)

                if cl < 1e-5:
                    continue

                if key in ["shoulder_width", "hip_width"]:
                    stance_target = targets.get(key, cl * 2.0) / 2.0
                    calib_key = f"calibration_{key}"
                    bone_target = targets.get(calib_key, stance_target * 2.0) / 2.0

                    if final_mode and scale_stance_and_head:
                        stance_target *= factor
                        bone_target *= factor

                    scale_xz_stance = stance_target / cl
                    pos_stance = pts_b[p_idx].copy()
                    pos_stance[0] += cv[0] * scale_xz_stance
                    pos_stance[1] += cv[1]
                    pos_stance[2] += cv[2] * scale_xz_stance

                    delta_stance = pos_stance - pts_b[c_idx]
                    pts_b[c_idx] += delta_stance

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_stance

                    scale_xz_config = bone_target / cl
                    pos_config = pts_b[p_idx].copy()
                    pos_config[0] += cv[0] * scale_xz_config
                    pos_config[1] += cv[1]
                    pos_config[2] += cv[2] * scale_xz_config

                    delta_config = pos_config - pts_b[c_idx]
                    pts_b[c_idx] += delta_config

                else:
                    if key not in targets:
                        continue

                    t_len_normal = targets[key]
                    cal_k = "calibration_" + key
                    t_len_final = targets.get(cal_k, t_len_normal)

                    if final_mode:
                        t_len_final *= factor

                    if final_mode and ("thigh" in key or "calf" in key):
                        t_len_final *= leg_height_factor

                    dir_vec = cv / cl
                    new_c_pos = pts_b[p_idx] + (dir_vec * t_len_final)
                    delta_shift = new_c_pos - pts_b[c_idx]
                    pts_b[c_idx] = new_c_pos

                    for d in get_all_descendants(c_idx, tree):
                        if d < len(pts_b) and np.linalg.norm(pts_b[d]) > 1e-5:
                            pts_b[d] += delta_shift

            return pts_b

        def compute_target_pelvis_ratio(targets):
            torso_len = float(targets.get("torso", 0.0))

            r_thigh = float(targets.get("calibration_r_thigh", targets.get("r_thigh", 0.0)))
            r_calf = float(targets.get("calibration_r_calf", targets.get("r_calf", 0.0)))
            l_thigh = float(targets.get("calibration_l_thigh", targets.get("l_thigh", 0.0)))
            l_calf = float(targets.get("calibration_l_calf", targets.get("l_calf", 0.0)))

            right_leg = r_thigh + r_calf
            left_leg = l_thigh + l_calf

            if right_leg > 1e-5 and left_leg > 1e-5:
                leg_len = (right_leg + left_leg) * 0.5
            elif right_leg > 1e-5:
                leg_len = right_leg
            elif left_leg > 1e-5:
                leg_len = left_leg
            else:
                leg_len = 0.0

            if torso_len <= 1e-5 or leg_len <= 1e-5:
                return 0.45, torso_len, leg_len

            ratio = torso_len / (torso_len + leg_len)
            return float(np.clip(ratio, 0.18, 0.70)), torso_len, leg_len

        def get_leg_lengths_for_side(pts_s, targets, side, global_factor, leg_height_factor):
            if side == "L":
                hip_idx, knee_idx, ankle_idx = 1, 4, 7
                upper_key, lower_key = "l_thigh", "l_calf"
            else:
                hip_idx, knee_idx, ankle_idx = 2, 5, 8
                upper_key, lower_key = "r_thigh", "r_calf"

            fallback_upper = 0.0
            fallback_lower = 0.0

            if is_valid_point(pts_s, hip_idx) and is_valid_point(pts_s, knee_idx):
                fallback_upper = float(np.linalg.norm(pts_s[knee_idx] - pts_s[hip_idx]))

            if is_valid_point(pts_s, knee_idx) and is_valid_point(pts_s, ankle_idx):
                fallback_lower = float(np.linalg.norm(pts_s[ankle_idx] - pts_s[knee_idx]))

            upper_len = get_target_len(targets, upper_key, fallback_upper, global_factor, leg_height_factor)
            lower_len = get_target_len(targets, lower_key, fallback_lower, global_factor, leg_height_factor)

            return upper_len, lower_len

        def solve_two_bone_ik(hip, knee_orig, foot_target, upper_len, lower_len):
            axis = foot_target - hip
            d = float(np.linalg.norm(axis))

            if d < 1e-5 or upper_len < 1e-5 or lower_len < 1e-5:
                return knee_orig.copy(), foot_target.copy(), False, 0.0, 0.0

            dir_vec = axis / d
            chain_len = upper_len + lower_len
            min_len = abs(upper_len - lower_len) + 1e-4
            max_len = max(chain_len * ik_reach_margin, min_len + 1e-4)

            d_solved = float(np.clip(d, min_len, max_len))

            a = ((upper_len * upper_len) - (lower_len * lower_len) + (d_solved * d_solved)) / (2.0 * d_solved)
            h_sq = max((upper_len * upper_len) - (a * a), 0.0)
            h = math.sqrt(h_sq)

            proj = hip + dir_vec * np.dot(knee_orig - hip, dir_vec)
            bend_vec = knee_orig - proj
            bend_len = float(np.linalg.norm(bend_vec))

            if bend_len < 1e-5:
                fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)

                if abs(float(np.dot(fallback, dir_vec))) > 0.95:
                    fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                bend_vec = fallback - dir_vec * np.dot(fallback, dir_vec)
                bend_len = float(np.linalg.norm(bend_vec))

            bend_dir = bend_vec / max(bend_len, 1e-5)

            knee_ik = hip + dir_vec * a + bend_dir * h
            knee_new = (knee_orig * (1.0 - ik_knee_bend_strength)) + (knee_ik * ik_knee_bend_strength)

            reachable_ratio = d / max(chain_len, 1e-5)
            solved_distance = float(np.linalg.norm(foot_target - hip))

            return knee_new, foot_target.copy(), True, reachable_ratio, solved_distance

        def get_locked_foot_target(pts_s, pts_orig, ankle_idx):
            foot_current = pts_s[ankle_idx].copy()

            if is_valid_point(pts_orig, ankle_idx):
                foot_orig = pts_orig[ankle_idx].copy()
            else:
                foot_orig = foot_current.copy()

            return (foot_current * (1.0 - ik_foot_lock_strength)) + (foot_orig * ik_foot_lock_strength)

        def clamp_pelvis_shift_by_reach(pts_s, pts_orig, targets, raw_shift, global_factor, leg_height_factor):
            debug = {
                "reach_clamped": False,
                "shift_before_reach": raw_shift,
                "shift_after_reach": raw_shift,
                "l_shift_min": None,
                "l_shift_max": None,
                "r_shift_min": None,
                "r_shift_max": None,
                "combined_min": None,
                "combined_max": None
            }

            combined_min = -float("inf")
            combined_max = float("inf")

            leg_defs = [
                ("L", 1, 7),
                ("R", 2, 8)
            ]

            for side, hip_idx, ankle_idx in leg_defs:
                if not (is_valid_point(pts_s, hip_idx) and is_valid_point(pts_s, ankle_idx)):
                    continue

                upper_len, lower_len = get_leg_lengths_for_side(pts_s, targets, side, global_factor, leg_height_factor)
                chain_len = max((upper_len + lower_len) * ik_reach_margin, 1e-5)

                foot_target = get_locked_foot_target(pts_s, pts_orig, ankle_idx)
                hip = pts_s[hip_idx].copy()

                dx = float(hip[0] - foot_target[0])
                dz = float(hip[2] - foot_target[2])
                horiz_sq = dx * dx + dz * dz

                if horiz_sq >= chain_len * chain_len:
                    max_vertical = 0.0
                else:
                    max_vertical = math.sqrt(max(chain_len * chain_len - horiz_sq, 0.0))

                shift_min = float((foot_target[1] - max_vertical) - hip[1])
                shift_max = float((foot_target[1] + max_vertical) - hip[1])

                combined_min = max(combined_min, shift_min)
                combined_max = min(combined_max, shift_max)

                if side == "L":
                    debug["l_shift_min"] = shift_min
                    debug["l_shift_max"] = shift_max
                else:
                    debug["r_shift_min"] = shift_min
                    debug["r_shift_max"] = shift_max

            if combined_min != -float("inf") and combined_max != float("inf"):
                debug["combined_min"] = combined_min
                debug["combined_max"] = combined_max

                if combined_min <= combined_max:
                    clamped_shift = float(np.clip(raw_shift, combined_min, combined_max))
                else:
                    center = (combined_min + combined_max) * 0.5
                    clamped_shift = float(center)

                if abs(clamped_shift - raw_shift) > 1e-4:
                    debug["reach_clamped"] = True

                debug["shift_after_reach"] = clamped_shift
                return clamped_shift, debug

            return raw_shift, debug

        def redistribute_spine_between_root_and_upper(pts_s):
            if not (is_valid_point(pts_s, 0) and is_valid_point(pts_s, 12)):
                return pts_s

            root = pts_s[0].copy()
            chest = pts_s[12].copy()

            for idx, alpha in [(3, 0.25), (6, 0.50), (9, 0.75)]:
                if is_valid_point(pts_s, idx):
                    pts_s[idx] = root * (1.0 - alpha) + chest * alpha

            return pts_s

        def apply_v21_shoulder_foot_solver(pts_base, pts_orig, frame_idx, global_factor, leg_height_factor, prev_pelvis_y):
            pts_s = pts_base.copy()

            debug = {
                "applied": False,
                "solver_mode": body_solver_mode,
                "upper_y": 0.0,
                "ground_y": 0.0,
                "current_pelvis_y": 0.0,
                "target_pelvis_y": 0.0,
                "target_pelvis_y_raw": 0.0,
                "applied_pelvis_y": 0.0,
                "pelvis_shift_y": 0.0,
                "pelvis_shift_raw": 0.0,
                "pelvis_shift_strength": 0.0,
                "pelvis_shift_smooth": 0.0,
                "pelvis_shift_limited": 0.0,
                "target_ratio": 0.0,
                "target_torso": 0.0,
                "target_leg": 0.0,
                "r_ik": False,
                "l_ik": False,
                "r_reach": 0.0,
                "l_reach": 0.0,
                "r_solved_dist": 0.0,
                "l_solved_dist": 0.0,
                "l_foot_drift": 0.0,
                "r_foot_drift": 0.0,
                "reach_clamped": False,
                "max_shift_clamped": False,
                "prev_pelvis_y": prev_pelvis_y if prev_pelvis_y is not None else 0.0
            }

            if body_solver_mode == "legacy_v19" or not scale_legs:
                return pts_s, debug, prev_pelvis_y

            upper_anchor = get_upper_body_anchor(pts_s)
            if upper_anchor is None:
                return pts_s, debug, prev_pelvis_y

            ground_y_current, ground_current_valid = get_foot_anchor_y_valid(pts_orig, robust=True, percentile=ground_shift_percentile)
            ground_y_trend = foot_y_trend[frame_idx] if frame_idx < len(foot_y_trend) else None

            if ground_anchor_mode == "advanced_trend" and ground_y_trend is not None:
                ground_y = float(ground_y_trend)
            elif ground_current_valid:
                ground_y = float(ground_y_current)
            else:
                return pts_s, debug, prev_pelvis_y

            pelvis_center = get_center(pts_s, [0, 1, 2])
            if pelvis_center is None:
                return pts_s, debug, prev_pelvis_y

            targets = get_frame_targets(pts_s)
            target_ratio, target_torso, target_leg = compute_target_pelvis_ratio(targets)

            upper_y = float(upper_anchor[1])
            current_pelvis_y = float(pelvis_center[1])

            if abs(ground_y - upper_y) < 1e-5:
                return pts_s, debug, prev_pelvis_y

            target_pelvis_y_raw = upper_y + (ground_y - upper_y) * target_ratio
            pelvis_shift_raw = target_pelvis_y_raw - current_pelvis_y
            pelvis_shift_strength = pelvis_shift_raw * pelvis_ratio_strength

            if body_solver_mode == "shoulder_foot_ik":
                pelvis_shift_reach, reach_debug = clamp_pelvis_shift_by_reach(
                    pts_s,
                    pts_orig,
                    targets,
                    pelvis_shift_strength,
                    global_factor,
                    leg_height_factor
                )
            else:
                pelvis_shift_reach = pelvis_shift_strength
                reach_debug = {"reach_clamped": False}

            if solver_max_pelvis_shift > 0.0 and abs(pelvis_shift_reach) > solver_max_pelvis_shift:
                pelvis_shift_limited = float(np.clip(pelvis_shift_reach, -solver_max_pelvis_shift, solver_max_pelvis_shift))
                debug["max_shift_clamped"] = True
            else:
                pelvis_shift_limited = pelvis_shift_reach

            target_pelvis_y_limited = current_pelvis_y + pelvis_shift_limited

            if pelvis_vertical_smooth > 0.0 and prev_pelvis_y is not None:
                applied_pelvis_y = (prev_pelvis_y * pelvis_vertical_smooth) + (target_pelvis_y_limited * (1.0 - pelvis_vertical_smooth))
            else:
                applied_pelvis_y = target_pelvis_y_limited

            pelvis_shift_final = applied_pelvis_y - current_pelvis_y

            if body_solver_mode == "pelvis_ratio":
                lower_body_nodes = [0, 1, 2, 4, 5, 7, 8, 10, 11]

                for idx in lower_body_nodes:
                    if is_valid_point(pts_s, idx):
                        pts_s[idx][1] += pelvis_shift_final

                pts_s = redistribute_spine_between_root_and_upper(pts_s)

            elif body_solver_mode == "shoulder_foot_ik":
                for idx in [0, 1, 2]:
                    if is_valid_point(pts_s, idx):
                        pts_s[idx][1] += pelvis_shift_final

                pts_s = redistribute_spine_between_root_and_upper(pts_s)

                leg_specs = [
                    {
                        "side": "L",
                        "hip": 1,
                        "knee": 4,
                        "ankle": 7,
                        "toe": 10
                    },
                    {
                        "side": "R",
                        "hip": 2,
                        "knee": 5,
                        "ankle": 8,
                        "toe": 11
                    }
                ]

                for spec in leg_specs:
                    side = spec["side"]
                    hip_idx = spec["hip"]
                    knee_idx = spec["knee"]
                    ankle_idx = spec["ankle"]
                    toe_idx = spec["toe"]

                    if not (is_valid_point(pts_s, hip_idx) and is_valid_point(pts_s, knee_idx) and is_valid_point(pts_s, ankle_idx)):
                        continue

                    upper_len, lower_len = get_leg_lengths_for_side(pts_s, targets, side, global_factor, leg_height_factor)
                    foot_target = get_locked_foot_target(pts_s, pts_orig, ankle_idx)

                    old_ankle = pts_s[ankle_idx].copy()
                    old_toe = pts_s[toe_idx].copy() if is_valid_point(pts_s, toe_idx) else None

                    knee_new, ankle_new, ik_ok, reach_ratio, solved_distance = solve_two_bone_ik(
                        pts_s[hip_idx],
                        pts_s[knee_idx],
                        foot_target,
                        float(upper_len),
                        float(lower_len)
                    )

                    pts_s[knee_idx] = knee_new
                    pts_s[ankle_idx] = ankle_new

                    ankle_delta = pts_s[ankle_idx] - old_ankle

                    if is_valid_point(pts_s, toe_idx):
                        if is_valid_point(pts_orig, toe_idx):
                            toe_orig = pts_orig[toe_idx].copy()
                            toe_shifted = old_toe + ankle_delta if old_toe is not None else pts_s[toe_idx] + ankle_delta
                            pts_s[toe_idx] = (toe_shifted * (1.0 - ik_foot_lock_strength)) + (toe_orig * ik_foot_lock_strength)
                        else:
                            pts_s[toe_idx] += ankle_delta

                    drift = float(np.linalg.norm(pts_s[ankle_idx] - foot_target))

                    if side == "L":
                        debug["l_ik"] = ik_ok
                        debug["l_reach"] = reach_ratio
                        debug["l_solved_dist"] = solved_distance
                        debug["l_foot_drift"] = drift
                    else:
                        debug["r_ik"] = ik_ok
                        debug["r_reach"] = reach_ratio
                        debug["r_solved_dist"] = solved_distance
                        debug["r_foot_drift"] = drift

            debug["applied"] = True
            debug["upper_y"] = upper_y
            debug["ground_y"] = ground_y
            debug["current_pelvis_y"] = current_pelvis_y
            debug["target_pelvis_y"] = target_pelvis_y_limited
            debug["target_pelvis_y_raw"] = target_pelvis_y_raw
            debug["applied_pelvis_y"] = applied_pelvis_y
            debug["pelvis_shift_y"] = pelvis_shift_final
            debug["pelvis_shift_raw"] = pelvis_shift_raw
            debug["pelvis_shift_strength"] = pelvis_shift_strength
            debug["pelvis_shift_smooth"] = applied_pelvis_y - target_pelvis_y_limited
            debug["pelvis_shift_limited"] = pelvis_shift_limited
            debug["target_ratio"] = target_ratio
            debug["target_torso"] = target_torso
            debug["target_leg"] = target_leg
            debug["reach_clamped"] = bool(reach_debug.get("reach_clamped", False))

            return pts_s, debug, applied_pelvis_y

        # --- PHASE 1: GLOBALER HEIGHT-FACTOR ---
        orig_h_global = get_height_stable(ref_pts)
        global_f_scale = 1.0

        log_messages.append("\n--- TOTAL-HEIGHT-ENFORCER ---")

        if orig_h_global > 1e-5:
            for iteration in range(10):
                pts_test = build_and_log(
                    ref_pts,
                    global_f_scale,
                    toggles,
                    final_mode=True,
                    force_all=True,
                    leg_height_factor=1.0
                )

                test_h = get_height_stable(pts_test)

                if test_h < 1e-5:
                    break

                diff = abs(orig_h_global - test_h)

                if diff < 0.1:
                    break

                ratio = orig_h_global / test_h
                global_f_scale *= ratio

        log_messages.append(f"Berechneter universeller Kompressions-Faktor: {global_f_scale:.6f}x")

        # --- PHASE 1.5: TREND-ANALYSE ---
        body_y_series = []
        foot_y_series = []
        body_y_trend = []
        foot_y_trend = []
        body_window_range = []
        foot_window_range = []

        for frame_data in raw_poses:
            pts_pre = extract_points(frame_data, copy_array=False)

            if pts_pre is None:
                body_y_series.append(None)
                foot_y_series.append(None)
                continue

            body_val, body_valid = get_body_anchor_y_valid(pts_pre)
            foot_val, foot_valid = get_foot_anchor_y_valid(pts_pre, robust=True, percentile=ground_shift_percentile)

            body_y_series.append(body_val if body_valid else None)
            foot_y_series.append(foot_val if foot_valid else None)

        for idx in range(len(raw_poses)):
            body_y_trend.append(median_filter_value(body_y_series, idx, body_anchor_lookahead_radius))
            foot_y_trend.append(median_filter_value(foot_y_series, idx, body_anchor_lookahead_radius))
            body_window_range.append(window_range_value(body_y_series, idx, body_anchor_lookahead_radius))
            foot_window_range.append(window_range_value(foot_y_series, idx, body_anchor_lookahead_radius))

        if ground_anchor_mode == "advanced_trend":
            log_messages.append(
                f"Advanced Trend Analyse aktiv: Frames={len(raw_poses)} | "
                f"Medianfenster={body_anchor_lookahead_radius * 2 + 1}"
            )

        # --- PHASE 2: FRAMES VERARBEITEN ---
        prev_pts = None
        prev_shift = None
        prev_leg_guard_factor = 1.0
        prev_pelvis_y = None

        leg_guard_events = []
        ground_guard_events = []
        pelvis_events = []
        ik_events = []
        cancellation_events = []

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]

            if frame_data is None or len(frame_data) == 0:
                continue

            is_tensor = isinstance(frame_data, torch.Tensor)
            pts = frame_data[0].cpu().numpy().copy() if is_tensor and frame_data.dim() == 3 else (frame_data.cpu().numpy().copy() if is_tensor else np.array(frame_data).copy())

            if pts.ndim == 3:
                pts = pts[0]

            log_this_frame = (frame_idx % 10 == 0)

            h_before = get_height_stable(pts)
            shoulder_y_before = get_avg_y(pts, [16, 17])
            hip_y_before = get_avg_y(pts, [1, 2])
            body_y_before, body_before_valid = get_body_anchor_y_valid(pts)
            foot_y_before_legacy = get_foot_anchor_y(pts, robust=False)
            foot_y_before_robust = get_foot_anchor_y(pts, robust=True, percentile=ground_shift_percentile)
            bones_before = get_bone_lengths(pts) if log_this_frame else None

            no_leg_toggles = toggles.copy()
            no_leg_toggles["scale_legs"] = False

            guard_target_h = 0.0
            guard_raw_h = 0.0
            guard_final_h = 0.0
            guard_raw_rel_error = 0.0
            guard_final_rel_error = 0.0
            guard_factor = 1.0
            guard_unsmoothed = 1.0
            guard_used = False
            guard_iterations = 0
            guard_smoothing_bypassed = False
            guard_no_expand_blocked = False

            if body_solver_mode == "legacy_v19":
                pts_final = build_and_log(
                    pts,
                    global_f_scale,
                    toggles,
                    final_mode=True,
                    force_all=False,
                    leg_height_factor=1.0
                )
                solver_debug = {"applied": False, "solver_mode": "legacy_v19"}
                solved_pelvis_y = prev_pelvis_y

            else:
                pts_no_legs = build_and_log(
                    pts,
                    global_f_scale,
                    no_leg_toggles,
                    final_mode=True,
                    force_all=False,
                    leg_height_factor=1.0
                )

                guard_target_h = get_height_stable(pts_no_legs)

                pts_candidate, solver_debug, solved_pelvis_y = apply_v21_shoulder_foot_solver(
                    pts_no_legs,
                    pts,
                    frame_idx,
                    global_f_scale,
                    1.0,
                    prev_pelvis_y
                )

                guard_raw_h = get_height_stable(pts_candidate)

                if leg_height_guard and scale_legs and guard_target_h > 1e-5 and guard_raw_h > 1e-5:
                    guard_raw_rel_error = (guard_raw_h - guard_target_h) / guard_target_h

                    if abs(guard_raw_rel_error) > leg_height_guard_tolerance:
                        if guard_raw_rel_error < 0.0 and leg_height_guard_max_factor <= 1.0001:
                            pts_final = pts_candidate
                            guard_final_h = guard_raw_h
                            guard_factor = 1.0
                            guard_unsmoothed = 1.0
                            guard_no_expand_blocked = True

                        else:
                            guard_used = True
                            local_factor = 1.0
                            best_pts = pts_candidate
                            best_h = guard_raw_h
                            best_debug = solver_debug
                            best_pelvis_y = solved_pelvis_y

                            for guard_iterations in range(1, 7):
                                if best_h <= 1e-5:
                                    break

                                ratio = guard_target_h / best_h
                                local_factor *= ratio
                                local_factor = float(np.clip(local_factor, leg_height_guard_min_factor, leg_height_guard_max_factor))

                                test_pts, test_debug, test_pelvis_y = apply_v21_shoulder_foot_solver(
                                    pts_no_legs,
                                    pts,
                                    frame_idx,
                                    global_f_scale,
                                    local_factor,
                                    prev_pelvis_y
                                )

                                test_h = get_height_stable(test_pts)

                                best_pts = test_pts
                                best_h = test_h
                                best_debug = test_debug
                                best_pelvis_y = test_pelvis_y

                                if guard_target_h > 1e-5:
                                    rel_err = abs(test_h - guard_target_h) / guard_target_h

                                    if rel_err <= leg_height_guard_tolerance:
                                        break

                            guard_unsmoothed = local_factor

                            if leg_height_guard_smooth > 0.0:
                                smoothed_factor = (prev_leg_guard_factor * leg_height_guard_smooth) + (guard_unsmoothed * (1.0 - leg_height_guard_smooth))
                                smoothed_factor = float(np.clip(smoothed_factor, leg_height_guard_min_factor, leg_height_guard_max_factor))

                                smooth_pts, smooth_debug, smooth_pelvis_y = apply_v21_shoulder_foot_solver(
                                    pts_no_legs,
                                    pts,
                                    frame_idx,
                                    global_f_scale,
                                    smoothed_factor,
                                    prev_pelvis_y
                                )

                                smooth_h = get_height_stable(smooth_pts)
                                rel_err_smoothed = abs(smooth_h - guard_target_h) / guard_target_h if guard_target_h > 1e-5 else 0.0
                                hard_limit = max(leg_height_guard_tolerance * 2.0, 0.04)

                                if rel_err_smoothed > hard_limit:
                                    guard_smoothing_bypassed = True
                                    pts_final = best_pts
                                    solver_debug = best_debug
                                    solved_pelvis_y = best_pelvis_y
                                    guard_factor = guard_unsmoothed
                                    guard_final_h = best_h

                                else:
                                    pts_final = smooth_pts
                                    solver_debug = smooth_debug
                                    solved_pelvis_y = smooth_pelvis_y
                                    guard_factor = smoothed_factor
                                    guard_final_h = smooth_h

                            else:
                                pts_final = best_pts
                                solver_debug = best_debug
                                solved_pelvis_y = best_pelvis_y
                                guard_factor = guard_unsmoothed
                                guard_final_h = best_h

                            prev_leg_guard_factor = guard_factor

                    else:
                        pts_final = pts_candidate
                        guard_final_h = guard_raw_h
                        guard_factor = 1.0
                        guard_unsmoothed = 1.0

                        if leg_height_guard_smooth > 0.0:
                            prev_leg_guard_factor = (prev_leg_guard_factor * leg_height_guard_smooth) + (1.0 * (1.0 - leg_height_guard_smooth))
                        else:
                            prev_leg_guard_factor = 1.0

                else:
                    pts_final = pts_candidate
                    guard_final_h = guard_raw_h

                if solved_pelvis_y is not None:
                    prev_pelvis_y = solved_pelvis_y

                if leg_height_guard and scale_legs and abs(guard_raw_rel_error) > leg_height_guard_tolerance:
                    guard_final_rel_error = (guard_final_h - guard_target_h) / guard_target_h if guard_target_h > 1e-5 else 0.0

                    leg_guard_events.append({
                        "frame": frame_idx,
                        "target_h": guard_target_h,
                        "raw_h": guard_raw_h,
                        "final_h": guard_final_h,
                        "raw_error": guard_raw_rel_error,
                        "final_error": guard_final_rel_error,
                        "factor": guard_factor,
                        "unsmoothed_factor": guard_unsmoothed,
                        "iterations": guard_iterations,
                        "smoothing_bypassed": guard_smoothing_bypassed,
                        "no_expand_blocked": guard_no_expand_blocked,
                        "used": guard_used
                    })

            if solver_debug.get("applied", False):
                pelvis_event = {
                    "frame": frame_idx,
                    "upper_y": solver_debug.get("upper_y", 0.0),
                    "ground_y": solver_debug.get("ground_y", 0.0),
                    "current_pelvis_y": solver_debug.get("current_pelvis_y", 0.0),
                    "target_pelvis_y_raw": solver_debug.get("target_pelvis_y_raw", 0.0),
                    "target_pelvis_y": solver_debug.get("target_pelvis_y", 0.0),
                    "applied_pelvis_y": solver_debug.get("applied_pelvis_y", 0.0),
                    "pelvis_shift_y": solver_debug.get("pelvis_shift_y", 0.0),
                    "pelvis_shift_raw": solver_debug.get("pelvis_shift_raw", 0.0),
                    "pelvis_shift_limited": solver_debug.get("pelvis_shift_limited", 0.0),
                    "target_ratio": solver_debug.get("target_ratio", 0.0),
                    "r_ik": solver_debug.get("r_ik", False),
                    "l_ik": solver_debug.get("l_ik", False),
                    "r_reach": solver_debug.get("r_reach", 0.0),
                    "l_reach": solver_debug.get("l_reach", 0.0),
                    "r_foot_drift": solver_debug.get("r_foot_drift", 0.0),
                    "l_foot_drift": solver_debug.get("l_foot_drift", 0.0),
                    "reach_clamped": solver_debug.get("reach_clamped", False),
                    "max_shift_clamped": solver_debug.get("max_shift_clamped", False)
                }

                pelvis_events.append(pelvis_event)

                if solver_debug.get("l_ik", False) or solver_debug.get("r_ik", False):
                    ik_events.append(pelvis_event)

            # --- V21 Ground Anchor Airbag ---
            ground_debug = {
                "has_feet": False,
                "mode": ground_anchor_mode,
                "source": "none",
                "orig_anchor": 0.0,
                "new_anchor": 0.0,
                "raw_shift": 0.0,
                "smoothed_shift": 0.0,
                "step_clamped_shift": 0.0,
                "body_guard_shift": 0.0,
                "cancel_guard_shift": 0.0,
                "final_shift": 0.0,
                "step_clamped": False,
                "body_guarded": False,
                "cancel_guarded": False,
                "cancel_ratio": 0.0,
                "body_ref_y": 0.0,
                "body_before_ground": 0.0,
                "body_delta_raw": 0.0,
                "body_delta_final": 0.0,
                "allowed_body_delta": body_anchor_max_delta,
                "trend_extra": 0.0
            }

            if ground_anchor_mode == "v18_legacy":
                orig_anchor, orig_valid = get_foot_anchor_y_valid(pts, robust=False)
                new_anchor, new_valid = get_foot_anchor_y_valid(pts_final, robust=False)
                source = "legacy_max"

            else:
                current_orig_anchor, current_valid = get_foot_anchor_y_valid(pts, robust=True, percentile=ground_shift_percentile)
                trend_orig_anchor = foot_y_trend[frame_idx] if frame_idx < len(foot_y_trend) else None

                if ground_anchor_mode == "advanced_trend" and trend_orig_anchor is not None:
                    foot_spike_limit = max(body_anchor_trend_tolerance * 2.0, 12.0)

                    if current_valid and abs(current_orig_anchor - trend_orig_anchor) > foot_spike_limit:
                        orig_anchor = float(trend_orig_anchor)
                        orig_valid = True
                        source = "advanced_trend_median"
                    elif current_valid:
                        orig_anchor = float(current_orig_anchor)
                        orig_valid = True
                        source = "advanced_current_robust"
                    else:
                        orig_anchor = float(trend_orig_anchor)
                        orig_valid = True
                        source = "advanced_trend_fallback"

                else:
                    orig_anchor = float(current_orig_anchor) if current_valid else 0.0
                    orig_valid = current_valid
                    source = "conservative_robust"

                new_anchor, new_valid = get_foot_anchor_y_valid(pts_final, robust=True, percentile=ground_shift_percentile)

            if orig_valid and new_valid:
                ground_debug["has_feet"] = True
                ground_debug["source"] = source
                ground_debug["orig_anchor"] = orig_anchor
                ground_debug["new_anchor"] = new_anchor

                raw_shift = orig_anchor - new_anchor
                ground_debug["raw_shift"] = raw_shift

                if ground_smooth_factor > 0.0 and prev_shift is not None:
                    smoothed_shift = (prev_shift * ground_smooth_factor) + (raw_shift * (1.0 - ground_smooth_factor))
                else:
                    smoothed_shift = raw_shift

                ground_debug["smoothed_shift"] = smoothed_shift

                if ground_anchor_mode == "v18_legacy":
                    final_shift = smoothed_shift
                    ground_debug["step_clamped_shift"] = final_shift
                    ground_debug["body_guard_shift"] = final_shift
                    ground_debug["cancel_guard_shift"] = final_shift

                else:
                    step_clamped_shift = smoothed_shift

                    if ground_shift_max_step > 0.0 and prev_shift is not None:
                        step_delta = smoothed_shift - prev_shift

                        if abs(step_delta) > ground_shift_max_step:
                            step_clamped_shift = prev_shift + float(np.clip(step_delta, -ground_shift_max_step, ground_shift_max_step))
                            ground_debug["step_clamped"] = True

                    ground_debug["step_clamped_shift"] = step_clamped_shift

                    body_guard_shift = step_clamped_shift
                    body_before_ground, body_before_ground_valid = get_body_anchor_y_valid(pts_final)
                    current_body_ref = body_y_before
                    current_body_ref_valid = body_before_valid
                    trend_body_ref = body_y_trend[frame_idx] if frame_idx < len(body_y_trend) else None

                    if ground_anchor_mode == "advanced_trend" and trend_body_ref is not None:
                        body_ref_y = float(trend_body_ref)
                        body_ref_valid = True
                        local_body_range = body_window_range[frame_idx] if frame_idx < len(body_window_range) else 0.0
                        trend_extra = min(body_anchor_trend_tolerance, max(0.0, local_body_range * 0.50))
                    else:
                        body_ref_y = current_body_ref
                        body_ref_valid = current_body_ref_valid
                        trend_extra = 0.0

                    allowed_body_delta = body_anchor_max_delta + trend_extra
                    body_after_raw = body_before_ground + step_clamped_shift if body_before_ground_valid else 0.0
                    body_delta_raw = body_after_raw - body_ref_y if body_ref_valid and body_before_ground_valid else 0.0

                    if body_anchor_guard and body_ref_valid and body_before_ground_valid and allowed_body_delta > 0.0:
                        if abs(body_delta_raw) > allowed_body_delta:
                            desired_body_after = body_ref_y + (allowed_body_delta if body_delta_raw > 0.0 else -allowed_body_delta)
                            body_guard_shift = desired_body_after - body_before_ground
                            ground_debug["body_guarded"] = True

                    cancel_guard_shift = body_guard_shift
                    pelvis_shift_for_cancel = float(solver_debug.get("pelvis_shift_y", 0.0)) if solver_debug.get("applied", False) else 0.0

                    if solver_debug.get("applied", False) and abs(pelvis_shift_for_cancel) > 1e-5:
                        if (cancel_guard_shift * pelvis_shift_for_cancel) < 0.0:
                            cancel_ratio = abs(cancel_guard_shift) / max(abs(pelvis_shift_for_cancel), 1e-5)
                            ground_debug["cancel_ratio"] = cancel_ratio

                            if cancel_ratio > ground_cancellation_limit:
                                max_counter = abs(pelvis_shift_for_cancel) * ground_cancellation_limit
                                cancel_guard_shift = math.copysign(max_counter, cancel_guard_shift)
                                ground_debug["cancel_guarded"] = True

                    final_shift = cancel_guard_shift

                    ground_debug["body_guard_shift"] = body_guard_shift
                    ground_debug["cancel_guard_shift"] = cancel_guard_shift
                    ground_debug["body_ref_y"] = body_ref_y if body_ref_valid else 0.0
                    ground_debug["body_before_ground"] = body_before_ground if body_before_ground_valid else 0.0
                    ground_debug["body_delta_raw"] = body_delta_raw
                    ground_debug["allowed_body_delta"] = allowed_body_delta
                    ground_debug["trend_extra"] = trend_extra

                ground_debug["final_shift"] = final_shift

                for j in range(len(pts_final)):
                    if np.linalg.norm(pts_final[j]) > 1e-5:
                        pts_final[j][1] += final_shift

                prev_shift = final_shift

                body_after_final, body_after_final_valid = get_body_anchor_y_valid(pts_final)
                if body_after_final_valid and abs(ground_debug["body_ref_y"]) > 1e-5:
                    ground_debug["body_delta_final"] = body_after_final - ground_debug["body_ref_y"]

                if ground_debug["step_clamped"] or ground_debug["body_guarded"] or ground_debug["cancel_guarded"]:
                    ground_guard_events.append({
                        "frame": frame_idx,
                        "raw_shift": ground_debug["raw_shift"],
                        "smoothed_shift": ground_debug["smoothed_shift"],
                        "step_clamped_shift": ground_debug["step_clamped_shift"],
                        "body_guard_shift": ground_debug["body_guard_shift"],
                        "cancel_guard_shift": ground_debug["cancel_guard_shift"],
                        "final_shift": ground_debug["final_shift"],
                        "step_clamped": ground_debug["step_clamped"],
                        "body_guarded": ground_debug["body_guarded"],
                        "cancel_guarded": ground_debug["cancel_guarded"],
                        "cancel_ratio": ground_debug["cancel_ratio"],
                        "body_delta_raw": ground_debug["body_delta_raw"],
                        "body_delta_final": ground_debug["body_delta_final"],
                        "allowed_body_delta": ground_debug["allowed_body_delta"],
                        "source": ground_debug["source"]
                    })

                if ground_debug["cancel_guarded"]:
                    cancellation_events.append({
                        "frame": frame_idx,
                        "pelvis_shift": float(solver_debug.get("pelvis_shift_y", 0.0)),
                        "shift_before": ground_debug["body_guard_shift"],
                        "shift_after": ground_debug["final_shift"],
                        "cancel_ratio": ground_debug["cancel_ratio"]
                    })

            # --- TEMPORAL SMOOTHING ---
            if temporal_smooth_factor > 0.0:
                if prev_pts is None:
                    prev_pts = pts_final.copy()
                else:
                    for j in range(len(pts_final)):
                        if j < len(prev_pts) and np.linalg.norm(pts_final[j]) > 1e-5 and np.linalg.norm(prev_pts[j]) > 1e-5:
                            pts_final[j] = (prev_pts[j] * temporal_smooth_factor) + (pts_final[j] * (1.0 - temporal_smooth_factor))

                        if j < len(prev_pts):
                            prev_pts[j] = pts_final[j].copy()

            if log_this_frame:
                bones_after = get_bone_lengths(pts_final)
                h_after = get_height_stable(pts_final)
                shoulder_y_after = get_avg_y(pts_final, [16, 17])
                hip_y_after = get_avg_y(pts_final, [1, 2])
                body_y_after = get_body_anchor_y(pts_final)
                foot_y_after_legacy = get_foot_anchor_y(pts_final, robust=False)
                foot_y_after_robust = get_foot_anchor_y(pts_final, robust=True, percentile=ground_shift_percentile)

                log_messages.append(f"\n--- FRAME {frame_idx} VERGLEICH (V21 Foot-Locked Solver) ---")
                log_messages.append(f"Gesamthöhe              | Vorher: {h_before:.2f} -> Nachher: {h_after:.2f}")
                log_messages.append(f"Schulter-Y Ø            | Vorher: {shoulder_y_before:.2f} -> Nachher: {shoulder_y_after:.2f} | Delta: {(shoulder_y_after - shoulder_y_before):+.2f}")
                log_messages.append(f"Hüft-Y Ø                | Vorher: {hip_y_before:.2f} -> Nachher: {hip_y_after:.2f} | Delta: {(hip_y_after - hip_y_before):+.2f}")
                log_messages.append(f"Body-Y Ø                | Vorher: {body_y_before:.2f} -> Nachher: {body_y_after:.2f} | Delta: {(body_y_after - body_y_before):+.2f}")
                log_messages.append(f"Fuß-Anker Legacy/Robust | Vorher: {foot_y_before_legacy:.2f}/{foot_y_before_robust:.2f} -> Nachher: {foot_y_after_legacy:.2f}/{foot_y_after_robust:.2f}")

                if solver_debug.get("applied", False):
                    log_messages.append(
                        f"V21 Pelvis Solver       | UpperY: {solver_debug.get('upper_y', 0.0):.2f} | GroundY: {solver_debug.get('ground_y', 0.0):.2f} | "
                        f"Ratio: {solver_debug.get('target_ratio', 0.0):.4f} | Pelvis: {solver_debug.get('current_pelvis_y', 0.0):.2f} -> {solver_debug.get('applied_pelvis_y', 0.0):.2f} | "
                        f"ShiftRaw: {solver_debug.get('pelvis_shift_raw', 0.0):+.2f} | ShiftFinal: {solver_debug.get('pelvis_shift_y', 0.0):+.2f}"
                    )
                    log_messages.append(
                        f"V21 Pelvis Guards       | ReachClamped: {solver_debug.get('reach_clamped', False)} | MaxShiftClamped: {solver_debug.get('max_shift_clamped', False)} | "
                        f"LimitedShift: {solver_debug.get('pelvis_shift_limited', 0.0):+.2f}"
                    )
                    log_messages.append(
                        f"V21 IK                  | L_OK: {solver_debug.get('l_ik', False)} Reach: {solver_debug.get('l_reach', 0.0):.3f} Drift: {solver_debug.get('l_foot_drift', 0.0):.3f} | "
                        f"R_OK: {solver_debug.get('r_ik', False)} Reach: {solver_debug.get('r_reach', 0.0):.3f} Drift: {solver_debug.get('r_foot_drift', 0.0):.3f}"
                    )
                else:
                    log_messages.append(f"V21 Pelvis Solver       | Nicht aktiv / Mode: {body_solver_mode}")

                if ground_debug["has_feet"]:
                    log_messages.append(
                        f"Ground Anchor           | Mode: {ground_debug['mode']} | Source: {ground_debug['source']} | "
                        f"Orig: {ground_debug['orig_anchor']:.2f} | New: {ground_debug['new_anchor']:.2f}"
                    )
                    log_messages.append(
                        f"Ground Shift            | Raw: {ground_debug['raw_shift']:+.2f} | Smooth: {ground_debug['smoothed_shift']:+.2f} | "
                        f"Step: {ground_debug['step_clamped_shift']:+.2f} | Body: {ground_debug['body_guard_shift']:+.2f} | "
                        f"Cancel: {ground_debug['cancel_guard_shift']:+.2f} | Final: {ground_debug['final_shift']:+.2f}"
                    )
                    log_messages.append(
                        f"Ground Guards           | StepClamped: {ground_debug['step_clamped']} | BodyGuarded: {ground_debug['body_guarded']} | "
                        f"CancelGuarded: {ground_debug['cancel_guarded']} | CancelRatio: {ground_debug['cancel_ratio']:.3f} | "
                        f"BodyRaw: {ground_debug['body_delta_raw']:+.2f} | BodyFinal: {ground_debug['body_delta_final']:+.2f} | Allowed: {ground_debug['allowed_body_delta']:.2f}"
                    )
                else:
                    log_messages.append("Ground Anchor           | Keine gültigen Fußpunkte gefunden.")

                log_messages.append("-" * 70)

                for k in bones_before.keys():
                    log_messages.append(f"Knochen: {k.ljust(18)} | Vorher: {bones_before[k]:.2f} -> Nachher: {bones_after[k]:.2f}")

            if is_tensor:
                if frame_data.dim() == 3:
                    raw_poses[frame_idx][0] = torch.from_numpy(pts_final).to(frame_data.device)
                else:
                    raw_poses[frame_idx] = torch.from_numpy(pts_final).to(frame_data.device)
            else:
                raw_poses[frame_idx] = pts_final.tolist()

        # --- SUMMARY ---
        log_messages.append("\n--- V21 PELVIS SOLVER SUMMARY ---")
        log_messages.append(f"Pelvis Solver Events: {len(pelvis_events)} | Mode: {body_solver_mode}")

        if pelvis_events:
            strongest_pelvis = max(pelvis_events, key=lambda e: abs(e["pelvis_shift_y"]))

            log_messages.append(
                f"Stärkster Pelvis-Shift: Frame {strongest_pelvis['frame']} | "
                f"UpperY: {strongest_pelvis['upper_y']:.2f} | GroundY: {strongest_pelvis['ground_y']:.2f} | "
                f"Pelvis: {strongest_pelvis['current_pelvis_y']:.2f} -> {strongest_pelvis['applied_pelvis_y']:.2f} | "
                f"Shift: {strongest_pelvis['pelvis_shift_y']:+.2f} | Ratio: {strongest_pelvis['target_ratio']:.4f} | "
                f"ReachClamped: {strongest_pelvis['reach_clamped']} | MaxClamped: {strongest_pelvis['max_shift_clamped']}"
            )

            log_messages.append("Erste Pelvis-Events:")

            for e in pelvis_events[:20]:
                log_messages.append(
                    f"Frame {str(e['frame']).rjust(4)} | "
                    f"Upper {e['upper_y']:.2f} | Ground {e['ground_y']:.2f} | "
                    f"Pelvis {e['current_pelvis_y']:.2f}->{e['applied_pelvis_y']:.2f} | "
                    f"Shift {e['pelvis_shift_y']:+.2f} | Ratio {e['target_ratio']:.4f} | "
                    f"L_IK {e['l_ik']} R_IK {e['r_ik']} | "
                    f"LReach {e['l_reach']:.3f} RReach {e['r_reach']:.3f} | "
                    f"LDrift {e['l_foot_drift']:.3f} RDrift {e['r_foot_drift']:.3f}"
                )

            if len(pelvis_events) > 20:
                log_messages.append(f"... weitere {len(pelvis_events) - 20} Pelvis-Events ausgelassen.")

        log_messages.append("\n--- V21 IK SUMMARY ---")
        log_messages.append(f"IK Events: {len(ik_events)}")

        if ik_events:
            worst_l_drift = max(ik_events, key=lambda e: e.get("l_foot_drift", 0.0))
            worst_r_drift = max(ik_events, key=lambda e: e.get("r_foot_drift", 0.0))
            worst_reach = max(ik_events, key=lambda e: max(e.get("l_reach", 0.0), e.get("r_reach", 0.0)))

            log_messages.append(
                f"Stärkster L-Foot-Drift: Frame {worst_l_drift['frame']} | Drift: {worst_l_drift.get('l_foot_drift', 0.0):.4f} | Reach: {worst_l_drift.get('l_reach', 0.0):.3f}"
            )
            log_messages.append(
                f"Stärkster R-Foot-Drift: Frame {worst_r_drift['frame']} | Drift: {worst_r_drift.get('r_foot_drift', 0.0):.4f} | Reach: {worst_r_drift.get('r_reach', 0.0):.3f}"
            )
            log_messages.append(
                f"Höchste Reach-Auslastung: Frame {worst_reach['frame']} | LReach: {worst_reach.get('l_reach', 0.0):.3f} | RReach: {worst_reach.get('r_reach', 0.0):.3f}"
            )

        log_messages.append("\n--- V21 LEG HEIGHT GUARD SUMMARY ---")
        log_messages.append(f"Leg Guard Events: {len(leg_guard_events)}")

        if leg_guard_events:
            worst_raw = max(leg_guard_events, key=lambda e: abs(e["raw_error"]))
            worst_final = max(leg_guard_events, key=lambda e: abs(e["final_error"]))
            no_expand_count = sum(1 for e in leg_guard_events if e.get("no_expand_blocked", False))

            log_messages.append(
                f"Stärkster Raw-Ausreißer: Frame {worst_raw['frame']} | "
                f"Target: {worst_raw['target_h']:.2f} | Raw: {worst_raw['raw_h']:.2f} | "
                f"RawError: {worst_raw['raw_error'] * 100.0:+.2f}% | Faktor: {worst_raw['factor']:.5f}"
            )
            log_messages.append(
                f"Stärkster Final-Ausreißer: Frame {worst_final['frame']} | "
                f"Target: {worst_final['target_h']:.2f} | Final: {worst_final['final_h']:.2f} | "
                f"FinalError: {worst_final['final_error'] * 100.0:+.2f}% | Faktor: {worst_final['factor']:.5f}"
            )
            log_messages.append(f"No-Expand-blockierte negative Events: {no_expand_count}")

        log_messages.append("\n--- V21 GROUND GUARD SUMMARY ---")
        log_messages.append(f"Ground Guard Events: {len(ground_guard_events)} | Mode: {ground_anchor_mode}")

        if ground_guard_events:
            step_count = sum(1 for e in ground_guard_events if e.get("step_clamped", False))
            body_count = sum(1 for e in ground_guard_events if e.get("body_guarded", False))
            cancel_count = sum(1 for e in ground_guard_events if e.get("cancel_guarded", False))
            worst_shift = max(ground_guard_events, key=lambda e: abs(e["raw_shift"] - e["final_shift"]))

            log_messages.append(f"StepClamp Events: {step_count} | BodyGuard Events: {body_count} | CancellationGuard Events: {cancel_count}")
            log_messages.append(
                f"Stärkste Shift-Korrektur: Frame {worst_shift['frame']} | "
                f"RawShift: {worst_shift['raw_shift']:+.2f} | FinalShift: {worst_shift['final_shift']:+.2f} | "
                f"CancelRatio: {worst_shift.get('cancel_ratio', 0.0):.3f} | Source: {worst_shift['source']}"
            )

        log_messages.append("\n--- V21 CANCELLATION SUMMARY ---")
        log_messages.append(f"Ground-vs-Pelvis Cancellation Events: {len(cancellation_events)}")

        if cancellation_events:
            worst_cancel = max(cancellation_events, key=lambda e: e.get("cancel_ratio", 0.0))
            log_messages.append(
                f"Stärkste Solver-Gegenkorrektur: Frame {worst_cancel['frame']} | "
                f"PelvisShift: {worst_cancel['pelvis_shift']:+.2f} | "
                f"GroundBefore: {worst_cancel['shift_before']:+.2f} | "
                f"GroundAfter: {worst_cancel['shift_after']:+.2f} | "
                f"CancelRatio: {worst_cancel['cancel_ratio']:.3f}"
            )

        # --- CONFIG CLEAN ---
        try:
            config_dict = json.loads(nlf_render_config) if isinstance(nlf_render_config, str) and nlf_render_config.strip() else {}
        except Exception:
            config_dict = {}

        config_dict["anchor_scale"] = 1.0
        config_dict["scale_x_factor"] = 1.0

        clean_config_str = json.dumps(config_dict)

        return (nlf_data_retargeted, "\n".join(log_messages), clean_config_str)
