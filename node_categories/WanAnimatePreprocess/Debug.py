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



class SavePoseDataNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "filename_prefix": ("STRING", {"default": "minimal_keypoints"}),
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "save_json"
    CATEGORY = "WanAnimatePreprocess/Debug"
    DESCRIPTION = "Speichert NUR die reinen Keypoints (X, Y, Scores) und die Bildgröße in einer JSON-Datei."

    def save_json(self, pose_data, filename_prefix):
        import json
        import os
        import numpy as np
        import folder_paths
        
        output_dir = folder_paths.get_output_directory()
        
        # Filename generieren (hochzählend)
        counter = 0
        while True:
            filename = f"{filename_prefix}_{counter:05d}.json"
            full_path = os.path.join(output_dir, filename)
            if not os.path.exists(full_path):
                break
            counter += 1

        # Hier sammeln wir die aufgeräumten Daten
        minimal_frames = []
        
        # Wir holen uns nur die aktuellen Pose-Metadaten (einen Eintrag pro Frame)
        pose_metas = pose_data.get("pose_metas", [])
        
        for meta in pose_metas:
            if meta is None:
                continue
            
            # Hilfsfunktion, um Numpy-Arrays sauber in JSON-Listen umzuwandeln
            def to_list(arr):
                if arr is None: return None
                if isinstance(arr, np.ndarray): return arr.tolist()
                if isinstance(arr, list): return arr
                return None

            # Wir bauen uns ein komplett sauberes Dictionary nur mit dem Nötigsten
            frame_data = {
                "width": getattr(meta, "width", 0),
                "height": getattr(meta, "height", 0),
                "kps_body": to_list(getattr(meta, "kps_body", None)),
                "kps_body_p": to_list(getattr(meta, "kps_body_p", None)),
                "kps_lhand": to_list(getattr(meta, "kps_lhand", None)),
                "kps_lhand_p": to_list(getattr(meta, "kps_lhand_p", None)),
                "kps_rhand": to_list(getattr(meta, "kps_rhand", None)),
                "kps_rhand_p": to_list(getattr(meta, "kps_rhand_p", None)),
                "kps_face": to_list(getattr(meta, "kps_face", None)),
                "kps_face_p": to_list(getattr(meta, "kps_face_p", None)),
            }
            
            minimal_frames.append(frame_data)

        try:
            # Speichere die saubere Liste als JSON
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(minimal_frames, f, indent=4, ensure_ascii=False)
                
            print(f"✅ Keypoints erfolgreich als JSON exportiert: {full_path}")
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Keypoints: {e}")

        # Gib die Original-PoseData unverändert an die nächste Node weiter
        return (pose_data,)


class PoseDataHipHandDebugV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "hip_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01, "tooltip": "Skaliert die Hüftbreite."}),
                "auto_hand_adjust": ("BOOLEAN", {"default": True, "tooltip": "Wenn an, bewegen sich die Hände automatisch mit der Hüfte mit."}),
                "move_elbows": ("BOOLEAN", {"default": True, "tooltip": "Wenn an, werden auch die Ellbogen verschoben. Wenn aus, nur Handgelenke und Hände."}),
                "hand_offset": ("FLOAT", {"default": 0.0, "min": -500.0, "max": 500.0, "step": 1.0, "tooltip": "Manueller Zusatz-Offset für die Hände (funktioniert jetzt auch ohne Auto-Adjust)."}),
                "smooth_hand_entry": ("BOOLEAN", {"default": True, "tooltip": "Verhindert Sprünge, wenn die Hand die Hüfthöhe passiert."}),
                "smooth_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "person_index": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Debug"
    DESCRIPTION = "V2: Debugging Node für Hüftbreite. Erlaubt nun Hand-Offset ohne Auto-Adjust und das Ausschließen der Ellbogen."

    def process(self, pose_data, hip_scale, auto_hand_adjust, move_elbows, hand_offset, smooth_hand_entry, smooth_threshold, person_index):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        indices = (
            [person_index]
            if isinstance(person_index, int) and person_index >= 0 and person_index < len(pose_metas)
            else list(range(len(pose_metas)))
        )

        for idx in indices:
            if pose_metas[idx] is not None:
                self._apply_hip_logic(
                    pose_metas[idx], 
                    hip_scale, 
                    auto_hand_adjust, 
                    move_elbows,
                    hand_offset, 
                    smooth_hand_entry, 
                    smooth_threshold
                )

        return (pose_data_copy,)

    def _apply_hip_logic(self, meta, hip_scale, auto_hand_adjust, move_elbows, hand_offset, smooth_hand_entry, smooth_threshold):
        body_arr = getattr(meta, "kps_body", None)
        width = getattr(meta, "width", 1024)
        height = getattr(meta, "height", 1024)

        if body_arr is None:
            return
        
        # Indizes: 8 = Rechte Hüfte, 11 = Linke Hüfte
        if 8 >= len(body_arr) or 11 >= len(body_arr):
            return

        # 1. Hole Original-Koordinaten der Hüften
        hip_r_orig = self._extract_coords(body_arr[8])
        hip_l_orig = self._extract_coords(body_arr[11])

        if hip_r_orig is None or hip_l_orig is None:
            return

        # Berechne das Zentrum zwischen den Hüften
        center_x = (hip_r_orig[0] + hip_l_orig[0]) / 2.0

        # 2. Skaliere Hüften
        # Rechte Hüfte (Index 8)
        dist_r = hip_r_orig[0] - center_x
        new_hip_r_x = center_x + (dist_r * hip_scale)
        self._assign_point(body_arr, 8, new_hip_r_x, hip_r_orig[1])

        # Linke Hüfte (Index 11)
        dist_l = hip_l_orig[0] - center_x
        new_hip_l_x = center_x + (dist_l * hip_scale)
        self._assign_point(body_arr, 11, new_hip_l_x, hip_l_orig[1])

        # ---------------------------------------------------------
        # HAND LOGIK (V2 Update)
        # ---------------------------------------------------------
        
        # Berechne Delta (Verschiebung durch Hüfte) nur wenn auto_hand_adjust an ist
        delta_right = 0.0
        delta_left = 0.0
        
        if auto_hand_adjust:
            delta_right = new_hip_r_x - hip_r_orig[0]
            delta_left = new_hip_l_x - hip_l_orig[0]

        # Base shift: Hüftbewegung + Manueller Offset
        # Der manuelle Offset wirkt nun immer, auch wenn delta 0 ist.
        base_shift_right = delta_right - float(hand_offset)
        base_shift_left = delta_left + float(hand_offset)

        # --- SMOOTHING LOGIK RECHTS ---
        factor_right = 1.0
        if smooth_hand_entry and 4 < len(body_arr):
            wrist_coords = self._extract_coords(body_arr[4])
            hip_y = hip_r_orig[1]
            if wrist_coords is not None and hip_y > 0:
                wrist_y = wrist_coords[1]
                safe_zone_px = float(height) * float(smooth_threshold)
                start_transition_y = hip_y - safe_zone_px

                if wrist_y < start_transition_y:
                    factor_right = 0.0
                elif wrist_y >= hip_y:
                    factor_right = 1.0
                else:
                    if safe_zone_px > 0:
                        progress = (wrist_y - start_transition_y) / safe_zone_px
                        factor_right = max(0.0, min(1.0, progress))

        # --- SMOOTHING LOGIK LINKS ---
        factor_left = 1.0
        if smooth_hand_entry and 7 < len(body_arr):
            wrist_coords = self._extract_coords(body_arr[7])
            hip_y = hip_l_orig[1]
            if wrist_coords is not None and hip_y > 0:
                wrist_y = wrist_coords[1]
                safe_zone_px = float(height) * float(smooth_threshold)
                start_transition_y = hip_y - safe_zone_px

                if wrist_y < start_transition_y:
                    factor_left = 0.0
                elif wrist_y >= hip_y:
                    factor_left = 1.0
                else:
                    if safe_zone_px > 0:
                        progress = (wrist_y - start_transition_y) / safe_zone_px
                        factor_left = max(0.0, min(1.0, progress))

        final_shift_right = base_shift_right * factor_right
        final_shift_left = base_shift_left * factor_left

        # --- ANWENDEN (V2: Ellbogen optional) ---
        
        # Indizes Rechts: 3 (Ellbogen), 4 (Handgelenk)
        indices_right = [4]
        if move_elbows:
            indices_right.append(3)
            
        for idx in indices_right:
            if idx < len(body_arr):
                self._shift_point_x(body_arr, idx, final_shift_right)
        
        if hasattr(meta, "kps_rhand") and meta.kps_rhand is not None:
            for i in range(len(meta.kps_rhand)):
                self._shift_point_x(meta.kps_rhand, i, final_shift_right)

        # Indizes Links: 6 (Ellbogen), 7 (Handgelenk)
        indices_left = [7]
        if move_elbows:
            indices_left.append(6)

        for idx in indices_left:
            if idx < len(body_arr):
                self._shift_point_x(body_arr, idx, final_shift_left)
        
        if hasattr(meta, "kps_lhand") and meta.kps_lhand is not None:
            for i in range(len(meta.kps_lhand)):
                self._shift_point_x(meta.kps_lhand, i, final_shift_left)

    def _extract_coords(self, point):
        if point is None: return None
        if isinstance(point, np.ndarray): return point
        if isinstance(point, list) and len(point) >= 2: return point
        return None

    def _assign_point(self, arr, idx, x, y):
        if idx < len(arr) and arr[idx] is not None:
            if isinstance(arr[idx], np.ndarray):
                arr[idx][0] = x
                arr[idx][1] = y
            elif isinstance(arr[idx], list):
                arr[idx][0] = x
                arr[idx][1] = y

    def _shift_point_x(self, arr, idx, shift_x):
        if idx < len(arr) and arr[idx] is not None:
            if isinstance(arr[idx], np.ndarray):
                arr[idx][0] += shift_x
            elif isinstance(arr[idx], list):
                arr[idx][0] += shift_x


class PoseDataHipHandDebugV3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "hip_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01, "tooltip": "Skaliert die Hüftbreite."}),
                "auto_hand_adjust": ("BOOLEAN", {"default": True, "tooltip": "Wenn an, bewegen sich die Hände automatisch mit der Hüfte mit."}),
                "move_elbows": ("BOOLEAN", {"default": True, "tooltip": "Wenn an, werden auch die Ellbogen verschoben."}),
                "hand_offset_px": ("FLOAT", {"default": 0.0, "min": -500.0, "max": 500.0, "step": 1.0, "tooltip": "Manueller Zusatz-Offset für die Hände in Pixeln."}),
                "hand_offset_norm": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Manueller Zusatz-Offset für die Hände normiert."}),
                "elbow_offset_px": ("FLOAT", {"default": 0.0, "min": -500.0, "max": 500.0, "step": 1.0, "tooltip": "Manueller Zusatz-Offset für die Ellbogen in Pixeln."}),
                "elbow_offset_norm": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Manueller Zusatz-Offset für die Ellbogen normiert."}),
                "smooth_hand_entry": ("BOOLEAN", {"default": True, "tooltip": "Verhindert Sprünge, wenn die Hand die Hüfthöhe passiert."}),
                "smooth_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "person_index": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Debug"
    DESCRIPTION = "V3: Zusätzliche Ellbogen-Offsets und normierte Offsets (mit 2 Nachkommastellen)."

    def process(self, pose_data, hip_scale, auto_hand_adjust, move_elbows, hand_offset_px, hand_offset_norm, elbow_offset_px, elbow_offset_norm, smooth_hand_entry, smooth_threshold, person_index):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        indices = (
            [person_index]
            if isinstance(person_index, int) and person_index >= 0 and person_index < len(pose_metas)
            else list(range(len(pose_metas)))
        )

        for idx in indices:
            if pose_metas[idx] is not None:
                self._apply_hip_logic(
                    pose_metas[idx], hip_scale, auto_hand_adjust, move_elbows, 
                    hand_offset_px, hand_offset_norm, elbow_offset_px, elbow_offset_norm,
                    smooth_hand_entry, smooth_threshold
                )

        return (pose_data_copy,)

    def _apply_hip_logic(self, meta, hip_scale, auto_hand_adjust, move_elbows, hand_offset_px, hand_offset_norm, elbow_offset_px, elbow_offset_norm, smooth_hand_entry, smooth_threshold):
        body_arr = getattr(meta, "kps_body", None)
        width = getattr(meta, "width", 1024)
        height = getattr(meta, "height", 1024)

        if body_arr is None:
            return

        if 8 >= len(body_arr) or 11 >= len(body_arr):
            return

        hip_r_orig = self._extract_coords(body_arr[8])
        hip_l_orig = self._extract_coords(body_arr[11])

        if hip_r_orig is None or hip_l_orig is None:
            return

        center_x = (hip_r_orig[0] + hip_l_orig[0]) / 2.0

        dist_r = hip_r_orig[0] - center_x
        new_hip_r_x = center_x + (dist_r * hip_scale)
        self._assign_point(body_arr, 8, new_hip_r_x, hip_r_orig[1])

        dist_l = hip_l_orig[0] - center_x
        new_hip_l_x = center_x + (dist_l * hip_scale)
        self._assign_point(body_arr, 11, new_hip_l_x, hip_l_orig[1])

        delta_right = 0.0
        delta_left = 0.0

        if auto_hand_adjust:
            delta_right = new_hip_r_x - hip_r_orig[0]
            delta_left = new_hip_l_x - hip_l_orig[0]

        actual_hand_offset = float(hand_offset_px) + (float(hand_offset_norm) * float(width))
        actual_elbow_offset = float(elbow_offset_px) + (float(elbow_offset_norm) * float(width))

        base_shift_right = delta_right - actual_hand_offset
        base_shift_left = delta_left + actual_hand_offset

        factor_right = 1.0
        if smooth_hand_entry and 4 < len(body_arr):
            wrist_coords = self._extract_coords(body_arr[4])
            hip_y = hip_r_orig[1]
            if wrist_coords is not None and hip_y > 0:
                wrist_y = wrist_coords[1]
                safe_zone_px = float(height) * float(smooth_threshold)
                start_transition_y = hip_y - safe_zone_px

                if wrist_y < start_transition_y:
                    factor_right = 0.0
                elif wrist_y >= hip_y:
                    factor_right = 1.0
                else:
                    if safe_zone_px > 0:
                        progress = (wrist_y - start_transition_y) / safe_zone_px
                        factor_right = max(0.0, min(1.0, progress))

        factor_left = 1.0
        if smooth_hand_entry and 7 < len(body_arr):
            wrist_coords = self._extract_coords(body_arr[7])
            hip_y = hip_l_orig[1]
            if wrist_coords is not None and hip_y > 0:
                wrist_y = wrist_coords[1]
                safe_zone_px = float(height) * float(smooth_threshold)
                start_transition_y = hip_y - safe_zone_px

                if wrist_y < start_transition_y:
                    factor_left = 0.0
                elif wrist_y >= hip_y:
                    factor_left = 1.0
                else:
                    if safe_zone_px > 0:
                        progress = (wrist_y - start_transition_y) / safe_zone_px
                        factor_left = max(0.0, min(1.0, progress))

        final_shift_right = base_shift_right * factor_right
        final_shift_left = base_shift_left * factor_left

        if 4 < len(body_arr):
            self._shift_point_x(body_arr, 4, final_shift_right)
        if hasattr(meta, "kps_rhand") and meta.kps_rhand is not None:
            for i in range(len(meta.kps_rhand)):
                self._shift_point_x(meta.kps_rhand, i, final_shift_right)

        if 7 < len(body_arr):
            self._shift_point_x(body_arr, 7, final_shift_left)
        if hasattr(meta, "kps_lhand") and meta.kps_lhand is not None:
            for i in range(len(meta.kps_lhand)):
                self._shift_point_x(meta.kps_lhand, i, final_shift_left)

        # Ellbogen verschieben (3 und 6)
        if move_elbows or actual_elbow_offset != 0.0:
            elbow_shift_right = (final_shift_right if move_elbows else 0.0) - actual_elbow_offset
            elbow_shift_left = (final_shift_left if move_elbows else 0.0) + actual_elbow_offset

            if 3 < len(body_arr):
                self._shift_point_x(body_arr, 3, elbow_shift_right)
            if 6 < len(body_arr):
                self._shift_point_x(body_arr, 6, elbow_shift_left)

    def _extract_coords(self, point):
        if point is None: return None
        if isinstance(point, np.ndarray): return point
        if isinstance(point, list) and len(point) >= 2: return point
        if isinstance(point, tuple) and len(point) >= 2: return list(point)
        return None

    def _assign_point(self, arr, idx, x, y):
        if idx < len(arr) and arr[idx] is not None:
            if isinstance(arr[idx], np.ndarray):
                arr[idx][0] = x
                arr[idx][1] = y
            elif isinstance(arr[idx], list):
                arr[idx][0] = x
                arr[idx][1] = y
            elif isinstance(arr[idx], tuple):
                tmp = list(arr[idx])
                tmp[0] = x
                tmp[1] = y
                arr[idx] = tuple(tmp)

    def _shift_point_x(self, arr, idx, shift_x):
        if idx < len(arr) and arr[idx] is not None:
            if isinstance(arr[idx], np.ndarray):
                arr[idx][0] += shift_x
            elif isinstance(arr[idx], list):
                arr[idx][0] += shift_x
            elif isinstance(arr[idx], tuple):
                tmp = list(arr[idx])
                tmp[0] += shift_x
                arr[idx] = tuple(tmp)


