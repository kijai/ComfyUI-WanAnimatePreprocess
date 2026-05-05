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



class PoseDataHandOffsetTimed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "x_offset": ("FLOAT", {"default": 0.0, "min": -2048.0, "max": 2048.0, "step": 0.1, "tooltip": "X-Verschiebung in Pixeln (Standard: Symmetrisch breiter)."}),
                "y_offset": ("FLOAT", {"default": 0.0, "min": -2048.0, "max": 2048.0, "step": 0.1, "tooltip": "Y-Verschiebung in Pixeln."}),
                "active_seconds": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 3600.0, "step": 0.01, "tooltip": "Wie lange der volle Offset gehalten wird (in Sekunden)."}),
                "fade_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3600.0, "step": 0.01, "tooltip": "Zeitraum für den sanften Abgang auf 0.0 (in Sekunden)."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01, "tooltip": "Framerate des Videos zur Berechnung der Zeit."}),
                "symmetric_x": ("BOOLEAN", {"default": True, "tooltip": "True: Rechte Hand +X, Linke Hand -X (breiter). False: Beide Hände +X (Verschiebung)."}),
                "move_elbows": ("BOOLEAN", {"default": True, "tooltip": "Soll der Ellbogen mitverschoben werden?"}),
                "person_index": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Timed"
    DESCRIPTION = "Wendet einen Hand-Offset für eine bestimmte Zeit an und blendet ihn dann sanft aus."

    def process(self, pose_data, x_offset, y_offset, active_seconds, fade_seconds, fps, symmetric_x, move_elbows, person_index):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        # Parameter vorbereiten
        active_frames = int(active_seconds * fps)
        fade_frames = int(fade_seconds * fps)
        total_effect_frames = active_frames + fade_frames

        # Iteriere über alle Frames (angenommen pose_metas ist eine Liste von Frames/Personen)
        # Wenn pose_metas Frames repräsentiert:
        for frame_idx, meta in enumerate(pose_metas):
            
            # Check Person Index (falls pose_metas mehrere Personen pro Frame hat, ist die Struktur komplexer, 
            # aber hier gehen wir von der Standard WanAnimate Struktur aus: Liste von Metas über die Zeit).
            # Wenn person_index gesetzt ist, würde man normalerweise filtern, 
            # aber bei sequenziellen Daten wenden wir es auf den Frame an, wenn es passt.
            if meta is None: continue
            
            # Zeit-Logik
            current_x = 0.0
            current_y = 0.0
            
            if frame_idx < active_frames:
                # Phase 1: Voller Offset
                current_x = x_offset
                current_y = y_offset
            elif frame_idx < total_effect_frames:
                # Phase 2: Fade Out
                if fade_frames > 0:
                    frames_passed_in_fade = frame_idx - active_frames
                    progress = frames_passed_in_fade / float(fade_frames)
                    # Linearer Fade (1.0 -> 0.0)
                    factor = 1.0 - progress
                    current_x = x_offset * factor
                    current_y = y_offset * factor
                else:
                    current_x = 0.0
                    current_y = 0.0
            else:
                # Phase 3: Ende (0.0)
                current_x = 0.0
                current_y = 0.0
            
            # Performance-Optimierung: Wenn Offset 0 ist, nichts tun
            if abs(current_x) < 0.001 and abs(current_y) < 0.001:
                continue

            # Offset anwenden
            self._apply_offset(meta, current_x, current_y, symmetric_x, move_elbows)

        return (pose_data_copy,)

    def _apply_offset(self, meta, off_x, off_y, symmetric, move_elbows):
        body_arr = getattr(meta, "kps_body", None)
        if body_arr is None: return

        # Indizes definieren
        # Rechts: Elbow(3), Wrist(4)
        # Links: Elbow(6), Wrist(7)
        right_indices = [4]
        left_indices = [7]
        
        if move_elbows:
            right_indices.append(3)
            left_indices.append(6)

        # Berechne Verschiebung für Links/Rechts
        shift_x_right = off_x
        shift_x_left = -off_x if symmetric else off_x
        
        shift_y = off_y # Y ist meist für beide gleich (Höhe)

        # --- RECHTE SEITE ---
        for idx in right_indices:
            self._shift_point(body_arr, idx, shift_x_right, shift_y)
        
        if hasattr(meta, "kps_rhand") and meta.kps_rhand is not None:
            for i in range(len(meta.kps_rhand)):
                self._shift_point(meta.kps_rhand, i, shift_x_right, shift_y)

        # --- LINKE SEITE ---
        for idx in left_indices:
            self._shift_point(body_arr, idx, shift_x_left, shift_y)

        if hasattr(meta, "kps_lhand") and meta.kps_lhand is not None:
            for i in range(len(meta.kps_lhand)):
                self._shift_point(meta.kps_lhand, i, shift_x_left, shift_y)

    def _shift_point(self, arr, idx, dx, dy):
        if idx >= len(arr): return
        val = arr[idx]
        
        # Koordinaten extrahieren
        coords = None
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            coords = [float(val[0]), float(val[1])]
        elif isinstance(val, np.ndarray) and val.size >= 2:
            coords = [float(val[0]), float(val[1])]
        
        if coords is None: return

        # Verschieben
        new_x = coords[0] + dx
        new_y = coords[1] + dy
        
        # Zurückschreiben
        if isinstance(val, list):
            val[0] = new_x
            val[1] = new_y
        elif isinstance(val, np.ndarray):
            val[0] = new_x
            val[1] = new_y
        # Falls Tuple, müssen wir es ersetzen (in Listen aber meist Mutable)
        else:
             arr[idx] = [new_x, new_y] # Fallback


class PoseDataHandDeleterTimed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "active_seconds": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 3600.0, "step": 0.01, "tooltip": "Wie lange die Hände gelöscht bleiben (in Sekunden)."}),
                "target_hand": (["BOTH", "LEFT", "RIGHT"], {"default": "BOTH", "tooltip": "Welche Hände gelöscht werden sollen."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01, "tooltip": "Framerate des Videos zur Berechnung der Zeit."}),
                "person_index": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Timed"
    DESCRIPTION = "Löscht Hände für eine bestimmte Zeit. Danach erscheinen sie sofort wieder (hart)."

    def process(self, pose_data, active_seconds, target_hand, fps, person_index):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        # Parameter in Frames umrechnen
        active_frames = int(active_seconds * fps)

        for frame_idx, meta in enumerate(pose_metas):
            if meta is None: continue

            # Logik: Wenn innerhalb der Zeit, dann komplett löschen (0.0).
            # Danach sofort wieder voll da (1.0).
            if frame_idx < active_frames:
                # Löschen
                if target_hand in ["BOTH", "LEFT"]:
                    self._zero_hand(meta, "kps_lhand")
                
                if target_hand in ["BOTH", "RIGHT"]:
                    self._zero_hand(meta, "kps_rhand")
            
            # Wenn frame_idx >= active_frames, machen wir einfach nichts (= Hände bleiben original)

        return (pose_data_copy,)

    def _zero_hand(self, meta, arr_name):
        kps = getattr(meta, arr_name, None)
        kps_p = getattr(meta, f"{arr_name}_p", None) # Confidence Score Array

        if kps is None: return

        # Koordinaten nullen
        for i in range(len(kps)):
            self._zero_point(kps, i)
        
        # Confidence nullen (damit ControlNet weiß: "Hier ist nichts")
        if kps_p is not None:
            for i in range(len(kps_p)):
                kps_p[i] = 0.0

    def _zero_point(self, arr, idx):
        val = arr[idx]
        if isinstance(val, list):
            val[0] = 0.0
            val[1] = 0.0
        elif isinstance(val, np.ndarray):
            val[0] = 0.0
            val[1] = 0.0


class PoseDataSmartHandFilterTimed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ab welcher Confidence gilt ein Punkt als sichtbar?"}),
                "delete_broken_fingers": ("BOOLEAN", {"default": True, "tooltip": "Löscht defekte Finger komplett."}),
                "delete_incomplete_hand": ("BOOLEAN", {"default": True, "tooltip": "Aktiviert das Löschen der ganzen Hand bei zu wenig Punkten."}),
                "hand_integrity_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Prozentualer Anteil (0.0-1.0) der Punkte, die da sein müssen, sonst wird die Hand gelöscht."}),
                "active_seconds": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 3600.0, "step": 0.01, "tooltip": "Wie lange der Filter aktiv ist."}),
                "target_hand": (["BOTH", "LEFT", "RIGHT"], {"default": "BOTH", "tooltip": "Zielhände."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01, "tooltip": "Video Framerate."}),
                "person_index": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Timed"
    DESCRIPTION = "Smart Filter (Timed) mit einstellbarer Hand-Integrität."

    def process(self, pose_data, threshold, delete_broken_fingers, delete_incomplete_hand, hand_integrity_threshold, active_seconds, target_hand, fps, person_index):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        active_frames = int(active_seconds * fps)

        for frame_idx, meta in enumerate(pose_metas):
            if meta is None: continue
            
            # Nur innerhalb der Zeit filtern
            if frame_idx < active_frames:
                if target_hand in ["BOTH", "LEFT"]:
                    self._process_hand(meta, "kps_lhand", threshold, delete_broken_fingers, delete_incomplete_hand, hand_integrity_threshold)
                
                if target_hand in ["BOTH", "RIGHT"]:
                    self._process_hand(meta, "kps_rhand", threshold, delete_broken_fingers, delete_incomplete_hand, hand_integrity_threshold)

        return (pose_data_copy,)

    def _process_hand(self, meta, arr_name, threshold, strict_fingers, strict_hand_active, integrity_ratio):
        kps = getattr(meta, arr_name, None)
        kps_p = getattr(meta, f"{arr_name}_p", None)

        if kps is None or kps_p is None: return
        if len(kps) < 21: return

        # 1. CHECK: Hand Integrität
        if strict_hand_active:
            valid_count = 0
            for p in kps_p:
                if p > threshold:
                    valid_count += 1
            
            min_required = int(21 * integrity_ratio)
            if valid_count < min_required:
                self._zero_indices(kps, kps_p, range(21))
                return

        # 2. CHECK: Finger
        if strict_fingers:
            fingers = [
                [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                [13, 14, 15, 16], [17, 18, 19, 20]
            ]
            for finger_indices in fingers:
                finger_broken = False
                for i in finger_indices:
                    if kps_p[i] <= threshold:
                        finger_broken = True
                        break
                if finger_broken:
                    self._zero_indices(kps, kps_p, finger_indices)

    def _zero_indices(self, kps, kps_p, indices):
        for i in indices:
            if i < len(kps):
                val = kps[i]
                if isinstance(val, list):
                    val[0] = 0.0; val[1] = 0.0
                elif isinstance(val, np.ndarray):
                    val[0] = 0.0; val[1] = 0.0
                if i < len(kps_p):
                    kps_p[i] = 0.0


