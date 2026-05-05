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



class PoseDataConfidenceFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Wenn die Confidence unter diesem Wert liegt, wird der Punkt gelöscht."}),
                "target_region": (
                    [
                        "ALL",
                        "HANDS_BOTH", "HANDS_LEFT", "HANDS_RIGHT",
                        "ARMS_BOTH", "ARMS_LEFT", "ARMS_RIGHT",
                        "LEGS_BOTH", "LEGS_LEFT", "LEGS_RIGHT",
                        "FEET_BOTH", "FEET_LEFT", "FEET_RIGHT"
                    ],
                    {"default": "HANDS_BOTH", "tooltip": "Wähle, welche Körperteile gefiltert werden sollen."}
                ),
                "person_index": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Filter"
    DESCRIPTION = "Setzt Keypoints auf 0, wenn ihre Confidence unter dem Schwellenwert liegt."

    def process(self, pose_data, threshold, target_region, person_index):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        indices = (
            [person_index]
            if isinstance(person_index, int) and person_index >= 0 and person_index < len(pose_metas)
            else list(range(len(pose_metas)))
        )

        for idx in indices:
            meta = pose_metas[idx]
            if meta is None: continue
            
            # Ziele auflösen (Welche Arrays und welche Indizes?)
            targets = self._resolve_targets(target_region)
            
            for arr_name, indices_to_check in targets:
                self._apply_filter(meta, arr_name, indices_to_check, threshold)

        return (pose_data_copy,)

    def _apply_filter(self, meta, arr_name, indices_to_check, threshold):
        kps = getattr(meta, arr_name, None)
        kps_p = getattr(meta, f"{arr_name}_p", None)

        if kps is None or kps_p is None: return

        # Wenn "ALL" als Index übergeben wurde (z.B. bei Händen)
        if isinstance(indices_to_check, str) and indices_to_check == "ALL":
            check_range = range(len(kps))
        else:
            check_range = indices_to_check

        for i in check_range:
            if i >= len(kps_p) or i >= len(kps): continue
            
            confidence = kps_p[i]
            
            # FILTER LOGIK:
            # Wenn Confidence kleiner als Threshold -> Punkt löschen
            if confidence < threshold:
                self._zero_point(kps, i)
                kps_p[i] = 0.0 # Confidence auch auf 0 setzen

    def _resolve_targets(self, region):
        """
        Gibt eine Liste von Tupeln zurück: (ArrayName, IndexListe/ALL)
        Body Keypoint Indizes (COCO/OpenPose Standard):
        2: R-Shoulder, 3: R-Elbow, 4: R-Wrist
        5: L-Shoulder, 6: L-Elbow, 7: L-Wrist
        8: R-Hip, 9: R-Knee, 10: R-Ankle
        11: L-Hip, 12: L-Knee, 13: L-Ankle
        14-17: Eyes/Ears
        19-24: Heels/Toes (bei Body25)
        """
        targets = []
        r = region.upper()

        # --- HÄNDE (Eigene Arrays) ---
        if "HANDS" in r or r == "ALL":
            if r in ["ALL", "HANDS_BOTH", "HANDS_LEFT"]:
                targets.append(("kps_lhand", "ALL"))
            if r in ["ALL", "HANDS_BOTH", "HANDS_RIGHT"]:
                targets.append(("kps_rhand", "ALL"))

        # --- ARME (Body Array) ---
        # Arme definieren wir hier als Schulter, Ellbogen, Handgelenk
        right_arm_idx = [2, 3, 4]
        left_arm_idx = [5, 6, 7]
        
        if "ARMS" in r or r == "ALL":
            if r in ["ALL", "ARMS_BOTH", "ARMS_RIGHT"]:
                targets.append(("kps_body", right_arm_idx))
            if r in ["ALL", "ARMS_BOTH", "ARMS_LEFT"]:
                targets.append(("kps_body", left_arm_idx))

        # --- BEINE (Body Array) ---
        # Beine: Hüfte, Knie
        right_leg_idx = [8, 9]
        left_leg_idx = [11, 12]

        if "LEGS" in r or r == "ALL":
            if r in ["ALL", "LEGS_BOTH", "LEGS_RIGHT"]:
                targets.append(("kps_body", right_leg_idx))
            if r in ["ALL", "LEGS_BOTH", "LEGS_LEFT"]:
                targets.append(("kps_body", left_leg_idx))

        # --- FÜSSE (Body Array) ---
        # Füsse: Knöchel (10, 13) und optional Zehen/Fersen (19-24 falls Body25 Format)
        # Wir nehmen sicherheitshalber auch höhere Indizes dazu für Body25 Support
        right_foot_idx = [10, 19, 20, 21] 
        left_foot_idx = [13, 22, 23, 24]

        if "FEET" in r or r == "ALL":
            if r in ["ALL", "FEET_BOTH", "FEET_RIGHT"]:
                targets.append(("kps_body", right_foot_idx))
            if r in ["ALL", "FEET_BOTH", "FEET_LEFT"]:
                targets.append(("kps_body", left_foot_idx))

        return targets

    def _zero_point(self, arr, idx):
        val = arr[idx]
        if isinstance(val, list):
            val[0] = 0.0
            val[1] = 0.0
        elif isinstance(val, np.ndarray):
            val[0] = 0.0
            val[1] = 0.0


