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



class PoseDataLowerLegRemover:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "remove_knees": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Wenn True, werden auch die Knie (9, 12) gelöscht. Wenn False, nur ab den Knöcheln."
                }),
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Editor"
    DESCRIPTION = "Entfernt die Unterschenkel und Füße extrem aggressiv aus den Pose-Daten."

    def process(self, pose_data, remove_knees):
        import copy
        import numpy as np

        # Deepcopy, um sicherzustellen, dass wir nicht im Cache rumpfuschen
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        pose_metas_original = pose_data_copy.get("pose_metas_original", [])
        
        # 9: Rechtes Knie, 12: Linkes Knie
        # 10: Rechter Knöchel, 13: Linker Knöchel
        # 18-24: Diverse Fuß/Zehen-Punkte
        if remove_knees:
            indices_to_remove = [9, 12, 10, 13, 18, 19, 20, 21, 22, 23, 24]
        else:
            indices_to_remove = [10, 13, 18, 19, 20, 21, 22, 23, 24]

        removed_count = 0

        # 1. Haupt-Metas verarbeiten (Das ist das, was DrawViTPose zeichnet!)
        for meta in pose_metas:
            coords = getattr(meta, "kps_body", None)
            scores = getattr(meta, "kps_body_p", None)
            
            if coords is not None and scores is not None:
                for idx in indices_to_remove:
                    if idx < len(coords) and idx < len(scores):
                        # Extrem wichtig: Direkt ins Numpy-Array schreiben
                        scores[idx] = 0.0
                        coords[idx][0] = 0.0
                        coords[idx][1] = 0.0
                        removed_count += 1

        # 2. Original-Metas verarbeiten (Das ist das, was an Retargeting/SCAIL geht)
        for entry in pose_metas_original:
            if not isinstance(entry, dict):
                continue
                
            keypoints_body = entry.get("keypoints_body")
            if keypoints_body is not None:
                points_np = np.array(keypoints_body, dtype=np.float32)
                if points_np.ndim == 2 and points_np.shape[1] >= 3:
                    for idx in indices_to_remove:
                        if idx < points_np.shape[0]:
                            points_np[idx, 0] = 0.0
                            points_np[idx, 1] = 0.0
                            points_np[idx, 2] = 0.0
                    # Zwingend als Liste zurückschreiben!
                    entry["keypoints_body"] = points_np.tolist()

        # Logge in die Server-Konsole, damit wir sehen, ob er getriggert wird
        print(f"\n[Lower Leg Remover] Erfolgreich ausgeführt! Es wurden {removed_count} Keypoints auf 0.0 gesetzt (Knie gelöscht: {remove_knees}).\n")

        return (pose_data_copy,)


