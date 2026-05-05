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



class NLFPoseDataSelectFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die 3D NLF Pose Daten (Sequenz)"}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "tooltip": "Der Index des gewünschten Frames"}),
            }
        }

    RETURN_TYPES = ("NLFPRED",)
    RETURN_NAMES = ("nlf_poses_single",)
    FUNCTION = "select_frame"
    CATEGORY = "WanAnimatePreprocess/NLF"
    DESCRIPTION = "Extrahiert einen exakten, einzelnen Frame aus einer NLF 3D Pose Sequenz."

    def select_frame(self, nlf_poses, frame_index):
        # Tiefe Kopie, um die Originaldaten im Speicher nicht aus Versehen zu kappen
        selected_nlf = copy.deepcopy(nlf_poses)
        
        # NLF-Daten kommen meist als Dictionary mit verschiedenen Keys (joints3d_nonparam, cam_poses etc.)
        if isinstance(selected_nlf, dict):
            for key, value in selected_nlf.items():
                if isinstance(value, list) and len(value) > 0:
                    # Wenn es eine Liste von Arrays/Tensoren ist (z.B. Person 0, Person 1)
                    for i in range(len(value)):
                        try:
                            # Wir sichern ab, dass der Index nicht out-of-bounds geht (Clamp)
                            max_idx = max(0, len(value[i]) - 1)
                            idx = min(frame_index, max_idx)
                            idx = max(0, idx)
                            
                            # WICHTIG: Slice [idx : idx+1] erhält die Frame-Dimension! 
                            value[i] = value[i][idx:idx+1]
                        except Exception:
                            pass 
                elif hasattr(value, '__getitem__') and hasattr(value, '__len__'):
                    # Falls es direkt ein Tensor/Array ist
                    try:
                        max_idx = max(0, len(value) - 1)
                        idx = min(frame_index, max_idx)
                        idx = max(0, idx)
                        selected_nlf[key] = value[idx:idx+1]
                    except:
                        pass
                        
        # Fallback, falls nlf_poses direkt eine Liste oder ein Tensor ist (ohne Dict drumherum)
        elif hasattr(selected_nlf, '__getitem__') and hasattr(selected_nlf, '__len__'):
             max_idx = max(0, len(selected_nlf) - 1)
             idx = min(frame_index, max_idx)
             idx = max(0, idx)
             selected_nlf = selected_nlf[idx:idx+1]

        return (selected_nlf,)


