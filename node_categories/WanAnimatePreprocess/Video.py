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



class FrameSubsamplerForDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "current_fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "target_fps": ("INT", {"default": 10, "min": 1, "max": 120, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sampled_images", "valid_indices")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Video"
    DESCRIPTION = "Reduziert die Video-FPS für Depth Maps um VRAM zu sparen und gibt die behaltenen Indizes aus."

    def process(self, images, current_fps, target_fps):
        import torch
        import math
        
        total_frames = images.shape[0]
        valid_indices = []
        
        # Wenn Target FPS höher oder gleich ist, nichts tun
        if target_fps >= current_fps:
            valid_indices = list(range(total_frames))
            return (images, ",".join(map(str, valid_indices)))
            
        duration = total_frames / current_fps
        target_frame_count = int(math.ceil(duration * target_fps))
        
        sampled_tensors = []
        for i in range(target_frame_count):
            # Berechne den Index im Original-Video
            idx = int(round((i / target_fps) * current_fps))
            
            # Stelle sicher, dass wir im Rahmen bleiben und keine Duplikate einfügen
            if idx < total_frames and idx not in valid_indices:
                valid_indices.append(idx)
                sampled_tensors.append(images[idx].unsqueeze(0))
                
        # Fallback, falls etwas schiefgeht
        if not sampled_tensors:
            valid_indices = [0]
            sampled_tensors = [images[0].unsqueeze(0)]
            
        sampled_images = torch.cat(sampled_tensors, dim=0)
        indices_str = ",".join(map(str, valid_indices))
        
        return (sampled_images, indices_str)


