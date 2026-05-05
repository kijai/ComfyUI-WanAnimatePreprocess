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



class WanFrameSyncSettingsV5:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_sequence": (["Foreground (Wan Output)", "Background (Source)"], {"default": "Foreground (Wan Output)"}),
                "index_from_end": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "current_iteration": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "expected_len_first_pass": ("INT", {"default": 16, "min": 0, "max": 1024, "step": 1}),
                "expected_len_loop_pass": ("INT", {"default": 16, "min": 0, "max": 1024, "step": 1}),
                "overlap_drop_frames": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "enable_sync_first_pass": (["yes", "no"], {"default": "yes", "tooltip": "Wenn 'no', wird im 1. Durchgang nichts verlängert/gekürzt."}),
                "enable_sync_loop_pass": (["yes", "no"], {"default": "yes", "tooltip": "Wenn 'no', wird in Loop-Durchgängen nichts verlängert/gekürzt (Overlap wird aber trotzdem abgeschnitten)."}),
            }
        }

    RETURN_TYPES = ("FRAME_SYNC_SETTINGS",)
    RETURN_NAMES = ("sync_settings",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Sync"
    DESCRIPTION = "Steuert Längen, Drop-Indizes, Overlap und erlaubt das gezielte An-/Abschalten der Frame-Verdopplung."

    def process(self, target_sequence, index_from_end, current_iteration, expected_len_first_pass, expected_len_loop_pass, overlap_drop_frames, enable_sync_first_pass, enable_sync_loop_pass):
        settings = {
            "target": target_sequence,
            "idx_from_end": index_from_end,
            "iteration": current_iteration,
            "exp_len_first": expected_len_first_pass,
            "exp_len_loop": expected_len_loop_pass,
            "overlap_drop": overlap_drop_frames,
            "sync_first": enable_sync_first_pass == "yes",
            "sync_loop": enable_sync_loop_pass == "yes"
        }
        return (settings,)


class WanSmartImageBatcherV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "sync_settings": ("FRAME_SYNC_SETTINGS",),
            },
            "optional": {
                "opt_mask1": ("MASK",),
                "opt_mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("batched_images", "batched_masks")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Sync"
    DESCRIPTION = "Intelligenter Batcher: Repariert Längen (falls aktiviert) und schneidet Overlap ab."

    def process(self, image1, image2, sync_settings, opt_mask1=None, opt_mask2=None):
        import torch

        list1 = [img for img in image1]
        list2 = [img for img in image2]
        
        mlist1 = [m for m in opt_mask1] if opt_mask1 is not None else None
        mlist2 = [m for m in opt_mask2] if opt_mask2 is not None else None

        target = sync_settings["target"]
        idx_from_end = sync_settings["idx_from_end"]
        iteration = sync_settings["iteration"]
        exp_len_first = sync_settings["exp_len_first"]
        exp_len_loop = sync_settings["exp_len_loop"]
        overlap_drop = sync_settings["overlap_drop"]
        sync_first = sync_settings["sync_first"]
        sync_loop = sync_settings["sync_loop"]

        def fix_list(lst, t_len, is_tensor=True):
            if t_len <= 0: return lst
            while len(lst) > t_len:
                idx = max(0, len(lst) - 1 - idx_from_end)
                lst.pop(idx)
            while len(lst) < t_len:
                idx = max(0, len(lst) - 1 - idx_from_end)
                lst.insert(idx, lst[idx].clone() if is_tensor else lst[idx].copy())
            return lst

        # --- 1. FIRST PASS (image1) ---
        if iteration == 0 and target == "Foreground (Wan Output)" and exp_len_first > 0 and sync_first:
            list1 = fix_list(list1, exp_len_first)
            if mlist1 is not None:
                mlist1 = fix_list(mlist1, exp_len_first)

        # --- 2. LOOP PASS (image2) ---
        # Overlap-Drop passiert IMMER, das ist ein harter Cut für den Kontext!
        if overlap_drop > 0:
            safe_drop = min(overlap_drop, len(list2) - 1)
            list2 = list2[safe_drop:]
            if mlist2 is not None:
                mlist2 = mlist2[safe_drop:]
        
        # Frame-Verdopplung/-Löschung passiert nur, wenn Schalter auf "yes" steht
        if target == "Foreground (Wan Output)" and exp_len_loop > 0 and sync_loop:
            target_len2 = max(1, exp_len_loop - overlap_drop)
            list2 = fix_list(list2, target_len2)
            if mlist2 is not None:
                mlist2 = fix_list(mlist2, target_len2)

        # --- 3. BATCHEN ---
        out_img_list = list1 + list2
        out_img = torch.stack(out_img_list)

        if mlist1 is not None and mlist2 is not None:
            out_mask = torch.stack(mlist1 + mlist2)
        elif mlist1 is not None:
            dummy = torch.zeros((len(list2), out_img.shape[1], out_img.shape[2]), dtype=torch.float32)
            out_mask = torch.cat([torch.stack(mlist1), dummy], dim=0)
        elif mlist2 is not None:
            dummy = torch.zeros((len(list1), out_img.shape[1], out_img.shape[2]), dtype=torch.float32)
            out_mask = torch.cat([dummy, torch.stack(mlist2)], dim=0)
        else:
            out_mask = torch.zeros((len(out_img_list), out_img.shape[1], out_img.shape[2]), dtype=torch.float32)

        return (out_img, out_mask)


