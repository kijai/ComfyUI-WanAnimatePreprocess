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

from .models.onnx_models import ViTPose, Yolo
from .pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from .utils import get_face_bboxes, padding_resize, resize_by_area, resize_to_bounds
from .pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new, draw_aaface_by_meta
from .retarget_pose import get_retarget_pose
from .pose_data_editor_alone_automatic import PoseDataEditorAloneAutomaticChatyNode


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



# --- Extracted Nodes Imports ---
from .node_categories.WanAnimatePreprocess.nodes import (
    PoseAndFaceDetectionV7_NoWarp, WanFaceStitcherV3, KeypointTrimNode, OnnxDetectionModelLoader, PoseAndFaceDetection, PoseDataAutoBlackoutOnJitter, ImageBlackoutOnNoBBox, PoseDataEditor, PoseDataAutomaticOffsetNodeV3, PoseDataAutomaticOffsetNodeV4, PoseDataEditorCutter, PoseDataEditorWithMaskCutter, DrawViTPose, PoseDataEditorKeypointDeleter, PoseDataEditorKneeCutter, PoseDataEditorHeadDeleter, PoseDataEditorJitterDeleter, BlackStripeImage, ImageBatchBlackout, PoseRetargetPromptHelper, PoseDataToMask, PoseDataToOvalMask, DrawViTPose_v2, DrawViTPose_v3, KeypointDeleter, PoseDataToMaskV2, PoseDataSelectFrameNode, LoadPoseDataFromJsonNode, PoseAndFaceDetectionV8_NoWarp, WanFaceStitcherV4
)
from .node_categories.WanAnimatePreprocess.Debug import (
    SavePoseDataNode, PoseDataHipHandDebugV2, PoseDataHipHandDebugV3
)
from .node_categories.WanAnimatePreprocess.Timed import (
    PoseDataHandOffsetTimed, PoseDataHandDeleterTimed, PoseDataSmartHandFilterTimed
)
from .node_categories.WanAnimatePreprocess.Filter import PoseDataConfidenceFilter
from .node_categories.WanAnimatePreprocess.Masking import (
    MaskPositionalCutterV14, MaskPositionalJoinerV20, MaskPositionalCutterV21, MaskPositionalJoinerV21
)
from .node_categories.WanAnimatePreprocess.Sync import (
    WanFrameSyncSettingsV5, WanSmartImageBatcherV2
)
from .node_categories.WanAnimatePreprocess.SCAIL import (
    PoseDataToDWPoses, RenderNLFPosesWithData, NLFDataToPoseData, RenderNLFPosesDirect, RenderNLFPosesDirect7, RenderNLFPosesDirectPoseDataMimic13, RenderNLFPosesDirectPoseDataMimic14, NLFDataToMaskV2, RenderNLFPosesDirectPoseDataMimic15, NLFDataToMaskV3, NLFDataToMaskV4, RenderNLFPosesDirectPoseDataMimic16, NLFDataToMaskV5, RenderNLFPosesDirectPoseDataMimic17, RenderNLFPosesDirectHybrid8
)
from .node_categories.WanAnimatePreprocess.Ultimate import (
    SavePoseCalibration, LoadPoseCalibration, PoseLocalBoneRetargeterV10, PoseGlobalPerspectiveScalerV30, PoseGlobalPerspectiveScalerV38, PoseCalibrationV20, PoseCalibrationV22, PoseCalibrationV15, PoseGlobalPerspectiveScalerV28, PoseCalibrationV23, PoseGlobalPerspectiveScalerV39, PoseGlobalPerspectiveScalerV40, PoseGlobalPerspectiveScalerV38, PoseCalibrationV24, PoseCalibrationV25, PoseGlobalPerspectiveScalerV41, PoseCalibrationManipulator, PoseCalibrationV29, PoseGlobalPerspectiveScalerV43, PoseGlobalPerspectiveScalerV46, PoseGlobalPerspectiveScalerV47, PoseGlobalPerspectiveScalerV48, PoseGlobalPerspectiveScalerV49, PoseCalibrationManipulator2, PoseGlobalPerspectiveScalerV50, PoseGlobalPerspectiveScalerV51, PoseCalibrationV30, PoseGlobalPerspectiveScalerV53, PoseCalibrationV31, PoseGlobalPerspectiveScalerV54, PoseCalibrationV32, PoseGlobalPerspectiveScalerV55, PoseCalibrationV33, PoseGlobalPerspectiveScalerV56, PoseCalibrationManipulator3, PoseGlobalPerspectiveScalerV57
)
from .node_categories.WanAnimatePreprocess.Editor import PoseDataLowerLegRemover
from .node_categories.WanAnimatePreprocess.Retargeting import (
    NLFProportionalRetargeterV5, NLFConfigScaler3DBones, NLFProportionalRetargeterV6, NLFProportionalRetargeterV7, NLFConfigScaler3DBones2, NLFProportionalRetargeterV9, NLFProportionalRetargeterV13, NLFProportionalRetargeterV14, NLFProportionalRetargeterV16, NLFProportionalRetargeterV17, NLFProportionalRetargeterV17ex, NLFProportionalRetargeterV18, NLFProportionalRetargeterV181, NLFProportionalRetargeterV19, NLFProportionalRetargeterV20
)
from .node_categories.WanAnimate.NLF import (
    NLFDataHandDebugV3, NLFDataHandDebugV4, NLFDataHandDebugV5, NLFDataHandDebugV6, NLFDataHandDebugV7, NLFDataHandDebugV8, NLFDataHandDebugV9, NLFDataHandDebugV10, NLFDataHandDebugV11, NLFDataHandDebugV12
)
from .node_categories.WanAnimatePreprocess.Video import FrameSubsamplerForDepth
from .node_categories.WanAnimatePreprocess.Scaling import NLFPhysicalScalerV1
from .node_categories.WanAnimatePreprocess.Mimic import RenderNLFPosesOrthographicMimic
from .node_categories.WanAnimatePreprocess.NLF import NLFPoseDataSelectFrame

# --- End Extracted Nodes Imports ---

import copy


import math


import torch


import numpy as np


from .pose_utils.pose2d_utils import AAPoseMeta


import copy


import math


import torch


import numpy as np


from .pose_utils.pose2d_utils import AAPoseMeta


import copy


import math


from .pose_utils.pose2d_utils import AAPoseMeta # Stellt sicher, dass AAPoseMeta importiert wird


import copy


import numpy as np


from .pose_utils.human_visualization import draw_handpose_new, AAPoseMeta


import math


def draw_aapose_v2(
    img,
    kp2ds,
    threshold=0.5,
    kp2ds_lhand=None,
    kp2ds_rhand=None,
    draw_hand=False,
    stickwidth_type='v2',
    body_stick_width=-1,
    hand_stick_width=-1,
    draw_head=True
):
    kp2ds = kp2ds.copy()
    if not draw_head:
        kp2ds[[0,14,15,16,17], 2] = 0
    kp2ds_body = kp2ds

    # Originale Reihenfolge der Gliedmaßen
    # Indizes:
    # 0, 1: Schultern
    # 2, 3: Rechter Arm (soll nach hinten oder zuletzt, wenn man von vorne schaut, aber hier geht es um Layering)
    # 4, 5: Linker Arm (der "grüne Streifen")
    # 6-11: Rumpf und Beine ("blaue Streifen")
    # 12-16: Gesicht
    # 17, 18: Füße
    
    limbSeq_orig = [
        [2, 3], [2, 6],  # 0, 1: Shoulders
        [3, 4], [4, 5],  # 2, 3: Right Arm
        [6, 7], [7, 8],  # 4, 5: Left Arm
        [2, 9],          # 6: Right Body Side (Neck-Hip)
        [9, 10], [10, 11], # 7, 8: Right Leg
        [2, 12],         # 9: Left Body Side (Neck-Hip)
        [12, 13], [13, 14], # 10, 11: Left Leg
        [2, 1],          # 12: Neck-Nose
        [1, 15], [15, 17], # 13, 14: Face Right
        [1, 16], [16, 18], # 15, 16: Face Left
        [14, 19],        # 17: Left Foot
        [11, 20]         # 18: Right Foot
    ]

    colors_orig = [
        [255, 0, 0], [255, 85, 0],      # 0, 1
        [255, 170, 0], [255, 255, 0],   # 2, 3 (Right Arm - Orange/Yellow)
        [170, 255, 0], [85, 255, 0],    # 4, 5 (Left Arm - Lime/Green)
        [0, 255, 0],                    # 6 (Right Body - Green)
        [0, 255, 85], [0, 255, 170],    # 7, 8 (Right Leg)
        [0, 255, 255],                  # 9 (Left Body - Cyan/Blue)
        [0, 170, 255], [0, 85, 255],    # 10, 11 (Left Leg - Blue)
        [0, 0, 255],                    # 12 (Neck-Nose - Blue)
        [85, 0, 255], [170, 0, 255],    # 13, 14 (Face)
        [255, 0, 255], [255, 0, 170],   # 15, 16 (Face)
        [255, 0, 85],                   # 17 (Foot)
        [200, 200, 0]                   # 18 (Foot)
    ]

    # Wir definieren eine neue Reihenfolge, bei der die Arme (Indizes 2,3,4,5) ans Ende verschoben werden.
    # Neue Reihenfolge der Indizes aus der Originalliste:
    reordered_indices = [
        0, 1,               # Schultern
        6, 7, 8, 9, 10, 11, # Körper & Beine (zuerst zeichnen, damit sie "hinten" sind)
        12, 13, 14, 15, 16, # Gesicht
        17, 18,             # Füße
        2, 3,               # Rechter Arm (jetzt drüber gezeichnet)
        4, 5                # Linker Arm (jetzt drüber gezeichnet)
    ]

    limbSeq = [limbSeq_orig[i] for i in reordered_indices]
    colors = [colors_orig[i] for i in reordered_indices]

    H, W, C = img.shape
    
    if body_stick_width == -1:
        stickwidth = max(int(min(H, W) / 200) - 1, 1)
    else:
        stickwidth = body_stick_width

    for _idx, ((k1_index, k2_index), color) in enumerate(zip(limbSeq, colors)):
        keypoint1 = kp2ds_body[k1_index - 1]
        keypoint2 = kp2ds_body[k2_index - 1]

        if keypoint1[-1] < threshold or keypoint2[-1] < threshold:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]])
        X = np.array([keypoint1[1], keypoint2[1]])
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(img, polygon, [int(float(c) * 0.6) for c in color])

    for _idx, (keypoint, color) in enumerate(zip(kp2ds_body, colors_orig)): # Punkte in Originalfarbe/Reihenfolge
         # Hinweis: Die Punkte selbst (Kreise) werden hier evtl. übermalt, wenn wir die Reihenfolge ändern. 
         # Aber meistens sind die Linien das Problem.
         pass # Wir zeichnen die Punkte (Gelenke) separat am Ende, damit sie ganz oben sind?
              # Der Originalcode zeichnet Punkte IN der Schleife oder danach.
              # Originalcode zeichnet Punkte danach. Wir machen es auch so.

    # Zeichne Gelenkpunkte (in Originalreihenfolge der Farben, damit die Farben stimmen)
    for _idx, (keypoint, color) in enumerate(zip(kp2ds_body, colors_orig)): # Nutze colors_orig + dummy für Länge
        if _idx >= len(colors_orig): break
        if keypoint[-1] < threshold:
            continue
        x, y = keypoint[0], keypoint[1]
        cv2.circle(img, (int(x), int(y)), stickwidth, colors_orig[_idx], thickness=-1)

    if draw_hand:
        img = draw_handpose_new(img, kp2ds_lhand, stickwidth_type=stickwidth_type, hand_score_th=threshold, hand_stick_width=hand_stick_width)
        img = draw_handpose_new(img, kp2ds_rhand, stickwidth_type=stickwidth_type, hand_score_th=threshold, hand_stick_width=hand_stick_width)

    return img


import logging


def scale_faces(poses, pose_2d_ref):
    ref = pose_2d_ref[0]
    pose_0 = poses[0]

    face_0 = pose_0['faces']  # shape: (1, 68, 2)
    face_ref = ref['faces']

    face_0 = np.array(face_0[0])      # (68, 2)
    face_ref = np.array(face_ref[0])

    center_idx = 30
    center_0 = face_0[center_idx]
    center_ref = face_ref[center_idx]

    dist = np.linalg.norm(face_0 - center_0, axis=1)
    dist_ref = np.linalg.norm(face_ref - center_ref, axis=1)

    dist = np.delete(dist, center_idx)
    dist_ref = np.delete(dist_ref, center_idx)

    mean_dist = np.mean(dist)
    mean_dist_ref = np.mean(dist_ref)

    if mean_dist < 1e-6:
        scale_n = 1.0
    else:
        scale_n = mean_dist_ref / mean_dist

    scale_n = np.clip(scale_n, 0.8, 1.5)

    for i, pose in enumerate(poses):
        face = pose['faces']
        face = np.array(face[0])
        center = face[center_idx]
        scaled_face = (face - center) * scale_n + center
        poses[i]['faces'][0] = scaled_face

        body = pose['bodies']
        candidate = body['candidate']
        candidate_np = np.array(candidate[0])
        body_center = candidate_np[0]
        scaled_candidate = (candidate_np - body_center) * scale_n + body_center
        poses[i]['bodies']['candidate'][0] = scaled_candidate

    return scale_n


import os


import json


import folder_paths


def intrinsic_matrix_from_field_of_view(imshape, fov_degrees:float = 55):
    imshape = np.array(imshape)
    fov_radians = fov_degrees * np.array(np.pi / 180)
    larger_side = np.max(imshape)
    focal_length = larger_side / (np.tan(fov_radians / 2) * 2)
    return np.array([
        [focal_length, 0, imshape[1] / 2],
        [0, focal_length, imshape[0] / 2],
        [0, 0, 1],
    ])


# =========================================
# Class Locations Reference:
# PoseAndFaceDetectionV7_NoWarp -> node_categories/WanAnimatePreprocess/nodes.py
# WanFaceStitcherV3 -> node_categories/WanAnimatePreprocess/nodes.py
# KeypointTrimNode -> node_categories/WanAnimatePreprocess/nodes.py
# OnnxDetectionModelLoader -> node_categories/WanAnimatePreprocess/nodes.py
# PoseAndFaceDetection -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataAutoBlackoutOnJitter -> node_categories/WanAnimatePreprocess/nodes.py
# ImageBlackoutOnNoBBox -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataEditor -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataAutomaticOffsetNodeV3 -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataAutomaticOffsetNodeV4 -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataEditorCutter -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataEditorWithMaskCutter -> node_categories/WanAnimatePreprocess/nodes.py
# DrawViTPose -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataEditorKeypointDeleter -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataEditorKneeCutter -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataEditorHeadDeleter -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataEditorJitterDeleter -> node_categories/WanAnimatePreprocess/nodes.py
# BlackStripeImage -> node_categories/WanAnimatePreprocess/nodes.py
# ImageBatchBlackout -> node_categories/WanAnimatePreprocess/nodes.py
# PoseRetargetPromptHelper -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataToMask -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataToOvalMask -> node_categories/WanAnimatePreprocess/nodes.py
# DrawViTPose_v2 -> node_categories/WanAnimatePreprocess/nodes.py
# DrawViTPose_v3 -> node_categories/WanAnimatePreprocess/nodes.py
# KeypointDeleter -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataToMaskV2 -> node_categories/WanAnimatePreprocess/nodes.py
# PoseDataSelectFrameNode -> node_categories/WanAnimatePreprocess/nodes.py
# LoadPoseDataFromJsonNode -> node_categories/WanAnimatePreprocess/nodes.py
# PoseAndFaceDetectionV8_NoWarp -> node_categories/WanAnimatePreprocess/nodes.py
# WanFaceStitcherV4 -> node_categories/WanAnimatePreprocess/nodes.py
# SavePoseDataNode -> node_categories/WanAnimatePreprocess/Debug.py
# PoseDataHipHandDebugV2 -> node_categories/WanAnimatePreprocess/Debug.py
# PoseDataHipHandDebugV3 -> node_categories/WanAnimatePreprocess/Debug.py
# PoseDataHandOffsetTimed -> node_categories/WanAnimatePreprocess/Timed.py
# PoseDataHandDeleterTimed -> node_categories/WanAnimatePreprocess/Timed.py
# PoseDataSmartHandFilterTimed -> node_categories/WanAnimatePreprocess/Timed.py
# PoseDataConfidenceFilter -> node_categories/WanAnimatePreprocess/Filter.py
# MaskPositionalCutterV14 -> node_categories/WanAnimatePreprocess/Masking.py
# MaskPositionalJoinerV20 -> node_categories/WanAnimatePreprocess/Masking.py
# MaskPositionalCutterV21 -> node_categories/WanAnimatePreprocess/Masking.py
# MaskPositionalJoinerV21 -> node_categories/WanAnimatePreprocess/Masking.py
# WanFrameSyncSettingsV5 -> node_categories/WanAnimatePreprocess/Sync.py
# WanSmartImageBatcherV2 -> node_categories/WanAnimatePreprocess/Sync.py
# PoseDataToDWPoses -> node_categories/WanAnimatePreprocess/SCAIL.py
# RenderNLFPosesWithData -> node_categories/WanAnimatePreprocess/SCAIL.py
# NLFDataToPoseData -> node_categories/WanAnimatePreprocess/SCAIL.py
# RenderNLFPosesDirect -> node_categories/WanAnimatePreprocess/SCAIL.py
# RenderNLFPosesDirect7 -> node_categories/WanAnimatePreprocess/SCAIL.py
# RenderNLFPosesDirectPoseDataMimic13 -> node_categories/WanAnimatePreprocess/SCAIL.py
# RenderNLFPosesDirectPoseDataMimic14 -> node_categories/WanAnimatePreprocess/SCAIL.py
# NLFDataToMaskV2 -> node_categories/WanAnimatePreprocess/SCAIL.py
# RenderNLFPosesDirectPoseDataMimic15 -> node_categories/WanAnimatePreprocess/SCAIL.py
# NLFDataToMaskV3 -> node_categories/WanAnimatePreprocess/SCAIL.py
# NLFDataToMaskV4 -> node_categories/WanAnimatePreprocess/SCAIL.py
# RenderNLFPosesDirectPoseDataMimic16 -> node_categories/WanAnimatePreprocess/SCAIL.py
# NLFDataToMaskV5 -> node_categories/WanAnimatePreprocess/SCAIL.py
# RenderNLFPosesDirectPoseDataMimic17 -> node_categories/WanAnimatePreprocess/SCAIL.py
# RenderNLFPosesDirectHybrid8 -> node_categories/WanAnimatePreprocess/SCAIL.py
# SavePoseCalibration -> node_categories/WanAnimatePreprocess/Ultimate.py
# LoadPoseCalibration -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseLocalBoneRetargeterV10 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV30 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV38 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV20 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV22 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV15 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV28 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV23 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV39 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV40 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV38 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV24 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV25 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV41 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationManipulator -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV29 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV43 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV46 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV47 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV48 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV49 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationManipulator2 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV50 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV51 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV30 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV53 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV31 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV54 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV32 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV55 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationV33 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV56 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseCalibrationManipulator3 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseGlobalPerspectiveScalerV57 -> node_categories/WanAnimatePreprocess/Ultimate.py
# PoseDataLowerLegRemover -> node_categories/WanAnimatePreprocess/Editor.py
# NLFProportionalRetargeterV5 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFConfigScaler3DBones -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV6 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV7 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFConfigScaler3DBones2 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV9 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV13 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV14 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV16 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV17 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV17ex -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV18 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV181 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV19 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFProportionalRetargeterV20 -> node_categories/WanAnimatePreprocess/Retargeting.py
# NLFDataHandDebugV3 -> node_categories/WanAnimate/NLF.py
# NLFDataHandDebugV4 -> node_categories/WanAnimate/NLF.py
# NLFDataHandDebugV5 -> node_categories/WanAnimate/NLF.py
# NLFDataHandDebugV6 -> node_categories/WanAnimate/NLF.py
# NLFDataHandDebugV7 -> node_categories/WanAnimate/NLF.py
# NLFDataHandDebugV8 -> node_categories/WanAnimate/NLF.py
# NLFDataHandDebugV9 -> node_categories/WanAnimate/NLF.py
# NLFDataHandDebugV10 -> node_categories/WanAnimate/NLF.py
# NLFDataHandDebugV11 -> node_categories/WanAnimate/NLF.py
# NLFDataHandDebugV12 -> node_categories/WanAnimate/NLF.py
# FrameSubsamplerForDepth -> node_categories/WanAnimatePreprocess/Video.py
# NLFPhysicalScalerV1 -> node_categories/WanAnimatePreprocess/Scaling.py
# RenderNLFPosesOrthographicMimic -> node_categories/WanAnimatePreprocess/Mimic.py
# NLFPoseDataSelectFrame -> node_categories/WanAnimatePreprocess/NLF.py
# =========================================

NODE_CLASS_MAPPINGS = {
    "PoseAndFaceDetectionV7_NoWarp": PoseAndFaceDetectionV7_NoWarp,
    "WanFaceStitcherV3": WanFaceStitcherV3,
    "KeypointTrimNode": KeypointTrimNode,
    "OnnxDetectionModelLoader": OnnxDetectionModelLoader,
    "PoseAndFaceDetection": PoseAndFaceDetection,
    "PoseDataEditor": PoseDataEditor,
    "PoseDataEditorCutter": PoseDataEditorCutter,
    "PoseDataEditorWithMaskCutter": PoseDataEditorWithMaskCutter,
    "PoseDataEditorKeypointDeleter": PoseDataEditorKeypointDeleter,
    "PoseDataEditorKneeCutter": PoseDataEditorKneeCutter,
    "PoseDataEditorHeadDeleter": PoseDataEditorHeadDeleter,
    "PoseDataEditorJitterDeleter": PoseDataEditorJitterDeleter,
    "PoseDataEditorAloneAutomaticChaty": PoseDataEditorAloneAutomaticChatyNode,
    "ImageBatchBlackout": ImageBatchBlackout,
    "PoseRetargetPromptHelper": PoseRetargetPromptHelper,
    "ImageBlackoutOnNoBBox": ImageBlackoutOnNoBBox,
    "PoseDataAutomaticOffsetNodeV3": PoseDataAutomaticOffsetNodeV3,
    "PoseDataAutomaticOffsetNodeV4": PoseDataAutomaticOffsetNodeV4,
    "PoseDataAutoBlackoutOnJitter": PoseDataAutoBlackoutOnJitter,
    "PoseDataToMask": PoseDataToMask,
    "PoseDataToOvalMask": PoseDataToOvalMask,
    "SavePoseDataNode": SavePoseDataNode,
    "PoseDataHandOffsetTimed": PoseDataHandOffsetTimed,
    "PoseDataHandDeleterTimed": PoseDataHandDeleterTimed,
    "PoseDataConfidenceFilter": PoseDataConfidenceFilter,
    "PoseDataSmartHandFilterTimed": PoseDataSmartHandFilterTimed,
    "DrawViTPose": DrawViTPose,
    "DrawViTPose_v2": DrawViTPose_v2,
    "DrawViTPose_v3": DrawViTPose_v3,
    "BlackStripeImage": BlackStripeImage,
    "PoseDataHipHandDebugV2": PoseDataHipHandDebugV2,
    "PoseDataHipHandDebugV3": PoseDataHipHandDebugV3,
    "KeypointDeleter": KeypointDeleter,
    "MaskPositionalCutterV14": MaskPositionalCutterV14,
    "MaskPositionalCutterV21": MaskPositionalCutterV21,
    "MaskPositionalJoinerV20": MaskPositionalJoinerV20,
    "MaskPositionalJoinerV21": MaskPositionalJoinerV21,
    "WanFrameSyncSettingsV5": WanFrameSyncSettingsV5,
    "WanSmartImageBatcherV2": WanSmartImageBatcherV2,
    "PoseDataToMaskV2": PoseDataToMaskV2,
    "PoseDataSelectFrameNode": PoseDataSelectFrameNode,
    "LoadPoseDataFromJsonNode": LoadPoseDataFromJsonNode,
    "PoseDataToDWPoses": PoseDataToDWPoses,
    "RenderNLFPosesWithData": RenderNLFPosesWithData,
    "SavePoseCalibration": SavePoseCalibration,
    "LoadPoseCalibration": LoadPoseCalibration,
    "PoseLocalBoneRetargeterV10": PoseLocalBoneRetargeterV10,
    "PoseDataLowerLegRemover": PoseDataLowerLegRemover,
    "NLFDataToPoseData": NLFDataToPoseData,
    "RenderNLFPosesDirect": RenderNLFPosesDirect,
    "PoseGlobalPerspectiveScalerV30": PoseGlobalPerspectiveScalerV30,
    "PoseGlobalPerspectiveScalerV28": PoseGlobalPerspectiveScalerV28,
    "PoseGlobalPerspectiveScalerV38": PoseGlobalPerspectiveScalerV38,
    "PoseCalibrationV20": PoseCalibrationV20,
    "PoseCalibrationV22": PoseCalibrationV22,
    "PoseCalibrationV15": PoseCalibrationV15,
    "NLFProportionalRetargeterV5": NLFProportionalRetargeterV5,
    "NLFConfigScaler3DBones": NLFConfigScaler3DBones,
    "RenderNLFPosesDirect7": RenderNLFPosesDirect7,
    "RenderNLFPosesDirectPoseDataMimic13": RenderNLFPosesDirectPoseDataMimic13,
    "RenderNLFPosesDirectPoseDataMimic14": RenderNLFPosesDirectPoseDataMimic14,
    "PoseCalibrationV23": PoseCalibrationV23,
    "PoseCalibrationV24": PoseCalibrationV24,
    "PoseGlobalPerspectiveScalerV39": PoseGlobalPerspectiveScalerV39,
    "PoseGlobalPerspectiveScalerV40": PoseGlobalPerspectiveScalerV40,
    "PoseCalibrationV25": PoseCalibrationV25,
    "PoseGlobalPerspectiveScalerV41": PoseGlobalPerspectiveScalerV41,
    "PoseCalibrationManipulator": PoseCalibrationManipulator,
    "PoseCalibrationV29": PoseCalibrationV29,
    "PoseGlobalPerspectiveScalerV43": PoseGlobalPerspectiveScalerV43,
    "NLFProportionalRetargeterV6": NLFProportionalRetargeterV6,
    "PoseGlobalPerspectiveScalerV46": PoseGlobalPerspectiveScalerV46,
    "NLFProportionalRetargeterV7": NLFProportionalRetargeterV7,
    "PoseGlobalPerspectiveScalerV47": PoseGlobalPerspectiveScalerV47,
    "NLFConfigScaler3DBones2": NLFConfigScaler3DBones2,
    "NLFProportionalRetargeterV9": NLFProportionalRetargeterV9,
    "PoseGlobalPerspectiveScalerV48": PoseGlobalPerspectiveScalerV48,
    "PoseGlobalPerspectiveScalerV49": PoseGlobalPerspectiveScalerV49,
    "NLFDataToMaskV2": NLFDataToMaskV2,
    "RenderNLFPosesDirectPoseDataMimic15": RenderNLFPosesDirectPoseDataMimic15,
    "NLFDataToMaskV3": NLFDataToMaskV3,
    "NLFDataToMaskV4": NLFDataToMaskV4,
    "NLFDataHandDebugV3": NLFDataHandDebugV3,
    "NLFDataHandDebugV4": NLFDataHandDebugV4,
    "NLFDataHandDebugV5": NLFDataHandDebugV5,
    "NLFDataHandDebugV6": NLFDataHandDebugV6,
    "NLFDataHandDebugV7": NLFDataHandDebugV7,
    "NLFDataHandDebugV8": NLFDataHandDebugV8,
    "NLFDataHandDebugV9": NLFDataHandDebugV9,
    "NLFDataHandDebugV10": NLFDataHandDebugV10,
    "NLFDataHandDebugV11": NLFDataHandDebugV11,
    "NLFDataHandDebugV12": NLFDataHandDebugV12,
    "NLFProportionalRetargeterV13": NLFProportionalRetargeterV13,
    "RenderNLFPosesDirectPoseDataMimic16": RenderNLFPosesDirectPoseDataMimic16,
    "PoseCalibrationManipulator2": PoseCalibrationManipulator2,
    "FrameSubsamplerForDepth": FrameSubsamplerForDepth,
    "PoseGlobalPerspectiveScalerV50": PoseGlobalPerspectiveScalerV50,
    "PoseGlobalPerspectiveScalerV51": PoseGlobalPerspectiveScalerV51,
    "NLFProportionalRetargeterV14": NLFProportionalRetargeterV14,
    "NLFPhysicalScalerV1": NLFPhysicalScalerV1,
    "NLFProportionalRetargeterV16": NLFProportionalRetargeterV16,
    "NLFProportionalRetargeterV17": NLFProportionalRetargeterV17,
    "NLFProportionalRetargeterV17ex": NLFProportionalRetargeterV17ex,
    "NLFDataToMaskV5": NLFDataToMaskV5,
    "RenderNLFPosesDirectPoseDataMimic17": RenderNLFPosesDirectPoseDataMimic17,
    "RenderNLFPosesOrthographicMimic": RenderNLFPosesOrthographicMimic,
    "PoseCalibrationV30": PoseCalibrationV30,
    "PoseGlobalPerspectiveScalerV53": PoseGlobalPerspectiveScalerV53,
    "NLFPoseDataSelectFrame": NLFPoseDataSelectFrame,
    "PoseCalibrationV31": PoseCalibrationV31,
    "PoseGlobalPerspectiveScalerV54": PoseGlobalPerspectiveScalerV54,
    "PoseCalibrationV32": PoseCalibrationV32,
    "PoseGlobalPerspectiveScalerV55": PoseGlobalPerspectiveScalerV55,
    "PoseCalibrationV33": PoseCalibrationV33,
    "PoseGlobalPerspectiveScalerV56": PoseGlobalPerspectiveScalerV56,
    "PoseCalibrationManipulator3": PoseCalibrationManipulator3,
    "PoseAndFaceDetectionV8_NoWarp": PoseAndFaceDetectionV8_NoWarp,
    "WanFaceStitcherV4": WanFaceStitcherV4,
    "RenderNLFPosesDirectHybrid8": RenderNLFPosesDirectHybrid8,
    "NLFProportionalRetargeterV18": NLFProportionalRetargeterV18,
    "NLFProportionalRetargeterV181": NLFProportionalRetargeterV181,
    "NLFProportionalRetargeterV19": NLFProportionalRetargeterV19,
    "NLFProportionalRetargeterV20": NLFProportionalRetargeterV20,
    "NLFProportionalRetargeterV21": NLFProportionalRetargeterV21,
    "PoseGlobalPerspectiveScalerV57": PoseGlobalPerspectiveScalerV57,
    
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAndFaceDetectionV7_NoWarp": "Pose and Face Detection V7 (No Warp)",
    "WanFaceStitcherV3": "Wan Face Stitcher V3 (Smart Scale)",
    "KeypointTrimNode": "Keypoint Trim (Video/Audio)",
    "DrawViTPose": "Draw ViT Pose",
    "BlackStripeImage": "Black Stripe Image",
    "OnnxDetectionModelLoader": "ONNX Detection Model Loader",
    "PoseAndFaceDetection": "Pose and Face Detection",
    "PoseDataEditor": "Pose Data Editor",
    "PoseDataEditorCutter": "Pose Data Editor Cutter v2",
    "PoseDataEditorWithMaskCutter": "Pose Data Editor With Mask Cutter",
    "PoseDataEditorKeypointDeleter": "Pose Data Editor Keypoint Deleter",
    "PoseDataEditorKneeCutter": "Pose Data Editor Knee Cutter",
    "PoseDataEditorHeadDeleter": "Pose Data Head Deleter",
    "PoseDataEditorJitterDeleter": "Pose Data Jitter Deleter",
    "PoseDataEditorAloneAutomaticChaty": "Pose Data Editor Alone Automatic Chaty",
    "ImageBatchBlackout": "Image Batch Blackout",
    "PoseRetargetPromptHelper": "Pose Retarget Prompt Helper",
    "ImageBlackoutOnNoBBox": "Image Blackout on No BBox",
    "PoseDataAutomaticOffsetNodeV3": "Automatic Offset Node V3",
    "PoseDataAutomaticOffsetNodeV4": "Automatic Offset Node V4",
    "PoseDataAutoBlackoutOnJitter": "Auto Blackout On Jitter",
    "PoseDataToMask": "PoseData to Mask",
    "PoseDataToOvalMask": "PoseData to Oval Mask",
    "SavePoseDataNode": "Save Pose Data (Debug)",
    "PoseDataHandOffsetTimed": "Pose Data Hand Offset (Timed)",
    "PoseDataHandDeleterTimed": "Pose Data Hand Deleter (Timed)",
    "PoseDataConfidenceFilter": "Pose Data Confidence Filter",
    "PoseDataSmartHandFilterTimed": "Pose Data Smart Hand Filter (Timed)",
    "DrawViTPose_v2": "Draw ViT Pose v2 (Fixed Order)",
    "PoseDataHipHandDebugV2": "Pose Data Hip & Hand Debug V2",
    "DrawViTPose_v3": "Draw ViT Pose v3 (Body>Legs>Arms)",
    "KeypointDeleter": "Keypoint Deleter (Remove Limbs)",
    "MaskPositionalCutterV14": "Mask Positional Cutter V14",
    "WanFrameSyncSettingsV5": "Wan Frame Sync Settings V5",
    "WanSmartImageBatcherV2": "Wan Smart Image Batcher V2",
    "MaskPositionalJoinerV20": "Mask Positional Joiner V20",
    "MaskPositionalCutterV21": "Mask Positional Cutter V21",
    "MaskPositionalJoinerV21": "Mask Positional Joiner V21",
    "PoseDataHipHandDebugV3": "Pose Data Hip & Hand Debug V3",
    "PoseDataToMaskV2": "Pose Data To Mask V2",
    "PoseDataSelectFrameNode": "Pose Data Select Frame",
    "LoadPoseDataFromJsonNode": "Load Pose Data From JSON",
    "PoseDataToDWPoses": "PoseDataToDWPoses",
    "RenderNLFPosesWithData": "Render NLF Poses & Data (SCAIL)",
    "SavePoseCalibration": "Save Pose Calibration (Ultimate)",
    "LoadPoseCalibration": "Load Pose Calibration (Ultimate)",
    "PoseLocalBoneRetargeterV10": "Pose Local Bone Retargeter V10B",
    "PoseDataLowerLegRemover": "Pose Data Lower Leg Remover",
    "NLFDataToPoseData": "NLF Data to 2D Pose Data",
    "RenderNLFPosesDirect": "Render NLF Poses Direct",
    "PoseGlobalPerspectiveScalerV30": "Pose Global Perspective Scaler V30",
    "PoseGlobalPerspectiveScalerV38": "Pose Global Perspective Scaler V38 (Smart Camera Zoom)",
    "PoseCalibrationV20": "Pose Calibration V20 (Full 3D & Smart Calf)",
    "PoseCalibrationV22": "Pose Calibration V22",
    "NLFProportionalRetargeterV5": "NLF Proportional Retargeter V5",
    "NLFConfigScaler3DBones": "NLF Config caler 3D Bones",
    "RenderNLFPosesDirect7": "Render NLF Poses Direct 7",
    "RenderNLFPosesDirectPoseDataMimic13": "Render NLF Poses Mimic 13 (Flat 3D)",
    "RenderNLFPosesDirectPoseDataMimic14": "Render NLF Poses Mimic 14 (Flat 3D PoseData)",
    "PoseCalibrationV23": "Pose Calibration V23",
    "PoseCalibrationV24": "Pose Calibration V24",
    "PoseGlobalPerspectiveScalerV39": "Pose Global Perspective Scaler V39",
    "PoseCalibrationV15": "Pose Calibration V15",
    "PoseGlobalPerspectiveScalerV28": "Pose Global Perspective Scaler V28",
    "PoseGlobalPerspectiveScalerV40": "Global Perspective Scaler V40 (V28+V38 Best-of)",
    "PoseCalibrationV25": "Pose Calibration V25",
    "PoseGlobalPerspectiveScalerV41": "Global Perspective Scaler V41 (V28+V38 Best-of)",
    "PoseCalibrationManipulator": "Pose Calibration Manipulator",
    "PoseCalibrationV29": "Pose Calibration V29",
    "PoseGlobalPerspectiveScalerV43": "Pose Global Perspective Scaler V43",
    "NLFProportionalRetargeterV6": "NLF Proportional Retargeter V6",
    "PoseGlobalPerspectiveScalerV46": "Pose Global Perspective Scaler V46",
    "NLFProportionalRetargeterV7": "NLF Proportional Retargeter V7",
    "PoseGlobalPerspectiveScalerV47": "Pose Global Perspective Scaler V47",
    "NLFConfigScaler3DBones2": "NLF Config Scaler 3D Bones2",
    "NLFProportionalRetargeterV9": "NLF Proportional Retargeter V9",
    "PoseGlobalPerspectiveScalerV48": "Pose Global Perspective Scaler V48",
    "PoseGlobalPerspectiveScalerV49": "Pose Global Perspective Scaler V49",
    "NLFDataToMaskV2": "NLF Data to Mask V2 (3D)",
    "RenderNLFPosesDirectPoseDataMimic15": "Render NLF Poses Mimic 15 (Flat 3D PoseData)",
    "NLFDataToMaskV3": "NLF Data to Mask V3",
    "NLFDataToMaskV4": "NLF Data to Mask V4",
    "NLFDataHandDebugV3": "NLF Data Hand Debug V3 (Collision / IK)",
    "NLFDataHandDebugV4": "NLF Data Hand Debug V4 (Collision / IK)",
    "NLFDataHandDebugV5": "NLF Data Hand Debug V5 (Collision / IK)",
    "NLFDataHandDebugV6": "NLF Data Hand Debug V6 (Collision / IK)",
    "NLFDataHandDebugV7": "NLF Data Hand Debug V7 (Collision / IK)",
    "NLFDataHandDebugV8": "NLF Data Hand Debug V8 (Collision / IK)",
    "NLFDataHandDebugV9": "NLF Data Hand Debug V9 (Collision / IK)",
    "NLFDataHandDebugV10": "NLF Data Hand Debug V10 (Collision / IK)",
    "NLFDataHandDebugV11": "NLF Data Hand Debug V11 (Collision / IK)",
    "NLFDataHandDebugV12": "NLF Data Hand Debug V12 (Collision / IK)",
    "NLFProportionalRetargeterV13": "NLF Proportional Retargeter V13",
    "RenderNLFPosesDirectPoseDataMimic16": "Render NLF Poses Mimic 16 (Flat 3D PoseData)",
    "PoseCalibrationManipulator2": "Pose Calibration Manipulator2",
    "FrameSubsamplerForDepth": "Frame Subsampler For Depth (VRAM Saver)",
    "PoseGlobalPerspectiveScalerV50": "Pose Global Perspective Scaler (V50)",
    "PoseGlobalPerspectiveScalerV51": "Pose Global Perspective Scaler (V51)",
    "NLFProportionalRetargeterV14": "NLF Proportional Retargeter V14",
    "NLFPhysicalScalerV1": "NLF Physical Scaler V1",
    "NLFProportionalRetargeterV17": "NLF Proportional Retargeter V17",
    "NLFProportionalRetargeterV17ex": "NLF Proportional Retargeter V17ex",
    "NLFDataToMaskV5": "NLF Data to Mask V5",
    "RenderNLFPosesDirectPoseDataMimic17": "Render NLF Poses (Mimic 17 PoseData Fix)",
    "RenderNLFPosesOrthographicMimic": "Render NLF Poses Orthographic Mimic",
    "PoseCalibrationV30": "🎯 Pose Calibration Hub (V30)",
    "PoseGlobalPerspectiveScalerV53": "⚖️ Pose Global Perspective Scaler (V53)",
    "NLFPoseDataSelectFrame": "NLF Pose Data Select Frame",
    "PoseCalibrationV31": "🎯 Pose Calibration Hub (V31)",
    "PoseGlobalPerspectiveScalerV54": "⚖️ Pose Global Perspective Scaler (V54)",
    "PoseCalibrationV32": "🎯 Pose Calibration Hub (V32)",
    "PoseGlobalPerspectiveScalerV55": "⚖️ Pose Global Perspective Scaler (V55)",
    "PoseCalibrationV33": "🎯 Pose Calibration Hub (V33)",
    "PoseGlobalPerspectiveScalerV56": "⚖️ Pose Global Perspective Scaler (V56)",
    "PoseCalibrationManipulator3": "🔧 Pose Calibration Manipulator (V3)",
    "PoseAndFaceDetectionV8_NoWarp": "Pose And Face Detection V8 (No Warp)",
    "WanFaceStitcherV4": "Wan Face Stitcher V4",
    "RenderNLFPosesDirectHybrid8": "Render NLF Poses Direct Hybrid 8",
    "NLFProportionalRetargeterV18": "NLF Proportional Retargeter V18",
    "NLFProportionalRetargeterV181": "NLF Proportional Retargeter V18.1",
    "NLFProportionalRetargeterV19": "NLF Proportional Retargeter V19",
    "NLFProportionalRetargeterV20": "NLF Proportional Retargeter V20",
    "NLFProportionalRetargeterV21": "NLF Proportional Retargeter V21",
    "PoseGlobalPerspectiveScalerV57": "Pose Global Perspective Scaler V57",

    
    
}


