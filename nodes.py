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

class PoseAndFaceDetectionV7_NoWarp:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Breite der Pose-Generierung"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Höhe der Pose-Generierung"}),
                "face_resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64, "tooltip": "Zielauflösung (quadratisch)."}),
                "face_pad_factor": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "Padding um das Gesicht."}),
            },
            "optional": {
                "retarget_image": ("IMAGE", {"default": None, "tooltip": "Optionales Referenzbild"}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "FACE_INFO", "STRING", "BBOX", "BBOX")
    RETURN_NAMES = ("pose_data", "face_images", "face_info", "key_frame_body_points", "bboxes", "face_bboxes")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "V7 (No Warp): V4 Tracking-Logik, aber mit Padding statt Verzerrung. Input wird quadratisch gemacht."

    def process(self, model, images, width, height, face_resolution, face_pad_factor, retarget_image=None):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()
        
        # --- 1. Original V2/V4 Retarget Logic ---
        refer_pose_meta = None
        refer_img = None
        
        if retarget_image is not None:
            refer_img = resize_by_area(retarget_image[0].numpy() * 255, width * height, divisor=16) / 255.0
            ref_bbox = (detector(
                cv2.resize(refer_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])

            if ref_bbox is None or ref_bbox[-1] <= 0 or (ref_bbox[2] - ref_bbox[0]) < 10 or (ref_bbox[3] - ref_bbox[1]) < 10:
                ref_bbox = np.array([0, 0, refer_img.shape[1], refer_img.shape[0]])

            center, scale = bbox_from_detector(ref_bbox, input_resolution, rescale=rescale)
            refer_img_crop = crop(refer_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (refer_img_crop - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            ref_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            refer_pose_meta = load_pose_metas_from_kp2ds_seq(ref_keypoints, width=retarget_image.shape[2], height=retarget_image.shape[1])[0]

        # --- 2. Original V2/V4 Detection Loop ---
        comfy_pbar = ProgressBar(B*2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="V7 NoWarp Detecting"):
            det_result = detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
            )
            bbox_res = det_result[0][0]["bbox"]
            bboxes.append(bbox_res)
            
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        # --- 3. Original V2/V4 Pose Loop ---
        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="V7 NoWarp Keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img_crop = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img_crop - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

        # --- 4. Face Extraction (MODIFIZIERT: Padding statt Warping) ---
        face_images = []
        face_bboxes = []
        face_info = []

        for idx, meta in enumerate(pose_metas):
            # V4-Logik für Box-Koordinaten
            current_scale = 1.0 + face_pad_factor
            face_bbox_for_image = get_face_bboxes(meta['keypoints_face'][:, :2], scale=current_scale, image_shape=(H, W))
            
            raw_x1, raw_x2, raw_y1, raw_y2 = face_bbox_for_image
            raw_w = raw_x2 - raw_x1
            raw_h = raw_y2 - raw_y1
            
            # --- Square Logic (V7: No Warp) ---
            max_side = max(raw_w, raw_h)
            center_x = raw_x1 + raw_w / 2
            center_y = raw_y1 + raw_h / 2
            
            sq_x1 = int(center_x - max_side / 2)
            sq_y1 = int(center_y - max_side / 2)
            sq_x2 = sq_x1 + max_side
            sq_y2 = sq_y1 + max_side
            
            # Clamping
            safe_x1 = max(0, sq_x1)
            safe_y1 = max(0, sq_y1)
            safe_x2 = min(W, sq_x2)
            safe_y2 = min(H, sq_y2)
            
            valid = True
            
            # Crop mit Padding
            if safe_x2 > safe_x1 and safe_y2 > safe_y1:
                crop_img = images_np[idx][safe_y1:safe_y2, safe_x1:safe_x2]
                
                pad_l = safe_x1 - sq_x1
                pad_t = safe_y1 - sq_y1
                pad_r = sq_x2 - safe_x2
                pad_b = sq_y2 - safe_y2
                
                if any([pad_l > 0, pad_t > 0, pad_r > 0, pad_b > 0]):
                    crop_img = cv2.copyMakeBorder(
                        crop_img, 
                        max(0, pad_t), max(0, pad_b), max(0, pad_l), max(0, pad_r), 
                        cv2.BORDER_CONSTANT, 
                        value=(0,0,0)
                    )
            else:
                crop_img = np.zeros((face_resolution, face_resolution, C), dtype=images_np.dtype)
                valid = False

            # Resize (jetzt verzerrungsfrei)
            if crop_img.shape[0] != face_resolution or crop_img.shape[1] != face_resolution:
                face_image_resized = cv2.resize(crop_img, (face_resolution, face_resolution), interpolation=cv2.INTER_CUBIC)
            else:
                face_image_resized = crop_img

            face_images.append(face_image_resized)
            face_bboxes.append((sq_x1, sq_y1, sq_x2, sq_y2))
            
            info_entry = {
                "frame_index": idx,
                "original_img_shape": (W, H),
                "target_tensor_size": (face_resolution, face_resolution),
                "valid": valid,
                "crop_coords": (float(sq_x1), float(sq_y1), float(sq_x2), float(sq_y2)),
                "padding": (0, 0, 0, 0)
            }
            face_info.append(info_entry)

        face_images_tensor = torch.from_numpy(np.stack(face_images, 0))

        # --- 5. Restliche Outputs (unverändert) ---
        if retarget_image is not None and refer_pose_meta is not None:
            retarget_pose_metas = get_retarget_pose(pose_metas[0], refer_pose_meta, pose_metas, None, None)
        else:
            retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

        final_bboxes_list = []
        for bb in bboxes:
            bb_flat = np.array(bb).flatten()
            if bb_flat.shape[0] >= 4:
                bbox_ints = tuple(int(v) for v in bb_flat[:4])
            else:
                bbox_ints = (0, 0, 0, 0)
            final_bboxes_list.append(bbox_ints)

        key_frame_num = 4 if B >= 4 else 1
        key_frame_step = len(pose_metas) // key_frame_num
        key_frame_index_list = list(range(0, len(pose_metas), key_frame_step))
        key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]
        
        points_dict_list = [] 
        for key_frame_index in key_frame_index_list:
            if key_frame_index < len(pose_metas):
                keypoints_body_list = []
                body_key_points = pose_metas[key_frame_index]['keypoints_body']
                for each_index in key_points_index:
                    each_keypoint = body_key_points[each_index]
                    if None is each_keypoint:
                        continue
                    keypoints_body_list.append(each_keypoint)

                if keypoints_body_list:
                    keypoints_body = np.array(keypoints_body_list)[:, :2]
                    wh = np.array([[pose_metas[0]['width'], pose_metas[0]['height']]])
                    points = (keypoints_body * wh).astype(np.int32)
                    for point in points:
                        points_dict_list.append({"x": int(point[0]), "y": int(point[1])})

        pose_data = {
            "retarget_image": refer_img if retarget_image is not None else None,
            "pose_metas": retarget_pose_metas,
            "refer_pose_meta": refer_pose_meta if retarget_image is not None else None,
            "pose_metas_original": pose_metas,
        }

        return (pose_data, face_images_tensor, face_info, json.dumps(points_dict_list), final_bboxes_list, face_bboxes)


class WanFaceStitcherV3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination_images": ("IMAGE",),
                "face_images_v2": ("IMAGE",),
                "face_info_v2": ("FACE_INFO",),
                "mode": (["Resize Face to Dest", "Resize Dest to Fit Face"], {"default": "Resize Dest to Fit Face"}),
                "blend_feather": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stitched_images",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "V2 Stitcher: Fügt V2-Faces wieder ein."

    def process(self, destination_images, face_images_v2, face_info_v2, mode, blend_feather):
        dest_np = destination_images.cpu().numpy()
        faces_np = face_images_v2.cpu().numpy()
        B, H_dest, W_dest, _ = dest_np.shape
        
        if len(face_info_v2) == 0: return (destination_images,)

        # Smart Scale Calc
        target_w, target_h = W_dest, H_dest
        if mode == "Resize Dest to Fit Face":
            valid_info = next((f for f in face_info_v2 if f.get("valid")), None)
            if valid_info:
                orig_crop_w = valid_info["crop_coords"][2] - valid_info["crop_coords"][0]
                if orig_crop_w > 0:
                    zoom = faces_np.shape[2] / orig_crop_w
                    target_w, target_h = int(valid_info["original_img_shape"][0] * zoom), int(valid_info["original_img_shape"][1] * zoom)

        out = np.zeros((B, target_h, target_w, 3), dtype=np.float32)

        for i in tqdm(range(B), desc="V2 Stitching"):
            bg = dest_np[i]
            if bg.shape[1] != target_w or bg.shape[0] != target_h: bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            out[i] = bg
            
            if i >= len(face_info_v2) or not face_info_v2[i]["valid"]: continue
            
            info = face_info_v2[i]
            # Coords projizieren
            sx, sy = target_w / info["original_img_shape"][0], target_h / info["original_img_shape"][1]
            c = info["crop_coords"]
            tx1, ty1, tx2, ty2 = int(c[0]*sx), int(c[1]*sy), int(c[2]*sx), int(c[3]*sy)
            tw, th = tx2 - tx1, ty2 - ty1
            
            if tw <= 0 or th <= 0: continue
            
            # Face resize fit
            face = faces_np[i]
            if face.shape[1] != tw or face.shape[0] != th: face = cv2.resize(face, (tw, th), interpolation=cv2.INTER_AREA)
            
            # Mask & Blend
            mask = np.ones((th, tw, 1), dtype=np.float32)
            if blend_feather > 0:
                f = min(blend_feather, tw//2, th//2)
                if f > 0:
                    g = np.linspace(0, 1, f)
                    mask[:f, :] *= g[:, None, None]; mask[-f:, :] *= g[::-1, None, None]
                    mask[:, :f] *= g[None, :, None]; mask[:, -f:] *= g[None, ::-1, None]

            # Clipping
            dx1, dy1 = max(0, tx1), max(0, ty1)
            dx2, dy2 = min(target_w, tx2), min(target_h, ty2)
            sx1, sy1 = dx1 - tx1, dy1 - ty1
            sx2, sy2 = sx1 + (dx2-dx1), sy1 + (dy2-dy1)
            
            if dx2 > dx1 and dy2 > dy1:
                out[i, dy1:dy2, dx1:dx2] = face[sy1:sy2, sx1:sx2] * mask[sy1:sy2, sx1:sx2] + out[i, dy1:dy2, dx1:dx2] * (1.0 - mask[sy1:sy2, sx1:sx2])

        return (torch.from_numpy(out),)

class KeypointTrimNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "pose_data": ("POSEDATA",),
                "trim_start": ("BOOLEAN", {"default": True, "tooltip": "Schneidet den Anfang weg, bevor der erste Keypoint auftaucht."}),
                "trim_end": ("BOOLEAN", {"default": False, "tooltip": "Schneidet das Ende weg, nachdem der letzte Keypoint verschwindet."}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 960, "step": 1, "tooltip": "Wichtig für korrekten Audio-Schnitt."}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "face_images": ("IMAGE",),
                "optional_images": ("IMAGE", {"tooltip": "Zweiter Bild-Input zum Mit-Schneiden (z.B. Masken oder Pose-Bilder)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA", "AUDIO", "IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "pose_data", "audio", "face_images", "optional_images", "start_frame", "end_frame")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Trims video, audio, and pose data based on the presence of detected keypoints."

    def process(self, images, pose_data, trim_start, trim_end, fps, audio=None, face_images=None, optional_images=None):
        # 1. Daten vorbereiten
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        
        # Sicherheitscheck
        if not pose_metas:
            print("KeypointTrim: Keine Pose-Daten gefunden.")
            return (images, pose_data, audio, face_images, optional_images, 0, len(images))

        total_frames = len(pose_metas)
        
        # 2. Keypoints analysieren
        first_valid_idx = 0
        last_valid_idx = total_frames - 1
        
        def has_keypoints(meta):
            if not meta: return False
            # Prüfe Body, Hands, Face auf Existenz und Score > 0.05
            for attr in ["kps_body_p", "kps_lhand_p", "kps_rhand_p", "kps_face_p"]:
                if hasattr(meta, attr):
                    val = getattr(meta, attr)
                    if val is not None and np.any(val > 0.05):
                        return True
            return False

        # Start finden
        if trim_start:
            for i in range(total_frames):
                if has_keypoints(pose_metas[i]):
                    first_valid_idx = i
                    break
        
        # Ende finden
        if trim_end:
            for i in range(total_frames - 1, -1, -1):
                if has_keypoints(pose_metas[i]):
                    last_valid_idx = i
                    break
        
        # Logik-Check: Wurde überhaupt etwas gefunden?
        if first_valid_idx > last_valid_idx:
            print("KeypointTrim: WARNUNG - Keine Keypoints im gesamten Video gefunden! Gebe leeres Ergebnis zurück.")
            first_valid_idx = 0
            last_valid_idx = 0 

        # Python Slicing ist exklusiv am Ende, daher +1
        cut_end = last_valid_idx + 1
        
        # Info-Ausgabe in die Konsole
        print(f"KeypointTrim: Schneide von Frame {first_valid_idx} bis {cut_end}. (Behalte {cut_end - first_valid_idx} Frames)")

        # 3. Images schneiden (Haupt-Video)
        # Wir clippen die Indices, damit sie nicht abstürzen, falls Images kürzer/länger als PoseData sind
        img_len = images.shape[0]
        i_start = min(first_valid_idx, img_len)
        i_end = min(cut_end, img_len)
        
        trimmed_images = images[i_start:i_end]

        # 4. Pose Data schneiden
        pose_data_copy["pose_metas"] = pose_metas[first_valid_idx:cut_end]
        if "pose_metas_original" in pose_data_copy:
             pose_data_copy["pose_metas_original"] = pose_data_copy["pose_metas_original"][first_valid_idx:cut_end]

        # 5. Face Images schneiden
        trimmed_face_images = None
        if face_images is not None:
            if isinstance(face_images, list): # Falls es eine Liste ist
                f_len = len(face_images)
                trimmed_face_images = face_images[min(first_valid_idx, f_len):min(cut_end, f_len)]
            else: # Falls es ein Tensor ist
                f_len = face_images.shape[0]
                trimmed_face_images = face_images[min(first_valid_idx, f_len):min(cut_end, f_len)]

        # 6. Optional Images schneiden
        trimmed_optional = None
        if optional_images is not None:
            o_len = optional_images.shape[0]
            trimmed_optional = optional_images[min(first_valid_idx, o_len):min(cut_end, o_len)]

        # 7. Audio schneiden
        trimmed_audio = None
        if audio is not None:
            try:
                waveform = audio['waveform']
                sample_rate = audio['sample_rate']
                
                # Zeit berechnen
                start_time = first_valid_idx / float(fps)
                # End time based on duration frames
                end_time = cut_end / float(fps) 
                
                # Samples berechnen
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Waveform schneiden
                if waveform.ndim == 3:
                    new_waveform = waveform[:, :, start_sample:end_sample]
                elif waveform.ndim == 2:
                    new_waveform = waveform[:, start_sample:end_sample]
                else:
                    new_waveform = waveform 
                
                trimmed_audio = {'waveform': new_waveform, 'sample_rate': sample_rate}
                
            except Exception as e:
                print(f"KeypointTrim: Audio Error: {e}")
                trimmed_audio = audio

        return (trimmed_images, pose_data_copy, trimmed_audio, trimmed_face_images, trimmed_optional, first_valid_idx, last_valid_idx)

class OnnxDetectionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": (folder_paths.get_filename_list("detection"), {"tooltip": "These models are loaded from the 'ComfyUI/models/detection' -folder",}),
                "yolo_model": (folder_paths.get_filename_list("detection"), {"tooltip": "These models are loaded from the 'ComfyUI/models/detection' -folder",}),
                "onnx_device": (["CUDAExecutionProvider", "CPUExecutionProvider"], {"default": "CUDAExecutionProvider", "tooltip": "Device to run the ONNX models on"}),
            },
        }

    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Loads ONNX models for pose and face detection. ViTPose for pose estimation and YOLO for object detection."

    def loadmodel(self, vitpose_model, yolo_model, onnx_device):

        vitpose_model_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_model_path = folder_paths.get_full_path_or_raise("detection", yolo_model)

        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device)

        model = {
            "vitpose": vitpose,
            "yolo": yolo,
        }

        return (model, )

class PoseAndFaceDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
            },
            "optional": {
                "retarget_image": ("IMAGE", {"default": None, "tooltip": "Optional reference image for pose retargeting"}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", "BBOX,")
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", "face_bboxes")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Detects human poses and face images from input images. Optionally retargets poses based on a reference image."

    def process(self, model, images, width, height, retarget_image=None):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()
        if retarget_image is not None:
            refer_img = resize_by_area(retarget_image[0].numpy() * 255, width * height, divisor=16) / 255.0
            ref_bbox = (detector(
                cv2.resize(refer_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])

            if ref_bbox is None or ref_bbox[-1] <= 0 or (ref_bbox[2] - ref_bbox[0]) < 10 or (ref_bbox[3] - ref_bbox[1]) < 10:
                ref_bbox = np.array([0, 0, refer_img.shape[1], refer_img.shape[0]])

            center, scale = bbox_from_detector(ref_bbox, input_resolution, rescale=rescale)
            refer_img = crop(refer_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (refer_img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            ref_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            refer_pose_meta = load_pose_metas_from_kp2ds_seq(ref_keypoints, width=retarget_image.shape[2], height=retarget_image.shape[1])[0]

        comfy_pbar = ProgressBar(B*2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

        face_images = []
        face_bboxes = []
        for idx, meta in enumerate(pose_metas):
            face_bbox_for_image = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, image_shape=(H, W))
            x1, x2, y1, y2 = face_bbox_for_image
            face_bboxes.append((x1, y1, x2, y2))
            face_image = images_np[idx][y1:y2, x1:x2]
            # Check if face_image is valid before resizing
            if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
                logging.warning(f"Empty face crop on frame {idx}, creating fallback image.")
                # Create a fallback image (black or use center crop)
                fallback_size = int(min(H, W) * 0.3)
                fallback_x1 = (W - fallback_size) // 2
                fallback_x2 = fallback_x1 + fallback_size
                fallback_y1 = int(H * 0.1)
                fallback_y2 = fallback_y1 + fallback_size
                face_image = images_np[idx][fallback_y1:fallback_y2, fallback_x1:fallback_x2]
                
                # If still empty, create a black image
                if face_image.size == 0:
                    face_image = np.zeros((fallback_size, fallback_size, C), dtype=images_np.dtype)
            face_image = cv2.resize(face_image, (512, 512))
            face_images.append(face_image)

        face_images_np = np.stack(face_images, 0)
        face_images_tensor = torch.from_numpy(face_images_np)

        if retarget_image is not None and refer_pose_meta is not None:
            retarget_pose_metas = get_retarget_pose(pose_metas[0], refer_pose_meta, pose_metas, None, None)
        else:
            retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

        bbox = np.array(bboxes[0]).flatten()
        if bbox.shape[0] >= 4:
            bbox_ints = tuple(int(v) for v in bbox[:4])
        else:
            bbox_ints = (0, 0, 0, 0)

        key_frame_num = 4 if B >= 4 else 1
        key_frame_step = len(pose_metas) // key_frame_num
        key_frame_index_list = list(range(0, len(pose_metas), key_frame_step))

        key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]

        for key_frame_index in key_frame_index_list:
            keypoints_body_list = []
            body_key_points = pose_metas[key_frame_index]['keypoints_body']
            for each_index in key_points_index:
                each_keypoint = body_key_points[each_index]
                if None is each_keypoint:
                    continue
                keypoints_body_list.append(each_keypoint)

            keypoints_body = np.array(keypoints_body_list)[:, :2]
            wh = np.array([[pose_metas[0]['width'], pose_metas[0]['height']]])
            points = (keypoints_body * wh).astype(np.int32)
            points_dict_list = []
            for point in points:
                points_dict_list.append({"x": int(point[0]), "y": int(point[1])})

        pose_data = {
            "retarget_image": refer_img if retarget_image is not None else None,
            "pose_metas": retarget_pose_metas,
            "refer_pose_meta": refer_pose_meta if retarget_image is not None else None,
            "pose_metas_original": pose_metas,
        }

        return (pose_data, face_images_tensor, json.dumps(points_dict_list), [bbox_ints], face_bboxes)


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

class PoseDataAutoBlackoutOnJitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "pose_data": ("POSEDATA",),
                "jitter_threshold": (
                    "FLOAT",
                    {
                        "default": 50.0,
                        "min": 1.0,
                        "max": 500.0,
                        "step": 1.0,
                        "tooltip": "Wie stark darf sich die Figur bewegen? Wenn die Figur am Ende 'spinnt', ist der Wert extrem hoch.",
                    },
                ),
                "consecutive_frames": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Wie viele Frames muss es zittern, bevor der Blackout startet?",
                    },
                ),
                "remove_pose_data": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Soll auch das Skelett für diese Frames gelöscht werden?",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA", "INT")
    RETURN_NAMES = ("images", "pose_data", "blackout_start_frame")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Analysiert die Pose und macht das Bild automatisch schwarz, sobald die Figur am Ende anfängt zu 'spinnen' (Jitter)."

    def process(self, images, pose_data, jitter_threshold, consecutive_frames, remove_pose_data):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        
        # Bilder kopieren (Tensor)
        new_images = images.clone()
        batch_size = new_images.shape[0]
        
        if not pose_metas:
            return (new_images, pose_data_copy, -1)

        # Wir suchen den Frame, ab dem das Chaos beginnt
        blackout_start_index = -1
        bad_frame_counter = 0

        # Wir starten bei Frame 1 (wir brauchen Frame 0 zum Vergleich)
        for i in range(1, len(pose_metas)):
            curr_meta = pose_metas[i]
            prev_meta = pose_metas[i-1]
            
            if not isinstance(curr_meta, AAPoseMeta) or not isinstance(prev_meta, AAPoseMeta):
                continue

            # Wir berechnen die durchschnittliche Bewegung aller Körperteile
            total_dist = 0.0
            valid_points = 0
            
            # Body Keypoints prüfen
            kp_curr = getattr(curr_meta, "kps_body", None)
            score_curr = getattr(curr_meta, "kps_body_p", None)
            kp_prev = getattr(prev_meta, "kps_body", None)
            score_prev = getattr(prev_meta, "kps_body_p", None)

            if kp_curr is not None and kp_prev is not None:
                # Nur sichtbare Punkte vergleichen
                for idx in range(min(len(kp_curr), len(kp_prev))):
                    # HIER IST DER SICHERHEITSCHECK:
                    # Wenn der JitterDeleter den Punkt gelöscht hat (Score 0), wird er hier ignoriert.
                    if score_curr[idx] > 0.05 and score_prev[idx] > 0.05:
                        # Euklidische Distanz
                        dist = math.sqrt(
                            (kp_curr[idx][0] - kp_prev[idx][0])**2 + 
                            (kp_curr[idx][1] - kp_prev[idx][1])**2
                        )
                        total_dist += dist
                        valid_points += 1
            
            # Durchschnittliche Bewegung ("Jitter Score")
            avg_jitter = total_dist / valid_points if valid_points > 0 else 0.0
            
            # Wenn die Bewegung extrem hoch ist (das "Spinnen" am Rand), zählen wir hoch
            if avg_jitter > jitter_threshold:
                bad_frame_counter += 1
            else:
                bad_frame_counter = 0 # Reset, war nur ein kurzer Ruckler
            
            # Wenn es oft genug hintereinander passiert ist -> Blackout auslösen!
            if bad_frame_counter >= consecutive_frames:
                # Wir gehen zurück zum ersten Frame des Fehlers
                blackout_start_index = i - (consecutive_frames - 1)
                break # Abbruch, wir haben den Punkt gefunden

        # Blackout anwenden
        if blackout_start_index != -1:
            print(f"AutoBlackout: Jitter detected starting at frame {blackout_start_index}. Blacking out remaining {batch_size - blackout_start_index} frames.")
            
            # 1. Bilder schwärzen
            if blackout_start_index < batch_size:
                new_images[blackout_start_index:, :, :, :] = 0.0
            
            # 2. Pose Daten löschen (optional)
            if remove_pose_data:
                for d_idx in range(blackout_start_index, len(pose_metas)):
                    # Wir setzen einfach leere Daten
                    meta = pose_metas[d_idx]
                    if hasattr(meta, "kps_body_p"):
                        meta.kps_body_p[:] = 0.0 # Alles unsichtbar machen
                    if hasattr(meta, "kps_lhand_p"):
                        meta.kps_lhand_p[:] = 0.0
                    if hasattr(meta, "kps_rhand_p"):
                        meta.kps_rhand_p[:] = 0.0
                    if hasattr(meta, "kps_face_p"):
                        meta.kps_face_p[:] = 0.0

        return (new_images, pose_data_copy, blackout_start_index)


class ImageBlackoutOnNoBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bboxes": ("BBOX",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Blacks out the image frame if no valid BBOX is detected."

    def process(self, images, bboxes):
        # Wir erstellen eine Kopie der Bilder, um das Original nicht zu verändern
        new_images = images.clone()
        batch_size = images.shape[0]

        # Sicherstellen, dass bboxes als Liste vorliegt
        if not isinstance(bboxes, list):
            bboxes = [bboxes]

        for i in range(batch_size):
            is_bbox_valid = False

            # Wir prüfen, ob für den aktuellen Frame 'i' ein Eintrag in der BBox-Liste existiert.
            # Hinweis: Wenn die Detection Node weniger BBoxen liefert als Bilder da sind,
            # werden die überschüssigen Bilder schwarz.
            if i < len(bboxes):
                bbox = bboxes[i]
                
                # Prüfen ob bbox existiert und das Standardformat [x, y, w, h] hat
                if bbox is not None and len(bbox) >= 4:
                    w = bbox[2]
                    h = bbox[3]
                    
                    # Ein Bild gilt nur als "erkannt", wenn die Box eine Fläche hat
                    if w > 0 and h > 0:
                        is_bbox_valid = True

            # Wenn ungültig (keine Detection), wird der Frame schwarz (Wert 0.0)
            if not is_bbox_valid:
                new_images[i] = 0.0

        return (new_images,)


class PoseDataEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "target_region": (TARGET_OPTIONS, {"default": "BODY", "tooltip": "Select which set of keypoints to manipulate."}),
                "x_offset": ("FLOAT", {"default": 0.0, "min": -2048.0, "max": 2048.0, "step": 0.01, "tooltip": "Horizontal offset applied to the selected points."}),
                "y_offset": ("FLOAT", {"default": 0.0, "min": -2048.0, "max": 2048.0, "step": 0.01, "tooltip": "Vertical offset applied to the selected points."}),
                "normalized_offset": ("BOOLEAN", {"default": False, "tooltip": "Interpret offsets in normalised 0-1 space instead of pixels."}),
                "rotation_deg": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1, "tooltip": "Rotation angle applied around the centroid of the selected points."}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01, "tooltip": "Uniform scale applied when link scale axes is enabled."}),
                "link_scale_axes": ("BOOLEAN", {"default": False, "tooltip": "When enabled, the uniform scale value drives both X and Y axes."}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01, "tooltip": "Scale factor along the X axis (bi-directional)."}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01, "tooltip": "Scale factor along the Y axis (bi-directional)."}),
                "limit_scale_to_canvas": ("BOOLEAN", {"default": True, "tooltip": "Clamp transformed points so they stay within the canvas."}),
                "only_scale_up": ("BOOLEAN", {"default": False, "tooltip": "Prevent scale factors below 1.0 to avoid shrinking the selection."}),
                "only_scale_down": ("BOOLEAN", {"default": False, "tooltip": "Prevent scale factors above 1.0 to avoid enlarging the selection."}),
                "shift_pose_to_canvas": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Translate the entire pose after edits so every keypoint stays on the canvas before any clamping is applied.",
                    },
                ),
                "head_top_padding": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1024.0,
                        "step": 0.1,
                        "tooltip": "Minimum distance (in pixels) to keep between head keypoints and the top canvas edge when enforcing bounds.",
                    },
                ),
                "only_adjust_when_legs_long": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When editing legs or feet, only apply scaling when their normalised height span exceeds the configured threshold.",
                    },
                ),
                "min_leg_length_ratio": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Minimum normalised leg length (relative to canvas height) required before leg scaling is applied.",
                    },
                ),
                "strict_leg_guard": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, leg edits are skipped unless both legs have visible lower joints so torso points stay unchanged when detections are missing.",
                    },
                ),
                "require_visible_part": ("BOOLEAN", {"default": True, "tooltip": "Skip edits when any required keypoints for the selected region are not visible."}),
                "person_index": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1, "tooltip": "When >= 0, only edit the matching pose entry. Use -1 to edit every pose."}),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "edit"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Interactive editor for pose data allowing offsets, rotation and scaling of body, hand and face keypoints."

    def edit(
        self,
        pose_data,
        target_region,
        x_offset,
        y_offset,
        normalized_offset,
        rotation_deg,
        scale,
        link_scale_axes,
        scale_x,
        scale_y,
        limit_scale_to_canvas,
        only_scale_up,
        only_scale_down,
        shift_pose_to_canvas,
        head_top_padding,
        only_adjust_when_legs_long,
        min_leg_length_ratio,
        strict_leg_guard,
        require_visible_part,
        person_index,
    ):
        if only_scale_up and only_scale_down:
            raise ValueError(
                "Only one of 'only_scale_up' or 'only_scale_down' can be enabled at a time."
            )

        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        if link_scale_axes:
            scale_x = scale
            scale_y = scale

        indices = (
            [person_index]
            if isinstance(person_index, int) and person_index >= 0 and person_index < len(pose_metas)
            else list(range(len(pose_metas)))
        )

        for idx in indices:
            meta = pose_metas[idx]
            if meta is None:
                continue
            self._apply_edit(
                meta,
                target_region,
                x_offset,
                y_offset,
                normalized_offset,
                rotation_deg,
                scale_x,
                scale_y,
                limit_scale_to_canvas,
                only_scale_up,
                only_scale_down,
                shift_pose_to_canvas,
                head_top_padding,
                only_adjust_when_legs_long,
                min_leg_length_ratio,
                strict_leg_guard,
                require_visible_part,
            )

        return (pose_data_copy,)

    def _apply_edit(
        self,
        meta,
        target_region,
        x_offset,
        y_offset,
        normalized_offset,
        rotation_deg,
        scale_x,
        scale_y,
        limit_scale_to_canvas,
        only_scale_up,
        only_scale_down,
        shift_pose_to_canvas,
        head_top_padding,
        only_adjust_when_legs_long,
        min_leg_length_ratio,
        strict_leg_guard,
        require_visible_part,
    ):
        width = getattr(meta, "width", None)
        height = getattr(meta, "height", None)

        if width in (None, 0) or height in (None, 0):
            return

        selections = self._resolve_selection(meta, target_region)
        if not selections:
            return

        target_upper = target_region.upper()
        if require_visible_part:
            required_refs = self._required_refs_for_visibility(meta, target_upper)
            if required_refs and not all(
                self._is_point_visible(meta, arr_name, idx) for arr_name, idx in required_refs
            ):
                return

        points = []
        refs = []

        for arr_name, indices in selections:
            arr = getattr(meta, arr_name, None)
            if arr is None:
                continue

            if isinstance(indices, str) and indices == "ALL":
                iterable = range(len(arr))
            else:
                iterable = indices

            for idx in iterable:
                if idx >= len(arr):
                    continue

                point = arr[idx]
                if point is None:
                    continue

                if isinstance(point, np.ndarray):
                    if np.isnan(point).any():
                        continue
                    x, y = point.tolist()
                elif isinstance(point, (list, tuple)):
                    if len(point) < 2 or point[0] is None or point[1] is None:
                        continue
                    x, y = point[:2]
                else:
                    continue

                if arr_name == "kps_body" and getattr(meta, "kps_body_p", None) is not None:
                    if meta.kps_body_p[idx] <= 0:
                        continue
                if arr_name == "kps_lhand" and getattr(meta, "kps_lhand_p", None) is not None:
                    if meta.kps_lhand_p[idx] <= 0:
                        continue
                if arr_name == "kps_rhand" and getattr(meta, "kps_rhand_p", None) is not None:
                    if meta.kps_rhand_p[idx] <= 0:
                        continue

                points.append([float(x), float(y)])
                refs.append((arr_name, idx))

        if not points:
            return

        if (
            strict_leg_guard
            and target_upper == "LEGS"
            and not self._has_lower_leg_points(refs)
        ):
            return

        points_np = np.array(points, dtype=np.float32)
        center = points_np.mean(axis=0, keepdims=True)
        original_points = points_np.copy()

        leg_indices = set(BODY_GROUPS.get("LEGS", [])) | set(BODY_GROUPS.get("FEET", []))
        affects_legs = bool(refs) and all(
            arr_name == "kps_body" and idx in leg_indices for arr_name, idx in refs
        )

        scales = np.array([scale_x, scale_y], dtype=np.float32)
        if only_scale_up:
            scales = np.maximum(scales, np.ones_like(scales))
        if only_scale_down:
            scales = np.minimum(scales, np.ones_like(scales))

        if affects_legs and only_adjust_when_legs_long and height not in (None, 0):
            leg_span = float(np.ptp(original_points[:, 1]))
            leg_span_ratio = leg_span / float(height) if height else 0.0
            if leg_span_ratio < max(0.0, float(min_leg_length_ratio)):
                if scales[0] > 1.0:
                    scales[0] = 1.0
                if scales[1] > 1.0:
                    scales[1] = 1.0

        offset = np.array([x_offset, y_offset], dtype=np.float32)
        if normalized_offset:
            offset *= np.array([width, height], dtype=np.float32)

        theta = math.radians(rotation_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

        transformed = (points_np - center) * scales
        transformed = transformed @ rotation_matrix.T
        transformed = transformed + center

        vertical_offset_for_rest = 0.0
        if affects_legs and only_scale_up and (scales[0] > 1.0 or scales[1] > 1.0):
            vertical_offset_for_rest = max(0.0, float(np.min(original_points[:, 1]) - np.min(transformed[:, 1])))

        transformed = transformed + offset

        if limit_scale_to_canvas and not shift_pose_to_canvas:
            transformed[:, 0] = np.clip(transformed[:, 0], 0.0, float(width))
            transformed[:, 1] = np.clip(transformed[:, 1], 0.0, float(height))

        for (arr_name, idx), new_point in zip(refs, transformed.tolist()):
            if arr_name == "kps_body":
                meta.kps_body[idx] = new_point
            elif arr_name == "kps_lhand":
                meta.kps_lhand[idx] = new_point
            elif arr_name == "kps_rhand":
                meta.kps_rhand[idx] = new_point
            elif arr_name == "kps_face":
                meta.kps_face[idx] = new_point

        if vertical_offset_for_rest > 0.0:
            self._offset_unselected_points(
                meta,
                vertical_offset_for_rest,
                refs,
                limit_scale_to_canvas and not shift_pose_to_canvas,
                float(width),
                float(height),
            )

        self._enforce_canvas_bounds(
            meta,
            float(width),
            float(height),
            limit_scale_to_canvas,
            shift_pose_to_canvas,
            float(head_top_padding),
        )

    def _required_refs_for_visibility(self, meta, target_upper):
        if target_upper in ("ALL", "BODY"):
            return []

        if target_upper in BODY_GROUPS and target_upper != "ALL":
            return [("kps_body", idx) for idx in BODY_GROUPS[target_upper]]

        return []

    def _is_point_visible(self, meta, arr_name, idx):
        arr = getattr(meta, arr_name, None)
        if arr is None or idx >= len(arr):
            return False

        point = arr[idx]
        if point is None:
            return False

        if isinstance(point, np.ndarray):
            if point.shape[-1] < 2:
                return False
            if np.isnan(point[:2]).any():
                return False
        elif isinstance(point, (list, tuple)):
            if len(point) < 2 or point[0] is None or point[1] is None:
                return False
        else:
            return False

        prob_attr = getattr(meta, f"{arr_name}_p", None)
        if prob_attr is not None:
            if idx >= len(prob_attr) or prob_attr[idx] <= 0:
                return False

        return True

    def _offset_unselected_points(
        self,
        meta,
        vertical_offset,
        selected_refs,
        clamp_points,
        width,
        height,
    ):
        if vertical_offset <= 0.0:
            return

        selected_set = {(name, idx) for name, idx in selected_refs}

        for arr_name in ("kps_body", "kps_lhand", "kps_rhand", "kps_face"):
            arr = getattr(meta, arr_name, None)
            if arr is None:
                continue

            for idx in range(len(arr)):
                if (arr_name, idx) in selected_set:
                    continue

                coords = self._extract_coords(arr[idx])
                if coords is None:
                    continue

                new_x = coords[0]
                new_y = coords[1] - vertical_offset

                if clamp_points:
                    new_x = float(np.clip(new_x, 0.0, width))
                    new_y = float(np.clip(new_y, 0.0, height))

                self._assign_point(arr, idx, new_x, new_y)

    def _enforce_canvas_bounds(
        self,
        meta,
        width,
        height,
        limit_to_canvas,
        shift_pose,
        head_top_padding,
    ):
        if shift_pose:
            self._keep_pose_within_canvas(
                meta,
                width,
                height,
                limit_to_canvas,
                head_top_padding,
            )
        elif limit_to_canvas:
            self._clamp_pose(
                meta,
                width,
                height,
                head_top_padding,
                head_top_padding > 0.0,
            )

    def _keep_pose_within_canvas(
        self,
        meta,
        width,
        height,
        limit_to_canvas,
        head_top_padding,
    ):
        all_points, head_points = self._collect_pose_points(meta)

        if not all_points:
            return

        xs = [pt[2] for pt in all_points]
        ys = [pt[3] for pt in all_points]

        dx_min = -min(xs)
        dx_max = width - max(xs)
        dy_min = -min(ys)
        dy_max = height - max(ys)

        if head_points and head_top_padding > 0.0:
            head_min_y = min(pt[3] for pt in head_points)
            dy_min = max(dy_min, head_top_padding - head_min_y)

        dx = self._select_shift(dx_min, dx_max)
        dy = self._select_shift(dy_min, dy_max)

        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            self._apply_translation(meta, dx, dy)

        if limit_to_canvas:
            self._clamp_pose(
                meta,
                width,
                height,
                head_top_padding,
                head_top_padding > 0.0,
            )

    def _collect_pose_points(self, meta):
        all_points = []
        head_points = []
        head_indices = set(BODY_GROUPS.get("HEAD", []))

        for arr_name in ("kps_body", "kps_lhand", "kps_rhand", "kps_face"):
            arr = getattr(meta, arr_name, None)
            if arr is None:
                continue

            for idx in range(len(arr)):
                coords = self._extract_coords(arr[idx])
                if coords is None:
                    continue

                all_points.append((arr_name, idx, coords[0], coords[1]))

                if arr_name == "kps_body" and idx in head_indices:
                    head_points.append((arr_name, idx, coords[0], coords[1]))

        return all_points, head_points

    def _apply_translation(self, meta, dx, dy):
        if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
            return

        for arr_name in ("kps_body", "kps_lhand", "kps_rhand", "kps_face"):
            arr = getattr(meta, arr_name, None)
            if arr is None:
                continue

            for idx in range(len(arr)):
                coords = self._extract_coords(arr[idx])
                if coords is None:
                    continue

                self._assign_point(arr, idx, coords[0] + dx, coords[1] + dy)

    def _clamp_pose(
        self,
        meta,
        width,
        height,
        head_top_padding,
        enforce_head_padding,
    ):
        head_indices = set(BODY_GROUPS.get("HEAD", []))

        for arr_name in ("kps_body", "kps_lhand", "kps_rhand", "kps_face"):
            arr = getattr(meta, arr_name, None)
            if arr is None:
                continue

            for idx in range(len(arr)):
                coords = self._extract_coords(arr[idx])
                if coords is None:
                    continue

                min_y = 0.0
                if (
                    enforce_head_padding
                    and arr_name == "kps_body"
                    and idx in head_indices
                ):
                    min_y = head_top_padding

                clamped_x = float(np.clip(coords[0], 0.0, width))
                clamped_y = float(np.clip(coords[1], min_y, height))

                self._assign_point(arr, idx, clamped_x, clamped_y)

    def _extract_coords(self, point):
        if point is None:
            return None

        if isinstance(point, np.ndarray):
            if point.ndim == 0 or point.shape[-1] < 2:
                return None
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                return None
        elif isinstance(point, (list, tuple)):
            if len(point) < 2:
                return None
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                return None
        else:
            return None

        try:
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
        except (TypeError, ValueError):
            return None

        return x, y

    def _assign_point(self, arr, idx, x, y):
        x_val = float(x)
        y_val = float(y)

        if isinstance(arr, np.ndarray):
            if arr.ndim >= 2 and arr.shape[-1] >= 2:
                arr[idx, 0] = x_val
                arr[idx, 1] = y_val
            else:
                current = arr[idx]
                if isinstance(current, np.ndarray) and current.shape[-1] >= 2:
                    current[0] = x_val
                    current[1] = y_val
                    arr[idx] = current
                else:
                    arr[idx] = np.array([x_val, y_val], dtype=np.float32)
            return

        current = arr[idx]

        if current is None:
            current = [0.0, 0.0]
        elif isinstance(current, tuple):
            current = list(current)
        elif not isinstance(current, list):
            current = [float(current)]

        while len(current) < 2:
            current.append(0.0)

        current[0] = x_val
        current[1] = y_val

        arr[idx] = current

    def _select_shift(self, min_allowed, max_allowed):
        if min_allowed <= 0.0 <= max_allowed:
            return 0.0

        if min_allowed > max_allowed:
            return min_allowed if abs(min_allowed) <= abs(max_allowed) else max_allowed

        return min_allowed if abs(min_allowed) <= abs(max_allowed) else max_allowed

    def _has_lower_leg_points(self, refs):
        if not refs:
            return False

        right_leg_present = False
        left_leg_present = False

        for arr_name, idx in refs:
            if arr_name != "kps_body":
                continue

            if idx in (9, 10):
                right_leg_present = True
            elif idx in (12, 13):
                left_leg_present = True

            if right_leg_present and left_leg_present:
                return True

        return right_leg_present and left_leg_present

    def _resolve_selection(self, meta, target_region):
        target = target_region.upper()
        selections = []

        if target == "ALL":
            selections.append(("kps_body", BODY_GROUPS["ALL"]))
            if getattr(meta, "kps_lhand", None) is not None:
                selections.append(("kps_lhand", "ALL"))
            if getattr(meta, "kps_rhand", None) is not None:
                selections.append(("kps_rhand", "ALL"))
            if getattr(meta, "kps_face", None) is not None:
                selections.append(("kps_face", "ALL"))
            return selections

        if target == "BODY":
            selections.append(("kps_body", BODY_GROUPS["ALL"]))
            return selections

        if target in BODY_GROUPS:
            selections.append(("kps_body", BODY_GROUPS[target]))
            return selections

        if target in HAND_GROUPS:
            hand_target = HAND_GROUPS[target]
            if hand_target in ("left", "both") and getattr(meta, "kps_lhand", None) is not None:
                selections.append(("kps_lhand", "ALL"))
            if hand_target in ("right", "both") and getattr(meta, "kps_rhand", None) is not None:
                selections.append(("kps_rhand", "ALL"))
            return selections

        if target in FACE_GROUP and getattr(meta, "kps_face", None) is not None:
            selections.append(("kps_face", "ALL"))
            return selections

        return selections

class PoseDataAutomaticOffsetNodeV3:
    """
    V3: Berechnet den Offset rein proportional.
    Keine 'Canvas Height' mehr nötig.
    Wenn Source=1.70 und Target=2.20, wird die Person einfach um den Faktor (2.20/1.70) gestreckt.
    Füße bleiben fix, Beine passen sich an.
    """
    
    HEAD_INDICES = [0, 1, 2, 3, 4, 5] 
    HIP_INDICES = [8, 11]
    FOOT_INDICES = [10, 13, 18, 19, 20, 21, 22, 23, 24] 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "source_height": (
                    "FLOAT",
                    {
                        "default": 1.70,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.01,
                        "tooltip": "Die aktuelle/echte Größe der Person (z.B. 1.70m).",
                    },
                ),
                "target_height": (
                    "FLOAT",
                    {
                        "default": 2.20,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.01,
                        "tooltip": "Die Wunschgröße (z.B. 2.20m). Das Verhältnis Target/Source bestimmt die Streckung.",
                    },
                ),
                "analysis_duration": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Dauer der Analyse, um die aktuelle Größe im Bild zu messen.",
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 240,
                        "step": 1,
                        "tooltip": "Framerate.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "V3: Proportional scaling from Source Height to Target Height. Feet stay fixed."

    def process(self, pose_data, source_height, target_height, analysis_duration, fps):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        # 1. Analyse Phase: Wie groß ist die Person im Bild aktuell?
        fps = max(1, int(fps))
        analysis_frames = max(1, int(analysis_duration * fps))
        limit_frames = min(len(pose_metas), analysis_frames)
        
        head_y_samples = []
        foot_y_samples = []
        
        for i in range(limit_frames):
            meta = pose_metas[i]
            top_y = self._find_pose_top(meta)
            bot_y = self._find_pose_bottom(meta)
            
            if top_y is not None and bot_y is not None:
                head_y_samples.append(top_y)
                foot_y_samples.append(bot_y)
        
        if not head_y_samples:
            # Fallback: Nichts tun, wenn keine Person erkannt wurde
            return (pose_data_copy,)

        avg_head_y = float(np.median(head_y_samples))
        avg_foot_y = float(np.median(foot_y_samples))
        
        # Die aktuelle visuelle Größe (z.B. 0.5 der Bildhöhe)
        current_visual_height = avg_foot_y - avg_head_y
        
        if current_visual_height <= 0.001:
            return (pose_data_copy,)

        # 2. Berechnung des Skalierungsfaktors
        # Beispiel: 2.20 / 1.70 = 1.29 (Person soll 29% größer werden)
        scale_ratio = target_height / source_height
        
        # Die neue Wunsch-Größe im Bild
        target_visual_height = current_visual_height * scale_ratio
        
        # Wo muss der Kopf hin? (Füße bleiben bei avg_foot_y)
        target_head_y = avg_foot_y - target_visual_height
        
        # Offset berechnen: Wohin muss der Oberkörper geschoben werden?
        fixed_offset_y_norm = target_head_y - avg_head_y

        # 3. Anwendung
        for meta in pose_metas:
            height = getattr(meta, "height", 1.0) or 1.0
            
            offset_px = fixed_offset_y_norm * height
            
            # Alte Hüftpositionen sichern
            hip_coords_before = self._get_hip_coords(meta)
            
            # A. Oberkörper verschieben
            self._apply_offset_to_upper_body(meta, offset_px)
            
            # B. Beine anpassen (Füße bleiben fix, Knie interpolieren)
            if hip_coords_before:
                self._reconnect_legs(meta, hip_coords_before, offset_px)

        return (pose_data_copy,)

    # --- Helper Methoden (identisch zu V2, hier kopiert für Unabhängigkeit) ---

    def _find_pose_top(self, meta):
        body = getattr(meta, "kps_body", None)
        if body is None: return None
        min_y = float('inf')
        found = False
        for idx in self.HEAD_INDICES:
            if idx < len(body) and self._is_valid(body[idx]):
                y_norm = body[idx][1] / meta.height
                if y_norm < min_y:
                    min_y = y_norm
                    found = True
        return min_y if found else None

    def _find_pose_bottom(self, meta):
        body = getattr(meta, "kps_body", None)
        if body is None: return None
        max_y = float('-inf')
        found = False
        for idx in self.FOOT_INDICES:
            if idx < len(body) and self._is_valid(body[idx]):
                y_norm = body[idx][1] / meta.height
                if y_norm > max_y:
                    max_y = y_norm
                    found = True
        return max_y if found else None

    def _is_valid(self, pt):
        if pt is None: return False
        if len(pt) > 2 and pt[2] < 0.05: return False
        return True

    def _get_hip_coords(self, meta):
        body = getattr(meta, "kps_body", None)
        if body is None: return {}
        hips = {}
        for idx in self.HIP_INDICES:
            if idx < len(body) and self._is_valid(body[idx]):
                hips[idx] = np.array(body[idx][:2], dtype=np.float32)
        return hips

    def _apply_offset_to_upper_body(self, meta, offset_px):
        excluded = set([9, 12] + self.FOOT_INDICES)
        if meta.kps_body is not None:
            for i in range(len(meta.kps_body)):
                if i not in excluded:
                    self._offset_point(meta.kps_body, i, offset_px)
        for arr_name in ["kps_lhand", "kps_rhand", "kps_face"]:
            arr = getattr(meta, arr_name, None)
            if arr is not None:
                for i in range(len(arr)):
                    self._offset_point(arr, i, offset_px)

    def _reconnect_legs(self, meta, hips_before, offset_px):
        body = meta.kps_body
        leg_map = {8: (9, 10), 11: (12, 13)} 
        for hip_idx, (knee_idx, ankle_idx) in leg_map.items():
            if hip_idx not in hips_before: continue
            hip_old = hips_before[hip_idx]
            hip_new = hip_old.copy()
            hip_new[1] += offset_px 
            
            if ankle_idx >= len(body) or not self._is_valid(body[ankle_idx]): continue
            ankle_pos = np.array(body[ankle_idx][:2], dtype=np.float32)
            
            if knee_idx >= len(body) or not self._is_valid(body[knee_idx]): continue
            knee_old = np.array(body[knee_idx][:2], dtype=np.float32)
            
            vec_old = ankle_pos - hip_old
            len_old = np.linalg.norm(vec_old)
            vec_new = ankle_pos - hip_new
            
            if len_old > 1e-6:
                t = np.dot(knee_old - hip_old, vec_old) / (len_old * len_old)
                ortho = (knee_old - hip_old) - t * vec_old
                knee_new = hip_new + t * vec_new + ortho
                self._set_point(body, knee_idx, knee_new[0], knee_new[1])

    def _offset_point(self, arr, idx, offset_y):
        if idx < len(arr) and arr[idx] is not None:
            if isinstance(arr[idx], np.ndarray):
                arr[idx][1] += offset_y
            elif isinstance(arr[idx], list):
                arr[idx][1] += offset_y

    def _set_point(self, arr, idx, x, y):
        if idx < len(arr) and arr[idx] is not None:
            if isinstance(arr[idx], np.ndarray):
                arr[idx][0] = x
                arr[idx][1] = y
            elif isinstance(arr[idx], list):
                arr[idx][0] = x
                arr[idx][1] = y


class PoseDataAutomaticOffsetNodeV4:
    """
    V4: Wie V3, aber mit robusterer Analyse.
    Bezieht auch 'kps_face' (Gesichts-Landmarks) in die Höhenmessung ein.
    Das stellt sicher, dass der Kopf auch dann korrekt verschoben wird, 
    wenn die Body-Keypoints für den Kopf fehlen oder ungenau sind.
    """
    
    HEAD_INDICES = [0, 1, 2, 3, 4, 5] 
    HIP_INDICES = [8, 11]
    FOOT_INDICES = [10, 13, 18, 19, 20, 21, 22, 23, 24] 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "source_height": (
                    "FLOAT",
                    {
                        "default": 1.70,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.01,
                        "tooltip": "Die aktuelle/echte Größe der Person (z.B. 1.70m).",
                    },
                ),
                "target_height": (
                    "FLOAT",
                    {
                        "default": 2.20,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.01,
                        "tooltip": "Die Wunschgröße (z.B. 2.20m). Das Verhältnis Target/Source bestimmt die Streckung.",
                    },
                ),
                "analysis_duration": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Dauer der Analyse, um die aktuelle Größe im Bild zu messen.",
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 240,
                        "step": 1,
                        "tooltip": "Framerate.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "V4: Proportional scaling with Face-Aware Analysis. Ensures head moves correctly."

    def process(self, pose_data, source_height, target_height, analysis_duration, fps):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        # 1. Analyse Phase: Wie groß ist die Person im Bild aktuell?
        fps = max(1, int(fps))
        analysis_frames = max(1, int(analysis_duration * fps))
        limit_frames = min(len(pose_metas), analysis_frames)
        
        head_y_samples = []
        foot_y_samples = []
        
        for i in range(limit_frames):
            meta = pose_metas[i]
            top_y = self._find_pose_top(meta)
            bot_y = self._find_pose_bottom(meta)
            
            if top_y is not None and bot_y is not None:
                head_y_samples.append(top_y)
                foot_y_samples.append(bot_y)
        
        if not head_y_samples:
            # Fallback: Wenn Analyse fehlschlägt, keine Änderung
            return (pose_data_copy,)

        avg_head_y = float(np.median(head_y_samples))
        avg_foot_y = float(np.median(foot_y_samples))
        
        # Die aktuelle visuelle Größe (z.B. 0.5 der Bildhöhe)
        current_visual_height = avg_foot_y - avg_head_y
        
        if current_visual_height <= 0.001:
            return (pose_data_copy,)

        # 2. Berechnung des Skalierungsfaktors
        scale_ratio = target_height / source_height
        
        # Die neue Wunsch-Größe im Bild
        target_visual_height = current_visual_height * scale_ratio
        
        # Wo muss der Kopf hin? (Füße bleiben bei avg_foot_y)
        target_head_y = avg_foot_y - target_visual_height
        
        # Offset berechnen: Wohin muss der Oberkörper geschoben werden?
        fixed_offset_y_norm = target_head_y - avg_head_y

        # 3. Anwendung
        for meta in pose_metas:
            height = getattr(meta, "height", 1.0) or 1.0
            
            offset_px = fixed_offset_y_norm * height
            
            # Alte Hüftpositionen sichern
            hip_coords_before = self._get_hip_coords(meta)
            
            # A. Oberkörper verschieben
            self._apply_offset_to_upper_body(meta, offset_px)
            
            # B. Beine anpassen (Füße bleiben fix, Knie interpolieren)
            if hip_coords_before:
                self._reconnect_legs(meta, hip_coords_before, offset_px)

        return (pose_data_copy,)

    # --- Verbesserte Helper Methoden ---

    def _find_pose_top(self, meta):
        # 1. Check Body Keypoints (Nose, Eyes, Ears, Neck)
        body = getattr(meta, "kps_body", None)
        min_y = float('inf')
        found = False
        
        if body is not None:
            for idx in self.HEAD_INDICES:
                if idx < len(body) and self._is_valid(body[idx]):
                    y_norm = body[idx][1] / meta.height
                    if y_norm < min_y:
                        min_y = y_norm
                        found = True
        
        # 2. Check Face Keypoints (NEU in V4: Viel genauer)
        face = getattr(meta, "kps_face", None)
        if face is not None:
            # face ist oft ein numpy array (N, 2) oder (N, 3)
            # Wir iterieren durch alle Face-Punkte, um den höchsten zu finden
            for i in range(len(face)):
                # Face points haben oft keine Score, wir nehmen sie einfach an
                pt = face[i]
                if pt is not None:
                    # Sicherstellen dass wir y (index 1) haben
                    y_val = pt[1]
                    y_norm = y_val / meta.height
                    if y_norm < min_y:
                        min_y = y_norm
                        found = True

        return min_y if found else None

    def _find_pose_bottom(self, meta):
        body = getattr(meta, "kps_body", None)
        if body is None: return None
        max_y = float('-inf')
        found = False
        for idx in self.FOOT_INDICES:
            if idx < len(body) and self._is_valid(body[idx]):
                y_norm = body[idx][1] / meta.height
                if y_norm > max_y:
                    max_y = y_norm
                    found = True
        return max_y if found else None

    def _is_valid(self, pt):
        if pt is None: return False
        if len(pt) > 2 and pt[2] < 0.05: return False
        return True

    def _get_hip_coords(self, meta):
        body = getattr(meta, "kps_body", None)
        if body is None: return {}
        hips = {}
        for idx in self.HIP_INDICES:
            if idx < len(body) and self._is_valid(body[idx]):
                hips[idx] = np.array(body[idx][:2], dtype=np.float32)
        return hips

    def _apply_offset_to_upper_body(self, meta, offset_px):
        """Verschiebt alles AUẞER Knie und Füße."""
        excluded = set([9, 12] + self.FOOT_INDICES)
        
        # Body: Alles verschieben außer Excluded
        if meta.kps_body is not None:
            for i in range(len(meta.kps_body)):
                if i not in excluded:
                    self._offset_point(meta.kps_body, i, offset_px)
        
        # Hands & Face: IMMER mitverschieben
        for arr_name in ["kps_lhand", "kps_rhand", "kps_face"]:
            arr = getattr(meta, arr_name, None)
            if arr is not None:
                for i in range(len(arr)):
                    self._offset_point(arr, i, offset_px)

    def _reconnect_legs(self, meta, hips_before, offset_px):
        body = meta.kps_body
        leg_map = {8: (9, 10), 11: (12, 13)} 
        for hip_idx, (knee_idx, ankle_idx) in leg_map.items():
            if hip_idx not in hips_before: continue
            
            hip_old = hips_before[hip_idx]
            hip_new = hip_old.copy()
            hip_new[1] += offset_px 
            
            if ankle_idx >= len(body) or not self._is_valid(body[ankle_idx]): continue
            ankle_pos = np.array(body[ankle_idx][:2], dtype=np.float32)
            
            if knee_idx >= len(body) or not self._is_valid(body[knee_idx]): continue
            knee_old = np.array(body[knee_idx][:2], dtype=np.float32)
            
            vec_old = ankle_pos - hip_old
            len_old = np.linalg.norm(vec_old)
            vec_new = ankle_pos - hip_new
            
            if len_old > 1e-6:
                # Knie wird projiziert, um die Beinlänge zu erhalten
                t = np.dot(knee_old - hip_old, vec_old) / (len_old * len_old)
                ortho = (knee_old - hip_old) - t * vec_old
                knee_new = hip_new + t * vec_new + ortho
                self._set_point(body, knee_idx, knee_new[0], knee_new[1])

    def _offset_point(self, arr, idx, offset_y):
        if idx < len(arr) and arr[idx] is not None:
            if isinstance(arr[idx], np.ndarray):
                arr[idx][1] += offset_y
            elif isinstance(arr[idx], list):
                arr[idx][1] += offset_y

    def _set_point(self, arr, idx, x, y):
        if idx < len(arr) and arr[idx] is not None:
            if isinstance(arr[idx], np.ndarray):
                arr[idx][0] = x
                arr[idx][1] = y
            elif isinstance(arr[idx], list):
                arr[idx][0] = x
                arr[idx][1] = y

class PoseDataEditorCutter:
    SCORE_THRESHOLD = 0.05

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "padding_left": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.01,
                        "tooltip": "Extra space to keep on the left side of the cropped canvas (pixels unless normalised).",
                    },
                ),
                "padding_right": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.01,
                        "tooltip": "Extra space to keep on the right side of the cropped canvas (pixels unless normalised).",
                    },
                ),
                "padding_top": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.01,
                        "tooltip": "Extra space to keep above the pose in the cropped canvas (pixels unless normalised).",
                    },
                ),
                "padding_bottom": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.01,
                        "tooltip": "Extra space to keep below the pose in the cropped canvas (pixels unless normalised).",
                    },
                ),
                "padding_normalized": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Interpret padding values as 0-1 ratios of the image dimensions instead of pixels.",
                    },
                ),
                "min_crop_width": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 8192.0,
                        "step": 1.0,
                        "tooltip": "Minimum crop width in pixels. Set to 0 to disable the width constraint.",
                    },
                ),
                "min_crop_height": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 8192.0,
                        "step": 1.0,
                        "tooltip": "Minimum crop height in pixels. Set to 0 to disable the height constraint.",
                    },
                ),
                "crop_expand": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Uniformly scale the padded bounding box before clamping (1.0 keeps the original size).",
                    },
                ),
                "max_crop_expand": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Maximum allowed expansion factor relative to the detected content (0 disables the limit).",
                    },
                ),
                "keep_aspect_ratio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Deprecated: use 'preserve_aspect_ratio' to maintain the original canvas aspect ratio.",
                    },
                ),
                "preserve_aspect_ratio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Expand the crop so it preserves the original canvas aspect ratio when possible.",
                    },
                ),
                "start_at_canvas_bottom": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Anchor the crop to the canvas bottom so it grows upward.",
                    },
                ),
                "analyze_start_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": "Time offset before the automatic cutter begins analysing pose extents.",
                    },
                ),
                "analyze_stop_seconds_reversed": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": "Duration counted from the clip end after which analysis stops.",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.0,
                        "max": 240.0,
                        "step": 0.1,
                        "tooltip": "Frame rate used to convert analyse start seconds into frames.",
                    },
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "default": None,
                        "tooltip": "Optional image frames to crop alongside the pose data.",
                    },
                ),
                "min_crop_size": (
                    "VEC2",
                    {
                        "default": [0.0, 0.0],
                        "min": 0.0,
                        "max": 8192.0,
                        "step": 1.0,
                        "tooltip": "Legacy combined minimum crop size. Overrides the width/height fields when provided.",
                    },
                ),
                "bbox": (
                    "BBOX",
                    {
                        "default": None,
                        "tooltip": "Optional bounding box override (x, y, width, height) to merge with pose-detected bounds.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE")
    RETURN_NAMES = ("pose_data", "images")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Crops pose data and images to the largest detected pose region with optional per-side padding."

    def process(
        self,
        pose_data,
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_normalized,
        min_crop_width,
        min_crop_height,
        crop_expand,
        max_crop_expand,
        keep_aspect_ratio,
        preserve_aspect_ratio,
        start_at_canvas_bottom,
        analyze_start_seconds,
        analyze_stop_seconds_reversed,
        fps,
        images=None,
        min_crop_size=None,
        bbox=None,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy, images)

        analyze_start_seconds = max(0.0, float(analyze_start_seconds))
        analyze_stop_seconds_reversed = max(0.0, float(analyze_stop_seconds_reversed))
        fps = max(0.0, float(fps))
        total_frames = len(pose_metas)

        start_frame = int(analyze_start_seconds * fps) if fps > 0.0 else 0
        start_frame = max(0, start_frame)
        if total_frames:
            start_frame = min(start_frame, total_frames - 1)
        else:
            start_frame = 0

        stop_offset_frames = int(analyze_stop_seconds_reversed * fps) if fps > 0.0 else 0
        stop_offset_frames = max(0, stop_offset_frames)
        stop_frame = total_frames - stop_offset_frames
        stop_frame = max(0, min(total_frames, stop_frame))

        if total_frames and stop_frame <= start_frame:
            if stop_frame < total_frames:
                stop_frame = min(total_frames, start_frame + 1)
            else:
                start_frame = max(0, stop_frame - 1)

        images_np = None
        images_device = None
        images_dtype = None
        has_images = False

        if isinstance(images, torch.Tensor):
            if images.numel() > 0:
                images_np = images.detach().cpu().numpy()
                images_device = images.device
                images_dtype = images.dtype
                has_images = True
        elif images is not None:
            images_np = np.asarray(images)
            if images_np.size > 0:
                has_images = True
            else:
                images_np = None

        reference_meta = pose_metas[0]
        width = getattr(reference_meta, "width", 0)
        height = getattr(reference_meta, "height", 0)

        if width in (None, 0) or height in (None, 0):
            return (pose_data_copy, images)

        min_crop_w = float(min_crop_width)
        min_crop_h = float(min_crop_height)
        if min_crop_size is not None:
            legacy_w, legacy_h = self._vec2_to_pair(min_crop_size)
            if any(not math.isclose(val, 0.0, abs_tol=1e-6) for val in (legacy_w, legacy_h)):
                min_crop_w = float(legacy_w)
                min_crop_h = float(legacy_h)

        bbox_override = self._normalize_bbox_override(bbox, width, height)

        crop_bounds = self._determine_crop_bounds(
            pose_metas,
            width,
            height,
            padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            padding_normalized,
            min_crop_w,
            min_crop_h,
            crop_expand,
            max_crop_expand,
            bool(preserve_aspect_ratio or keep_aspect_ratio),
            bool(start_at_canvas_bottom),
            start_frame,
            stop_frame,
            bbox_override,
        )

        if crop_bounds is None:
            return (pose_data_copy, images)

        x0, y0, x1, y1 = crop_bounds

        if x1 <= x0 or y1 <= y0:
            return (pose_data_copy, images)

        new_width = x1 - x0
        new_height = y1 - y0

        for meta in pose_metas:
            self._offset_aapose_meta(meta, x0, y0, new_width, new_height)

        refer_meta = pose_data_copy.get("refer_pose_meta")
        if isinstance(refer_meta, AAPoseMeta):
            self._offset_aapose_meta(refer_meta, x0, y0, new_width, new_height)

        original_metas = pose_data_copy.get("pose_metas_original", [])
        for original in original_metas or []:
            self._offset_original_meta(original, x0, y0, new_width, new_height)

        cropped_tensor = images
        if has_images and images_np is not None:
            cropped_np = images_np[:, y0:y1, x0:x1, ...]
            if cropped_np.size == 0:
                return (pose_data_copy, images)

            cropped_tensor = torch.from_numpy(cropped_np)
            if images_dtype is not None or images_device is not None:
                cropped_tensor = cropped_tensor.to(device=images_device or torch.device("cpu"))
                if images_dtype is not None:
                    cropped_tensor = cropped_tensor.to(dtype=images_dtype)

        cutter_metadata = pose_data_copy.get("cutter_metadata")
        if not isinstance(cutter_metadata, dict):
            cutter_metadata = {}
            pose_data_copy["cutter_metadata"] = cutter_metadata

        metadata_update = {
            "preserve_aspect_ratio": bool(preserve_aspect_ratio or keep_aspect_ratio),
            "start_at_canvas_bottom": bool(start_at_canvas_bottom),
            "analyze_start_seconds": float(analyze_start_seconds),
            "analyze_stop_seconds_reversed": float(analyze_stop_seconds_reversed),
            "fps": float(fps),
            "min_crop_width": float(min_crop_w),
            "min_crop_height": float(min_crop_h),
            "min_crop_size": [float(min_crop_w), float(min_crop_h)],
            "crop_expand": float(crop_expand),
            "max_crop_expand": float(max_crop_expand),
            "bounding_box": [int(x0), int(y0), int(new_width), int(new_height)],
        }

        if bbox_override is not None:
            metadata_update["bbox_override"] = [
                float(bbox_override[0]),
                float(bbox_override[1]),
                float(bbox_override[2]),
                float(bbox_override[3]),
            ]

        cutter_metadata.update(metadata_update)

        cutter_metadata = pose_data_copy.get("cutter_metadata")
        if not isinstance(cutter_metadata, dict):
            cutter_metadata = {}
            pose_data_copy["cutter_metadata"] = cutter_metadata

        cutter_metadata.update(
            {
                "preserve_aspect_ratio": bool(preserve_aspect_ratio or keep_aspect_ratio),
                "start_at_canvas_bottom": bool(start_at_canvas_bottom),
                "analyze_start_seconds": float(analyze_start_seconds),
                "analyze_stop_seconds_reversed": float(analyze_stop_seconds_reversed),
                "fps": float(fps),
                "min_crop_width": float(min_crop_w),
                "min_crop_height": float(min_crop_h),
                "min_crop_size": [float(min_crop_w), float(min_crop_h)],
                "crop_expand": float(crop_expand),
                "max_crop_expand": float(max_crop_expand),
                "bounding_box": [int(x0), int(y0), int(new_width), int(new_height)],
            }
        )

        return (pose_data_copy, cropped_tensor)

    def _determine_crop_bounds(
        self,
        pose_metas,
        width,
        height,
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_normalized,
        min_crop_width,
        min_crop_height,
        crop_expand,
        max_crop_expand,
        preserve_aspect_ratio,
        start_at_canvas_bottom,
        start_frame,
        stop_frame,
        bbox_override=None,
    ):
        largest_bbox = None
        largest_area = -1.0

        if bbox_override is not None:
            x0, y0, x1, y1 = bbox_override
            span_x = max(0.0, x1 - x0)
            span_y = max(0.0, y1 - y0)
            largest_bbox = (x0, y0, x1, y1)
            largest_area = span_x * span_y

        for index, meta in enumerate(pose_metas):
            if index < start_frame:
                continue
            if index >= stop_frame:
                break
            bbox = self._compute_bbox(meta)
            if bbox is None:
                continue

            x0, y0, x1, y1 = bbox
            span_x = max(0.0, x1 - x0)
            span_y = max(0.0, y1 - y0)
            area = span_x * span_y

            if area > largest_area:
                largest_area = area
                largest_bbox = (x0, y0, x1, y1)

        if bbox_override is not None and largest_bbox is not None:
            bx0, by0, bx1, by1 = bbox_override
            ox0, oy0, ox1, oy1 = largest_bbox
            largest_bbox = (
                min(bx0, ox0),
                min(by0, oy0),
                max(bx1, ox1),
                max(by1, oy1),
            )

        if largest_bbox is None:
            return None

        content_left = float(largest_bbox[0])
        content_top = float(largest_bbox[1])
        content_right = float(largest_bbox[2])
        content_bottom = float(largest_bbox[3])

        content_width = max(0.0, content_right - content_left)
        content_height = max(0.0, content_bottom - content_top)

        pad_left_px = self._resolve_padding(padding_left, padding_normalized, width)
        pad_right_px = self._resolve_padding(padding_right, padding_normalized, width)
        pad_top_px = self._resolve_padding(padding_top, padding_normalized, height)
        pad_bottom_px = self._resolve_padding(padding_bottom, padding_normalized, height)

        x0 = content_left - pad_left_px
        y0 = content_top - pad_top_px
        x1 = content_right + pad_right_px
        y1 = content_bottom + pad_bottom_px

        crop_expand = max(0.0, float(crop_expand))
        if crop_expand <= 0.0:
            crop_expand = 1.0
        if not math.isclose(crop_expand, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            center_x = (x0 + x1) * 0.5
            center_y = (y0 + y1) * 0.5
            half_w = (x1 - x0) * 0.5 * crop_expand
            half_h = (y1 - y0) * 0.5 * crop_expand
            x0 = center_x - half_w
            x1 = center_x + half_w
            y0 = center_y - half_h
            y1 = center_y + half_h

        center_x = (x0 + x1) * 0.5
        center_y = (y0 + y1) * 0.5
        current_width = x1 - x0
        current_height = y1 - y0

        min_width = float(min_crop_width)
        min_height = float(min_crop_height)

        if min_width > 0.0 and current_width < min_width:
            half = min_width * 0.5
            x0 = center_x - half
            x1 = center_x + half
            current_width = x1 - x0
            center_x = (x0 + x1) * 0.5

        if min_height > 0.0 and current_height < min_height:
            half = min_height * 0.5
            y0 = center_y - half
            y1 = center_y + half
            current_height = y1 - y0
            center_y = (y0 + y1) * 0.5

        max_expand = float(max_crop_expand)
        if max_expand > 0.0:
            center_x = (x0 + x1) * 0.5
            center_y = (y0 + y1) * 0.5
            if content_width > 0.0:
                allowed_width = content_width * max_expand
                if current_width > allowed_width:
                    half = allowed_width * 0.5
                    x0 = center_x - half
                    x1 = center_x + half
                    current_width = x1 - x0
            if content_height > 0.0:
                allowed_height = content_height * max_expand
                if current_height > allowed_height:
                    half = allowed_height * 0.5
                    y0 = center_y - half
                    y1 = center_y + half
                    current_height = y1 - y0

        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = min(float(width), x1)
        y1 = min(float(height), y1)

        if preserve_aspect_ratio and width > 0 and height > 0:
            target_ratio = float(width) / float(height)
            if target_ratio > 0.0:
                x0, y0, x1, y1 = self._expand_bounds_to_aspect_ratio(
                    x0, y0, x1, y1, float(width), float(height), target_ratio
                )

        x0 = int(max(0.0, math.floor(x0)))
        y0 = int(max(0.0, math.floor(y0)))
        x1 = int(min(float(width), math.ceil(x1)))
        y1 = int(min(float(height), math.ceil(y1)))

        if start_at_canvas_bottom:
            desired_height = y1 - y0
            if desired_height <= 0:
                return None
            y1 = int(height)
            y0 = max(0, y1 - desired_height)

        if x1 <= x0 or y1 <= y0:
            return None

        return (x0, y0, x1, y1)

    def _expand_bounds_to_aspect_ratio(
        self, x0, y0, x1, y1, canvas_width, canvas_height, target_ratio
    ):
        crop_width = x1 - x0
        crop_height = y1 - y0

        if (
            crop_width <= 0.0
            or crop_height <= 0.0
            or target_ratio <= 0.0
            or canvas_width <= 0.0
            or canvas_height <= 0.0
        ):
            return x0, y0, x1, y1

        current_ratio = crop_width / crop_height
        if abs(current_ratio - target_ratio) <= 1e-6:
            return x0, y0, x1, y1

        center_x = (x0 + x1) * 0.5
        center_y = (y0 + y1) * 0.5

        if current_ratio > target_ratio:
            new_width = crop_width
            new_height = crop_width / target_ratio
        else:
            new_height = crop_height
            new_width = crop_height * target_ratio

        max_width = float(canvas_width)
        max_height = float(canvas_height)

        new_width = min(new_width, max_width)
        new_height = min(new_height, max_height)

        new_x0 = center_x - new_width * 0.5
        new_y0 = center_y - new_height * 0.5

        max_x0 = max_width - new_width
        max_y0 = max_height - new_height

        if max_x0 < 0.0:
            new_x0 = 0.0
            new_x1 = max_width
        else:
            new_x0 = min(max(new_x0, 0.0), max_x0)
            new_x1 = new_x0 + new_width

        if max_y0 < 0.0:
            new_y0 = 0.0
            new_y1 = max_height
        else:
            new_y0 = min(max(new_y0, 0.0), max_y0)
            new_y1 = new_y0 + new_height

        return new_x0, new_y0, new_x1, new_y1

    def _resolve_padding(self, value, normalized, size_reference):
        if normalized:
            return float(value) * float(size_reference)
        return float(value)

    def _vec2_to_pair(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().flatten().tolist()
        elif isinstance(value, np.ndarray):
            value = value.flatten().tolist()

        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                return float(value[0]), float(value[1])
            if len(value) == 1:
                scalar = float(value[0])
                return scalar, scalar

        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return 0.0, 0.0
        return scalar, scalar

    def _normalize_bbox_override(self, bbox, width, height):
        if bbox in (None,):
            return None

        if width in (None, 0) or height in (None, 0):
            return None

        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().cpu().numpy()

        try:
            bbox_array = np.asarray(bbox, dtype=np.float32)
        except Exception:
            return None

        if bbox_array.size == 0:
            return None

        if bbox_array.ndim == 1:
            entries = [bbox_array]
        else:
            last_dim = bbox_array.shape[-1]
            if last_dim < 4:
                return None
            entries = bbox_array.reshape(-1, last_dim)

        collected = []

        for entry in entries:
            converted = self._convert_bbox_entry(entry, width, height)
            if converted is not None:
                collected.append(converted)

        if not collected:
            return None

        x0 = min(val[0] for val in collected)
        y0 = min(val[1] for val in collected)
        x1 = max(val[2] for val in collected)
        y1 = max(val[3] for val in collected)

        if x1 <= x0 or y1 <= y0:
            return None

        return (x0, y0, x1, y1)

    def _convert_bbox_entry(self, entry, width, height):
        try:
            vector = np.asarray(entry, dtype=np.float32).flatten()
        except Exception:
            return None

        if vector.size < 4:
            return None

        x0 = float(vector[0])
        y0 = float(vector[1])
        third = float(vector[2])
        fourth = float(vector[3])

        canvas_width = float(width)
        canvas_height = float(height)

        normalized = all(0.0 <= v <= 1.0 for v in (x0, y0, third, fourth)) and canvas_width > 1.0 and canvas_height > 1.0
        if normalized:
            x0 *= canvas_width
            y0 *= canvas_height
            third *= canvas_width
            fourth *= canvas_height

        x1 = x0 + third
        y1 = y0 + fourth

        invalid_xywh = third <= 0.0 or fourth <= 0.0 or x1 <= x0 or y1 <= y0
        exceeds_canvas = x1 > canvas_width * 1.5 or y1 > canvas_height * 1.5

        if invalid_xywh or exceeds_canvas:
            x1 = third
            y1 = fourth

        x0 = float(np.clip(x0, 0.0, canvas_width))
        y0 = float(np.clip(y0, 0.0, canvas_height))
        x1 = float(np.clip(x1, 0.0, canvas_width))
        y1 = float(np.clip(y1, 0.0, canvas_height))

        if x1 <= x0 or y1 <= y0:
            return None

        return (x0, y0, x1, y1)


class PoseDataEditorWithMaskCutter(PoseDataEditorCutter):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "masks": ("MASK",),
                "padding_left": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4096.0,
                        "step": 0.01,
                        "tooltip": "Extra space to keep on the left side of the combined mask (pixels unless normalised).",
                    },
                ),
                "padding_right": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4096.0,
                        "step": 0.01,
                        "tooltip": "Extra space to keep on the right side of the combined mask (pixels unless normalised).",
                    },
                ),
                "padding_top": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4096.0,
                        "step": 0.01,
                        "tooltip": "Extra space to keep above the combined mask (pixels unless normalised).",
                    },
                ),
                "padding_bottom": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4096.0,
                        "step": 0.01,
                        "tooltip": "Extra space to keep below the combined mask (pixels unless normalised).",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Interpret padding values as ratios of the mask dimensions instead of pixels.",
                    },
                ),
                "expand_mask": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.01,
                        "tooltip": "Uniform amount to expand the combined mask in every direction (pixels unless normalised).",
                    },
                ),
                "expand_normalize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Interpret the expand value as a ratio of the mask size instead of pixels.",
                    },
                ),
                "mask_to_bottom": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Extend the filled mask down to the canvas bottom before cropping.",
                    },
                ),
                "min_crop_width": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 8192.0,
                        "step": 1.0,
                        "tooltip": "Minimum crop width in pixels. Set to 0 to disable the width constraint.",
                    },
                ),
                "min_crop_height": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 8192.0,
                        "step": 1.0,
                        "tooltip": "Minimum crop height in pixels. Set to 0 to disable the height constraint.",
                    },
                ),
                "max_crop_expand": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Maximum allowed expansion factor relative to the detected mask bounds (0 disables the limit).",
                    },
                ),
                "keep_aspect_ratio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Expand the crop so it preserves the original mask aspect ratio when possible.",
                    },
                ),
                "analyze_start_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": "Time in seconds after which mask analysis begins.",
                    },
                ),
                "analyze_stop_seconds_reversed": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": "Duration from the clip end after which mask analysis stops.",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.0,
                        "max": 240.0,
                        "step": 0.1,
                        "tooltip": "Frame rate used to convert analysis seconds into frame indices.",
                    },
                ),
            },
            "optional": {
                "min_crop_size": (
                    "VEC2",
                    {
                        "default": [0.0, 0.0],
                        "min": 0.0,
                        "max": 8192.0,
                        "step": 1.0,
                        "tooltip": "Legacy combined minimum crop size. Overrides the width/height fields when provided.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "default": None,
                        "tooltip": "Optional image frames to crop alongside the pose data.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "MASK")
    RETURN_NAMES = ("pose_data", "images", "mask")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Combines multiple masks into a filled rectangle and crops pose data (and optional images) to that region."

    def process(
        self,
        pose_data,
        masks,
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        normalize,
        expand_mask,
        expand_normalize,
        mask_to_bottom,
        min_crop_width,
        min_crop_height,
        max_crop_expand,
        keep_aspect_ratio,
        analyze_start_seconds,
        analyze_stop_seconds_reversed,
        fps,
        images=None,
        min_crop_size=None,
    ):
        if isinstance(pose_data, dict):
            pose_data_copy = copy.deepcopy(pose_data)
        else:
            pose_data_copy = pose_data

        pose_metas = pose_data_copy.get("pose_metas", []) if isinstance(pose_data_copy, dict) else []

        mask_stack, mask_device, mask_dtype = self._coerce_mask_stack(masks)
        if mask_stack.size == 0:
            empty_mask = torch.zeros((1, 0, 0), dtype=torch.float32)
            if mask_device is not None:
                empty_mask = empty_mask.to(device=mask_device)
            if mask_dtype is not None:
                empty_mask = empty_mask.to(dtype=mask_dtype)
            return pose_data_copy, images, empty_mask

        frame_count = mask_stack.shape[0]
        mask_height = mask_stack.shape[1]
        mask_width = mask_stack.shape[2]

        analyze_start_seconds = max(0.0, float(analyze_start_seconds))
        analyze_stop_seconds_reversed = max(0.0, float(analyze_stop_seconds_reversed))
        fps = max(0.0, float(fps))

        start_frame = int(analyze_start_seconds * fps) if fps > 0.0 else 0
        start_frame = max(0, start_frame)
        if frame_count:
            start_frame = min(start_frame, frame_count - 1)
        else:
            start_frame = 0

        stop_offset_frames = int(analyze_stop_seconds_reversed * fps) if fps > 0.0 else 0
        stop_offset_frames = max(0, stop_offset_frames)
        stop_frame = frame_count - stop_offset_frames
        stop_frame = max(0, min(frame_count, stop_frame))

        if frame_count and stop_frame <= start_frame:
            if stop_frame < frame_count:
                stop_frame = min(frame_count, start_frame + 1)
            else:
                start_frame = max(0, stop_frame - 1)

        analysis_stack = mask_stack
        if frame_count:
            analysis_stack = mask_stack[start_frame:stop_frame]
            if analysis_stack.shape[0] == 0:
                analysis_stack = mask_stack[start_frame : start_frame + 1]

        combined_mask = self._combine_masks(analysis_stack)
        bbox = self._mask_bounding_box(combined_mask)

        if bbox is None:
            zero_mask = np.zeros((mask_height, mask_width), dtype=np.float32)
            filled_mask_tensor = self._to_mask_tensor(zero_mask, mask_device, mask_dtype)
            return pose_data_copy, images, filled_mask_tensor

        content_x0, content_y0, content_x1, content_y1 = bbox

        pad_left = self._resolve_padding(padding_left, normalize, mask_width)
        pad_right = self._resolve_padding(padding_right, normalize, mask_width)
        pad_top = self._resolve_padding(padding_top, normalize, mask_height)
        pad_bottom = self._resolve_padding(padding_bottom, normalize, mask_height)

        expand_x = self._resolve_padding(expand_mask, expand_normalize, mask_width)
        expand_y = self._resolve_padding(expand_mask, expand_normalize, mask_height)

        x0 = content_x0 - pad_left - expand_x
        x1 = content_x1 + pad_right + expand_x
        y0 = content_y0 - pad_top - expand_y
        y1 = content_y1 + pad_bottom + expand_y

        min_crop_w = float(min_crop_width)
        min_crop_h = float(min_crop_height)
        if min_crop_size is not None:
            legacy_w, legacy_h = self._vec2_to_pair(min_crop_size)
            if any(not math.isclose(val, 0.0, abs_tol=1e-6) for val in (legacy_w, legacy_h)):
                min_crop_w = float(legacy_w)
                min_crop_h = float(legacy_h)

        current_width = x1 - x0
        current_height = y1 - y0

        if min_crop_w > 0.0 and current_width < min_crop_w:
            deficit = (min_crop_w - current_width) * 0.5
            x0 -= deficit
            x1 += deficit
            current_width = x1 - x0

        if min_crop_h > 0.0 and current_height < min_crop_h:
            deficit = (min_crop_h - current_height) * 0.5
            y0 -= deficit
            y1 += deficit
            current_height = y1 - y0

        max_expand = float(max_crop_expand)
        content_width = content_x1 - content_x0
        content_height = content_y1 - content_y0

        if max_expand > 0.0:
            center_x = (x0 + x1) * 0.5
            center_y = (y0 + y1) * 0.5

            if content_width > 0.0:
                allowed_width = content_width * max_expand
                if current_width > allowed_width:
                    half = allowed_width * 0.5
                    x0 = center_x - half
                    x1 = center_x + half
                    current_width = x1 - x0

            if content_height > 0.0:
                allowed_height = content_height * max_expand
                if current_height > allowed_height:
                    half = allowed_height * 0.5
                    y0 = center_y - half
                    y1 = center_y + half
                    current_height = y1 - y0

        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = min(float(mask_width), x1)
        y1 = min(float(mask_height), y1)

        if keep_aspect_ratio and mask_width > 0 and mask_height > 0:
            target_ratio = float(mask_width) / float(mask_height)
            if target_ratio > 0.0:
                x0, y0, x1, y1 = self._expand_bounds_to_aspect_ratio(
                    x0, y0, x1, y1, float(mask_width), float(mask_height), target_ratio
                )

        if mask_to_bottom:
            desired_height = y1 - y0
            y1 = float(mask_height)
            y0 = y1 - desired_height
            if y0 < 0.0:
                y0 = 0.0

        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = min(float(mask_width), x1)
        y1 = min(float(mask_height), y1)

        if x1 <= x0 or y1 <= y0:
            zero_mask = np.zeros((mask_height, mask_width), dtype=np.float32)
            filled_mask_tensor = self._to_mask_tensor(zero_mask, mask_device, mask_dtype)
            return pose_data_copy, images, filled_mask_tensor

        x0_int = int(max(0, math.floor(x0)))
        y0_int = int(max(0, math.floor(y0)))
        x1_int = int(min(mask_width, math.ceil(x1)))
        y1_int = int(min(mask_height, math.ceil(y1)))

        if x1_int <= x0_int or y1_int <= y0_int:
            zero_mask = np.zeros((mask_height, mask_width), dtype=np.float32)
            filled_mask_tensor = self._to_mask_tensor(zero_mask, mask_device, mask_dtype)
            return pose_data_copy, images, filled_mask_tensor

        new_width = x1_int - x0_int
        new_height = y1_int - y0_int

        final_mask = np.zeros((mask_height, mask_width), dtype=np.float32)
        final_mask[y0_int:y1_int, x0_int:x1_int] = 1.0
        cropped_mask = final_mask[y0_int:y1_int, x0_int:x1_int]
        mask_tensor = self._to_mask_tensor(cropped_mask, mask_device, mask_dtype)

        images_result = images
        images_np = None
        images_device = None
        images_dtype = None
        images_available = False

        if isinstance(images, torch.Tensor):
            images_np = images.detach().cpu().numpy()
            images_device = images.device
            images_dtype = images.dtype
            images_available = images_np.size > 0
        elif isinstance(images, np.ndarray):
            images_np = images
            images_available = images_np.size > 0
        elif images is not None:
            try:
                images_np = np.asarray(images)
                images_available = images_np.size > 0
            except Exception:
                images_np = None

        if images_available and images_np is not None:
            cropped_np = images_np[:, y0_int:y1_int, x0_int:x1_int, ...]
            if isinstance(images, torch.Tensor):
                images_result = torch.from_numpy(cropped_np).to(device=images_device, dtype=images_dtype)
            else:
                images_result = torch.from_numpy(cropped_np)
        elif isinstance(images, torch.Tensor):
            images_result = images
        elif images is None:
            images_result = torch.zeros((0, 0, 0, 3), dtype=torch.float32)

        if pose_metas:
            for meta in pose_metas:
                self._offset_aapose_meta(meta, x0_int, y0_int, new_width, new_height)

            if isinstance(pose_data_copy, dict):
                refer_meta = pose_data_copy.get("refer_pose_meta")
                if isinstance(refer_meta, AAPoseMeta):
                    self._offset_aapose_meta(refer_meta, x0_int, y0_int, new_width, new_height)

                original_metas = pose_data_copy.get("pose_metas_original", [])
                for original in original_metas or []:
                    self._offset_original_meta(original, x0_int, y0_int, new_width, new_height)

        if isinstance(pose_data_copy, dict):
            metadata = pose_data_copy.get("mask_cutter_metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                pose_data_copy["mask_cutter_metadata"] = metadata

            metadata.update(
                {
                    "normalize": bool(normalize),
                    "expand_mask": float(expand_mask),
                    "expand_normalize": bool(expand_normalize),
                    "mask_to_bottom": bool(mask_to_bottom),
                    "min_crop_width": float(min_crop_w),
                    "min_crop_height": float(min_crop_h),
                    "min_crop_size": [float(min_crop_w), float(min_crop_h)],
                    "max_crop_expand": float(max_crop_expand),
                    "keep_aspect_ratio": bool(keep_aspect_ratio),
                    "analyze_start_seconds": float(analyze_start_seconds),
                    "analyze_stop_seconds_reversed": float(analyze_stop_seconds_reversed),
                    "fps": float(fps),
                    "analysis_start_frame": int(start_frame),
                    "analysis_stop_frame": int(stop_frame),
                    "padding": {
                        "left": float(padding_left),
                        "right": float(padding_right),
                        "top": float(padding_top),
                        "bottom": float(padding_bottom),
                    },
                    "bounding_box": [int(x0_int), int(y0_int), int(new_width), int(new_height)],
                }
            )

        return pose_data_copy, images_result, mask_tensor

    def _coerce_mask_stack(self, masks):
        mask_device = None
        mask_dtype = None

        if isinstance(masks, torch.Tensor):
            mask_device = masks.device
            mask_dtype = masks.dtype
            mask_stack = masks.detach().cpu().numpy()
        else:
            mask_stack = np.asarray(masks)

        mask_stack = np.asarray(mask_stack)
        if mask_stack.ndim == 0:
            return np.array([], dtype=np.float32), mask_device, mask_dtype

        if mask_stack.ndim == 2:
            mask_stack = mask_stack[None, ...]
        elif mask_stack.ndim == 4:
            mask_stack = mask_stack[..., 0]
        elif mask_stack.ndim > 4:
            mask_stack = mask_stack.reshape(mask_stack.shape[0], mask_stack.shape[-2], mask_stack.shape[-1])

        mask_stack = mask_stack.astype(np.float32)
        return mask_stack, mask_device, mask_dtype

    def _combine_masks(self, mask_stack):
        if mask_stack.ndim == 2:
            combined = mask_stack
        else:
            combined = np.max(mask_stack, axis=0)
        return (combined > 0.0).astype(np.float32)

    def _mask_bounding_box(self, combined_mask):
        active = combined_mask > 0.0
        if not np.any(active):
            return None

        rows = np.where(np.any(active, axis=1))[0]
        cols = np.where(np.any(active, axis=0))[0]

        if rows.size == 0 or cols.size == 0:
            return None

        y0 = float(rows[0])
        y1 = float(rows[-1] + 1)
        x0 = float(cols[0])
        x1 = float(cols[-1] + 1)
        return x0, y0, x1, y1

    def _to_mask_tensor(self, mask, device, dtype):
        if mask.ndim == 2:
            mask = mask[None, ...]
        tensor = torch.from_numpy(mask.astype(np.float32))
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor

    def _compute_bbox(self, meta):
        keypoint_sets = []

        for coords_attr, score_attr in (
            ("kps_body", "kps_body_p"),
            ("kps_lhand", "kps_lhand_p"),
            ("kps_rhand", "kps_rhand_p"),
            ("kps_face", "kps_face_p"),
        ):
            coords = getattr(meta, coords_attr, None)
            scores = getattr(meta, score_attr, None)

            if coords is None or scores is None:
                continue

            coords = np.asarray(coords, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)

            if coords.size == 0 or scores.size == 0:
                continue

            visible = scores > self.SCORE_THRESHOLD
            if not np.any(visible):
                continue

            keypoint_sets.append(coords[visible, :2])

        if not keypoint_sets:
            return None

        stacked = np.concatenate(keypoint_sets, axis=0)
        x0 = float(np.min(stacked[:, 0]))
        y0 = float(np.min(stacked[:, 1]))
        x1 = float(np.max(stacked[:, 0]))
        y1 = float(np.max(stacked[:, 1]))

        return (x0, y0, x1, y1)

    def _offset_aapose_meta(self, meta, offset_x, offset_y, new_width, new_height):
        if meta is None:
            return

        for attr in ("kps_body", "kps_lhand", "kps_rhand", "kps_face"):
            coords = getattr(meta, attr, None)
            if coords is None:
                continue

            coords[:, 0] -= offset_x
            coords[:, 1] -= offset_y
            coords[:, 0] = np.clip(coords[:, 0], 0.0, float(new_width))
            coords[:, 1] = np.clip(coords[:, 1], 0.0, float(new_height))

        if hasattr(meta, "width"):
            meta.width = new_width
        if hasattr(meta, "height"):
            meta.height = new_height

    def _offset_original_meta(self, meta_dict, offset_x, offset_y, new_width, new_height):
        if not isinstance(meta_dict, dict):
            return

        original_width = meta_dict.get("width")
        original_height = meta_dict.get("height")

        if original_width in (None, 0) or original_height in (None, 0):
            return

        for key in (
            "keypoints_body",
            "keypoints_left_hand",
            "keypoints_right_hand",
            "keypoints_face",
        ):
            points = meta_dict.get(key)
            if points is None:
                continue

            points_np = np.asarray(points, dtype=np.float32)
            if points_np.ndim != 2 or points_np.shape[1] < 2:
                continue

            coords = points_np[:, :2] * np.array([original_width, original_height], dtype=np.float32)
            coords[:, 0] -= offset_x
            coords[:, 1] -= offset_y
            coords[:, 0] = np.clip(coords[:, 0], 0.0, float(new_width))
            coords[:, 1] = np.clip(coords[:, 1], 0.0, float(new_height))

            if new_width > 0 and new_height > 0:
                points_np[:, 0] = coords[:, 0] / float(new_width)
                points_np[:, 1] = coords[:, 1] / float(new_height)

            meta_dict[key] = points_np

        meta_dict["width"] = new_width
        meta_dict["height"] = new_height




class DrawViTPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
                "retarget_padding": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "When > 0, the retargeted pose image is padded and resized to the target size"}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks. Set to 0 to disable body drawing, -1 for auto"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks. Set to 0 to disable hand drawing, -1 for auto"}),
                "draw_head": ("BOOLEAN", {"default": "True", "tooltip": "Whether to draw head keypoints"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("pose_images", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Draws pose images from pose data."

    def process(self, pose_data, width, height, body_stick_width, hand_stick_width, draw_head, retarget_padding=64):

        retarget_image = pose_data.get("retarget_image", None)
        pose_metas = pose_data["pose_metas"]

        draw_hand = hand_stick_width != 0
        use_retarget_resize = retarget_padding > 0 and retarget_image is not None

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        crop_target_image = None
        pose_images = []

        for idx, meta in enumerate(tqdm(pose_metas, desc="Drawing pose images")):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_image = draw_aapose_by_meta_new(canvas, meta, draw_hand=draw_hand, draw_head=draw_head, body_stick_width=body_stick_width, hand_stick_width=hand_stick_width)

            if crop_target_image is None:
                crop_target_image = pose_image

            if use_retarget_resize:
                pose_image = resize_to_bounds(pose_image, height, width, crop_target_image=crop_target_image, extra_padding=retarget_padding)
            else:
                pose_image = padding_resize(pose_image, height, width)

            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0

        return (pose_images_tensor, )


class PoseDataEditorKeypointDeleter:
    BODY_KEYPOINT_NAMES = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "neck",
        6: "left_shoulder",
        7: "right_shoulder",
        8: "left_elbow",
        9: "right_elbow",
        10: "left_wrist",
        11: "right_wrist",
        12: "left_hip",
        13: "right_hip",
        14: "left_knee",
        15: "right_knee",
        16: "left_ankle",
        17: "right_ankle",
        18: "left_big_toe",
        19: "left_small_toe",
        20: "left_heel",
        21: "right_big_toe",
        22: "right_small_toe",
        23: "right_heel",
        24: "hip",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "duration": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": "Seconds a keypoint may stay at the canvas border before deletion.",
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 240,
                        "step": 1,
                        "tooltip": "Frame rate used to convert the duration to frame counts.",
                    },
                ),
                "selective_delete": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, only the selected keypoint groups are monitored.",
                    },
                ),
                "target_keypoints": (
                    TARGET_OPTIONS,
                    {
                        "default": "BODY",
                        "tooltip": "When selective delete is enabled, choose which pose region to monitor (same options as Pose Data Editor).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING")
    RETURN_NAMES = ("pose_data", "log")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Removes keypoints that keep pressing against the canvas border for longer than the configured duration."

    def process(self, pose_data, duration, fps, selective_delete, target_keypoints):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas") or []
        pose_metas_original = pose_data_copy.get("pose_metas_original") or []

        frame_count = max(len(pose_metas), len(pose_metas_original))
        if frame_count == 0:
            return (pose_data_copy, "No pose frames available.")

        duration_value = max(0.0, float(duration))
        fps_value = max(1, int(fps))
        frames_required = max(1, int(math.ceil(duration_value * fps_value)))

        monitor_spec = self._build_monitor_spec(selective_delete, target_keypoints)

        counters = defaultdict(int)
        frame_deletions = [self._empty_deletion_record() for _ in range(frame_count)]
        deletion_log = []

        if pose_metas:
            self._evaluate_pose_metas(
                pose_metas,
                frames_required,
                monitor_spec,
                counters,
                frame_deletions,
                deletion_log,
            )
        else:
            self._evaluate_original_metas(
                pose_metas_original,
                frames_required,
                monitor_spec,
                counters,
                frame_deletions,
                deletion_log,
            )

        if pose_metas:
            for idx, meta in enumerate(pose_metas):
                deletions = frame_deletions[idx]
                self._apply_deletions_to_meta(meta, deletions)

        if pose_metas_original:
            for idx, entry in enumerate(pose_metas_original):
                deletions = frame_deletions[idx]
                self._apply_deletions_to_meta_dict(entry, deletions)

        pose_data_copy["pose_metas"] = pose_metas
        pose_data_copy["pose_metas_original"] = pose_metas_original

        log_output = self._format_log(deletion_log, fps_value)
        return (pose_data_copy, log_output)

    def _build_monitor_spec(self, selective_delete, target_keypoints):
        if not selective_delete:
            return {"body": None, "lhand": None, "rhand": None, "face": None}

        monitor_from_dropdown = self._build_monitor_from_dropdown(target_keypoints)
        if monitor_from_dropdown is not None:
            return monitor_from_dropdown

        tokens = []
        if isinstance(target_keypoints, str):
            parts = [part.strip().upper() for part in target_keypoints.replace("\n", ",").split(",")]
            tokens = [part for part in parts if part]

        if not tokens:
            return {"body": None, "lhand": None, "rhand": None, "face": None}

        monitor = {"body": set(), "lhand": set(), "rhand": set(), "face": set()}

        for token in tokens:
            spec = self._legacy_group_spec(token)
            if not spec:
                continue

            for key, indices in spec.items():
                if indices is None:
                    monitor[key] = None
                    continue

                if monitor[key] is None:
                    continue

                monitor[key].update(indices)

        for key, value in monitor.items():
            if isinstance(value, set) and not value:
                monitor[key] = set()

        return monitor

    def _build_monitor_from_dropdown(self, target_keypoints):
        option = (target_keypoints or "").strip().upper() if isinstance(target_keypoints, str) else ""
        if not option:
            return None

        valid_options = {opt.upper() for opt in TARGET_OPTIONS}
        if option not in valid_options:
            return None

        monitor = {"body": set(), "lhand": set(), "rhand": set(), "face": set()}

        if option == "ALL":
            return {"body": None, "lhand": None, "rhand": None, "face": None}

        if option == "BODY":
            monitor["body"] = None
            return monitor

        if option in BODY_GROUPS:
            monitor["body"].update(BODY_GROUPS[option])
            return monitor

        if option in HAND_GROUPS:
            hand_target = HAND_GROUPS[option]
            if hand_target in ("left", "both"):
                monitor["lhand"] = None
            if hand_target in ("right", "both"):
                monitor["rhand"] = None
            return monitor

        if option in FACE_GROUP:
            monitor["face"] = None
            return monitor

        return {"body": None, "lhand": None, "rhand": None, "face": None}

    def _legacy_group_spec(self, token):
        legacy_groups = {
            "KNEES": {"body": {13, 14, 15, 16}},
            "FEET": {"body": {15, 16, 17, 18, 19, 20, 21, 22, 23, 24}},
            "HEAD": {"body": {0, 1, 2, 3, 4, 5}, "face": None},
            "HANDS": {"body": {9, 10, 11}, "lhand": None, "rhand": None},
        }

        return legacy_groups.get(token)

    @staticmethod
    def _empty_deletion_record():
        return {"body": set(), "lhand": set(), "rhand": set(), "face": set()}

    def _evaluate_pose_metas(
        self,
        pose_metas,
        frames_required,
        monitor_spec,
        counters,
        frame_deletions,
        deletion_log,
    ):
        for frame_idx, meta in enumerate(pose_metas):
            if not isinstance(meta, AAPoseMeta):
                continue

            width = self._to_float(getattr(meta, "width", None))
            height = self._to_float(getattr(meta, "height", None))
            if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
                self._reset_frame_counters(monitor_spec, counters)
                continue

            arrays = (
                ("body", getattr(meta, "kps_body", None), getattr(meta, "kps_body_p", None)),
                ("lhand", getattr(meta, "kps_lhand", None), getattr(meta, "kps_lhand_p", None)),
                ("rhand", getattr(meta, "kps_rhand", None), getattr(meta, "kps_rhand_p", None)),
                ("face", getattr(meta, "kps_face", None), getattr(meta, "kps_face_p", None)),
            )

            for key_type, coords, scores in arrays:
                self._evaluate_frame_array(
                    frame_idx,
                    key_type,
                    coords,
                    scores,
                    width,
                    height,
                    frames_required,
                    monitor_spec,
                    counters,
                    frame_deletions,
                    deletion_log,
                )

    def _evaluate_original_metas(
        self,
        meta_dicts,
        frames_required,
        monitor_spec,
        counters,
        frame_deletions,
        deletion_log,
    ):
        for frame_idx, entry in enumerate(meta_dicts):
            if not isinstance(entry, dict):
                continue

            width = self._to_float(entry.get("width"))
            height = self._to_float(entry.get("height"))
            if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
                self._reset_frame_counters(monitor_spec, counters)
                continue

            arrays = (
                ("body", entry.get("keypoints_body")),
                ("lhand", entry.get("keypoints_left_hand")),
                ("rhand", entry.get("keypoints_right_hand")),
                ("face", entry.get("keypoints_face")),
            )

            for key_type, points in arrays:
                if points is None:
                    self._reset_array_counters(key_type, monitor_spec, counters)
                    continue

                points_np = np.asarray(points, dtype=np.float32)
                if points_np.ndim != 2 or points_np.shape[1] < 3:
                    self._reset_array_counters(key_type, monitor_spec, counters)
                    continue

                coords = points_np[:, :2].copy()
                coords[:, 0] *= width
                coords[:, 1] *= height

                scores = points_np[:, 2]

                self._evaluate_frame_array(
                    frame_idx,
                    key_type,
                    coords,
                    scores,
                    width,
                    height,
                    frames_required,
                    monitor_spec,
                    counters,
                    frame_deletions,
                    deletion_log,
                )

    def _evaluate_frame_array(
        self,
        frame_idx,
        key_type,
        coords,
        scores,
        width,
        height,
        frames_required,
        monitor_spec,
        counters,
        frame_deletions,
        deletion_log,
    ):
        indices = self._resolve_indices(key_type, coords, monitor_spec)
        if not indices:
            self._reset_array_counters(key_type, monitor_spec, counters)
            return

        coords_np = np.asarray(coords, dtype=np.float32)
        scores_np = None if scores is None else np.asarray(scores, dtype=np.float32)

        for idx in indices:
            key = (key_type, idx)

            if coords_np.ndim != 2 or idx >= coords_np.shape[0]:
                counters[key] = 0
                continue

            point = coords_np[idx]
            if point.shape[0] < 2 or not np.isfinite(point[:2]).all():
                counters[key] = 0
                continue

            if scores_np is not None:
                if idx >= scores_np.shape[0] or scores_np[idx] <= 0.0:
                    counters[key] = 0
                    continue

            touching = self._is_touching_border(point[0], point[1], width, height)
            if touching:
                counters[key] += 1
                if counters[key] >= frames_required:
                    frame_deletions[frame_idx][key_type].add(idx)
                    if counters[key] == frames_required:
                        deletion_log.append((frame_idx, key_type, idx))
                    counters[key] = min(counters[key], frames_required)
            else:
                counters[key] = 0

    def _apply_deletions_to_meta(self, meta, deletions):
        arrays = (
            ("body", getattr(meta, "kps_body", None), getattr(meta, "kps_body_p", None)),
            ("lhand", getattr(meta, "kps_lhand", None), getattr(meta, "kps_lhand_p", None)),
            ("rhand", getattr(meta, "kps_rhand", None), getattr(meta, "kps_rhand_p", None)),
            ("face", getattr(meta, "kps_face", None), getattr(meta, "kps_face_p", None)),
        )

        for key_type, coords, scores in arrays:
            indices = deletions.get(key_type)
            if not indices or coords is None or scores is None:
                continue

            for idx in indices:
                if idx < len(scores):
                    scores[idx] = 0.0
                if idx < len(coords) and len(coords[idx]) >= 2:
                    coords[idx][0] = 0.0
                    coords[idx][1] = 0.0

    def _apply_deletions_to_meta_dict(self, entry, deletions):
        key_map = {
            "body": "keypoints_body",
            "lhand": "keypoints_left_hand",
            "rhand": "keypoints_right_hand",
            "face": "keypoints_face",
        }

        for key_type, key_name in key_map.items():
            indices = deletions.get(key_type)
            if not indices:
                continue

            points = entry.get(key_name)
            if points is None:
                continue

            points_np = np.asarray(points, dtype=np.float32)
            if points_np.ndim != 2 or points_np.shape[1] < 3:
                continue

            for idx in indices:
                if idx < points_np.shape[0]:
                    points_np[idx, 2] = 0.0
            entry[key_name] = points_np.tolist()

    @staticmethod
    def _resolve_indices(key_type, coords, monitor_spec):
        monitor = monitor_spec.get(key_type)
        if coords is None:
            return []

        count = len(coords)
        if monitor is None:
            return list(range(count))

        return [idx for idx in monitor if idx < count]

    @staticmethod
    def _reset_frame_counters(monitor_spec, counters):
        for key_type in ("body", "lhand", "rhand", "face"):
            monitor = monitor_spec.get(key_type)
            if monitor is None:
                keys = [k for k in counters.keys() if k[0] == key_type]
                for key in keys:
                    counters[key] = 0
            else:
                for idx in monitor:
                    counters[(key_type, idx)] = 0

    @staticmethod
    def _reset_array_counters(key_type, monitor_spec, counters):
        monitor = monitor_spec.get(key_type)
        if monitor is None:
            keys = [k for k in counters.keys() if k[0] == key_type]
            for key in keys:
                counters[key] = 0
        else:
            for idx in monitor:
                counters[(key_type, idx)] = 0

    @staticmethod
    def _is_touching_border(x, y, width, height):
        if not math.isfinite(x) or not math.isfinite(y):
            return False

        epsilon = 1e-6
        if x <= 0.0 + epsilon or x >= width - epsilon:
            return True
        if y <= 0.0 + epsilon or y >= height - epsilon:
            return True
        return False

    def _format_log(self, deletion_log, fps):
        if not deletion_log:
            return "No keypoints were removed."

        lines = []
        for frame_idx, key_type, idx in deletion_log:
            timestamp = frame_idx / fps if fps > 0 else 0.0
            name = self._describe_keypoint(key_type, idx)
            lines.append(
                f"Frame {frame_idx} ({timestamp:.3f}s): removed {name} ({key_type}[{idx}])."
            )
        return "\n".join(lines)

    def _describe_keypoint(self, key_type, idx):
        if key_type == "body":
            return self.BODY_KEYPOINT_NAMES.get(idx, f"body #{idx}")
        if key_type == "lhand":
            return f"left hand #{idx}"
        if key_type == "rhand":
            return f"right hand #{idx}"
        if key_type == "face":
            return f"face #{idx}"
        return f"{key_type} #{idx}"

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")


import copy
import math
from .pose_utils.pose2d_utils import AAPoseMeta # Stellt sicher, dass AAPoseMeta importiert wird

class PoseDataEditorKneeCutter:
    # Keypoint-Indizes basierend auf dem 20-Punkte-Skelett (retarget_pose.py, human_visualization.py)
    # 9: RKnee (right_knee)
    # 12: LKnee (left_knee)
    KNEE_INDICES = (9, 12)
    BODY_KEYPOINT_NAMES = {9: "right_knee", 12: "left_knee"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "fps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 240,
                        "step": 1,
                        "tooltip": "Frame rate of the pose data sequence.",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Set to true if keypoint coordinates are normalized (0-1).",
                    },
                ),
                "padding_bottom": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Normalized padding from the bottom (0.0 = direkter Rand, 0.2 = untere 20%).",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.01,
                        "tooltip": "Duration in seconds the knee must be in the padding zone to be cut (0.0 = sofort).",
                    },
                ),
                "activation_time_seconds": ( # <--- NEUER PARAMETER
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": "Time in seconds before the cutter starts monitoring (0.0 = sofort aktiv).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING")
    RETURN_NAMES = ("pose_data", "log")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = (
        "Cuts knee keypoints if they stay in the bottom canvas padding zone for a specified duration."
    )

    def process(self, pose_data, fps, normalize, padding_bottom, duration, activation_time_seconds): # <--- PARAMETER HINZUGEFÜGT
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas") or []
        pose_metas_original = pose_data_copy.get("pose_metas_original") or []

        if not pose_metas and not pose_metas_original:
            return (pose_data_copy, "No pose frames available.")

        frames_required = max(1, int(round(duration * float(fps))))
        activation_frame = max(0, int(round(activation_time_seconds * float(fps)))) # <--- NEUE BERECHNUNG
        
        press_counters = {}
        removal_log = []
        removed_any = False

        if pose_metas:
            removed_any |= self._process_pose_metas(
                pose_metas, normalize, padding_bottom, frames_required, activation_frame, press_counters, removal_log # <--- NEUES ARGUMENT
            )

        press_counters = {}
        if pose_metas_original:
            removed_any |= self._process_original_metas(
                pose_metas_original, normalize, padding_bottom, frames_required, activation_frame, press_counters, removal_log # <--- NEUES ARGUMENT
            )

        if not removed_any and activation_frame == 0:
             removal_log.append(f"No knees entered the {padding_bottom*100}% bottom zone.")
        elif activation_frame > 0 and not removed_any:
            removal_log.append(f"Cutter activated after {activation_time_seconds}s. No knees were cut.")

        return (pose_data_copy, "\n".join(removal_log))

    def _process_pose_metas(self, pose_metas, normalize, padding_bottom, frames_required, activation_frame, press_counters, removal_log): # <--- NEUES ARGUMENT
        removed_any = False
        for frame_idx, meta in enumerate(pose_metas):
            if frame_idx < activation_frame: # <--- NEUE PRÜFUNG
                continue # Überspringen, da die Aktivierungszeit noch nicht erreicht ist
                
            if not isinstance(meta, AAPoseMeta):
                continue

            height = self._to_float(getattr(meta, "height", None))
            coords = getattr(meta, "kps_body", None)
            scores = getattr(meta, "kps_body_p", None)
            
            removed = self._prune_body_coords(
                frame_idx, coords, scores, height, normalize, padding_bottom, frames_required, press_counters, removal_log
            )
            if removed:
                removed_any = True
        return removed_any

    def _process_original_metas(self, pose_metas_original, normalize, padding_bottom, frames_required, activation_frame, press_counters, removal_log): # <--- NEUES ARGUMENT
        removed_any = False
        for frame_idx, entry in enumerate(pose_metas_original):
            if frame_idx < activation_frame: # <--- NEUE PRÜFUNG
                continue # Überspringen, da die Aktivierungszeit noch nicht erreicht ist
                
            if not isinstance(entry, dict):
                continue

            height = self._to_float(entry.get("height"))
            keypoints_body = entry.get("keypoints_body")
            if keypoints_body is None:
                continue

            points_np = np.asarray(keypoints_body, dtype=np.float32)
            if points_np.ndim != 2 or points_np.shape[1] < 3:
                continue

            removed = False
            for knee_index in self.KNEE_INDICES:
                if knee_index >= points_np.shape[0]:
                    continue
                
                key = (0, knee_index) # Annahme: Nur eine Person (person_id=0)
                current_counter = press_counters.get(key, 0)
                
                y_coord_norm = self._to_float(points_np[knee_index, 1])
                is_visible = points_np[knee_index, 2] > 0.0

                if normalize:
                    threshold = 1.0 - padding_bottom
                    y_coord_to_check = y_coord_norm
                else:
                    if not math.isfinite(height) or height <= 0.0:
                        press_counters[key] = 0
                        continue
                    threshold = height - (padding_bottom * height)
                    y_coord_to_check = y_coord_norm * height
                
                if not math.isfinite(y_coord_to_check):
                    press_counters[key] = 0
                    continue

                is_touching = (y_coord_to_check >= threshold - 1e-6)

                if is_touching and is_visible:
                    current_counter += 1
                    press_counters[key] = current_counter
                else:
                    press_counters[key] = 0 # Reset, wenn nicht mehr berührt oder unsichtbar

                if current_counter >= frames_required:
                    if points_np[knee_index, 2] > 0: # Nur loggen, wenn es gerade entfernt wurde
                        removal_log.append(
                            self._format_log_entry(frame_idx, knee_index, normalize, y_coord_to_check, frames_required)
                        )
                        removed = True
                    points_np[knee_index, 2] = 0.0
                    points_np[knee_index, 0] = 0.0
                    points_np[knee_index, 1] = 0.0
            
            if removed:
                entry["keypoints_body"] = points_np.tolist()
                removed_any = True
        return removed_any

    def _prune_body_coords(self, frame_idx, coords, scores, height, normalize, padding_bottom, frames_required, press_counters, removal_log):
        if coords is None or scores is None:
            return False

        removed_any = False
        for knee_index in self.KNEE_INDICES:
            if knee_index >= len(coords) or knee_index >= len(scores):
                continue
            
            key = (0, knee_index) # Annahme: Nur eine Person (person_id=0)
            current_counter = press_counters.get(key, 0)

            coord_pair = coords[knee_index]
            if coord_pair is None or len(coord_pair) < 2:
                press_counters[key] = 0
                continue

            y_coord_pixel = self._to_float(coord_pair[1])
            is_visible = scores[knee_index] > 0.0
            
            if normalize:
                if not math.isfinite(height) or height <= 0.0:
                    press_counters[key] = 0
                    continue
                threshold = 1.0 - padding_bottom
                y_coord_to_check = y_coord_pixel / height
            else:
                if not math.isfinite(height) or height <= 0.0:
                    press_counters[key] = 0
                    continue
                threshold = height - (padding_bottom * height)
                y_coord_to_check = y_coord_pixel

            if not math.isfinite(y_coord_to_check):
                press_counters[key] = 0
                continue

            is_touching = (y_coord_to_check >= threshold - 1e-6)

            if is_touching and is_visible:
                current_counter += 1
                press_counters[key] = current_counter
            else:
                press_counters[key] = 0 # Reset

            if current_counter >= frames_required:
                if scores[knee_index] > 0.0: # Nur loggen, wenn es gerade entfernt wurde
                    removal_log.append(
                        self._format_log_entry(frame_idx, knee_index, normalize, y_coord_to_check, frames_required)
                    )
                    removed_any = True
                scores[knee_index] = 0.0
                coord_pair[0] = 0.0
                coord_pair[1] = 0.0

        return removed_any

    def _format_log_entry(self, frame_idx, knee_index, normalize, y_value, frame_count):
        name = self.BODY_KEYPOINT_NAMES.get(knee_index, f"index_{knee_index}")
        value_type = "normalized" if normalize else "pixel"
        return (
            f"Frame {frame_idx}: removed {name} (y={y_value:.4f} {value_type}) after {frame_count} frames."
        )

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")
class PoseDataEditorHeadDeleter:
    # Keypoint-Indizes basierend auf dem 20-Punkte-Skelett (retarget_pose.py, human_visualization.py)
    # 0: Nose, 14: REye, 15: LEye, 16: REar, 17: LEar
    HEAD_INDICES = (0, 14, 15, 16, 17)
    BODY_KEYPOINT_NAMES = {
        0: "Nose",
        14: "right_eye",
        15: "left_eye",
        16: "right_ear",
        17: "left_ear",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "fps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 240,
                        "step": 1,
                        "tooltip": "Frame rate of the pose data sequence.",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Set to true if keypoint coordinates are normalized (0-1).",
                    },
                ),
                "padding_top": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Normalized padding from the top (0.0 = direkter Rand, 0.1 = obere 10%).",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.01,
                        "tooltip": "Duration in seconds the head must be in the padding zone to be cut (0.0 = sofort).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING")
    RETURN_NAMES = ("pose_data", "log")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = (
        "Cuts head keypoints if they stay in the top canvas padding zone for a specified duration."
    )

    def process(self, pose_data, fps, normalize, padding_top, duration):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas") or []
        pose_metas_original = pose_data_copy.get("pose_metas_original") or []

        if not pose_metas and not pose_metas_original:
            return (pose_data_copy, "No pose frames available.")

        frames_required = max(1, int(round(duration * float(fps))))
        press_counters = {}
        removal_log = []
        removed_any = False

        if pose_metas:
            removed_any |= self._process_pose_metas(
                pose_metas, normalize, padding_top, frames_required, press_counters, removal_log
            )

        press_counters = {}
        if pose_metas_original:
            removed_any |= self._process_original_metas(
                pose_metas_original, normalize, padding_top, frames_required, press_counters, removal_log
            )

        if not removed_any:
            removal_log.append(f"No head points entered the {padding_top*100}% top zone.")

        return (pose_data_copy, "\n".join(removal_log))

    def _process_pose_metas(self, pose_metas, normalize, padding_top, frames_required, press_counters, removal_log):
        removed_any = False
        for frame_idx, meta in enumerate(pose_metas):
            if not isinstance(meta, AAPoseMeta):
                continue
            height = self._to_float(getattr(meta, "height", None))
            coords = getattr(meta, "kps_body", None)
            scores = getattr(meta, "kps_body_p", None)
            
            removed = self._prune_body_coords(
                frame_idx, coords, scores, height, normalize, padding_top, frames_required, press_counters, removal_log
            )
            if removed:
                removed_any = True
        return removed_any

    def _process_original_metas(self, pose_metas_original, normalize, padding_top, frames_required, press_counters, removal_log):
        removed_any = False
        for frame_idx, entry in enumerate(pose_metas_original):
            if not isinstance(entry, dict):
                continue
            height = self._to_float(entry.get("height"))
            keypoints_body = entry.get("keypoints_body")
            if keypoints_body is None:
                continue
            points_np = np.asarray(keypoints_body, dtype=np.float32)
            if points_np.ndim != 2 or points_np.shape[1] < 3:
                continue

            removed = False
            for head_index in self.HEAD_INDICES:
                if head_index >= points_np.shape[0]:
                    continue
                
                key = (0, head_index)
                current_counter = press_counters.get(key, 0)
                y_coord_norm = self._to_float(points_np[head_index, 1])
                is_visible = points_np[head_index, 2] > 0.0

                if normalize:
                    threshold = padding_top
                    y_coord_to_check = y_coord_norm
                else:
                    if not math.isfinite(height) or height <= 0.0:
                        press_counters[key] = 0
                        continue
                    threshold = padding_top * height
                    y_coord_to_check = y_coord_norm * height
                
                if not math.isfinite(y_coord_to_check):
                    press_counters[key] = 0
                    continue

                is_touching = (y_coord_to_check <= threshold + 1e-6) # Logic inverted: <= threshold

                if is_touching and is_visible:
                    current_counter += 1
                    press_counters[key] = current_counter
                else:
                    press_counters[key] = 0

                if current_counter >= frames_required:
                    if points_np[head_index, 2] > 0:
                        removal_log.append(
                            self._format_log_entry(frame_idx, head_index, normalize, y_coord_to_check, frames_required)
                        )
                        removed = True
                    points_np[head_index, 2] = 0.0
                    points_np[head_index, 0] = 0.0
                    points_np[head_index, 1] = 0.0
            
            if removed:
                entry["keypoints_body"] = points_np.tolist()
                removed_any = True
        return removed_any

    def _prune_body_coords(self, frame_idx, coords, scores, height, normalize, padding_top, frames_required, press_counters, removal_log):
        if coords is None or scores is None:
            return False

        removed_any = False
        for head_index in self.HEAD_INDICES:
            if head_index >= len(coords) or head_index >= len(scores):
                continue
            
            key = (0, head_index)
            current_counter = press_counters.get(key, 0)
            coord_pair = coords[head_index]
            if coord_pair is None or len(coord_pair) < 2:
                press_counters[key] = 0
                continue

            y_coord_pixel = self._to_float(coord_pair[1])
            is_visible = scores[head_index] > 0.0
            
            if normalize:
                if not math.isfinite(height) or height <= 0.0:
                    press_counters[key] = 0
                    continue
                threshold = padding_top
                y_coord_to_check = y_coord_pixel / height
            else:
                if not math.isfinite(height) or height <= 0.0:
                    press_counters[key] = 0
                    continue
                threshold = padding_top * height
                y_coord_to_check = y_coord_pixel

            if not math.isfinite(y_coord_to_check):
                press_counters[key] = 0
                continue

            is_touching = (y_coord_to_check <= threshold + 1e-6) # Logic inverted: <= threshold

            if is_touching and is_visible:
                current_counter += 1
                press_counters[key] = current_counter
            else:
                press_counters[key] = 0

            if current_counter >= frames_required:
                if scores[head_index] > 0.0:
                    removal_log.append(
                        self._format_log_entry(frame_idx, head_index, normalize, y_coord_to_check, frames_required)
                    )
                    removed_any = True
                scores[head_index] = 0.0
                coord_pair[0] = 0.0
                coord_pair[1] = 0.0

        return removed_any

    def _format_log_entry(self, frame_idx, head_index, normalize, y_value, frame_count):
        name = self.BODY_KEYPOINT_NAMES.get(head_index, f"index_{head_index}")
        value_type = "normalized" if normalize else "pixel"
        return (
            f"Frame {frame_idx}: removed {name} (y={y_value:.4f} {value_type}) after {frame_count} frames (hit top)."
        )

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")
            
class PoseDataEditorJitterDeleter:
    # Diese Node löscht Keypoints, die sich zwischen Frames unnatürlich schnell bewegen (Jitter/Glitch).
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "velocity_threshold": (
                    "FLOAT",
                    {
                        "default": 50.0,
                        "min": 1.0,
                        "max": 1024.0,
                        "step": 1.0,
                        "tooltip": "Maximale Distanz (in Pixel oder normalisiert), die sich ein Punkt pro Frame bewegen darf, bevor er als Jitter gelöscht wird.",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Behandelt den Schwellenwert normalisiert (0-1, relativ zur Höhe). Empfohlen: False (Pixel).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("POSEDATA", "STRING")
    RETURN_NAMES = ("pose_data", "log")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = (
        "Removes keypoints that 'jitter' or 'glitch' (move faster than a threshold) between frames."
    )

    def process(self, pose_data, velocity_threshold, normalize):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas") or []
        pose_metas_original = pose_data_copy.get("pose_metas_original") or []

        if not pose_metas and not pose_metas_original:
            return (pose_data_copy, "No pose frames available.")

        removal_log = []

        if pose_metas:
            self._process_meta_list(
                pose_metas, velocity_threshold, normalize, removal_log
            )
        
        if pose_metas_original:
            self._process_original_list(
                pose_metas_original, velocity_threshold, normalize, removal_log
            )

        if not removal_log:
            removal_log.append("No jitter detected.")

        return (pose_data_copy, "\n".join(removal_log))

    def _get_threshold_px(self, meta, threshold, normalize):
        if not normalize:
            return threshold # Wert ist bereits in Pixeln
        
        height = self._to_float(getattr(meta, "height", None))
        if not math.isfinite(height) or height <= 0.0:
            return float('inf') # Kann nicht berechnen, also nichts löschen
        return threshold * height

    def _process_meta_list(self, metas, threshold, normalize, log):
        # Beginne bei Frame 1, da wir Frame 0 als Referenz brauchen
        for frame_idx in range(1, len(metas)):
            meta_curr = metas[frame_idx]
            meta_prev = metas[frame_idx - 1]

            if not isinstance(meta_curr, AAPoseMeta) or not isinstance(meta_prev, AAPoseMeta):
                continue
            
            threshold_px = self._get_threshold_px(meta_curr, threshold, normalize)

            arrays_curr = [
                ("body", getattr(meta_curr, "kps_body", None), getattr(meta_curr, "kps_body_p", None)),
                ("lhand", getattr(meta_curr, "kps_lhand", None), getattr(meta_curr, "kps_lhand_p", None)),
                ("rhand", getattr(meta_curr, "kps_rhand", None), getattr(meta_curr, "kps_rhand_p", None)),
                ("face", getattr(meta_curr, "kps_face", None), getattr(meta_curr, "kps_face_p", None)),
            ]
            arrays_prev = {
                "body": (getattr(meta_prev, "kps_body", None), getattr(meta_prev, "kps_body_p", None)),
                "lhand": (getattr(meta_prev, "kps_lhand", None), getattr(meta_prev, "kps_lhand_p", None)),
                "rhand": (getattr(meta_prev, "kps_rhand", None), getattr(meta_prev, "kps_rhand_p", None)),
                "face": (getattr(meta_prev, "kps_face", None), getattr(meta_prev, "kps_face_p", None)),
            }

            for key_type, coords_curr, scores_curr in arrays_curr:
                coords_prev, scores_prev = arrays_prev[key_type]
                
                if coords_curr is None or scores_curr is None or coords_prev is None or scores_prev is None:
                    continue

                for kp_idx in range(len(coords_curr)):
                    if kp_idx >= len(coords_prev):
                        continue
                    
                    score_curr = scores_curr[kp_idx]
                    score_prev = scores_prev[kp_idx]
                    
                    # Nur vergleichen, wenn beide Punkte sichtbar waren/sind
                    if score_curr > 0.0 and score_prev > 0.0:
                        pos_curr = coords_curr[kp_idx][:2]
                        pos_prev = coords_prev[kp_idx][:2]
                        
                        distance = np.linalg.norm(pos_curr - pos_prev)
                        
                        if distance > threshold_px:
                            # Jitter/Glitch erkannt!
                            scores_curr[kp_idx] = 0.0
                            coords_curr[kp_idx] = [0.0, 0.0]
                            log.append(f"Frame {frame_idx}: Jitter detected for {key_type}[{kp_idx}]. Dist: {distance:.1f}px. Point removed.")

    def _process_original_list(self, meta_dicts, threshold, normalize, log):
        key_names = ["keypoints_body", "keypoints_left_hand", "keypoints_right_hand", "keypoints_face"]
        
        for frame_idx in range(1, len(meta_dicts)):
            entry_curr = meta_dicts[frame_idx]
            entry_prev = meta_dicts[frame_idx - 1]
            
            if not isinstance(entry_curr, dict) or not isinstance(entry_prev, dict):
                continue
                
            height = self._to_float(entry_curr.get("height"))
            width = self._to_float(entry_curr.get("width"))
            if not math.isfinite(height) or height <= 0 or not math.isfinite(width) or width <= 0:
                continue

            threshold_px = self._get_threshold_px(entry_curr, threshold, normalize)
            
            for key in key_names:
                points_curr_list = entry_curr.get(key)
                points_prev_list = entry_prev.get(key)

                if points_curr_list is None or points_prev_list is None:
                    continue
                
                points_curr = np.asarray(points_curr_list, dtype=np.float32)
                points_prev = np.asarray(points_prev_list, dtype=np.float32)

                if points_curr.ndim != 2 or points_curr.shape[1] < 3 or \
                   points_prev.ndim != 2 or points_prev.shape[1] < 3:
                    continue
                    
                for kp_idx in range(len(points_curr)):
                    if kp_idx >= len(points_prev):
                        continue
                    
                    score_curr = points_curr[kp_idx, 2]
                    score_prev = points_prev[kp_idx, 2]
                    
                    if score_curr > 0.0 and score_prev > 0.0:
                        pos_curr_norm = points_curr[kp_idx, :2]
                        pos_prev_norm = points_prev[kp_idx, :2]
                        
                        # In Pixel umrechnen für den Distanzvergleich
                        pos_curr_px = pos_curr_norm * np.array([width, height])
                        pos_prev_px = pos_prev_norm * np.array([width, height])
                        
                        distance = np.linalg.norm(pos_curr_px - pos_prev_px)
                        
                        if distance > threshold_px:
                            points_curr[kp_idx, 2] = 0.0 # Score auf 0 setzen
                            points_curr[kp_idx, 0] = 0.0
                            points_curr[kp_idx, 1] = 0.0
                            log.append(f"Frame {frame_idx}: Jitter detected for {key}[{kp_idx}]. Dist: {distance:.1f}px. Point removed (Originals).")
                
                # Wichtig: Die Liste in der Diktionär-Struktur aktualisieren
                entry_curr[key] = points_curr.tolist()

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")
            
class BlackStripeImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "blackstripe_left": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4096.0,
                        "step": 0.01,
                        "tooltip": "Width of the left black stripe."
                    },
                ),
                "blackstripe_right": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4096.0,
                        "step": 0.01,
                        "tooltip": "Width of the right black stripe."
                    },
                ),
                "blackstripe_top": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4096.0,
                        "step": 0.01,
                        "tooltip": "Height of the top black stripe."
                    },
                ),
                "blackstripe_bottom": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4096.0,
                        "step": 0.01,
                        "tooltip": "Height of the bottom black stripe."
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Interpret stripe sizes relative to the image dimensions."
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Overlays configurable black stripes (blackout bars) on top of each image."

    @staticmethod
    def _resolve_length(value, normalize, reference):
        value = float(value)
        if value <= 0.0:
            return 0
        if normalize:
            value *= float(reference)
        return max(int(round(value)), 0)

    def apply(
        self,
        images,
        blackstripe_left,
        blackstripe_right,
        blackstripe_top,
        blackstripe_bottom,
        normalize,
    ):
        if all(
            stripe == 0.0
            for stripe in (
                blackstripe_left,
                blackstripe_right,
                blackstripe_top,
                blackstripe_bottom,
            )
        ):
            return (images,)

        if isinstance(images, torch.Tensor):
            images_np = images.detach().cpu().numpy()
            images_device = images.device
            images_dtype = images.dtype
        else:
            images_np = np.asarray(images)
            images_device = None
            images_dtype = None

        if images_np.size == 0:
            return (images,)

        single_image = False
        if images_np.ndim == 3:
            images_np = images_np[None, ...]
            single_image = True

        result_images = []
        for image_np in images_np:
            # --- START KORREKTUR ---
            # Erstelle eine Kopie des Originalbildes, um darauf zu zeichnen
            overlaid_image = image_np.copy()
            height, width = overlaid_image.shape[:2]

            # Berechne die Pixel-Breiten der Streifen
            left_px = self._resolve_length(blackstripe_left, normalize, width)
            right_px = self._resolve_length(blackstripe_right, normalize, width)
            top_px = self._resolve_length(blackstripe_top, normalize, height)
            bottom_px = self._resolve_length(blackstripe_bottom, normalize, height)

            # Zeichne die Streifen (Overlays)
            # (Setze die Pixel auf 0.0, was bei 0-1 Float-Bildern schwarz ist)
            if top_px > 0:
                top_span = min(top_px, height) # Sicherstellen, dass der Balken nicht über das Bild hinausgeht
                overlaid_image[0:top_span, :, :] = 0.0

            if bottom_px > 0:
                bottom_span = min(bottom_px, height - top_px) # Sicherstellen, dass er nicht den oberen Balken überlappt
                if bottom_span > 0:
                    overlaid_image[-bottom_span:, :, :] = 0.0

            if left_px > 0:
                left_span = min(left_px, width)
                overlaid_image[:, 0:left_span, :] = 0.0

            if right_px > 0:
                right_span = min(right_px, width - left_px) # Sicherstellen, dass er nicht den linken Balken überlappt
                if right_span > 0:
                    overlaid_image[:, -right_span:, :] = 0.0

            result_images.append(overlaid_image)
            # --- ENDE KORREKTUR ---

        result_np = np.stack(result_images, axis=0)
        result_tensor = torch.from_numpy(result_np)

        if images_dtype is not None:
            result_tensor = result_tensor.to(dtype=images_dtype)
        else:
            # Fallback, falls die Eingabe kein Tensor war (z.B. reines Numpy-Array)
            result_tensor = result_tensor.to(dtype=torch.float32)

        if images_device is not None:
            result_tensor = result_tensor.to(device=images_device)

        if single_image:
            result_tensor = result_tensor[0]

        return (result_tensor,)
   
class ImageBatchBlackout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 240,
                        "step": 1,
                        "tooltip": "Frame rate of the image sequence.",
                    },
                ),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": "Duration in seconds to black out.",
                    },
                ),
                "reversed": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Reversed (End of Clip)",
                        "label_off": "Normal (Start of Clip)",
                        "tooltip": "If true, blacks out frames from the end. If false, blacks out from the start.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blackout_frames"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Blacks out frames at the beginning or end of a batch for a specified duration."

    def blackout_frames(self, images, fps, duration_seconds, reversed):
        if duration_seconds <= 0:
            # Nichts zu tun
            return (images,)

        # 1. Berechne die Anzahl der Frames, die schwarz sein sollen
        num_frames_to_blackout = max(0, int(round(duration_seconds * float(fps))))

        if num_frames_to_blackout == 0:
            return (images,)

        # 2. Hole die Gesamtanzahl der Frames
        total_frames = images.shape[0]

        # 3. Erstelle eine Kopie, um das Original nicht zu verändern
        # Bilder sind Float-Tensoren (0-1), Schwarz ist 0.0
        images_copy = images.clone()

        # 4. Logik anwenden
        if reversed:
            # Mache die letzten N Frames schwarz
            # Berechne den Start-Index für den Blackout
            start_index = max(0, total_frames - num_frames_to_blackout)
            
            if start_index < total_frames:
                images_copy[start_index:, :, :, :] = 0.0
        else:
            # Mache die ersten N Frames schwarz
            # Berechne den End-Index für den Blackout
            end_index = min(total_frames, num_frames_to_blackout)
            
            if end_index > 0:
                images_copy[0:end_index, :, :, :] = 0.0

        return (images_copy,)

class PoseRetargetPromptHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("prompt", "retarget_prompt", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Generates text prompts for pose retargeting based on visibility of arms and legs in the template pose. Originally used for Flux Kontext"

    def process(self, pose_data):
        refer_pose_meta = pose_data.get("refer_pose_meta", None)
        if refer_pose_meta is None:
            return ("Change the person to face forward.", "Change the person to face forward.", )
        tpl_pose_metas = pose_data["pose_metas_original"]
        arm_visible = False
        leg_visible = False

        for tpl_pose_meta in tpl_pose_metas:
            tpl_keypoints = tpl_pose_meta['keypoints_body']
            tpl_keypoints = np.array(tpl_keypoints)
            if np.any(tpl_keypoints[3]) != 0 or np.any(tpl_keypoints[4]) != 0 or np.any(tpl_keypoints[6]) != 0 or np.any(tpl_keypoints[7]) != 0:
                if (tpl_keypoints[3][0] <= 1 and tpl_keypoints[3][1] <= 1 and tpl_keypoints[3][2] >= 0.75) or (tpl_keypoints[4][0] <= 1 and tpl_keypoints[4][1] <= 1 and tpl_keypoints[4][2] >= 0.75) or \
                    (tpl_keypoints[6][0] <= 1 and tpl_keypoints[6][1] <= 1 and tpl_keypoints[6][2] >= 0.75) or (tpl_keypoints[7][0] <= 1 and tpl_keypoints[7][1] <= 1 and tpl_keypoints[7][2] >= 0.75):
                    arm_visible = True
            if np.any(tpl_keypoints[9]) != 0 or np.any(tpl_keypoints[12]) != 0 or np.any(tpl_keypoints[10]) != 0 or np.any(tpl_keypoints[13]) != 0:
                if (tpl_keypoints[9][0] <= 1 and tpl_keypoints[9][1] <= 1 and tpl_keypoints[9][2] >= 0.75) or (tpl_keypoints[12][0] <= 1 and tpl_keypoints[12][1] <= 1 and tpl_keypoints[12][2] >= 0.75) or \
                    (tpl_keypoints[10][0] <= 1 and tpl_keypoints[10][1] <= 1 and tpl_keypoints[10][2] >= 0.75) or (tpl_keypoints[13][0] <= 1 and tpl_keypoints[13][1] <= 1 and tpl_keypoints[13][2] >= 0.75):
                    leg_visible = True
            if arm_visible and leg_visible:
                break

        if leg_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."
        elif arm_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."
        else:
            tpl_prompt = "Change the person to face forward."
            refer_prompt = "Change the person to face forward."

        return (tpl_prompt, refer_prompt, )


class PoseDataToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1}),
                "stick_width": ("INT", {
                    "default": 10, 
                    "min": 1, 
                    "max": 300, 
                    "step": 1,
                    "display": "slider",
                    "slider_max": 200 
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Erstellt Maske. Füllt Körper & Stirn. Fallback auf vorherigen Frame."

    def process(self, pose_data, width, height, stick_width):
        pose_metas = pose_data["pose_metas"]
        mask_list = []

        # Speicher für Fallback
        last_valid_mask = np.zeros((height, width), dtype=np.float32)

        for meta in pose_metas:
            kps = meta.kps_body
            scores = meta.kps_body_p
            
            def get_pt(idx):
                if idx < len(scores) and scores[idx] > 0.3:
                    return (int(kps[idx][0]), int(kps[idx][1]))
                return None

            # Schultern prüfen
            p_lsh, p_rsh = get_pt(5), get_pt(2)

            if p_lsh and p_rsh:
                # === NEU ZEICHNEN ===
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                p_lhip, p_rhip = get_pt(11), get_pt(8)

                # 1. TORSO
                if p_lhip and p_rhip:
                    pts_torso = np.array([p_lsh, p_rsh, p_rhip, p_lhip], np.int32)
                    cv2.fillPoly(canvas, [pts_torso], (255, 255, 255))
                else:
                    p_r_bottom = (p_rsh[0], height)
                    p_l_bottom = (p_lsh[0], height)
                    pts_torso = np.array([p_lsh, p_rsh, p_r_bottom, p_l_bottom], np.int32)
                    cv2.fillPoly(canvas, [pts_torso], (255, 255, 255))

                # 2. KOPF & STIRN
                head_pts = []
                # Relevante Punkte holen
                p_nose = get_pt(0)
                p_lear = get_pt(17) or get_pt(15) # Ohr oder Auge Links
                p_rear = get_pt(16) or get_pt(14) # Ohr oder Auge Rechts

                # Basis-Polygon für Kopf/Hals
                if p_nose: head_pts.append(p_nose)
                if p_lear: head_pts.append(p_lear)
                head_pts.append(p_lsh)
                head_pts.append(p_rsh)
                if p_rear: head_pts.append(p_rear)
                
                if len(head_pts) >= 3:
                    pts_head = np.array(head_pts, np.int32)
                    cv2.fillPoly(canvas, [pts_head], (255, 255, 255))

                    # --- ZUSATZ: STIRN-RECHTECK ---
                    # Wir brauchen Ohren/Augen, um die Breite zu wissen
                    if p_lear and p_rear:
                        # Berechne Y-Höhe der Augen/Ohren (Durchschnitt)
                        eye_y = (p_lear[1] + p_rear[1]) / 2
                        # Berechne Y-Höhe der Schultern
                        shoulder_y = (p_lsh[1] + p_rsh[1]) / 2
                        
                        # Distanz Augen <-> Schultern
                        dist_head_shoulder = abs(shoulder_y - eye_y)
                        
                        # Stirnhöhe = Hälfte dieser Distanz
                        forehead_height = int(dist_head_shoulder * 0.65) # Faktor 0.65 für Sicherheit
                        
                        # Koordinaten für das Rechteck
                        # Wir nehmen die X-Werte der Ohren/Augen als Breite
                        x_min = min(p_lear[0], p_rear[0])
                        x_max = max(p_lear[0], p_rear[0])
                        
                        # Y-Start ist auf Augenhöhe, Y-Ende ist weiter oben
                        y_bottom = int(eye_y)
                        y_top = int(eye_y - forehead_height)

                        # Rechteckpunkte
                        pt1 = (x_min, y_bottom)
                        pt2 = (x_max, y_bottom)
                        pt3 = (x_max, y_top)
                        pt4 = (x_min, y_top)
                        
                        pts_forehead = np.array([pt1, pt2, pt3, pt4], np.int32)
                        cv2.fillPoly(canvas, [pts_forehead], (255, 255, 255))

                # 3. SKELETT
                skeleton_canvas = np.zeros_like(canvas)
                skeleton_img = draw_aapose_by_meta_new(
                    skeleton_canvas, 
                    meta, 
                    draw_hand=True, 
                    draw_head=True,
                    body_stick_width=stick_width,
                    hand_stick_width=max(1, stick_width // 2)
                )

                # 4. MERGE
                filled_mask = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
                skeleton_mask = cv2.cvtColor(skeleton_img, cv2.COLOR_RGB2GRAY)
                final_mask_combined = cv2.bitwise_or(filled_mask, skeleton_mask)
                
                last_valid_mask = (final_mask_combined > 0).astype(np.float32)

            else:
                # FALLBACK (Schultern weg -> altes Bild behalten)
                pass 

            mask_tensor = torch.from_numpy(last_valid_mask)
            mask_list.append(mask_tensor)

        return (torch.stack(mask_list, dim=0),)


class PoseDataToOvalMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1}),
                # Optional: Ein Faktor, um die Box etwas breiter/schmaler zu machen
                "width_scale": ("FLOAT", {"default": 1.2, "min": 0.8, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Erstellt eine Body-Box (Rechteck), die am Kopf hängt und sich die Breite merkt, wenn der Körper aus dem Bild geht."

    def process(self, pose_data, width, height, width_scale):
        pose_metas = pose_data["pose_metas"]
        mask_list = []

        # Wir merken uns die letzte "gute" halbe Breite (Abstand von Mitte nach Außen)
        last_half_width = None
        
        # Indizes für die Breite (Wir ignorieren Arme 3,4,6,7):
        # 2: RShoulder, 5: LShoulder, 8: RHip, 11: LHip, 0: Nose, 14-17: Eyes/Ears
        width_indices = [2, 5, 8, 11, 0, 14, 15, 16, 17]
        
        # Indizes für unten (Knie, Füße):
        bottom_indices = [9, 10, 12, 13, 19, 20] # Knie, Knöchel, Zehen

        for meta in pose_metas:
            canvas = np.zeros((height, width), dtype=np.float32) # Float Canvas 0.0-1.0
            
            kps = meta.kps_body
            scores = meta.kps_body_p

            # Helper
            def get_pt(idx):
                if idx < len(scores) and scores[idx] > 0.3:
                    return np.array(kps[idx][:2])
                return None

            # 1. KOPF / ZENTRUM FINDEN
            # Wir brauchen ein Zentrum, an dem die Box hängt. Am besten Durchschnitt aus Schultern oder Ohren.
            p_lsh, p_rsh = get_pt(5), get_pt(2)
            p_lear, p_rear = (get_pt(17) or get_pt(15)), (get_pt(16) or get_pt(14))
            
            center_x = None
            
            # Versuch 1: Mitte der Schultern
            if p_lsh is not None and p_rsh is not None:
                center_x = (p_lsh[0] + p_rsh[0]) / 2
            # Versuch 2: Mitte der Ohren/Augen
            elif p_lear is not None and p_rear is not None:
                center_x = (p_lear[0] + p_rear[0]) / 2
            # Versuch 3: Nase
            elif get_pt(0) is not None:
                center_x = get_pt(0)[0]

            if center_x is not None:
                # === A. STIRN-LOGIK (TOP Y) ===
                top_y = 0
                if p_lear is not None and p_rear is not None and p_lsh is not None and p_rsh is not None:
                    eye_y = (p_lear[1] + p_rear[1]) / 2
                    sh_y = (p_lsh[1] + p_rsh[1]) / 2
                    dist = abs(sh_y - eye_y)
                    forehead_add = dist * 0.7  # Faktor für Stirnhöhe
                    top_y = max(0, int(eye_y - forehead_add))
                elif get_pt(0) is not None:
                    # Fallback nur Nase -> Pauschal etwas drüber
                    top_y = max(0, int(get_pt(0)[1] - (height * 0.1)))

                # === B. BREITE BESTIMMEN ===
                # Suche den breitesten Punkt im aktuellen Frame (ohne Arme)
                current_min_x = width
                current_max_x = 0
                found_body_parts = False

                for idx in width_indices:
                    pt = get_pt(idx)
                    if pt is not None:
                        if pt[0] < current_min_x: current_min_x = pt[0]
                        if pt[0] > current_max_x: current_max_x = pt[0]
                        found_body_parts = True
                
                # Berechnung der halben Breite (vom Zentrum aus)
                current_half_width = 0
                if found_body_parts:
                    w = (current_max_x - current_min_x)
                    # Sicherstellen, dass wir eine Mindestbreite haben (z.B. Kopfbreite)
                    if w < 10: w = 50 
                    current_half_width = (w / 2) * width_scale

                # LOGIK: Sind Hüften da?
                p_lhip, p_rhip = get_pt(11), get_pt(8)
                hips_present = (p_lhip is not None or p_rhip is not None)

                if hips_present:
                    # Wenn Hüften da sind, vertrauen wir der aktuellen Breite und speichern sie
                    draw_half_width = current_half_width
                    last_half_width = current_half_width
                else:
                    # Keine Hüften (zu nah dran)? 
                    # Nimm die gespeicherte Breite. Wenn keine gespeichert, nimm die aktuelle (Schultern).
                    if last_half_width is not None:
                        draw_half_width = last_half_width
                    else:
                        draw_half_width = current_half_width

                # === C. UNTEN BESTIMMEN (BOTTOM Y) ===
                # Suche den tiefsten Punkt
                max_y = 0
                found_legs = False
                for idx in bottom_indices:
                    pt = get_pt(idx)
                    if pt is not None:
                        if pt[1] > max_y: max_y = pt[1]
                        found_legs = True
                
                if found_legs:
                    # Beine da -> Box geht bis zu den Füßen (+ etwas Puffer)
                    bottom_y = min(height, int(max_y + 20))
                else:
                    # Keine Beine -> Box geht bis ganz nach unten (und darüber hinaus)
                    bottom_y = height 

                # === D. ZEICHNEN ===
                # Wir zeichnen ein Rechteck
                x1 = int(center_x - draw_half_width)
                x2 = int(center_x + draw_half_width)
                y1 = int(top_y)
                y2 = int(bottom_y)

                # Zeichnen (1.0 für Weiß)
                # cv2.rectangle erwartet Integer Koordinaten. 
                # Wir füllen einfach diesen Bereich im Array.
                
                # Clipping für Array-Zugriff, damit es nicht crasht
                x1 = max(0, x1)
                x2 = min(width, x2)
                y1 = max(0, y1)
                y2 = min(height, y2)

                if x2 > x1 and y2 > y1:
                    canvas[y1:y2, x1:x2] = 1.0

            else:
                # Kein Kopf gefunden? 
                # Option A: Schwarzes Bild
                # Option B: Letzte Maske wiederholen (könnte man hier einbauen)
                pass

            mask_tensor = torch.from_numpy(canvas)
            mask_list.append(mask_tensor)

        return (torch.stack(mask_list, dim=0),)


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


import copy
import numpy as np

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


from .pose_utils.human_visualization import draw_handpose_new, AAPoseMeta
import math

# Hilfsfunktion mit korrigierter Zeichenreihenfolge (Arme zuletzt)
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


class DrawViTPose_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
                "retarget_padding": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "When > 0, the retargeted pose image is padded and resized to the target size"}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks. Set to 0 to disable body drawing, -1 for auto"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks. Set to 0 to disable hand drawing, -1 for auto"}),
                "draw_head": ("BOOLEAN", {"default": "True", "tooltip": "Whether to draw head keypoints"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("pose_images", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Draws pose images from pose data with fixed z-order (Arms drawn last)."

    def process(self, pose_data, width, height, body_stick_width, hand_stick_width, draw_head, retarget_padding=64):

        retarget_image = pose_data.get("retarget_image", None)
        pose_metas = pose_data["pose_metas"]

        draw_hand = hand_stick_width != 0
        use_retarget_resize = retarget_padding > 0 and retarget_image is not None

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        crop_target_image = None
        pose_images = []

        for idx, meta in enumerate(tqdm(pose_metas, desc="Drawing pose images v2")):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Manuelle Vorbereitung der Daten wie in draw_aapose_by_meta_new
            kp2ds = np.concatenate([meta.kps_body, meta.kps_body_p[:, None]], axis=1)
            kp2ds_lhand = np.concatenate([meta.kps_lhand, meta.kps_lhand_p[:, None]], axis=1)
            kp2ds_rhand = np.concatenate([meta.kps_rhand, meta.kps_rhand_p[:, None]], axis=1)
            
            # Aufruf der gefixten Funktion
            pose_image = draw_aapose_v2(
                canvas, 
                kp2ds, 
                draw_hand=draw_hand, 
                draw_head=draw_head, 
                body_stick_width=body_stick_width, 
                hand_stick_width=hand_stick_width,
                kp2ds_lhand=kp2ds_lhand,
                kp2ds_rhand=kp2ds_rhand
            )

            if crop_target_image is None:
                crop_target_image = pose_image

            if use_retarget_resize:
                pose_image = resize_to_bounds(pose_image, height, width, crop_target_image=crop_target_image, extra_padding=retarget_padding)
            else:
                pose_image = padding_resize(pose_image, height, width)

            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0

        return (pose_images_tensor, )

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


class DrawViTPose_v3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
                "retarget_padding": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "When > 0, the retargeted pose image is padded and resized to the target size"}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks. Set to 0 to disable body drawing, -1 for auto"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks. Set to 0 to disable hand drawing, -1 for auto"}),
                "draw_head": ("BOOLEAN", {"default": "True", "tooltip": "Whether to draw head keypoints"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("pose_images", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Draws pose with layer order: Body (Back) -> Legs (Mid) -> Arms (Front)."

    def process(self, pose_data, width, height, body_stick_width, hand_stick_width, draw_head, retarget_padding=64):
        import cv2
        import numpy as np
        import torch
        from .pose_utils.human_visualization import draw_handpose_new
        from .utils import resize_to_bounds, padding_resize
        import math

        retarget_image = pose_data.get("retarget_image", None)
        pose_metas = pose_data["pose_metas"]

        draw_hand = hand_stick_width != 0
        use_retarget_resize = retarget_padding > 0 and retarget_image is not None

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        crop_target_image = None
        pose_images = []

        # --- Definition der Verbindungen (Indizes basieren auf pose2d_utils Logik) ---
        limbSeq_orig = [
            [2, 3], [2, 6],  # 0, 1: Shoulders (Neck -> RSho, Neck -> LSho)
            [3, 4], [4, 5],  # 2, 3: Right Arm (Sho->Elb, Elb->Wri)
            [6, 7], [7, 8],  # 4, 5: Left Arm (Sho->Elb, Elb->Wri)
            [2, 9],          # 6: Right Body (Neck->RHip)
            [9, 10], [10, 11], # 7, 8: Right Leg (Hip->Knee, Knee->Ank)
            [2, 12],         # 9: Left Body (Neck->LHip)
            [12, 13], [13, 14], # 10, 11: Left Leg (Hip->Knee, Knee->Ank)
            [2, 1],          # 12: Neck->Nose
            [1, 15], [15, 17], # 13, 14: Face Right (Nose->Eye, Eye->Ear)
            [1, 16], [16, 18], # 15, 16: Face Left
            [14, 19],        # 17: Left Foot (Ank->Toe)
            [11, 20]         # 18: Right Foot (Ank->Toe)
        ]

        colors_orig = [
            [255, 0, 0], [255, 85, 0],      # 0, 1 (Shoulders)
            [255, 170, 0], [255, 255, 0],   # 2, 3 (Right Arm)
            [170, 255, 0], [85, 255, 0],    # 4, 5 (Left Arm)
            [0, 255, 0],                    # 6 (Right Body)
            [0, 255, 85], [0, 255, 170],    # 7, 8 (Right Leg)
            [0, 255, 255],                  # 9 (Left Body)
            [0, 170, 255], [0, 85, 255],    # 10, 11 (Left Leg)
            [0, 0, 255],                    # 12 (Neck)
            [85, 0, 255], [170, 0, 255],    # 13, 14 (Face)
            [255, 0, 255], [255, 0, 170],   # 15, 16 (Face)
            [255, 0, 85],                   # 17 (Foot L)
            [200, 200, 0]                   # 18 (Foot R)
        ]

        # --- Z-ORDER SORTER ---
        # Reihenfolge: Zuerst (Hintergrund) -> Zuletzt (Vordergrund)
        
        # 1. KÖRPER (Ganz hinten)
        # Indizes: 0,1 (Schultern), 6 (Rumpf R), 9 (Rumpf L), 12-16 (Kopf)
        indices_body = [0, 1, 6, 9, 12, 13, 14, 15, 16]

        # 2. BEINE (Mitte - verdecken Körper)
        # Indizes: 7, 8 (Bein R), 10, 11 (Bein L), 17, 18 (Füße)
        indices_legs = [7, 8, 10, 11, 17, 18]

        # 3. ARME (Ganz vorne - verdecken Beine & Körper)
        # Indizes: 2, 3 (Arm R), 4, 5 (Arm L)
        indices_arms = [2, 3, 4, 5]

        # Die finale Liste zum Zeichnen
        draw_order = indices_body + indices_legs + indices_arms

        limbSeq = [limbSeq_orig[i] for i in draw_order]
        colors = [colors_orig[i] for i in draw_order]

        for idx, meta in enumerate(tqdm(pose_metas, desc="Drawing pose images v3")):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Daten vorbereiten
            kp2ds = np.concatenate([meta.kps_body, meta.kps_body_p[:, None]], axis=1)
            kp2ds_lhand = np.concatenate([meta.kps_lhand, meta.kps_lhand_p[:, None]], axis=1)
            kp2ds_rhand = np.concatenate([meta.kps_rhand, meta.kps_rhand_p[:, None]], axis=1)

            if not draw_head:
                kp2ds[[0,14,15,16,17], 2] = 0
            kp2ds_body = kp2ds

            # Stick width berechnen
            if body_stick_width == -1:
                stickwidth = max(int(min(height, width) / 200) - 1, 1)
            else:
                stickwidth = body_stick_width

            threshold = 0.5

            # --- Zeichnen der Linien (in sortierter Reihenfolge) ---
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
                cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

            # --- Zeichnen der Gelenkpunkte (Kreise) ---
            # Diese zeichnen wir auch am Ende, damit sie sauber aussehen
            for _idx, (keypoint, color) in enumerate(zip(kp2ds_body, colors_orig)): 
                if _idx >= len(colors_orig): break
                if keypoint[-1] < threshold:
                    continue
                x, y = keypoint[0], keypoint[1]
                cv2.circle(canvas, (int(x), int(y)), stickwidth, colors_orig[_idx], thickness=-1)

            # --- Hände ---
            # Hände zeichnen wir hier als allerletztes Overlay
            if draw_hand:
                canvas = draw_handpose_new(canvas, kp2ds_lhand, stickwidth_type='v2', hand_score_th=threshold, hand_stick_width=hand_stick_width)
                canvas = draw_handpose_new(canvas, kp2ds_rhand, stickwidth_type='v2', hand_score_th=threshold, hand_stick_width=hand_stick_width)

            pose_image = canvas

            if crop_target_image is None:
                crop_target_image = pose_image

            if use_retarget_resize:
                pose_image = resize_to_bounds(pose_image, height, width, crop_target_image=crop_target_image, extra_padding=retarget_padding)
            else:
                pose_image = padding_resize(pose_image, height, width)

            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0

        return (pose_images_tensor, )

class KeypointDeleter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "delete_face": ("BOOLEAN", {"default": False, "tooltip": "Löscht Gesicht (Augen, Ohren, Nase)"}),
                "delete_torso": ("BOOLEAN", {"default": False, "tooltip": "Löscht Schultern und Hüften, aber behält den Hals"}),
                "delete_arms": ("BOOLEAN", {"default": False, "tooltip": "Löscht BEIDE Arme (Ellbogen, Handgelenk) - Schultern bleiben!"}),
                "delete_legs": ("BOOLEAN", {"default": False, "tooltip": "Löscht BEIDE Beine (Knie, Fuß, Zehen) - Hüften bleiben!"}),
                "delete_left_arm": ("BOOLEAN", {"default": False, "tooltip": "Löscht nur den linken Arm"}),
                "delete_right_arm": ("BOOLEAN", {"default": False, "tooltip": "Löscht nur den rechten Arm"}),
                "delete_left_leg": ("BOOLEAN", {"default": False, "tooltip": "Löscht nur das linke Bein"}),
                "delete_right_leg": ("BOOLEAN", {"default": False, "tooltip": "Löscht nur das rechte Bein"}),
                "delete_hands": ("BOOLEAN", {"default": False, "tooltip": "Löscht detaillierte Finger/Hand-Keypoints"}),
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Löscht ausgewählte Segmente aus den Pose-Daten (Schultern/Hüften bleiben bei Arm/Bein-Löschung erhalten)."

    def process(self, pose_data, delete_face, delete_torso, delete_arms, delete_legs, delete_left_arm, delete_right_arm, delete_left_leg, delete_right_leg, delete_hands):
        import copy
        import numpy as np

        new_pose_data = copy.deepcopy(pose_data)
        pose_metas = new_pose_data['pose_metas']

        # Indizes (COCO + Foot):
        # 0:Nose, 1:Neck
        # 2:RSho, 3:RElb, 4:RWri
        # 5:LSho, 6:LElb, 7:LWri
        # 8:RHip, 9:RKnee, 10:RAnk
        # 11:LHip, 12:LKnee, 13:LAnk
        # 14:REye, 15:LEye, 16:REar, 17:LEar
        # 18:LToe, 19:RToe

        face_indices = [0, 14, 15, 16, 17]
        
        # Torso delete: Schultern (2, 5) und Hüften (8, 11). 
        # Hals (1) bleibt erhalten!
        torso_indices = [2, 5, 8, 11]

        # Arm delete: Nur Ellbogen (3/6) und Handgelenk (4/7). 
        # Schulter (2/5) bleibt erhalten!
        right_arm_indices = [3, 4]
        left_arm_indices = [6, 7]
        
        # Bein delete: Nur Knie (9/12), Knöchel (10/13) und Zehen (19/18). 
        # Hüfte (8/11) bleibt erhalten!
        right_leg_indices = [9, 10, 19]
        left_leg_indices = [12, 13, 18]

        for meta in pose_metas:
            # Gesicht
            if delete_face:
                meta.kps_body[face_indices, :] = 0
                meta.kps_body_p[face_indices] = 0
            
            # Torso (Hals bleibt)
            if delete_torso:
                meta.kps_body[torso_indices, :] = 0
                meta.kps_body_p[torso_indices] = 0

            # Arme (Schultern bleiben)
            if delete_right_arm or delete_arms:
                meta.kps_body[right_arm_indices, :] = 0
                meta.kps_body_p[right_arm_indices] = 0
                if hasattr(meta, 'kps_rhand'):
                     meta.kps_rhand[:] = 0
                     meta.kps_rhand_p[:] = 0

            if delete_left_arm or delete_arms:
                meta.kps_body[left_arm_indices, :] = 0
                meta.kps_body_p[left_arm_indices] = 0
                if hasattr(meta, 'kps_lhand'):
                     meta.kps_lhand[:] = 0
                     meta.kps_lhand_p[:] = 0

            # Beine (Hüften bleiben)
            if delete_right_leg or delete_legs:
                meta.kps_body[right_leg_indices, :] = 0
                meta.kps_body_p[right_leg_indices] = 0

            if delete_left_leg or delete_legs:
                meta.kps_body[left_leg_indices, :] = 0
                meta.kps_body_p[left_leg_indices] = 0

            # Hände
            if delete_hands:
                if hasattr(meta, 'kps_lhand'):
                    meta.kps_lhand[:] = 0
                    meta.kps_lhand_p[:] = 0
                if hasattr(meta, 'kps_rhand'):
                    meta.kps_rhand[:] = 0
                    meta.kps_rhand_p[:] = 0

        return (new_pose_data,)




# ==============================================================================
# Node: Mask Positional Cutter V14
# ==============================================================================

class MaskPositionalCutterV14:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",), 
                "padding": ("INT", {"default": 30, "min": 0, "max": 1024, "step": 1, "tooltip": "Puffer-Rand. Wichtig für weiche Kamerafahrten!"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1, "tooltip": "Zielauflösung in MP."}),
                "camera_smooth_window": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1, "tooltip": "Gimbal Window (0 = Aus)."}),
                "smoothing_passes": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "tooltip": "Multi-Pass für butterweiche Kamera."}),
                "keep_mask_100_percent_inside": (["yes", "no"], {"default": "yes", "tooltip": "Zwingt die Kamera, die Maske nie abzuschneiden."}),
                "background_color": (["black", "white"], {"default": "black", "tooltip": "Füllfarbe am Bildrand."}),
            },
            "optional": {
                "opt_mask": ("MASK",),
                "opt_positive_points": ("*",), 
                "opt_negative_points": ("*",),
                "opt_bboxes": ("*",), 
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK_CUT_INFO_V14", "STRING", "STRING", "STRING", "*", "*", "*")
    RETURN_NAMES = ("cropped_images", "mask_cutted", "mask_cutted_opt", "cut_info", "trans_pos_json", "trans_neg_json", "trans_bbox_json", "trans_pos_raw", "trans_neg_raw", "trans_bbox_raw")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Masking"
    DESCRIPTION = "V14: Sub-Pixel Cutter für den Total-Size-Agnostic Joiner V14."

    def process(self, images, mask, padding, megapixels, camera_smooth_window, smoothing_passes, keep_mask_100_percent_inside, background_color, opt_mask=None, opt_positive_points=None, opt_negative_points=None, opt_bboxes=None):
        import cv2
        import numpy as np
        import torch
        import math
        import json
        
        B, H, W, C = images.shape
        if mask.ndim == 2: mask = mask.unsqueeze(0)
        if mask.shape[0] < B: mask = mask.repeat(B, 1, 1)

        has_opt_mask = False
        if opt_mask is not None:
            has_opt_mask = True
            if opt_mask.ndim == 2: opt_mask = opt_mask.unsqueeze(0)
            if opt_mask.shape[0] < B: opt_mask = opt_mask.repeat(B, 1, 1)
        
        raw_centers = []
        mask_bounds = [] 
        global_max_w = 0
        global_max_h = 0
        
        for i in range(B):
            m = mask[i].cpu().numpy()
            y_indices, x_indices = np.nonzero(m > 0.5)
            if len(y_indices) == 0:
                raw_centers.append(None)
                mask_bounds.append(None)
            else:
                x1, x2 = x_indices.min(), x_indices.max()
                y1, y2 = y_indices.min(), y_indices.max()
                w = x2 - x1
                h = y2 - y1
                if w > global_max_w: global_max_w = w
                if h > global_max_h: global_max_h = h
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                raw_centers.append((cx, cy))
                mask_bounds.append((x1, y1, x2, y2))

        if global_max_w == 0: global_max_w = 128
        if global_max_h == 0: global_max_h = 128
        
        box_w = int(global_max_w + (padding * 2))
        box_h = int(global_max_h + (padding * 2))
        
        valid_centers = []
        last_valid = (W/2.0, H/2.0)
        for c in raw_centers:
            if c is not None: last_valid = c
            valid_centers.append(last_valid)
            
        valid_bounds = []
        last_b = (W/2.0-10, H/2.0-10, W/2.0+10, H/2.0+10)
        for b in mask_bounds:
            if b is not None: last_b = b
            valid_bounds.append(last_b)

        def smooth_series(series, passes, window):
            if window <= 0: return series
            res = series.copy()
            for _ in range(passes):
                temp = []
                for i in range(len(res)):
                    s = max(0, i - window)
                    e = min(len(res), i + window + 1)
                    w_slice = res[s:e]
                    avg = tuple(sum(val[k] for val in w_slice) / len(w_slice) for k in range(len(res[0])))
                    temp.append(avg)
                res = temp
            return res

        smoothed_centers = smooth_series(valid_centers, smoothing_passes, camera_smooth_window)
        smoothed_bounds = smooth_series(valid_bounds, smoothing_passes, camera_smooth_window)
        
        final_centers = []
        for i in range(B):
            cx, cy = smoothed_centers[i]
            if keep_mask_100_percent_inside == "yes":
                x1, y1, x2, y2 = smoothed_bounds[i]
                min_cx = x2 - box_w / 2.0
                max_cx = x1 + box_w / 2.0
                min_cy = y2 - box_h / 2.0
                max_cy = y1 + box_h / 2.0
                if min_cx <= max_cx: cx = max(min_cx, min(max_cx, cx))
                if min_cy <= max_cy: cy = max(min_cy, min(max_cy, cy))
            final_centers.append((cx, cy))

        target_pixel_count = megapixels * 1_000_000
        aspect_ratio = box_w / box_h
        target_h_float = math.sqrt(target_pixel_count / aspect_ratio)
        target_w_float = target_h_float * aspect_ratio
        target_w = int(round(target_w_float / 8) * 8)
        target_h = int(round(target_h_float / 8) * 8)
        target_w = max(64, target_w)
        target_h = max(64, target_h)
        
        scale_x = target_w / float(box_w)
        scale_y = target_h / float(box_h)
        
        img_bg_val = 0.0 if background_color == "black" else 1.0
        mask_bg_val = 0.0
        
        def get_item_safe(idx, data):
            if data is None: return None
            if isinstance(data, str):
                try: data = json.loads(data)
                except: pass
            if isinstance(data, (list, tuple)):
                if len(data) == 0: return None
                return data[idx] if idx < len(data) else data[-1]
            if hasattr(data, "shape"):
                if data.shape[0] == 0: return None
                return data[idx] if idx < data.shape[0] else data[-1]
            return data

        def transform_coords_affine(coords, t_x, t_y, sx, sy, limit_w, limit_h, is_bbox=False):
            if coords is None: return []
            is_tensor = hasattr(coords, "cpu")
            if is_tensor: pts = coords.cpu().numpy()
            else: pts = np.array(coords)
            if pts.size == 0 or pts.ndim == 0: return []
            new_pts = []
            if is_bbox:
                if pts.ndim == 1 and len(pts) == 4: pts = pts.reshape(1, 4)
                elif pts.ndim == 1: return []
                for b in pts:
                    if len(b) < 4: continue
                    nx1, ny1 = b[0]*sx + t_x, b[1]*sy + t_y
                    nx2, ny2 = b[2]*sx + t_x, b[3]*sy + t_y
                    nx1, ny1 = max(0, min(limit_w, nx1)), max(0, min(limit_h, ny1))
                    nx2, ny2 = max(0, min(limit_w, nx2)), max(0, min(limit_h, ny2))
                    if nx2>nx1 and ny2>ny1: new_pts.append([float(nx1), float(ny1), float(nx2), float(ny2)])
            else:
                if pts.ndim == 1 and len(pts) == 2: pts = pts.reshape(1, 2)
                elif pts.ndim == 1: return []
                for p in pts:
                    if len(p) < 2: continue
                    nx, ny = p[0]*sx + t_x, p[1]*sy + t_y
                    if 0<=nx<limit_w and 0<=ny<limit_h: new_pts.append([float(nx), float(ny)])
            return new_pts

        cropped_images, cropped_masks, cropped_opt_masks = [], [], []
        cut_infos, out_pos, out_neg, out_bbox = [], [], [], []

        for i in range(B):
            img = images[i].cpu().numpy()
            msk = mask[i].cpu().numpy()
            
            cx, cy = final_centers[i]
            
            t_x = (target_w / 2.0) - (cx * scale_x)
            t_y = (target_h / 2.0) - (cy * scale_y)
            M = np.array([[scale_x, 0, t_x], [0, scale_y, t_y]], dtype=np.float64)
            
            final_img = cv2.warpAffine(img, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(img_bg_val,)*C)
            cropped_images.append(final_img)
            
            final_mask = cv2.warpAffine(msk, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=mask_bg_val)
            cropped_masks.append(np.clip(final_mask, 0.0, 1.0))
            
            if has_opt_mask:
                o_msk = opt_mask[i].cpu().numpy()
                final_opt = cv2.warpAffine(o_msk, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=mask_bg_val)
                cropped_opt_masks.append(np.clip(final_opt, 0.0, 1.0))
            
            curr_pos = get_item_safe(i, opt_positive_points)
            curr_neg = get_item_safe(i, opt_negative_points)
            curr_box = get_item_safe(i, opt_bboxes)
            out_pos.append(transform_coords_affine(curr_pos, t_x, t_y, scale_x, scale_y, target_w, target_h, False))
            out_neg.append(transform_coords_affine(curr_neg, t_x, t_y, scale_x, scale_y, target_w, target_h, False))
            out_bbox.append(transform_coords_affine(curr_box, t_x, t_y, scale_x, scale_y, target_w, target_h, True))

            info = {
                "cx": float(cx),
                "cy": float(cy),
                "crop_shape": (box_w, box_h),
                "original_shape": (W, H)
            }
            cut_infos.append(info)
            
        cropped_tensor = torch.from_numpy(np.stack(cropped_images, 0))
        mask_tensor = torch.from_numpy(np.stack(cropped_masks, 0))
        opt_mask_tensor = torch.from_numpy(np.stack(cropped_opt_masks, 0)) if has_opt_mask else torch.zeros((B, target_h, target_w), dtype=torch.float32)
        
        def to_json_str(d):
            if not d: return ""
            try: return json.dumps(d)
            except: return ""

        return (cropped_tensor, mask_tensor, opt_mask_tensor, cut_infos, 
                to_json_str(out_pos), to_json_str(out_neg), to_json_str(out_bbox), 
                out_pos, out_neg, out_bbox)



# ==============================================================================
# Node: Wan Frame Sync Settings V5 (Mit An/Aus Schaltern)
# ==============================================================================

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



# ==============================================================================
# Node: Wan Smart Image Batcher V2
# ==============================================================================

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


# ==============================================================================
# Node: Mask Positional Joiner V20 (End-Truncation Safety)
# ==============================================================================

class MaskPositionalJoinerV20:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination_images": ("IMAGE",),
                "processed_images": ("IMAGE",),
                "cut_info": ("MASK_CUT_INFO_V14",), 
                "feather": ("INT", {"default": 10, "min": 0, "max": 256, "step": 1}),
                "padding_minus": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            },
            "optional": {
                "opt_mask_cutted": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("joined_images",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Masking"
    DESCRIPTION = "V20: Wenn Längen nicht stimmen, wird das Ende einfach hart abgeschnitten (Truncation). Kein Verdoppeln."

    def process(self, destination_images, processed_images, cut_info, feather, padding_minus, opt_mask_cutted=None):
        import cv2
        import numpy as np
        import torch
        import math
        
        dest_list = [img for img in destination_images]
        proc_list = [img for img in processed_images]
        mask_list = [m for m in opt_mask_cutted] if opt_mask_cutted is not None else None
        cut_list = list(cut_info)
        
        # --- 1. HARTER CUT AM ENDE (Safety Truncation) ---
        # Wir ermitteln die Länge des kürzeren Videos
        min_len = min(len(dest_list), len(proc_list))
        
        # Und schneiden alles, was darüber hinausgeht, einfach am Ende ab!
        if len(dest_list) > min_len:
            dest_list = dest_list[:min_len]
            cut_list = cut_list[:min_len]
            
        if len(proc_list) > min_len:
            proc_list = proc_list[:min_len]
            if mask_list is not None:
                mask_list = mask_list[:min_len]

        dest_np = torch.stack(dest_list).cpu().numpy()
        proc_np = torch.stack(proc_list).cpu().numpy()
        if mask_list is not None: opt_mask_cutted = torch.stack(mask_list)
        else: opt_mask_cutted = None
            
        B_dest = len(dest_np)
        B_proc = len(proc_np)
        
        # --- 2. JOINER LOGIK (Matrix Math) ---
        for i in range(B_dest):
            if i >= len(cut_list): break
            info = cut_list[i]
            
            proc_idx = i % B_proc
            proc_img = proc_np[proc_idx]
            
            curr_h, curr_w = proc_img.shape[:2]       
            dest_h, dest_w = dest_np[i].shape[:2]     
            
            cx, cy = info["cx"], info["cy"]
            box_w, box_h = info["crop_shape"]
            img_W, img_H = info["original_shape"]
            
            bg_scale_x = float(dest_w) / float(img_W)
            bg_scale_y = float(dest_h) / float(img_H)
            
            cx_bg = cx * bg_scale_x
            cy_bg = cy * bg_scale_y
            box_w_bg = box_w * bg_scale_x
            box_h_bg = box_h * bg_scale_y
            
            fg_scale_x = float(curr_w) / box_w_bg
            fg_scale_y = float(curr_h) / box_h_bg
            
            M_inv = np.array([
                [1.0 / fg_scale_x, 0, cx_bg - box_w_bg / 2.0],
                [0, 1.0 / fg_scale_y, cy_bg - box_h_bg / 2.0]
            ], dtype=np.float64)
            
            pm_scale_x = float(curr_w) / float(box_w)
            pm_scale_y = float(curr_h) / float(box_h)
            
            pm_x = padding_minus * pm_scale_x
            pm_y = padding_minus * pm_scale_y
            
            safe_pm_l = safe_pm_r = pm_x
            safe_pm_t = safe_pm_b = pm_y
            
            msk_img = None
            if opt_mask_cutted is not None:
                raw_msk = opt_mask_cutted[proc_idx].cpu().numpy()
                if raw_msk.shape[:2] != (curr_h, curr_w):
                    msk_img = cv2.resize(raw_msk, (curr_w, curr_h), interpolation=cv2.INTER_LINEAR)
                else:
                    msk_img = raw_msk
                    
                y_ind, x_ind = np.nonzero(msk_img > 0.1)
                if len(y_ind) > 0:
                    m_x1, m_x2 = x_ind.min(), x_ind.max()
                    m_y1, m_y2 = y_ind.min(), y_ind.max()
                    safe_pm_l = min(pm_x, m_x1)
                    safe_pm_r = min(pm_x, curr_w - m_x2)
                    safe_pm_t = min(pm_y, m_y1)
                    safe_pm_b = min(pm_y, curr_h - m_y2)
                else:
                    safe_pm_l = safe_pm_r = safe_pm_t = safe_pm_b = 0
            
            alpha = np.zeros((curr_h, curr_w), dtype=np.float32)
            
            start_x = int(math.floor(safe_pm_l))
            end_x = int(math.ceil(curr_w - safe_pm_r))
            start_y = int(math.floor(safe_pm_t))
            end_y = int(math.ceil(curr_h - safe_pm_b))
            
            if end_x > start_x and end_y > start_y:
                alpha[start_y:end_y, start_x:end_x] = 1.0
                
                if feather > 0:
                    f_x = int(feather * pm_scale_x)
                    f_y = int(feather * pm_scale_y)
                    
                    dist_l = cx_bg - box_w_bg/2.0
                    dist_r = dest_w - (cx_bg + box_w_bg/2.0)
                    dist_t = cy_bg - box_h_bg/2.0
                    dist_b = dest_h - (cy_bg + box_h_bg/2.0)
                    
                    do_fade_l = (dist_l > 0.5) or (start_x > 0)
                    do_fade_r = (dist_r > 0.5) or (end_x < curr_w)
                    do_fade_t = (dist_t > 0.5) or (start_y > 0)
                    do_fade_b = (dist_b > 0.5) or (end_y < curr_h)
                    
                    f_l = min(f_x, (end_x - start_x)//2)
                    f_r = min(f_x, (end_x - start_x)//2)
                    f_t = min(f_y, (end_y - start_y)//2)
                    f_b = min(f_y, (end_y - start_y)//2)
                    
                    if do_fade_l and f_l > 0: alpha[start_y:end_y, start_x:start_x+f_l] *= np.linspace(0, 1, f_l)[None, :]
                    if do_fade_r and f_r > 0: alpha[start_y:end_y, end_x-f_r:end_x] *= np.linspace(1, 0, f_r)[None, :]
                    if do_fade_t and f_t > 0: alpha[start_y:start_y+f_t, start_x:end_x] *= np.linspace(0, 1, f_t)[:, None]
                    if do_fade_b and f_b > 0: alpha[end_y-f_b:end_y, start_x:end_x] *= np.linspace(1, 0, f_b)[:, None]
            
            if msk_img is not None:
                alpha = np.maximum(alpha, msk_img)
            
            warped_proc = cv2.warpAffine(proc_img, M_inv, (dest_w, dest_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            warped_alpha = cv2.warpAffine(alpha, M_inv, (dest_w, dest_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            
            warped_alpha = warped_alpha[:, :, None]
            dest_np[i] = warped_proc * warped_alpha + dest_np[i] * (1.0 - warped_alpha)
            
        return (torch.from_numpy(dest_np),)

# ==============================================================================
# Node: Mask Positional Cutter V21
# ==============================================================================

class MaskPositionalCutterV21:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",), 
                "padding": ("INT", {"default": 30, "min": 0, "max": 1024, "step": 1, "tooltip": "Puffer-Rand. Wichtig für weiche Kamerafahrten!"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1, "tooltip": "Zielauflösung in MP."}),
                "camera_smooth_window": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1, "tooltip": "Gimbal Window (0 = Aus)."}),
                "smoothing_passes": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "tooltip": "Multi-Pass für butterweiche Kamera."}),
                "keep_mask_100_percent_inside": (["yes", "no"], {"default": "yes", "tooltip": "Zwingt die Kamera, die Maske nie abzuschneiden."}),
                "background_color": (["black", "white"], {"default": "black", "tooltip": "Füllfarbe am Bildrand."}),
                "disable_cutter_and_joiner": (["no", "yes"], {"default": "no", "tooltip": "Wenn 'yes', wird das Bild ohne Zuschnitt direkt durchgereicht."}),
            },
            "optional": {
                "opt_mask": ("MASK",),
                "opt_positive_points": ("*",), 
                "opt_negative_points": ("*",),
                "opt_bboxes": ("*",), 
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK_CUT_INFO_V21", "STRING", "STRING", "STRING", "*", "*", "*")
    RETURN_NAMES = ("cropped_images", "mask_cutted", "mask_cutted_opt", "cut_info", "trans_pos_json", "trans_neg_json", "trans_bbox_json", "trans_pos_raw", "trans_neg_raw", "trans_bbox_raw")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Masking"
    DESCRIPTION = "V21: Sub-Pixel Cutter mit integriertem Bypass-Schalter zum direkten Durchreichen."

    def process(self, images, mask, padding, megapixels, camera_smooth_window, smoothing_passes, keep_mask_100_percent_inside, background_color, disable_cutter_and_joiner="no", opt_mask=None, opt_positive_points=None, opt_negative_points=None, opt_bboxes=None):
        import cv2
        import numpy as np
        import torch
        import math
        import json
        
        B, H, W, C = images.shape
        
        # --- BYPASS LOGIK ---
        if disable_cutter_and_joiner == "yes":
            cut_infos = [{"bypass": True} for _ in range(B)]
            opt_mask_out = opt_mask if opt_mask is not None else torch.zeros((B, H, W), dtype=torch.float32)
            empty_json = json.dumps([])
            return (images, mask, opt_mask_out, cut_infos, empty_json, empty_json, empty_json, opt_positive_points, opt_negative_points, opt_bboxes)
        
        # --- NORMALE LOGIK ---
        if mask.ndim == 2: mask = mask.unsqueeze(0)
        if mask.shape[0] < B: mask = mask.repeat(B, 1, 1)

        has_opt_mask = False
        if opt_mask is not None:
            has_opt_mask = True
            if opt_mask.ndim == 2: opt_mask = opt_mask.unsqueeze(0)
            if opt_mask.shape[0] < B: opt_mask = opt_mask.repeat(B, 1, 1)
        
        raw_centers = []
        mask_bounds = [] 
        global_max_w = 0
        global_max_h = 0
        
        for i in range(B):
            m = mask[i].cpu().numpy()
            y_indices, x_indices = np.nonzero(m > 0.5)
            if len(y_indices) == 0:
                raw_centers.append(None)
                mask_bounds.append(None)
            else:
                x1, x2 = x_indices.min(), x_indices.max()
                y1, y2 = y_indices.min(), y_indices.max()
                w = x2 - x1
                h = y2 - y1
                if w > global_max_w: global_max_w = w
                if h > global_max_h: global_max_h = h
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                raw_centers.append((cx, cy))
                mask_bounds.append((x1, y1, x2, y2))

        if global_max_w == 0: global_max_w = 128
        if global_max_h == 0: global_max_h = 128
        
        box_w = int(global_max_w + (padding * 2))
        box_h = int(global_max_h + (padding * 2))
        
        valid_centers = []
        last_valid = (W/2.0, H/2.0)
        for c in raw_centers:
            if c is not None: last_valid = c
            valid_centers.append(last_valid)
            
        valid_bounds = []
        last_b = (W/2.0-10, H/2.0-10, W/2.0+10, H/2.0+10)
        for b in mask_bounds:
            if b is not None: last_b = b
            valid_bounds.append(last_b)

        def smooth_series(series, passes, window):
            if window <= 0: return series
            res = series.copy()
            for _ in range(passes):
                temp = []
                for i in range(len(res)):
                    s = max(0, i - window)
                    e = min(len(res), i + window + 1)
                    w_slice = res[s:e]
                    avg = tuple(sum(val[k] for val in w_slice) / len(w_slice) for k in range(len(res[0])))
                    temp.append(avg)
                res = temp
            return res

        smoothed_centers = smooth_series(valid_centers, smoothing_passes, camera_smooth_window)
        smoothed_bounds = smooth_series(valid_bounds, smoothing_passes, camera_smooth_window)
        
        final_centers = []
        for i in range(B):
            cx, cy = smoothed_centers[i]
            if keep_mask_100_percent_inside == "yes":
                x1, y1, x2, y2 = smoothed_bounds[i]
                min_cx = x2 - box_w / 2.0
                max_cx = x1 + box_w / 2.0
                min_cy = y2 - box_h / 2.0
                max_cy = y1 + box_h / 2.0
                if min_cx <= max_cx: cx = max(min_cx, min(max_cx, cx))
                if min_cy <= max_cy: cy = max(min_cy, min(max_cy, cy))
            final_centers.append((cx, cy))

        target_pixel_count = megapixels * 1_000_000
        aspect_ratio = box_w / box_h
        target_h_float = math.sqrt(target_pixel_count / aspect_ratio)
        target_w_float = target_h_float * aspect_ratio
        target_w = int(round(target_w_float / 8) * 8)
        target_h = int(round(target_h_float / 8) * 8)
        target_w = max(64, target_w)
        target_h = max(64, target_h)
        
        scale_x = target_w / float(box_w)
        scale_y = target_h / float(box_h)
        
        img_bg_val = 0.0 if background_color == "black" else 1.0
        mask_bg_val = 0.0
        
        def get_item_safe(idx, data):
            if data is None: return None
            if isinstance(data, str):
                try: data = json.loads(data)
                except: pass
            if isinstance(data, (list, tuple)):
                if len(data) == 0: return None
                return data[idx] if idx < len(data) else data[-1]
            if hasattr(data, "shape"):
                if data.shape[0] == 0: return None
                return data[idx] if idx < data.shape[0] else data[-1]
            return data

        def transform_coords_affine(coords, t_x, t_y, sx, sy, limit_w, limit_h, is_bbox=False):
            if coords is None: return []
            is_tensor = hasattr(coords, "cpu")
            if is_tensor: pts = coords.cpu().numpy()
            else: pts = np.array(coords)
            if pts.size == 0 or pts.ndim == 0: return []
            new_pts = []
            if is_bbox:
                if pts.ndim == 1 and len(pts) == 4: pts = pts.reshape(1, 4)
                elif pts.ndim == 1: return []
                for b in pts:
                    if len(b) < 4: continue
                    nx1, ny1 = b[0]*sx + t_x, b[1]*sy + t_y
                    nx2, ny2 = b[2]*sx + t_x, b[3]*sy + t_y
                    nx1, ny1 = max(0, min(limit_w, nx1)), max(0, min(limit_h, ny1))
                    nx2, ny2 = max(0, min(limit_w, nx2)), max(0, min(limit_h, ny2))
                    if nx2>nx1 and ny2>ny1: new_pts.append([float(nx1), float(ny1), float(nx2), float(ny2)])
            else:
                if pts.ndim == 1 and len(pts) == 2: pts = pts.reshape(1, 2)
                elif pts.ndim == 1: return []
                for p in pts:
                    if len(p) < 2: continue
                    nx, ny = p[0]*sx + t_x, p[1]*sy + t_y
                    if 0<=nx<limit_w and 0<=ny<limit_h: new_pts.append([float(nx), float(ny)])
            return new_pts

        cropped_images, cropped_masks, cropped_opt_masks = [], [], []
        cut_infos, out_pos, out_neg, out_bbox = [], [], [], []

        for i in range(B):
            img = images[i].cpu().numpy()
            msk = mask[i].cpu().numpy()
            
            cx, cy = final_centers[i]
            
            t_x = (target_w / 2.0) - (cx * scale_x)
            t_y = (target_h / 2.0) - (cy * scale_y)
            M = np.array([[scale_x, 0, t_x], [0, scale_y, t_y]], dtype=np.float64)
            
            final_img = cv2.warpAffine(img, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(img_bg_val,)*C)
            cropped_images.append(final_img)
            
            final_mask = cv2.warpAffine(msk, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=mask_bg_val)
            cropped_masks.append(np.clip(final_mask, 0.0, 1.0))
            
            if has_opt_mask:
                o_msk = opt_mask[i].cpu().numpy()
                final_opt = cv2.warpAffine(o_msk, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=mask_bg_val)
                cropped_opt_masks.append(np.clip(final_opt, 0.0, 1.0))
            
            curr_pos = get_item_safe(i, opt_positive_points)
            curr_neg = get_item_safe(i, opt_negative_points)
            curr_box = get_item_safe(i, opt_bboxes)
            out_pos.append(transform_coords_affine(curr_pos, t_x, t_y, scale_x, scale_y, target_w, target_h, False))
            out_neg.append(transform_coords_affine(curr_neg, t_x, t_y, scale_x, scale_y, target_w, target_h, False))
            out_bbox.append(transform_coords_affine(curr_box, t_x, t_y, scale_x, scale_y, target_w, target_h, True))

            info = {
                "cx": float(cx),
                "cy": float(cy),
                "crop_shape": (box_w, box_h),
                "original_shape": (W, H)
            }
            cut_infos.append(info)
            
        cropped_tensor = torch.from_numpy(np.stack(cropped_images, 0))
        mask_tensor = torch.from_numpy(np.stack(cropped_masks, 0))
        opt_mask_tensor = torch.from_numpy(np.stack(cropped_opt_masks, 0)) if has_opt_mask else torch.zeros((B, target_h, target_w), dtype=torch.float32)
        
        def to_json_str(d):
            if not d: return ""
            try: return json.dumps(d)
            except: return ""

        return (cropped_tensor, mask_tensor, opt_mask_tensor, cut_infos, 
                to_json_str(out_pos), to_json_str(out_neg), to_json_str(out_bbox), 
                out_pos, out_neg, out_bbox)


# ==============================================================================
# Node: Mask Positional Joiner V21
# ==============================================================================

class MaskPositionalJoinerV21:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination_images": ("IMAGE",),
                "processed_images": ("IMAGE",),
                "cut_info": ("MASK_CUT_INFO_V21",), 
                "feather": ("INT", {"default": 10, "min": 0, "max": 256, "step": 1}),
                "padding_minus": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            },
            "optional": {
                "opt_mask_cutted": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("joined_images",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Masking"
    DESCRIPTION = "V21: Nimmt das Bild einfach unberührt entgegen, falls Bypass aktiviert wurde."

    def process(self, destination_images, processed_images, cut_info, feather, padding_minus, opt_mask_cutted=None):
        import cv2
        import numpy as np
        import torch
        import math
        
        # --- BYPASS LOGIK ---
        if cut_info and isinstance(cut_info[0], dict) and cut_info[0].get("bypass", False):
            # Wir geben das generierte Bild einfach direkt durch (ignoriert destination_images komplett)
            return (processed_images,)

        # --- NORMALE LOGIK ---
        dest_list = [img for img in destination_images]
        proc_list = [img for img in processed_images]
        mask_list = [m for m in opt_mask_cutted] if opt_mask_cutted is not None else None
        cut_list = list(cut_info)
        
        min_len = min(len(dest_list), len(proc_list))
        
        if len(dest_list) > min_len:
            dest_list = dest_list[:min_len]
            cut_list = cut_list[:min_len]
            
        if len(proc_list) > min_len:
            proc_list = proc_list[:min_len]
            if mask_list is not None:
                mask_list = mask_list[:min_len]

        dest_np = torch.stack(dest_list).cpu().numpy()
        proc_np = torch.stack(proc_list).cpu().numpy()
        if mask_list is not None: opt_mask_cutted = torch.stack(mask_list)
        else: opt_mask_cutted = None
            
        B_dest = len(dest_np)
        B_proc = len(proc_np)
        
        for i in range(B_dest):
            if i >= len(cut_list): break
            info = cut_list[i]
            
            proc_idx = i % B_proc
            proc_img = proc_np[proc_idx]
            
            curr_h, curr_w = proc_img.shape[:2]       
            dest_h, dest_w = dest_np[i].shape[:2]     
            
            cx, cy = info["cx"], info["cy"]
            box_w, box_h = info["crop_shape"]
            img_W, img_H = info["original_shape"]
            
            bg_scale_x = float(dest_w) / float(img_W)
            bg_scale_y = float(dest_h) / float(img_H)
            
            cx_bg = cx * bg_scale_x
            cy_bg = cy * bg_scale_y
            box_w_bg = box_w * bg_scale_x
            box_h_bg = box_h * bg_scale_y
            
            fg_scale_x = float(curr_w) / box_w_bg
            fg_scale_y = float(curr_h) / box_h_bg
            
            M_inv = np.array([
                [1.0 / fg_scale_x, 0, cx_bg - box_w_bg / 2.0],
                [0, 1.0 / fg_scale_y, cy_bg - box_h_bg / 2.0]
            ], dtype=np.float64)
            
            pm_scale_x = float(curr_w) / float(box_w)
            pm_scale_y = float(curr_h) / float(box_h)
            
            pm_x = padding_minus * pm_scale_x
            pm_y = padding_minus * pm_scale_y
            
            safe_pm_l = safe_pm_r = pm_x
            safe_pm_t = safe_pm_b = pm_y
            
            msk_img = None
            if opt_mask_cutted is not None:
                raw_msk = opt_mask_cutted[proc_idx].cpu().numpy()
                if raw_msk.shape[:2] != (curr_h, curr_w):
                    msk_img = cv2.resize(raw_msk, (curr_w, curr_h), interpolation=cv2.INTER_LINEAR)
                else:
                    msk_img = raw_msk
                    
                y_ind, x_ind = np.nonzero(msk_img > 0.1)
                if len(y_ind) > 0:
                    m_x1, m_x2 = x_ind.min(), x_ind.max()
                    m_y1, m_y2 = y_ind.min(), y_ind.max()
                    safe_pm_l = min(pm_x, m_x1)
                    safe_pm_r = min(pm_x, curr_w - m_x2)
                    safe_pm_t = min(pm_y, m_y1)
                    safe_pm_b = min(pm_y, curr_h - m_y2)
                else:
                    safe_pm_l = safe_pm_r = safe_pm_t = safe_pm_b = 0
            
            alpha = np.zeros((curr_h, curr_w), dtype=np.float32)
            
            start_x = int(math.floor(safe_pm_l))
            end_x = int(math.ceil(curr_w - safe_pm_r))
            start_y = int(math.floor(safe_pm_t))
            end_y = int(math.ceil(curr_h - safe_pm_b))
            
            if end_x > start_x and end_y > start_y:
                alpha[start_y:end_y, start_x:end_x] = 1.0
                
                if feather > 0:
                    f_x = int(feather * pm_scale_x)
                    f_y = int(feather * pm_scale_y)
                    
                    dist_l = cx_bg - box_w_bg/2.0
                    dist_r = dest_w - (cx_bg + box_w_bg/2.0)
                    dist_t = cy_bg - box_h_bg/2.0
                    dist_b = dest_h - (cy_bg + box_h_bg/2.0)
                    
                    do_fade_l = (dist_l > 0.5) or (start_x > 0)
                    do_fade_r = (dist_r > 0.5) or (end_x < curr_w)
                    do_fade_t = (dist_t > 0.5) or (start_y > 0)
                    do_fade_b = (dist_b > 0.5) or (end_y < curr_h)
                    
                    f_l = min(f_x, (end_x - start_x)//2)
                    f_r = min(f_x, (end_x - start_x)//2)
                    f_t = min(f_y, (end_y - start_y)//2)
                    f_b = min(f_y, (end_y - start_y)//2)
                    
                    if do_fade_l and f_l > 0: alpha[start_y:end_y, start_x:start_x+f_l] *= np.linspace(0, 1, f_l)[None, :]
                    if do_fade_r and f_r > 0: alpha[start_y:end_y, end_x-f_r:end_x] *= np.linspace(1, 0, f_r)[None, :]
                    if do_fade_t and f_t > 0: alpha[start_y:start_y+f_t, start_x:end_x] *= np.linspace(0, 1, f_t)[:, None]
                    if do_fade_b and f_b > 0: alpha[end_y-f_b:end_y, start_x:end_x] *= np.linspace(1, 0, f_b)[:, None]
            
            if msk_img is not None:
                alpha = np.maximum(alpha, msk_img)
            
            warped_proc = cv2.warpAffine(proc_img, M_inv, (dest_w, dest_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            warped_alpha = cv2.warpAffine(alpha, M_inv, (dest_w, dest_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            
            warped_alpha = warped_alpha[:, :, None]
            dest_np[i] = warped_proc * warped_alpha + dest_np[i] * (1.0 - warped_alpha)
            
        return (torch.from_numpy(dest_np),)


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


class PoseDataToMaskV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1}),
                "stick_width": ("INT", {
                    "default": 10, 
                    "min": 1, 
                    "max": 300, 
                    "step": 1,
                    "display": "slider",
                    "slider_max": 200 
                }),
                "head_circle_px": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 1.0, "tooltip": "Fester Radius für den Kopfkreis in Pixeln."}),
                "head_circle_norm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Normierter Radius relativ zur Schulterbreite (passt sich pro Frame dynamisch an)."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Erstellt Maske (V2). Füllt Körper & Stirn. Fügt optional einen Kreis für den Kopf hinzu und zeichnet das Skelett (Bones)."

    def process(self, pose_data, width, height, stick_width, head_circle_px, head_circle_norm):
        pose_metas = pose_data["pose_metas"]
        mask_list = []

        # Speicher für Fallback (falls eine Detection fehlschlägt)
        last_valid_mask = np.zeros((height, width), dtype=np.float32)

        for meta in pose_metas:
            kps = meta.kps_body
            scores = meta.kps_body_p
            
            def get_pt(idx):
                if idx < len(scores) and scores[idx] > 0.3:
                    return (int(kps[idx][0]), int(kps[idx][1]))
                return None

            # Schultern prüfen
            p_lsh, p_rsh = get_pt(5), get_pt(2)

            if p_lsh and p_rsh:
                # === NEU ZEICHNEN ===
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                p_lhip, p_rhip = get_pt(11), get_pt(8)

                # 1. TORSO (gefüllte Fläche)
                if p_lhip and p_rhip:
                    pts_torso = np.array([p_lsh, p_rsh, p_rhip, p_lhip], np.int32)
                    cv2.fillPoly(canvas, [pts_torso], (255, 255, 255))
                else:
                    p_r_bottom = (p_rsh[0], height)
                    p_l_bottom = (p_lsh[0], height)
                    pts_torso = np.array([p_lsh, p_rsh, p_r_bottom, p_l_bottom], np.int32)
                    cv2.fillPoly(canvas, [pts_torso], (255, 255, 255))

                # 2. KOPF & STIRN (Basis-Polygon)
                head_pts = []
                p_nose = get_pt(0)
                p_lear = get_pt(17) or get_pt(15) 
                p_rear = get_pt(16) or get_pt(14) 

                if p_nose: head_pts.append(p_nose)
                if p_lear: head_pts.append(p_lear)
                head_pts.append(p_lsh)
                head_pts.append(p_rsh)
                if p_rear: head_pts.append(p_rear)
                
                if len(head_pts) >= 3:
                    pts_head = np.array(head_pts, np.int32)
                    cv2.fillPoly(canvas, [pts_head], (255, 255, 255))

                    # Stirn-Rechteck
                    if p_lear and p_rear:
                        eye_y = (p_lear[1] + p_rear[1]) / 2
                        shoulder_y = (p_lsh[1] + p_rsh[1]) / 2
                        dist_head_shoulder = abs(shoulder_y - eye_y)
                        forehead_height = int(dist_head_shoulder * 0.65)
                        x_min = min(p_lear[0], p_rear[0])
                        x_max = max(p_lear[0], p_rear[0])
                        y_bottom = int(eye_y)
                        y_top = int(eye_y - forehead_height)
                        pt1 = (x_min, y_bottom); pt2 = (x_max, y_bottom)
                        pt3 = (x_max, y_top); pt4 = (x_min, y_top)
                        pts_forehead = np.array([pt1, pt2, pt3, pt4], np.int32)
                        cv2.fillPoly(canvas, [pts_forehead], (255, 255, 255))

                # --- 3. KREIS FÜR DEN KOPF (V2 FEATURE) ---
                if head_circle_px > 0 or head_circle_norm > 0:
                    center_x, center_y = None, None
                    if p_lear and p_rear:
                        center_x = int((p_lear[0] + p_rear[0]) / 2)
                        center_y = int((p_lear[1] + p_rear[1]) / 2)
                    elif p_nose:
                        center_x, center_y = p_nose
                    
                    if center_x is not None and center_y is not None:
                        shoulder_width = abs(p_lsh[0] - p_rsh[0])
                        radius = int(head_circle_px + head_circle_norm * shoulder_width)
                        if radius > 0:
                            cv2.circle(canvas, (center_x, center_y), radius, (255, 255, 255), -1)

                # --- 4. SKELETT (DIE FEHLENDEN BONES) ---
                skeleton_canvas = np.zeros_like(canvas)
                skeleton_img = draw_aapose_by_meta_new(
                    skeleton_canvas, 
                    meta, 
                    draw_hand=True, 
                    draw_head=True,
                    body_stick_width=stick_width,
                    hand_stick_width=max(1, stick_width // 2)
                )

                # --- 5. MERGE ---
                # Wandle beides in Graustufen um und kombiniere sie
                filled_mask_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                skeleton_mask_gray = cv2.cvtColor(skeleton_img, cv2.COLOR_RGB2GRAY)
                final_mask_combined = cv2.bitwise_or(filled_mask_gray, skeleton_mask_gray)
                
                # Update den Fallback-Speicher
                last_valid_mask = (final_mask_combined > 0).astype(np.float32)

            else:
                # FALLBACK (Wenn Schultern nicht erkannt werden -> nimm die Maske vom letzten Frame)
                pass 

            mask_tensor = torch.from_numpy(last_valid_mask)
            mask_list.append(mask_tensor)

        return (torch.stack(mask_list, dim=0),)

class PoseDataSelectFrameNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "frame_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 99999, 
                    "step": 1, 
                    "tooltip": "Gibt den Index des gewünschten Frames an (0 = 1. Frame, 9 = 10. Frame etc.)"
                }),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Wählt einen einzelnen Frame aus einer PoseData-Sequenz anhand des Index aus."

    def process(self, pose_data, frame_index):
        import copy
        # Erstelle eine tiefe Kopie, damit das Original nicht verändert wird
        new_pose_data = copy.deepcopy(pose_data)
        
        # Hole die Metadaten der Posen
        if "pose_metas" in new_pose_data and new_pose_data["pose_metas"]:
            metas = new_pose_data["pose_metas"]
            # Stelle sicher, dass der Index nicht außerhalb der Reichweite liegt
            safe_index = min(frame_index, len(metas) - 1)
            safe_index = max(0, safe_index)
            # Behalte nur den einen ausgewählten Frame
            new_pose_data["pose_metas"] = [metas[safe_index]]
            
        if "pose_metas_original" in new_pose_data and new_pose_data["pose_metas_original"]:
            metas_orig = new_pose_data["pose_metas_original"]
            safe_index_orig = min(frame_index, len(metas_orig) - 1)
            safe_index_orig = max(0, safe_index_orig)
            new_pose_data["pose_metas_original"] = [metas_orig[safe_index_orig]]

        return (new_pose_data,)

class LoadPoseDataFromJsonNode:
    @classmethod
    def INPUT_TYPES(s):
        import os
        import folder_paths
        
        # Hole alle JSON-Dateien aus dem ComfyUI 'input' Ordner
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        
        # Falls keine Dateien existieren, packe einen Dummy-Eintrag rein, um Fehler zu vermeiden
        if not files:
            files = ["Keine JSON gefunden. Bitte JSON in den 'input' Ordner legen."]
            
        return {
            "required": {
                "json_file": (files, {"tooltip": "Wähle eine JSON-Datei aus dem ComfyUI 'input'-Ordner"}),
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "load_json"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Lädt Pose-Daten aus einer JSON-Datei, die zuvor gespeichert wurde."

    def load_json(self, json_file):
        import json
        import os
        import numpy as np
        import folder_paths
        from .pose_utils.pose2d_utils import AAPoseMeta

        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, json_file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Die Datei {file_path} existiert nicht.")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Unterstützt verschiedene Speicherformate (Liste von Metas oder Dict mit 'pose_metas' Key)
        meta_list = data.get("pose_metas", data) if isinstance(data, dict) else data

        restored_metas = []
        for item in meta_list:
            if not isinstance(item, dict): 
                continue
            
            # Ein leeres Meta-Objekt erstellen
            meta = AAPoseMeta(None)
            
            meta.image_id = item.get("image_id", "")
            meta.height = item.get("height", 0)
            meta.width = item.get("width", 0)
            
            # Hilfsfunktion, um Numpy-Arrays wiederherzustellen
            def restore_array(val):
                if val is None: return None
                if isinstance(val, str) and "<Image" in val: return None
                return np.array(val, dtype=np.float32)

            meta.kps_body = restore_array(item.get("kps_body"))
            meta.kps_body_p = restore_array(item.get("kps_body_p"))
            meta.kps_lhand = restore_array(item.get("kps_lhand"))
            meta.kps_lhand_p = restore_array(item.get("kps_lhand_p"))
            meta.kps_rhand = restore_array(item.get("kps_rhand"))
            meta.kps_rhand_p = restore_array(item.get("kps_rhand_p"))
            meta.kps_face = restore_array(item.get("kps_face"))
            meta.kps_face_p = restore_array(item.get("kps_face_p"))
            
            restored_metas.append(meta)

        pose_data = {
            "retarget_image": None,
            "pose_metas": restored_metas,
            "refer_pose_meta": None,
            "pose_metas_original": restored_metas,
        }

        return (pose_data,)




class PoseDataToDWPoses:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
            }
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Konvertiert dein PoseData direkt in DWPoses für SCAIL."

    def process(self, pose_data):
        pose_metas = pose_data.get("pose_metas", [])
        results = []
        
        for meta in pose_metas:
            # Dimensionen für die Normalisierung holen (SCAIL erwartet 0.0 - 1.0)
            width = getattr(meta, "width", 512) or 512
            height = getattr(meta, "height", 512) or 512

            # Überprüfen, ob meta ein Dictionary oder eine AAPoseMeta Instanz ist
            is_dict = isinstance(meta, dict)

            if is_dict:
                kps_body = np.array(meta.get("keypoints_body", []))
                kps_lhand = np.array(meta.get("keypoints_left_hand", []))
                kps_rhand = np.array(meta.get("keypoints_right_hand", []))
                kps_face = np.array(meta.get("keypoints_face", []))

                # Die letzten 2 Punkte beim Körper weglassen (20 -> 18)
                candidate_body = kps_body[:-2, :2] if len(kps_body) > 2 else np.zeros((18, 2))
                score_body = kps_body[:-2, 2] if len(kps_body) > 2 else np.zeros((18,))
                
                lhand_coords = kps_lhand[:, :2] if len(kps_lhand) > 0 else np.zeros((21, 2))
                lhand_score = kps_lhand[:, 2] if len(kps_lhand) > 0 else np.zeros((21,))
                
                rhand_coords = kps_rhand[:, :2] if len(kps_rhand) > 0 else np.zeros((21, 2))
                rhand_score = kps_rhand[:, 2] if len(kps_rhand) > 0 else np.zeros((21,))

                # Bei Gesicht den ersten Punkt auslassen (69 -> 68)
                face_coords = kps_face[1:, :2] if len(kps_face) > 1 else np.zeros((68, 2))
                face_score = kps_face[1:, 2] if len(kps_face) > 1 else np.zeros((68,))

            else:
                # Meta ist eine AAPoseMeta Instanz (Koordinaten sind meist noch in Pixeln)
                b_coords = getattr(meta, "kps_body", None)
                if b_coords is not None and len(b_coords) > 2:
                    candidate_body = b_coords[:-2].copy()
                    candidate_body[:, 0] /= width
                    candidate_body[:, 1] /= height
                    score_body = getattr(meta, "kps_body_p")[:-2]
                else:
                    candidate_body = np.zeros((18, 2))
                    score_body = np.zeros((18,))

                # Linke Hand
                lh_coords = getattr(meta, "kps_lhand", None)
                if lh_coords is not None and len(lh_coords) > 0:
                    lhand_coords = lh_coords.copy()
                    lhand_coords[:, 0] /= width
                    lhand_coords[:, 1] /= height
                    lhand_score = getattr(meta, "kps_lhand_p", np.zeros((21,)))
                else:
                    lhand_coords = np.zeros((21, 2))
                    lhand_score = np.zeros((21,))

                # Rechte Hand
                rh_coords = getattr(meta, "kps_rhand", None)
                if rh_coords is not None and len(rh_coords) > 0:
                    rhand_coords = rh_coords.copy()
                    rhand_coords[:, 0] /= width
                    rhand_coords[:, 1] /= height
                    rhand_score = getattr(meta, "kps_rhand_p", np.zeros((21,)))
                else:
                    rhand_coords = np.zeros((21, 2))
                    rhand_score = np.zeros((21,))

                # Gesicht
                f_coords = getattr(meta, "kps_face", None)
                if f_coords is not None and len(f_coords) > 1:
                    face_coords = f_coords[1:].copy()
                    face_coords[:, 0] /= width
                    face_coords[:, 1] /= height
                    face_score = getattr(meta, "kps_face_p", np.zeros((69,)))[1:]
                else:
                    face_coords = np.zeros((68, 2))
                    face_score = np.zeros((68,))

            # Subset Matrix aufbauen: -1 wenn Score unter Threshold von 0.3 ist
            subset_body = np.arange(len(candidate_body), dtype=float)
            subset_body[score_body <= 0.3] = -1

            # Hände wie in DWPose/SCAIL stacken (Rechts, dann Links!)
            hands_coords = np.stack([rhand_coords, lhand_coords], axis=0)
            hands_score = np.stack([rhand_score, lhand_score], axis=0)

            # Dictionary im Zielformat für DWPose / SCAIL bauen
            dwpose_format = {
                "bodies": {
                    "candidate": np.expand_dims(candidate_body, axis=0).astype(np.float32),
                    "subset": np.expand_dims(subset_body, axis=0).astype(np.float32)
                },
                "hands": hands_coords.astype(np.float32),
                "faces": np.expand_dims(face_coords, axis=0).astype(np.float32),
                "body_score": np.expand_dims(score_body, axis=0).astype(np.float32),
                "hand_score": hands_score.astype(np.float32),
                "face_score": np.expand_dims(face_score, axis=0).astype(np.float32)
            }
            results.append(dwpose_format)

        out_dict = {'poses': results, 'swap_hands': True}
        return (out_dict,)

import logging

# --- HIER IST DEIN HILFSSKRIPT ---
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
# ---------------------------------


class RenderNLFPosesWithData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Der 3D Output aus SCAIL"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096})
            },
            "optional": {
                "dw_poses": ("DWPOSES",),
                "ref_dw_pose": ("DWPOSES",),
                "draw_face": ("BOOLEAN", {"default": True}),
                "draw_hand": ("BOOLEAN", {"default": True}),
                "draw_body": ("BOOLEAN", {"default": True}),
                # Diese Settings braucht das Rendern intern:
                "render_device": (["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"], {"default": "gpu"}),
                "scale_hands": ("BOOLEAN", {"default": True}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "NLFPRED")
    RETURN_NAMES = ("image", "mask", "nlf_poses_data")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Rendert die NLF Poses komplett eigenständig und reicht die Daten an den Scaler weiter."

    def process(self, nlf_poses, width, height, dw_poses=None, ref_dw_pose=None, draw_face=True, draw_hand=True, draw_body=True, render_device="gpu", scale_hands=True, render_backend="taichi"):
        
        # Imports aus dem NLF-Ordner
        from .NLFPoseExtract.nlf_render import render_nlf_as_images, render_multi_nlf_as_images, shift_dwpose_according_to_nlf, process_data_to_COCO_format, intrinsic_matrix_from_field_of_view
        from .NLFPoseExtract.align3d import solve_new_camera_params_central, solve_new_camera_params_down
        
        if render_backend == "taichi":
            try:
                import taichi as ti
                device_map = {
                    "cpu": ti.cpu, "gpu": ti.gpu, "opengl": ti.opengl,
                    "cuda": ti.cuda, "vulkan": ti.vulkan, "metal": ti.metal,
                }
                ti.init(arch=device_map.get(render_device.lower()))
            except:
                logging.warning("Taichi selected but not installed. Falling back to torch rendering.")
                render_backend = "torch"

        if isinstance(nlf_poses, dict):
            pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
        else:
            pose_input = nlf_poses

        dw_pose_input = copy.deepcopy(dw_poses["poses"]) if dw_poses is not None else None
        swap_hands = dw_poses.get("swap_hands", False) if dw_poses is not None else False

        ori_camera_pose = intrinsic_matrix_from_field_of_view([height, width])
        ori_focal = ori_camera_pose[0, 0]

        num_people = dw_pose_input[0]['bodies']['candidate'].shape[0] if dw_poses is not None else 0

        # Alignment
        if dw_poses is not None and ref_dw_pose is not None and num_people == 1:
            ref_dw_pose_input = copy.deepcopy(ref_dw_pose["poses"])

            pose_3d_first_driving_frame = None
            for pose in pose_input:
                if pose.shape[0] == 0:
                    continue
                candidate = pose[0].cpu().numpy()
                if np.any(candidate):
                    pose_3d_first_driving_frame = candidate
                    break
            if pose_3d_first_driving_frame is None:
                raise ValueError("No valid pose found in pose_input.")

            pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)
            poses_2d_ref = ref_dw_pose_input[0]['bodies']['candidate'][0][:14]
            poses_2d_ref[:, 0] = poses_2d_ref[:, 0] * width
            poses_2d_ref[:, 1] = poses_2d_ref[:, 1] * height

            poses_2d_subset = ref_dw_pose_input[0]['bodies']['subset'][0][:14]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[:14]

            valid_indices, valid_upper_indices, valid_lower_indices = [], [], []
            upper_body_indices = [0, 2, 3, 5, 6]
            lower_body_indices = [9, 10, 12, 13]

            for i in range(len(poses_2d_subset)):
                if poses_2d_subset[i] != -1.0 and np.sum(pose_3d_coco_first_driving_frame[i]) != 0:
                    if i in upper_body_indices:
                        valid_upper_indices.append(i)
                    if i in lower_body_indices:
                        valid_lower_indices.append(i)

            valid_indices = [1] + valid_lower_indices if len(valid_upper_indices) < 4 else [1] + valid_lower_indices + valid_upper_indices 

            pose_2d_ref = poses_2d_ref[valid_indices]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[valid_indices]

            if len(valid_lower_indices) >= 4:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_down(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)
            else:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_central(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)

            # HIER WIRD DIE HILFSFUNKTION AUFGERUFEN
            scale_face = scale_faces(list(dw_pose_input), list(ref_dw_pose_input))   

            logging.info(f"Scale - m: {scale_m}, face: {scale_face}")
            shift_dwpose_according_to_nlf(pose_input, dw_pose_input, ori_camera_pose, new_camera_intrinsics, height, width, swap_hands=swap_hands, scale_hands=scale_hands, scale_x=scale_m, scale_y=scale_m*scale_s)

            intrinsic_matrix = new_camera_intrinsics
        else:
            intrinsic_matrix = ori_camera_pose

        # Rendern
        if pose_input[0].shape[0] > 1:
            frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hand, render_backend=render_backend)
        else:
            frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hand, render_backend=render_backend)

        # Tensor generieren
        frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
        frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5

        return (frames_tensor.cpu().float(), mask.cpu().float(), nlf_poses)


import os
import json
import folder_paths

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




# ======================================================================
# 3. V10B: LOCAL BONE RETARGETER (MIT LOG-AUSGANG)
# ======================================================================
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


# ======================================================================
# 1. KALIBRIERUNG V9 (L2-NORM TORSO-BASIERT)
# ======================================================================
class RetargetPoseCalibratorV9:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_close": ("POSEDATA",),
                "depth_map_close": ("IMAGE",),
                "pose_far": ("POSEDATA",),
                "depth_map_far": ("IMAGE",),
            },
            "optional": {
                "nlf_close": ("NLFPRED", {"tooltip": "3D Daten Frame nah (Fallback)"}),
                "nlf_far": ("NLFPRED", {"tooltip": "3D Daten Frame fern (Empfohlen)"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V9: Nutzt L2-Norm (Torso-Länge) anstatt Bounding Box für extrem stabile Messung."

    def process(self, pose_close, depth_map_close, pose_far, depth_map_far, nlf_close=None, nlf_far=None):
        import math
        import numpy as np
        
        log_messages = ["=== V9 CALIBRATION LOG (NORM-BASED) ==="]
        
        meta_close = pose_close.get("pose_metas", [])[0] if pose_close.get("pose_metas") else None
        meta_far = pose_far.get("pose_metas", [])[0] if pose_far.get("pose_metas") else None

        if not meta_close or not meta_far:
            return ({"perspective_slope": 0.0, "perspective_intercept": 1.0}, "FEHLER: Posen fehlen!")

        kps_c = getattr(meta_close, "kps_body", [])
        kps_f = getattr(meta_far, "kps_body", [])

        # --- DIE NORM-FUNKTION ---
        def get_torso_norm(kps):
            if len(kps) < 12: return 0.0
            neck = kps[1]
            r_hip = kps[8]
            l_hip = kps[11]
            # Prüfen ob Punkte sichtbar sind
            if len(neck) >= 2 and neck[1] > 0 and len(r_hip) >= 2 and r_hip[1] > 0 and len(l_hip) >= 2 and l_hip[1] > 0:
                mid_x = (r_hip[0] + l_hip[0]) / 2.0
                mid_y = (r_hip[1] + l_hip[1]) / 2.0
                # L2 Norm (Euklidische Distanz)
                return math.sqrt((neck[0] - mid_x)**2 + (neck[1] - mid_y)**2)
            return 0.0

        norm_close = get_torso_norm(kps_c)
        norm_far = get_torso_norm(kps_f)

        log_messages.append(f"Torso-Norm Nah: {norm_close:.1f} px")
        log_messages.append(f"Torso-Norm Fern: {norm_far:.1f} px")

        if norm_close <= 0 or norm_far <= 0:
            return ({"perspective_slope": 0.0, "perspective_intercept": 1.0}, "FEHLER: Torso (Hals/Hüfte) nicht sichtbar für Norm-Berechnung!")

        def get_robust_depth(kps, depth_map):
            # Misst Tiefe nur genau auf dem Torso, filtert Hände weg
            valid_x = [kps[1][0], kps[8][0], kps[11][0]]
            valid_y = [kps[1][1], kps[8][1], kps[11][1]]
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]
            min_x, max_x = int(max(0, min(valid_x))), int(min(W-1, max(valid_x)))
            min_y, max_y = int(max(0, min(valid_y))), int(min(H-1, max(valid_y)))
            if max_x > min_x and max_y > min_y:
                return float(np.mean(depth_np[0, min_y:max_y, min_x:max_x]))
            return 0.5

        depth_c = get_robust_depth(kps_c, depth_map_close)
        depth_f = get_robust_depth(kps_f, depth_map_far)

        log_messages.append(f"KI-Roh-Tiefe Torso Nah: {depth_c:.4f} | Fern: {depth_f:.4f}")

        # --- FIX: AUTO INVERT ---
        is_inverted = False
        if depth_c > depth_f and norm_close > norm_far:
            is_inverted = True
            log_messages.append("-> KI-Tiefe ist spiegelverkehrt! Wandle in echte Distanz um...")
            depth_c = 1.0 / max(depth_c, 0.0001)
            depth_f = 1.0 / max(depth_f, 0.0001)
            log_messages.append(f"Echte Distanz Nah: {depth_c:.4f} | Fern: {depth_f:.4f}")

        slope, intercept = 0.0, 1.0
        depth_diff = abs(depth_f - depth_c)
        if depth_diff > 0.05:
            slope = (norm_far - norm_close) / (depth_f - depth_c)
            intercept = norm_close - (slope * depth_c)
        else:
            slope = -500.0 if is_inverted else 500.0
            intercept = norm_close - (slope * depth_c)

        log_messages.append(f"\nPhysikalischer NORM-Slope: {slope:.2f}")

        true_3d_bones = {}
        target_nlf = nlf_far if nlf_far is not None else nlf_close
        if target_nlf is not None:
            try:
                pose_3d = target_nlf.get('joints3d_nonparam', [target_nlf])[0][0][0]
                def dist_3d(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                if len(pose_3d) > 13:
                    true_3d_bones = {
                        "torso": dist_3d(pose_3d[1], pose_3d[8]),
                        "r_thigh": dist_3d(pose_3d[8], pose_3d[9]), "r_calf": dist_3d(pose_3d[9], pose_3d[10]),
                        "l_thigh": dist_3d(pose_3d[11], pose_3d[12]), "l_calf": dist_3d(pose_3d[12], pose_3d[13])
                    }
            except Exception as e:
                pass

        return ({"perspective_slope": slope, "perspective_intercept": intercept, "true_3d_bones": true_3d_bones, "is_depth_inverted": is_inverted}, "\n".join(log_messages))




# ======================================================================
# V13: POSE CALIBRATION (PROPORTIONAL EXTRAPOLATION & RECHENVORGANG)
# ======================================================================
class PoseCalibrationV13:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_nah": ("POSEDATA",),
                "depth_nah": ("IMAGE",),
                "pose_fern": ("POSEDATA",),
                "depth_fern": ("IMAGE",),
                "norm_method": (["Dynamic Full-Body", "Torso (Neck-Hip)"], {"default": "Dynamic Full-Body"}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_depth": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"

    def calibrate(self, pose_nah, depth_nah, pose_fern, depth_fern, norm_method, min_confidence, invert_depth):
        import numpy as np
        import math
        
        log_messages = ["=== V13 CALIBRATION LOG (RECHENVORGANG) ==="]
        log_messages.append(f"Methode: {norm_method}")

        def get_body_metrics(pose_data, depth_map):
            meta = pose_data.get("pose_metas", [])[0]
            kps = getattr(meta, "kps_body", None)
            confs = getattr(meta, "kps_body_p", None)
            
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]
            
            def get_c(idx):
                if confs is not None and idx < len(confs): return float(confs[idx])
                if len(kps[idx]) >= 3: return float(kps[idx][2])
                return 1.0

            def is_val(idx):
                if kps is None or idx >= len(kps): return False
                pt = kps[idx]
                if pt is None or len(pt) < 2: return False
                if get_c(idx) < min_confidence: return False
                return (0 <= pt[0] < W) and (0 <= pt[1] < H)

            if kps is None or len(kps) < 14: 
                return {'torso': 0.0, 'max_len': 0.0, 'depth': 0.5, 'parts': 'None'}

            torso_len = 0.0
            torso_x, torso_y = [], []
            if is_val(1) and is_val(8) and is_val(11):
                torso_len = math.sqrt((kps[1][0] - (kps[8][0]+kps[11][0])/2.0)**2 + (kps[1][1] - (kps[8][1]+kps[11][1])/2.0)**2)
                torso_x.extend([kps[1][0], kps[8][0], kps[11][0]])
                torso_y.extend([kps[1][1], kps[8][1], kps[11][1]])

            top_y, bottom_y = None, None
            if is_val(0): top_y = kps[0][1]
            elif is_val(1): top_y = kps[1][1]

            parts_str = "Torso"
            heels = [idx for idx in [10, 13, 21, 24] if is_val(idx)]
            knees = [idx for idx in [9, 12] if is_val(idx)]
            hips = [idx for idx in [8, 11] if is_val(idx)]

            if heels:
                bottom_y = max([kps[idx][1] for idx in heels])
                parts_str = "Ganzkörper (mit Füßen)"
            elif knees:
                bottom_y = max([kps[idx][1] for idx in knees])
                parts_str = "Bis Knie"
            elif hips:
                bottom_y = max([kps[idx][1] for idx in hips])
                parts_str = "Nur Torso"

            max_len = torso_len 
            if top_y is not None and bottom_y is not None and bottom_y > top_y:
                max_len = bottom_y - top_y

            depth_vals = []
            for px, py in zip(torso_x, torso_y):
                ix, iy = int(px), int(py)
                if 0 <= ix < W and 0 <= iy < H:
                    depth_vals.append(depth_np[0, iy, ix])
            
            d_val = float(np.mean(depth_vals)) if depth_vals else 0.5
            if invert_depth: d_val = 1.0 / max(d_val, 0.0001)
            
            return {'torso': torso_len, 'max_len': max_len, 'depth': d_val, 'parts': parts_str}

        data_nah = get_body_metrics(pose_nah, depth_nah)
        data_fern = get_body_metrics(pose_fern, depth_fern)

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

        depth_diff = data_fern['depth'] - data_nah['depth']
        slope = 0.0 if abs(depth_diff) < 0.001 else (norm_fern - norm_nah) / depth_diff
        intercept = norm_nah - (slope * data_nah['depth'])

        log_messages.append("\n--- SLOPE & INTERCEPT RECHENVORGANG ---")
        log_messages.append(f"Slope = (Norm_Fern - Norm_Nah) / (Tiefe_Fern - Tiefe_Nah)")
        log_messages.append(f"Slope = ({norm_fern:.1f} - {norm_nah:.1f}) / ({data_fern['depth']:.4f} - {data_nah['depth']:.4f}) = {slope:.2f}")
        
        log_messages.append(f"Intercept = Norm_Nah - (Slope * Tiefe_Nah)")
        log_messages.append(f"Intercept = {norm_nah:.1f} - ({slope:.2f} * {data_nah['depth']:.4f}) = {intercept:.2f}")

        log_messages.append("\n=== ERGEBNIS ===")
        log_messages.append(f"Finale Norm Nah: {norm_nah:.1f} px | Tiefe Nah: {data_nah['depth']:.4f}")
        log_messages.append(f"Finale Norm Fern: {norm_fern:.1f} px | Tiefe Fern: {data_fern['depth']:.4f}")
        log_messages.append(f"Slope: {slope:.2f}")
        log_messages.append(f"Intercept: {intercept:.2f}")

        calib_data = {
            "perspective_slope": slope,
            "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth,
            "norm_method": norm_method
        }
        return (calib_data, "\n".join(log_messages))

# ======================================================================
# 3. V26: GLOBAL PERSPECTIVE SCALER (FULL-BODY & RECHENVORGANG)
# ======================================================================
class PoseGlobalPerspectiveScalerV26:
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
        log_messages = ["=== V26 GLOBAL SCALER LOG (RECHENVORGANG) ==="]

        if not pose_metas: return (pose_data_copy, "Fehler: Keine Pose-Daten.")
        
        slope = calibration_data.get("perspective_slope", 0.0)
        intercept = calibration_data.get("perspective_intercept", 1.0)
        is_inverted = calibration_data.get("is_depth_inverted", False)
        norm_method = calibration_data.get("norm_method", "Dynamic Full-Body")
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
        expected_norm = (avg_anchor_depth * slope) + intercept
        anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0
        
        log_messages.append(f"\n--- SKALIERUNG RECHENVORGANG ---")
        log_messages.append(f"Gewinner Frame: {best_idx}")
        log_messages.append(f"Ist-Norm (Video): {avg_anchor_norm:.1f} px")
        log_messages.append(f"Ist-Tiefe (Video): {avg_anchor_depth:.3f} m")
        
        log_messages.append(f"\nSoll-Norm = (Tiefe * Slope) + Intercept")
        log_messages.append(f"Soll-Norm = ({avg_anchor_depth:.3f} * {slope:.2f}) + {intercept:.2f} = {expected_norm:.1f} px")
        
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

# ======================================================================
# V15: POSE CALIBRATION (DYNAMIC FULL-BODY + UNSCALED DEPTH + PINHOLE)
# ======================================================================
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

# ======================================================================
# V28: GLOBAL PERSPECTIVE SCALER (FULL-BODY + PINHOLE MATH)
# ======================================================================
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

import copy
import numpy as np
import math
import json
import torch

# ======================================================================
# 1. PoseCalibrationV16 (Mit NLF 3D Extrahierung)
# ======================================================================
class PoseCalibrationV16:
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
                "nlf_data": ("NLFPRED", {"tooltip": "Optionale 3D NLF Daten zur Ermittlung der echten Größe"}),
                "config_data": ("STRING", {"default": "{}", "tooltip": "JSON ConfigurationData (z.B. für 3D Längen)"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, intrinsics_json=None, nlf_data=None, config_data="{}"):
        log_messages = ["=== V16 CALIBRATION LOG (UNSCALED DEPTH + PINHOLE) ==="]
        log_messages.append(f"Methode: {norm_method}")

        # --- ADDITIV: Parse Config Data ---
        try:
            config = json.loads(config_data)
        except:
            config = {}
            log_messages.append("Warnung: config_data konnte nicht als JSON geparst werden.")

        def get_body_metrics(pose_s, pose_u, depth_map):
            meta_s = pose_s.get("pose_metas", [])[0]
            meta_u = pose_u.get("pose_metas", [])[0]
            
            kps_s = getattr(meta_s, "kps_body", None)
            confs_s = getattr(meta_s, "kps_body_p", None)
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]

            def get_c_s(idx):
                if confs_s is not None and idx < len(confs_s): return float(confs_s[idx])
                if len(kps_s[idx]) >= 3: return float(kps_s[idx][2])
                return 1.0
                
            def is_val_s(idx):
                if kps_s is None or idx >= len(kps_s): return False
                pt = kps_s[idx]
                if pt is None or len(pt) < 2: return False
                if get_c_s(idx) < min_confidence: return False
                return True

            def get_c_u(idx):
                if confs_u is not None and idx < len(confs_u): return float(confs_u[idx])
                if len(kps_u[idx]) >= 3: return float(kps_u[idx][2])
                return 1.0
                
            def is_val_u(idx):
                if kps_u is None or idx >= len(kps_u): return False
                pt = kps_u[idx]
                if pt is None or len(pt) < 2: return False
                if get_c_u(idx) < min_confidence: return False
                if pt[0] < 0 or pt[0] >= 1.0 or pt[1] < 0 or pt[1] >= 1.0: return False
                return True

            norm = 100.0 
            if norm_method == "Torso (Neck-Hip)":
                if is_val_s(1) and is_val_s(8) and is_val_s(11):
                    mid_x = (kps_s[8][0] + kps_s[11][0]) / 2.0
                    mid_y = (kps_s[8][1] + kps_s[11][1]) / 2.0
                    norm = math.sqrt((kps_s[1][0] - mid_x)**2 + (kps_s[1][1] - mid_y)**2)
            else:
                valid_y = [kps_s[i][1] for i in range(len(kps_s)) if is_val_s(i)]
                valid_x = [kps_s[i][0] for i in range(len(kps_s)) if is_val_s(i)]
                if valid_y and valid_x:
                    norm = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)

            depth = 0.5 
            valid_u_x = []
            valid_u_y = []
            for idx in [1, 8, 11]:
                if is_val_u(idx):
                    valid_u_x.append(kps_u[idx][0])
                    valid_u_y.append(kps_u[idx][1])
            
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
        
        norm_nah = data_nah['norm']
        norm_fern = data_fern['norm']
        depth_c = data_nah['depth']
        depth_f = data_fern['depth']

        if invert_depth:
            depth_c = 1.0 / max(depth_c, 0.0001)
            depth_f = 1.0 / max(depth_f, 0.0001)

        slope = 0.0
        intercept = 1.0
        depth_diff = abs(depth_f - depth_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_f - depth_c)
            intercept = norm_nah - (slope * depth_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_c)

        fx = 500.0
        echte_groesse = 0.0
        if use_pinhole_math:
            if intrinsics_json:
                try:
                    intr_data = json.loads(intrinsics_json)
                    if isinstance(intr_data, list) and len(intr_data) > 0:
                        fx = float(intr_data[0][0])
                except: pass
            
            delta_z = abs(data_fern['depth'] - data_nah['depth'])
            if delta_z > 0.001 and abs(norm_nah - norm_fern) > 0.1:
                echte_groesse = (norm_nah * norm_fern * delta_z) / (fx * abs(norm_nah - norm_fern))
            else:
                echte_groesse = (norm_nah * data_nah['depth']) / fx

        # --- ADDITIV: 3D Längen aus NLF berechnen ---
        true_3d_bones = {}
        total_3d_height = 0.0

        if nlf_data is not None:
            try:
                pose_input_3d = nlf_data.get('joints3d_nonparam', [nlf_data])[0]
                if pose_input_3d is not None and len(pose_input_3d) > 0 and len(pose_input_3d[0]) > 0:
                    pose_3d = pose_input_3d[0][0]
                    
                    def dist_3d(p1, p2):
                        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

                    torso_3d = dist_3d(pose_3d[1], pose_3d[8]) if len(pose_3d) > 8 else 0
                    thigh_3d = dist_3d(pose_3d[8], pose_3d[9]) if len(pose_3d) > 9 else 0
                    calf_3d = dist_3d(pose_3d[9], pose_3d[10]) if len(pose_3d) > 10 else 0
                    
                    true_3d_bones = {
                        "torso": torso_3d,
                        "thigh": thigh_3d,
                        "calf": calf_3d
                    }
                    
                    total_3d_height = torso_3d + thigh_3d + calf_3d
                    head_allowance = config.get("head_allowance_3d", 0.15) 
                    total_3d_height += head_allowance
                    
                    log_messages.append(f"NLF 3D-Analyse -> Torso: {torso_3d:.3f}, Bein: {thigh_3d+calf_3d:.3f}, Total: {total_3d_height:.3f}")
            except Exception as e:
                log_messages.append(f"Warnung bei 3D NLF Extraktion: {e}")

        log_messages.append("\n=== ERGEBNIS ===")
        log_messages.append(f"Echte Größe (aus 2D): {echte_groesse:.3f}m | fx: {fx:.2f}")

        calib_data = {
            "perspective_slope": slope,
            "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth,
            "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math,
            "focal_length_fx": fx,
            "echte_groesse": echte_groesse,
            "true_3d_bones": true_3d_bones,
            "total_3d_height": total_3d_height,
            "config": config
        }
        
        return (calib_data, "\n".join(log_messages))

# ======================================================================
# 2. PoseGlobalPerspectiveScalerV29 (Mit 3D Scaling & 2D->3D Füßen)
# ======================================================================
class PoseGlobalPerspectiveScalerV29:
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
        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V29 GLOBAL SCALER LOG (INKL. 3D NLF) ==="]

        if not pose_metas:
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data)

        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        slope = calibration_data.get("perspective_slope", 0.0)
        intercept = calibration_data.get("perspective_intercept", 1.0)
        is_inverted = calibration_data.get("is_depth_inverted", False)
        norm_method = calibration_data.get("norm_method", "Dynamic Full-Body")
        
        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        echte_groesse = calibration_data.get("echte_groesse", 1.75)
        fx = calibration_data.get("focal_length_fx", 500.0)

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

        log_messages.append(f"\nFaktor = {expected_norm:.1f} / {avg_anchor_norm:.1f} = {anchor_scale:.3f}x")

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

        # --- ADDITIV: 3D NLF Daten skalieren & Füße hinzufügen ---
        nlf_data_scaled = None
        if video_nlf_data is not None:
            try:
                is_dict = isinstance(video_nlf_data, dict)
                pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0].clone() if is_dict else video_nlf_data.clone()
                
                for frame_idx in range(len(pose_input_3d)):
                    if pose_input_3d[frame_idx] is None or len(pose_input_3d[frame_idx]) == 0:
                        continue
                    
                    # 3D Skelett skalieren (Multiplikation der 3D Koordinaten)
                    pose_input_3d[frame_idx] *= anchor_scale
                    
                    # Füße aus den 2D Daten holen (Indizes 18-23 etc.)
                    if frame_idx < len(pose_metas):
                        meta = pose_metas[frame_idx]
                        kps_2d = getattr(meta, "kps_body", [])
                        
                        extra_feet = []
                        if len(kps_2d) > 20: 
                            # L-Toe (Index 18)
                            if (kps_2d[18][2] if len(kps_2d[18])>2 else 1) > min_confidence:
                                z_left_ankle = pose_input_3d[frame_idx][0][13][2] if len(pose_input_3d[frame_idx][0]) > 13 else 0.0
                                extra_feet.append([kps_2d[18][0], kps_2d[18][1], float(z_left_ankle)])
                            
                            # R-Toe (Index 19)
                            if (kps_2d[19][2] if len(kps_2d[19])>2 else 1) > min_confidence:
                                z_right_ankle = pose_input_3d[frame_idx][0][10][2] if len(pose_input_3d[frame_idx][0]) > 10 else 0.0
                                extra_feet.append([kps_2d[19][0], kps_2d[19][1], float(z_right_ankle)])
                        
                        if extra_feet:
                            feet_tensor = torch.tensor(extra_feet, dtype=pose_input_3d[frame_idx].dtype, device=pose_input_3d[frame_idx].device).unsqueeze(0)
                            pose_input_3d[frame_idx] = torch.cat((pose_input_3d[frame_idx], feet_tensor), dim=1)

                if is_dict:
                    nlf_data_scaled = copy.deepcopy(video_nlf_data)
                    nlf_data_scaled['joints3d_nonparam'] = [pose_input_3d]
                else:
                    nlf_data_scaled = pose_input_3d
                    
                log_messages.append(f"Erfolgreich: 3D NLF Daten um Faktor {anchor_scale:.3f}x skaliert und 2D Füße angeknüpft.")
            except Exception as e:
                log_messages.append(f"Fehler bei der NLF 3D-Skalierung: {e}")
                nlf_data_scaled = video_nlf_data

        return (pose_data_copy, "\n".join(log_messages), nlf_data_scaled)


# ======================================================================
# 3. Neue Visualisierungs- & Mapper-Nodes
# ======================================================================

class NLFDataToPoseData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096})
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Wandelt 3D NLF Daten zurück in reguläre 2D Pose Data um. (Mappt Indizes)"

    def process(self, nlf_poses, width, height):
        from .pose_utils.human_visualization import AAPoseMeta
        
        pose_input = nlf_poses['joints3d_nonparam'][0] if isinstance(nlf_poses, dict) and 'joints3d_nonparam' in nlf_poses else nlf_poses
        
        # Mapping von typischen NLF (z.B. COCO 14/17) zu OpenPose 25-Point
        mapping = {
            0: 0,   # Nose
            1: 1,   # Neck
            2: 2,   # R-Shoulder
            3: 3,   # R-Elbow
            4: 4,   # R-Wrist
            5: 5,   # L-Shoulder
            6: 6,   # L-Elbow
            7: 7,   # L-Wrist
            8: 8,   # R-Hip
            9: 9,   # R-Knee
            10: 10, # R-Ankle
            11: 11, # L-Hip
            12: 12, # L-Knee
            13: 13, # L-Ankle
        }

        pose_metas = []
        for i in range(len(pose_input)):
            meta = AAPoseMeta()
            meta.width = width
            meta.height = height
            
            kps_body = np.zeros((25, 2))
            kps_body_p = np.zeros(25)
            
            if pose_input[i] is not None and len(pose_input[i]) > 0:
                pose_3d = pose_input[i][0].cpu().numpy()
                
                for nlf_idx, op_idx in mapping.items():
                    if nlf_idx < len(pose_3d):
                        kps_body[op_idx] = [pose_3d[nlf_idx][0], pose_3d[nlf_idx][1]]
                        kps_body_p[op_idx] = 1.0 
                
                # Angehängte Füße (aus Node V29) abrufen (Indices > 13)
                if len(pose_3d) > 14:
                    kps_body[18] = [pose_3d[14][0], pose_3d[14][1]] # L-Toe
                    kps_body_p[18] = 1.0
                if len(pose_3d) > 15:
                    kps_body[19] = [pose_3d[15][0], pose_3d[15][1]] # R-Toe
                    kps_body_p[19] = 1.0

            meta.kps_body = kps_body
            meta.kps_body_p = kps_body_p
            pose_metas.append(meta)

        pose_data = {
            "retarget_image": None,
            "pose_metas": pose_metas,
            "refer_pose_meta": None,
            "pose_metas_original": pose_metas,
        }
        return (pose_data,)


class RenderNLFPosesDirect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses_scaled": ("NLFPRED", {"tooltip": "Die skalierten 3D NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht, falls NLF diese nicht abdeckt"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Rendert manuell skalierte NLF-Daten DIREKT, ohne das Modell die Koordinaten überschreiben zu lassen."

    def process(self, nlf_poses_scaled, width, height, render_backend="taichi", dw_poses_fallback=None):
        from .NLFPoseExtract.nlf_render import render_multi_nlf_as_images, render_nlf_as_images, intrinsic_matrix_from_field_of_view
        
        if render_backend == "taichi":
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
            except:
                render_backend = "torch"

        pose_input = nlf_poses_scaled['joints3d_nonparam'][0] if isinstance(nlf_poses_scaled, dict) else nlf_poses_scaled
        
        dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
        intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])

        if pose_input[0].shape[0] > 1:
            frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)
        else:
            frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)

        frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
        frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
        
        return (frames_tensor.cpu().float(), mask.cpu().float())


# ======================================================================
# 1. PoseCalibrationV17 (Mit NLF Nah & Fern für Durchschnitts-3D)
# ======================================================================
class PoseCalibrationV17:
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
                "nlf_data_nah": ("NLFPRED", {"tooltip": "3D NLF Daten vom NAHEN Kalibrierungs-Frame"}),
                "nlf_data_fern": ("NLFPRED", {"tooltip": "3D NLF Daten vom FERNEN Kalibrierungs-Frame"}),
                "config_data": ("STRING", {"default": "{}", "tooltip": "JSON ConfigurationData (z.B. für 3D Längen)"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, intrinsics_json=None, nlf_data_nah=None, nlf_data_fern=None, config_data="{}"):
        import numpy as np
        import math
        import json
        
        log_messages = ["=== V17 CALIBRATION LOG (UNSCALED DEPTH + PINHOLE + 3D AVG) ==="]
        log_messages.append(f"Methode: {norm_method}")

        # Parse Config Data
        try:
            config = json.loads(config_data)
        except:
            config = {}
            log_messages.append("Warnung: config_data konnte nicht als JSON geparst werden.")

        def get_body_metrics(pose_s, pose_u, depth_map):
            meta_s = pose_s.get("pose_metas", [])[0]
            meta_u = pose_u.get("pose_metas", [])[0]
            
            kps_s = getattr(meta_s, "kps_body", None)
            confs_s = getattr(meta_s, "kps_body_p", None)
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]

            def get_c_s(idx):
                if confs_s is not None and idx < len(confs_s): return float(confs_s[idx])
                if len(kps_s[idx]) >= 3: return float(kps_s[idx][2])
                return 1.0
                
            def is_val_s(idx):
                if kps_s is None or idx >= len(kps_s): return False
                pt = kps_s[idx]
                if pt is None or len(pt) < 2: return False
                if get_c_s(idx) < min_confidence: return False
                return True

            def get_c_u(idx):
                if confs_u is not None and idx < len(confs_u): return float(confs_u[idx])
                if len(kps_u[idx]) >= 3: return float(kps_u[idx][2])
                return 1.0
                
            def is_val_u(idx):
                if kps_u is None or idx >= len(kps_u): return False
                pt = kps_u[idx]
                if pt is None or len(pt) < 2: return False
                if get_c_u(idx) < min_confidence: return False
                if pt[0] < 0 or pt[0] >= 1.0 or pt[1] < 0 or pt[1] >= 1.0: return False
                return True

            norm = 100.0 
            if norm_method == "Torso (Neck-Hip)":
                if is_val_s(1) and is_val_s(8) and is_val_s(11):
                    mid_x = (kps_s[8][0] + kps_s[11][0]) / 2.0
                    mid_y = (kps_s[8][1] + kps_s[11][1]) / 2.0
                    norm = math.sqrt((kps_s[1][0] - mid_x)**2 + (kps_s[1][1] - mid_y)**2)
            else:
                valid_y = [kps_s[i][1] for i in range(len(kps_s)) if is_val_s(i)]
                valid_x = [kps_s[i][0] for i in range(len(kps_s)) if is_val_s(i)]
                if valid_y and valid_x:
                    norm = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)

            depth = 0.5 
            valid_u_x = []
            valid_u_y = []
            for idx in [1, 8, 11]:
                if is_val_u(idx):
                    valid_u_x.append(kps_u[idx][0])
                    valid_u_y.append(kps_u[idx][1])
            
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
        
        norm_nah = data_nah['norm']
        norm_fern = data_fern['norm']
        depth_c = data_nah['depth']
        depth_f = data_fern['depth']

        if invert_depth:
            depth_c = 1.0 / max(depth_c, 0.0001)
            depth_f = 1.0 / max(depth_f, 0.0001)

        slope = 0.0
        intercept = 1.0
        depth_diff = abs(depth_f - depth_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_f - depth_c)
            intercept = norm_nah - (slope * depth_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_c)

        fx = 500.0
        echte_groesse = 0.0
        if use_pinhole_math:
            if intrinsics_json:
                try:
                    intr_data = json.loads(intrinsics_json)
                    if isinstance(intr_data, list) and len(intr_data) > 0:
                        fx = float(intr_data[0][0])
                except: pass
            
            delta_z = abs(data_fern['depth'] - data_nah['depth'])
            if delta_z > 0.001 and abs(norm_nah - norm_fern) > 0.1:
                echte_groesse = (norm_nah * norm_fern * delta_z) / (fx * abs(norm_nah - norm_fern))
            else:
                echte_groesse = (norm_nah * data_nah['depth']) / fx

        # --- ADDITIV V17: 3D Längen aus NLF Nah UND Fern extrahieren & mitteln ---
        true_3d_bones = {}
        total_3d_height = 0.0

        def extract_3d_bones(nlf_data):
            if nlf_data is None: return None
            try:
                pose_input_3d = nlf_data.get('joints3d_nonparam', [nlf_data])[0]
                if pose_input_3d is not None and len(pose_input_3d) > 0 and len(pose_input_3d[0]) > 0:
                    pose_3d = pose_input_3d[0][0]
                    def dist_3d(p1, p2):
                        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

                    torso = dist_3d(pose_3d[1], pose_3d[8]) if len(pose_3d) > 8 else 0
                    thigh = dist_3d(pose_3d[8], pose_3d[9]) if len(pose_3d) > 9 else 0
                    calf = dist_3d(pose_3d[9], pose_3d[10]) if len(pose_3d) > 10 else 0
                    return {"torso": torso, "thigh": thigh, "calf": calf}
            except: pass
            return None

        bones_nah = extract_3d_bones(nlf_data_nah)
        bones_fern = extract_3d_bones(nlf_data_fern)

        if bones_nah or bones_fern:
            # Durchschnitt bilden, falls beide vorhanden
            b_n = bones_nah if bones_nah else bones_fern
            b_f = bones_fern if bones_fern else bones_nah
            
            torso_3d = (b_n["torso"] + b_f["torso"]) / 2.0
            thigh_3d = (b_n["thigh"] + b_f["thigh"]) / 2.0
            calf_3d = (b_n["calf"] + b_f["calf"]) / 2.0

            true_3d_bones = {
                "torso": torso_3d,
                "thigh": thigh_3d,
                "calf": calf_3d
            }
            
            total_3d_height = torso_3d + thigh_3d + calf_3d
            head_allowance = config.get("head_allowance_3d", 0.15) 
            total_3d_height += head_allowance
            
            log_messages.append(f"NLF 3D-Analyse (Gemittelt) -> Torso: {torso_3d:.3f}, Bein: {thigh_3d+calf_3d:.3f}, Total: {total_3d_height:.3f}")
        else:
            log_messages.append("Keine NLF-Daten für 3D-Analyse geliefert oder Extraktion fehlgeschlagen.")

        log_messages.append("\n=== ERGEBNIS ===")
        log_messages.append(f"Echte Größe (aus 2D): {echte_groesse:.3f}m | fx: {fx:.2f}")

        calib_data = {
            "perspective_slope": slope,
            "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth,
            "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math,
            "focal_length_fx": fx,
            "echte_groesse": echte_groesse,
            "true_3d_bones": true_3d_bones,
            "total_3d_height": total_3d_height,
            "config": config
        }
        
        return (calib_data, "\n".join(log_messages))

import copy
import numpy as np
import math
import json
import torch

# ======================================================================
# 1. PoseCalibrationV18 (Mit smarter Waden-Berechnung via Faktor)
# ======================================================================
class PoseCalibrationV18:
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
                "nlf_data_nah": ("NLFPRED", {"tooltip": "3D NLF Daten vom NAHEN Kalibrierungs-Frame (abgeschnittene Waden)"}),
                "nlf_data_fern": ("NLFPRED", {"tooltip": "3D NLF Daten vom FERNEN Kalibrierungs-Frame (ganzer Körper)"}),
                "config_data": ("STRING", {"default": "{}", "tooltip": "JSON ConfigurationData"}),
            }
        }

    RETURN_TYPES = ("POSE_CALIBRATION", "STRING",)
    RETURN_NAMES = ("calibration_data", "log_output",)
    FUNCTION = "calibrate"
    CATEGORY = "WanAnimatePreprocess/Ultimate"

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, intrinsics_json=None, nlf_data_nah=None, nlf_data_fern=None, config_data="{}"):
        log_messages = ["=== V18 CALIBRATION LOG (SMART CALF ESTIMATION) ==="]
        log_messages.append(f"Methode: {norm_method}")

        try:
            config = json.loads(config_data)
        except:
            config = {}
            log_messages.append("Warnung: config_data konnte nicht als JSON geparst werden.")

        # --- 2D TIEFEN- UND NORM-BERECHNUNG ---
        def get_body_metrics(pose_s, pose_u, depth_map):
            meta_s = pose_s.get("pose_metas", [])[0]
            meta_u = pose_u.get("pose_metas", [])[0]
            
            kps_s = getattr(meta_s, "kps_body", None)
            confs_s = getattr(meta_s, "kps_body_p", None)
            kps_u = getattr(meta_u, "kps_body", None)
            confs_u = getattr(meta_u, "kps_body_p", None)
            
            depth_np = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            H, W = depth_np.shape[1], depth_np.shape[2]

            def get_c_s(idx):
                if confs_s is not None and idx < len(confs_s): return float(confs_s[idx])
                if len(kps_s[idx]) >= 3: return float(kps_s[idx][2])
                return 1.0
                
            def is_val_s(idx):
                if kps_s is None or idx >= len(kps_s): return False
                pt = kps_s[idx]
                if pt is None or len(pt) < 2: return False
                if get_c_s(idx) < min_confidence: return False
                return True

            def get_c_u(idx):
                if confs_u is not None and idx < len(confs_u): return float(confs_u[idx])
                if len(kps_u[idx]) >= 3: return float(kps_u[idx][2])
                return 1.0
                
            def is_val_u(idx):
                if kps_u is None or idx >= len(kps_u): return False
                pt = kps_u[idx]
                if pt is None or len(pt) < 2: return False
                if get_c_u(idx) < min_confidence: return False
                if pt[0] < 0 or pt[0] >= 1.0 or pt[1] < 0 or pt[1] >= 1.0: return False
                return True

            norm = 100.0 
            if norm_method == "Torso (Neck-Hip)":
                if is_val_s(1) and is_val_s(8) and is_val_s(11):
                    mid_x = (kps_s[8][0] + kps_s[11][0]) / 2.0
                    mid_y = (kps_s[8][1] + kps_s[11][1]) / 2.0
                    norm = math.sqrt((kps_s[1][0] - mid_x)**2 + (kps_s[1][1] - mid_y)**2)
            else:
                valid_y = [kps_s[i][1] for i in range(len(kps_s)) if is_val_s(i)]
                valid_x = [kps_s[i][0] for i in range(len(kps_s)) if is_val_s(i)]
                if valid_y and valid_x:
                    norm = math.sqrt((max(valid_x) - min(valid_x))**2 + (max(valid_y) - min(valid_y))**2)

            depth = 0.5 
            valid_u_x = []
            valid_u_y = []
            for idx in [1, 8, 11]:
                if is_val_u(idx):
                    valid_u_x.append(kps_u[idx][0])
                    valid_u_y.append(kps_u[idx][1])
            
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
        
        norm_nah = data_nah['norm']
        norm_fern = data_fern['norm']
        depth_c = data_nah['depth']
        depth_f = data_fern['depth']

        if invert_depth:
            depth_c = 1.0 / max(depth_c, 0.0001)
            depth_f = 1.0 / max(depth_f, 0.0001)

        slope = 0.0
        intercept = 1.0
        depth_diff = abs(depth_f - depth_c)
        if depth_diff > 0.05:
            slope = (norm_fern - norm_nah) / (depth_f - depth_c)
            intercept = norm_nah - (slope * depth_c)
        else:
            slope = -500.0 if invert_depth else 500.0
            intercept = norm_nah - (slope * depth_c)

        fx = 500.0
        echte_groesse = 0.0
        if use_pinhole_math:
            if intrinsics_json:
                try:
                    intr_data = json.loads(intrinsics_json)
                    if isinstance(intr_data, list) and len(intr_data) > 0:
                        fx = float(intr_data[0][0])
                except: pass
            
            delta_z = abs(data_fern['depth'] - data_nah['depth'])
            if delta_z > 0.001 and abs(norm_nah - norm_fern) > 0.1:
                echte_groesse = (norm_nah * norm_fern * delta_z) / (fx * abs(norm_nah - norm_fern))
            else:
                echte_groesse = (norm_nah * data_nah['depth']) / fx

        # --- 3D LÄNGEN-EXTRAKTION (WADE REPARIEREN) ---
        true_3d_bones = {}
        total_3d_height = 0.0

        def extract_3d_bones(nlf_data):
            if nlf_data is None: return None
            try:
                pose_input_3d = nlf_data.get('joints3d_nonparam', [nlf_data])[0]
                if pose_input_3d is not None and len(pose_input_3d) > 0 and len(pose_input_3d[0]) > 0:
                    pose_3d = pose_input_3d[0][0]
                    def dist_3d(p1, p2):
                        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

                    torso = dist_3d(pose_3d[1], pose_3d[8]) if len(pose_3d) > 8 else 0
                    thigh = dist_3d(pose_3d[8], pose_3d[9]) if len(pose_3d) > 9 else 0
                    calf = dist_3d(pose_3d[9], pose_3d[10]) if len(pose_3d) > 10 else 0
                    return {"torso": torso, "thigh": thigh, "calf": calf}
            except: pass
            return None

        bones_nah = extract_3d_bones(nlf_data_nah)
        bones_fern = extract_3d_bones(nlf_data_fern)

        if bones_nah and bones_fern:
            # 1. Wir berechnen den Skalierungsfaktor nur aus sichtbaren, verlässlichen Teilen (Torso & Oberschenkel)
            factor_torso = bones_nah["torso"] / bones_fern["torso"] if bones_fern["torso"] > 0 else 1.0
            factor_thigh = bones_nah["thigh"] / bones_fern["thigh"] if bones_fern["thigh"] > 0 else 1.0
            
            # Den Durchschnittsfaktor der oberen Hälfte nehmen wir als Master-Faktor
            master_3d_factor = (factor_torso + factor_thigh) / 2.0
            
            # 2. Die korrumpierte "Nahe"-Wade werfen wir weg und berechnen sie sauber aus der "Fernen"-Wade!
            calf_nah_estimated = bones_fern["calf"] * master_3d_factor

            true_3d_bones = {
                "torso": bones_nah["torso"],
                "thigh": bones_nah["thigh"],
                "calf": calf_nah_estimated,  # Die reparierte Wade
                "factor_nah_fern": master_3d_factor,
                "original_corrupted_calf": bones_nah["calf"],
                "fern_reference": bones_fern
            }
            
            total_3d_height = bones_nah["torso"] + bones_nah["thigh"] + calf_nah_estimated
            head_allowance = config.get("head_allowance_3d", 0.15) 
            total_3d_height += head_allowance
            
            log_messages.append(f"Faktor Nah/Fern (aus Oberschenkel/Torso): {master_3d_factor:.3f}x")
            log_messages.append(f"Wade repariert! Alt (abgeschnitten): {bones_nah['calf']:.3f} -> Neu (berechnet): {calf_nah_estimated:.3f}")
            log_messages.append(f"Finale 3D-Metrik Nah -> Torso: {bones_nah['torso']:.3f}, Bein: {bones_nah['thigh']+calf_nah_estimated:.3f}")
        else:
            log_messages.append("Fehler/Warnung: Konnte nicht beide NLF-Frames (Nah & Fern) für die Waden-Reparatur extrahieren.")

        calib_data = {
            "perspective_slope": slope,
            "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth,
            "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math,
            "focal_length_fx": fx,
            "echte_groesse": echte_groesse,
            "true_3d_bones": true_3d_bones,
            "total_3d_height": total_3d_height,
            "config": config
        }
        
        return (calib_data, "\n".join(log_messages))


# ======================================================================
# 2. PoseGlobalPerspectiveScalerV30 (Bugfix: Robuste NLF-Listen-Skalierung)
# ======================================================================
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


# ======================================================================
# 2. PoseGlobalPerspectiveScalerV31 (Fix: 3D Pivot Scaling & Neue Listen-Architektur)
# ======================================================================
class PoseGlobalPerspectiveScalerV31:
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
        import traceback

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
                    return (shoulder_w / hip_w) > frontal_2d_threshold
            return False

        def is_frontal_3d(frame_idx):
            if video_nlf_data is None: return False
            pose_input_3d = video_nlf_data.get('joints3d_nonparam', [video_nlf_data])[0]
            if pose_input_3d is None or len(pose_input_3d) <= frame_idx or pose_input_3d[frame_idx] is None or len(pose_input_3d[frame_idx]) == 0:
                return False
            pose_3d = pose_input_3d[frame_idx][0]
            if len(pose_3d) > 11:
                l_hip, r_hip = pose_3d[11], pose_3d[8]
                dz, dx = abs(l_hip[2] - r_hip[2]), abs(l_hip[0] - r_hip[0])
                return (math.degrees(math.atan2(dz, dx)) if dx > 0 else 90.0) < frontal_3d_angle_tolerance
            return False

        best_idx, best_area = 0, 0.0

        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            
            if not valid_y or not valid_x: continue
            if not (is_frontal_3d(i) if frontal_method == "3D_NLF" and video_nlf_data is not None else is_frontal_2d(meta)): continue
            
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
            
            depth_vals = [depth_np[v_idx, int(py), int(px)] for px, py in zip(valid_x_d, valid_y_d) if 0 <= int(px) < W and 0 <= int(py) < H]
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

        # --- 3D NLF DATEN ROBUST SKALIEREN (Mitte-Schwerpunkt-Methode) ---
        nlf_data_scaled = None
        log_messages.append("\n=== NLF 3D DATA SCALING DETAILED LOG ===")
        log_messages.append("URSACHE GEFUNDEN: Das Projektions-Paradoxon! Skaliere nun um den 3D-Pivotpunkt der Füße, damit die Projektion tatsächlich wächst.")
        
        if video_nlf_data is not None:
            try:
                nlf_data_scaled = copy.deepcopy(video_nlf_data)
                is_dict = isinstance(nlf_data_scaled, dict)
                
                keys_to_scale = [k for k in ['joints3d_nonparam', 'joints3d', 'joints', 'kps3d'] if k in nlf_data_scaled] if is_dict else [None]
                skalierte_frames_gesamt = 0

                for key in keys_to_scale:
                    target_list = nlf_data_scaled[key][0] if key is not None else nlf_data_scaled
                    if target_list is None or len(target_list) == 0: continue
                        
                    log_messages.append(f"Verarbeite Key '{key}' mit {len(target_list)} Frames.")
                    new_target_list = [] # Komplett neue Liste erzwingt Speicher-Aktualisierung

                    for frame_idx in range(len(target_list)):
                        frame_data = target_list[frame_idx]
                        if frame_data is None or len(frame_data) == 0:
                            new_target_list.append(frame_data)
                            continue
                            
                        if isinstance(frame_data, torch.Tensor):
                            dim = frame_data.dim()
                            pts = frame_data[0] if dim == 3 else frame_data
                            
                            if len(pts) > 0:
                                # PIVOT FINDEN (X-Mitte, Y-Füße, Z-Mitte)
                                pivot_x = pts[:, 0].mean()
                                pivot_y = pts[:, 1].max() # Max Y ist im Bildbereich meist unten (Füße)
                                pivot_z = pts[:, 2].mean()
                                
                                pivot_3d = torch.tensor([pivot_x, pivot_y, pivot_z], dtype=pts.dtype, device=pts.device)
                                
                                # RELATIVE SKALIERUNG: Distanz zum Pivot * anchor_scale
                                scaled_frame = pivot_3d + (frame_data - pivot_3d) * anchor_scale
                                
                                if frame_idx == 0:
                                    log_messages.append(f"Frame 0 Pivot-Z: {pivot_z:.4f}")
                                    log_messages.append(f"Frame 0 (Zentriert) -> Y-Wert vorher: {float(pts[1][1]):.4f} | nachher: {float((scaled_frame[0] if dim==3 else scaled_frame)[1][1]):.4f}")
                                    
                                # FÜSSE ANKNÜPFEN (Rein in 3D berechnet, ohne 2D Pixel zu vermischen)
                                if len(pts) > 13:
                                    r_ankle = scaled_frame[0][10] if dim == 3 else scaled_frame[10]
                                    l_ankle = scaled_frame[0][13] if dim == 3 else scaled_frame[13]
                                    
                                    # Virtuelle Zehen (leicht nach unten und vorne)
                                    r_toe, l_toe = r_ankle.clone(), l_ankle.clone()
                                    r_toe[1] += 0.05 * anchor_scale; r_toe[2] -= 0.10 * anchor_scale
                                    l_toe[1] += 0.05 * anchor_scale; l_toe[2] -= 0.10 * anchor_scale
                                    
                                    feet_t = torch.tensor([l_toe.tolist(), r_toe.tolist()], dtype=scaled_frame.dtype, device=scaled_frame.device)
                                    scaled_frame = torch.cat((scaled_frame, feet_t.unsqueeze(0)), dim=1) if dim == 3 else torch.cat((scaled_frame, feet_t), dim=0)

                                new_target_list.append(scaled_frame)
                                skalierte_frames_gesamt += 1
                            else:
                                new_target_list.append(frame_data.clone())
                        else:
                            new_target_list.append(frame_data)
                            
                    # Liste direkt im Dictionary überschreiben
                    if key is not None:
                        nlf_data_scaled[key][0] = new_target_list

                log_messages.append(f"ERFOLG: Insgesamt wurden {skalierte_frames_gesamt} 3D-Frames tiefenskalierend überschrieben (Pivot-Methode).")

            except Exception as e:
                log_messages.append(f"KRITISCHER FEHLER BEI NLF SKALIERUNG: {e}")
                log_messages.append(traceback.format_exc())
                nlf_data_scaled = video_nlf_data

        return (pose_data_copy, "\n".join(log_messages), nlf_data_scaled)

import copy
import numpy as np
import math
import torch
import traceback
import json

# ======================================================================
# 1. PoseCalibrationV19 (Exakt wie V18, nur umbenannt für deinen Workflow)
# ======================================================================
class PoseCalibrationV19:
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

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, intrinsics_json=None, nlf_data_nah=None, nlf_data_fern=None, config_data="{}"):
        log_messages = ["=== V19 CALIBRATION LOG (SMART CALF ESTIMATION) ==="]
        try: config = json.loads(config_data)
        except: config = {}

        # (Existierende 2D Metrik-Logik...)
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

        # --- 3D LÄNGEN-EXTRAKTION (WADE REPARIEREN) ---
        true_3d_bones = {}
        total_3d_height = 0.0

        def extract_3d_bones(nlf_data):
            if nlf_data is None: return None
            try:
                pose_input_3d = nlf_data.get('joints3d_nonparam', [nlf_data])[0]
                if pose_input_3d is not None and len(pose_input_3d) > 0 and len(pose_input_3d[0]) > 0:
                    pose_3d = pose_input_3d[0][0]
                    def dist_3d(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                    return {
                        "torso": dist_3d(pose_3d[1], pose_3d[8]) if len(pose_3d) > 8 else 0,
                        "thigh": dist_3d(pose_3d[8], pose_3d[9]) if len(pose_3d) > 9 else 0,
                        "calf": dist_3d(pose_3d[9], pose_3d[10]) if len(pose_3d) > 10 else 0
                    }
            except: pass
            return None

        bones_nah = extract_3d_bones(nlf_data_nah)
        bones_fern = extract_3d_bones(nlf_data_fern)

        if bones_nah and bones_fern:
            factor_torso = bones_nah["torso"] / bones_fern["torso"] if bones_fern["torso"] > 0 else 1.0
            factor_thigh = bones_nah["thigh"] / bones_fern["thigh"] if bones_fern["thigh"] > 0 else 1.0
            master_3d_factor = (factor_torso + factor_thigh) / 2.0
            calf_nah_estimated = bones_fern["calf"] * master_3d_factor

            true_3d_bones = {
                "torso": bones_nah["torso"],
                "thigh": bones_nah["thigh"],
                "calf": calf_nah_estimated,
                "factor_nah_fern": master_3d_factor
            }
            total_3d_height = bones_nah["torso"] + bones_nah["thigh"] + calf_nah_estimated + config.get("head_allowance_3d", 0.15) 
            log_messages.append(f"Wade repariert! Neu (berechnet): {calf_nah_estimated:.3f}")
        
        calib_data = {
            "perspective_slope": slope, "perspective_intercept": intercept, "is_depth_inverted": invert_depth,
            "norm_method": norm_method, "use_pinhole_math": use_pinhole_math, "focal_length_fx": fx,
            "echte_groesse": echte_groesse, "true_3d_bones": true_3d_bones, "total_3d_height": total_3d_height,
            "config": config
        }
        return (calib_data, "\n".join(log_messages))


# ======================================================================
# 2. PoseGlobalPerspectiveScalerV33 (DIE PERFEKTE LÖSUNG: Kamera-Translation)
# ======================================================================
class PoseGlobalPerspectiveScalerV33:
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
        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V33 GLOBAL SCALER LOG ==="]

        if not pose_metas: return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data)

        # Configs aus Calibration laden
        slope = calibration_data.get("perspective_slope", 0.0)
        intercept = calibration_data.get("perspective_intercept", 1.0)
        is_inverted = calibration_data.get("is_depth_inverted", False)
        norm_method = calibration_data.get("norm_method", "Dynamic Full-Body")
        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        echte_groesse = calibration_data.get("echte_groesse", 1.75)
        fx = calibration_data.get("focal_length_fx", 500.0)
        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        def is_valid_point(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else (float(pt[2]) if len(pt)>=3 else 1.0)
            return c >= min_confidence

        # (Frame Evaluierungslogik bleibt identisch)
        best_idx, best_area = 0, 0.0
        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            if not valid_y or not valid_x: continue
            
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

        if valid_frames_in_window == 0: return (pose_data_copy, "Fehler.", video_nlf_data)

        avg_anchor_norm = sum_norm / valid_frames_in_window
        avg_anchor_depth = sum_depth / valid_frames_in_window
        expected_norm = (avg_anchor_depth * slope) + intercept
        anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0

        log_messages.append(f"Skalierungs-Faktor: {anchor_scale:.3f}x")

        # 1. 2D Daten Skalieren (Das verhält sich wie ein Sticker - mathematischer Stretch)
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


        # --- 3D NLF DATEN PHYSIKALISCH KORREKT SKALIEREN (Kamera-Translation) ---
        nlf_data_scaled = None
        log_messages.append("\n=== NLF 3D DATA SCALING DETAILED LOG ===")
        log_messages.append("URSACHE BEHOBEN: Wir verzerren die 3D Knochen NICHT! Wir schieben die Person näher an die Kamera (Translation).")
        
        if video_nlf_data is not None:
            try:
                nlf_data_scaled = copy.deepcopy(video_nlf_data)
                is_dict = isinstance(nlf_data_scaled, dict)
                keys_to_scale = [k for k in ['joints3d_nonparam', 'joints3d', 'joints', 'kps3d'] if k in nlf_data_scaled] if is_dict else [None]
                skalierte_frames_gesamt = 0

                for key in keys_to_scale:
                    target_list = nlf_data_scaled[key][0] if key is not None else nlf_data_scaled
                    if target_list is None or len(target_list) == 0: continue
                    new_target_list = []

                    for frame_idx in range(len(target_list)):
                        frame_data = target_list[frame_idx]
                        if frame_data is None or len(frame_data) == 0:
                            new_target_list.append(frame_data)
                            continue
                            
                        if isinstance(frame_data, torch.Tensor):
                            dim = frame_data.dim()
                            pts = frame_data[0] if dim == 3 else frame_data
                            
                            if len(pts) > 0:
                                # 1. Wir ermitteln das 3D Zentrum (Pivot)
                                pivot_x = pts[:, 0].mean()
                                pivot_y = pts[:, 1].max() # Füße
                                pivot_z = pts[:, 2].mean()
                                
                                # 2. Wir berechnen den Vektor, um die Person exakt so viel 
                                #    näher an die Kamera zu holen, dass sie 'anchor_scale' größer wirkt!
                                delta_x = (pivot_x / anchor_scale) - pivot_x
                                delta_y = (pivot_y / anchor_scale) - pivot_y
                                delta_z = (pivot_z / anchor_scale) - pivot_z
                                
                                # Wir erzeugen ein frisches Skelett
                                scaled_frame = frame_data.clone()
                                
                                # 3. Wir schieben alle Punkte um den exakt gleichen Vektor (Keine Verzerrung!)
                                if dim == 3:
                                    scaled_frame[0, :, 0] += delta_x
                                    scaled_frame[0, :, 1] += delta_y
                                    scaled_frame[0, :, 2] += delta_z
                                else:
                                    scaled_frame[:, 0] += delta_x
                                    scaled_frame[:, 1] += delta_y
                                    scaled_frame[:, 2] += delta_z

                                if frame_idx == 0:
                                    log_messages.append(f"Frame 0: Person um {delta_z:.1f} Einheiten zur Kamera gezogen.")
                                    
                                # 4. Füße anhängen (Basierend auf den neuen, herangezogenen Positionen)
                                if len(pts) > 13:
                                    r_ankle = scaled_frame[0][10] if dim == 3 else scaled_frame[10]
                                    l_ankle = scaled_frame[0][13] if dim == 3 else scaled_frame[13]
                                    
                                    r_toe, l_toe = r_ankle.clone(), l_ankle.clone()
                                    r_toe[1] += 0.05 * anchor_scale; r_toe[2] -= 0.10 * anchor_scale
                                    l_toe[1] += 0.05 * anchor_scale; l_toe[2] -= 0.10 * anchor_scale
                                    
                                    feet_t = torch.tensor([l_toe.tolist(), r_toe.tolist()], dtype=scaled_frame.dtype, device=scaled_frame.device)
                                    scaled_frame = torch.cat((scaled_frame, feet_t.unsqueeze(0)), dim=1) if dim == 3 else torch.cat((scaled_frame, feet_t), dim=0)

                                new_target_list.append(scaled_frame)
                                skalierte_frames_gesamt += 1
                            else:
                                new_target_list.append(frame_data.clone())
                        else:
                            new_target_list.append(frame_data)
                            
                    if key is not None:
                        nlf_data_scaled[key][0] = new_target_list

                log_messages.append(f"ERFOLG: 3D Form bewahrt. Rendering wird nun visuell {anchor_scale:.2f}x größer sein.")

            except Exception as e:
                log_messages.append(f"FEHLER: {e}\n{traceback.format_exc()}")
                nlf_data_scaled = video_nlf_data

        return (pose_data_copy, "\n".join(log_messages), nlf_data_scaled)

# ======================================================================
# 2. PoseGlobalPerspectiveScalerV34 (PERFEKTER FAKTOR + 3D TRANSLATION)
# ======================================================================
class PoseGlobalPerspectiveScalerV34:
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
        import traceback

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V34 GLOBAL SCALER LOG ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data)

        # Configs aus Calibration laden (Versteht V18/V19 Waden-Logik)
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if "calf" in true_3d_bones:
            log_messages.append(f"V34 hat die reparierten 3D-Längen verstanden. Soll-Wade: {true_3d_bones['calf']:.3f}x")

        slope = calibration_data.get("perspective_slope", 0.0)
        intercept = calibration_data.get("perspective_intercept", 1.0)
        is_inverted = calibration_data.get("is_depth_inverted", False)
        norm_method = calibration_data.get("norm_method", "Dynamic Full-Body")
        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        echte_groesse = calibration_data.get("echte_groesse", 1.75)
        fx = calibration_data.get("focal_length_fx", 500.0)
        
        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        def is_valid_point(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else (float(pt[2]) if len(pt)>=3 else 1.0)
            return c >= min_confidence

        def is_frontal_2d(meta):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            if is_valid_point(kps, confs, 2) and is_valid_point(kps, confs, 5) and is_valid_point(kps, confs, 8) and is_valid_point(kps, confs, 11):
                shoulder_w = abs(kps[2][0] - kps[5][0])
                hip_w = abs(kps[8][0] - kps[11][0])
                if hip_w > 0: return (shoulder_w / hip_w) > frontal_2d_threshold
            return False

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
            if not (is_frontal_3d(i) if frontal_method == "3D_NLF" and video_nlf_data is not None else is_frontal_2d(meta)): continue
            
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
            depth_vals = [depth_np[v_idx, int(py), int(px)] for px, py in zip(valid_x_d, valid_y_d) if 0 <= int(px) < W and 0 <= int(py) < H]
            
            frame_depth = float(np.mean(depth_vals)) if depth_vals else 0.5
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            sum_norm += norm_val
            sum_depth += frame_depth
            valid_frames_in_window += 1

        if valid_frames_in_window == 0: 
            return (pose_data_copy, "Fehler: Anchor-Window ungültig.", video_nlf_data)

        avg_anchor_norm = sum_norm / valid_frames_in_window
        avg_anchor_depth = sum_depth / valid_frames_in_window

        # --- DER FEHLENDE CODE WURDE HIER WIEDER EINGEFÜGT! ---
        if use_pinhole_math and echte_groesse > 0.0:
            expected_norm = (echte_groesse * fx) / avg_anchor_depth
        else:
            expected_norm = (avg_anchor_depth * slope) + intercept

        anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0
        
        # Dieser Log wird dir jetzt wieder deinen gewohnten Faktor (z.B. 1.220x) anzeigen!
        log_messages.append(f"Skalierungs-Faktor = {expected_norm:.1f} / {avg_anchor_norm:.1f} = {anchor_scale:.3f}x")

        # --- 1. 2D Daten Skalieren ---
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

        # --- 2. 3D NLF DATEN SKALIEREN (Kamera-Translation) ---
        nlf_data_scaled = None
        log_messages.append("\n=== NLF 3D DATA SCALING DETAILED LOG ===")
        log_messages.append("Methode: Optische Translation auf der Kamera-Achse (verhindert Schulterverzerrung).")
        
        if video_nlf_data is not None:
            try:
                nlf_data_scaled = copy.deepcopy(video_nlf_data)
                is_dict = isinstance(nlf_data_scaled, dict)
                keys_to_scale = [k for k in ['joints3d_nonparam', 'joints3d', 'joints', 'kps3d'] if k in nlf_data_scaled] if is_dict else [None]
                skalierte_frames_gesamt = 0

                for key in keys_to_scale:
                    target_list = nlf_data_scaled[key][0] if key is not None else nlf_data_scaled
                    if target_list is None or len(target_list) == 0: continue
                    new_target_list = []

                    for frame_idx in range(len(target_list)):
                        frame_data = target_list[frame_idx]
                        if frame_data is None or len(frame_data) == 0:
                            new_target_list.append(frame_data)
                            continue
                            
                        if isinstance(frame_data, torch.Tensor):
                            dim = frame_data.dim()
                            pts = frame_data[0] if dim == 3 else frame_data
                            
                            if len(pts) > 0:
                                # Wir nutzen die Mitte der Person als Anker
                                pivot_x = pts[:, 0].mean()
                                pivot_y = pts[:, 1].max() # Füße
                                pivot_z = pts[:, 2].mean()
                                
                                # Berechnung des Schiebungsvektors basierend auf dem Skalierungs-Faktor
                                delta_x = (pivot_x / anchor_scale) - pivot_x
                                delta_y = (pivot_y / anchor_scale) - pivot_y
                                delta_z = (pivot_z / anchor_scale) - pivot_z
                                
                                scaled_frame = frame_data.clone()
                                
                                # Person unverzerrt in 3D näher/weiter zur Kamera schieben
                                if dim == 3:
                                    scaled_frame[0, :, 0] += delta_x
                                    scaled_frame[0, :, 1] += delta_y
                                    scaled_frame[0, :, 2] += delta_z
                                else:
                                    scaled_frame[:, 0] += delta_x
                                    scaled_frame[:, 1] += delta_y
                                    scaled_frame[:, 2] += delta_z

                                if frame_idx == 0:
                                    log_messages.append(f"Frame 0: Person um {delta_z:.1f} Einheiten auf der Z-Achse verschoben.")
                                    
                                # Füße an die neue, verschobene Position anhängen
                                if len(pts) > 13:
                                    r_ankle = scaled_frame[0][10] if dim == 3 else scaled_frame[10]
                                    l_ankle = scaled_frame[0][13] if dim == 3 else scaled_frame[13]
                                    
                                    r_toe, l_toe = r_ankle.clone(), l_ankle.clone()
                                    r_toe[1] += 0.05 * anchor_scale; r_toe[2] -= 0.10 * anchor_scale
                                    l_toe[1] += 0.05 * anchor_scale; l_toe[2] -= 0.10 * anchor_scale
                                    
                                    feet_t = torch.tensor([l_toe.tolist(), r_toe.tolist()], dtype=scaled_frame.dtype, device=scaled_frame.device)
                                    scaled_frame = torch.cat((scaled_frame, feet_t.unsqueeze(0)), dim=1) if dim == 3 else torch.cat((scaled_frame, feet_t), dim=0)

                                new_target_list.append(scaled_frame)
                                skalierte_frames_gesamt += 1
                            else:
                                new_target_list.append(frame_data.clone())
                        else:
                            new_target_list.append(frame_data)
                            
                    if key is not None:
                        nlf_data_scaled[key][0] = new_target_list

                log_messages.append(f"ERFOLG: 3D Form unversehrt (keine schmalen Schultern mehr). Translation umgesetzt.")

            except Exception as e:
                log_messages.append(f"FEHLER: {e}\n{traceback.format_exc()}")
                nlf_data_scaled = video_nlf_data

        return (pose_data_copy, "\n".join(log_messages), nlf_data_scaled)

# ======================================================================
# 2. PoseGlobalPerspectiveScalerV35 (DIE PERFEKTE LÖSUNG: Lokales 3D Pivot-Scaling)
# ======================================================================
class PoseGlobalPerspectiveScalerV35:
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
        import traceback

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V35 GLOBAL SCALER LOG ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data)

        # Configs aus Calibration laden (Versteht reparierte Waden-Logik)
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if "calf" in true_3d_bones:
            log_messages.append(f"V35 hat die reparierten 3D-Längen verstanden. Soll-Wade: {true_3d_bones['calf']:.3f}")

        slope = calibration_data.get("perspective_slope", 0.0)
        intercept = calibration_data.get("perspective_intercept", 1.0)
        is_inverted = calibration_data.get("is_depth_inverted", False)
        norm_method = calibration_data.get("norm_method", "Dynamic Full-Body")
        use_pinhole_math = calibration_data.get("use_pinhole_math", True)
        echte_groesse = calibration_data.get("echte_groesse", 1.75)
        fx = calibration_data.get("focal_length_fx", 500.0)
        
        depth_np = video_depth_map.cpu().numpy() if hasattr(video_depth_map, 'cpu') else video_depth_map
        H, W = depth_np.shape[1], depth_np.shape[2]

        def is_valid_point(kps, confs, idx):
            if kps is None or idx >= len(kps): return False
            pt = kps[idx]
            if pt is None or len(pt) < 2: return False
            c = float(confs[idx]) if (confs is not None and idx < len(confs)) else (float(pt[2]) if len(pt)>=3 else 1.0)
            return c >= min_confidence

        def is_frontal_2d(meta):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            if is_valid_point(kps, confs, 2) and is_valid_point(kps, confs, 5) and is_valid_point(kps, confs, 8) and is_valid_point(kps, confs, 11):
                shoulder_w = abs(kps[2][0] - kps[5][0])
                hip_w = abs(kps[8][0] - kps[11][0])
                if hip_w > 0: return (shoulder_w / hip_w) > frontal_2d_threshold
            return False

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
            if not (is_frontal_3d(i) if frontal_method == "3D_NLF" and video_nlf_data is not None else is_frontal_2d(meta)): continue
            
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
            depth_vals = [depth_np[v_idx, int(py), int(px)] for px, py in zip(valid_x_d, valid_y_d) if 0 <= int(px) < W and 0 <= int(py) < H]
            
            frame_depth = float(np.mean(depth_vals)) if depth_vals else 0.5
            if is_inverted: frame_depth = 1.0 / max(frame_depth, 0.0001)

            sum_norm += norm_val
            sum_depth += frame_depth
            valid_frames_in_window += 1

        if valid_frames_in_window == 0: 
            return (pose_data_copy, "Fehler: Anchor-Window ungültig.", video_nlf_data)

        avg_anchor_norm = sum_norm / valid_frames_in_window
        avg_anchor_depth = sum_depth / valid_frames_in_window

        if use_pinhole_math and echte_groesse > 0.0:
            expected_norm = (echte_groesse * fx) / avg_anchor_depth
        else:
            expected_norm = (avg_anchor_depth * slope) + intercept

        anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0
        
        log_messages.append(f"Skalierungs-Faktor = {expected_norm:.1f} / {avg_anchor_norm:.1f} = {anchor_scale:.3f}x")

        # --- 1. 2D Daten Skalieren ---
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

        # --- 2. 3D NLF DATEN SKALIEREN (LOKALES 3D PIVOT SCALING) ---
        nlf_data_scaled = None
        log_messages.append("\n=== NLF 3D DATA SCALING DETAILED LOG ===")
        log_messages.append("Methode: Lokales 3D-Pivot-Scaling. (Skaliert X, Y UND Z um das lokale Zentrum der Person).")
        
        if video_nlf_data is not None:
            try:
                nlf_data_scaled = copy.deepcopy(video_nlf_data)
                is_dict = isinstance(nlf_data_scaled, dict)
                keys_to_scale = [k for k in ['joints3d_nonparam', 'joints3d', 'joints', 'kps3d'] if k in nlf_data_scaled] if is_dict else [None]
                skalierte_frames_gesamt = 0

                for key in keys_to_scale:
                    target_list = nlf_data_scaled[key][0] if key is not None else nlf_data_scaled
                    if target_list is None or len(target_list) == 0: continue
                    new_target_list = []

                    for frame_idx in range(len(target_list)):
                        frame_data = target_list[frame_idx]
                        if frame_data is None or len(frame_data) == 0:
                            new_target_list.append(frame_data)
                            continue
                            
                        if isinstance(frame_data, torch.Tensor):
                            dim = frame_data.dim()
                            pts = frame_data[0] if dim == 3 else frame_data
                            
                            if len(pts) > 0:
                                # 1. Wir ermitteln den physikalischen Schwerpunkt der Person
                                pivot_x = pts[:, 0].mean()
                                pivot_y = pts[:, 1].max() # Max Y ist meist unten bei den Füßen
                                pivot_z = pts[:, 2].mean() # Z-Mitte der Person
                                
                                scaled_frame = frame_data.clone()
                                
                                # 2. PERFEKTE 3D SKALIERUNG: Wir skalieren alle Achsen (auch Z!), 
                                # aber ausgehend von ihrem eigenen Pivot. 
                                # Sie bleibt am exakt gleichen Platz (Z_pivot), wächst aber proportional in alle Richtungen!
                                if dim == 3:
                                    scaled_frame[0, :, 0] = pivot_x + (pts[:, 0] - pivot_x) * anchor_scale
                                    scaled_frame[0, :, 1] = pivot_y + (pts[:, 1] - pivot_y) * anchor_scale
                                    scaled_frame[0, :, 2] = pivot_z + (pts[:, 2] - pivot_z) * anchor_scale
                                else:
                                    scaled_frame[:, 0] = pivot_x + (pts[:, 0] - pivot_x) * anchor_scale
                                    scaled_frame[:, 1] = pivot_y + (pts[:, 1] - pivot_y) * anchor_scale
                                    scaled_frame[:, 2] = pivot_z + (pts[:, 2] - pivot_z) * anchor_scale

                                if frame_idx == 0:
                                    log_messages.append(f"Frame 0 Pivot-Z: {pivot_z:.1f}. 3D-Volumen proportional vergrößert!")
                                    
                                # 3. Füße physikalisch korrekt und dynamisch berechnen
                                if len(pts) > 13:
                                    r_knee, r_ankle = (scaled_frame[0][9], scaled_frame[0][10]) if dim == 3 else (scaled_frame[9], scaled_frame[10])
                                    l_knee, l_ankle = (scaled_frame[0][12], scaled_frame[0][13]) if dim == 3 else (scaled_frame[12], scaled_frame[13])
                                    
                                    # Die Zehe ist eine Verlängerung des Schienbeins (ca. 15%)
                                    shin_len_r = torch.norm(r_ankle - r_knee)
                                    shin_len_l = torch.norm(l_ankle - l_knee)
                                    
                                    r_toe = r_ankle + (r_ankle - r_knee) * 0.15
                                    l_toe = l_ankle + (l_ankle - l_knee) * 0.15
                                    
                                    # Zusätzlich leicht nach vorne drücken (Richtung Kamera)
                                    r_toe[2] -= shin_len_r * 0.25 
                                    l_toe[2] -= shin_len_l * 0.25
                                    
                                    feet_t = torch.tensor([l_toe.tolist(), r_toe.tolist()], dtype=scaled_frame.dtype, device=scaled_frame.device)
                                    scaled_frame = torch.cat((scaled_frame, feet_t.unsqueeze(0)), dim=1) if dim == 3 else torch.cat((scaled_frame, feet_t), dim=0)

                                new_target_list.append(scaled_frame)
                                skalierte_frames_gesamt += 1
                            else:
                                new_target_list.append(frame_data.clone())
                        else:
                            new_target_list.append(frame_data)
                            
                    if key is not None:
                        nlf_data_scaled[key][0] = new_target_list

                log_messages.append(f"ERFOLG: 3D Form ist absolut intakt. 3D-Volumen & 2D-Projektion matchen nun perfekt.")

            except Exception as e:
                log_messages.append(f"FEHLER: {e}\n{traceback.format_exc()}")
                nlf_data_scaled = video_nlf_data

        return (pose_data_copy, "\n".join(log_messages), nlf_data_scaled)

# ======================================================================
# 2. PoseGlobalPerspectiveScalerV37 (Pixel-Perfektes 3D-to-2D Mapping)
# ======================================================================
class PoseGlobalPerspectiveScalerV37:
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

    RETURN_TYPES = ("POSEDATA", "STRING", "NLFPRED")
    RETURN_NAMES = ("scaled_pose_data", "log_output", "nlf_data_scaled")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"

    def process(self, video_pose_data, calibration_data, video_depth_map, include_head, anchor_window, min_confidence, frontal_method="3D_NLF", frontal_2d_threshold=0.65, frontal_3d_angle_tolerance=20.0, scale_2d_axes="X and Y (Uniform)", video_nlf_data=None):
        import copy
        import numpy as np
        import math
        import torch
        import traceback

        pose_data_copy = copy.deepcopy(video_pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])
        log_messages = ["=== V37 GLOBAL SCALER LOG ==="]

        if not pose_metas: 
            return (pose_data_copy, "Fehler: Keine Pose-Daten.", video_nlf_data)

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

        def is_frontal_2d(meta):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            if is_valid_point(kps, confs, 2) and is_valid_point(kps, confs, 5) and is_valid_point(kps, confs, 8) and is_valid_point(kps, confs, 11):
                shoulder_w = abs(kps[2][0] - kps[5][0])
                hip_w = abs(kps[8][0] - kps[11][0])
                if hip_w > 0: return (shoulder_w / hip_w) > frontal_2d_threshold
            return False

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
            if not (is_frontal_3d(i) if frontal_method == "3D_NLF" and video_nlf_data is not None else is_frontal_2d(meta)): continue
            
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
            return (pose_data_copy, "Fehler: Anchor-Window ungültig.", video_nlf_data)

        avg_anchor_norm = sum_norm / valid_frames_in_window
        avg_anchor_depth = sum_depth / valid_frames_in_window

        if use_pinhole_math and echte_groesse > 0.0:
            expected_norm = (echte_groesse * fx_calib) / avg_anchor_depth
        else:
            expected_norm = (avg_anchor_depth * slope) + intercept

        anchor_scale = expected_norm / avg_anchor_norm if avg_anchor_norm > 0 else 1.0
        scale_x_factor = anchor_scale if scale_2d_axes == "X and Y (Uniform)" else 1.0
        
        log_messages.append(f"Skalierungs-Faktor = {anchor_scale:.3f}x")
        if scale_2d_axes == "Only Y (Height)":
            log_messages.append("INFO: Schultern bleiben in 2D unangetastet (Nur Höhe wird skaliert)!")

        # --- 1. 2D Daten Skalieren & Pivot speichern ---
        frame_pivots = [] # Wir merken uns den Pivot jedes Frames für die 3D Mathematik
        
        for i, meta in enumerate(pose_metas):
            kps = getattr(meta, "kps_body", [])
            confs = getattr(meta, "kps_body_p", None)
            valid_y = [kps[idx][1] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            valid_x = [kps[idx][0] for idx in range(len(kps)) if is_valid_point(kps, confs, idx)]
            
            if not valid_y or not valid_x: 
                frame_pivots.append((0.5, 0.5))
                continue
            
            pivot_y, pivot_x = max(valid_y), np.mean(valid_x)
            frame_pivots.append((pivot_x, pivot_y))
            
            for attr_name in ["kps_body", "kps_lhand", "kps_rhand", "kps_face"]:
                arr = getattr(meta, attr_name, None)
                if arr is not None and len(arr) > 0:
                    for j in range(len(arr)):
                        if len(arr[j]) >= 2 and arr[j][1] > 0:
                            arr[j][0] = pivot_x + (arr[j][0] - pivot_x) * scale_x_factor
                            arr[j][1] = pivot_y + (arr[j][1] - pivot_y) * anchor_scale

        # --- 2. 3D NLF DATEN SKALIEREN (INVERSE PROJEKTIONSMATHEMATIK) ---
        nlf_data_scaled = None
        log_messages.append("\n=== NLF 3D DATA SCALING DETAILED LOG ===")
        log_messages.append("Methode: Inverse Reprojektion (Zwingt 3D Daten, der fehlerfreien 2D Pose auf den Pixel genau zu folgen).")
        
        if video_nlf_data is not None:
            try:
                # Wir berechnen exakt die gleichen Kamera-Parameter wie nlf_render.py
                fov_degrees = 55.0
                fov_radians = fov_degrees * math.pi / 180.0
                larger_side = max(H, W)
                f_cam = larger_side / (math.tan(fov_radians / 2.0) * 2.0)
                c_x = W / 2.0
                c_y = H / 2.0
                
                nlf_data_scaled = copy.deepcopy(video_nlf_data)
                is_dict = isinstance(nlf_data_scaled, dict)
                keys_to_scale = [k for k in ['joints3d_nonparam', 'joints3d', 'joints', 'kps3d'] if k in nlf_data_scaled] if is_dict else [None]
                skalierte_frames_gesamt = 0

                for key in keys_to_scale:
                    target_list = nlf_data_scaled[key][0] if key is not None else nlf_data_scaled
                    if target_list is None or len(target_list) == 0: continue
                    new_target_list = []

                    for frame_idx in range(len(target_list)):
                        frame_data = target_list[frame_idx]
                        if frame_data is None or len(frame_data) == 0:
                            new_target_list.append(frame_data)
                            continue
                            
                        if isinstance(frame_data, torch.Tensor):
                            dim = frame_data.dim()
                            
                            # 1. Wir hängen ZUERST die Füße an das Original-3D-Skelett an, 
                            # damit sie danach ebenfalls perfekt mit-projiziert werden.
                            pts = frame_data[0] if dim == 3 else frame_data
                            if len(pts) > 13:
                                r_knee, r_ankle = (pts[9], pts[10])
                                l_knee, l_ankle = (pts[12], pts[13])
                                
                                shin_len_r = torch.norm(r_ankle - r_knee)
                                shin_len_l = torch.norm(l_ankle - l_knee)
                                
                                r_toe = r_ankle + (r_ankle - r_knee) * 0.15
                                l_toe = l_ankle + (l_ankle - l_knee) * 0.15
                                r_toe[2] -= shin_len_r * 0.25 
                                l_toe[2] -= shin_len_l * 0.25
                                
                                feet_t = torch.stack([l_toe, r_toe]).to(frame_data.dtype).to(frame_data.device)
                                if dim == 3:
                                    frame_data = torch.cat((frame_data, feet_t.unsqueeze(0)), dim=1)
                                else:
                                    frame_data = torch.cat((frame_data, feet_t), dim=0)
                            
                            # 2. Jetzt wenden wir die mathematische Magie an!
                            scaled_frame = frame_data.clone()
                            
                            # Hole den 2D Pivot für genau diesen Frame und wandle ihn in Pixel um
                            p_x, p_y = frame_pivots[frame_idx] if frame_idx < len(frame_pivots) else (0.5, 0.5)
                            u_pivot = p_x * W
                            v_pivot = p_y * H
                            
                            if dim == 3:
                                X, Y, Z = scaled_frame[0, :, 0], scaled_frame[0, :, 1], scaled_frame[0, :, 2]
                                # Formel: Skaliere X/Y + Kompensiere die perspektivische Z-Verzerrung
                                scaled_frame[0, :, 0] = X * scale_x_factor + Z * (u_pivot - c_x) * (1.0 - scale_x_factor) / f_cam
                                scaled_frame[0, :, 1] = Y * anchor_scale + Z * (v_pivot - c_y) * (1.0 - anchor_scale) / f_cam
                                # Z bleibt völlig unberührt!
                            else:
                                X, Y, Z = scaled_frame[:, 0], scaled_frame[:, 1], scaled_frame[:, 2]
                                scaled_frame[:, 0] = X * scale_x_factor + Z * (u_pivot - c_x) * (1.0 - scale_x_factor) / f_cam
                                scaled_frame[:, 1] = Y * anchor_scale + Z * (v_pivot - c_y) * (1.0 - anchor_scale) / f_cam

                            if frame_idx == 0:
                                log_messages.append("Frame 0: Projektions-Matrix berechnet und invertiert angewandt.")

                            new_target_list.append(scaled_frame)
                            skalierte_frames_gesamt += 1
                        else:
                            new_target_list.append(frame_data)
                            
                    if key is not None:
                        nlf_data_scaled[key][0] = new_target_list

                log_messages.append(f"ERFOLG: 3D Rendern wird nun mathematisch auf den Pixel genau dem 2D-Skelett entsprechen! (Keine hohen Schultern)")

            except Exception as e:
                log_messages.append(f"FEHLER: {e}\n{traceback.format_exc()}")
                nlf_data_scaled = video_nlf_data

        return (pose_data_copy, "\n".join(log_messages), nlf_data_scaled)

# ======================================================================
# 2. PoseGlobalPerspectiveScalerV38 (Der geniale Architektur-Fix)
# ======================================================================
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


# ======================================================================
# 3. RenderNLFPosesDirect (Mit detailliertem Kamera-Debugging & Log-Ausgang)
# ======================================================================
class RenderNLFPosesDirect2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    # NEU: Ein String-Output für das Logging!
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Rendert NLF-Daten und wendet smarte Kamera-Zooms an."

    def process(self, nlf_poses, width, height, render_backend="taichi", dw_poses_fallback=None, nlf_render_config="{}"):
        import copy
        import json
        import torch
        import numpy as np
        import traceback
        from .NLFPoseExtract.nlf_render import render_multi_nlf_as_images, render_nlf_as_images, intrinsic_matrix_from_field_of_view
        
        log_messages = ["=== RENDER NLF POSES DIRECT LOG ==="]

        if render_backend == "taichi":
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
                log_messages.append("Render-Backend: Taichi GPU initialisiert.")
            except Exception as e:
                render_backend = "torch"
                log_messages.append(f"WARNUNG: Taichi GPU fehlgeschlagen. Nutze Torch. Fehler: {e}")
        else:
            log_messages.append("Render-Backend: Torch.")

        try:
            pose_input = nlf_poses['joints3d_nonparam'][0] if isinstance(nlf_poses, dict) else nlf_poses
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            
            if len(pose_input) > 0 and pose_input[0] is not None:
                log_messages.append(f"Erfolgreich geladen: {len(pose_input)} Frames.")
            else:
                log_messages.append("WARNUNG: pose_input ist leer!")

            # 1. Standard Kamera Matrix erstellen
            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
            log_messages.append(f"\nOriginal Intrinsic Matrix:\n{intrinsic_matrix.round(2)}")
            
            # 2. DEN KLUGE KAMERA ZOOM ANWENDEN (aus Config)
            try:
                config = json.loads(nlf_render_config)
                log_messages.append(f"\nEmpfangene Config: {config}")
                
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    
                    # Pivot-Punkte (sollten normalisiert 0.0 - 1.0 sein)
                    pivot_x_norm = float(config["pivot_x"])
                    pivot_y_norm = float(config["pivot_y"])
                    
                    # In absolute Pixel umrechnen
                    p_x = pivot_x_norm * width
                    p_y = pivot_y_norm * height
                    
                    log_messages.append(f"Kamera Zoom X: {scale_x:.3f} | Zoom Y: {scale_y:.3f}")
                    log_messages.append(f"Kamera Fokus-Punkt (Pixel): X={p_x:.1f}, Y={p_y:.1f}")
                    
                    cx_alt = intrinsic_matrix[0, 2]
                    cy_alt = intrinsic_matrix[1, 2]
                    
                    # KAMERA MAGIE: Wir verändern die Brennweite (Zoom) und verschieben die Linse!
                    intrinsic_matrix[0, 0] *= scale_x  
                    intrinsic_matrix[1, 1] *= scale_y  
                    intrinsic_matrix[0, 2] = cx_alt * scale_x + p_x * (1.0 - scale_x) 
                    intrinsic_matrix[1, 2] = cy_alt * scale_y + p_y * (1.0 - scale_y) 
                    
                    log_messages.append(f"Center X verschoben: {cx_alt:.1f} -> {intrinsic_matrix[0, 2]:.1f}")
                    log_messages.append(f"Center Y verschoben: {cy_alt:.1f} -> {intrinsic_matrix[1, 2]:.1f}")
                    log_messages.append(f"\nModifizierte Intrinsic Matrix:\n{intrinsic_matrix.round(2)}")
                else:
                    log_messages.append("Kein 'anchor_scale' in Config. Rendere in Standard-Größe.")
            except Exception as e:
                log_messages.append(f"Fehler beim Lesen der Config (Rendere Standard): {e}")

            # 3. Rendern
            log_messages.append("\nStarte Render-Prozess...")
            is_multi = False
            if len(pose_input) > 0:
                if isinstance(pose_input[0], list):
                    is_multi = len(pose_input[0]) > 1
                elif isinstance(pose_input[0], torch.Tensor) and pose_input[0].dim() == 3:
                    is_multi = pose_input[0].shape[0] > 1

            if is_multi:
                frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)
            else:
                frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)

            # Prüfe auf schwarze Bilder
            black_frames = 0
            for frame in frames_np:
                if np.max(frame) < 5: # Wenn der hellste Pixel fast schwarz ist
                    black_frames += 1
            
            if black_frames > 0:
                log_messages.append(f"KRITISCH: {black_frames} von {len(frames_np)} generierten Frames sind komplett SCHWARZ! Die Kamera hat die Person vermutlich aus dem Bildbereich geschoben.")
            else:
                log_messages.append("Rendering abgeschlossen. Bilder enthalten Pixel-Daten (nicht schwarz).")

            frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages))

        except Exception as e:
            log_messages.append(f"FATALER ABSTURZ BEIM RENDERN: {e}")
            log_messages.append("--- TRACEBACK ---")
            log_messages.append(traceback.format_exc())
            log_messages.append("-----------------")
            
            empty_img = torch.zeros((1, height, width, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            return (empty_img, empty_mask, "\n".join(log_messages))

# ======================================================================
# 3. RenderNLFPosesDirect (Bugfix: Absolute vs. Normalisierte Pixel)
# ======================================================================
class RenderNLFPosesDirect3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "Rendert NLF-Daten und wendet smarte Kamera-Zooms an."

    def process(self, nlf_poses, width, height, render_backend="taichi", dw_poses_fallback=None, nlf_render_config="{}"):
        import copy
        import json
        import torch
        import numpy as np
        import traceback
        from .NLFPoseExtract.nlf_render import render_multi_nlf_as_images, render_nlf_as_images, intrinsic_matrix_from_field_of_view
        
        log_messages = ["=== RENDER NLF POSES DIRECT LOG ==="]

        if render_backend == "taichi":
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
                log_messages.append("Render-Backend: Taichi GPU initialisiert.")
            except Exception as e:
                render_backend = "torch"
                log_messages.append(f"WARNUNG: Taichi GPU fehlgeschlagen. Nutze Torch. Fehler: {e}")
        else:
            log_messages.append("Render-Backend: Torch.")

        try:
            pose_input = nlf_poses['joints3d_nonparam'][0] if isinstance(nlf_poses, dict) else nlf_poses
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            
            if len(pose_input) > 0 and pose_input[0] is not None:
                log_messages.append(f"Erfolgreich geladen: {len(pose_input)} Frames.")
            else:
                log_messages.append("WARNUNG: pose_input ist leer!")

            # 1. Standard Kamera Matrix erstellen
            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
            log_messages.append(f"\nOriginal Intrinsic Matrix:\n{intrinsic_matrix.round(2)}")
            
            # 2. DEN KLUGEN KAMERA ZOOM ANWENDEN (aus Config)
            try:
                config = json.loads(nlf_render_config)
                log_messages.append(f"\nEmpfangene Config: {config}")
                
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    
                    # Pivot-Punkte auslesen
                    p_x = float(config["pivot_x"])
                    p_y = float(config["pivot_y"])
                    
                    # BUGFIX: Prüfen ob Werte normalisiert (0.0 - 1.0) oder bereits absolute Pixel sind!
                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x = p_x * width
                        p_y = p_y * height
                        log_messages.append("Info: Pivot-Werte waren normalisiert, in Pixel umgerechnet.")
                    else:
                        log_messages.append("Info: Pivot-Werte sind bereits absolute Pixel (Keine Multiplikation nötig).")
                    
                    log_messages.append(f"Kamera Zoom X: {scale_x:.3f} | Zoom Y: {scale_y:.3f}")
                    log_messages.append(f"Kamera Fokus-Punkt (Pixel): X={p_x:.1f}, Y={p_y:.1f}")
                    
                    cx_alt = intrinsic_matrix[0, 2]
                    cy_alt = intrinsic_matrix[1, 2]
                    
                    # KAMERA MAGIE: Wir verändern die Brennweite (Zoom) und verschieben die Linse!
                    intrinsic_matrix[0, 0] *= scale_x  
                    intrinsic_matrix[1, 1] *= scale_y  
                    intrinsic_matrix[0, 2] = cx_alt * scale_x + p_x * (1.0 - scale_x) 
                    intrinsic_matrix[1, 2] = cy_alt * scale_y + p_y * (1.0 - scale_y) 
                    
                    log_messages.append(f"Center X verschoben: {cx_alt:.1f} -> {intrinsic_matrix[0, 2]:.1f}")
                    log_messages.append(f"Center Y verschoben: {cy_alt:.1f} -> {intrinsic_matrix[1, 2]:.1f}")
                else:
                    log_messages.append("Kein 'anchor_scale' in Config. Rendere in Standard-Größe.")
            except Exception as e:
                log_messages.append(f"Fehler beim Lesen der Config (Rendere Standard): {e}")

            # 3. Rendern
            log_messages.append("\nStarte Render-Prozess...")
            is_multi = False
            if len(pose_input) > 0:
                if isinstance(pose_input[0], list):
                    is_multi = len(pose_input[0]) > 1
                elif isinstance(pose_input[0], torch.Tensor) and pose_input[0].dim() == 3:
                    is_multi = pose_input[0].shape[0] > 1

            if is_multi:
                frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)
            else:
                frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)

            # Prüfe auf schwarze Bilder
            black_frames = 0
            for frame in frames_np:
                if np.max(frame) < 5: 
                    black_frames += 1
            
            if black_frames > 0:
                log_messages.append(f"WARNUNG: {black_frames} von {len(frames_np)} Frames sind SCHWARZ!")
            else:
                log_messages.append("Rendering abgeschlossen. Bilder sind in Ordnung (nicht schwarz).")

            frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages))

        except Exception as e:
            log_messages.append(f"FATALER ABSTURZ BEIM RENDERN: {e}")
            log_messages.append("--- TRACEBACK ---")
            log_messages.append(traceback.format_exc())
            log_messages.append("-----------------")
            
            empty_img = torch.zeros((1, height, width, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            return (empty_img, empty_mask, "\n".join(log_messages))


class RenderNLFPosesDirect4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "V4: Rendert NLF-Daten mit smarten Kamera-Zooms UND skaliert das 3D-Volumen, um überdicke Zylinder zu verhindern."

    def process(self, nlf_poses, width, height, render_backend="taichi", dw_poses_fallback=None, nlf_render_config="{}"):
        import copy
        import json
        import torch
        import numpy as np
        import traceback
        from .NLFPoseExtract.nlf_render import render_multi_nlf_as_images, render_nlf_as_images, intrinsic_matrix_from_field_of_view

        log_messages = ["=== RENDER NLF POSES DIRECT V4 LOG ==="]

        if render_backend == "taichi":
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
                log_messages.append("Render-Backend: Taichi GPU initialisiert.")
            except Exception as e:
                render_backend = "torch"
                log_messages.append(f"WARNUNG: Taichi GPU fehlgeschlagen. Nutze Torch. Fehler: {e}")
        else:
            log_messages.append("Render-Backend: Torch.")

        try:
            pose_input = nlf_poses['joints3d_nonparam'][0] if isinstance(nlf_poses, dict) else nlf_poses
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None

            if len(pose_input) > 0 and pose_input[0] is not None:
                log_messages.append(f"Erfolgreich geladen: {len(pose_input)} Frames.")
            else:
                log_messages.append("WARNUNG: pose_input ist leer!")

            # 1. Standard Kamera Matrix erstellen
            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
            log_messages.append(f"\nOriginal Intrinsic Matrix:\n{intrinsic_matrix.round(2)}")

            # Pose Clone für In-Place Modifikationen (verhindert, dass wir die globalen Daten zerstören)
            pose_input_cloned = []
            if pose_input is not None:
                for frame in pose_input:
                    if frame is not None:
                        pose_input_cloned.append(frame.clone())
                    else:
                        pose_input_cloned.append(None)
            pose_input = pose_input_cloned

            # 2. DEN KLUGEN KAMERA ZOOM ANWENDEN (aus Config)
            try:
                config = json.loads(nlf_render_config)
                log_messages.append(f"\nEmpfangene Config: {config}")

                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))

                    # Pivot-Punkte auslesen
                    p_x = float(config["pivot_x"])
                    p_y = float(config["pivot_y"])

                    # BUGFIX: Prüfen ob Werte normalisiert (0.0 - 1.0) oder bereits absolute Pixel sind!
                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x = p_x * width
                        p_y = p_y * height
                        log_messages.append("Info: Pivot-Werte waren normalisiert, in Pixel umgerechnet.")
                    else:
                        log_messages.append("Info: Pivot-Werte sind bereits absolute Pixel (Keine Multiplikation nötig).")

                    log_messages.append(f"Kamera Zoom X: {scale_x:.3f} | Zoom Y: {scale_y:.3f}")
                    log_messages.append(f"Kamera Fokus-Punkt (Pixel): X={p_x:.1f}, Y={p_y:.1f}")

                    # ==========================================================
                    # NEUER FIX FÜR DAS ZYLINDER-DICKEN-PROBLEM (V4):
                    # ==========================================================
                    # Wenn wir nur den Kamera-Zoom (focal length) erhöhen, projiziert die Kamera 
                    # die 3D-Zylinder viel dicker, weil das Objekt im 3D-Raum am selben Platz bleibt.
                    # Lösung: Wir skalieren das physische 3D-Skelett um den Faktor 'scale_y'. 
                    # Dadurch rutscht es im Z-Raum weiter weg, was die Zylinder relativ zur Kamera 
                    # wieder dünner macht! Der Kamera-Zoom gleicht die Entfernung dann auf 2D-Pixelebene
                    # wieder exakt aus. Die 3D-Volumina und die Dicke sind gerettet!
                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None:
                            pose_input[frame_idx] = pose_input[frame_idx] * scale_y
                    
                    log_messages.append(f"ERFOLG: 3D-Volumen und Z-Tiefe wurden um {scale_y:.3f}x skaliert, um übergroße Zylinder zu verhindern!")

                    cx_alt = intrinsic_matrix[0, 2]
                    cy_alt = intrinsic_matrix[1, 2]

                    # KAMERA MAGIE: Wir verändern die Brennweite (Zoom) und verschieben die Linse!
                    intrinsic_matrix[0, 0] *= scale_x
                    intrinsic_matrix[1, 1] *= scale_y
                    intrinsic_matrix[0, 2] = cx_alt * scale_x + p_x * (1.0 - scale_x)
                    intrinsic_matrix[1, 2] = cy_alt * scale_y + p_y * (1.0 - scale_y)

                    log_messages.append(f"Center X verschoben: {cx_alt:.1f} -> {intrinsic_matrix[0, 2]:.1f}")
                    log_messages.append(f"Center Y verschoben: {cy_alt:.1f} -> {intrinsic_matrix[1, 2]:.1f}")
                else:
                    log_messages.append("Kein 'anchor_scale' in Config. Rendere in Standard-Größe.")

            except Exception as e:
                log_messages.append(f"Fehler beim Lesen der Config (Rendere Standard): {e}")

            # 3. Rendern
            log_messages.append("\nStarte Render-Prozess...")
            is_multi = False
            if len(pose_input) > 0:
                if isinstance(pose_input[0], list):
                    is_multi = len(pose_input[0]) > 1
                elif isinstance(pose_input[0], torch.Tensor) and pose_input[0].dim() == 3:
                    is_multi = pose_input[0].shape[0] > 1

            if is_multi:
                frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)
            else:
                frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)

            # Prüfe auf schwarze Bilder
            black_frames = 0
            for frame in frames_np:
                if np.max(frame) < 5:
                    black_frames += 1
            
            if black_frames > 0:
                log_messages.append(f"WARNUNG: {black_frames} von {len(frames_np)} Frames sind SCHWARZ!")
            else:
                log_messages.append("Rendering abgeschlossen. Bilder sind in Ordnung (nicht schwarz).")

            frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5

            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages))

        except Exception as e:
            log_messages.append(f"FATALER ABSTURZ BEIM RENDERN: {e}")
            log_messages.append("--- TRACEBACK ---")
            log_messages.append(traceback.format_exc())
            log_messages.append("-----------------")
            
            empty_img = torch.zeros((1, height, width, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            return (empty_img, empty_mask, "\n".join(log_messages))


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




class NLFProportionalRetargeterV3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_nlf_data": ("NLFPRED", {"tooltip": "Die originalen 3D NLF Daten"}),
                "calibration_data": ("POSE_CALIBRATION", {"tooltip": "Die Referenz-Daten aus V20"}),
                "frontal_3d_angle_tolerance": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0, "tooltip": "Toleranz für die Frontal-Suche"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING")
    RETURN_NAMES = ("nlf_data_retargeted", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Ultimate"
    DESCRIPTION = "V3: Baut das 3D-Skelett proportional um. Nutzt Best-Frame, um abgeschnittene Beine zu rekonstruieren."

    def process(self, video_nlf_data, calibration_data, frontal_3d_angle_tolerance):
        import copy
        import math
        import torch

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V3 LOG ==="]

        if video_nlf_data is None:
            return (video_nlf_data, "FEHLER: Keine NLF-Daten.")

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        pose_input_3d = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0] if is_dict else nlf_data_retargeted
        
        target_bones = calibration_data.get("true_3d_bones", {})
        if not target_bones:
            log_messages.append("WARNUNG: Keine Knochenlängen in Calibration_data. Gebe Original zurück.")
            return (video_nlf_data, "\n".join(log_messages))

        def get_dist(p1, p2): return float(torch.norm(p1 - p2))
        
        def safe_normalize(vec):
            norm = torch.norm(vec)
            return vec / norm if norm > 1e-6 else vec

        # ==========================================
        # 1. REFERENZ-PROPORTIONEN BERECHNEN (Prozente)
        # ==========================================
        ref_torso = target_bones.get("torso", 1.0)
        if ref_torso <= 0: ref_torso = 1.0
        
        target_props = {}
        for bone_name, length in target_bones.items():
            target_props[bone_name] = length / ref_torso
            
        log_messages.append(f"Referenz-Proportionen aus V20 (Basis Torso = 1.0):")
        for k, v in target_props.items():
            if k != "factor_nah_fern":
                log_messages.append(f"  {k}: {v*100:.1f}%")

        # ==========================================
        # 2. BESTEN NLF-FRAME FINDEN (Längste Beine)
        # ==========================================
        best_idx = 0
        max_leg_proportion = 0.0

        for i, frame_data in enumerate(pose_input_3d):
            if frame_data is None or len(frame_data) == 0: continue
            pose_3d = frame_data[0]
            if len(pose_3d) < 14: continue 
            
            # Frontalitäts-Check
            l_hip, r_hip = pose_3d[11], pose_3d[8]
            dz, dx = abs(l_hip[2] - r_hip[2]), abs(l_hip[0] - r_hip[0])
            angle = math.degrees(math.atan2(dz, dx)) if dx > 0 else 90.0
            if angle > frontal_3d_angle_tolerance: continue

            # Torso Länge
            mid_hip = (pose_3d[8] + pose_3d[11]) / 2.0
            torso_len = get_dist(pose_3d[1], mid_hip)
            if torso_len == 0: continue

            # Bein-Länge prüfen
            r_leg_len = get_dist(pose_3d[8], pose_3d[9]) + get_dist(pose_3d[9], pose_3d[10])
            l_leg_len = get_dist(pose_3d[11], pose_3d[12]) + get_dist(pose_3d[12], pose_3d[13])
            
            current_leg_prop = max(r_leg_len, l_leg_len) / torso_len
            
            if current_leg_prop > max_leg_proportion:
                max_leg_proportion = current_leg_prop
                best_idx = i

        log_messages.append(f"\n---> BESTER FRAME GEFUNDEN: Frame {best_idx}")
        log_messages.append(f"---> Beine waren dort {max_leg_proportion*100:.1f}% der Torso-Länge.")

        # Die Pose des besten Frames für Fallbacks speichern
        best_pose_3d = pose_input_3d[best_idx][0]

        # ==========================================
        # 3. KINEMATIC TREE UMBAUEN (Jeden Frame)
        # ==========================================
        frames_retargeted = 0
        legs_reconstructed = 0

        for i in range(len(pose_input_3d)):
            frame_data = pose_input_3d[i]
            if frame_data is None or len(frame_data) == 0: continue
                
            pts_orig = frame_data[0].clone()
            pts_new = frame_data[0].clone()
            
            mid_hip = (pts_orig[8] + pts_orig[11]) / 2.0
            current_torso_len = get_dist(pts_orig[1], mid_hip)
            if current_torso_len == 0: current_torso_len = 1.0

            def apply_prop_with_fallback(start_pt, target_pt_orig, prop_name, fallback_start, fallback_target):
                nonlocal legs_reconstructed
                target_ratio = target_props.get(prop_name, None)
                if target_ratio is None:
                    generic = prop_name.replace("l_", "").replace("r_", "")
                    target_ratio = target_props.get(generic, None)
                
                vec = target_pt_orig - start_pt
                
                # Wenn der Knochen im aktuellen Frame 0 lang ist (abgeschnitten),
                # klauen wir die Richtung aus dem Best-Frame!
                if torch.norm(vec) < 1e-3:
                    vec = fallback_target - fallback_start
                    if prop_name in ["r_thigh", "l_thigh", "r_calf", "l_calf"]:
                        legs_reconstructed += 1

                if target_ratio is None or torch.norm(vec) < 1e-6:
                    return start_pt + vec 
                
                desired_length = current_torso_len * target_ratio
                dir_vec = safe_normalize(vec)
                return start_pt + (dir_vec * desired_length)

            # 1. HÜFT-BREITE
            pts_new[8] = apply_prop_with_fallback(mid_hip, pts_orig[8], "hip_width", mid_hip, best_pose_3d[8])
            pts_new[11] = apply_prop_with_fallback(mid_hip, pts_orig[11], "hip_width", mid_hip, best_pose_3d[11])

            # 2. SCHULTER-BREITE 
            target_shoulder = target_props.get("shoulder_width", None)
            if target_shoulder:
                desired_half_shoulder = (current_torso_len * target_shoulder) / 2.0
                pts_new[2] = pts_orig[1] + (safe_normalize(pts_orig[2] - pts_orig[1]) * desired_half_shoulder)
                pts_new[5] = pts_orig[1] + (safe_normalize(pts_orig[5] - pts_orig[1]) * desired_half_shoulder)

            # 3. BEINE (Mit Fallback auf Best-Frame, wenn abgeschnitten!)
            pts_new[9] = apply_prop_with_fallback(pts_new[8], pts_orig[9], "r_thigh", best_pose_3d[8], best_pose_3d[9])
            pts_new[10] = apply_prop_with_fallback(pts_new[9], pts_orig[10], "r_calf", best_pose_3d[9], best_pose_3d[10])
            pts_new[12] = apply_prop_with_fallback(pts_new[11], pts_orig[12], "l_thigh", best_pose_3d[11], best_pose_3d[12])
            pts_new[13] = apply_prop_with_fallback(pts_new[12], pts_orig[13], "l_calf", best_pose_3d[12], best_pose_3d[13])

            # 4. ARME
            pts_new[3] = apply_prop_with_fallback(pts_new[2], pts_orig[3], "r_arm", best_pose_3d[2], best_pose_3d[3])
            pts_new[4] = apply_prop_with_fallback(pts_new[3], pts_orig[4], "r_forearm", best_pose_3d[3], best_pose_3d[4])
            pts_new[6] = apply_prop_with_fallback(pts_new[5], pts_orig[6], "l_arm", best_pose_3d[5], best_pose_3d[6])
            pts_new[7] = apply_prop_with_fallback(pts_new[6], pts_orig[7], "l_forearm", best_pose_3d[6], best_pose_3d[7])

            # Füße/Kopf einfach unskaliert anhängen
            if len(pts_orig) > 14:
                pts_new[0] = pts_orig[1] + (pts_orig[0] - pts_orig[1]) 
            if len(pts_orig) > 18:
                pts_new[18] = pts_new[13] + (pts_orig[18] - pts_orig[13]) 
            if len(pts_orig) > 19:
                pts_new[19] = pts_new[10] + (pts_orig[19] - pts_orig[10]) 

            pose_input_3d[i][0] = pts_new
            frames_retargeted += 1

        log_messages.append(f"\nErfolg: {frames_retargeted} Frames umgebaut.")
        log_messages.append(f"Es wurden {legs_reconstructed} Gelenke rekonstruiert (weil sie im Original fehlten)!")

        if is_dict:
            nlf_data_retargeted['joints3d_nonparam'] = [pose_input_3d]
        else:
            nlf_data_retargeted = pose_input_3d

        return (nlf_data_retargeted, "\n".join(log_messages))

class PoseCalibrationV21:
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

    def calibrate(self, pose_nah_scaled, pose_nah_unscaled, depth_nah, pose_fern_scaled, pose_fern_unscaled, depth_fern, norm_method, min_confidence, invert_depth, use_pinhole_math=True, intrinsics_json=None, nlf_data_nah=None, nlf_data_fern=None, config_data="{}"):
        log_messages = ["=== V21 CALIBRATION LOG (H36M/OPENPOSE AUTO-FIX & SYMMETRIE) ==="]
        
        try:
            config = json.loads(config_data)
        except:
            config = {}

        # ---------------------------------------------------------
        # 1. BERECHNUNG VON NORM & TIEFE (Unverändert)
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
        # 2. 3D BONES BERECHNEN (DER FIX FÜR DIE INDIZES!)
        # ---------------------------------------------------------
        def extract_all_3d_bones(nlf_data):
            if nlf_data is None: return None
            try:
                pose_input_3d = nlf_data.get('joints3d_nonparam', [nlf_data])[0]
                if pose_input_3d is not None and len(pose_input_3d) > 0 and len(pose_input_3d[0]) > 0:
                    pose_3d = pose_input_3d[0][0]
                    num_joints = len(pose_3d)

                    def dist_3d(p1, p2):
                        return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))

                    # Wenn 17 Keypoints -> H36M Format (NLF Daten Standard)
                    if num_joints == 17:
                        mid_hip = pose_3d[0] # 0 = Pelvis/Mid-Hip in H36M
                        raw_bones = {
                            "torso": dist_3d(pose_3d[8], mid_hip), # 8 = Neck
                            "shoulder_width": dist_3d(pose_3d[14], pose_3d[11]), # 14 = RShoulder, 11 = LShoulder
                            "hip_width": dist_3d(pose_3d[1], pose_3d[4]), # 1 = RHip, 4 = LHip
                            "r_arm": dist_3d(pose_3d[14], pose_3d[15]), # RShoulder -> RElbow
                            "r_forearm": dist_3d(pose_3d[15], pose_3d[16]), # RElbow -> RWrist
                            "l_arm": dist_3d(pose_3d[11], pose_3d[12]), # LShoulder -> LElbow
                            "l_forearm": dist_3d(pose_3d[12], pose_3d[13]), # LElbow -> LWrist
                            "r_thigh": dist_3d(pose_3d[1], pose_3d[2]), # RHip -> RKnee
                            "r_calf": dist_3d(pose_3d[2], pose_3d[3]), # RKnee -> RAnkle
                            "l_thigh": dist_3d(pose_3d[4], pose_3d[5]), # LHip -> LKnee
                            "l_calf": dist_3d(pose_3d[5], pose_3d[6]) # LKnee -> LAnkle
                        }
                    else:
                        # Fallback auf Standard OpenPose Format
                        mid_hip = [(pose_3d[8][i] + pose_3d[11][i]) / 2.0 for i in range(3)]
                        raw_bones = {
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
                    
                    # SYMMETRIE ERZWINGEN (Beide Seiten mitteln)
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
                    return sym_bones
            except Exception as e:
                log_messages.append(f"Fehler in extract_all_3d_bones: {e}")
            return None

        bones_nah = extract_all_3d_bones(nlf_data_nah)
        bones_fern = extract_all_3d_bones(nlf_data_fern)

        true_3d_bones = {}
        total_3d_height = 0.0

        if bones_nah and bones_fern:
            # Wir nehmen den Durchschnitt beider Kalibrierungs-Frames für maximale Genauigkeit
            for key in bones_nah.keys():
                true_3d_bones[key] = (bones_nah[key] + bones_fern[key]) / 2.0
                
            total_3d_height = true_3d_bones["torso"] + true_3d_bones["r_thigh"] + true_3d_bones["r_calf"]
            head_allowance = config.get("head_allowance_3d", 0.15)
            total_3d_height += head_allowance
            log_messages.append("Erfolg: true_3d_bones symmetrisch berechnet (H36M & OpenPose Support)!")
        elif bones_nah:
            true_3d_bones = bones_nah
        elif bones_fern:
            true_3d_bones = bones_fern
        else:
            log_messages.append("Fehler: Konnte keine 3D-Bones extrahieren.")

        calib_data = {
            "perspective_slope": slope,
            "perspective_intercept": intercept,
            "is_depth_inverted": invert_depth,
            "norm_method": norm_method,
            "use_pinhole_math": use_pinhole_math,
            "focal_length_fx": fx,
            "echte_groesse": echte_groesse,
            "true_3d_bones": true_3d_bones,
            "total_3d_height": total_3d_height,
            "config": config
        }

        return (calib_data, "\n".join(log_messages))

import copy
import numpy as np
import math
import json
import torch

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


class NLFProportionalRetargeterV4:
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

        log_messages = ["=== NLF PROPORTIONAL RETARGETER V4 (VEKTOR KINEMATIK) ==="]
        
        true_3d_bones = calibration_data.get("true_3d_bones", {})
        if not true_3d_bones:
            log_messages.append("FEHLER: Keine true_3d_bones in calibration_data gefunden.")
            return (video_nlf_data, "\n".join(log_messages))

        is_dict = isinstance(video_nlf_data, dict)
        nlf_data_retargeted = copy.deepcopy(video_nlf_data)
        
        # Sicheres extrahieren der Posen
        if is_dict:
            raw_poses = nlf_data_retargeted.get('joints3d_nonparam', [nlf_data_retargeted])[0]
        else:
            raw_poses = nlf_data_retargeted

        # Check: Sind es V22 Prozent-Werte?
        is_normalized = math.isclose(true_3d_bones.get("torso", 0.0), 100.0, abs_tol=1e-3)

        # --- SMPL 3D KINEMATISCHER BAUM ---
        # Definiert, wer an wem hängt. Wenn sich ein Parent verschiebt, wandern alle Kinder mit!
        # 0: Pelvis, 12: Neck
        # 16: L_Shoulder, 18: L_Elbow, 20: L_Wrist
        # 17: R_Shoulder, 19: R_Elbow, 21: R_Wrist
        # 1: L_Hip, 4: L_Knee, 7: L_Ankle
        # 2: R_Hip, 5: R_Knee, 8: R_Ankle
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

        for frame_idx in range(len(raw_poses)):
            frame_data = raw_poses[frame_idx]
            if frame_data is None or len(frame_data) == 0:
                continue
                
            # Sicherstellen, dass wir ein Numpy-Array haben (kompatibel mit Tensor/Listen)
            is_tensor = isinstance(frame_data, torch.Tensor)
            if is_tensor:
                pts = frame_data[0].cpu().numpy().copy() if frame_data.dim() == 3 else frame_data.cpu().numpy().copy()
            else:
                arr = np.array(frame_data)
                pts = arr[0].copy() if arr.ndim == 3 else arr.copy()
            
            pts_new = pts.copy()

            # 1. Torso-Länge als Vektor zwischen Hals [12] und Becken [0] messen
            vec_torso = pts[12] - pts[0]
            current_torso_length = np.linalg.norm(vec_torso)
            
            if current_torso_length < 1e-5:
                continue

            # 2. Ziel-Längen berechnen (Einheiten anhand des Torsos)
            targets = {}
            for k, v in true_3d_bones.items():
                targets[k] = (v / 100.0) * current_torso_length if is_normalized else v

            # 3. KINEMATISCHE OPERATIONEN (Reihenfolge wichtig: Zuerst Körpermitte, dann Extremitäten)
            operations = [
                # Virtuelle Vektoren (Hals zu Schultern, Becken zu Hüften)
                ('shoulder_width', 12, 17), # Hals -> R Schulter
                ('shoulder_width', 12, 16), # Hals -> L Schulter
                ('hip_width', 0, 2),        # Becken -> R Hüfte
                ('hip_width', 0, 1),        # Becken -> L Hüfte
                
                # Arm Vektoren
                ('r_arm', 17, 19),          # R Schulter -> R Ellbogen
                ('r_forearm', 19, 21),      # R Ellbogen -> R Handgelenk
                ('l_arm', 16, 18),          # L Schulter -> L Ellbogen
                ('l_forearm', 18, 20),      # L Ellbogen -> L Handgelenk
                
                # Bein Vektoren
                ('r_thigh', 2, 5),          # R Hüfte -> R Knie
                ('r_calf', 5, 8),           # R Knie -> R Knöchel
                ('l_thigh', 1, 4),          # L Hüfte -> L Knie
                ('l_calf', 4, 7)            # L Knie -> L Knöchel
            ]

            for bone_key, p_idx, c_idx in operations:
                if bone_key not in targets: continue
                if c_idx >= len(pts_new) or p_idx >= len(pts_new): continue
                
                target_len = targets[bone_key]
                # Breite halbieren, da wir vom Zentrum (Hals/Becken) nach außen messen
                if bone_key in ['shoulder_width', 'hip_width']:
                    target_len = target_len / 2.0
                
                # Wir holen uns die *aktuellen* Positionen (falls das Parent sich schon verschoben hat!)
                p_pos = pts_new[p_idx]
                c_pos = pts_new[c_idx]
                
                # Schutz gegen nicht erkannte [0,0,0] Gelenke
                if np.linalg.norm(p_pos) < 1e-5 or np.linalg.norm(c_pos) < 1e-5:
                    continue

                # Den Vektor bilden
                vec = c_pos - p_pos
                curr_len = np.linalg.norm(vec)
                
                if curr_len < 1e-5: continue
                
                # Den Vektor auf die Ziel-Länge ausrichten/skalieren
                dir_vec = vec / curr_len
                new_c_pos = p_pos + (dir_vec * target_len)
                
                # Die Vektor-Verschiebung (Delta)
                delta = new_c_pos - c_pos
                
                # 1. Das Gelenk auf die neue Position setzen
                pts_new[c_idx] += delta
                
                # 2. Den "Arm/das Bein" einfach komplett an der Kette mitwandern lassen!
                descendants = get_all_descendants(c_idx, tree)
                for d in descendants:
                    if d < len(pts_new):
                        # Nur mitnehmen, wenn der Punkt kein [0,0,0]-Fehler ist
                        if np.linalg.norm(pts_new[d]) > 1e-5:
                            pts_new[d] += delta

            # Die berechneten Daten wieder verpacken
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

        log_messages.append(f"Vektor-Retargeting erfolgreich abgeschlossen: {frames_processed} Frames skaliert.")
        return (nlf_data_retargeted, "\n".join(log_messages))
        
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
        
class RenderNLFPosesDirect5:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    # Auf 5 Outputs erweitert
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("image", "mask", "log_output", "scaled_nlf_poses", "node_mappings")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "V4: Rendert NLF-Daten mit smarten Kamera-Zooms UND skaliert das 3D-Volumen, um überdicke Zylinder zu verhindern. Gibt zudem die skalierten NLF-Daten weiter."

    def process(self, nlf_poses, width, height, render_backend="taichi", dw_poses_fallback=None, nlf_render_config="{}"):
        import copy
        import json
        import torch
        import numpy as np
        import traceback
        from .NLFPoseExtract.nlf_render import render_multi_nlf_as_images, render_nlf_as_images, intrinsic_matrix_from_field_of_view
        
        log_messages = ["=== RENDER NLF POSES DIRECT V4 LOG ==="]
        
        if render_backend == "taichi":
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
                log_messages.append("Render-Backend: Taichi GPU initialisiert.")
            except Exception as e:
                render_backend = "torch"
                log_messages.append(f"WARNUNG: Taichi GPU fehlgeschlagen. Nutze Torch. Fehler: {e}")
        else:
            log_messages.append("Render-Backend: Torch.")

        # Deepcopy der NLF Poses für den modifizierten Output
        scaled_nlf_poses = copy.deepcopy(nlf_poses)
            
        try:
            pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            
            if len(pose_input) > 0 and pose_input[0] is not None:
                log_messages.append(f"Erfolgreich geladen: {len(pose_input)} Frames.")
            else:
                log_messages.append("WARNUNG: pose_input ist leer!")

            # 1. Standard Kamera Matrix erstellen
            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
            log_messages.append(f"\nOriginal Intrinsic Matrix:\n{intrinsic_matrix.round(2)}")

            # 2. DEN KLUGEN KAMERA ZOOM ANWENDEN (aus Config)
            try:
                config = json.loads(nlf_render_config)
                log_messages.append(f"\nEmpfangene Config: {config}")

                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    
                    # Pivot-Punkte auslesen
                    p_x = float(config["pivot_x"])
                    p_y = float(config["pivot_y"])

                    # Normalisierte vs absolute Pixel prüfen
                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x = p_x * width
                        p_y = p_y * height
                        log_messages.append("Info: Pivot-Werte waren normalisiert, in Pixel umgerechnet.")
                    
                    log_messages.append(f"Kamera Zoom X: {scale_x:.3f} | Zoom Y: {scale_y:.3f}")
                    log_messages.append(f"Kamera Fokus-Punkt (Pixel): X={p_x:.1f}, Y={p_y:.1f}")

                    cx_alt = intrinsic_matrix[0, 2]
                    cy_alt = intrinsic_matrix[1, 2]

                    # KAMERA MAGIE: Brennweite anpassen
                    intrinsic_matrix[0, 0] *= scale_x
                    intrinsic_matrix[1, 1] *= scale_y
                    intrinsic_matrix[0, 2] = cx_alt * scale_x + p_x * (1.0 - scale_x)
                    intrinsic_matrix[1, 2] = cy_alt * scale_y + p_y * (1.0 - scale_y)

                    log_messages.append(f"Center X verschoben: {cx_alt:.1f} -> {intrinsic_matrix[0, 2]:.1f}")
                    log_messages.append(f"Center Y verschoben: {cy_alt:.1f} -> {intrinsic_matrix[1, 2]:.1f}")
                    log_messages.append(f"\nModifizierte Intrinsic Matrix:\n{intrinsic_matrix.round(2)}")

                    # 3D-Volumen Skalierung auf die Daten anwenden, damit die Zylinder nicht überdick werden
                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None and len(pose_input[frame_idx]) > 0:
                            # Wir multiplizieren die 3D-Koordinaten direkt mit dem scale_y / scale_x
                            # (Dadurch passen wir die wirklichen Volumendaten an, die dann als Output rausgehen)
                            pose_input[frame_idx] *= scale_y 
                            
                else:
                    log_messages.append("Kein 'anchor_scale' in Config. Rendere in Standard-Größe.")
                    
            except Exception as e:
                log_messages.append(f"Fehler beim Lesen der Config (Rendere Standard): {e}")

            # 3. Rendern mit den (jetzt modifizierten) pose_input Daten
            log_messages.append("\nStarte Render-Prozess...")
            is_multi = False
            if len(pose_input) > 0:
                if isinstance(pose_input[0], list):
                    is_multi = len(pose_input[0]) > 1
                elif isinstance(pose_input[0], torch.Tensor) and pose_input[0].dim() == 3:
                    is_multi = pose_input[0].shape[0] > 1
                    
            if is_multi:
                frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)
            else:
                frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)

            # Prüfe auf schwarze Bilder
            black_frames = 0
            for frame in frames_np:
                if np.max(frame) < 5:
                    black_frames += 1
            if black_frames > 0:
                log_messages.append(f"WARNUNG: {black_frames} von {len(frames_np)} Frames sind SCHWARZ!")
            else:
                log_messages.append("Rendering abgeschlossen. Bilder sind in Ordnung (nicht schwarz).")

            frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            # Dictionary wieder zusammensetzen, falls es eins war
            if isinstance(scaled_nlf_poses, dict):
                scaled_nlf_poses['joints3d_nonparam'] = [pose_input]
            else:
                scaled_nlf_poses = pose_input

            # Node Mappings vorbereiten (JSON String für Workflow Output)
            node_mappings = json.dumps({
                "node_name": "RenderNLFPosesDirect4",
                "scaled_frames": len(pose_input),
                "camera_matrix_modified": True if "anchor_scale" in config else False
            })

            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), scaled_nlf_poses, node_mappings)

        except Exception as e:
            log_messages.append(f"FATALER ABSTURZ BEIM RENDERN: {e}")
            log_messages.append("--- TRACEBACK ---")
            log_messages.append(traceback.format_exc())
            log_messages.append("-----------------")
            
            empty_img = torch.zeros((1, height, width, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            
            return (empty_img, empty_mask, "\n".join(log_messages), nlf_poses, "{}")



class RenderNLFPosesDirect6:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die originalen NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Für Hände/Gesicht"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("image", "mask", "log_output", "scaled_nlf_poses", "node_mappings")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "V5: Backt die Kamera-Config physisch in die 3D-Daten ein und gibt die manipulierten Daten als Output aus."

    def process(self, nlf_poses, width, height, render_backend="taichi", dw_poses_fallback=None, nlf_render_config="{}"):
        import copy
        import json
        import torch
        import numpy as np
        import traceback
        from .NLFPoseExtract.nlf_render import render_multi_nlf_as_images, render_nlf_as_images, intrinsic_matrix_from_field_of_view
        
        log_messages = ["=== RENDER NLF POSES DIRECT V5 LOG ==="]
        
        if render_backend == "taichi":
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
                log_messages.append("Render-Backend: Taichi GPU initialisiert.")
            except Exception as e:
                render_backend = "torch"
                log_messages.append(f"WARNUNG: Taichi GPU fehlgeschlagen. Nutze Torch. Fehler: {e}")
        else:
            log_messages.append("Render-Backend: Torch.")

        # 1. Echter Deepcopy der Daten, damit wir sie manipulieren und ausgeben können
        scaled_nlf_poses = copy.deepcopy(nlf_poses)
            
        try:
            pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            
            if len(pose_input) > 0 and pose_input[0] is not None:
                log_messages.append(f"Erfolgreich geladen: {len(pose_input)} Frames.")

            # Standard Kamera Matrix erstellen
            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
            
            # 2. CONFIG MATHEMATISCH AUF DIE 3D PUNKTE ANWENDEN
            try:
                config = json.loads(nlf_render_config)
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    
                    p_x = float(config["pivot_x"])
                    p_y = float(config["pivot_y"])

                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x = p_x * width
                        p_y = p_y * height
                    
                    # Originale Brennweiten und Kamera-Zentren
                    fx = intrinsic_matrix[0, 0]
                    fy = intrinsic_matrix[1, 1]
                    cx = intrinsic_matrix[0, 2]
                    cy = intrinsic_matrix[1, 2]

                    # Wir übersetzen den Kamera-Zoom in eine 3D-Punkt-Verschiebung 
                    # (Unter Einbezug der Tiefe Z, um die Perspektive zu erhalten)
                    M13 = (cx - p_x) * (scale_x - 1.0) / fx
                    M23 = (cy - p_y) * (scale_y - 1.0) / fy

                    log_messages.append(f"Wende 3D-Transformation an: ScaleX={scale_x:.3f}, ScaleY={scale_y:.3f}")
                    log_messages.append(f"Berechnete perspektivische Shifts: M13={M13:.3f}, M23={M23:.3f}")

                    # ECHTE 3D DATEN MANIPULATION
                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None and len(pose_input[frame_idx]) > 0:
                            pts = pose_input[frame_idx]
                            
                            # Koordinaten extrahieren
                            X = pts[..., 0].clone()
                            Y = pts[..., 1].clone()
                            Z = pts[..., 2].clone()
                            
                            # Punkte transformieren (Z bleibt gleich für korrekte 3D Proportion)
                            pts[..., 0] = X * scale_x + Z * M13
                            pts[..., 1] = Y * scale_y + Z * M23
                            
                    log_messages.append("Erfolg: Die 3D-NLF-Daten wurden physisch im Raum transformiert!")
            except Exception as e:
                log_messages.append(f"Fehler bei 3D Transformation: {e}")

            # 3. Rendern 
            # WICHTIG: Da die Daten nun verändert sind, rendern wir mit der STANDARD intrinsic_matrix
            log_messages.append("\nStarte Render-Prozess mit transformierten Daten...")
            is_multi = False
            if len(pose_input) > 0:
                if isinstance(pose_input[0], list):
                    is_multi = len(pose_input[0]) > 1
                elif isinstance(pose_input[0], torch.Tensor) and pose_input[0].dim() == 3:
                    is_multi = pose_input[0].shape[0] > 1
                    
            if is_multi:
                frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)
            else:
                frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=True, draw_hands=True, render_backend=render_backend)

            frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            # Die Dictionary Struktur sauber verpacken
            if isinstance(scaled_nlf_poses, dict):
                scaled_nlf_poses['joints3d_nonparam'] = [pose_input]
            else:
                scaled_nlf_poses = pose_input

            node_mappings = json.dumps({
                "node_name": "RenderNLFPosesDirect5",
                "status": "success_3d_baked",
                "frames_processed": len(pose_input)
            })

            # Wir geben jetzt die wirklich berechneten (verschobenen/skalierten) 3D NLF Daten weiter!
            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), scaled_nlf_poses, node_mappings)

        except Exception as e:
            log_messages.append(f"FATALER ABSTURZ BEIM RENDERN: {e}")
            log_messages.append(traceback.format_exc())
            empty_img = torch.zeros((1, height, width, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            return (empty_img, empty_mask, "\n".join(log_messages), nlf_poses, "{}")


class RenderNLFPosesDirect7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "Die originalen NLF Daten"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "render_backend": (["taichi", "torch"], {"default": "taichi"}),
                # Hier sind die drei neuen Einstellungen für die Render-Logik:
                "draw_2d": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet 2D Overlay (falls DW Poses vorhanden)"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet das Gesicht (falls DW Poses vorhanden)"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Zeichnet die Hände (falls DW Poses vorhanden)"}),
            },
            "optional": {
                "dw_poses_fallback": ("DWPOSES", {"tooltip": "Referenz-Posen für Hände/Gesicht"}),
                "nlf_render_config": ("STRING", {"forceInput": True, "tooltip": "Die Camera Config aus der Scaler-Node"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "NLFPRED", "STRING")
    RETURN_NAMES = ("image", "mask", "log_output", "scaled_nlf_poses", "node_mappings")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/SCAIL"
    DESCRIPTION = "V6: Backt die Kamera-Config physisch in die 3D-Daten ein, gibt sie aus UND übernimmt die Render-Einstellungen (Draw 2D/Face/Hands) für mehr Kontrolle."

    def process(self, nlf_poses, width, height, render_backend="taichi", draw_2d=True, draw_face=True, draw_hands=True, dw_poses_fallback=None, nlf_render_config="{}"):
        import copy
        import json
        import torch
        import numpy as np
        import traceback
        from .NLFPoseExtract.nlf_render import render_multi_nlf_as_images, render_nlf_as_images, intrinsic_matrix_from_field_of_view
        
        log_messages = ["=== RENDER NLF POSES DIRECT V6 LOG ==="]
        
        if render_backend == "taichi":
            try:
                import taichi as ti
                ti.init(arch=ti.gpu)
                log_messages.append("Render-Backend: Taichi GPU initialisiert.")
            except Exception as e:
                render_backend = "torch"
                log_messages.append(f"WARNUNG: Taichi GPU fehlgeschlagen. Nutze Torch. Fehler: {e}")
        else:
            log_messages.append("Render-Backend: Torch.")

        # 1. Echter Deepcopy der Daten, damit wir sie manipulieren und ausgeben können
        scaled_nlf_poses = copy.deepcopy(nlf_poses)
            
        try:
            pose_input = scaled_nlf_poses['joints3d_nonparam'][0] if isinstance(scaled_nlf_poses, dict) else scaled_nlf_poses
            
            # Referenz DW Poses laden, falls verbunden
            dw_pose_input = copy.deepcopy(dw_poses_fallback["poses"]) if dw_poses_fallback is not None else None
            
            if len(pose_input) > 0 and pose_input[0] is not None:
                log_messages.append(f"Erfolgreich geladen: {len(pose_input)} Frames.")

            # Standard Kamera Matrix erstellen
            intrinsic_matrix = intrinsic_matrix_from_field_of_view([height, width])
            
            # 2. CONFIG MATHEMATISCH AUF DIE 3D PUNKTE ANWENDEN
            try:
                config = json.loads(nlf_render_config)
                if "anchor_scale" in config:
                    scale_y = float(config["anchor_scale"])
                    scale_x = float(config.get("scale_x_factor", scale_y))
                    
                    p_x = float(config["pivot_x"])
                    p_y = float(config["pivot_y"])

                    if p_x <= 2.0 and p_y <= 2.0:
                        p_x = p_x * width
                        p_y = p_y * height
                    
                    # Originale Brennweiten und Kamera-Zentren
                    fx = intrinsic_matrix[0, 0]
                    fy = intrinsic_matrix[1, 1]
                    cx = intrinsic_matrix[0, 2]
                    cy = intrinsic_matrix[1, 2]

                    # Wir übersetzen den Kamera-Zoom in eine 3D-Punkt-Verschiebung 
                    # (Unter Einbezug der Tiefe Z, um die Perspektive zu erhalten)
                    M13 = (cx - p_x) * (scale_x - 1.0) / fx
                    M23 = (cy - p_y) * (scale_y - 1.0) / fy

                    log_messages.append(f"Wende 3D-Transformation an: ScaleX={scale_x:.3f}, ScaleY={scale_y:.3f}")

                    # ECHTE 3D DATEN MANIPULATION
                    for frame_idx in range(len(pose_input)):
                        if pose_input[frame_idx] is not None and len(pose_input[frame_idx]) > 0:
                            pts = pose_input[frame_idx]
                            
                            # Koordinaten extrahieren
                            X = pts[..., 0].clone()
                            Y = pts[..., 1].clone()
                            Z = pts[..., 2].clone()
                            
                            # Punkte transformieren (Z bleibt gleich für korrekte 3D Proportion)
                            pts[..., 0] = X * scale_x + Z * M13
                            pts[..., 1] = Y * scale_y + Z * M23
                            
                    log_messages.append("Erfolg: Die 3D-NLF-Daten wurden physisch im Raum transformiert!")
            except Exception as e:
                log_messages.append(f"Fehler bei 3D Transformation: {e}")

            # 3. Rendern mit allen benutzerdefinierten Einstellungen
            log_messages.append(f"Rendere Settings -> 2D: {draw_2d} | Face: {draw_face} | Hands: {draw_hands}")
            
            is_multi = False
            if len(pose_input) > 0:
                if isinstance(pose_input[0], list):
                    is_multi = len(pose_input[0]) > 1
                elif isinstance(pose_input[0], torch.Tensor) and pose_input[0].dim() == 3:
                    is_multi = pose_input[0].shape[0] > 1
                    
            if is_multi:
                frames_np = render_multi_nlf_as_images(
                    pose_input, dw_pose_input, height, width, len(pose_input), 
                    intrinsic_matrix=intrinsic_matrix, 
                    draw_2d=draw_2d, draw_face=draw_face, draw_hands=draw_hands, 
                    render_backend=render_backend
                )
            else:
                frames_np = render_nlf_as_images(
                    pose_input, dw_pose_input, height, width, len(pose_input), 
                    intrinsic_matrix=intrinsic_matrix, 
                    draw_2d=draw_2d, draw_face=draw_face, draw_hands=draw_hands, 
                    render_backend=render_backend
                )

            frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
            frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5
            
            # Die Dictionary Struktur sauber verpacken
            if isinstance(scaled_nlf_poses, dict):
                scaled_nlf_poses['joints3d_nonparam'] = [pose_input]
            else:
                scaled_nlf_poses = pose_input

            node_mappings = json.dumps({
                "node_name": "RenderNLFPosesDirect6",
                "status": "success",
                "frames_processed": len(pose_input)
            })

            return (frames_tensor.cpu().float(), mask.cpu().float(), "\n".join(log_messages), scaled_nlf_poses, node_mappings)

        except Exception as e:
            log_messages.append(f"FATALER ABSTURZ BEIM RENDERN: {e}")
            log_messages.append(traceback.format_exc())
            empty_img = torch.zeros((1, height, width, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            return (empty_img, empty_mask, "\n".join(log_messages), nlf_poses, "{}")


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
    "RetargetPoseCalibratorV9": RetargetPoseCalibratorV9,
    "PoseCalibrationV13": PoseCalibrationV13,
    "PoseGlobalPerspectiveScalerV26": PoseGlobalPerspectiveScalerV26,
    "PoseCalibrationV15": PoseCalibrationV15,
    "PoseGlobalPerspectiveScalerV28": PoseGlobalPerspectiveScalerV28,
    "PoseCalibrationV16": PoseCalibrationV16,
    "PoseGlobalPerspectiveScalerV29": PoseGlobalPerspectiveScalerV29,
    "NLFDataToPoseData": NLFDataToPoseData,
    "RenderNLFPosesDirect": RenderNLFPosesDirect,
    "PoseCalibrationV17": PoseCalibrationV17,
    "PoseCalibrationV18": PoseCalibrationV18,
    "PoseGlobalPerspectiveScalerV30": PoseGlobalPerspectiveScalerV30,
    "PoseGlobalPerspectiveScalerV31": PoseGlobalPerspectiveScalerV31,
    "PoseCalibrationV19": PoseCalibrationV19,
    "PoseGlobalPerspectiveScalerV33": PoseGlobalPerspectiveScalerV33,
    "PoseGlobalPerspectiveScalerV34": PoseGlobalPerspectiveScalerV34,
    "PoseGlobalPerspectiveScalerV35": PoseGlobalPerspectiveScalerV35,
    "PoseGlobalPerspectiveScalerV37": PoseGlobalPerspectiveScalerV37,
    "PoseGlobalPerspectiveScalerV38": PoseGlobalPerspectiveScalerV38,
    "RenderNLFPosesDirect2": RenderNLFPosesDirect2,
    "RenderNLFPosesDirect3": RenderNLFPosesDirect3,
    "RenderNLFPosesDirect4": RenderNLFPosesDirect4,
    "PoseCalibrationV20": PoseCalibrationV20,
    "NLFProportionalRetargeterV3": NLFProportionalRetargeterV3,
    "PoseCalibrationV21": PoseCalibrationV21,
    "PoseCalibrationV22": PoseCalibrationV22,
    "NLFProportionalRetargeterV4": NLFProportionalRetargeterV4,
    "NLFProportionalRetargeterV5": NLFProportionalRetargeterV5,
    "NLFConfigScaler3DBones": NLFConfigScaler3DBones,
    "RenderNLFPosesDirect5": RenderNLFPosesDirect5,
    "RenderNLFPosesDirect6": RenderNLFPosesDirect6,
    "RenderNLFPosesDirect7": RenderNLFPosesDirect7,
    
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
    "RetargetPoseCalibratorV9": "WanAnimate: Retarget Pose Calibrator (V9 Norm)",
    "PoseCalibrationV13": "WanAnimate: Pose Calibration (V13 Rechenvorgang)",
    "PoseGlobalPerspectiveScalerV26": "WanAnimate: Global Scaler (V26 Rechenvorgang)",
    "PoseCalibrationV15": "WanAnimate: Pose Calibration (DYNAMIC FULL-BODY + UNSCALED DEPTH + PINHOLE)",
    "PoseGlobalPerspectiveScalerV28": "WanAnimate: Global Scaler v28 (FULL-BODY + UNSCALED DEPTH + PINHOLE)",
    "PoseCalibrationV16": "Pose Calibration V16 (with NLF 3D Extractor)",
    "PoseGlobalPerspectiveScalerV29": "Pose Global Perspective Scaler V29 (with 3D Scale)",
    "NLFDataToPoseData": "NLF Data to 2D Pose Data",
    "RenderNLFPosesDirect": "Render NLF Poses Direct",
    "PoseCalibrationV17": "Pose Calibration V17 (with 3D NLF Avg)",
    "PoseCalibrationV18": "Pose Calibration V18 (Smart Calf)",
    "PoseGlobalPerspectiveScalerV30": "Pose Global Perspective Scaler V30",
    "PoseGlobalPerspectiveScalerV31": "Pose Global Perspective Scaler V31",
    "PoseCalibrationV19": "Pose Calibration V19",
    "PoseGlobalPerspectiveScalerV33": "Pose Global Perspective Scaler V33",
    "PoseGlobalPerspectiveScalerV34": "Pose Global Perspective Scaler V34",
    "PoseGlobalPerspectiveScalerV35": "Pose Global Perspective Scaler V35",
    "PoseGlobalPerspectiveScalerV37": "Pose Global Perspective Scaler V37",
    "PoseGlobalPerspectiveScalerV38": "Pose Global Perspective Scaler V38 (Smart Camera Zoom)",
    "RenderNLFPosesDirect2": "Render NLF Poses Direct 2",
    "RenderNLFPosesDirect3": "Render NLF Poses Direct 3",
    "RenderNLFPosesDirect4": "Render NLF Poses Direct 4",
    "PoseCalibrationV20": "Pose Calibration V20 (Full 3D & Smart Calf)",
    "NLFProportionalRetargeterV3": "NLF Proportional Retargeter V3 (Best Frame Fallback)",
    "PoseCalibrationV21": "Pose Calibration V21 (Symmetric Auto-Fix)",
    "PoseCalibrationV22": "Pose Calibration V22",
    "NLFProportionalRetargeterV4": "NLF Proportional Retargeter V4",
    "NLFProportionalRetargeterV5": "NLF Proportional Retargeter V5",
    "NLFConfigScaler3DBones": "NLF Config caler 3D Bones",
    "RenderNLFPosesDirect5": "Render NLF Poses Direct 5",
    "RenderNLFPosesDirect6": "Render NLF Poses Direct 6",
    "RenderNLFPosesDirect7": "Render NLF Poses Direct 7",


}








