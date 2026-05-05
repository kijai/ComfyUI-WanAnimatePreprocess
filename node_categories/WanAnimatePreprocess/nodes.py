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


class PoseAndFaceDetectionV8_NoWarp:
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
                "camera_smooth_window": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1, "tooltip": "Gimbal Window (0 = Aus)."}),
                "smoothing_passes": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "tooltip": "Multi-Pass für butterweiche Kamera."}),
                "max_face_ratio": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Ab welchem Screen-Anteil (z.B. 0.4 = 40%) wird das Gesicht ausgeblendet? Hoher Wert = Aus."}),
                "fade_frames": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1, "tooltip": "Wie viele Frames dauert die weiche Überblendung?"}),
            },
            "optional": {
                "retarget_image": ("IMAGE", {"default": None, "tooltip": "Optionales Referenzbild"}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "FACE_INFO", "STRING", "BBOX", "BBOX")
    RETURN_NAMES = ("pose_data", "face_images", "face_info", "key_frame_body_points", "bboxes", "face_bboxes")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "V8 (No Warp): V7 mit Temporal Smoothing und dynamischem Proximity Fade-Out."

    def process(self, model, images, width, height, face_resolution, face_pad_factor, camera_smooth_window, smoothing_passes, max_face_ratio, fade_frames, retarget_image=None):
        import cv2
        import numpy as np
        import torch
        import math
        import json
        from tqdm import tqdm
        from comfy.utils import ProgressBar
        from .utils import resize_by_area, bbox_from_detector, crop
        from .retarget_pose import load_pose_metas_from_kp2ds_seq, get_face_bboxes, get_retarget_pose, AAPoseMeta

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
        for img in tqdm(images_np, total=len(images_np), desc="V8 Detecting"):
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
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="V8 Keypoints"):
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

        # --- 4. TEMPORAL SMOOTHING & ALPHA FADE LOGIC ---
        raw_face_data = []
        last_valid_data = (W/2.0, H/2.0, float(face_resolution))
        
        for idx, meta in enumerate(pose_metas):
            current_scale = 1.0 + face_pad_factor
            face_bbox_for_image = get_face_bboxes(meta['keypoints_face'][:, :2], scale=current_scale, image_shape=(H, W))
            raw_x1, raw_x2, raw_y1, raw_y2 = face_bbox_for_image
            raw_w = raw_x2 - raw_x1
            raw_h = raw_y2 - raw_y1
            max_side = max(raw_w, raw_h)
            cx = raw_x1 + raw_w / 2.0
            cy = raw_y1 + raw_h / 2.0
            
            if max_side > 0:
                last_valid_data = (cx, cy, max_side)
                raw_face_data.append((cx, cy, max_side))
            else:
                raw_face_data.append(last_valid_data)

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

        smoothed_data = smooth_series(raw_face_data, smoothing_passes, camera_smooth_window)

        # Fade Alpha Calculation based on face size
        alphas = [1.0] * len(smoothed_data)
        if max_face_ratio < 10.0:
            is_too_close = []
            for cx, cy, max_side in smoothed_data:
                ratio = max_side / float(min(H, W))
                is_too_close.append(ratio > max_face_ratio)
            
            for i in range(len(alphas)):
                min_dist = 999999
                for j, close in enumerate(is_too_close):
                    if close:
                        dist = abs(i - j)
                        if dist < min_dist: min_dist = dist
                
                if fade_frames <= 0:
                    alphas[i] = 0.0 if min_dist == 0 else 1.0
                else:
                    if min_dist == 0:
                        alphas[i] = 0.0
                    elif min_dist >= fade_frames:
                        alphas[i] = 1.0
                    else:
                        alphas[i] = float(min_dist) / float(fade_frames)

        # --- 5. FACE EXTRACTION (Using Smoothed Data) ---
        face_images = []
        face_bboxes = []
        face_info = []

        for idx, (data, alpha) in enumerate(zip(smoothed_data, alphas)):
            cx, cy, max_side = data
            
            sq_x1 = int(cx - max_side / 2)
            sq_y1 = int(cy - max_side / 2)
            sq_x2 = sq_x1 + int(max_side)
            sq_y2 = sq_y1 + int(max_side)
            
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

            # Resize (verzerrungsfrei, da quadratisch)
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
                "padding": (0, 0, 0, 0),
                "blend_alpha": float(alpha) # <-- Hier geht der Fade-Wert in den Stitcher
            }
            face_info.append(info_entry)

        face_images_tensor = torch.from_numpy(np.stack(face_images, 0))

        # --- 6. Restliche Outputs ---
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


class WanFaceStitcherV4:
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
    DESCRIPTION = "V4 Stitcher: Beachtet den blend_alpha Fade-Out aus V8 NoWarp."

    def process(self, destination_images, face_images_v2, face_info_v2, mode, blend_feather):
        import cv2
        import numpy as np
        import torch
        from tqdm import tqdm

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

        for i in tqdm(range(B), desc="V4 Stitching"):
            bg = dest_np[i]
            if bg.shape[1] != target_w or bg.shape[0] != target_h: bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            out[i] = bg
            
            if i >= len(face_info_v2) or not face_info_v2[i]["valid"]: continue
            
            info = face_info_v2[i]
            blend_alpha = info.get("blend_alpha", 1.0)
            
            # Optimization: Wenn wir komplett im Fade-Out sind, sparen wir uns das Stitching für diesen Frame
            if blend_alpha <= 0.001: continue
            
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

            # Multipliziere die Alpha-Maske mit unserem Fade-Wert aus V8
            mask *= blend_alpha

            # Clipping
            dx1, dy1 = max(0, tx1), max(0, ty1)
            dx2, dy2 = min(target_w, tx2), min(target_h, ty2)
            sx1, sy1 = dx1 - tx1, dy1 - ty1
            sx2, sy2 = sx1 + (dx2-dx1), sy1 + (dy2-dy1)
            
            if dx2 > dx1 and dy2 > dy1:
                out[i, dy1:dy2, dx1:dx2] = face[sy1:sy2, sx1:sx2] * mask[sy1:sy2, sx1:sx2] + out[i, dy1:dy2, dx1:dx2] * (1.0 - mask[sy1:sy2, sx1:sx2])

        return (torch.from_numpy(out),)


