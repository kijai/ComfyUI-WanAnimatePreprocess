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


