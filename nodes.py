import os
import copy
import math
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


class PoseDataPostProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "length_mode": (("TORSO", "FULL_BODY"), {"default": "TORSO", "tooltip": "Select which body segments are used to stabilise lengths."}),
                "preserve_tolerance": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Skip adjustments when the median length ratio deviates from 1 by less than this tolerance."}),
                "max_scale_change": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.01, "tooltip": "Clamp the normalisation scale multiplier to avoid extreme corrections."}),
                "limit_to_canvas": ("BOOLEAN", {"default": True, "tooltip": "Clamp corrected keypoints so they remain on the canvas."}),
                "propagate_to_hands": ("BOOLEAN", {"default": True, "tooltip": "Apply the stabilisation transform to hand and face keypoints as well."}),
            },
            "optional": {
                "reference_pose_data": ("POSEDATA", {"default": None, "tooltip": "Optional pose data providing the reference proportions. Defaults to the original detection data."}),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Normalises pose proportions after editing so torso and limb lengths stay consistent even when detections are incomplete."

    SCORE_THRESHOLD = 0.05
    MIN_VALID_PAIRS = 2

    def process(
        self,
        pose_data,
        length_mode,
        preserve_tolerance,
        max_scale_change,
        limit_to_canvas,
        propagate_to_hands,
        reference_pose_data=None,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        reference_entries, reference_is_meta = self._resolve_reference_list(
            reference_pose_data, pose_data_copy
        )

        if reference_entries is None:
            return (pose_data_copy,)

        use_pairs = (
            TORSO_LENGTH_PAIRS if length_mode == "TORSO" else FULL_BODY_LENGTH_PAIRS
        )

        for idx, meta in enumerate(pose_metas):
            reference_entry = (
                reference_entries[idx]
                if idx < len(reference_entries)
                else reference_entries[-1]
            )

            if reference_entry is None:
                continue

            self._stabilise_meta(
                meta,
                reference_entry,
                reference_is_meta,
                use_pairs,
                preserve_tolerance,
                max_scale_change,
                limit_to_canvas,
                propagate_to_hands,
            )

        return (pose_data_copy,)


class PoseDataEditorAutomatic:
    HEAD_INDICES = BODY_GROUPS.get("HEAD", [])
    HIP_INDICES = [idx for idx in BODY_GROUPS.get("HIP_WIDTH", []) if idx is not None]
    LEG_INDICES = [idx for idx in BODY_GROUPS.get("LEGS", []) if idx not in BODY_GROUPS.get("HIP_WIDTH", [])]
    FOOT_INDICES = [idx for idx in BODY_GROUPS.get("FEET", []) if idx is not None]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "scale_legs_to_bottom": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Stretch legs so the lowest point reaches the canvas floor (after foot padding).",
                    },
                ),
                "scale_legs_normal": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Apply a manual scaling factor to the current leg span.",
                    },
                ),
                "scale_legs_relative_to_body": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Match legs to a multiple of the torso-to-head distance.",
                    },
                ),
                "scale_legs": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Multiplier applied to the current leg span when the 'Scale Legs Normal' checkbox is enabled.",
                    },
                ),
                "torso_head_multiple": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Multiplier applied to the torso-to-head distance when 'Scale Legs Relative to Body' is enabled.",
                    },
                ),
                "head_padding": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.001,
                        "tooltip": "Distance to keep between the top-most head point and the canvas edge.",
                    },
                ),
                "head_padding_normalized": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Interpret the head padding as a 0-1 ratio of the canvas height instead of pixels.",
                    },
                ),
                "foot_padding": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.001,
                        "tooltip": "Distance to keep between the lowest foot point and the bottom canvas edge.",
                    },
                ),
                "foot_padding_normalized": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Interpret the foot padding as a 0-1 ratio of the canvas height instead of pixels.",
                    },
                ),
                "center_horizontally": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, horizontally centre the pose after vertical adjustments.",
                    },
                ),
                "limit_to_canvas": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Clamp all keypoints to the canvas bounds after adjustments are applied.",
                    },
                ),
                "person_index": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "When >= 0, only process the matching pose entry. Use -1 to process every pose.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Automatically aligns detected poses to the canvas by padding the head, stretching legs to the floor and optionally centring horizontally."

    def process(
        self,
        pose_data,
        scale_legs_to_bottom,
        scale_legs_normal,
        scale_legs_relative_to_body,
        scale_legs,
        torso_head_multiple,
        head_padding,
        head_padding_normalized,
        foot_padding,
        foot_padding_normalized,
        center_horizontally,
        limit_to_canvas,
        person_index,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        leg_mode = self._select_leg_mode(
            scale_legs_to_bottom,
            scale_legs_normal,
            scale_legs_relative_to_body,
        )

        for idx, meta in enumerate(pose_metas):
            if person_index >= 0 and idx != person_index:
                continue

            self._auto_align_meta(
                meta,
                leg_mode,
                scale_legs,
                torso_head_multiple,
                head_padding,
                head_padding_normalized,
                foot_padding,
                foot_padding_normalized,
                center_horizontally,
                limit_to_canvas,
            )

        return (pose_data_copy,)

    def _select_leg_mode(
        self,
        scale_legs_to_bottom,
        scale_legs_normal,
        scale_legs_relative_to_body,
    ):
        selection = [
            bool(scale_legs_to_bottom),
            bool(scale_legs_normal),
            bool(scale_legs_relative_to_body),
        ]

        if sum(selection) != 1:
            raise ValueError(
                "PoseDataEditorAutomatic requires exactly one leg scaling checkbox to be enabled."
            )

        if scale_legs_to_bottom:
            return "bottom"
        if scale_legs_normal:
            return "normal"
        return "relative"

    def _auto_align_meta(
        self,
        meta,
        leg_mode,
        scale_legs,
        torso_head_multiple,
        head_padding,
        head_padding_normalized,
        foot_padding,
        foot_padding_normalized,
        center_horizontally,
        limit_to_canvas,
    ):
        width = getattr(meta, "width", None)
        height = getattr(meta, "height", None)

        if width in (None, 0) or height in (None, 0):
            return

        head_pad_px = self._resolve_padding(head_padding, head_padding_normalized, height)
        foot_pad_px = self._resolve_padding(foot_padding, foot_padding_normalized, height)

        head_pad_px = float(np.clip(head_pad_px, 0.0, float(height)))
        foot_pad_px = float(np.clip(foot_pad_px, 0.0, float(height)))

        self._translate_head_to_padding(meta, head_pad_px, width, height)
        self._adjust_leg_length(
            meta,
            foot_pad_px,
            width,
            height,
            leg_mode,
            scale_legs,
            torso_head_multiple,
        )

        if center_horizontally:
            self._centre_pose(meta, width)

        if limit_to_canvas:
            self._clamp_pose(meta, width, height)

    def _resolve_padding(self, value, normalized, height):
        if normalized:
            return float(value) * float(height)
        return float(value)

    def _translate_head_to_padding(
        self,
        meta,
        head_padding_px,
        width,
        height,
        locked_delta=None,
    ):
        if locked_delta is not None:
            delta = float(locked_delta)
            if abs(delta) <= 1e-6:
                return float(delta)
            self._translate_pose(meta, 0.0, delta, width, height)
            return float(delta)

        top_y = self._find_head_top(meta)

        if top_y is None:
            top_y = self._find_pose_top(meta)

        if top_y is None:
            return None

        delta = head_padding_px - float(top_y)
        if abs(delta) <= 1e-6:
            return float(delta)

        self._translate_pose(meta, 0.0, delta, width, height)
        return float(delta)

    def _adjust_leg_length(
        self,
        meta,
        foot_padding_px,
        width,
        height,
        leg_mode,
        scale_legs,
        torso_head_multiple,
    ):
        anchor_y = self._compute_leg_anchor(meta)
        if anchor_y is None:
            return

        bottom_y = self._find_leg_bottom(meta)
        if bottom_y is None:
            return

        span = bottom_y - anchor_y
        if span <= 1e-6:
            return

        mode_key = (leg_mode or "").strip().lower()

        if mode_key == "normal":
            multiplier = max(float(scale_legs), 0.0)
            if multiplier <= 0.0:
                return
            scale = multiplier
        elif mode_key == "relative":
            upper_length = self._compute_upper_body_length(meta, anchor_y)
            if upper_length is None or upper_length <= 1e-6:
                return
            multiplier = max(float(torso_head_multiple), 0.0)
            if multiplier <= 0.0:
                return
            desired_span = upper_length * multiplier
            if desired_span <= 1e-6:
                return
            scale = desired_span / span
        else:
            target_y = float(height) - float(foot_padding_px)
            target_y = float(np.clip(target_y, 0.0, float(height)))

            if target_y <= anchor_y + 1e-6:
                return

            scale = (target_y - anchor_y) / span

        if not np.isfinite(scale) or scale <= 0.0:
            return

        self._scale_leg_points(meta, anchor_y, scale)

    def _compute_upper_body_length(self, meta, anchor_y):
        top_y = self._find_head_top(meta)

        if top_y is None:
            top_y = self._find_pose_top(meta)

        if top_y is None:
            return None

        length = float(anchor_y) - float(top_y)

        if length <= 1e-6:
            return None

        return length

    def _centre_pose(self, meta, width):
        bbox = self._collect_pose_bounds(meta)
        if bbox is None:
            return

        min_x, _, max_x, _ = bbox
        if max_x - min_x <= 1e-6:
            return

        current_cx = (min_x + max_x) * 0.5
        target_cx = float(width) * 0.5
        delta_x = target_cx - current_cx

        if abs(delta_x) <= 1e-6:
            return

        self._translate_pose(meta, delta_x, 0.0, width, getattr(meta, "height", 0))

    def _translate_pose(self, meta, dx, dy, width, height):
        for arr in self._iter_point_arrays(meta):
            if arr is None:
                continue
            for idx in range(len(arr)):
                coords = self._extract_coords(arr[idx])
                if coords is None:
                    continue
                new_x = coords[0] + dx
                new_y = coords[1] + dy
                self._assign_point(arr, idx, new_x, new_y)

    def _scale_leg_points(self, meta, anchor_y, scale):
        indices = set(self.LEG_INDICES + self.FOOT_INDICES)

        body = getattr(meta, "kps_body", None)
        if body is None:
            return

        for idx in indices:
            if idx >= len(body):
                continue
            coords = self._extract_coords(body[idx])
            if coords is None:
                continue
            offset_y = coords[1] - anchor_y
            new_y = anchor_y + offset_y * scale
            self._assign_point(body, idx, coords[0], new_y)

    def _clamp_pose(self, meta, width, height):
        for arr in self._iter_point_arrays(meta):
            if arr is None:
                continue
            for idx in range(len(arr)):
                coords = self._extract_coords(arr[idx])
                if coords is None:
                    continue
                new_x = float(np.clip(coords[0], 0.0, float(width)))
                new_y = float(np.clip(coords[1], 0.0, float(height)))
                self._assign_point(arr, idx, new_x, new_y)

    def _find_head_top(self, meta):
        body = getattr(meta, "kps_body", None)
        if body is None:
            return None

        ys = []
        for idx in self.HEAD_INDICES:
            if idx >= len(body):
                continue
            coords = self._extract_coords(body[idx])
            if coords is None:
                continue
            ys.append(coords[1])

        if ys:
            return float(np.min(ys))

        face = getattr(meta, "kps_face", None)
        if face is None:
            return None

        face_ys = []
        for idx in range(len(face)):
            coords = self._extract_coords(face[idx])
            if coords is None:
                continue
            face_ys.append(coords[1])

        if face_ys:
            return float(np.min(face_ys))

        return None

    def _find_pose_top(self, meta):
        mins = []
        for arr in self._iter_point_arrays(meta):
            if arr is None:
                continue
            for point in arr:
                coords = self._extract_coords(point)
                if coords is None:
                    continue
                mins.append(coords[1])

        if not mins:
            return None

        return float(np.min(mins))

    def _compute_leg_anchor(self, meta):
        body = getattr(meta, "kps_body", None)
        if body is None:
            return None

        hip_coords = []
        for idx in self.HIP_INDICES:
            if idx >= len(body):
                continue
            coords = self._extract_coords(body[idx])
            if coords is None:
                continue
            hip_coords.append(coords[1])

        if hip_coords:
            return float(np.mean(hip_coords))

        indices = set(self.LEG_INDICES + self.FOOT_INDICES)
        leg_coords = []
        for idx in indices:
            if idx >= len(body):
                continue
            coords = self._extract_coords(body[idx])
            if coords is None:
                continue
            leg_coords.append(coords[1])

        if leg_coords:
            return float(np.min(leg_coords))

        return None

    def _find_leg_bottom(self, meta):
        body = getattr(meta, "kps_body", None)
        if body is None:
            return None

        indices = set(self.LEG_INDICES + self.FOOT_INDICES)
        ys = []
        for idx in indices:
            if idx >= len(body):
                continue
            coords = self._extract_coords(body[idx])
            if coords is None:
                continue
            ys.append(coords[1])

        if not ys:
            return None

        return float(np.max(ys))

    def _collect_pose_bounds(self, meta):
        xs = []
        ys = []
        for arr in self._iter_point_arrays(meta):
            if arr is None:
                continue
            for point in arr:
                coords = self._extract_coords(point)
                if coords is None:
                    continue
                xs.append(coords[0])
                ys.append(coords[1])

        if not xs or not ys:
            return None

        return (
            float(np.min(xs)),
            float(np.min(ys)),
            float(np.max(xs)),
            float(np.max(ys)),
        )

    def _iter_point_arrays(self, meta):
        return (
            getattr(meta, name, None)
            for name in ("kps_body", "kps_lhand", "kps_rhand", "kps_face")
        )

    def _extract_coords(self, point):
        if point is None:
            return None

        if isinstance(point, np.ndarray):
            if point.size < 2:
                return None
            coords = point[:2]
        elif isinstance(point, (list, tuple)):
            if len(point) < 2:
                return None
            coords = point[:2]
        else:
            return None

        if not np.all(np.isfinite(coords)):
            return None

        return np.array(coords, dtype=np.float32)

    def _assign_point(self, arr, idx, x, y):
        if isinstance(arr[idx], np.ndarray):
            arr[idx][0] = x
            arr[idx][1] = y
        elif isinstance(arr[idx], list):
            if len(arr[idx]) >= 2:
                arr[idx][0] = x
                arr[idx][1] = y
        elif isinstance(arr[idx], tuple):
            values = list(arr[idx])
            if len(values) >= 2:
                values[0] = x
                values[1] = y
            arr[idx] = type(arr[idx])(values)
        else:
            arr[idx] = [x, y]


class PoseDataEditorAutomaticV2(PoseDataEditorAutomatic):
    """Canvas-aware variant of the automatic pose editor."""

    DESCRIPTION = (
        "Automatically aligns detected poses to the canvas while ensuring leg scaling "
        "respects the configured padding in every mode."
    )

    def _adjust_leg_length(
        self,
        meta,
        foot_padding_px,
        width,
        height,
        leg_mode,
        scale_legs,
        torso_head_multiple,
    ):
        anchor_y = self._compute_leg_anchor(meta)
        if anchor_y is None:
            return

        bottom_y = self._find_leg_bottom(meta)
        if bottom_y is None:
            return

        span = bottom_y - anchor_y
        if span <= 1e-6:
            return

        mode_key = (leg_mode or "").strip().lower()

        target_floor = float(height) - float(foot_padding_px)
        target_floor = float(np.clip(target_floor, 0.0, float(height)))
        available_downward = target_floor - anchor_y

        if mode_key == "normal":
            multiplier = max(float(scale_legs), 0.0)
            if multiplier <= 0.0:
                return
            scale = multiplier
        elif mode_key == "relative":
            upper_length = self._compute_upper_body_length(meta, anchor_y)
            if upper_length is None or upper_length <= 1e-6:
                return
            multiplier = max(float(torso_head_multiple), 0.0)
            if multiplier <= 0.0:
                return
            desired_span = upper_length * multiplier
            if desired_span <= 1e-6:
                return
            scale = desired_span / span
        else:
            if available_downward <= 1e-6:
                return
            scale = available_downward / span

        if not np.isfinite(scale) or scale <= 0.0:
            return

        if mode_key != "bottom":
            if available_downward <= 1e-6:
                if scale >= 1.0:
                    return
            else:
                max_scale = available_downward / span
                if max_scale <= 0.0:
                    if scale >= 1.0:
                        return
                elif scale > max_scale:
                    scale = max_scale

        self._scale_leg_points(meta, anchor_y, scale)

    def _resolve_reference_list(self, supplied_reference, pose_data_copy):
        if supplied_reference not in (None,):
            if isinstance(supplied_reference, dict):
                ref_pose_metas = supplied_reference.get("pose_metas")
                if ref_pose_metas:
                    return ref_pose_metas, True
                ref_original = supplied_reference.get("pose_metas_original")
                if ref_original:
                    return ref_original, False
            return None, False

        ref_original_default = pose_data_copy.get("pose_metas_original")
        if ref_original_default:
            return ref_original_default, False

        ref_pose_metas_default = pose_data_copy.get("pose_metas")
        if ref_pose_metas_default:
            return ref_pose_metas_default, True

        return None, False

    def _stabilise_meta(
        self,
        meta,
        reference_entry,
        reference_is_meta,
        length_pairs,
        tolerance,
        max_scale_change,
        limit_to_canvas,
        propagate_to_hands,
    ):
        width = getattr(meta, "width", None)
        height = getattr(meta, "height", None)

        if width in (None, 0) or height in (None, 0):
            return

        current_body = getattr(meta, "kps_body", None)
        current_scores = getattr(meta, "kps_body_p", None)

        if current_body is None or len(current_body) == 0:
            return

        reference_body, reference_scores = self._extract_reference_body(
            reference_entry, reference_is_meta, width, height
        )

        if reference_body is None or len(reference_body) == 0:
            return

        ratios = self._compute_length_ratios(
            reference_body,
            reference_scores,
            current_body,
            current_scores,
            length_pairs,
        )

        if len(ratios) < self.MIN_VALID_PAIRS:
            return

        median_ratio = float(np.median(ratios))

        if not np.isfinite(median_ratio):
            return

        median_ratio = float(np.clip(median_ratio, 1.0 / max_scale_change, max_scale_change))

        if abs(median_ratio - 1.0) <= tolerance:
            return

        anchor = self._compute_anchor(current_body, current_scores, width, height)

        self._apply_scale(meta.kps_body, anchor, median_ratio, width, height, limit_to_canvas)

        if propagate_to_hands:
            self._apply_scale(meta.kps_lhand, anchor, median_ratio, width, height, limit_to_canvas)
            self._apply_scale(meta.kps_rhand, anchor, median_ratio, width, height, limit_to_canvas)
            self._apply_scale(meta.kps_face, anchor, median_ratio, width, height, limit_to_canvas)

    def _extract_reference_body(self, reference_entry, reference_is_meta, width, height):
        if reference_is_meta and isinstance(reference_entry, AAPoseMeta):
            coords = np.array(reference_entry.kps_body, dtype=np.float32)
            scores = (
                np.array(getattr(reference_entry, "kps_body_p", None), dtype=np.float32)
                if getattr(reference_entry, "kps_body_p", None) is not None
                else None
            )
            return coords, scores

        if isinstance(reference_entry, dict):
            keypoints = reference_entry.get("keypoints_body")
            if keypoints is None:
                return None, None

            keypoints_np = np.array(keypoints, dtype=np.float32)
            if keypoints_np.ndim != 2 or keypoints_np.shape[1] < 2:
                return None, None

            coords = keypoints_np[:, :2] * np.array([width, height], dtype=np.float32)
            scores = (
                keypoints_np[:, 2]
                if keypoints_np.shape[1] > 2
                else None
            )
            return coords, scores

        return None, None

    def _compute_length_ratios(
        self,
        reference_body,
        reference_scores,
        current_body,
        current_scores,
        length_pairs,
    ):
        ratios = []

        for idx_a, idx_b in length_pairs:
            ref_valid = self._is_point_valid(reference_body, reference_scores, idx_a) and self._is_point_valid(
                reference_body, reference_scores, idx_b
            )
            cur_valid = self._is_point_valid(current_body, current_scores, idx_a) and self._is_point_valid(
                current_body, current_scores, idx_b
            )

            if not (ref_valid and cur_valid):
                continue

            ref_length = float(np.linalg.norm(reference_body[idx_a] - reference_body[idx_b]))
            cur_length = float(np.linalg.norm(current_body[idx_a] - current_body[idx_b]))

            if ref_length <= 1e-6 or cur_length <= 1e-6:
                continue

            ratios.append(ref_length / cur_length)

        return ratios

    def _is_point_valid(self, points, scores, index):
        if points is None or index >= len(points):
            return False

        point = points[index]
        if point is None:
            return False

        if np.any(~np.isfinite(point)):
            return False

        if scores is not None and index < len(scores):
            return scores[index] > self.SCORE_THRESHOLD

        return not (abs(point[0]) <= 1e-6 and abs(point[1]) <= 1e-6)

    def _compute_anchor(self, current_body, current_scores, width, height):
        torso_indices = BODY_GROUPS.get("TORSO", [])
        candidates = []

        for idx in torso_indices:
            if idx >= len(current_body):
                continue
            if current_scores is not None and idx < len(current_scores):
                if current_scores[idx] <= self.SCORE_THRESHOLD:
                    continue
            point = current_body[idx]
            if point is None or np.any(~np.isfinite(point)):
                continue
            candidates.append(point)

        if not candidates:
            for idx, point in enumerate(current_body):
                if current_scores is not None and idx < len(current_scores):
                    if current_scores[idx] <= self.SCORE_THRESHOLD:
                        continue
                if point is None or np.any(~np.isfinite(point)):
                    continue
                candidates.append(point)

        if not candidates:
            return np.array([width * 0.5, height * 0.5], dtype=np.float32)

        return np.mean(np.array(candidates, dtype=np.float32), axis=0)

    def _apply_scale(self, points, anchor, scale, width, height, limit_to_canvas):
        if points is None:
            return

        if isinstance(points, np.ndarray):
            coords = points
        else:
            coords = np.array(points, dtype=np.float32)

        if coords.size == 0:
            return

        if coords.ndim != 2 or coords.shape[1] < 2:
            return

        anchor_vec = np.array(anchor, dtype=np.float32)

        transformed = (coords[:, :2] - anchor_vec) * scale + anchor_vec

        if limit_to_canvas:
            transformed[:, 0] = np.clip(transformed[:, 0], 0.0, float(width))
            transformed[:, 1] = np.clip(transformed[:, 1], 0.0, float(height))

        if isinstance(points, np.ndarray):
            points[:, :2] = transformed
        else:
            for idx, (x, y) in enumerate(transformed):
                points[idx][0] = float(x)
                points[idx][1] = float(y)


class PoseDataEditorAutomaticV3(PoseDataEditorAutomaticV2):
    """Extends the canvas-aware editor with post-scale foot alignment."""

    DESCRIPTION = (
        "Aligns poses like Automatic V2 while pushing the body down when manual or "
        "relative leg scaling leaves extra space above the foot padding."
    )

    def _auto_align_meta(
        self,
        meta,
        leg_mode,
        scale_legs,
        torso_head_multiple,
        head_padding,
        head_padding_normalized,
        foot_padding,
        foot_padding_normalized,
        center_horizontally,
        limit_to_canvas,
    ):
        width = getattr(meta, "width", None)
        height = getattr(meta, "height", None)

        if width in (None, 0) or height in (None, 0):
            return

        head_pad_px = self._resolve_padding(head_padding, head_padding_normalized, height)
        foot_pad_px = self._resolve_padding(foot_padding, foot_padding_normalized, height)

        head_pad_px = float(np.clip(head_pad_px, 0.0, float(height)))
        foot_pad_px = float(np.clip(foot_pad_px, 0.0, float(height)))

        self._translate_head_to_padding(meta, head_pad_px, width, height)
        self._adjust_leg_length(
            meta,
            foot_pad_px,
            width,
            height,
            leg_mode,
            scale_legs,
            torso_head_multiple,
        )

        if leg_mode in ("normal", "relative"):
            self._shift_pose_to_foot_padding(meta, foot_pad_px, width, height)

        if center_horizontally:
            self._centre_pose(meta, width)

        if limit_to_canvas:
            self._clamp_pose(meta, width, height)

    def _shift_pose_to_foot_padding(
        self,
        meta,
        foot_padding_px,
        width,
        height,
        locked_delta=None,
    ):
        if locked_delta is not None:
            delta = float(locked_delta)
            if abs(delta) <= 1e-6:
                return float(delta)
            self._translate_pose(meta, 0.0, delta, width, height)
            return float(delta)

        bottom_y = self._find_leg_bottom(meta)
        if bottom_y is None:
            return None

        target_floor = float(height) - float(foot_padding_px)
        target_floor = float(np.clip(target_floor, 0.0, float(height)))

        delta = target_floor - float(bottom_y)
        if delta <= 1e-6:
            return float(max(delta, 0.0))

        self._translate_pose(meta, 0.0, delta, width, height)
        return float(delta)


class PoseDataEditorAutomaticV4(PoseDataEditorAutomaticV3):
    """Adds an upper-body offset leg mode to the automatic alignment workflow."""

    DESCRIPTION = (
        "Extends Automatic V3 with an 'Only Offset From Torso to Head' option that "
        "shifts the upper body by a configurable amount before stretching the legs "
        "to reconnect to the feet."
    )

    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        required = inputs.get("required", {})

        ordered = {}
        for key, value in required.items():
            ordered[key] = value
            if key == "scale_legs_relative_to_body":
                ordered["offset_upper_body_only"] = (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable the 'Only Offset From Torso to Head' leg mode. "
                            "Moves the upper body without translating the feet."
                        ),
                    },
                )
                ordered["upper_body_offset"] = (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2048.0,
                        "max": 2048.0,
                        "step": 0.001,
                        "tooltip": (
                            "Amount to raise the torso-to-head segment when the offset "
                            "mode is selected."
                        ),
                    },
                )
                ordered["upper_body_offset_normalized"] = (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Interpret the upper-body offset as a 0-1 ratio of the "
                            "canvas height instead of pixels."
                        ),
                    },
                )

        inputs["required"] = ordered
        return inputs

    def process(
        self,
        pose_data,
        scale_legs_to_bottom,
        scale_legs_normal,
        scale_legs_relative_to_body,
        offset_upper_body_only,
        upper_body_offset,
        upper_body_offset_normalized,
        scale_legs,
        torso_head_multiple,
        head_padding,
        head_padding_normalized,
        foot_padding,
        foot_padding_normalized,
        center_horizontally,
        limit_to_canvas,
        person_index,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        leg_mode = self._select_leg_mode(
            scale_legs_to_bottom,
            scale_legs_normal,
            scale_legs_relative_to_body,
            offset_upper_body_only,
        )

        for idx, meta in enumerate(pose_metas):
            if person_index >= 0 and idx != person_index:
                continue

            self._auto_align_meta(
                meta,
                leg_mode,
                scale_legs,
                torso_head_multiple,
                head_padding,
                head_padding_normalized,
                foot_padding,
                foot_padding_normalized,
                center_horizontally,
                limit_to_canvas,
                upper_body_offset,
                upper_body_offset_normalized,
            )

        return (pose_data_copy,)

    def _select_leg_mode(
        self,
        scale_legs_to_bottom,
        scale_legs_normal,
        scale_legs_relative_to_body,
        offset_upper_body_only,
    ):
        selection = [
            bool(scale_legs_to_bottom),
            bool(scale_legs_normal),
            bool(scale_legs_relative_to_body),
            bool(offset_upper_body_only),
        ]

        if sum(selection) != 1:
            raise ValueError(
                "PoseDataEditorAutomatic V4 requires exactly one leg option to be enabled."
            )

        if offset_upper_body_only:
            return "offset"

        if scale_legs_to_bottom:
            return "bottom"

        if scale_legs_normal:
            return "normal"

        return "relative"

    def _auto_align_meta(
        self,
        meta,
        leg_mode,
        scale_legs,
        torso_head_multiple,
        head_padding,
        head_padding_normalized,
        foot_padding,
        foot_padding_normalized,
        center_horizontally,
        limit_to_canvas,
        upper_body_offset,
        upper_body_offset_normalized,
    ):
        if leg_mode != "offset":
            super()._auto_align_meta(
                meta,
                leg_mode,
                scale_legs,
                torso_head_multiple,
                head_padding,
                head_padding_normalized,
                foot_padding,
                foot_padding_normalized,
                center_horizontally,
                limit_to_canvas,
            )
            return

        width = getattr(meta, "width", None)
        height = getattr(meta, "height", None)

        if width in (None, 0) or height in (None, 0):
            return

        head_pad_px = self._resolve_padding(head_padding, head_padding_normalized, height)
        head_pad_px = float(np.clip(head_pad_px, 0.0, float(height)))

        self._translate_head_to_padding(meta, head_pad_px, width, height)

        locked_feet = self._capture_foot_positions(meta)

        offset_px = self._resolve_padding(
            upper_body_offset,
            upper_body_offset_normalized,
            height,
        )
        offset_px = float(np.clip(offset_px, -float(height), float(height)))

        hip_pairs = self._offset_upper_body(meta, offset_px)

        if hip_pairs:
            self._stretch_legs_to_hips(meta, hip_pairs, locked_feet)

        if locked_feet:
            self._restore_locked_feet(meta, locked_feet)

        if center_horizontally:
            self._centre_pose(meta, width)

        if limit_to_canvas:
            self._clamp_pose(meta, width, height)

    def _offset_upper_body(self, meta, offset_px):
        if abs(offset_px) <= 1e-6:
            return {}

        body = getattr(meta, "kps_body", None)
        hip_info = {}

        if body is None:
            return hip_info

        hip_indices = set(self.HIP_INDICES)
        leg_indices = set(self.LEG_INDICES + self.FOOT_INDICES)

        for hip_idx in hip_indices:
            if hip_idx >= len(body):
                continue
            coords = self._extract_coords(body[hip_idx])
            if coords is None:
                continue
            hip_info[hip_idx] = (
                np.array(coords, dtype=np.float32),
                None,
            )

        for idx in range(len(body)):
            coords = self._extract_coords(body[idx])
            if coords is None:
                continue

            if idx in leg_indices and idx not in hip_indices:
                continue

            new_x = coords[0]
            new_y = coords[1] - offset_px
            self._assign_point(body, idx, new_x, new_y)

        for hip_idx in list(hip_info.keys()):
            coords = self._extract_coords(body[hip_idx])
            if coords is None:
                hip_info.pop(hip_idx, None)
                continue
            hip_info[hip_idx] = (
                hip_info[hip_idx][0],
                np.array(coords, dtype=np.float32),
            )

        for arr_name in ("kps_lhand", "kps_rhand", "kps_face"):
            arr = getattr(meta, arr_name, None)
            if arr is None:
                continue

            for idx in range(len(arr)):
                coords = self._extract_coords(arr[idx])
                if coords is None:
                    continue

                new_x = coords[0]
                new_y = coords[1] - offset_px
                self._assign_point(arr, idx, new_x, new_y)

        hip_pairs = {}
        for hip_idx, (before, after) in hip_info.items():
            if after is None:
                continue
            if np.any(~np.isfinite(before)) or np.any(~np.isfinite(after)):
                continue
            hip_pairs[hip_idx] = (before, after)

        return hip_pairs

    def _capture_foot_positions(self, meta):
        body = getattr(meta, "kps_body", None)
        if body is None:
            return {}

        locked = {}
        for foot_idx in self.FOOT_INDICES:
            if foot_idx >= len(body):
                continue
            coords = self._extract_coords(body[foot_idx])
            if coords is None:
                continue
            locked[foot_idx] = np.array(coords, dtype=np.float32)

        return locked

    def _restore_locked_feet(self, meta, locked_feet):
        if not locked_feet:
            return

        body = getattr(meta, "kps_body", None)
        if body is None:
            return

        for foot_idx, coords in locked_feet.items():
            if foot_idx >= len(body):
                continue
            if coords is None or not np.all(np.isfinite(coords)):
                continue
            self._assign_point(body, foot_idx, float(coords[0]), float(coords[1]))

    def _stretch_legs_to_hips(self, meta, hip_pairs, locked_feet=None):
        if not hip_pairs:
            return

        body = getattr(meta, "kps_body", None)
        if body is None:
            return

        leg_map = {
            8: (9, 10),
            11: (12, 13),
        }

        for hip_idx, (before, after) in hip_pairs.items():
            if hip_idx not in leg_map:
                continue

            knee_idx, ankle_idx = leg_map[hip_idx]

            if ankle_idx >= len(body):
                continue

            if locked_feet and ankle_idx in locked_feet:
                foot_coords = locked_feet[ankle_idx]
            else:
                foot_coords = self._extract_coords(body[ankle_idx])

            if foot_coords is None:
                continue

            foot_coords = np.array(foot_coords, dtype=np.float32)
            foot_vec_before = before - foot_coords
            foot_vec_after = after - foot_coords

            if np.linalg.norm(foot_vec_before) <= 1e-6:
                continue

            if np.linalg.norm(foot_vec_after) <= 1e-6:
                continue

            self._assign_point(
                body,
                hip_idx,
                float(after[0]),
                float(after[1]),
            )

            if knee_idx < len(body):
                knee_coords = self._extract_coords(body[knee_idx])
                if knee_coords is not None:
                    ratio = self._project_ratio(
                        np.array(knee_coords, dtype=np.float32),
                        foot_coords,
                        before,
                    )
                    if ratio is not None:
                        ratio = float(np.clip(ratio, 0.0, 1.0))
                        new_pos = foot_coords + foot_vec_after * ratio
                        self._assign_point(
                            body,
                            knee_idx,
                            float(new_pos[0]),
                            float(new_pos[1]),
                        )

    def _project_ratio(self, point, foot, hip_before):
        segment = hip_before - foot
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-6:
            return None

        vector = point - foot
        return float(np.dot(vector, segment) / length_sq)


class PoseDataEditorAutomaticV5(PoseDataEditorAutomaticV4):
    """Extends Automatic V4 with timed scaling, padding, and foot-lock options."""

    DESCRIPTION = (
        "Adds per-feature durations, FPS-driven scale locking, and an option to keep "
        "feet anchored while the upper body is shifted before stretching the legs."
    )

    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        required = inputs.get("required", {})

        ordered = {}
        for key, value in required.items():
            ordered[key] = value
            if key == "scale_legs_to_bottom":
                ordered["scale_legs_to_bottom_active_seconds"] = (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "Duration in seconds to keep adapting the stretch-to-bottom "
                            "leg mode before reusing the captured scale. Use 0 for no "
                            "time limit."
                        ),
                    },
                )
            if key == "head_padding_normalized":
                ordered["head_padding_active_seconds"] = (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "How long the automatic head padding adjustments remain "
                            "active. Use 0 to keep them enabled for the whole clip."
                        ),
                    },
                )
            if key == "foot_padding_normalized":
                ordered["foot_padding_active_seconds"] = (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "How long the automatic foot padding adjustments remain "
                            "active. Use 0 to keep them enabled for the whole clip."
                        ),
                    },
                )
            if key == "limit_to_canvas":
                ordered["activate_lock_scale_after_seconds"] = (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "Time in seconds before the current leg scale is frozen. Set "
                            "to 0 to keep adapting for the full clip. Applies to all "
                            "leg scaling modes."
                        ),
                    },
                )
                ordered["dont_offset_feet"] = (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Keep feet at their original positions while the torso moves "
                            "so legs stretch to reconnect without translating ankles."
                        ),
                    },
                )
                ordered["fps"] = (
                    "INT",
                    {
                        "default": 24,
                        "min": 1,
                        "max": 960,
                        "step": 1,
                        "tooltip": (
                            "Frames per second of the sequence. Durations are converted "
                            "to frame counts using this value."
                        ),
                    },
                )

        inputs["required"] = ordered
        return inputs

    def process(
        self,
        pose_data,
        scale_legs_to_bottom,
        scale_legs_to_bottom_active_seconds,
        scale_legs_normal,
        scale_legs_relative_to_body,
        offset_upper_body_only,
        upper_body_offset,
        upper_body_offset_normalized,
        scale_legs,
        torso_head_multiple,
        head_padding,
        head_padding_normalized,
        head_padding_active_seconds,
        foot_padding,
        foot_padding_normalized,
        foot_padding_active_seconds,
        center_horizontally,
        limit_to_canvas,
        activate_lock_scale_after_seconds,
        dont_offset_feet,
        fps,
        person_index,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        leg_mode = self._select_leg_mode(
            scale_legs_to_bottom,
            scale_legs_normal,
            scale_legs_relative_to_body,
            offset_upper_body_only,
        )

        fps_int = max(1, int(round(float(fps))))

        lock_threshold = self._seconds_to_frames(
            activate_lock_scale_after_seconds,
            fps_int,
        )
        if lock_threshold == 0:
            lock_threshold = None

        bottom_threshold = None
        if leg_mode == "bottom":
            bottom_threshold = self._seconds_to_frames(
                scale_legs_to_bottom_active_seconds,
                fps_int,
            )
            if bottom_threshold == 0:
                bottom_threshold = None

        head_threshold = self._seconds_to_frames(
            head_padding_active_seconds,
            fps_int,
        )
        if head_threshold == 0:
            head_threshold = None

        foot_threshold = self._seconds_to_frames(
            foot_padding_active_seconds,
            fps_int,
        )
        if foot_threshold == 0:
            foot_threshold = None

        bottom_frames = 0
        head_frames = 0
        foot_frames = 0
        lock_frames = 0
        locked_scale = None

        for idx, meta in enumerate(pose_metas):
            if person_index >= 0 and idx != person_index:
                continue

            bottom_active = True
            allow_leg_adjustment = True

            if leg_mode == "bottom":
                bottom_active = (
                    bottom_threshold is None or bottom_frames < bottom_threshold
                )
                allow_leg_adjustment = bottom_active

            use_locked_scale = (
                lock_threshold is not None
                and lock_frames >= lock_threshold
                and locked_scale is not None
            )

            if leg_mode == "bottom":
                allow_leg_adjustment = allow_leg_adjustment or use_locked_scale

            head_active = head_threshold is None or head_frames < head_threshold
            foot_active = foot_threshold is None or foot_frames < foot_threshold

            scale_used = self._auto_align_meta_v5(
                meta,
                leg_mode,
                scale_legs,
                torso_head_multiple,
                head_padding,
                head_padding_normalized,
                head_active,
                foot_padding,
                foot_padding_normalized,
                foot_active,
                center_horizontally,
                limit_to_canvas and not use_locked_scale,
                dont_offset_feet,
                upper_body_offset,
                upper_body_offset_normalized,
                locked_scale if use_locked_scale else None,
                allow_leg_adjustment,
            )

            if leg_mode == "bottom":
                if bottom_active:
                    bottom_frames += 1
                elif use_locked_scale:
                    bottom_frames += 1
                elif bottom_threshold is not None and bottom_frames < bottom_threshold:
                    bottom_frames = bottom_threshold

            if head_threshold is not None and head_active:
                head_frames += 1

            if foot_threshold is not None and foot_active:
                foot_frames += 1

            if lock_threshold is not None:
                if lock_frames < lock_threshold:
                    lock_frames += 1
                    if (
                        scale_used is not None
                        and np.isfinite(scale_used)
                        and scale_used > 0.0
                    ):
                        locked_scale = float(scale_used)
                elif use_locked_scale:
                    lock_frames += 1
                elif (
                    locked_scale is None
                    and scale_used is not None
                    and np.isfinite(scale_used)
                    and scale_used > 0.0
                ):
                    locked_scale = float(scale_used)

        return (pose_data_copy,)

    def _auto_align_meta_v5(
        self,
        meta,
        leg_mode,
        scale_legs,
        torso_head_multiple,
        head_padding,
        head_padding_normalized,
        head_active,
        foot_padding,
        foot_padding_normalized,
        foot_active,
        center_horizontally,
        limit_to_canvas,
        dont_offset_feet,
        upper_body_offset,
        upper_body_offset_normalized,
        locked_scale,
        allow_leg_adjustment,
    ):
        width = getattr(meta, "width", None)
        height = getattr(meta, "height", None)

        if width in (None, 0) or height in (None, 0):
            return None

        mode_key = (leg_mode or "").strip().lower()

        locked_feet = None
        if dont_offset_feet or mode_key == "offset":
            locked_feet = self._capture_foot_positions(meta)

        if head_active:
            head_pad_px = self._resolve_padding(
                head_padding,
                head_padding_normalized,
                height,
            )
            head_pad_px = float(np.clip(head_pad_px, 0.0, float(height)))
            self._translate_head_to_padding(meta, head_pad_px, width, height)

        if mode_key == "offset":
            offset_px = self._resolve_padding(
                upper_body_offset,
                upper_body_offset_normalized,
                height,
            )
            offset_px = float(np.clip(offset_px, -float(height), float(height)))

            hip_pairs = self._offset_upper_body(meta, offset_px)

            if hip_pairs:
                self._stretch_legs_to_hips(meta, hip_pairs, locked_feet)

            if locked_feet:
                self._restore_locked_feet(meta, locked_feet)

            if center_horizontally:
                self._centre_pose(meta, width)

            if limit_to_canvas:
                self._clamp_pose(meta, width, height)

            return None

        foot_pad_px = self._resolve_padding(
            foot_padding,
            foot_padding_normalized,
            height,
        )
        foot_pad_px = float(np.clip(foot_pad_px, 0.0, float(height)))

        scale_used = None
        if allow_leg_adjustment:
            scale_used = self._adjust_leg_length_with_lock(
                meta,
                foot_pad_px,
                width,
                height,
                mode_key,
                scale_legs,
                torso_head_multiple,
                locked_scale,
            )

        if dont_offset_feet and locked_feet:
            self._restore_locked_feet(meta, locked_feet)

        if mode_key in ("normal", "relative") and foot_active and not dont_offset_feet:
            self._shift_pose_to_foot_padding(meta, foot_pad_px, width, height)

        if center_horizontally:
            self._centre_pose(meta, width)

        if limit_to_canvas:
            self._clamp_pose(meta, width, height)

        return scale_used

    def _adjust_leg_length_with_lock(
        self,
        meta,
        foot_padding_px,
        width,
        height,
        leg_mode,
        scale_legs,
        torso_head_multiple,
        locked_scale,
    ):
        anchor_y = self._compute_leg_anchor(meta)
        if anchor_y is None:
            return None

        bottom_y = self._find_leg_bottom(meta)
        if bottom_y is None:
            return None

        span = bottom_y - anchor_y
        if span <= 1e-6:
            return None

        target_floor = float(height) - float(foot_padding_px)
        target_floor = float(np.clip(target_floor, 0.0, float(height)))
        available_downward = target_floor - anchor_y

        if locked_scale is not None:
            scale = float(locked_scale)
        else:
            mode_key = (leg_mode or "").strip().lower()
            if mode_key == "normal":
                multiplier = max(float(scale_legs), 0.0)
                if multiplier <= 0.0:
                    return None
                scale = multiplier
            elif mode_key == "relative":
                upper_length = self._compute_upper_body_length(meta, anchor_y)
                if upper_length is None or upper_length <= 1e-6:
                    return None
                multiplier = max(float(torso_head_multiple), 0.0)
                if multiplier <= 0.0:
                    return None
                desired_span = upper_length * multiplier
                if desired_span <= 1e-6:
                    return None
                scale = desired_span / span
            else:
                if available_downward <= 1e-6:
                    return None
                scale = available_downward / span

            if mode_key != "bottom":
                if available_downward <= 1e-6:
                    if scale >= 1.0:
                        return None
                else:
                    max_scale = available_downward / span
                    if max_scale <= 0.0:
                        if scale >= 1.0:
                            return None
                    elif scale > max_scale:
                        scale = max_scale

        if not np.isfinite(scale) or scale <= 0.0:
            return None

        self._scale_leg_points(meta, anchor_y, scale)
        return float(scale)

    @staticmethod
    def _seconds_to_frames(seconds, fps):
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return 0

        if seconds <= 0.0:
            return 0

        fps = max(1, int(fps))
        return max(1, int(round(fps * seconds)))


class PoseDataEditorAutomaticV6(PoseDataEditorAutomaticV5):
    """Locks leg scales and padding offsets after configurable durations."""

    DESCRIPTION = (
        "Extends Automatic V5 by keeping head and foot padding active for their full "
        "durations before freezing the measured offsets. Also renames the upper-body "
        "mode toggle to clarify that legs are stretched when the torso is shifted."
    )

    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        required = inputs.get("required", {})

        ordered = {}
        for key, value in required.items():
            if key == "offset_upper_body_only":
                dtype, opts = value
                new_opts = dict(opts)
                new_opts["tooltip"] = (
                    "Enable the torso-to-head offset mode that raises the upper body "
                    "while stretching the legs to reconnect without moving the feet."
                )
                ordered["scale_offset_upper_body_only"] = (dtype, new_opts)
                ordered["active_seconds_to_scale_offset_upper_body_only"] = (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "How long to keep adapting the upper-body offset mode before "
                            "reusing the captured shift. Use 0 for no time limit."
                        ),
                    },
                )
                continue

            if key == "head_padding_active_seconds":
                dtype, opts = value
                new_opts = dict(opts)
                new_opts["tooltip"] = (
                    "How long to keep adapting the automatic head padding before the "
                    "current offset is locked and reused. Use 0 for no time limit."
                )
                ordered[key] = (dtype, new_opts)
                continue

            if key == "foot_padding_active_seconds":
                dtype, opts = value
                new_opts = dict(opts)
                new_opts["tooltip"] = (
                    "How long to keep adapting the automatic foot padding before the "
                    "current offset is locked and reused. Use 0 for no time limit."
                )
                ordered[key] = (dtype, new_opts)
                continue

            if key == "activate_lock_scale_after_seconds":
                dtype, opts = value
                new_opts = dict(opts)
                new_opts["tooltip"] = (
                    "Time in seconds before the measured leg scale and padding offsets "
                    "are frozen and reused. Set to 0 to keep adapting throughout the clip."
                )
                ordered["activate_lock_scale_offset_after_seconds"] = (dtype, new_opts)
                ordered["allow_upper_body_overflow"] = (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Permit the torso offset to push points outside the canvas "
                            "instead of clamping them, so the upper body can leave the frame."
                        ),
                    },
                )
                continue

            ordered[key] = value

        inputs["required"] = ordered
        return inputs

    def process(
        self,
        pose_data,
        scale_legs_to_bottom,
        scale_legs_to_bottom_active_seconds,
        scale_legs_normal,
        scale_legs_relative_to_body,
        scale_offset_upper_body_only,
        active_seconds_to_scale_offset_upper_body_only,
        upper_body_offset,
        upper_body_offset_normalized,
        scale_legs,
        torso_head_multiple,
        head_padding,
        head_padding_normalized,
        head_padding_active_seconds,
        foot_padding,
        foot_padding_normalized,
        foot_padding_active_seconds,
        center_horizontally,
        limit_to_canvas,
        activate_lock_scale_offset_after_seconds,
        allow_upper_body_overflow,
        dont_offset_feet,
        fps,
        person_index,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        leg_mode = self._select_leg_mode(
            scale_legs_to_bottom,
            scale_legs_normal,
            scale_legs_relative_to_body,
            scale_offset_upper_body_only,
        )

        fps_int = max(1, int(round(float(fps))))

        lock_threshold = self._seconds_to_frames(
            activate_lock_scale_offset_after_seconds,
            fps_int,
        )
        if lock_threshold == 0:
            lock_threshold = None

        offset_threshold = None
        if leg_mode == "offset":
            offset_threshold = self._seconds_to_frames(
                active_seconds_to_scale_offset_upper_body_only,
                fps_int,
            )
            if offset_threshold == 0:
                offset_threshold = None

        bottom_threshold = None
        if leg_mode == "bottom":
            bottom_threshold = self._seconds_to_frames(
                scale_legs_to_bottom_active_seconds,
                fps_int,
            )
            if bottom_threshold == 0:
                bottom_threshold = None

        head_threshold = self._seconds_to_frames(
            head_padding_active_seconds,
            fps_int,
        )
        if head_threshold == 0:
            head_threshold = None

        foot_threshold = self._seconds_to_frames(
            foot_padding_active_seconds,
            fps_int,
        )
        if foot_threshold == 0:
            foot_threshold = None

        bottom_frames = 0
        head_frames = 0
        foot_frames = 0
        lock_frames = 0
        offset_frames = 0
        locked_scale = None
        locked_head_delta = None
        locked_foot_delta = None
        locked_offset_delta = None

        for idx, meta in enumerate(pose_metas):
            if person_index >= 0 and idx != person_index:
                continue

            bottom_active = True
            allow_leg_adjustment = True
            offset_active = True

            if leg_mode == "bottom":
                bottom_active = (
                    bottom_threshold is None or bottom_frames < bottom_threshold
                )
                allow_leg_adjustment = bottom_active

            if leg_mode == "offset":
                offset_active = (
                    offset_threshold is None or offset_frames < offset_threshold
                )

            head_adapt_allowed = head_threshold is None or head_frames < head_threshold
            foot_adapt_allowed = foot_threshold is None or foot_frames < foot_threshold

            use_locked_scale = (
                lock_threshold is not None
                and lock_frames >= lock_threshold
                and locked_scale is not None
            )

            lock_offsets_active = lock_threshold is not None and lock_frames >= lock_threshold

            if leg_mode == "bottom":
                allow_leg_adjustment = allow_leg_adjustment or use_locked_scale

            if lock_offsets_active:
                head_adapt_allowed = False
                foot_adapt_allowed = False

            if dont_offset_feet:
                foot_adapt_allowed = False

            head_locked_delta = None if head_adapt_allowed else locked_head_delta
            foot_locked_delta = None if foot_adapt_allowed else locked_foot_delta

            scale_used, head_delta, foot_delta, offset_delta = self._auto_align_meta_v6(
                meta,
                leg_mode,
                scale_legs,
                torso_head_multiple,
                head_padding,
                head_padding_normalized,
                head_adapt_allowed,
                head_locked_delta,
                foot_padding,
                foot_padding_normalized,
                foot_adapt_allowed,
                foot_locked_delta,
                center_horizontally,
                limit_to_canvas and not use_locked_scale,
                dont_offset_feet,
                upper_body_offset,
                upper_body_offset_normalized,
                locked_scale if use_locked_scale else None,
                allow_leg_adjustment,
                offset_active,
                locked_offset_delta,
                allow_upper_body_overflow,
            )

            if leg_mode == "bottom":
                if bottom_active:
                    bottom_frames += 1
                elif use_locked_scale:
                    bottom_frames += 1
                elif bottom_threshold is not None and bottom_frames < bottom_threshold:
                    bottom_frames = bottom_threshold

            if head_threshold is not None and head_adapt_allowed:
                head_frames += 1

            if foot_threshold is not None and foot_adapt_allowed:
                foot_frames += 1

            if leg_mode == "offset":
                if offset_threshold is not None and offset_active:
                    offset_frames += 1
                elif (
                    offset_threshold is not None
                    and offset_frames < offset_threshold
                ):
                    offset_frames = offset_threshold

            if head_delta is not None:
                if head_adapt_allowed or locked_head_delta is None:
                    locked_head_delta = float(head_delta)

            if foot_delta is not None and not dont_offset_feet:
                if foot_adapt_allowed or locked_foot_delta is None:
                    locked_foot_delta = float(foot_delta)

            if leg_mode == "offset" and offset_delta is not None:
                if offset_active or locked_offset_delta is None:
                    locked_offset_delta = float(offset_delta)

            if lock_threshold is not None:
                if lock_frames < lock_threshold:
                    lock_frames += 1
                    if (
                        scale_used is not None
                        and np.isfinite(scale_used)
                        and scale_used > 0.0
                    ):
                        locked_scale = float(scale_used)
                    if head_delta is not None:
                        locked_head_delta = float(head_delta)
                    if foot_delta is not None and not dont_offset_feet:
                        locked_foot_delta = float(foot_delta)
                elif use_locked_scale or lock_offsets_active:
                    lock_frames += 1
                else:
                    if (
                        locked_scale is None
                        and scale_used is not None
                        and np.isfinite(scale_used)
                        and scale_used > 0.0
                    ):
                        locked_scale = float(scale_used)
                    if locked_head_delta is None and head_delta is not None:
                        locked_head_delta = float(head_delta)
                    if (
                        locked_foot_delta is None
                        and foot_delta is not None
                        and not dont_offset_feet
                    ):
                        locked_foot_delta = float(foot_delta)
                    if (
                        locked_offset_delta is None
                        and offset_delta is not None
                        and leg_mode == "offset"
                    ):
                        locked_offset_delta = float(offset_delta)

        return (pose_data_copy,)

    def _auto_align_meta_v6(
        self,
        meta,
        leg_mode,
        scale_legs,
        torso_head_multiple,
        head_padding,
        head_padding_normalized,
        head_adapt_allowed,
        head_locked_delta,
        foot_padding,
        foot_padding_normalized,
        foot_adapt_allowed,
        foot_locked_delta,
        center_horizontally,
        limit_to_canvas,
        dont_offset_feet,
        upper_body_offset,
        upper_body_offset_normalized,
        locked_scale,
        allow_leg_adjustment,
        offset_adapt_allowed,
        locked_offset_delta,
        allow_upper_body_overflow,
    ):
        width = getattr(meta, "width", None)
        height = getattr(meta, "height", None)

        if width in (None, 0) or height in (None, 0):
            return (None, None, None, None)

        mode_key = (leg_mode or "").strip().lower()

        locked_feet = None
        if dont_offset_feet or mode_key == "offset":
            locked_feet = self._capture_foot_positions(meta)

        head_delta = None
        head_pad_px = self._resolve_padding(
            head_padding,
            head_padding_normalized,
            height,
        )
        head_pad_px = float(np.clip(head_pad_px, 0.0, float(height)))

        if head_adapt_allowed:
            head_delta = self._translate_head_to_padding(
                meta,
                head_pad_px,
                width,
                height,
            )
        elif head_locked_delta is not None:
            head_delta = self._translate_head_to_padding(
                meta,
                head_pad_px,
                width,
                height,
                locked_delta=head_locked_delta,
            )

        if mode_key == "offset":
            if offset_adapt_allowed or locked_offset_delta is None:
                offset_px = self._resolve_padding(
                    upper_body_offset,
                    upper_body_offset_normalized,
                    height,
                )
                offset_px = float(np.clip(offset_px, -float(height), float(height)))
            else:
                offset_px = float(locked_offset_delta)

            hip_pairs = self._offset_upper_body(meta, offset_px)

            if hip_pairs:
                self._stretch_legs_to_hips(meta, hip_pairs, locked_feet)

            if locked_feet:
                self._restore_locked_feet(meta, locked_feet)

            if center_horizontally:
                self._centre_pose(meta, width)

            if limit_to_canvas and not allow_upper_body_overflow:
                self._clamp_pose(meta, width, height)

            return (None, head_delta, None, offset_px)

        foot_pad_px = self._resolve_padding(
            foot_padding,
            foot_padding_normalized,
            height,
        )
        foot_pad_px = float(np.clip(foot_pad_px, 0.0, float(height)))

        scale_used = None
        if allow_leg_adjustment:
            scale_used = self._adjust_leg_length_with_lock(
                meta,
                foot_pad_px,
                width,
                height,
                mode_key,
                scale_legs,
                torso_head_multiple,
                locked_scale,
            )

        if dont_offset_feet and locked_feet:
            self._restore_locked_feet(meta, locked_feet)

        foot_delta = None
        if mode_key in ("normal", "relative") and not dont_offset_feet:
            if foot_adapt_allowed:
                foot_delta = self._shift_pose_to_foot_padding(
                    meta,
                    foot_pad_px,
                    width,
                    height,
                )
            elif foot_locked_delta is not None:
                foot_delta = self._shift_pose_to_foot_padding(
                    meta,
                    foot_pad_px,
                    width,
                    height,
                    locked_delta=foot_locked_delta,
                )

        if center_horizontally:
            self._centre_pose(meta, width)

        if limit_to_canvas:
            self._clamp_pose(meta, width, height)

        return (scale_used, head_delta, foot_delta, None)

    def _select_leg_mode(
        self,
        scale_legs_to_bottom,
        scale_legs_normal,
        scale_legs_relative_to_body,
        scale_offset_upper_body_only,
    ):
        return super()._select_leg_mode(
            scale_legs_to_bottom,
            scale_legs_normal,
            scale_legs_relative_to_body,
            scale_offset_upper_body_only,
        )


class PoseDataEditorAutomaticV7(PoseDataEditorAutomaticV6):
    """Adds timed torso offsets and overflow control on top of Automatic V6."""

    DESCRIPTION = (
        "Builds on Automatic V6 by introducing a duration control for the upper-body "
        "offset mode and an overflow toggle that lets the torso move beyond the "
        "canvas instead of being clamped."
    )


class PoseDataEditorAutomaticV8(PoseDataEditorAutomaticV7):
    """Allows head padding to be disabled or to expire without reapplying offsets."""

    DESCRIPTION = (
        "Extends Automatic V7 by adding a toggle to disable head padding entirely "
        "and ensuring the head offset stops once the active duration elapses instead "
        "of remaining locked against the canvas."
    )

    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        required = inputs.get("required", {})

        if "head_padding" in required and "enable_head_padding" not in required:
            reordered = {}
            for key, value in required.items():
                if key == "head_padding":
                    reordered["enable_head_padding"] = (
                        "BOOLEAN",
                        {
                            "default": True,
                            "tooltip": (
                                "Disable to skip automatic head padding entirely, even when "
                                "a duration is provided."
                            ),
                        },
                    )
                reordered[key] = value
            inputs["required"] = reordered

        return inputs

    def process(
        self,
        pose_data,
        scale_legs_to_bottom,
        scale_legs_to_bottom_active_seconds,
        scale_legs_normal,
        scale_legs_relative_to_body,
        scale_offset_upper_body_only,
        active_seconds_to_scale_offset_upper_body_only,
        upper_body_offset,
        upper_body_offset_normalized,
        scale_legs,
        torso_head_multiple,
        enable_head_padding,
        head_padding,
        head_padding_normalized,
        head_padding_active_seconds,
        foot_padding,
        foot_padding_normalized,
        foot_padding_active_seconds,
        center_horizontally,
        limit_to_canvas,
        activate_lock_scale_offset_after_seconds,
        allow_upper_body_overflow,
        dont_offset_feet,
        fps,
        person_index,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        leg_mode = self._select_leg_mode(
            scale_legs_to_bottom,
            scale_legs_normal,
            scale_legs_relative_to_body,
            scale_offset_upper_body_only,
        )

        fps_int = max(1, int(round(float(fps))))

        lock_threshold = self._seconds_to_frames(
            activate_lock_scale_offset_after_seconds,
            fps_int,
        )
        if lock_threshold == 0:
            lock_threshold = None

        offset_threshold = None
        if leg_mode == "offset":
            offset_threshold = self._seconds_to_frames(
                active_seconds_to_scale_offset_upper_body_only,
                fps_int,
            )
            if offset_threshold == 0:
                offset_threshold = None

        bottom_threshold = None
        if leg_mode == "bottom":
            bottom_threshold = self._seconds_to_frames(
                scale_legs_to_bottom_active_seconds,
                fps_int,
            )
            if bottom_threshold == 0:
                bottom_threshold = None

        head_threshold = None
        if enable_head_padding:
            head_threshold = self._seconds_to_frames(
                head_padding_active_seconds,
                fps_int,
            )
            if head_threshold == 0:
                head_threshold = None

        foot_threshold = self._seconds_to_frames(
            foot_padding_active_seconds,
            fps_int,
        )
        if foot_threshold == 0:
            foot_threshold = None

        bottom_frames = 0
        head_frames = 0
        foot_frames = 0
        lock_frames = 0
        offset_frames = 0
        locked_scale = None
        locked_head_delta = None
        locked_foot_delta = None
        locked_offset_delta = None

        for idx, meta in enumerate(pose_metas):
            if person_index >= 0 and idx != person_index:
                continue

            bottom_active = True
            allow_leg_adjustment = True
            offset_active = True

            if leg_mode == "bottom":
                bottom_active = (
                    bottom_threshold is None or bottom_frames < bottom_threshold
                )
                allow_leg_adjustment = bottom_active

            if leg_mode == "offset":
                offset_active = (
                    offset_threshold is None or offset_frames < offset_threshold
                )

            head_window_active = enable_head_padding and (
                head_threshold is None or head_frames < head_threshold
            )
            foot_adapt_allowed = foot_threshold is None or foot_frames < foot_threshold

            use_locked_scale = (
                lock_threshold is not None
                and lock_frames >= lock_threshold
                and locked_scale is not None
            )

            lock_offsets_active = lock_threshold is not None and lock_frames >= lock_threshold

            if leg_mode == "bottom":
                allow_leg_adjustment = allow_leg_adjustment or use_locked_scale

            head_adapt_allowed = head_window_active and not lock_offsets_active

            if lock_offsets_active:
                foot_adapt_allowed = False

            if dont_offset_feet:
                foot_adapt_allowed = False

            head_locked_delta = None
            if not head_adapt_allowed and head_window_active and locked_head_delta is not None:
                head_locked_delta = float(locked_head_delta)

            foot_locked_delta = None if foot_adapt_allowed else locked_foot_delta

            scale_used, head_delta, foot_delta, offset_delta = self._auto_align_meta_v6(
                meta,
                leg_mode,
                scale_legs,
                torso_head_multiple,
                head_padding,
                head_padding_normalized,
                head_adapt_allowed,
                head_locked_delta,
                foot_padding,
                foot_padding_normalized,
                foot_adapt_allowed,
                foot_locked_delta,
                center_horizontally,
                limit_to_canvas and not use_locked_scale,
                dont_offset_feet,
                upper_body_offset,
                upper_body_offset_normalized,
                locked_scale if use_locked_scale else None,
                allow_leg_adjustment,
                offset_active,
                locked_offset_delta,
                allow_upper_body_overflow,
            )

            if leg_mode == "bottom":
                if bottom_active:
                    bottom_frames += 1
                elif use_locked_scale:
                    bottom_frames += 1
                elif bottom_threshold is not None and bottom_frames < bottom_threshold:
                    bottom_frames = bottom_threshold

            if head_window_active:
                head_frames += 1
            else:
                locked_head_delta = None

            if foot_threshold is not None and foot_adapt_allowed:
                foot_frames += 1

            if leg_mode == "offset":
                if offset_threshold is not None and offset_active:
                    offset_frames += 1
                elif (
                    offset_threshold is not None
                    and offset_frames < offset_threshold
                ):
                    offset_frames = offset_threshold

            if head_delta is not None and enable_head_padding:
                if head_window_active or locked_head_delta is None:
                    locked_head_delta = float(head_delta)

            if foot_delta is not None and not dont_offset_feet:
                if foot_adapt_allowed or locked_foot_delta is None:
                    locked_foot_delta = float(foot_delta)

            if leg_mode == "offset" and offset_delta is not None:
                if offset_active or locked_offset_delta is None:
                    locked_offset_delta = float(offset_delta)

            if lock_threshold is not None:
                if lock_frames < lock_threshold:
                    lock_frames += 1
                    if (
                        scale_used is not None
                        and np.isfinite(scale_used)
                        and scale_used > 0.0
                    ):
                        locked_scale = float(scale_used)
                    if head_delta is not None and enable_head_padding:
                        locked_head_delta = float(head_delta)
                    if foot_delta is not None and not dont_offset_feet:
                        locked_foot_delta = float(foot_delta)
                elif use_locked_scale or lock_offsets_active:
                    lock_frames += 1
                else:
                    if (
                        locked_scale is None
                        and scale_used is not None
                        and np.isfinite(scale_used)
                        and scale_used > 0.0
                    ):
                        locked_scale = float(scale_used)
                    if (
                        locked_head_delta is None
                        and head_delta is not None
                        and enable_head_padding
                    ):
                        locked_head_delta = float(head_delta)
                    if (
                        locked_foot_delta is None
                        and foot_delta is not None
                        and not dont_offset_feet
                    ):
                        locked_foot_delta = float(foot_delta)
                    if (
                        locked_offset_delta is None
                        and offset_delta is not None
                        and leg_mode == "offset"
                    ):
                        locked_offset_delta = float(offset_delta)

        return (pose_data_copy,)


class PoseDataEditorAutomaticOnlyTorsoHeadOffset(PoseDataEditorAutomaticV4):
    """Provides the torso-to-head offset workflow as a dedicated node."""

    DESCRIPTION = (
        "Moves the upper body by a specified offset while keeping the feet planted. "
        "Automatically stretches the legs so the knees and hips reconnect after the "
        "shift, optionally recentring and clamping the pose to the canvas."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "upper_body_offset": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2048.0,
                        "max": 2048.0,
                        "step": 0.001,
                        "tooltip": (
                            "Amount to raise (+) or lower (-) the torso-to-head segment "
                            "before legs are stretched to meet the hips."
                        ),
                    },
                ),
                "upper_body_offset_normalized": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Interpret the upper-body offset as a 0-1 ratio of the canvas "
                            "height instead of raw pixels."
                        ),
                    },
                ),
                "head_padding": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.001,
                        "tooltip": (
                            "Distance to keep between the adjusted head points and the top "
                            "edge of the canvas before applying the offset."
                        ),
                    },
                ),
                "head_padding_normalized": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Interpret the head padding as a 0-1 ratio of the canvas height "
                            "instead of raw pixels."
                        ),
                    },
                ),
                "center_horizontally": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, horizontally centre the pose after applying the offset.",
                    },
                ),
                "limit_to_canvas": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Clamp all keypoints to the canvas bounds after adjustments are applied.",
                    },
                ),
                "person_index": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "When >= 0, only process the matching pose entry. Use -1 to process every pose.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"

    def process(
        self,
        pose_data,
        upper_body_offset,
        upper_body_offset_normalized,
        head_padding,
        head_padding_normalized,
        center_horizontally,
        limit_to_canvas,
        person_index,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        for idx, meta in enumerate(pose_metas):
            if person_index >= 0 and idx != person_index:
                continue

            self._auto_align_meta(
                meta,
                "offset",
                1.0,
                1.0,
                head_padding,
                head_padding_normalized,
                0.0,
                True,
                center_horizontally,
                limit_to_canvas,
                upper_body_offset,
                upper_body_offset_normalized,
            )

        return (pose_data_copy,)


class PoseDataEditorAutomaticOnlyTorsoHeadOffsetV2(
    PoseDataEditorAutomaticOnlyTorsoHeadOffset
):
    """Locks the upper-body scaling factor after five seconds of footage."""

    LOCK_SECONDS = 5.0

    LEG_FOOT_MAP = {
        8: (9, 10),
        11: (12, 13),
    }

    @classmethod
    def INPUT_TYPES(cls):
        parent = super().INPUT_TYPES()
        parent_required = parent["required"]

        required = {}

        required["pose_data"] = parent_required["pose_data"]
        required["upper_body_offset"] = parent_required["upper_body_offset"]
        required["upper_body_offset_normalized"] = parent_required[
            "upper_body_offset_normalized"
        ]
        required["offset_auto_duration_seconds"] = (
            "FLOAT",
            {
                "default": cls.LOCK_SECONDS,
                "min": 0.0,
                "max": 3600.0,
                "step": 0.01,
                "tooltip": (
                    "Number of seconds to keep adapting the torso-to-head offset before"
                    " locking the measured scale factor. Use 0 to lock immediately."
                ),
            },
        )
        required["head_padding"] = parent_required["head_padding"]
        required["head_padding_normalized"] = parent_required["head_padding_normalized"]
        required["auto_head_to_padding"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Automatically translate the pose so the head rests at the requested"
                    " padding distance from the top edge."
                ),
            },
        )
        required["head_padding_active_seconds"] = (
            "FLOAT",
            {
                "default": 0.0,
                "min": 0.0,
                "max": 3600.0,
                "step": 0.01,
                "tooltip": (
                    "How long to keep applying the automatic head padding before"
                    " disabling the translation entirely. Use 0 to keep it active"
                    " for the full clip."
                ),
            },
        )
        required["head_padding_allow_overflow"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "Allow the automatic head offset to move the pose outside the canvas"
                    " instead of clamping the padding target inside the frame."
                ),
            },
        )
        required["head_auto_duration_seconds"] = (
            "FLOAT",
            {
                "default": cls.LOCK_SECONDS,
                "min": 0.0,
                "max": 3600.0,
                "step": 0.01,
                "tooltip": (
                    "How long to keep adapting the automatic head offset before the"
                    " measured translation is locked. Use 0 to lock immediately."
                ),
            },
        )
        required["auto_feet_to_padding"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "When enabled, push the pose so the lowest leg points sit at the"
                    " requested foot padding above the canvas bottom."
                ),
            },
        )
        required["foot_padding"] = (
            "FLOAT",
            {
                "default": 0.02,
                "min": 0.0,
                "max": 2048.0,
                "step": 0.001,
                "tooltip": (
                    "Distance to keep between the adjusted feet and the bottom edge when"
                    " automatic foot alignment is enabled."
                ),
            },
        )
        required["foot_padding_normalized"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Interpret the foot padding as a 0-1 ratio of the canvas height"
                    " instead of raw pixels."
                ),
            },
        )
        required["foot_padding_active_seconds"] = (
            "FLOAT",
            {
                "default": 0.0,
                "min": 0.0,
                "max": 3600.0,
                "step": 0.01,
                "tooltip": (
                    "How long to keep applying the automatic foot padding before"
                    " disabling the translation entirely. Use 0 to keep it active"
                    " for the full clip."
                ),
            },
        )
        required["foot_auto_duration_seconds"] = (
            "FLOAT",
            {
                "default": cls.LOCK_SECONDS,
                "min": 0.0,
                "max": 3600.0,
                "step": 0.01,
                "tooltip": (
                    "How long to keep adapting the automatic foot alignment before the"
                    " measured translation is locked. Use 0 to lock immediately."
                ),
            },
        )
        required["center_horizontally"] = parent_required["center_horizontally"]
        required["limit_to_canvas"] = parent_required["limit_to_canvas"]
        required["fps"] = (
            "INT",
            {
                "default": 24,
                "min": 1,
                "max": 960,
                "step": 1,
                "tooltip": (
                    "Frames per second of the incoming clip. Durations are converted"
                    " into frame counts using this value."
                ),
            },
        )
        required["person_index"] = parent_required["person_index"]

        return {"required": required}

    def process(
        self,
        pose_data,
        upper_body_offset,
        upper_body_offset_normalized,
        offset_auto_duration_seconds,
        head_padding,
        head_padding_normalized,
        auto_head_to_padding,
        head_padding_active_seconds,
        head_padding_allow_overflow,
        head_auto_duration_seconds,
        auto_feet_to_padding,
        foot_padding,
        foot_padding_normalized,
        foot_padding_active_seconds,
        foot_auto_duration_seconds,
        center_horizontally,
        limit_to_canvas,
        fps,
        person_index,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        fps = int(max(1, fps))
        scale_threshold = self._seconds_to_frames(offset_auto_duration_seconds, fps)
        head_threshold = (
            self._seconds_to_frames(head_auto_duration_seconds, fps)
            if auto_head_to_padding
            else None
        )
        head_active_threshold = (
            self._seconds_to_frames(head_padding_active_seconds, fps)
            if auto_head_to_padding
            else None
        )
        if head_active_threshold == 0:
            head_active_threshold = None
        foot_threshold = (
            self._seconds_to_frames(foot_auto_duration_seconds, fps)
            if auto_feet_to_padding
            else None
        )
        foot_active_threshold = (
            self._seconds_to_frames(foot_padding_active_seconds, fps)
            if auto_feet_to_padding
            else None
        )
        if foot_active_threshold == 0:
            foot_active_threshold = None

        locked_scale = None
        locked_head_delta = None
        locked_foot_delta = None
        adaptive_scale_frames = 0
        adaptive_head_frames = 0
        adaptive_foot_frames = 0
        head_active_frames = 0
        foot_active_frames = 0

        for idx, meta in enumerate(pose_metas):
            if person_index >= 0 and idx != person_index:
                continue

            head_active = auto_head_to_padding and (
                head_active_threshold is None or head_active_frames < head_active_threshold
            )
            foot_active = auto_feet_to_padding and (
                foot_active_threshold is None or foot_active_frames < foot_active_threshold
            )

            use_locked_scale = (
                locked_scale is not None
                and (scale_threshold == 0 or adaptive_scale_frames >= scale_threshold)
            )

            use_locked_head = (
                auto_head_to_padding
                and head_active
                and locked_head_delta is not None
                and (
                    head_threshold == 0
                    or (
                        head_threshold is not None
                        and adaptive_head_frames >= head_threshold
                    )
                )
            )

            use_locked_feet = (
                auto_feet_to_padding
                and foot_active
                and locked_foot_delta is not None
                and (
                    foot_threshold == 0
                    or (
                        foot_threshold is not None
                        and adaptive_foot_frames >= foot_threshold
                    )
                )
            )

            scale_used, head_delta, foot_delta = self._auto_align_with_scale_lock(
                meta,
                head_padding,
                head_padding_normalized,
                auto_head_to_padding,
                head_active,
                head_padding_allow_overflow,
                center_horizontally,
                limit_to_canvas,
                upper_body_offset,
                upper_body_offset_normalized,
                locked_scale if use_locked_scale else None,
                locked_head_delta if use_locked_head else None,
                auto_feet_to_padding,
                foot_padding,
                foot_padding_normalized,
                foot_active,
                locked_foot_delta if use_locked_feet else None,
            )

            if not use_locked_scale:
                adaptive_scale_frames += 1
                if (
                    scale_used is not None
                    and np.isfinite(scale_used)
                    and scale_used > 0.0
                ):
                    locked_scale = float(scale_used)

            if auto_head_to_padding and head_active and not use_locked_head:
                adaptive_head_frames += 1
                if head_delta is not None and np.isfinite(head_delta):
                    locked_head_delta = float(head_delta)

            if auto_feet_to_padding and foot_active and not use_locked_feet:
                adaptive_foot_frames += 1
                if foot_delta is not None and np.isfinite(foot_delta):
                    locked_foot_delta = float(foot_delta)

            if auto_head_to_padding and head_active_threshold is not None and head_active:
                head_active_frames += 1

            if auto_feet_to_padding and foot_active_threshold is not None and foot_active:
                foot_active_frames += 1

        return (pose_data_copy,)

    @staticmethod
    def _seconds_to_frames(seconds, fps):
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return 0

        if seconds <= 0.0:
            return 0

        fps = max(1, int(fps))
        return max(1, int(round(fps * seconds)))

    def _auto_align_with_scale_lock(
        self,
        meta,
        head_padding,
        head_padding_normalized,
        auto_head_to_padding,
        head_padding_active,
        head_padding_allow_overflow,
        center_horizontally,
        limit_to_canvas,
        upper_body_offset,
        upper_body_offset_normalized,
        locked_scale,
        locked_head_delta,
        auto_feet_to_padding,
        foot_padding,
        foot_padding_normalized,
        foot_padding_active,
        locked_foot_delta,
    ):
        width = getattr(meta, "width", None)
        height = getattr(meta, "height", None)

        if width in (None, 0) or height in (None, 0):
            return (None, None, None)

        head_delta = None
        head_pad_px = None
        if auto_head_to_padding and head_padding_active:
            head_pad_px = self._resolve_padding(
                head_padding,
                head_padding_normalized,
                height,
            )
            if head_padding_allow_overflow:
                head_pad_px = float(head_pad_px)
            else:
                head_pad_px = float(np.clip(head_pad_px, 0.0, float(height)))

            delta = locked_head_delta
            if delta is None:
                top_y = self._find_head_top(meta)
                if top_y is None:
                    top_y = self._find_pose_top(meta)
                if top_y is not None:
                    delta = head_pad_px - float(top_y)

            if delta is not None:
                delta = float(delta)
                if abs(delta) > 1e-6:
                    self._translate_pose(meta, 0.0, delta, width, height)
                head_delta = delta

        locked_feet = self._capture_foot_positions(meta)

        body = getattr(meta, "kps_body", None)
        if body is None:
            return (None, head_delta, None)

        hip_coords = {}
        for hip_idx in self.HIP_INDICES:
            if hip_idx >= len(body):
                continue
            coords = self._extract_coords(body[hip_idx])
            if coords is None:
                continue
            hip_coords[hip_idx] = np.array(coords, dtype=np.float32)

        offset_px = self._resolve_padding(
            upper_body_offset,
            upper_body_offset_normalized,
            height,
        )

        offset_px = float(np.clip(offset_px, -float(height), float(height)))

        if locked_scale is not None and hip_coords:
            offset_candidates = self._estimate_offset_from_scale(
                hip_coords,
                body,
                locked_feet,
                locked_scale,
            )
            if offset_candidates:
                offset_px = float(np.clip(
                    np.median(offset_candidates),
                    -float(height),
                    float(height),
                ))

        hip_pairs = self._offset_upper_body(meta, offset_px)

        if hip_pairs:
            self._stretch_legs_to_hips(meta, hip_pairs, locked_feet)

        if locked_feet:
            self._restore_locked_feet(meta, locked_feet)

        foot_delta = None
        locked_feet_after_translation = None
        if auto_feet_to_padding and foot_padding_active:
            foot_pad_px = self._resolve_padding(
                foot_padding,
                foot_padding_normalized,
                height,
            )
            foot_pad_px = float(np.clip(foot_pad_px, 0.0, float(height)))

            delta = locked_foot_delta
            if delta is None:
                bottom_y = self._find_leg_bottom(meta)
                if bottom_y is not None:
                    target_floor = float(height) - float(foot_pad_px)
                    target_floor = float(np.clip(target_floor, 0.0, float(height)))
                    delta = target_floor - float(bottom_y)

            if delta is not None:
                delta = float(delta)
                if abs(delta) > 1e-6:
                    self._translate_pose(meta, 0.0, delta, width, height)
                    locked_feet_after_translation = self._capture_foot_positions(meta)
                foot_delta = delta

        anchor_feet = locked_feet_after_translation or locked_feet

        if auto_head_to_padding and head_padding_active and head_pad_px is not None:
            top_y_after = self._find_head_top(meta)
            if top_y_after is None:
                top_y_after = self._find_pose_top(meta)

            if top_y_after is not None:
                extra_delta = float(head_pad_px) - float(top_y_after)
                if abs(extra_delta) > 1e-6:
                    hip_pairs_extra = self._offset_upper_body(meta, extra_delta)
                    if hip_pairs_extra:
                        if hip_pairs:
                            for hip_idx, (_, after) in hip_pairs_extra.items():
                                if hip_idx in hip_pairs:
                                    hip_pairs[hip_idx] = (
                                        hip_pairs[hip_idx][0],
                                        after,
                                    )
                                else:
                                    hip_pairs[hip_idx] = hip_pairs_extra[hip_idx]
                        else:
                            hip_pairs.update(hip_pairs_extra)

                        self._stretch_legs_to_hips(
                            meta,
                            hip_pairs_extra,
                            anchor_feet,
                        )

                    if anchor_feet:
                        self._restore_locked_feet(meta, anchor_feet)

        if center_horizontally:
            self._centre_pose(meta, width)

        if limit_to_canvas:
            self._clamp_pose(meta, width, height)

        return (
            self._measure_leg_scale(hip_pairs, locked_feet, meta),
            head_delta,
            foot_delta,
        )


class PoseDataEditorAutoPositioning(
    PoseDataEditorAutomaticOnlyTorsoHeadOffset
):
    """Automatically lifts poses so the head rests at the requested padding."""

    DEFAULT_DURATION_SECONDS = 5.0

    @classmethod
    def INPUT_TYPES(cls):
        parent = PoseDataEditorAutomaticOnlyTorsoHeadOffset.INPUT_TYPES()
        parent_required = parent["required"]

        return {
            "required": {
                "pose_data": parent_required["pose_data"],
                "head_padding": parent_required["head_padding"],
                "head_padding_normalized": parent_required["head_padding_normalized"],
                "head_padding_allow_overflow": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Allow the automatic offset to move the head above the canvas "
                            "when the requested padding would otherwise clamp inside the frame."
                        ),
                    },
                ),
                "padding_active_seconds": (
                    "FLOAT",
                    {
                        "default": cls.DEFAULT_DURATION_SECONDS,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "How long to keep nudging the pose toward the requested padding "
                            "before leaving it in place. Use 0 to keep it active for the whole clip."
                        ),
                    },
                ),
                "center_horizontally": parent_required["center_horizontally"],
                "limit_to_canvas": parent_required["limit_to_canvas"],
                "fps": (
                    "INT",
                    {
                        "default": 24,
                        "min": 1,
                        "max": 960,
                        "step": 1,
                        "tooltip": (
                            "Frames per second of the input clip. Durations are converted into "
                            "frame counts using this value."
                        ),
                    },
                ),
                "person_index": parent_required["person_index"],
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"

    def process(
        self,
        pose_data,
        head_padding,
        head_padding_normalized,
        head_padding_allow_overflow,
        padding_active_seconds,
        center_horizontally,
        limit_to_canvas,
        fps,
        person_index,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        fps = int(max(1, fps))
        active_threshold = self._seconds_to_frames(padding_active_seconds, fps)
        if active_threshold == 0:
            active_threshold = None

        active_frames = {}

        for idx, meta in enumerate(pose_metas):
            if person_index >= 0 and idx != person_index:
                continue

            used_frames = active_frames.get(idx, 0)
            head_active = active_threshold is None or used_frames < active_threshold

            self._align_head_only(
                meta,
                head_padding,
                head_padding_normalized,
                head_padding_allow_overflow,
                center_horizontally,
                limit_to_canvas,
                head_active,
            )

            if head_active and active_threshold is not None:
                active_frames[idx] = used_frames + 1

        return (pose_data_copy,)

    @staticmethod
    def _seconds_to_frames(seconds, fps):
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return 0

        if seconds <= 0.0:
            return 0

        return max(1, int(round(float(fps) * seconds)))

    def _align_head_only(
        self,
        meta,
        head_padding,
        head_padding_normalized,
        head_padding_allow_overflow,
        center_horizontally,
        limit_to_canvas,
        head_active,
    ):
        width = getattr(meta, "width", None)
        height = getattr(meta, "height", None)

        if width in (None, 0) or height in (None, 0):
            return

        head_pad_px = self._resolve_padding(
            head_padding,
            head_padding_normalized,
            height,
        )
        if head_padding_allow_overflow:
            head_pad_px = float(head_pad_px)
        else:
            head_pad_px = float(np.clip(head_pad_px, 0.0, float(height)))

        if head_active:
            top_y = self._find_head_top(meta)
            if top_y is None:
                top_y = self._find_pose_top(meta)

            if top_y is not None:
                delta = float(head_pad_px) - float(top_y)
                if abs(delta) > 1e-6:
                    self._translate_pose(meta, 0.0, delta, width, height)

        if center_horizontally:
            self._centre_pose(meta, width)

        if limit_to_canvas:
            self._clamp_pose(meta, width, height)

    def _estimate_offset_from_scale(self, hip_coords, body, locked_feet, locked_scale):
        offsets = []

        if locked_scale is None or locked_scale <= 0.0:
            return offsets

        for hip_idx, before in hip_coords.items():
            mapping = self.LEG_FOOT_MAP.get(hip_idx)
            if not mapping:
                continue

            _, ankle_idx = mapping

            foot_coords = locked_feet.get(ankle_idx)
            if foot_coords is None and ankle_idx < len(body):
                coords = self._extract_coords(body[ankle_idx])
                if coords is not None:
                    foot_coords = np.array(coords, dtype=np.float32)

            if foot_coords is None:
                continue

            before_diff = float(before[1] - foot_coords[1])
            if not np.isfinite(before_diff) or abs(before_diff) <= 1e-6:
                continue

            candidate = (1.0 - float(locked_scale)) * before_diff
            if np.isfinite(candidate):
                offsets.append(candidate)

        return offsets

    def _measure_leg_scale(self, hip_pairs, locked_feet, meta):
        if not hip_pairs:
            return None

        body = getattr(meta, "kps_body", None)
        if body is None:
            return None

        ratios = []

        for hip_idx, (before, after) in hip_pairs.items():
            mapping = self.LEG_FOOT_MAP.get(hip_idx)
            if not mapping:
                continue

            _, ankle_idx = mapping
            foot_coords = locked_feet.get(ankle_idx)
            if foot_coords is None and ankle_idx < len(body):
                coords = self._extract_coords(body[ankle_idx])
                if coords is not None:
                    foot_coords = np.array(coords, dtype=np.float32)

            if foot_coords is None:
                continue

            before_vec = before - foot_coords
            after_vec = after - foot_coords

            before_len = float(np.linalg.norm(before_vec))
            after_len = float(np.linalg.norm(after_vec))

            if before_len <= 1e-6 or not np.isfinite(before_len):
                continue
            if after_len <= 1e-6 or not np.isfinite(after_len):
                continue

            ratio = after_len / before_len
            if np.isfinite(ratio) and ratio > 0.0:
                ratios.append(ratio)

        if not ratios:
            return None

        return float(np.median(ratios))


class PoseDataEditorAutomaticPositioningAndStretching(
    PoseDataEditorAutomaticOnlyTorsoHeadOffsetV2
):
    """Automatically positions the head and feet with timed leg stretching."""

    DEFAULT_DURATION_SECONDS = 5.0

    @classmethod
    def INPUT_TYPES(cls):
        parent = PoseDataEditorAutomaticOnlyTorsoHeadOffset.INPUT_TYPES()
        parent_required = parent["required"]

        return {
            "required": {
                "pose_data": parent_required["pose_data"],
                "auto_head_to_padding": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Enable automatic torso positioning so the head rests at the"
                            " requested padding distance from the canvas top."
                        ),
                    },
                ),
                "head_padding": parent_required["head_padding"],
                "head_padding_normalized": parent_required["head_padding_normalized"],
                "head_padding_allow_overflow": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Allow the automatic head positioning to move beyond the canvas"
                            " instead of clamping the padding inside the frame."
                        ),
                    },
                ),
                "head_padding_active_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "How long to keep translating the pose toward the head padding"
                            " before disabling the automatic positioning entirely."
                            " Use 0 to keep it active for the full clip."
                        ),
                    },
                ),
                "head_auto_duration_seconds": (
                    "FLOAT",
                    {
                        "default": cls.DEFAULT_DURATION_SECONDS,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "Number of seconds to keep adapting the head offset before"
                            " locking the measured translation. Use 0 to lock immediately."
                        ),
                    },
                ),
                "auto_feet_to_padding": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Automatically translate the pose so the feet rest at the"
                            " requested padding distance from the canvas bottom."
                        ),
                    },
                ),
                "foot_padding": parent_required["foot_padding"],
                "foot_padding_normalized": parent_required["foot_padding_normalized"],
                "foot_padding_active_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "How long to keep nudging the pose toward the requested foot"
                            " padding before disabling the automatic translation."
                            " Use 0 to keep it active for the full clip."
                        ),
                    },
                ),
                "foot_auto_duration_seconds": (
                    "FLOAT",
                    {
                        "default": cls.DEFAULT_DURATION_SECONDS,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "Number of seconds to keep adapting the automatic foot offset"
                            " before locking the translation. Use 0 to lock immediately."
                        ),
                    },
                ),
                "offset_auto_duration_seconds": (
                    "FLOAT",
                    {
                        "default": cls.DEFAULT_DURATION_SECONDS,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.01,
                        "tooltip": (
                            "How long to keep adapting the leg stretching before locking"
                            " the measured scale factor. Use 0 to lock immediately."
                        ),
                    },
                ),
                "center_horizontally": parent_required["center_horizontally"],
                "limit_to_canvas": parent_required["limit_to_canvas"],
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 960.0,
                        "step": 0.01,
                        "tooltip": (
                            "Frames per second of the input clip. Durations are converted"
                            " into frame counts using this value."
                        ),
                    },
                ),
                "person_index": parent_required["person_index"],
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"

    def process(
        self,
        pose_data,
        auto_head_to_padding,
        head_padding,
        head_padding_normalized,
        head_padding_allow_overflow,
        head_padding_active_seconds,
        head_auto_duration_seconds,
        auto_feet_to_padding,
        foot_padding,
        foot_padding_normalized,
        foot_padding_active_seconds,
        foot_auto_duration_seconds,
        offset_auto_duration_seconds,
        center_horizontally,
        limit_to_canvas,
        fps,
        person_index,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy,)

        try:
            fps_value = float(fps)
        except (TypeError, ValueError):
            fps_value = 24.0

        fps_int = max(1, int(round(fps_value)))

        scale_threshold = self._seconds_to_frames(offset_auto_duration_seconds, fps_int)
        head_threshold = (
            self._seconds_to_frames(head_auto_duration_seconds, fps_int)
            if auto_head_to_padding
            else None
        )
        head_active_threshold = (
            self._seconds_to_frames(head_padding_active_seconds, fps_int)
            if auto_head_to_padding
            else None
        )
        if head_active_threshold == 0:
            head_active_threshold = None

        foot_threshold = (
            self._seconds_to_frames(foot_auto_duration_seconds, fps_int)
            if auto_feet_to_padding
            else None
        )
        foot_active_threshold = (
            self._seconds_to_frames(foot_padding_active_seconds, fps_int)
            if auto_feet_to_padding
            else None
        )
        if foot_active_threshold == 0:
            foot_active_threshold = None

        locked_scale = None
        locked_head_delta = None
        locked_foot_delta = None
        adaptive_scale_frames = 0
        adaptive_head_frames = 0
        adaptive_foot_frames = 0
        head_active_frames = 0
        foot_active_frames = 0

        for idx, meta in enumerate(pose_metas):
            if person_index >= 0 and idx != person_index:
                continue

            head_active = auto_head_to_padding and (
                head_active_threshold is None or head_active_frames < head_active_threshold
            )
            foot_active = auto_feet_to_padding and (
                foot_active_threshold is None or foot_active_frames < foot_active_threshold
            )

            use_locked_scale = (
                locked_scale is not None
                and (scale_threshold == 0 or adaptive_scale_frames >= scale_threshold)
            )

            use_locked_head = (
                auto_head_to_padding
                and head_active
                and locked_head_delta is not None
                and (
                    head_threshold == 0
                    or (
                        head_threshold is not None
                        and adaptive_head_frames >= head_threshold
                    )
                )
            )

            use_locked_feet = (
                auto_feet_to_padding
                and foot_active
                and locked_foot_delta is not None
                and (
                    foot_threshold == 0
                    or (
                        foot_threshold is not None
                        and adaptive_foot_frames >= foot_threshold
                    )
                )
            )

            scale_used, head_delta, foot_delta = self._auto_align_with_scale_lock(
                meta,
                head_padding,
                head_padding_normalized,
                auto_head_to_padding,
                head_active,
                head_padding_allow_overflow,
                center_horizontally,
                limit_to_canvas,
                0.0,
                False,
                locked_scale if use_locked_scale else None,
                locked_head_delta if use_locked_head else None,
                auto_feet_to_padding,
                foot_padding,
                foot_padding_normalized,
                foot_active,
                locked_foot_delta if use_locked_feet else None,
            )

            if not use_locked_scale:
                adaptive_scale_frames += 1
                if (
                    scale_used is not None
                    and np.isfinite(scale_used)
                    and scale_used > 0.0
                ):
                    locked_scale = float(scale_used)

            if auto_head_to_padding and head_active and not use_locked_head:
                adaptive_head_frames += 1
                if head_delta is not None and np.isfinite(head_delta):
                    locked_head_delta = float(head_delta)

            if auto_feet_to_padding and foot_active and not use_locked_feet:
                adaptive_foot_frames += 1
                if foot_delta is not None and np.isfinite(foot_delta):
                    locked_foot_delta = float(foot_delta)

            if auto_head_to_padding and head_active_threshold is not None and head_active:
                head_active_frames += 1

            if auto_feet_to_padding and foot_active_threshold is not None and foot_active:
                foot_active_frames += 1

        return (pose_data_copy,)


class PoseDataEditorCutter:
    SCORE_THRESHOLD = 0.05

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "images": ("IMAGE",),
                "padding_left": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.1,
                        "tooltip": "Extra space to keep on the left side of the cropped canvas (pixels unless normalised).",
                    },
                ),
                "padding_right": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.1,
                        "tooltip": "Extra space to keep on the right side of the cropped canvas (pixels unless normalised).",
                    },
                ),
                "padding_top": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.1,
                        "tooltip": "Extra space to keep above the pose in the cropped canvas (pixels unless normalised).",
                    },
                ),
                "padding_bottom": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.1,
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
                "keep_aspect_ratio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Expand the crop so it preserves the original canvas aspect ratio when possible.",
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
        images,
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_normalized,
        keep_aspect_ratio,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas", [])

        if not pose_metas:
            return (pose_data_copy, images)

        if isinstance(images, torch.Tensor):
            images_np = images.detach().cpu().numpy()
            images_device = images.device
            images_dtype = images.dtype
        else:
            images_np = np.asarray(images)
            images_device = None
            images_dtype = None
        if images_np.size == 0:
            return (pose_data_copy, images)

        reference_meta = pose_metas[0]
        width = getattr(reference_meta, "width", 0)
        height = getattr(reference_meta, "height", 0)

        if width in (None, 0) or height in (None, 0):
            return (pose_data_copy, images)

        crop_bounds = self._determine_crop_bounds(
            pose_metas,
            width,
            height,
            padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            padding_normalized,
            keep_aspect_ratio,
        )

        if crop_bounds is None:
            return (pose_data_copy, images)

        x0, y0, x1, y1 = crop_bounds

        if x1 <= x0 or y1 <= y0:
            return (pose_data_copy, images)

        cropped_np = images_np[:, y0:y1, x0:x1, ...]
        if cropped_np.size == 0:
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

        cropped_tensor = torch.from_numpy(cropped_np)
        if images_dtype is not None or images_device is not None:
            cropped_tensor = cropped_tensor.to(device=images_device or torch.device("cpu"))
            if images_dtype is not None:
                cropped_tensor = cropped_tensor.to(dtype=images_dtype)

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
        keep_aspect_ratio,
    ):
        largest_bbox = None
        largest_area = -1.0

        for meta in pose_metas:
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

        if largest_bbox is None:
            return None

        pad_left_px = self._resolve_padding(padding_left, padding_normalized, width)
        pad_right_px = self._resolve_padding(padding_right, padding_normalized, width)
        pad_top_px = self._resolve_padding(padding_top, padding_normalized, height)
        pad_bottom_px = self._resolve_padding(padding_bottom, padding_normalized, height)

        x0 = max(0.0, float(largest_bbox[0]) - pad_left_px)
        y0 = max(0.0, float(largest_bbox[1]) - pad_top_px)
        x1 = min(float(width), float(largest_bbox[2]) + pad_right_px)
        y1 = min(float(height), float(largest_bbox[3]) + pad_bottom_px)

        if keep_aspect_ratio:
            crop_width = x1 - x0
            crop_height = y1 - y0
            if crop_width > 0.0 and crop_height > 0.0 and width > 0 and height > 0:
                target_ratio = float(width) / float(height)
                if target_ratio > 0.0:
                    current_ratio = crop_width / crop_height
                    if current_ratio > target_ratio + 1e-6:
                        desired_height = crop_width / target_ratio
                        delta_height = desired_height - crop_height
                        if delta_height > 0.0:
                            expand_top = delta_height / 2.0
                            expand_bottom = delta_height - expand_top
                            y0 -= expand_top
                            y1 += expand_bottom
                            if y0 < 0.0:
                                y1 = min(float(height), y1 + (-y0))
                                y0 = 0.0
                            if y1 > float(height):
                                overflow = y1 - float(height)
                                y0 = max(0.0, y0 - overflow)
                                y1 = float(height)
                    elif current_ratio < target_ratio - 1e-6:
                        desired_width = crop_height * target_ratio
                        delta_width = desired_width - crop_width
                        if delta_width > 0.0:
                            expand_left = delta_width / 2.0
                            expand_right = delta_width - expand_left
                            x0 -= expand_left
                            x1 += expand_right
                            if x0 < 0.0:
                                x1 = min(float(width), x1 + (-x0))
                                x0 = 0.0
                            if x1 > float(width):
                                overflow = x1 - float(width)
                                x0 = max(0.0, x0 - overflow)
                                x1 = float(width)

        x0 = int(max(0.0, math.floor(x0)))
        y0 = int(max(0.0, math.floor(y0)))
        x1 = int(min(float(width), math.ceil(x1)))
        y1 = int(min(float(height), math.ceil(y1)))

        if x1 <= x0 or y1 <= y0:
            return None

        return (x0, y0, x1, y1)

    def _resolve_padding(self, value, normalized, size_reference):
        if normalized:
            return float(value) * float(size_reference)
        return float(value)

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

NODE_CLASS_MAPPINGS = {
    "DrawViTPose": DrawViTPose,
    "OnnxDetectionModelLoader": OnnxDetectionModelLoader,
    "PoseAndFaceDetection": PoseAndFaceDetection,
    "PoseDataEditor": PoseDataEditor,
    "PoseDataEditorCutter": PoseDataEditorCutter,
    "PoseDataEditorAutomatic": PoseDataEditorAutomatic,
    "PoseDataEditorAutoPositioning": PoseDataEditorAutoPositioning,
    "PoseDataEditorAutomaticPositioningAndStretching": PoseDataEditorAutomaticPositioningAndStretching,
    "PoseDataEditorAutomaticV2": PoseDataEditorAutomaticV2,
    "PoseDataEditorAutomaticV3": PoseDataEditorAutomaticV3,
    "PoseDataEditorAutomaticV4": PoseDataEditorAutomaticV4,
    "PoseDataEditorAutomaticV5": PoseDataEditorAutomaticV5,
    "PoseDataEditorAutomaticV6": PoseDataEditorAutomaticV6,
    "PoseDataEditorAutomaticV7": PoseDataEditorAutomaticV7,
    "PoseDataEditorAutomaticV8": PoseDataEditorAutomaticV8,
    "PoseDataEditorAutomaticOnlyTorsoHeadOffset": PoseDataEditorAutomaticOnlyTorsoHeadOffset,
    "PoseDataEditorAutomaticOnlyTorsoHeadOffsetV2": PoseDataEditorAutomaticOnlyTorsoHeadOffsetV2,
    "PoseDataPostProcessor": PoseDataPostProcessor,
    "PoseRetargetPromptHelper": PoseRetargetPromptHelper,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DrawViTPose": "Draw ViT Pose",
    "OnnxDetectionModelLoader": "ONNX Detection Model Loader",
    "PoseAndFaceDetection": "Pose and Face Detection",
    "PoseDataEditor": "Pose Data Editor",
    "PoseDataEditorCutter": "Pose Data Editor Cutter",
    "PoseDataEditorAutomatic": "Pose Data Editor Automatic",
    "PoseDataEditorAutoPositioning": "Pose Data Editor Auto-Positioning",
    "PoseDataEditorAutomaticPositioningAndStretching": "Pose Data Editor Automatic Positioning and Stretching",
    "PoseDataEditorAutomaticV2": "Pose Data Editor Automatic V2",
    "PoseDataEditorAutomaticV3": "Pose Data Editor Automatic V3",
    "PoseDataEditorAutomaticV4": "Pose Data Editor Automatic V4",
    "PoseDataEditorAutomaticV5": "Pose Data Editor Automatic V5",
    "PoseDataEditorAutomaticV6": "Pose Data Editor Automatic V6",
    "PoseDataEditorAutomaticV7": "Pose Data Editor Automatic V7",
    "PoseDataEditorAutomaticV8": "Pose Data Editor Automatic V8",
    "PoseDataEditorAutomaticOnlyTorsoHeadOffset": "Pose Data Editor Automatic Only Torso-to-Head Offset",
    "PoseDataEditorAutomaticOnlyTorsoHeadOffsetV2": "Pose Data Editor Automatic Only Torso-to-Head Offset V2",
    "PoseDataPostProcessor": "Pose Data Post-Processor",
    "PoseRetargetPromptHelper": "Pose Retarget Prompt Helper",
}
