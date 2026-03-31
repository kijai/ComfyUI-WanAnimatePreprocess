import cv2
import numpy as np
from PIL import Image
from .draw_utils import draw_bodypose, draw_bodypose_with_feet, draw_handpose_lr, draw_handpose, draw_facepose, draw_bodypose_augmentation


def draw_pose(pose, H, W, show_feet=False, show_body=True, show_hand=True, show_face=True, show_cheek=False, dw_bgr=False, dw_hand=False, aug_body_draw=False, optimized_face=False):
    final_canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for i in range(len(pose["bodies"]["candidate"])):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        bodies = pose["bodies"]
        faces = pose["faces"][i:i+1]
        hands = pose["hands"][2*i:2*i+2]
        candidate = bodies["candidate"][i]
        subset = bodies["subset"][i:i+1]

        if show_body:
            if len(subset[0]) <= 18 or show_feet == False:
                if aug_body_draw:
                    raise NotImplementedError("aug_body_draw is not implemented yet")
                else:
                    canvas = draw_bodypose(canvas, candidate, subset)
            else:
                canvas = draw_bodypose_with_feet(canvas, candidate, subset)
            if dw_bgr:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        if show_cheek:
            assert show_body == False, "show_cheek and show_body cannot be True at the same time"
            canvas = draw_bodypose_augmentation(canvas, candidate, subset,  drop_aug=True, shift_aug=False, all_cheek_aug=True)
        if show_hand:
            if not dw_hand:
                canvas = draw_handpose_lr(canvas, hands)
            else:
                canvas = draw_handpose(canvas, hands)
        if show_face:
            canvas = draw_facepose(canvas, faces, optimized_face=optimized_face)
        final_canvas = final_canvas + canvas
    return final_canvas


def scale_image_hw_keep_size(img, scale_h, scale_w):
    """Scale the image by scale_h and scale_w respectively, keeping the output size unchanged."""
    H, W = img.shape[:2]
    new_H, new_W = int(H * scale_h), int(W * scale_w)
    scaled = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    result = np.zeros_like(img)

    # 计算在目标图上的放置范围
    # --- Y方向 ---
    if new_H >= H:
        y_start_src = (new_H - H) // 2
        y_end_src = y_start_src + H
        y_start_dst = 0
        y_end_dst = H
    else:
        y_start_src = 0
        y_end_src = new_H
        y_start_dst = (H - new_H) // 2
        y_end_dst = y_start_dst + new_H

    # --- X方向 ---
    if new_W >= W:
        x_start_src = (new_W - W) // 2
        x_end_src = x_start_src + W
        x_start_dst = 0
        x_end_dst = W
    else:
        x_start_src = 0
        x_end_src = new_W
        x_start_dst = (W - new_W) // 2
        x_end_dst = x_start_dst + new_W

    # 将 scaled 映射到 result
    result[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = scaled[y_start_src:y_end_src, x_start_src:x_end_src]

    return result

def draw_pose_to_canvas_np(poses, pool, H, W, reshape_scale, show_feet_flag=False, show_body_flag=True, show_hand_flag=True, show_face_flag=True, show_cheek_flag=False, dw_bgr=False, dw_hand=False, aug_body_draw=False):
    canvas_np_lst = []
    for pose in poses:
        if reshape_scale > 0:
            pool.apply_random_reshapes(pose)
        canvas = draw_pose(pose, H, W, show_feet_flag, show_body_flag, show_hand_flag, show_face_flag, show_cheek_flag, dw_bgr, dw_hand, aug_body_draw, optimized_face=True)
        canvas_np_lst.append(canvas)
    return canvas_np_lst


def draw_pose_to_canvas(poses, pool, H, W, reshape_scale, points_only_flag, show_feet_flag, show_body_flag=True, show_hand_flag=True, show_face_flag=True, show_cheek_flag=False, dw_bgr=False, dw_hand=False, aug_body_draw=False):
    canvas_lst = []
    for pose in poses:
        if reshape_scale > 0:
            pool.apply_random_reshapes(pose)
        canvas = draw_pose(pose, H, W, show_feet_flag, show_body_flag, show_hand_flag, show_face_flag, show_cheek_flag, dw_bgr, dw_hand, aug_body_draw, optimized_face=False)
        canvas_img = Image.fromarray(canvas)
        canvas_lst.append(canvas_img)
    return canvas_lst


def project_dwpose_to_3d(dwpose_keypoint, original_threed_keypoint, focal, princpt, H, W):
    # Camera intrinsic parameters
    # fx, fy = focal, focal
    fx, fy = focal
    cx, cy = princpt

    # 2D keypoint coordinates
    x_2d, y_2d = dwpose_keypoint[0] * W, dwpose_keypoint[1] * H

    # Original 3D point (in camera coordinate system)
    ori_x, ori_y, ori_z = original_threed_keypoint

    # Use the new 2D point and original depth to compute the new 3D point by back-projection
    # Formula: x = (u - cx) * z / fx
    new_x = (x_2d - cx) * ori_z / fx
    new_y = (y_2d - cy) * ori_z / fy
    new_z = ori_z  # Keep the depth unchanged

    return [new_x, new_y, new_z]
