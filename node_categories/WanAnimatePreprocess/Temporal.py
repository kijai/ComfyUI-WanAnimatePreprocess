import copy
import numpy as np
import torch


# ============================================================
# Temporal Smoothing Nodes for NLF / DWPOSE
# ============================================================
# Recommended workflow:
#   NLF Extract -> NLFTemporalSmootherV1 -> Retargeter/Scaler -> Render/Mimic
#
# Optional for fallback hands/face:
#   DWPose fallback -> DWPoseTemporalSmootherV1 -> Render/Mimic
# ============================================================


def _temporal_alpha(value):
    # 0.00 = no smoothing
    # 0.95 = very strong smoothing
    return float(np.clip(value, 0.0, 0.95))


def _get_nlf_frames(nlf_poses):
    if isinstance(nlf_poses, dict) and "joints3d_nonparam" in nlf_poses:
        return nlf_poses["joints3d_nonparam"][0]
    return nlf_poses


def _nlf_frame_to_numpy(frame):
    if frame is None:
        return None, None

    if isinstance(frame, torch.Tensor):
        original_info = ("torch", frame.device, frame.dtype, frame.dim())
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)
        original_info = ("array_or_list", None, getattr(arr, "dtype", None), arr.ndim)

    if arr.size == 0:
        return None, original_info

    if arr.ndim == 2:
        arr = arr[None, ...]

    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None, original_info

    return arr.astype(np.float32, copy=True), original_info


def _numpy_to_nlf_frame(arr, original_frame, original_info):
    kind, device, dtype, original_dim = original_info

    out = arr[0] if original_dim == 2 else arr

    if kind == "torch":
        return torch.from_numpy(out).to(device=device, dtype=dtype)

    if isinstance(original_frame, np.ndarray):
        return out.astype(original_frame.dtype, copy=False)

    return out.tolist()


def _valid_xyz(arr):
    return np.linalg.norm(arr[..., :3], axis=-1) > 1e-6


def _joint_smooth_strength(
    joint_idx,
    body_smooth,
    arm_smooth,
    hand_smooth,
    foot_smooth,
    root_smooth,
):
    # SMPL-ish NLF indices from your current retargeter tree:
    #
    # 0 = pelvis/root
    #
    # left arm chain:
    # 13 -> 16 -> 18 -> 20 -> 22
    #
    # right arm chain:
    # 14 -> 17 -> 19 -> 21 -> 23
    #
    # foot / lower-leg anchors:
    # 7, 8, 10, 11

    if joint_idx == 0:
        return _temporal_alpha(root_smooth)

    # Shoulder / upper-arm / elbow-ish area
    if joint_idx in (13, 14, 16, 17, 18, 19):
        return _temporal_alpha(arm_smooth)

    # Wrist / hand-end / distal arm points
    if joint_idx in (20, 21, 22, 23):
        return _temporal_alpha(hand_smooth)

    # Feet / ankle / toe-ish anchors
    if joint_idx in (7, 8, 10, 11):
        return _temporal_alpha(foot_smooth)

    return _temporal_alpha(body_smooth)


def _smooth_nlf_frame_sequence(
    frames,
    body_smooth,
    arm_smooth,
    hand_smooth,
    foot_smooth,
    root_smooth,
    deadzone,
    max_jump,
    preserve_root_motion,
    smooth_z,
):
    previous = None
    previous_root = None

    smoothed_arrays = []
    original_frames = []
    original_infos = []

    changed_samples = 0
    jump_clamps = 0

    for frame_idx, frame in enumerate(frames):
        arr, original_info = _nlf_frame_to_numpy(frame)

        original_frames.append(frame)
        original_infos.append(original_info)

        if arr is None:
            smoothed_arrays.append(None)
            continue

        current = arr.copy()

        if previous is None or previous.shape != current.shape:
            previous = current.copy()
            previous_root = current[:, 0, :3].copy() if current.shape[1] > 0 else None
            smoothed_arrays.append(current)
            continue

        result = current.copy()

        valid = _valid_xyz(current)
        valid_previous = _valid_xyz(previous)
        both_valid = valid & valid_previous

        root_delta = np.zeros((current.shape[0], 1, 3), dtype=np.float32)

        if preserve_root_motion and current.shape[1] > 0 and previous_root is not None:
            root_now = current[:, 0:1, :3]
            root_prev = previous_root[:, None, :]

            if root_now.shape == root_prev.shape:
                root_delta = root_now - root_prev

        for joint_idx in range(current.shape[1]):
            alpha = _joint_smooth_strength(
                joint_idx,
                body_smooth,
                arm_smooth,
                hand_smooth,
                foot_smooth,
                root_smooth,
            )

            if alpha <= 1e-6:
                continue

            mask = both_valid[:, joint_idx]

            if not np.any(mask):
                continue

            prev_xyz = previous[:, joint_idx, :3].copy()
            cur_xyz = current[:, joint_idx, :3].copy()

            # Important:
            # Smooth joints relative to the moving pelvis/root.
            # This avoids laggy/rubbery walking or camera motion.
            if preserve_root_motion and joint_idx != 0:
                prev_xyz += root_delta[:, 0, :]

            delta = cur_xyz - prev_xyz

            if not smooth_z:
                delta[:, 2] = 0.0

            dist = np.linalg.norm(delta, axis=-1)
            target_xyz = cur_xyz.copy()

            if deadzone > 0.0:
                tiny_motion = dist < float(deadzone)
                if np.any(tiny_motion):
                    target_xyz[tiny_motion, :3] = prev_xyz[tiny_motion, :3]

            if max_jump > 0.0:
                too_far = dist > float(max_jump)
                if np.any(too_far):
                    factor = (float(max_jump) / np.maximum(dist[too_far], 1e-6))[:, None]
                    target_xyz[too_far, :3] = prev_xyz[too_far, :3] + delta[too_far, :3] * factor
                    jump_clamps += int(np.count_nonzero(too_far))

            smoothed_xyz = alpha * prev_xyz + (1.0 - alpha) * target_xyz

            if not smooth_z:
                smoothed_xyz[:, 2] = current[:, joint_idx, 2]

            result[mask, joint_idx, :3] = smoothed_xyz[mask]
            changed_samples += int(np.count_nonzero(mask))

        previous = result.copy()
        previous_root = result[:, 0, :3].copy() if result.shape[1] > 0 else None

        smoothed_arrays.append(result)

    return smoothed_arrays, original_frames, original_infos, changed_samples, jump_clamps


class NLFTemporalSmootherV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {
                    "tooltip": "NLF 3D data. Best placed directly after NLF extraction and before Retargeter/Scaler."
                }),
                "bypass": ("BOOLEAN", {"default": False}),

                "body_smooth": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                    "tooltip": "General body smoothing. Rest of the body. 0 = off, 0.95 = very strong."
                }),

                "arm_smooth": ("FLOAT", {
                    "default": 0.60,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                    "tooltip": "Smoothing for shoulders, upper arms and elbows. Good for arm jitter."
                }),

                "hand_smooth": ("FLOAT", {
                    "default": 0.78,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                    "tooltip": "Smoothing for wrist/hand-end/distal arm points. Higher values reduce hand jitter."
                }),

                "foot_smooth": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                    "tooltip": "Smoothing for foot anchors. Too high can cause foot sliding."
                }),

                "root_smooth": ("FLOAT", {
                    "default": 0.10,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                    "tooltip": "Smoothing for pelvis/root. Keep low so body translation stays responsive."
                }),

                "deadzone": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Tiny 3D motion under this value is treated as jitter. 0 disables."
                }),

                "max_jump": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Optional per-frame jump clamp in NLF units. 0 disables."
                }),

                "preserve_root_motion": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Smooths joints relative to pelvis/root motion to avoid laggy walking."
                }),

                "smooth_z": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also smooth depth/Z. Disable if depth lag causes issues."
                }),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING")
    RETURN_NAMES = ("nlf_poses_smoothed", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Timed"
    DESCRIPTION = "Temporal smoother for NLF 3D joints. Use before Retargeter/Scaler."

    def process(
        self,
        nlf_poses,
        bypass,
        body_smooth,
        arm_smooth,
        hand_smooth,
        foot_smooth,
        root_smooth,
        deadzone,
        max_jump,
        preserve_root_motion,
        smooth_z,
    ):
        if bypass:
            return (nlf_poses, "NLFTemporalSmootherV1: bypass active.")

        out = copy.deepcopy(nlf_poses)
        frames = _get_nlf_frames(out)

        if frames is None or len(frames) == 0:
            return (out, "NLFTemporalSmootherV1: no frames found.")

        smoothed_arrays, original_frames, original_infos, changed, clamped = _smooth_nlf_frame_sequence(
            frames=frames,
            body_smooth=body_smooth,
            arm_smooth=arm_smooth,
            hand_smooth=hand_smooth,
            foot_smooth=foot_smooth,
            root_smooth=root_smooth,
            deadzone=deadzone,
            max_jump=max_jump,
            preserve_root_motion=preserve_root_motion,
            smooth_z=smooth_z,
        )

        for i, arr in enumerate(smoothed_arrays):
            if arr is not None:
                frames[i] = _numpy_to_nlf_frame(arr, original_frames[i], original_infos[i])

        log = [
            "=== NLF TEMPORAL SMOOTHER V1 ===",
            f"frames={len(frames)}",
            f"changed_samples={changed}",
            f"jump_clamps={clamped}",
            f"body_smooth={body_smooth:.2f}",
            f"arm_smooth={arm_smooth:.2f}",
            f"hand_smooth={hand_smooth:.2f}",
            f"foot_smooth={foot_smooth:.2f}",
            f"root_smooth={root_smooth:.2f}",
            f"deadzone={deadzone:.4f}",
            f"max_jump={max_jump:.4f}",
            f"preserve_root_motion={preserve_root_motion}",
            f"smooth_z={smooth_z}",
            "",
            "Recommended workflow:",
            "NLF Extract -> NLFTemporalSmootherV1 -> Retargeter V22 -> Scaler/Render",
        ]

        return (out, "\n".join(log))


# ============================================================
# Optional 2D smoother for DWPOSE fallback hands/face/body
# ============================================================

def _smooth_2d_series(values, scores, smooth, deadzone, max_jump):
    if values is None:
        return values, 0, 0

    arr = np.asarray(values, dtype=np.float32).copy()

    if arr.ndim < 3 or arr.shape[-1] < 2:
        return arr, 0, 0

    score_arr = None
    if scores is not None:
        try:
            score_arr = np.asarray(scores, dtype=np.float32)
        except Exception:
            score_arr = None

    alpha = _temporal_alpha(smooth)

    if alpha <= 1e-6:
        return arr, 0, 0

    previous = arr[0].copy()
    changed = 0
    clamped = 0

    for frame_idx in range(1, arr.shape[0]):
        current = arr[frame_idx].copy()

        valid = np.linalg.norm(current[..., :2], axis=-1) > 1e-8
        valid_previous = np.linalg.norm(previous[..., :2], axis=-1) > 1e-8

        mask = valid & valid_previous

        if score_arr is not None:
            try:
                mask = mask & (score_arr[frame_idx] > 0.05)
            except Exception:
                pass

        if not np.any(mask):
            previous = current.copy()
            continue

        delta = current[..., :2] - previous[..., :2]
        dist = np.linalg.norm(delta, axis=-1)

        target = current[..., :2].copy()

        if deadzone > 0.0:
            tiny_motion = dist < float(deadzone)
            target[tiny_motion] = previous[..., :2][tiny_motion]

        if max_jump > 0.0:
            too_far = dist > float(max_jump)
            if np.any(too_far):
                factor = (float(max_jump) / np.maximum(dist[too_far], 1e-6))[:, None]
                target[too_far] = previous[..., :2][too_far] + delta[too_far] * factor
                clamped += int(np.count_nonzero(too_far))

        smoothed = alpha * previous[..., :2] + (1.0 - alpha) * target

        arr[frame_idx, ..., :2][mask] = smoothed[mask]
        previous = arr[frame_idx].copy()

        changed += int(np.count_nonzero(mask))

    return arr, changed, clamped


class DWPoseTemporalSmootherV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dw_poses": ("DWPOSES",),
                "bypass": ("BOOLEAN", {"default": False}),

                "body_smooth": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                }),

                "hand_smooth": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                }),

                "face_smooth": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                }),

                "deadzone": ("FLOAT", {
                    "default": 0.0015,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.0005,
                    "tooltip": "Normalized 2D units. 0.0015 is tiny jitter."
                }),

                "max_jump": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Normalized 2D units. 0 disables."
                }),
            }
        }

    RETURN_TYPES = ("DWPOSES", "STRING")
    RETURN_NAMES = ("dw_poses_smoothed", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Timed"
    DESCRIPTION = "Optional 2D temporal smoother for DWPOSE fallback hands/face/body before rendering."

    def process(
        self,
        dw_poses,
        bypass,
        body_smooth,
        hand_smooth,
        face_smooth,
        deadzone,
        max_jump,
    ):
        if bypass:
            return (dw_poses, "DWPoseTemporalSmootherV1: bypass active.")

        out = copy.deepcopy(dw_poses)
        poses = out.get("poses", []) if isinstance(out, dict) else []

        if not poses:
            return (out, "DWPoseTemporalSmootherV1: no poses found.")

        body_values = []
        body_scores = []

        hand_values = []
        hand_scores = []

        face_values = []
        face_scores = []

        for pose in poses:
            body_values.append(pose.get("bodies", {}).get("candidate"))
            body_scores.append(pose.get("body_score"))

            hand_values.append(pose.get("hands"))
            hand_scores.append(pose.get("hand_score"))

            face_values.append(pose.get("faces"))
            face_scores.append(pose.get("face_score"))

        changed_total = 0
        clamped_total = 0

        body_out, changed, clamped = _smooth_2d_series(
            body_values,
            body_scores,
            body_smooth,
            deadzone,
            max_jump,
        )
        changed_total += changed
        clamped_total += clamped

        hand_out, changed, clamped = _smooth_2d_series(
            hand_values,
            hand_scores,
            hand_smooth,
            deadzone,
            max_jump,
        )
        changed_total += changed
        clamped_total += clamped

        face_out, changed, clamped = _smooth_2d_series(
            face_values,
            face_scores,
            face_smooth,
            deadzone,
            max_jump,
        )
        changed_total += changed
        clamped_total += clamped

        for i, pose in enumerate(poses):
            if body_out is not None and "bodies" in pose:
                pose["bodies"]["candidate"] = body_out[i].astype(np.float32)

            if hand_out is not None:
                pose["hands"] = hand_out[i].astype(np.float32)

            if face_out is not None:
                pose["faces"] = face_out[i].astype(np.float32)

        log = [
            "=== DWPOSE TEMPORAL SMOOTHER V1 ===",
            f"frames={len(poses)}",
            f"changed_samples={changed_total}",
            f"jump_clamps={clamped_total}",
            f"body_smooth={body_smooth:.2f}",
            f"hand_smooth={hand_smooth:.2f}",
            f"face_smooth={face_smooth:.2f}",
            f"deadzone={deadzone:.4f}",
            f"max_jump={max_jump:.4f}",
            "",
            "Recommended workflow:",
            "DWPose fallback -> DWPoseTemporalSmootherV1 -> RenderNLF/Mimic",
        ]

        return (out, "\n".join(log))
