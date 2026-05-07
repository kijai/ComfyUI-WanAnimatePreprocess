import copy
import numpy as np
import torch


# ============================================================
# WanAnimatePreprocess Temporal Smoothing Nodes
# ============================================================
# Best workflow:
#   NLF Extract -> NLFTemporalSmootherV1 -> Retargeter/Scaler -> Render/Mimic
# Optional:
#   DWPose fallback -> DWPoseTemporalSmootherV1 -> Render/Mimic
#
# Why 3D first?
#   Retargeter/Scaler work on the NLF 3D joints. If jitter is already inside
#   the 3D stream, a later 2D smoother can only hide it visually. It cannot
#   prevent the retargeter from reacting to shaky measurements.
# ============================================================


def _copy_nlf(nlf_poses):
    return copy.deepcopy(nlf_poses)


def _get_nlf_frames(nlf_poses):
    if isinstance(nlf_poses, dict) and "joints3d_nonparam" in nlf_poses:
        return nlf_poses["joints3d_nonparam"][0]
    return nlf_poses


def _frame_to_np(frame):
    if frame is None:
        return None, None

    if isinstance(frame, torch.Tensor):
        original = ("torch", frame.device, frame.dtype, frame.dim())
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)
        original = ("array_or_list", None, arr.dtype if hasattr(arr, "dtype") else None, arr.ndim)

    if arr.size == 0:
        return None, original

    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None, original

    return arr.astype(np.float32, copy=True), original


def _np_to_frame(arr, original_frame, original):
    kind, device, dtype, original_dim = original
    out = arr[0] if original_dim == 2 else arr

    if kind == "torch":
        return torch.from_numpy(out).to(device=device, dtype=dtype)
    if isinstance(original_frame, np.ndarray):
        return out.astype(original_frame.dtype, copy=False)
    return out.tolist()


def _valid_xyz(arr):
    return np.linalg.norm(arr[..., :3], axis=-1) > 1e-6


def _alpha(v):
    # 0.00 = off / no smoothing, 0.95 = very strong smoothing
    return float(np.clip(v, 0.0, 0.95))


def _nlf_group_strength(joint_idx, body_smooth, hand_smooth, foot_smooth, root_smooth):
    # SMPL-ish indices used by the existing retargeters in this repo.
    # 0 = pelvis/root
    # 7/8/10/11 = ankle/toe-ish foot anchors
    # 18-23 = lower arm/hand chain in the current tree
    if joint_idx == 0:
        return _alpha(root_smooth)
    if joint_idx in (18, 19, 20, 21, 22, 23):
        return _alpha(hand_smooth)
    if joint_idx in (7, 8, 10, 11):
        return _alpha(foot_smooth)
    return _alpha(body_smooth)


def _smooth_nlf_frames(
    frames,
    body_smooth,
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
    output = []
    originals = []
    metas = []
    changed = 0
    clamped = 0

    for frame_idx, frame in enumerate(frames):
        arr, meta = _frame_to_np(frame)
        originals.append(frame)
        metas.append(meta)

        if arr is None:
            output.append(None)
            continue

        current = arr.copy()
        if previous is None or previous.shape != current.shape:
            previous = current.copy()
            previous_root = current[:, 0, :3].copy() if current.shape[1] > 0 else None
            output.append(current)
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
            a = _nlf_group_strength(joint_idx, body_smooth, hand_smooth, foot_smooth, root_smooth)
            if a <= 1e-6:
                continue

            mask = both_valid[:, joint_idx]
            if not np.any(mask):
                continue

            prev_xyz = previous[:, joint_idx, :3].copy()
            cur_xyz = current[:, joint_idx, :3].copy()

            # Important trick: smooth joints relative to the moving pelvis/root.
            # This avoids the classic temporal-smoothing problem where walking
            # or fast camera movement feels delayed and rubbery.
            if preserve_root_motion and joint_idx != 0:
                prev_xyz += root_delta[:, 0, :]

            delta = cur_xyz - prev_xyz
            if not smooth_z:
                delta[:, 2] = 0.0

            dist = np.linalg.norm(delta, axis=-1)
            target_xyz = cur_xyz.copy()

            if deadzone > 0.0:
                tiny = dist < float(deadzone)
                if np.any(tiny):
                    target_xyz[tiny, :3] = prev_xyz[tiny, :3]

            if max_jump > 0.0:
                too_far = dist > float(max_jump)
                if np.any(too_far):
                    factor = (float(max_jump) / np.maximum(dist[too_far], 1e-6))[:, None]
                    target_xyz[too_far, :3] = prev_xyz[too_far, :3] + delta[too_far, :3] * factor
                    clamped += int(np.count_nonzero(too_far))

            smoothed = a * prev_xyz + (1.0 - a) * target_xyz
            if not smooth_z:
                smoothed[:, 2] = current[:, joint_idx, 2]

            result[mask, joint_idx, :3] = smoothed[mask]
            changed += int(np.count_nonzero(mask))

        previous = result.copy()
        previous_root = result[:, 0, :3].copy() if result.shape[1] > 0 else None
        output.append(result)

    return output, originals, metas, changed, clamped


class NLFTemporalSmootherV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLFPRED", {"tooltip": "NLF 3D data. Best placed before Retargeter/Scaler."}),
                "bypass": ("BOOLEAN", {"default": False}),
                "body_smooth": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.95, "step": 0.01}),
                "hand_smooth": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 0.95, "step": 0.01}),
                "foot_smooth": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 0.95, "step": 0.01}),
                "root_smooth": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.95, "step": 0.01}),
                "deadzone": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Tiny 3D motion under this value is treated as jitter."}),
                "max_jump": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1, "tooltip": "Optional per-frame clamp in NLF units. 0 disables."}),
                "preserve_root_motion": ("BOOLEAN", {"default": True}),
                "smooth_z": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "STRING")
    RETURN_NAMES = ("nlf_poses_smoothed", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Timed"
    DESCRIPTION = "Temporal smoother for NLF 3D joints. Use before NLF retargeting/scaling."

    def process(
        self,
        nlf_poses,
        bypass,
        body_smooth,
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

        out = _copy_nlf(nlf_poses)
        frames = _get_nlf_frames(out)
        if frames is None or len(frames) == 0:
            return (out, "NLFTemporalSmootherV1: no frames found.")

        smoothed, originals, metas, changed, clamped = _smooth_nlf_frames(
            frames,
            body_smooth=body_smooth,
            hand_smooth=hand_smooth,
            foot_smooth=foot_smooth,
            root_smooth=root_smooth,
            deadzone=deadzone,
            max_jump=max_jump,
            preserve_root_motion=preserve_root_motion,
            smooth_z=smooth_z,
        )

        for i, arr in enumerate(smoothed):
            if arr is not None:
                frames[i] = _np_to_frame(arr, originals[i], metas[i])

        log = [
            "=== NLF TEMPORAL SMOOTHER V1 ===",
            f"frames={len(frames)}",
            f"changed_samples={changed}",
            f"jump_clamps={clamped}",
            f"body={body_smooth:.2f} hands={hand_smooth:.2f} feet={foot_smooth:.2f} root={root_smooth:.2f}",
            f"deadzone={deadzone:.4f} max_jump={max_jump:.4f}",
            f"preserve_root_motion={preserve_root_motion} smooth_z={smooth_z}",
            "Recommended: NLF Extract -> NLFTemporalSmootherV1 -> Retargeter V22 -> Scaler/Render.",
        ]
        return (out, "\n".join(log))


# ------------------------------------------------------------
# Optional 2D fallback smoother for DWPOSE hands/face/body
# ------------------------------------------------------------


def _smooth_2d_series(arr, score_arr, smooth, deadzone, max_jump):
    if arr is None:
        return arr, 0, 0

    values = np.asarray(arr, dtype=np.float32).copy()
    if values.ndim < 3 or values.shape[-1] < 2:
        return arr, 0, 0

    scores = None
    if score_arr is not None:
        try:
            scores = np.asarray(score_arr, dtype=np.float32)
        except Exception:
            scores = None

    a = _alpha(smooth)
    if a <= 1e-6:
        return values, 0, 0

    previous = values[0].copy()
    changed = 0
    clamped = 0

    for frame_idx in range(1, values.shape[0]):
        current = values[frame_idx].copy()
        valid = np.linalg.norm(current[..., :2], axis=-1) > 1e-8
        valid_prev = np.linalg.norm(previous[..., :2], axis=-1) > 1e-8
        mask = valid & valid_prev

        if scores is not None:
            try:
                mask = mask & (scores[frame_idx] > 0.05)
            except Exception:
                pass

        if not np.any(mask):
            previous = current.copy()
            continue

        delta = current[..., :2] - previous[..., :2]
        dist = np.linalg.norm(delta, axis=-1)
        target = current[..., :2].copy()

        if deadzone > 0.0:
            tiny = dist < float(deadzone)
            target[tiny] = previous[..., :2][tiny]

        if max_jump > 0.0:
            too_far = dist > float(max_jump)
            if np.any(too_far):
                factor = (float(max_jump) / np.maximum(dist[too_far], 1e-6))[:, None]
                target[too_far] = previous[..., :2][too_far] + delta[too_far] * factor
                clamped += int(np.count_nonzero(too_far))

        smoothed = a * previous[..., :2] + (1.0 - a) * target
        values[frame_idx, ..., :2][mask] = smoothed[mask]
        previous = values[frame_idx].copy()
        changed += int(np.count_nonzero(mask))

    return values, changed, clamped


class DWPoseTemporalSmootherV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dw_poses": ("DWPOSES",),
                "bypass": ("BOOLEAN", {"default": False}),
                "body_smooth": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.95, "step": 0.01}),
                "hand_smooth": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 0.95, "step": 0.01}),
                "face_smooth": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 0.95, "step": 0.01}),
                "deadzone": ("FLOAT", {"default": 0.0015, "min": 0.0, "max": 0.1, "step": 0.0005, "tooltip": "Normalized 2D units."}),
                "max_jump": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Normalized 2D units. 0 disables."}),
            }
        }

    RETURN_TYPES = ("DWPOSES", "STRING")
    RETURN_NAMES = ("dw_poses_smoothed", "log_output")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess/Timed"
    DESCRIPTION = "Optional 2D temporal smoother for DWPOSE fallback hands/face/body before rendering."

    def process(self, dw_poses, bypass, body_smooth, hand_smooth, face_smooth, deadzone, max_jump):
        if bypass:
            return (dw_poses, "DWPoseTemporalSmootherV1: bypass active.")

        out = copy.deepcopy(dw_poses)
        poses = out.get("poses", []) if isinstance(out, dict) else []
        if not poses:
            return (out, "DWPoseTemporalSmootherV1: no poses found.")

        body = []
        body_scores = []
        hands = []
        hand_scores = []
        faces = []
        face_scores = []

        for pose in poses:
            body.append(pose.get("bodies", {}).get("candidate"))
            body_scores.append(pose.get("body_score"))
            hands.append(pose.get("hands"))
            hand_scores.append(pose.get("hand_score"))
            faces.append(pose.get("faces"))
            face_scores.append(pose.get("face_score"))

        changed_total = 0
        clamped_total = 0

        body_out, changed, clamped = _smooth_2d_series(np.asarray(body), np.asarray(body_scores), body_smooth, deadzone, max_jump)
        changed_total += changed
        clamped_total += clamped

        hands_out, changed, clamped = _smooth_2d_series(np.asarray(hands), np.asarray(hand_scores), hand_smooth, deadzone, max_jump)
        changed_total += changed
        clamped_total += clamped

        faces_out, changed, clamped = _smooth_2d_series(np.asarray(faces), np.asarray(face_scores), face_smooth, deadzone, max_jump)
        changed_total += changed
        clamped_total += clamped

        for i, pose in enumerate(poses):
            if body_out is not None and "bodies" in pose:
                pose["bodies"]["candidate"] = body_out[i].astype(np.float32)
            if hands_out is not None:
                pose["hands"] = hands_out[i].astype(np.float32)
            if faces_out is not None:
                pose["faces"] = faces_out[i].astype(np.float32)

        log = [
            "=== DWPOSE TEMPORAL SMOOTHER V1 ===",
            f"frames={len(poses)}",
            f"changed_samples={changed_total}",
            f"jump_clamps={clamped_total}",
            f"body={body_smooth:.2f} hands={hand_smooth:.2f} face={face_smooth:.2f}",
            f"deadzone={deadzone:.4f} max_jump={max_jump:.4f}",
            "Recommended: DWPose fallback -> DWPoseTemporalSmootherV1 -> RenderNLF/Mimic.",
        ]
        return (out, "\n".join(log))


NODE_CLASS_MAPPINGS = {
    "NLFTemporalSmootherV1": NLFTemporalSmootherV1,
    "DWPoseTemporalSmootherV1": DWPoseTemporalSmootherV1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NLFTemporalSmootherV1": "NLF Temporal Smoother V1",
    "DWPoseTemporalSmootherV1": "DWPose Temporal Smoother V1",
}
