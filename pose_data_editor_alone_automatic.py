"""Standalone pose data editor for adaptive upper body offsetting.

This module implements the :class:`PoseDataEditorAloneAutomaticChaty` component
which adjusts pose keypoints so that the upper body maintains configurable
padding distances to the canvas.  The implementation is completely
independent from the rest of the repository and only relies on the Python
standard library.

The expected input matches the structure used by the Johnson style pose
format: a sequence of frames where each frame is represented by a mapping
from a person identifier to a list of keypoints.  Each keypoint is a
``(y, x, score)`` triple.  Only the vertical component ``y`` is modified by
this component.

Example frame structure::

    {
        "0": [
            [121.19, 458.15, 0.99],
            [110.02, 469.43, 0.98],
            ...
        ]
    }

The component keeps internal state while processing frames sequentially.
It smoothly adjusts the vertical offset/scale during the configured
``duration`` and then locks the factors so that the pose can move freely.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


Keypoint = Tuple[float, float, float]
FrameKeypoints = Mapping[str, Iterable[Optional[Keypoint]]]
MutableFrameKeypoints = MutableMapping[str, List[Optional[Keypoint]]]
SequenceKeypoints = Iterable[FrameKeypoints]


@dataclass(frozen=True)
class BodyRegions:
    """Grouped index definitions for the COCO-25 skeleton.

    The indices mirror the mapping that is declared in ``Visualization.py``
    and provide a stable reference for the transformation logic.  Only the
    indices that are relevant for vertical adjustments are included.
    """

    head: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    upper_body: Tuple[int, ...] = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
    )
    hips: Tuple[int, ...] = (12, 13, 14)
    legs: Tuple[int, ...] = (15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    feet: Tuple[int, ...] = (17, 18, 19, 20, 21, 22, 23, 24)


class PoseDataEditorAloneAutomaticChaty:
    """Adaptive pose editor that offsets the upper body.

    Parameters
    ----------
    duration:
        Total duration of the adaptive phase (seconds).  After the duration
        expires the last computed scale/offset are locked and reused.
    head_padding:
        Distance between the highest keypoint and the top canvas edge.  When
        ``normalize`` is enabled this value is interpreted as a ratio of the
        canvas height.
    foot_padding:
        Distance between the lowest keypoint and the bottom canvas edge.
    normalize:
        Toggle whether padding values are treated as normalized ratios.
    head_padding_active_seconds / foot_padding_active_seconds:
        Independent durations that determine for how long each padding is
        enforced.
    lock_feet:
        If ``True`` the feet remain unchanged by the editor.
    scale_legs:
        When ``True`` the leg keypoints receive the same transform as the
        upper body, ensuring that the hip-foot distance stays consistent.
    fps:
        Frame rate that is used to convert durations to frame counts.
    """

    _regions = BodyRegions()

    def __init__(
        self,
        *,
        duration: float = 1.0,
        head_padding: float = 0.0,
        foot_padding: float = 0.0,
        normalize: bool = False,
        head_padding_active_seconds: float = 1.0,
        foot_padding_active_seconds: float = 1.0,
        lock_feet: bool = False,
        scale_legs: bool = False,
        fps: int = 30,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        self.duration = max(0.0, float(duration))
        self.head_padding = float(head_padding)
        self.foot_padding = float(foot_padding)
        self.normalize = bool(normalize)
        self.head_padding_active_seconds = max(0.0, float(head_padding_active_seconds))
        self.foot_padding_active_seconds = max(0.0, float(foot_padding_active_seconds))
        self.lock_feet = bool(lock_feet)
        self.scale_legs = bool(scale_legs)
        self.fps = int(fps)

        self.duration_frames = max(0, int(round(self.duration * self.fps)))
        self.head_padding_frames = int(round(self.head_padding_active_seconds * self.fps))
        self.foot_padding_frames = int(round(self.foot_padding_active_seconds * self.fps))

        self._current_scale = 1.0
        self._current_offset = 0.0
        self._locked = self.duration_frames == 0
        self._frame_index = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the adaptive state so that a new sequence can be processed."""

        self._current_scale = 1.0
        self._current_offset = 0.0
        self._locked = self.duration_frames == 0
        self._frame_index = 0

    def process_sequence(
        self,
        frames: SequenceKeypoints,
        *,
        canvas_height: float,
        canvas_width: Optional[float] = None,
    ) -> List[MutableFrameKeypoints]:
        """Process an iterable of frames and return the transformed sequence."""

        result: List[MutableFrameKeypoints] = []
        for frame in frames:
            processed = self.process_frame(frame, canvas_height=canvas_height, canvas_width=canvas_width)
            result.append(processed)
        return result

    def process_frame(
        self,
        frame: FrameKeypoints,
        *,
        canvas_height: float,
        canvas_width: Optional[float] = None,
    ) -> MutableFrameKeypoints:
        """Process a single frame of pose keypoints.

        The ``canvas_width`` parameter is currently unused but is accepted so
        that the signature remains future proof.
        """

        del canvas_width  # width is currently not required but kept for API symmetry

        mutable_frame: MutableFrameKeypoints = {}
        for track_id, points in frame.items():
            mutable_points: List[Optional[Keypoint]] = []
            for point in points:
                if point is None:
                    mutable_points.append(None)
                else:
                    mutable_points.append(tuple(point))
            mutable_frame[track_id] = mutable_points

        canvas_height = float(canvas_height)
        head_padding_value = self._padding_to_pixels(self.head_padding, canvas_height) if self.head_padding_frames else 0.0
        foot_padding_value = self._padding_to_pixels(self.foot_padding, canvas_height) if self.foot_padding_frames else 0.0

        stats = self._collect_statistics(mutable_frame)

        target_scale, target_offset = self._compute_target_transform(
            stats=stats,
            canvas_height=canvas_height,
            head_padding=head_padding_value,
            foot_padding=foot_padding_value,
        )

        if not self._locked:
            remaining = max(1, self.duration_frames - self._frame_index)
            alpha = 1.0 / remaining
            self._current_scale += (target_scale - self._current_scale) * alpha
            self._current_offset += (target_offset - self._current_offset) * alpha

        for track_id, points in mutable_frame.items():
            pivot = stats.pivots.get(track_id, stats.fallback_pivot)
            if pivot is None:
                continue

            for idx, kp in enumerate(points):
                if not kp or len(kp) < 2:
                    continue

                y, x, *rest = kp

                if idx in self._regions.upper_body or (self.scale_legs and idx in self._regions.legs):
                    new_y = pivot + (y - pivot) * self._current_scale + self._current_offset
                else:
                    new_y = y

                if self.lock_feet and idx in self._regions.feet:
                    new_y = y

                points[idx] = (float(new_y), float(x), *rest)

        self._frame_index += 1
        if not self._locked and self._frame_index >= self.duration_frames:
            self._locked = True

        return mutable_frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @dataclass
    class _FrameStats:
        head_min: Optional[float]
        foot_max: Optional[float]
        pivots: Dict[str, Optional[float]]
        fallback_pivot: Optional[float]

    def _padding_to_pixels(self, padding: float, canvas_height: float) -> float:
        return padding * canvas_height if self.normalize else padding

    def _collect_statistics(self, frame: MutableFrameKeypoints) -> "PoseDataEditorAloneAutomaticChaty._FrameStats":
        head_min: Optional[float] = None
        foot_max: Optional[float] = None
        pivots: Dict[str, Optional[float]] = {}
        fallback_values: List[float] = []

        for track_id, keypoints in frame.items():
            head_positions: List[float] = []
            hip_positions: List[float] = []
            foot_positions: List[float] = []
            all_positions: List[float] = []

            for index, kp in enumerate(keypoints):
                if not kp or len(kp) < 2:
                    continue

                y = kp[0]
                score = kp[2] if len(kp) > 2 else 1.0
                if score <= 0:
                    continue

                all_positions.append(y)

                if index in self._regions.head:
                    head_positions.append(y)
                if index in self._regions.hips:
                    hip_positions.append(y)
                if index in self._regions.feet:
                    foot_positions.append(y)

            track_head = min(head_positions) if head_positions else None
            track_foot = max(foot_positions) if foot_positions else None
            track_pivot = (sum(hip_positions) / len(hip_positions)) if hip_positions else None

            if track_pivot is None and all_positions:
                track_pivot = sum(all_positions) / len(all_positions)

            if track_head is not None:
                head_min = track_head if head_min is None else min(head_min, track_head)
            if track_foot is not None:
                foot_max = track_foot if foot_max is None else max(foot_max, track_foot)

            if track_pivot is not None:
                fallback_values.append(track_pivot)

            pivots[track_id] = track_pivot

        fallback_pivot = sum(fallback_values) / len(fallback_values) if fallback_values else None
        return self._FrameStats(head_min=head_min, foot_max=foot_max, pivots=pivots, fallback_pivot=fallback_pivot)

    def _compute_target_transform(
        self,
        *,
        stats: "PoseDataEditorAloneAutomaticChaty._FrameStats",
        canvas_height: float,
        head_padding: float,
        foot_padding: float,
    ) -> Tuple[float, float]:
        head_active = self._frame_index < self.head_padding_frames and stats.head_min is not None
        foot_active = (
            self._frame_index < self.foot_padding_frames
            and stats.foot_max is not None
            and self.scale_legs
            and not self.lock_feet
        )

        head_target: Optional[float] = None
        foot_target: Optional[float] = None

        if head_active:
            head_target = head_padding
        if foot_active:
            foot_target = canvas_height - foot_padding

        pivot = stats.fallback_pivot or 0.0

        if head_target is not None and foot_target is not None and stats.foot_max is not None:
            source_head = stats.head_min if stats.head_min is not None else pivot
            source_foot = stats.foot_max
            denom = source_head - source_foot
            if abs(denom) < 1e-6:
                denom = -1e-6 if denom < 0 else 1e-6

            scale = (head_target - foot_target) / denom
            offset = head_target - pivot - (source_head - pivot) * scale
            return scale, offset

        if head_target is not None and stats.head_min is not None:
            scale = 1.0
            offset = head_target - stats.head_min
            return scale, offset

        if foot_target is not None and stats.foot_max is not None:
            scale = 1.0
            offset = foot_target - stats.foot_max
            return scale, offset

        return 1.0, 0.0


class PoseDataEditorAloneAutomaticChatyNode:
    """ComfyUI compatible wrapper around :class:`PoseDataEditorAloneAutomaticChaty`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "duration": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.01,
                        "tooltip": "Total duration in seconds before the adaptive offset/scale locks.",
                    },
                ),
                "head_padding": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.01,
                        "tooltip": "Distance to keep between the top-most keypoint and the canvas edge.",
                    },
                ),
                "foot_padding": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2048.0,
                        "step": 0.01,
                        "tooltip": "Distance to keep between the feet and the canvas floor when scaling legs.",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Interpret padding values as a fraction of the canvas height.",
                    },
                ),
                "head_padding_active_seconds": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.01,
                        "tooltip": "How long to enforce the head padding before releasing it.",
                    },
                ),
                "foot_padding_active_seconds": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.01,
                        "tooltip": "How long to enforce the foot padding before releasing it.",
                    },
                ),
                "lock_feet": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Prevent any adjustments to foot keypoints even when scaling the body.",
                    },
                ),
                "scale_legs": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Apply the adaptive transform to the legs so hip-to-foot distances remain constant.",
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 240,
                        "step": 1,
                        "tooltip": "Frame rate used to convert time based parameters to frame counts.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Adaptive upper-body offset that keeps configurable head and foot padding while processing pose sequences."

    def process(
        self,
        pose_data,
        duration,
        head_padding,
        foot_padding,
        normalize,
        head_padding_active_seconds,
        foot_padding_active_seconds,
        lock_feet,
        scale_legs,
        fps,
    ):
        pose_data_copy = copy.deepcopy(pose_data)
        pose_metas = pose_data_copy.get("pose_metas")

        if not pose_metas:
            return (pose_data_copy,)

        editor = PoseDataEditorAloneAutomaticChaty(
            duration=float(duration),
            head_padding=float(head_padding),
            foot_padding=float(foot_padding),
            normalize=bool(normalize),
            head_padding_active_seconds=float(head_padding_active_seconds),
            foot_padding_active_seconds=float(foot_padding_active_seconds),
            lock_feet=bool(lock_feet),
            scale_legs=bool(scale_legs),
            fps=int(fps),
        )

        editor.reset()

        for meta in pose_metas:
            canvas_height = self._to_float(getattr(meta, "height", None))
            canvas_width = self._to_float(getattr(meta, "width", None))

            if not math.isfinite(canvas_height) or canvas_height <= 0:
                continue

            canvas_width_value: Optional[float]
            if math.isfinite(canvas_width) and canvas_width > 0:
                canvas_width_value = canvas_width
            else:
                canvas_width_value = None

            frame = self._meta_to_frame(meta)
            if not frame:
                editor.process_frame({}, canvas_height=canvas_height, canvas_width=canvas_width_value)
                continue

            processed = editor.process_frame(frame, canvas_height=canvas_height, canvas_width=canvas_width_value)
            self._apply_processed_frame(meta, processed)

        pose_data_copy["pose_metas"] = pose_metas

        return (pose_data_copy,)

    @staticmethod
    def _to_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    @classmethod
    def _meta_to_frame(cls, meta) -> MutableFrameKeypoints:
        body = getattr(meta, "kps_body", None)
        confidences = getattr(meta, "kps_body_p", None)

        if body is None:
            return {}

        points: List[Optional[Keypoint]] = []
        for index in range(len(body)):
            coords = body[index]
            score = None
            if confidences is not None and index < len(confidences):
                score = cls._to_float(confidences[index])

            if coords is None or len(coords) < 2:
                points.append(None)
                continue

            x_value = cls._to_float(coords[0])
            y_value = cls._to_float(coords[1])

            if not math.isfinite(x_value) or not math.isfinite(y_value):
                points.append(None)
                continue

            confidence = score if score is not None and math.isfinite(score) else 1.0
            points.append((y_value, x_value, confidence))

        return {"0": points}

    @classmethod
    def _apply_processed_frame(cls, meta, processed: MutableFrameKeypoints) -> None:
        body = getattr(meta, "kps_body", None)
        if body is None:
            return

        points = processed.get("0")
        if not points:
            return

        limit = min(len(body), len(points))
        for index in range(limit):
            kp = points[index]
            if not kp or len(kp) < 1:
                continue

            new_y = cls._to_float(kp[0])
            if not math.isfinite(new_y):
                continue

            try:
                if isinstance(body[index], list):
                    body[index][1] = new_y
                elif isinstance(body[index], tuple):
                    mutable = list(body[index])
                    mutable[1] = new_y
                    body[index] = mutable
                else:
                    body[index][1] = new_y
            except Exception:
                # Fallback for unexpected container types.
                continue


__all__ = ["PoseDataEditorAloneAutomaticChaty", "PoseDataEditorAloneAutomaticChatyNode"]

