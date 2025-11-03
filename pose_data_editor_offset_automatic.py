"""PoseDataEditor.Offset.Automatic standalone component.

This module exposes the :class:`PoseDataEditorOffsetAutomatic` class and a
``PoseDataEditor`` namespace whose ``Offset.Automatic`` attribute points to the
class so the API matches the requested dotted name. The component consumes pose
structures shaped like :mod:`json_format` in this repository and offsets the
upper body while optionally stretching the legs to keep them attached to the
feet. Head and foot paddings can be controlled independently for a limited
number of seconds, and once the overall duration is reached the current offsets
and leg scales are locked for the remaining frames.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple


@dataclass
class _AdaptiveState:
    """Tracks adaptive offsets and locked values for a single person."""

    frames: int = 0
    duration_frames: Optional[int] = None
    head_limit: Optional[int] = None
    foot_limit: Optional[int] = None
    head_counter: int = 0
    foot_counter: int = 0
    current_upper_offset: float = 0.0
    current_foot_offset: float = 0.0
    locked_upper_offset: Optional[float] = None
    locked_foot_offset: Optional[float] = None
    locked_leg_scales: Optional[Dict[str, float]] = None
    current_leg_scales: Dict[str, float] = field(
        default_factory=lambda: {"left": 1.0, "right": 1.0}
    )


class PoseDataEditorOffsetAutomatic:
    """Offsets the torso and optionally stretches legs across pose frames.

    Parameters mirror the user specification and operate on pose dictionaries
    that follow the ``format_json`` layout (frames of persons, each with a list
    of ``[y, x, score]`` keypoints). The editor keeps head and foot paddings
    active for their respective durations, adapts offsets for the given number
    of seconds, and then locks the captured values.
    """

    HEAD_INDICES = (0, 1, 2, 3, 4, 5)
    TORSO_INDICES = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    HIP_INDICES = (12, 13, 14)
    KNEE_INDICES = (15, 16)
    ANKLE_INDICES = (17, 18)
    FOOT_EXTRA_INDICES = (19, 20, 21, 22, 23, 24)

    def __init__(
        self,
        duration_seconds: float,
        head_padding: float,
        foot_padding: float,
        normalize: bool,
        head_padding_active_seconds: float,
        foot_padding_active_seconds: float,
        lock_feet: bool,
        scale_legs: bool,
        fps: float,
    ) -> None:
        self.duration_seconds = float(duration_seconds)
        self.head_padding = float(head_padding)
        self.foot_padding = float(foot_padding)
        self.normalize = bool(normalize)
        self.head_padding_active_seconds = float(head_padding_active_seconds)
        self.foot_padding_active_seconds = float(foot_padding_active_seconds)
        self.lock_feet = bool(lock_feet)
        self.scale_legs = bool(scale_legs)
        self.fps = int(round(float(fps))) if float(fps) > 0 else 1
        self._states: Dict[str, _AdaptiveState] = {}

    def process(self, pose_data: MutableMapping) -> MutableMapping:
        """Return a modified deep copy of ``pose_data`` with offsets applied."""

        if not isinstance(pose_data, MutableMapping):
            raise TypeError("pose_data must be a dictionary-like object")

        width, height = self._extract_canvas_size(pose_data)
        if width is None or height is None:
            raise ValueError("pose_data must include canvas width and height")

        width = float(width)
        height = float(height)

        cloned = copy.deepcopy(pose_data)
        frames = cloned.get("keypoints")
        if not isinstance(frames, list):
            return cloned

        # Reset state for a new processing pass.
        self._states = {}

        for frame in frames:
            if not isinstance(frame, dict):
                continue

            for person_id, keypoints in frame.items():
                if not isinstance(keypoints, list):
                    continue

                state = self._states.get(person_id)
                if state is None:
                    state = self._create_state()
                    self._states[person_id] = state

                self._process_person(keypoints, state, width, height)

        return cloned

    # ------------------------------------------------------------------
    # State management helpers
    # ------------------------------------------------------------------
    def _create_state(self) -> _AdaptiveState:
        duration_frames = self._seconds_to_frames(self.duration_seconds)
        if duration_frames <= 0:
            duration_frames = None

        head_frames = self._seconds_to_frames(self.head_padding_active_seconds)
        if self.head_padding_active_seconds < 0.0:
            head_frames = None

        foot_frames = self._seconds_to_frames(self.foot_padding_active_seconds)
        if self.foot_padding_active_seconds < 0.0:
            foot_frames = None

        return _AdaptiveState(
            frames=0,
            duration_frames=duration_frames,
            head_limit=head_frames,
            foot_limit=foot_frames,
        )

    def _seconds_to_frames(self, seconds: float) -> int:
        value = float(seconds)
        if value <= 0.0:
            return 0
        frames = int(round(value * self.fps))
        return frames if frames > 0 else 1

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def _process_person(
        self,
        keypoints: List[List[float]],
        state: _AdaptiveState,
        width: float,
        height: float,
    ) -> None:
        frames = state.frames
        duration_frames = state.duration_frames
        duration_exceeded = duration_frames is not None and frames >= duration_frames

        if duration_exceeded:
            upper_offset = (
                state.locked_upper_offset
                if state.locked_upper_offset is not None
                else state.current_upper_offset
            )
            foot_offset = (
                state.locked_foot_offset
                if state.locked_foot_offset is not None
                else state.current_foot_offset
            )
        else:
            self._update_offsets(keypoints, state, width, height)
            upper_offset = state.current_upper_offset
            foot_offset = state.current_foot_offset

        locked_scales: Optional[Dict[str, float]] = None
        if duration_exceeded and state.locked_leg_scales:
            locked_scales = state.locked_leg_scales

        self._apply_offsets(keypoints, upper_offset, foot_offset)

        if self.scale_legs:
            self._adjust_legs(keypoints, state, upper_offset, foot_offset, locked_scales)

        state.frames = frames + 1

        should_lock = False
        if duration_frames is not None and state.frames >= duration_frames:
            should_lock = True

        if should_lock and state.locked_upper_offset is None:
            state.locked_upper_offset = state.current_upper_offset
            state.locked_foot_offset = state.current_foot_offset
            if self.scale_legs:
                state.locked_leg_scales = {
                    "left": state.current_leg_scales.get("left", 1.0),
                    "right": state.current_leg_scales.get("right", 1.0),
                }

    def _update_offsets(
        self,
        keypoints: List[List[float]],
        state: _AdaptiveState,
        width: float,
        height: float,
    ) -> None:
        head_active = self._padding_active(state.head_limit, state.head_counter)
        foot_active = (
            not self.lock_feet
            and self._padding_active(state.foot_limit, state.foot_counter)
        )

        head_padding_value = self._resolve_padding(self.head_padding, height)
        foot_padding_value = self._resolve_padding(self.foot_padding, height)

        head_position = self._min_y(keypoints, self.HEAD_INDICES)
        foot_indices: Tuple[int, ...] = self.ANKLE_INDICES + self.FOOT_EXTRA_INDICES
        foot_position = self._max_y(keypoints, foot_indices)

        if head_active and head_position is not None:
            state.current_upper_offset = head_padding_value - head_position
            state.head_counter += 1
        elif state.head_limit is not None and state.head_counter >= state.head_limit:
            state.current_upper_offset = 0.0

        if foot_active and foot_position is not None:
            target = height - foot_padding_value
            state.current_foot_offset = target - foot_position
            state.foot_counter += 1
        elif state.foot_limit is not None and state.foot_counter >= state.foot_limit:
            state.current_foot_offset = 0.0
        elif self.lock_feet:
            state.current_foot_offset = 0.0

    def _padding_active(self, limit: Optional[int], counter: int) -> bool:
        if limit is None:
            return True
        if limit <= 0:
            return False
        return counter < limit

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _apply_offsets(
        self,
        keypoints: List[List[float]],
        upper_offset: float,
        foot_offset: float,
    ) -> None:
        upper_offset = float(upper_offset or 0.0)
        foot_offset = float(foot_offset or 0.0)

        upper_indices = self.HEAD_INDICES + self.TORSO_INDICES
        for index in upper_indices:
            if index < len(keypoints):
                point = keypoints[index]
                if self._valid_point(point):
                    point[0] = point[0] + upper_offset

        if not self.lock_feet and foot_offset:
            foot_indices = self.ANKLE_INDICES + self.FOOT_EXTRA_INDICES
            for index in foot_indices:
                if index < len(keypoints):
                    point = keypoints[index]
                    if self._valid_point(point):
                        point[0] = point[0] + foot_offset

    def _adjust_legs(
        self,
        keypoints: List[List[float]],
        state: _AdaptiveState,
        upper_offset: float,
        foot_offset: float,
        locked_scales: Optional[Dict[str, float]],
    ) -> None:
        leg_ids = ((12, 15, 17), (13, 16, 18))
        for name, indices in (("left", leg_ids[0]), ("right", leg_ids[1])):
            hip_idx, knee_idx, ankle_idx = indices
            if ankle_idx >= len(keypoints) or hip_idx >= len(keypoints):
                continue

            hip = keypoints[hip_idx]
            ankle = keypoints[ankle_idx]
            if not self._valid_point(hip) or not self._valid_point(ankle):
                continue

            hip_original_y = hip[0] - upper_offset
            ankle_original_y = ankle[0] - (0.0 if self.lock_feet else foot_offset)

            hip_adjusted_y = hip_original_y + upper_offset
            ankle_adjusted_y = ankle_original_y + (
                0.0 if self.lock_feet else foot_offset
            )

            hip_original = [hip_original_y, hip[1]]
            ankle_original = [ankle_original_y, ankle[1]]
            hip_adjusted = [hip_adjusted_y, hip[1]]
            ankle_adjusted = [ankle_adjusted_y, ankle[1]]

            length_original = self._distance(hip_original, ankle_original)
            length_adjusted = self._distance(hip_adjusted, ankle_adjusted)

            target_scale = None
            if locked_scales is not None:
                target_scale = locked_scales.get(name)

            scale = 1.0
            if length_original > 0.0 and length_adjusted > 0.0:
                if target_scale is not None and target_scale > 0.0:
                    scale = float(target_scale)
                    length_adjusted = length_original * scale
                else:
                    scale = length_adjusted / length_original

            state.current_leg_scales[name] = scale

            if knee_idx >= len(keypoints):
                continue

            knee = keypoints[knee_idx]
            if not self._valid_point(knee):
                continue

            knee_original_y = knee[0] - upper_offset
            knee_original = [knee_original_y, knee[1]]

            ratio = 0.0
            if length_original > 0.0:
                ratio = self._distance(knee_original, ankle_original) / length_original
            ratio = max(0.0, min(1.0, ratio))

            leg_vector_y = hip_adjusted[0] - ankle_adjusted[0]
            leg_vector_x = hip_adjusted[1] - ankle_adjusted[1]

            if target_scale is not None and target_scale > 0.0:
                leg_vector_y = (hip_original[0] - ankle_original[0]) * target_scale
                leg_vector_x = (hip_original[1] - ankle_original[1]) * target_scale

            knee_new_y = ankle_adjusted[0] + leg_vector_y * ratio
            knee_new_x = ankle_adjusted[1] + leg_vector_x * ratio

            knee[0] = knee_new_y
            knee[1] = knee_new_x

    def _extract_canvas_size(self, pose_data: MutableMapping) -> Tuple[Optional[float], Optional[float]]:
        width = pose_data.get("canvas_width")
        height = pose_data.get("canvas_height")
        if width is None:
            width = pose_data.get("width")
        if height is None:
            height = pose_data.get("height")
        if isinstance(width, (list, tuple)) and len(width) >= 2 and height is None:
            height = width[1]
            width = width[0]
        if isinstance(height, (list, tuple)) and len(height) >= 2 and width is None:
            width = height[0]
            height = height[1]
        return width, height

    def _valid_point(self, point: List[float]) -> bool:
        return (
            isinstance(point, list)
            and len(point) >= 3
            and point[0] is not None
            and point[1] is not None
            and point[2] is not None
            and point[2] > 0.0
        )

    def _resolve_padding(self, padding: float, height: float) -> float:
        value = float(padding)
        if self.normalize:
            value *= height
        return value

    def _min_y(self, keypoints: List[List[float]], indices: Iterable[int]) -> Optional[float]:
        minimum = None
        for index in indices:
            if index < len(keypoints):
                point = keypoints[index]
                if self._valid_point(point):
                    value = float(point[0])
                    if minimum is None or value < minimum:
                        minimum = value
        return minimum

    def _max_y(self, keypoints: List[List[float]], indices: Iterable[int]) -> Optional[float]:
        maximum = None
        for index in indices:
            if index < len(keypoints):
                point = keypoints[index]
                if self._valid_point(point):
                    value = float(point[0])
                    if maximum is None or value > maximum:
                        maximum = value
        return maximum

    def _distance(self, point_a: List[float], point_b: List[float]) -> float:
        dy = float(point_a[0]) - float(point_b[0])
        dx = float(point_a[1]) - float(point_b[1])
        return (dy * dy + dx * dx) ** 0.5


class _OffsetNamespace:
    Automatic = PoseDataEditorOffsetAutomatic


class _PoseDataEditorNamespace:
    Offset = _OffsetNamespace()


PoseDataEditor = _PoseDataEditorNamespace()

__all__ = [
    "PoseDataEditorOffsetAutomatic",
    "PoseDataEditor",
]
