"""Standalone Pose Data Editor that adapts torso offsets and leg scaling.

This module exposes the :class:`PoseDataEditorAloneAutomatic` component which
modifies pose keypoints stored in dictionaries following the structure of the
``format_json`` sample bundled with the repository. The editor lifts the upper
body towards the canvas top while optionally stretching the legs so the hips
remain connected to the anchored feet. Head and foot paddings can be toggled on
independent timers, and all adaptive adjustments lock after a configurable
number of seconds derived from the provided FPS value.

No external dependencies are required; the implementation works directly with
built-in Python data structures.
"""


class PoseDataEditorAloneAutomatic:
    """Adjusts upper-body offsets and leg scaling over time for pose keypoints."""

    HEAD_INDICES = (0, 1, 2, 3, 4, 5)
    TORSO_INDICES = (6, 7, 8, 9, 10, 11, 12, 13, 14)
    HIP_INDICES = (12, 13, 14)
    KNEE_INDICES = (15, 16)
    ANKLE_INDICES = (17, 18)
    FOOT_EXTRA_INDICES = (19, 20, 21, 22, 23, 24)

    def __init__(
        self,
        duration_seconds,
        head_padding,
        foot_padding,
        normalize,
        head_padding_active_seconds,
        foot_padding_active_seconds,
        lock_feet,
        scale_legs,
        fps,
    ):
        self.duration_seconds = float(duration_seconds)
        self.head_padding = float(head_padding)
        self.foot_padding = float(foot_padding)
        self.normalize = bool(normalize)
        self.head_padding_active_seconds = float(head_padding_active_seconds)
        self.foot_padding_active_seconds = float(foot_padding_active_seconds)
        self.lock_feet = bool(lock_feet)
        self.scale_legs = bool(scale_legs)
        self.fps = int(round(float(fps))) if float(fps) > 0 else 1
        self._states = {}

    def process(self, pose_data):
        """Return a modified copy of ``pose_data`` with offsets applied."""

        if not isinstance(pose_data, dict):
            raise TypeError("pose_data must be a dictionary")

        width, height = self._extract_canvas_size(pose_data)
        if width is None or height is None:
            raise ValueError("pose_data must include canvas width and height")

        width = float(width)
        height = float(height)

        cloned = self._clone_pose_data(pose_data)
        frames = cloned.get("keypoints")
        if not isinstance(frames, list):
            return cloned

        self._states = {}

        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for person_id, keypoints in frame.items():
                state = self._states.get(person_id)
                if state is None:
                    state = self._create_state()
                    self._states[person_id] = state
                if isinstance(keypoints, list):
                    self._process_person(keypoints, state, width, height)

        return cloned

    def _create_state(self):
        duration_frames = self._seconds_to_frames(self.duration_seconds)
        if duration_frames == 0:
            duration_frames = None

        if self.head_padding_active_seconds < 0.0:
            head_frames = -1
        else:
            head_frames = self._seconds_to_frames(self.head_padding_active_seconds)

        if self.foot_padding_active_seconds < 0.0:
            foot_frames = -1
        else:
            foot_frames = self._seconds_to_frames(self.foot_padding_active_seconds)

        return {
            "frames": 0,
            "duration_frames": duration_frames,
            "head_limit": head_frames,
            "foot_limit": foot_frames,
            "head_counter": 0,
            "foot_counter": 0,
            "current_upper_offset": 0.0,
            "current_foot_offset": 0.0,
            "locked_upper_offset": None,
            "locked_foot_offset": None,
            "locked_leg_scales": None,
            "current_leg_scales": {
                "left": 1.0,
                "right": 1.0,
            },
        }

    def _seconds_to_frames(self, seconds):
        value = float(seconds)
        if value <= 0.0:
            return 0
        frames = int(round(value * self.fps))
        if frames <= 0:
            frames = 1
        return frames

    def _process_person(self, keypoints, state, width, height):
        frames = state["frames"]
        duration_frames = state["duration_frames"]
        duration_exceeded = duration_frames is not None and frames >= duration_frames

        if duration_exceeded:
            upper_offset = state["locked_upper_offset"]
            foot_offset = state["locked_foot_offset"]
            if upper_offset is None:
                upper_offset = state["current_upper_offset"]
            if foot_offset is None:
                foot_offset = state["current_foot_offset"]
        else:
            self._update_offsets(keypoints, state, width, height)
            upper_offset = state["current_upper_offset"]
            foot_offset = state["current_foot_offset"]

        if duration_exceeded and state["locked_leg_scales"]:
            state["current_leg_scales"] = {
                "left": state["locked_leg_scales"].get("left", 1.0),
                "right": state["locked_leg_scales"].get("right", 1.0),
            }

        self._apply_offsets(keypoints, upper_offset, foot_offset)
        if self.scale_legs:
            self._adjust_legs(keypoints, state, upper_offset, foot_offset)

        state["frames"] = frames + 1

        if not duration_exceeded and duration_frames is not None and state["frames"] >= duration_frames:
            state["locked_upper_offset"] = state["current_upper_offset"]
            state["locked_foot_offset"] = state["current_foot_offset"]
            if self.scale_legs:
                state["locked_leg_scales"] = {
                    "left": state["current_leg_scales"].get("left", 1.0),
                    "right": state["current_leg_scales"].get("right", 1.0),
                }

    def _update_offsets(self, keypoints, state, width, height):
        head_limit = state["head_limit"]
        foot_limit = state["foot_limit"]

        head_active = False
        if head_limit < 0:
            head_active = True
        elif head_limit > 0:
            head_active = state["head_counter"] < head_limit

        foot_active = False
        if foot_limit < 0:
            foot_active = True
        elif foot_limit > 0:
            foot_active = state["foot_counter"] < foot_limit

        head_padding_value = self.head_padding
        foot_padding_value = self.foot_padding
        if self.normalize:
            head_padding_value = head_padding_value * height
            foot_padding_value = foot_padding_value * height

        head_position = self._min_y(keypoints, self.HEAD_INDICES)
        foot_position = self._max_y(keypoints, self.ANKLE_INDICES + self.FOOT_EXTRA_INDICES)

        if head_active and head_position is not None:
            state["current_upper_offset"] = head_padding_value - head_position
            state["head_counter"] = state["head_counter"] + 1

        if self.lock_feet:
            state["current_foot_offset"] = 0.0
        elif foot_active and foot_position is not None:
            target = height - foot_padding_value
            state["current_foot_offset"] = target - foot_position
            state["foot_counter"] = state["foot_counter"] + 1

    def _apply_offsets(self, keypoints, upper_offset, foot_offset):
        if upper_offset is None:
            upper_offset = 0.0
        if foot_offset is None:
            foot_offset = 0.0

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

    def _adjust_legs(self, keypoints, state, upper_offset, foot_offset):
        leg_ids = ((12, 15, 17), (13, 16, 18))
        for name, indices in (("left", leg_ids[0]), ("right", leg_ids[1])):
            hip_idx, knee_idx, ankle_idx = indices
            if ankle_idx >= len(keypoints) or hip_idx >= len(keypoints):
                continue
            hip = keypoints[hip_idx]
            ankle = keypoints[ankle_idx]
            if not self._valid_point(hip) or not self._valid_point(ankle):
                continue

            hip_original = [hip[0] - upper_offset, hip[1]]
            ankle_original = [ankle[0] - foot_offset if not self.lock_feet else ankle[0], ankle[1]]

            hip_adjusted = [hip_original[0] + upper_offset, hip_original[1]]
            ankle_adjusted = [ankle_original[0] + (foot_offset if not self.lock_feet else 0.0), ankle_original[1]]

            length_original = self._distance(hip_original, ankle_original)
            length_adjusted = self._distance(hip_adjusted, ankle_adjusted)
            scale = 1.0
            if length_original > 0.0 and length_adjusted > 0.0:
                scale = length_adjusted / length_original
            state["current_leg_scales"][name] = scale

            if knee_idx >= len(keypoints):
                continue
            knee = keypoints[knee_idx]
            if not self._valid_point(knee):
                continue

            knee_original = [knee[0] - upper_offset, knee[1]]
            knee_distance = self._distance(knee_original, ankle_original)
            ratio = 0.0
            if length_original > 0.0:
                ratio = knee_distance / length_original
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0

            leg_vector_y = hip_adjusted[0] - ankle_adjusted[0]
            leg_vector_x = hip_adjusted[1] - ankle_adjusted[1]

            knee_new_y = ankle_adjusted[0] + leg_vector_y * ratio
            knee_new_x = ankle_adjusted[1] + leg_vector_x * ratio

            knee[0] = knee_new_y
            knee[1] = knee_new_x

    def _clone_pose_data(self, pose_data):
        cloned = {}
        for key, value in pose_data.items():
            if key == "keypoints" and isinstance(value, list):
                frames = []
                for frame in value:
                    if isinstance(frame, dict):
                        persons = {}
                        for person_id, points in frame.items():
                            if isinstance(points, list):
                                new_points = []
                                for point in points:
                                    if isinstance(point, list) and len(point) >= 3:
                                        new_points.append([float(point[0]), float(point[1]), float(point[2])])
                                    else:
                                        new_points.append(point)
                                persons[person_id] = new_points
                        frames.append(persons)
                    else:
                        frames.append(frame)
                cloned[key] = frames
            else:
                cloned[key] = value
        return cloned

    def _extract_canvas_size(self, pose_data):
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

    def _valid_point(self, point):
        return (
            isinstance(point, list)
            and len(point) >= 3
            and point[0] is not None
            and point[1] is not None
            and point[2] is not None
            and point[2] > 0.0
        )

    def _min_y(self, keypoints, indices):
        minimum = None
        for index in indices:
            if index < len(keypoints):
                point = keypoints[index]
                if self._valid_point(point):
                    value = float(point[0])
                    if minimum is None or value < minimum:
                        minimum = value
        return minimum

    def _max_y(self, keypoints, indices):
        maximum = None
        for index in indices:
            if index < len(keypoints):
                point = keypoints[index]
                if self._valid_point(point):
                    value = float(point[0])
                    if maximum is None or value > maximum:
                        maximum = value
        return maximum

    def _distance(self, point_a, point_b):
        dy = float(point_a[0]) - float(point_b[0])
        dx = float(point_a[1]) - float(point_b[1])
        return (dy * dy + dx * dx) ** 0.5
