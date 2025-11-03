"""Compatibility wrapper for the standalone offset automatic editor."""

from __future__ import annotations

from .pose_data_editor_offset_automatic import PoseDataEditorOffsetAutomatic


class PoseDataEditorAloneAutomatic(PoseDataEditorOffsetAutomatic):
    """Backward-compatible alias for :class:`PoseDataEditorOffsetAutomatic`."""

    pass


__all__ = ["PoseDataEditorAloneAutomatic"]
