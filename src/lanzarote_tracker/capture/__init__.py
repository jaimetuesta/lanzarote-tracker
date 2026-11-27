"""Captura: URL del directo, grabacion de clips y extraccion de fotogramas."""

from __future__ import annotations

from lanzarote_tracker.capture.frames import (
    SPLIT_NAMES,
    extract_all_frames,
    extract_frames,
    frame_offset_seconds,
    read_splits,
    split_by_clip,
    write_splits,
)
from lanzarote_tracker.capture.recorder import (
    ClipMetadata,
    build_clip_name,
    clip_time_utc,
    load_clip_metadata,
    metadata_path_for,
    record_clip,
    record_clips,
)
from lanzarote_tracker.capture.stream_url import get_stream_url

__all__ = [
    "SPLIT_NAMES",
    "extract_all_frames",
    "extract_frames",
    "frame_offset_seconds",
    "read_splits",
    "split_by_clip",
    "write_splits",
    "ClipMetadata",
    "build_clip_name",
    "clip_time_utc",
    "load_clip_metadata",
    "metadata_path_for",
    "record_clip",
    "record_clips",
    "get_stream_url",
]
