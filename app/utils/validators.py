"""Validation helpers for uploads and user-facing inputs."""

from __future__ import annotations

from pathlib import Path

from app.config.settings import MAX_UPLOAD_MB, SUPPORTED_AUDIO_TYPES


def validate_audio_upload(uploaded_file) -> tuple[bool, str]:
    """Validate Streamlit uploaded audio files before processing."""
    if uploaded_file is None:
        return False, "No file was uploaded."

    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_AUDIO_TYPES:
        return False, f"Unsupported file type: .{suffix}"

    size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return False, f"File is too large ({size_mb:.1f} MB). Limit is {MAX_UPLOAD_MB} MB."

    return True, ""

