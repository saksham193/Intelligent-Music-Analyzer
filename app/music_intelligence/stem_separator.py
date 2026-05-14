"""Optional Demucs-based source separation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from app.config.settings import STEM_CACHE_DIR


def separate_stems(audio_path: str | Path, output_dir: str | Path = STEM_CACHE_DIR) -> dict[str, object]:
    """Separate vocals/drums/bass/other with Demucs when installed.

    Demucs is intentionally optional because it is large and computationally
    expensive. The UI can call this only when the user explicitly requests it.
    """
    input_path = Path(audio_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "-o",
        str(output_path),
        str(input_path),
    ]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except FileNotFoundError:
        return {"ok": False, "error": "Python or Demucs is not available in PATH.", "stems": {}}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Stem separation timed out.", "stems": {}}

    if completed.returncode != 0:
        return {"ok": False, "error": completed.stderr.strip() or completed.stdout.strip(), "stems": {}}

    stems = {
        path.stem: str(path)
        for path in output_path.rglob("*.wav")
        if path.name.lower() in {"vocals.wav", "no_vocals.wav"}
    }
    return {"ok": True, "error": "", "stems": stems}
