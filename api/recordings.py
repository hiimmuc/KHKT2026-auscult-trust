"""Private storage for original microphone takes, separate from research datasets."""

import hashlib
import io
import ipaddress
import json
import wave
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo


def save_recording(data, client_ip, mode, root):
    """Keep the original PCM WAV, including silent takes that inference may reject."""
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels, width, sr, frames = audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
            if channels not in (1, 2) or width != 2 or not 4000 <= sr <= 48000 or not 0 < frames / sr <= 92:
                raise ValueError("Expected a mono/stereo 16-bit PCM WAV at 4–48 kHz, up to 92 seconds.")
            if len(audio.readframes(frames)) != frames * channels * width:
                raise ValueError("The WAV is truncated.")
    except (wave.Error, EOFError) as exc:
        raise ValueError("The recording is not a readable PCM WAV.") from exc
    try:
        address = str(ipaddress.ip_address(client_ip))
    except ValueError:
        address = "unknown"
    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    name = "{}_{}_{}.wav".format(address.replace(":", "-"), now.strftime("%Y%m%d_%H%M%S_%f%z"), uuid4().hex[:8])
    folder = Path(root) / now.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / name
    meta = {"filename": name, "saved_at": now.isoformat(), "client_ip": address, "mode": mode,
            "sample_rate": sr, "channels": channels, "duration_sec": frames / sr,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    sidecar = path.with_suffix(".json")
    try:
        with path.open("xb") as stream:
            path.chmod(0o600)
            stream.write(data)
        with sidecar.open("x") as stream:
            sidecar.chmod(0o600)
            json.dump(meta, stream, indent=2)
    except OSError:
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return {"filename": name, "saved_at": meta["saved_at"], "sha256": meta["sha256"]}
