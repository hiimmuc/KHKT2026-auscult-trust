"""API contract checks that need no checkpoints or GPU: start-up without a model, validation, auth, error shape.

    PYTHONPATH=. .venv-train/bin/python -m pytest -q api/test_contract.py
"""

import io
import json
import asyncio
import threading

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from api import engine, main

SAMPLE = {"id": "s1", "class": "normal", "file": "samples/s1.wav", "device": "Meditron", "cycles": []}
MANIFEST = {"classes": ["normal", "crackle", "wheeze", "both"], "classes2": ["normal", "abnormal"], "devices": ["Meditron", "Litt3200"],
            "alpha": 0.1, "preprocessing": {}, "pipelines": {}, "samples": [SAMPLE]}


def test_annotated_cycles_use_training_length_screening_stays_eight_seconds():
    import threading
    import torch

    lengths = []
    class Capture:
        thr = {"cycle4": 0.9, "cycle2": 0.9, "screen": 0.9}
        def run(self, segments, screening, *args):
            lengths.append((len(segments[0]), len(screening[0])))
            return {}, np.tile([0.7, 0.1, 0.1, 0.1], (len(segments), 1)), np.tile([0.7, 0.1, 0.1, 0.1], (len(screening), 1))
    e = engine.Engine.__new__(engine.Engine)
    e.m = {**MANIFEST, "preprocessing": {"segment_sec": 5}}
    e.pipelines, e.lock = {"capture": Capture()}, threading.Lock()
    x = torch.ones(1, 10 * engine.SR)
    e.infer(x, 10, cycles=[(0, 2)])
    timed = e.infer(x, 10)
    assert lengths == [(5 * engine.SR, 8 * engine.SR), (8 * engine.SR, 8 * engine.SR)]
    assert timed["timing_stages_ms"]["segments_cut"] >= 0
    assert timed["timing_stages_ms"]["pipelines"]["capture"]["total"] >= 0


class Stub:
    """Stands in for the model: returns a fixed, plainly fake result so the upload plumbing can be checked."""

    pipelines = {"stub": None}

    def __init__(self, manifest, bundle):
        pass

    def infer(self, x16, duration, device=None, cycles=None, window_sec=8.0, hop_sec=4.0):
        return {"segmentation": {"mode": "windows"}, "pipelines": {"stub": {"seen_sec": round(duration, 1), "device": device}}}


def wav(seconds=3.0, sr=16000):
    b = io.BytesIO()
    sf.write(b, 0.1 * np.sin(np.arange(int(seconds * sr)) * 0.05).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def upload(c, data=None, **form):
    return c.post("/v1/infer", files={"file": ("x.wav", wav() if data is None else data, "audio/wav")}, data=form)


@pytest.fixture
def empty_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "BUNDLE", tmp_path)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def stub_client(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    monkeypatch.setattr(main, "BUNDLE", tmp_path)
    monkeypatch.setattr(engine, "Engine", Stub)
    monkeypatch.setattr(main, "TOKEN", "")
    monkeypatch.setattr(main, "AUTH", "auto")
    monkeypatch.setattr(main, "RATE_LIMIT", 0)
    monkeypatch.setattr(main, "SAMPLES", "on")
    with TestClient(main.app) as c:
        yield c


def test_no_bundle_is_unavailable_not_fabricated(empty_bundle):
    c = empty_bundle
    h = c.get("/v1/health")
    assert h.status_code == 200 and h.json()["status"] == "unavailable" and h.json()["pipelines"] == []
    assert "/" not in h.json()["reason"]
    for r in (c.get("/v1/model"), c.get("/v1/samples"), c.get("/v1/samples/a/result"), upload(c), upload(c, b"junk")):
        assert r.status_code == 503, r.url
        assert r.json()["error"]["code"] == "model_unavailable" and r.json()["request_id"] and "pipelines" not in r.json()


def test_failed_model_load_hides_paths(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    monkeypatch.setattr(main, "BUNDLE", tmp_path)

    def boom(*a):
        raise FileNotFoundError("/secret/checkpoints/report_epoch_50.pth")

    monkeypatch.setattr(engine, "Engine", boom)
    with TestClient(main.app) as c:
        h = c.get("/v1/health").json()
        assert h["status"] == "unavailable" and "secret" not in json.dumps(h) and "FileNotFoundError" in h["reason"]
        assert upload(c).status_code == 503


def test_upload_validation(stub_client):
    c = stub_client
    assert c.get("/v1/health").json()["status"] == "ok"
    bad = upload(c, b"not a wav file at all")
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "not_wav" and bad.json()["request_id"]
    assert upload(c, wav(0.5)).json()["error"]["code"] == "duration"
    assert upload(c, wav(sr=2000)).json()["error"]["code"] == "sample_rate"
    assert upload(c, device="Nokia").json()["error"]["code"] == "bad_device"
    assert upload(c, window_sec="20").json()["error"]["code"] == "bad_window"
    assert upload(c, cycles="[[3, 1]]").json()["error"]["code"] == "bad_cycles"
    assert upload(c, cycles="nope").json()["error"]["code"] == "bad_cycles"
    r = c.post("/v1/infer", files={"file": ("x.wav", wav(), "audio/wav"), "annotation": ("a.txt", b"0.5 2.0 1 x\n", "text/plain")})
    assert r.json()["error"]["code"] == "bad_annotation"
    assert upload(c, b"RIFF" + b"\0" * 4 + b"WAVE" + b"\0" * 40).json()["error"]["code"] == "unreadable"


def test_too_large(stub_client, monkeypatch):
    monkeypatch.setattr(main, "MAX_BYTES", 1000)
    r = upload(stub_client)
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"


def test_upload_response_shape_and_cache(stub_client):
    c = stub_client
    a = upload(c, device="Meditron")
    assert a.status_code == 200 and a.headers["X-Request-ID"] == a.json()["request_id"]
    j = a.json()
    assert j["source"] == {"kind": "upload"} and j["audio"]["duration_sec"] == 3.0 and j["device"] == {"declared": "Meditron"}
    assert j["pipelines"]["stub"]["device"] == "Meditron" and j["cached"] is False
    assert j["timing_stages_ms"]["resample"] >= 0
    again = upload(c, device="Meditron").json()
    assert again["cached"] is True and again["timing_stages_ms"] == {}


def test_disconnected_queued_inference_never_runs():
    main.reset()
    entered, release = threading.Event(), threading.Event()
    called = []

    class Request:
        disconnected = False

        async def is_disconnected(self):
            return self.disconnected

    async def check():
        first_request, second_request = Request(), Request()
        first = asyncio.create_task(main.guarded_cached(first_request, "first", lambda: (entered.set(), release.wait(2))))
        await asyncio.to_thread(entered.wait, 2)
        second = asyncio.create_task(main.guarded_cached(second_request, "second", lambda: called.append(True)))
        while main.state["pending"] < 2:
            await asyncio.sleep(0.01)
        second_request.disconnected = True
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        await first
        assert not called and main.state["pending"] == 0

    asyncio.run(check())


def test_timeout_keeps_model_slot_until_worker_finishes(monkeypatch):
    main.reset()
    monkeypatch.setattr(main, "TIMEOUT_SEC", 0.05)
    release = threading.Event()

    class Request:
        async def is_disconnected(self):
            return False

    async def check():
        with pytest.raises(main.ApiError) as exc:
            await main.guarded_cached(Request(), "slow", lambda: release.wait(2))
        assert exc.value.code == "timeout"
        assert main.state["work_lock"].locked() and main.state["pending"] == 1
        release.set()
        while main.state["pending"]:
            await asyncio.sleep(0.01)
        assert not main.state["work_lock"].locked()

    asyncio.run(check())


def test_recording_keeps_original_bytes_without_a_model(empty_bundle, tmp_path, monkeypatch):
    root = tmp_path / "recordings"
    monkeypatch.setattr(main, "RECORDINGS", root)
    monkeypatch.setattr(main, "AUTH", "off")
    raw = wav()
    r = empty_bundle.post("/v1/recordings", files={"file": ("../../unsafe.wav", raw)},
                          data={"mode": "demo"}, headers={"X-Forwarded-For": "203.0.113.9"})
    assert r.status_code == 200, r.text
    saved = r.json()
    files = list(root.glob("*/*.wav"))
    assert len(files) == 1 and files[0].read_bytes() == raw
    assert files[0].name == saved["filename"] and files[0].name.startswith("unknown_")
    meta = json.loads(files[0].with_suffix(".json").read_text())
    assert meta["mode"] == "demo" and meta["client_ip"] == "unknown"
    assert meta["saved_at"].endswith("+07:00") and meta["sha256"] == saved["sha256"]
    assert files[0].stat().st_mode & 0o777 == 0o600
    assert empty_bundle.post("/v1/recordings", files={"file": ("x.wav", b"junk")}).status_code == 422
    assert len(list(root.glob("*/*.wav"))) == 1


def test_recording_validation_and_auth(stub_client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RECORDINGS", tmp_path / "takes")
    c = stub_client
    assert c.post("/v1/recordings", files={"file": ("x.wav", wav())}, data={"mode": "invalid"}).status_code == 422
    monkeypatch.setattr(main, "MAX_BYTES", 10)
    assert c.post("/v1/recordings", files={"file": ("x.wav", wav())}).status_code == 413
    monkeypatch.setattr(main, "AUTH", "on")
    monkeypatch.setattr(main, "TOKEN", "recording-test")
    assert c.post("/v1/recordings", files={"file": ("x.wav", wav())}).status_code == 401


def test_recording_client_ip_requires_authenticated_gateway(stub_client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RECORDINGS", tmp_path / "takes")
    monkeypatch.setattr(main, "PROXY_TOKEN", "gateway-test-token")
    for token, expected in (("forged", "unknown"), ("gateway-test-token", "203.0.113.12")):
        r = stub_client.post("/v1/recordings", files={"file": ("mic.wav", wav())}, headers={
            "X-Auscult-Proxy-Token": token, "X-Auscult-Client-IP": "203.0.113.12"})
        assert r.status_code == 200
        assert r.json()["filename"].startswith(expected + "_")


def test_recording_ipv6_silence_unique_names_and_storage_failure(tmp_path, monkeypatch, stub_client):
    from api.recordings import save_recording

    b = io.BytesIO()
    sf.write(b, np.zeros(16000), 16000, format="WAV", subtype="PCM_16")
    a = save_recording(b.getvalue(), "2001:db8::1", "real", tmp_path)
    other = save_recording(b.getvalue(), "2001:db8::1", "real", tmp_path)
    assert a["filename"].startswith("2001-db8--1_") and a["filename"] != other["filename"]
    broken = tmp_path / "not-a-directory"
    broken.write_text("occupied")
    monkeypatch.setattr(main, "RECORDINGS", broken)
    r = stub_client.post("/v1/recordings", files={"file": ("x.wav", wav())})
    assert r.status_code == 503 and r.json()["error"]["code"] == "recording_storage"
    assert str(tmp_path) not in r.text


def test_inference_crash_is_a_500_without_a_result(stub_client, monkeypatch):
    def crash(*a, **k):
        raise RuntimeError("CUDA out of memory at /home/x/model.py")

    monkeypatch.setattr(Stub, "infer", crash)
    r = upload(stub_client)
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal" and "/home" not in r.text and "pipelines" not in r.json()


def test_token_and_cors(stub_client, monkeypatch):
    c = stub_client
    monkeypatch.setattr(main, "TOKEN", "s3cret")
    assert c.get("/v1/health").status_code == 200  # open: hosting platforms probe it
    assert c.get("/v1/model").status_code == 401 and c.get("/v1/model").json()["error"]["code"] == "unauthorized"
    assert upload(c).status_code == 401
    assert c.get("/v1/model", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    origin = main.ORIGINS[0]
    pre = c.options("/v1/infer", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "authorization"})
    assert pre.headers["access-control-allow-origin"] == origin
    assert "access-control-allow-origin" not in c.get("/v1/health", headers={"Origin": "https://evil.example"}).headers


def test_auth_off_ignores_the_token(stub_client, monkeypatch):
    monkeypatch.setattr(main, "TOKEN", "s3cret")
    monkeypatch.setattr(main, "AUTH", "off")
    assert stub_client.get("/v1/model").status_code == 200 and upload(stub_client).status_code == 200


def test_rate_limit(stub_client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 2)
    main.hits.clear()
    assert [upload(stub_client, device="Meditron").status_code for _ in range(2)] == [200, 200]
    r = upload(stub_client)
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
    assert stub_client.get("/v1/model").status_code == 200  # only inference is limited
    main.hits.clear()


def test_samples_switch(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    monkeypatch.setattr(main, "BUNDLE", tmp_path)
    monkeypatch.setattr(engine, "Engine", Stub)
    for mode, n in (("on", 1), ("off", 0)):
        monkeypatch.setattr(main, "SAMPLES", mode)
        with TestClient(main.app) as c:
            assert len(c.get("/v1/samples").json()["samples"]) == n
            assert (c.get("/v1/samples/s1/result").status_code == 404) == (n == 0)  # off: unknown sample
