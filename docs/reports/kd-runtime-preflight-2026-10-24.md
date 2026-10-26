# KD runtime workload preflight

[Fact] The official seed-0 student and three-member static-SC BEATs teacher passed real-audio preprocessing on A. Both use exactly the same first 32 clean official-test cycle IDs, selected without class labels. Student inputs retain 5 seconds; teacher members retain their original 5/8/8 seconds. Shapes are respectively `(32,498,128)` and `(32,498,128)`, `(32,798,128)`, `(32,798,128)`. Input-file hashes were checked after both runs and again by the paired self-check.

[Pending] These are workload checks, with no model loading or timing. All three teacher checkpoints exist locally. The student checkpoint is absent at:

```text
checkpoints/train/distill-pilot-20261024/beats+ft+pafa+kd+sc+sc_static/seed0/report_epoch_100.pth
```

No KD checkpoint was found in the local checkpoints tree. Prediction exports and completion markers alone do not establish runtime, VRAM, or checkpoint reload integrity. The official three-seed quality results remain in the [synced analysis](synced-three-seed-lodo-progress-2026-10-24.md); Litt3200 remains pending. Local completion markers currently number 24/100 dynamic-SC and 3/6 KD; this is an artifact count, not fresh remote liveness verification.

## Benchmark scope

The prepared script `outputs/evaluation/report-20261024/runtime/benchmark.py` reuses frozen loading, SC and prediction helpers. It admits one student/control or three teacher members sharing seed, split and data path. Each arm runs in a fresh CLI process, on the same GPU, in FP32 with TF32 disabled and three CPU threads. Batches 1 and 32 each receive three warmups and twenty synchronized measured calls. It reports per-call times, allocator peak allocated/reserved bytes, model/checkpoint sizes and process RSS. Teacher probabilities use equal weights.

Two scopes are reported: cached filter-bank features to probabilities, and decoded waveforms to probabilities including each member's padding, saved SC and feature extraction. Disk decoding, SC coefficient estimation, model loading, conformal prediction and API overhead are excluded. Allocator peaks describe the current process, not isolated whole-device VRAM. This fixed workload is an engineering comparison; it does not characterize latency across devices, recordings, concurrent API traffic or the entire serving pipeline.

Run only when the student checkpoint is available and a usable GPU can be assigned without displacing the authorized priority queues or API. Set `CUDA_VISIBLE_DEVICES` to the assigned single GPU. Run the following commands separately on the same hardware, preserving their outputs:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/runtime/benchmark.py \
  outputs/train/distill-pilot-20261024/beats+ft+pafa+kd+sc+sc_static/seed0 \
  --out outputs/evaluation/report-20261024/runtime/student-measured.json

OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/runtime/benchmark.py \
  outputs/train/lodo-v1/beats+ft+pafa+sc+sc_static/seed0 \
  --out outputs/evaluation/report-20261024/runtime/control-measured.json

OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=. .venv-train/bin/python \
  outputs/evaluation/report-20261024/runtime/benchmark.py \
  outputs/train/lodo-v1/beats+ft+pafa+sc+sc_static/seed0 \
  outputs/train/lodo-v1/beats+ft+sg_scl+sc+sc_static/seed0 \
  outputs/train/lodo-v1/beats+ft+patchmix_cl+sc+sc_static/seed0 \
  --out outputs/evaluation/report-20261024/runtime/teacher-measured.json
```

The script refuses existing output paths and missing checkpoints. CPU preflight artifacts are `student-preflight.json`, `teacher-preflight.json` and their logs in the same artifact directory. The paired check is reproducible with:

```bash
PYTHONPATH=. .venv/bin/python outputs/evaluation/report-20261024/runtime/check.py
```

[Fact] Both refusal paths were exercised: missing-checkpoint measurement failed without producing a result, and an attempt to reuse the student preflight path failed without overwriting it. Logs are `missing-checkpoint-guard.log` and `overwrite-guard.log`. Both scripts compile; the paired input-hash check passes.

[Interpretation] No measured runtime or memory saving is established. The engineering claim requires actual GPU measurements plus the remaining KD quality and provenance checks; absence of a significant quality difference does not establish equivalence.
