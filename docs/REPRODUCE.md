# Reproduce

Run from the repo root after [SETUP.md](SETUP.md). `PY=.venv-train/bin/python`; every command takes `PYTHONPATH=.`.
Recipes compose left to right (`--config a.yaml,b.yaml`); the cell name is the stems joined by `+`, without `base`.

## 1. Encoders x methods, official split

5 encoders (AST, HTS-AT, OPERA-CT, BEATs, DASS) x 14 methods (8 fine-tuned, 6 frozen), seeds 0-4, run `grid-v1`.

```bash
$PY scripts/grid.py cells                         # list the 70 cells
$PY scripts/grid.py jobs --smoke a.json b.json    # job lists for two machines (writes logs/campaigns/)
$PY scripts/sweep.py logs/campaigns/grid-v1-A.txt --workers 3 --mem-gb 8
$PY scripts/grid.py audit && $PY scripts/grid.py table --epoch last
```

One unit:

```bash
C=experiments/train
$PY -m src.training.main --config $C/base.yaml,$C/encoder_beats.yaml,$C/grid/pafa_beats.yaml \
    --cell beats+ft+pafa --selection fixed --report_epochs 100 --seed 0
```

## 2. LODO training

Append `$C/lodo_<device>.yaml` (`meditron`, `littc2se`, `litt3200`, `akgc417l`) and `+lodo_<device>` to the cell.

```bash
$PY scripts/grid.py lodo-jobs --smoke a.json b.json --cells ... [--sc-cells ...]
bash scripts/lodo_h2_all.sh                       # null gate and H2 per cell
$PY scripts/lodo_tables.py > lodo_tables.md
```

## 3. SC at training

Add `$C/sc.yaml,$C/sc_static.yaml` (static bound) or `$C/sc.yaml` (dynamic) to any cell, official split or LODO.

```bash
$PY -m src.training.main --config $C/base.yaml,$C/encoder_beats.yaml,$C/grid/pafa_beats.yaml,$C/sc.yaml,$C/sc_static.yaml \
    --cell beats+ft+pafa+sc+sc_static --selection fixed --report_epochs 100 --seed 0
```

## 4. SC at inference

```bash
bash scripts/run_lodo_infer.sh        # test-time SC, A2, TTA-EQ on trained LODO units (src/pipelines/lodo_ttsc.py)
bash scripts/run_kauh.sh              # KAUH filter pairs with frozen models
$PY -m src.training.export_probs outputs/train/<run>/<cell>/seed0 --ckpt report_epoch_<E>.pth   # freeze: softmax + model.sha256
$PY scripts/final_evaluation.py       # frozen baseline vs SC evaluation: KAUH A/B2/B1, screening
```

## 5. Ensembles

```bash
$PY scripts/ensemble.py > ensemble_tables.md            # groups E1, E2, E3 and unions, equal vs fitted weights
$PY scripts/ensemble_sesp.py > ensemble_sesp.md         # Se at PAFA's Sp (k rate)
$PY scripts/final_ensembles.py                          # E1, E1+E3 on KAUH and screening
$PY scripts/followup_ensembles.py template              # fixed additions; then `evaluate <manifest>`
```

## 6. Distillation

The G4 teacher units (anchor `gain6-cv66`, `pcsl5_control`, `class_aware_light`, `multiscale`) are read from `outputs/train/{followup,target}-*-final`. Recipes: `experiments/train/{class_aware_light,multiscale,pcsl5_control,swa,lora,lora_class_aware}.yaml`.

```bash
$PY scripts/target_campaign.py prepare            # teacher / recipe finals: official, LODO, KAUH
$PY scripts/distill_campaign.py prepare           # student fits from teacher softmax; then `run <manifest>`
$PY scripts/student_final_campaign.py prepare     # kd_g4_k1 and its no-SC twin
$PY scripts/student_final_report.py [--h2]
```

Campaign scripts freeze a source snapshot under `build/campaigns/<name>/` and read hosts from the manifest.

## 7. Phantom replay

```bash
$PY scripts/replay_analysis.py --takes ../data/raw/phantom_replay --out outputs/replay/phantom-replay-20261028
```

Then `notebooks/04_replay_phantom.ipynb`. Transducer-pair phantom tests (H1, H3) are [Pending] hardware.

## 8. Analysis notebooks

```bash
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 0N_*.ipynb
PYTHONPATH=. .venv/bin/python scripts/index_results.py       # outputs/index.csv
```
