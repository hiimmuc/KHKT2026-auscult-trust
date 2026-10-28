"""Shared paths, loaders and plotting defaults for the analysis notebooks.

Notebooks only read finished artefacts (campaign outputs, KAUH and screening summaries, replay tables); nothing here trains or
predicts. Figures go to `outputs/figures/<topic>/`, tables to `outputs/analysis/<topic>/`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "data"
CAMPAIGNS = REPO / "build" / "campaigns"
ICBHI = DATA / "processed" / "patchmix_icbhi"
CLASSES = ["normal", "crackle", "wheeze", "both"]
DEVICES = ["Meditron", "LittC2SE", "Litt3200", "AKGC417L"]  # device ids 0..3 in exports
LODO = ["akgc417l", "litt3200", "littc2se", "meditron"]
LODO_NAME = dict(zip(LODO, ["AKGC417L", "Litt3200", "LittC2SE", "Meditron"]))

# arm -> (campaign, run, cell, fixed epoch). Official and LODO finals; LODO cells append "+lodo_<device>".
ARMS = {
    "anchor": ("followup-20261026", "followup-20261026-final", "gain6-cv66", 66),
    "swa": ("target-20261027", "target-20261027-final", "swa", 43),
    "multiscale": ("target-20261027", "target-20261027-final", "multiscale", 68),
    "pcsl5_control": ("target-20261027", "target-20261027-final", "pcsl5_control", 78),
    "class_aware_light": ("target-20261027", "target-20261027-final", "class_aware_light", 4),
    "lora": ("target-20261027", "target-20261027-final", "lora", 83),
    "lora_class_aware": ("target-20261027", "target-20261027-final", "lora_class_aware", 43),
    "kd_g4_k1": ("distill-20261028", "distill-20261028-final", "kd_g4_k1", 66),
    "kd_g4_k2": ("distill-20261028", "distill-20261028-final", "kd_g4_k2", 66),
    "kd_g7_k2": ("distill-20261028", "distill-20261028-final", "kd_g7_k2", 66),
}
G4 = ["anchor", "pcsl5_control", "class_aware_light", "multiscale"]
PALETTE = {"anchor": "#4C72B0", "kd_g4_k1": "#C44E52", "kd_g4_k2": "#DD8452", "kd_g7_k2": "#937860", "swa": "#55A868",
           "multiscale": "#8172B3", "pcsl5_control": "#64B5CD", "class_aware_light": "#CCB974", "lora": "#8C8C8C",
           "lora_class_aware": "#BBBBBB", "G4 ensemble": "#222222", "frozen": "#000000"}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def out_dirs(topic):
    """(figure dir, table dir) for one notebook topic, created on first use."""
    f, t = REPO / "outputs" / "figures" / topic, REPO / "outputs" / "analysis" / topic
    f.mkdir(parents=True, exist_ok=True)
    t.mkdir(parents=True, exist_ok=True)
    return f, t


def unit(arm, cohort="official", seed=0):
    camp, run, cell, _ = ARMS[arm]
    suffix = "" if cohort == "official" else "+lodo_" + cohort
    return CAMPAIGNS / camp / "outputs" / "train" / run / (cell + suffix) / f"seed{seed}"


def exists(arm, cohort="official", seed=0):
    return (unit(arm, cohort, seed) / "test_preds.npz").exists()


def report(arm, cohort="official", seed=0):
    return json.loads((unit(arm, cohort, seed) / "report.json").read_text())


def preds(arm, cohort="official", seed=0):
    """Last-epoch softmax (n, 4), labels, device ids and patients of one unit's test set."""
    z = np.load(unit(arm, cohort, seed) / "test_preds.npz", allow_pickle=True)
    return z["probs"][-1].astype(float), z["labels"], z["device"], z["patient"]


def icbhi_score(p, y):
    """(Sp, Se, Score) in % on the 4-class ICBHI definition."""
    pr = p.argmax(1)
    sp = (pr[y == 0] == 0).mean()
    se = (pr[y > 0] == y[y > 0]).mean()
    return 100 * sp, 100 * se, 50 * (sp + se)


def history(arm, cohort="official", seed=0):
    """Per-epoch training loss / accuracy and test Score of a final unit."""
    h = report(arm, cohort, seed)["history"]
    return pd.DataFrame([{k: e.get(k) for k in ("epoch", "train_loss", "train_acc", "val_score", "test_score")} for e in h])


def ensemble(arms, cohort, seed):
    ps = [preds(a, cohort, seed) for a in arms]
    return np.mean([p[0] for p in ps], 0), ps[0][1], ps[0][2], ps[0][3]


def savefig(fig, folder, name):
    fig.savefig(folder / f"{name}.png")
    return folder / f"{name}.png"
