"""Which ICBHI recordings belong to a split: official 60/40, a patient fold, or all non-ambiguous recordings (LODO)."""
import os
from collections import defaultdict
from typing import NamedTuple


class Selection(NamedTuple):
    filenames: list  # recording stems, sorted
    split: str  # cache and report name of the split
    dropped: set  # patients removed because the official split puts them on both sides (LODO only)


def select_recordings(args, train_flag):
    """Recording stems of the train or test split described by `args` (`holdout_device`, `test_fold`, `data_folder`)."""
    lodo = bool(args.holdout_device)  # leave-one-device-out: one dataset over all cycles, split by index later
    split = "all" if lodo else "train" if train_flag else "test"
    if args.test_fold != "official":
        split = "fold{}_{}".format(args.test_fold, split)
    wav_dir = os.path.join(args.data_folder, "audio_test_data")
    official = dict(
        line.strip().split("\t")
        for line in open(os.path.join(args.data_folder, "official_split.txt")).read().splitlines()
    )
    stems = sorted({f.split(".")[0] for f in os.listdir(wav_dir) if f.endswith((".wav", ".txt"))})
    dropped = set()
    if lodo:  # drop the patients that the official split puts on both sides
        sides = defaultdict(set)
        for f in stems:
            sides[f.split("_")[0]].add(official.get(f))
        dropped = {p for p, s in sides.items() if len(s) > 1 or None in s}
        filenames = [f for f in stems if f.split("_")[0] not in dropped]
    elif args.test_fold == "official":
        filenames = [f for f in stems if official.get(f) == split]
    else:
        fold = dict(line.split() for line in open(os.path.join(args.data_folder, "patient_list_foldwise.txt")).read().splitlines())
        filenames = [f for f in stems if (fold[f.split("_")[0]] == args.test_fold) != train_flag]
    return Selection(filenames, split, dropped)
