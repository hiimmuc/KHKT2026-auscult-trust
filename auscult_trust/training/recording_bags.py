"""Patient-preserving recording bags built from annotated ICBHI cycle indices."""
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def source_indices(source):
    """Resolve nested Subsets without widening their selected patient pool."""
    indices = np.arange(len(source))
    while isinstance(source, Subset):
        indices = np.asarray(source.indices)[indices]
        source = source.dataset
    return source, indices


def cycle_recordings(dataset):
    from auscult_trust.processing.cycles import read_annotations
    annotations = read_annotations(f"{dataset.args.data_folder}/audio_test_data")
    ids = [stem for stem in dataset.filenames for _ in range(len(annotations[stem]))]
    if len(ids) != len(dataset.labels):
        raise ValueError("Annotation cycles do not match dataset rows")
    return np.asarray(ids)


class RecordingBags(Dataset):
    """One bounded bag per recording, with padded windows excluded by quality mask.

    Labels contain K cycle labels followed by the complete recording's binary label.
    Training draws windows without labels; evaluation spaces them over the recording.
    """
    def __init__(self, source, max_cycles=4, training=True, recording_ids=None):
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        self.dataset, self.indices = source_indices(source)
        self.max_cycles, self.training = max_cycles, training
        ids = cycle_recordings(self.dataset) if recording_ids is None else np.asarray(recording_ids)
        if len(ids) != len(self.dataset):
            raise ValueError("Recording IDs must align with the base dataset")
        selected = set(self.indices.tolist())
        groups = defaultdict(list)
        for i in self.indices:
            groups[str(ids[i])].append(int(i))
        self.recording_ids = np.asarray(sorted(groups))
        self.groups = [np.asarray(groups[key]) for key in self.recording_ids]
        for key, rows in zip(self.recording_ids, self.groups):
            if any(int(i) not in selected for i in np.flatnonzero(ids == key)):
                raise ValueError(f"Partial recording in selected split: {key}")
            if len(set(np.asarray(self.dataset.patients)[rows])) != 1:
                raise ValueError(f"Recording spans multiple patients: {key}")
        self.patients = np.asarray([self.dataset.patients[r[0]] for r in self.groups])
        self.labels = np.asarray([int(np.any(np.asarray(self.dataset.labels)[r] != 0)) for r in self.groups])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        rows = self.groups[index]
        n = min(len(rows), self.max_cycles)
        chosen = (np.random.choice(rows, n, replace=False) if self.training else
                  rows[np.linspace(0, len(rows) - 1, n).round().astype(int)])
        values = [self.dataset[int(i)] for i in chosen]
        images = torch.stack([torch.as_tensor(v[0]) for v in values])
        padded = torch.zeros((self.max_cycles, *images.shape[1:]), dtype=images.dtype)
        padded[:n] = images
        labels = torch.zeros(self.max_cycles + 1, dtype=torch.long)
        labels[:n] = torch.tensor([v[1] for v in values])
        labels[-1] = int(self.labels[index])
        mask = torch.arange(self.max_cycles) < n
        result = padded, labels, values[0][2], int(self.patients[index]), mask
        if len(values[0]) > 5:
            targets = torch.zeros(self.max_cycles, len(values[0][5]))
            targets[:n] = torch.stack([torch.as_tensor(v[5]) for v in values])
            return (*result, targets)
        return result


def wrap_training_dataset(source, args):
    return RecordingBags(source, max_cycles=args.recording_bag_cycles, training=True)


@torch.no_grad()
def predict_recording(model, classifier, images, batch_size=16):
    """Run max-MIL binary screening over all windows, with bounded encoder memory."""
    if len(images) == 0 or batch_size < 1:
        raise ValueError("A recording needs windows and a positive batch size")
    head = getattr(classifier, "module", classifier)
    device = next(head.parameters()).device
    maximum = None
    for start in range(0, len(images), batch_size):
        features = model(torch.as_tensor(images[start:start + batch_size], device=device).float())
        logits = head.recording_logits(features.float()[None])
        score = logits[0, 1] - logits[0, 0]
        maximum = score if maximum is None else torch.maximum(maximum, score)
    return torch.stack([torch.zeros_like(maximum), maximum]).softmax(0)


@torch.no_grad()
def evaluate_recordings(source, model, classifier, args, excluded_patients=()):
    """Source recording validation; rejects any overlap with training patients."""
    from .reporting import per_device_report
    model.eval(), classifier.eval()
    bags = RecordingBags(source, getattr(args, "recording_eval_cycles", 8), training=False)
    if set(map(str, bags.patients)) & set(map(str, excluded_patients)):
        raise ValueError("Recording validation patients overlap training patients")
    probabilities, devices = [], []
    for i in range(len(bags)):
        images, _, device, _, mask, *_ = bags[i]
        head = getattr(classifier, "module", classifier)
        if callable(getattr(head, "recording_logits", None)):
            probability = predict_recording(model, classifier, images[mask])
        else:
            # Explicit cycle baseline: max abnormal probability over the same source windows.
            from .methods import get_method
            target = next(head.parameters()).device
            logits = classifier(model(images[mask].to(target)))
            cycle_probs = get_method(args.method).probabilities(logits)
            abnormal = (1 - cycle_probs[:, 0]).max()
            probability = torch.stack([1 - abnormal, abnormal])
        probabilities.append(probability.cpu().numpy())
        devices.append(device)
    probabilities = np.asarray(probabilities)
    predictions, devices = probabilities.argmax(1), np.asarray(devices)
    return (per_device_report(bags.labels, predictions, devices, 2), predictions, probabilities,
            bags.labels, devices, bags.recording_ids, bags.patients)
