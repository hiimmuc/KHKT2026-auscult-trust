"""QLung angular classification with a quality-adaptive margin."""
from torch.nn import functional as F

from auscult_trust.models.heads import QLungHead
from auscult_trust.models.upstream import load_reference
from .base import StepResult, TrainingMethod


class Method(QLungHead, TrainingMethod):
    def losses(self, model, args):
        reference = load_reference("QLung", "method/loss.py")
        return [*super().losses(model, args), reference.AngularMarginLoss(model.final_feat_dim, args.n_cls, s=15, m=0.5,
                gamma=args.qlung_gamma, m_target=args.qlung_m_target, class_nums=args.class_nums)]

    def step(self, context, batch):
        c = context
        features = F.normalize(c.encode(batch.images))
        output = c.classifier(features)
        margin = c.criteria[1](features, batch.labels, batch.quality.to(batch.images.device, non_blocking=True))
        loss = c.criteria[0](output.double(), batch.labels) + c.args.alpha * c.criteria[0](margin, batch.labels)
        return StepResult(output, loss, batch.labels)

    def quality_scores(self, dataset, annotations, wav_dir, cache_dir):
        """QLung quality score of every raw cycle (before any SC), cached per split and cycle settings."""
        import os
        import librosa
        import numpy as np
        from scipy.stats import entropy
        from auscult_trust.processing.cycles import individual_cycles
        q = load_reference("QLung", "util/get_acoustic_features.py")

        a = dataset.args
        path = cache_dir / "quality_{}_{}cls_{}s_{}_{}hz_{}_{}.npy".format(
            dataset.split, a.n_cls, a.desired_length, a.pad_types, a.sample_rate, a.quality_alpha, a.quality_beta
        )
        if path.exists():
            out = np.load(path)
        else:
            out = []
            for f in dataset.filenames:
                for wave, _ in individual_cycles(annotations[f], wav_dir, f, a):
                    y = wave.numpy()[0]
                    e, r = q.compute_spectral_entropy(y), q.compute_rms_energy(y)
                    if not np.isfinite(e):  # digitally silent frames give 0/0 in the reference: average over frames with energy
                        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
                        S = S[:, S.sum(0) > 0]
                        e = float(np.mean(entropy(S / S.sum(0), axis=0, base=2)) / np.log2(S.shape[0])) if S.size else 1.0
                    out.append(q.compute_quality_score_entropy_rms(e, r, alpha=a.quality_alpha, beta=a.quality_beta))
            out = np.asarray(out, np.float32)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name("{}.{}.tmp.npy".format(path.stem, os.getpid()))
            np.save(tmp, out)
            tmp.replace(path)
        assert len(out) == len(dataset.labels)
        return out
