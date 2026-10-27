"""Patch-Mix contrastive objective, including Lung-SRAD's dual-axis variant."""
from ..upstream import load_reference
from .base import TrainingMethod
from .ce import classify


class Method(TrainingMethod):
    supports_consistency = True
    supports_dual_mix = True

    def projector(self, model, args):
        return load_reference("patch-mix_contrastive_learning", "models/projector.py").Projector(
            model.final_feat_dim, args.proj_dim)

    def losses(self, model, args):
        loss = load_reference("patch-mix_contrastive_learning", "method/patchmix_cl.py").PatchMixConLoss(
            temperature=args.temperature)
        return [*super().losses(model, args), loss]

    def step(self, context, batch):
        c, a = context, context.args
        features, result = classify(c, batch)
        if a.target_type == "grad_block":
            target = features.detach().clone()
        elif a.target_type == "grad_flow":
            target = features
        elif a.target_type == "project_block":
            target = c.projector(features).detach().clone()
        else:
            target = c.projector(features)
        if a.cl_mode == "dual":
            for axis in ("time", "freq"):
                mixed, _, labels_b, lam, index = c.encode(batch.images, y=batch.labels, patch_mix=True, mix_type=axis)
                result.loss += a.alpha * c.criteria[1](target, c.projector(mixed.detach()), batch.labels, labels_b, lam, index, a)
        else:
            mixed, _, labels_b, lam, index = c.encode(batch.images, y=batch.labels, patch_mix=True, time_domain=a.time_domain)
            result.loss += a.alpha * c.criteria[1](target, c.projector(mixed), batch.labels, labels_b, lam, index, a)
        return result
