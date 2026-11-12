"""One training epoch and one evaluation pass."""

import sys
import time
from copy import deepcopy

import numpy as np
import torch

from .augment import freq_mixstyle
from .methods import get_method
from .methods.base import Batch, StepContext
from .optim import AverageMeter, accuracy, update_moving_average, warmup_learning_rate
from .reporting import per_device_report
def train_epoch(loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler):
    """Optimize one epoch with the selected method; return mean loss and accuracy."""
    method = get_method(args.method)
    model.train(), classifier.train(), projector.train()
    if args.freeze_encoder:
        model.eval()  # frozen encoder: no batch-norm statistics update, no dropout
    batch_time, losses, top1 = AverageMeter(), AverageMeter(), AverageMeter()
    cons_loss, cons_agree = AverageMeter(), AverageMeter()
    cons_w = args.consistency_weight * min(1.0, epoch / max(args.consistency_ramp, 1))
    end = time.time()
    for idx, values in enumerate(loader):
        images, labels, devices, patients, quality, *extra = values
        if (
            args.ma_update
        ):  # state before the step, for the moving average (a frozen encoder does not change)
            with torch.no_grad():
                ma_state = [
                    None if m is model and args.freeze_encoder and not method.trainable_frontend else deepcopy(m.state_dict())
                    for m in (model, classifier, projector)
                ]
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        if args.freq_mixstyle > 0:
            images = freq_mixstyle(images, p=args.freq_mixstyle)
        warmup_learning_rate(args, epoch, idx, len(loader), optimizer)

        batch = Batch(images, labels, devices, patients, quality,
                      extra[0].to(images.device, non_blocking=True) if extra else None)
        context = StepContext(model, classifier, projector, criterion, args, epoch, idx, len(loader),
                              method.trainable_frontend)
        with torch.cuda.amp.autocast():
            result = method.step(context, batch)
        output, loss, labels = result.logits, result.loss, result.labels
        if result.consistency is not None:
            cons_loss.update(result.consistency[0], bsz)
            cons_agree.update(result.consistency[1], bsz)

        losses.update(loss.item(), bsz)
        top1.update(accuracy(output[:bsz], labels), bsz)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                if ma_state[0] is not None:
                    update_moving_average(args.ma_beta, model, ma_state[0])
                update_moving_average(args.ma_beta, classifier, ma_state[1])
                update_moving_average(args.ma_beta, projector, ma_state[2])

        if (idx + 1) % args.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\tBT {bt.val:.3f} ({bt.avg:.3f})\tloss {l.val:.3f} ({l.avg:.3f})\tAcc@1 {a.val:.3f} ({a.avg:.3f})".format(
                    epoch, idx + 1, len(loader), bt=batch_time, l=losses, a=top1
                )
                + (
                    "\tloss_cons {c.val:.4f} ({c.avg:.4f})\tagree {g.avg:.1f} (w {w:.2f})".format(c=cons_loss, g=cons_agree, w=cons_w)
                    if args.consistency_weight > 0
                    else ""
                )
            )
            sys.stdout.flush()
    return losses.avg, top1.avg


@torch.no_grad()
def evaluate(loader, model, classifier, args):
    """Run model and classifier on a loader.

    Returns:
        Tuple (per-device report, preds, softmax probs, labels, device ids).
    """
    model.eval(), classifier.eval()
    method = get_method(args.method)
    preds, probs, ys, devs = [], [], [], []
    for images, labels, devices, *_ in loader:
        with torch.cuda.amp.autocast():
            output = classifier(model(images.cuda(non_blocking=True)))
        p = method.probabilities(output)
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        ys.append(labels.numpy())
        devs.append(devices.numpy())
    preds, probs, ys, devs = (np.concatenate(x) for x in (preds, probs, ys, devs))
    return per_device_report(ys, preds, devs, args.n_cls), preds, probs, ys, devs
