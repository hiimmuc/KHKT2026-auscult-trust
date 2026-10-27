"""Preflight gradient checks reject missing, zero, and nonfinite updates."""
import pytest
import torch
from torch import nn

from scripts.improvement_preflight import check_source_recipe, gradient_check, recipe


def test_gradient_checks_require_real_finite_updates():
    module = nn.Linear(2, 1)
    with pytest.raises(AssertionError, match="lack gradients"):
        gradient_check(module, True)
    module(torch.ones(1, 2)).sum().backward()
    assert gradient_check(module, True)["gradient_l1"] > 0
    for parameter in module.parameters():
        parameter.grad.zero_()
    with pytest.raises(AssertionError, match="zero"):
        gradient_check(module, True)
    module.weight.grad[0, 0] = torch.inf
    with pytest.raises(AssertionError, match="Nonfinite"):
        gradient_check(module, True)


def test_recording_pilot_uses_bounded_bags_and_source_cv():
    args = recipe("recording_head")
    assert args.batch_size == 8 and args.recording_bag_cycles == 4
    assert args.cv_folds == 3 and args.cv_fold == 0 and args.selection == "cv"
    assert args.num_workers == 0 and args.spectrum_correction and args.sc_mode == "static"


def test_shared_source_cache_rejects_different_waveform_recipes():
    anchor, candidate = recipe("llrd"), recipe("band_noise")
    check_source_recipe(anchor, candidate)
    candidate.desired_length += 1
    with pytest.raises(ValueError, match="desired_length"):
        check_source_recipe(anchor, candidate)
