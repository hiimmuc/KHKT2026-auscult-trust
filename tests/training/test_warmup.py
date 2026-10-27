from types import SimpleNamespace

import pytest

from src.training.config import parse_args
from src.training.optim import warmup_learning_rate


@pytest.mark.parametrize('extra, epoch, expected_lr', [
    ([], 6, 0.00055),
    (['--warm_epochs', '5'], 6, 0.123),
    (['--warm_epochs', '5'], 5, 0.00082),
])
def test_warmup_duration_controls_when_learning_rate_is_overridden(extra, epoch, expected_lr):
    args = parse_args(['--warm', '--learning_rate', '.001', '--epochs', '100', *extra])
    optimizer = SimpleNamespace(param_groups=[{'lr': .123}])
    warmup_learning_rate(args, epoch, 0, 10, optimizer)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(expected_lr)


def test_warmup_duration_must_not_be_negative():
    with pytest.raises(SystemExit):
        parse_args(['--warm', '--warm_epochs', '-1'])
