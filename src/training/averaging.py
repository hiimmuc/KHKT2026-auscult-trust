"""Epoch weight averaging, separate from the optimizer's restart weights."""
from contextlib import contextmanager

import torch


class EpochAverager:
    """Average scheduled epochs on CPU; evaluate without changing training state."""

    def __init__(self, start=0, interval=1):
        if start < 0 or interval < 1:
            raise ValueError('SWA requires start >= 0 and interval >= 1')
        self.start, self.interval = start, interval
        self.epochs, self.states = [], []

    @torch.no_grad()
    def update(self, epoch, modules):
        if not self.start or epoch < self.start or (epoch - self.start) % self.interval:
            return False
        if self.epochs and epoch <= self.epochs[-1]:
            raise ValueError('SWA epochs must increase')
        if any(isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
               for module in modules for layer in module.modules()):
            raise ValueError('SWA with BatchNorm needs training-only statistics recalibration')
        if not self.states:
            self.states = [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                           for module in modules]
        else:
            count = len(self.epochs) + 1
            for state, module in zip(self.states, modules):
                for key, value in module.state_dict().items():
                    value = value.detach().cpu()
                    if value.is_floating_point() or value.is_complex():
                        state[key].add_((value - state[key]) / count)
                    else:
                        state[key].copy_(value)
        self.epochs.append(epoch)
        return True

    @contextmanager
    def applied(self, modules):
        """Temporarily apply inference weights; restore even if evaluation fails."""
        if not self.epochs:
            yield
            return
        original = [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                    for module in modules]
        try:
            for module, state in zip(modules, self.states):
                module.load_state_dict(state)
            yield
        finally:
            for module, state in zip(modules, original):
                module.load_state_dict(state)

    def metadata(self):
        return {'start': self.start, 'interval': self.interval, 'epochs': self.epochs.copy(),
                'count': len(self.epochs), 'floating_buffers': 'mean', 'integer_buffers': 'latest'}

    def state_dict(self):
        return {**self.metadata(), 'states': self.states}

    def load_state_dict(self, state):
        if state is None:
            if self.start:
                raise ValueError('Missing SWA state in restart checkpoint; use a new run')
            return
        if (state['start'], state['interval']) != (self.start, self.interval):
            raise ValueError('SWA state schedule differs from this run')
        self.epochs, self.states = state['epochs'], state['states']
