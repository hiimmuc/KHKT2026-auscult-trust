"""Training objectives, selected by the stable method names stored in experiment configs."""
from importlib import import_module

_MODULES = {"ce": "ce", "patchmix": "patchmix", "patchmix_cl": "patchmix_cl", "ce_pafa": "pafa",
            "ce_pafa_attn": "pafa_attn", "ce_pafa_class_aware": "class_aware_pafa",
            "ce_recording_pafa": "recording_pafa", "ce_device_pafa": "device_pafa", "ce_pafa_events": "event_pafa",
            "sg_scl": "sg_scl", "qlung": "qlung", "add_rsc": "add_rsc", "pcmcl": "pcmcl"}
NAMES = tuple(_MODULES)


def get_method(name):
    try:
        module = _MODULES[name]
    except KeyError:
        raise ValueError(f"Unknown training method: {name}; choose from {', '.join(NAMES)}") from None
    return import_module(f"{__name__}.{module}").Method()
