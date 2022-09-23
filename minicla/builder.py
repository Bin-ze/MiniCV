from register.build import build_from_cfg
from register.register import Register

MODELS = Register('MODELS')


def build_model(cfg, default_cfg=None):
    model = build_from_cfg(cfg, MODELS, default_cfg)
    return model
