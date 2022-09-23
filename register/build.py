#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict
import inspect

from .register import Register


def build_from_cfg(cfg: Dict, registry: Register, default_args: Dict = None):
    if not isinstance(cfg, Dict):
        raise ValueError("cfg should be dict, but got {}".format(type(cfg)))

    if not isinstance(registry, Register):
        raise ValueError("registry should be Register, but got {}".format(type(registry)))

    if not (isinstance(default_args, Dict) or default_args is None):
        raise ValueError("default_args should be dict or None, but got {}".format(type(default_args)))

    if "object" not in cfg:
        raise KeyError("cfg should contain a builtin key: `object`")

    # extend cfg
    cfg_new = cfg.copy()
    if default_args is not None:
        cfg_new.update(default_args)

    object_value = cfg_new.pop("object")
    if isinstance(object_value, str):
        object_obj = registry.get(object_value)
        if object_obj is None:
            raise KeyError("{} hasn't registered in Register: {}".format(object_value, registry.name))
    elif inspect.isclass(object_value):
        object_obj = object_value
    else:
        raise ValueError("unsupport object: {}".format(object_value))

    return object_obj(**cfg_new)
