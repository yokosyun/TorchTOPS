from typing import Dict, Any, List
from functools import reduce
from itertools import compress
from torch import nn


def get_module_by_layer_name(module: nn.Module, access_string: str) -> nn.Module:
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def filter_modules(
    in_dict: Dict[str, Any], target_modules: List[str]
) -> Dict[str, Any]:
    target_modules = [getattr(nn, target_module) for target_module in target_modules]
    masks = [isinstance(module, tuple(target_modules)) for module in in_dict["modules"]]

    out_dict = in_dict.copy()
    for key, val in in_dict.items():
        if isinstance(val, list):
            out_dict[key] = list(compress(val, masks))

    return out_dict
