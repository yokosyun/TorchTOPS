from typing import Dict, Any
from fvcore.nn import FlopCountAnalysis
from torch import Tensor
from torch import nn
from .latency_profiler import LatencyProfile
from .utils import get_module_by_layer_name


def profile(model: nn.Module, input_data: Tensor) -> Dict[str, Any]:
    flops_counter = FlopCountAnalysis(model, input_data)
    total_flops = flops_counter.total()
    flops_dict = flops_counter.by_module()

    with LatencyProfile(model) as prof:
        model(input_data)

    latencies = []
    tops_list = []
    layer_names = []
    unsupported_ops_list = []
    modules = []
    for layer_name, latency in prof.trace_latency.items():
        flops = flops_dict.get(layer_name, None)
        if flops == 0 or flops is None:
            unsupported_ops_list.append(layer_name)
            continue

        tops = flops / latency / 1.0e12

        layer_names.append(layer_name)
        tops_list.append(tops)
        latencies.append(latency)
        modules.append(get_module_by_layer_name(model, layer_name))

    return {
        "layer_names": layer_names,
        "latencies": latencies,
        "tops_list": tops_list,
        "unsupported_ops_list": unsupported_ops_list,
        "total_flops": total_flops,
        "modules": modules,
    }
