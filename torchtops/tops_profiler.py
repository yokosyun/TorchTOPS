import math
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
    modules = []
    input_shapes = []
    output_shapes = []
    params_list = []
    activation_list = []
    for layer_name, latency in prof.trace_latency.items():
        flops = flops_dict.get(layer_name, 0)
        if flops > 0:
            tops = flops / latency / 1.0e12
            layer_names.append(layer_name)
            tops_list.append(tops)
            latencies.append(latency)
            modules.append(get_module_by_layer_name(model, layer_name))
            input_shapes.append(prof.trace_input_shape[layer_name])
            output_shapes.append(prof.trace_output_shape[layer_name])
            params_list.append(prof.trace_params[layer_name])

            activation = sum(
                math.prod(output_shape)
                for output_shape in prof.trace_output_shape[layer_name]
            )
            activation_list.append(activation)

    return {
        "layer_names": layer_names,
        "latencies": latencies,
        "tops_list": tops_list,
        "modules": modules,
        "total_flops": total_flops,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "params_list": params_list,
        "activation_list": activation_list,
    }
