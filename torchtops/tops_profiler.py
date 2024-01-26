from typing import Dict, Any
from fvcore.nn import FlopCountAnalysis
from torch import Tensor
from torch import nn
from .latency_profiler import LatencyProfile
from .utils import get_module_by_layer_name


def profile(model: nn.Module, input_data: Tensor) -> Dict[str, Any]:
    flops_counter = FlopCountAnalysis(model, input_data)
    flops_dict = flops_counter.by_module()

    with LatencyProfile(model) as prof:
        model(input_data)

    latency_list = []
    tops_list = []
    layer_name_list = []
    modules = []
    input_shape_list = []
    params_list = []
    read_counts_list = []
    write_counts_list = []
    arithmetric_intensity_list = []
    flops_list = []
    for layer_name, latency in prof.trace_latency.items():
        flops = flops_dict.get(layer_name, 0)
        if flops > 0:
            tops = flops / latency / 1.0e9
            layer_name_list.append(layer_name)
            tops_list.append(tops)
            latency_list.append(latency)
            modules.append(get_module_by_layer_name(model, layer_name))
            input_shape_list.append(prof.trace_input_shape[layer_name])
            params_list.append(prof.trace_params[layer_name])
            read_counts_list.append(prof.trace_read_counts[layer_name])
            write_counts_list.append(prof.trace_write_counts[layer_name])
            arithmetric_intensity = flops / (
                prof.trace_read_counts[layer_name]
                + prof.trace_write_counts[layer_name]
                + prof.trace_params[layer_name]
            )
            arithmetric_intensity_list.append(arithmetric_intensity)
            flops_list.append(flops)

    return {
        "latency_list": latency_list,
        "flops_list": flops_list,
        "tops_list": tops_list,
        "arithmetric_intensity_list": arithmetric_intensity_list,
        "params_list": params_list,
        "read_counts_list": read_counts_list,
        "write_counts_list": write_counts_list,
        "layer_name_list": layer_name_list,
        "module_list": modules,
        "input_shape_list": input_shape_list,
        "total_latency": sum(latency_list),
        "total_flops": sum(flops_list),
        "total_tops": sum(tops_list),
        "total_arithmetric_intensity": sum(arithmetric_intensity_list),
        "total_params": sum(params_list),
        "total_read_counts": sum(read_counts_list),
        "total_write_counts": sum(write_counts_list),
    }
