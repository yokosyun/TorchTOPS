import functools
from collections import defaultdict, namedtuple
import numpy as np
import torch
from torch import Tensor, nn
from typing import List, Dict, Union, Tuple, Any
from fvcore.nn import FlopCountAnalysis

from torchtops.utils import get_module_by_layer_name

Trace = namedtuple("Trace", ["path", "leaf", "module"])


def walk_modules(module, name="", path=""):
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    named_children = list(module.named_children())
    if path != "":
        path = path + "."

    path = path + name
    yield Trace(path, len(named_children) == 0, module)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, path=path)


def get_shape_of_tensor(inputs: Union[Tensor, List, Tuple, Dict]) -> List[List[int]]:
    if isinstance(inputs, Tensor):
        return [list(inputs.shape)]
    elif isinstance(inputs, List) or isinstance(inputs, Tuple):
        return [get_shape_of_tensor(input) for input in inputs]
    elif isinstance(inputs, Dict):
        return [get_shape_of_tensor(input) for input in inputs.values()]
    else:
        print("ignore dtype =", type(inputs))
        return []


def numel(inputs: Union[Tensor, List, Tuple, Dict]) -> int:
    if isinstance(inputs, Tensor):
        return torch.numel(inputs)
    elif isinstance(inputs, List) or isinstance(inputs, Tuple):
        return sum([numel(input) for input in inputs])
    elif isinstance(inputs, Dict):
        return sum([numel(input) for input in inputs.values()])
    else:
        print("ignore dtype =", type(inputs))
        return 0


class LayerProfiler(object):
    """Layer by layer profiling latency of PyTorch models"""

    def __init__(self, model, enabled=True):
        self._model = model
        self.enabled = enabled
        self.traces = ()
        self._ids = set()
        self.trace_latency = defaultdict(float)
        self.trace_input_shape = defaultdict(list)
        self.trace_output_shape = defaultdict(list)
        self.trace_params = defaultdict(int)
        self.trace_read_counts = defaultdict(int)
        self.trace_write_counts = defaultdict(int)
        self.iterations = 10

    def __enter__(self):
        if not self.enabled:
            return self
        self._forwards = {}  # store the original forward functions
        self.traces = tuple(map(self._hook_trace, walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [path, leaf, module] = trace
        _id = id(module)
        if leaf:
            if _id in self._ids:
                # already wrapped
                return trace
            self._ids.add(_id)
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                latency_list = []
                for _ in range(self.iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    results = _forward(*args, **kwargs)
                    end.record()
                    torch.cuda.synchronize()
                    latency = start.elapsed_time(end)  # miliseconds
                    latency_list.append(latency)

                params = sum(x.numel() for x in module.parameters())
                self.trace_params[path] = params
                self.trace_latency[path] = np.median(latency_list)
                self.trace_input_shape[path] = get_shape_of_tensor(
                    args
                ) + get_shape_of_tensor(kwargs)
                self.trace_read_counts[path] = numel(args) + numel(kwargs)
                self.trace_write_counts[path] = numel(results)

                return results

            module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        [path, leaf, module] = trace
        _id = id(module)
        if _id in self._ids:
            self._ids.discard(_id)
        else:
            return
        if leaf:
            module.forward = self._forwards[path]


def profile(model: nn.Module, input_data: Tensor) -> Dict[str, Any]:
    flops_counter = FlopCountAnalysis(model, input_data)
    flops_dict = flops_counter.by_module()

    with LayerProfiler(model) as prof:
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