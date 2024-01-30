import functools
from collections import defaultdict, namedtuple
import numpy as np
import torch
from torch import Tensor, nn
from typing import List, Dict, Union, Tuple, Any
from fvcore.nn import FlopCountAnalysis

from torchtops.utils import get_module_by_layer_name

Trace = namedtuple("Trace", ["path", "leaf", "module"])


def walk_modules(module, name="", path="", target_modules=[]):
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    named_children = list(module.named_children())
    if path != "":
        path = path + "."

    path = path + name

    # Stop tracing at target modules
    if module.__class__.__name__ in target_modules:
        yield Trace(path, True, module)
    # Trace until no children nn.Module
    else:
        yield Trace(path, len(named_children) == 0, module)
        # recursively walk into all submodules
        for name, child_module in named_children:
            yield from walk_modules(
                child_module, name=name, path=path, target_modules=target_modules
            )


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

    def __init__(
        self,
        model,
        target_modules: List[str] = [],
        profile_children: bool = False,
    ):
        self._model = model
        self.traces = ()
        self._ids = set()
        self.trace_latency = defaultdict(float)
        self.trace_input_shape = defaultdict(list)
        self.trace_output_shape = defaultdict(list)
        self.trace_params = defaultdict(int)
        self.trace_read_counts = defaultdict(int)
        self.trace_write_counts = defaultdict(int)
        self.iterations = 10
        self.target_modules = target_modules
        self.profile_children = profile_children

    def __enter__(self):
        self._forwards = {}  # store the original forward functions
        self.traces = tuple(
            map(
                self._hook_trace,
                walk_modules(self._model, target_modules=self.target_modules),
            )
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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

            if self.profile_children:

                def get_io(module, input, output):
                    self.trace_read_counts[path] += numel(input)
                    self.trace_write_counts[path] += numel(output)

                def trace_forward(module):
                    if len(list(module.named_children())) == 0:
                        module.register_forward_hook(get_io)
                    else:
                        for name, child_module in module.named_children():
                            trace_forward(child_module)

                trace_forward(module)
            else:

                @functools.wraps(_forward)
                def wrap_forward(*args, **kwargs):
                    results = _forward(*args, **kwargs)
                    latency_list = []
                    for _ in range(self.iterations):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        _forward(*args, **kwargs)
                        end.record()
                        torch.cuda.synchronize()
                        latency = start.elapsed_time(end)  # miliseconds
                        latency_list.append(latency)

                    self.trace_latency[path] = np.median(latency_list)

                    params = sum(x.numel() for x in module.parameters())
                    self.trace_params[path] = params
                    self.trace_input_shape[path] = get_shape_of_tensor(
                        args
                    ) + get_shape_of_tensor(kwargs)

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


def profile(
    model: nn.Module, input_data: Tensor, target_modules: List[str] = []
) -> Dict[str, Any]:
    flops_counter = FlopCountAnalysis(model, input_data)
    flops_dict = flops_counter.by_module()

    with LayerProfiler(
        model, target_modules=target_modules, profile_children=False
    ) as prof:
        model(input_data)

    with LayerProfiler(
        model, target_modules=target_modules, profile_children=True
    ) as prof_children:
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
            read_counts_list.append(prof_children.trace_read_counts[layer_name])
            write_counts_list.append(prof_children.trace_write_counts[layer_name])
            arithmetric_intensity = flops / (
                prof_children.trace_read_counts[layer_name]
                + prof_children.trace_write_counts[layer_name]
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
