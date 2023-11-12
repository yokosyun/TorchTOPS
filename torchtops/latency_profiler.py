import torch
import functools
from collections import defaultdict, namedtuple
import numpy as np


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


class LatencyProfile(object):
    """Layer by layer profiling latency of PyTorch models"""

    def __init__(self, model, enabled=True, paths=None):
        self._model = model
        self.enabled = enabled
        self.paths = paths

        self.entered = False
        self.exited = False
        self.traces = ()
        self._ids = set()
        self.trace_profile_events = defaultdict(list)
        self.trace_latency = defaultdict(float)
        self.iterations = 10

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("torchprof profiler is not reentrant")
        self.entered = True
        self._forwards = {}  # store the original forward functions
        self.traces = tuple(map(self._hook_trace, walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [path, leaf, module] = trace
        _id = id(module)
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            if _id in self._ids:
                # already wrapped
                return trace
            self._ids.add(_id)
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                latencies = []
                for _ in range(self.iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    res = _forward(*args, **kwargs)
                    end.record()
                    torch.cuda.synchronize()
                    latency = start.elapsed_time(end) / 1000  # miliseconds to seconds
                    latencies.append(latency)

                self.trace_latency[path] = np.median(latencies)
                return res

            module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        [path, leaf, module] = trace
        _id = id(module)
        if _id in self._ids:
            self._ids.discard(_id)
        else:
            return
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            module.forward = self._forwards[path]
