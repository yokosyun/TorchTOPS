from typing import Dict, Any, List
from functools import reduce
from itertools import compress
from torch import nn, Tensor
import torch
import statistics
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_module_by_layer_name(module: nn.Module, access_string: str) -> nn.Module:
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def filter_modules(
    in_dict: Dict[str, Any], target_modules: List[str]
) -> Dict[str, Any]:
    target_modules = [getattr(nn, target_module) for target_module in target_modules]
    masks = [
        isinstance(module, tuple(target_modules)) for module in in_dict["module_list"]
    ]

    out_dict = in_dict.copy()
    for key, val in in_dict.items():
        if isinstance(val, list):
            out_dict[key] = list(compress(val, masks))

    return out_dict


def get_latency(
    model: nn.Module,
    input: Tensor,
    iterations: int = 50,
) -> float:
    latency_list = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(input)
        end.record()
        torch.cuda.synchronize()
        latency_list.append(start.elapsed_time(end))
    return statistics.median(latency_list)  # [ms]


def plot_results(res: Dict[str, Any], save_path: str) -> None:
    x = res["layer_name_list"]

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(
        7, sharex="col", figsize=(10, 16), constrained_layout=True
    )
    ax1.bar(x, np.array(res["latency_list"]) / res["total_latency"] * 100)
    ax1.set_ylabel("latency[%]")

    ax2.bar(x, np.array(res["flops_list"]) / res["total_flops"] * 100)
    ax2.set_ylabel("FLOPs[%]")

    ax3.bar(x, np.array(res["tops_list"]) / res["total_tops"] * 100)
    ax3.set_ylabel("TOPS[%]")

    ax4.bar(
        x,
        np.array(res["arithmetric_intensity_list"])
        / res["total_arithmetric_intensity"]
        * 100,
    )
    ax4.set_ylabel("arith_intensity[%]")

    ax5.bar(x, np.array(res["params_list"]) / res["total_params"] * 100)
    ax5.set_ylabel("params[%]")

    ax6.bar(x, np.array(res["read_counts_list"]) / res["total_read_counts"] * 100)
    ax6.set_ylabel("read_counts[%]")

    ax7.bar(
        x,
        np.array(res["write_counts_list"]) / res["total_write_counts"] * 100,
    )
    ax7.set_ylabel("write_counts[%]")

    plt.suptitle(Path(save_path).stem)
    fig.autofmt_xdate()
    fig.savefig(save_path)
