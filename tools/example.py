import os
import argparse
import yaml
import torch
import torchvision
from torchtops import profile, filter_modules
from torchtops.utils import get_latency, plot_results


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        type=str,
        default="configs/example.yaml",
        help="path to yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_argparse()

    with open(args.yaml) as file:
        cfg = yaml.safe_load(file)

    os.makedirs(cfg["save_dir"], exist_ok=True)

    model = torchvision.models.get_model(cfg["model_name"], **cfg["model_cfg"])

    model = model.eval()

    img = torch.rand(cfg["input_shape"])

    if cfg["use_half"]:
        model = model.to(torch.float16)
        img = img.to(torch.float16)

    if cfg["use_cuda"]:
        model = model.cuda()
        img = img.cuda()

    # warmup
    get_latency(model, img)

    with torch.no_grad():
        res = profile(model, img)

        res = filter_modules(res, target_modules=cfg["target_modules"])

        (
            res["latencies"],
            res["tops_list"],
            res["layer_names"],
            res["modules"],
            res["input_shapes"],
            res["params_list"],
            res["read_counts_list"],
            res["write_counts_list"],
            res["arithmetric_intensity_list"],
            res["flops_list"],
        ) = zip(
            *sorted(
                zip(
                    res["latencies"],
                    res["tops_list"],
                    res["layer_names"],
                    res["modules"],
                    res["input_shapes"],
                    res["params_list"],
                    res["read_counts_list"],
                    res["write_counts_list"],
                    res["arithmetric_intensity_list"],
                    res["flops_list"],
                ),
                reverse=True,
            )
        )

        top_k = min(len(res["layer_names"]), cfg["top_k"])
        res["latencies"] = res["latencies"][:top_k]
        res["tops_list"] = res["tops_list"][:top_k]
        res["layer_names"] = res["layer_names"][:top_k]
        res["modules"] = res["modules"][:top_k]
        res["input_shapes"] = res["input_shapes"][:top_k]
        res["params_list"] = res["params_list"][:top_k]
        res["read_counts_list"] = res["read_counts_list"][:top_k]
        res["write_counts_list"] = res["write_counts_list"][:top_k]
        res["arithmetric_intensity_list"] = res["arithmetric_intensity_list"][:top_k]
        res["flops_list"] = res["flops_list"][:top_k]

        save_path = os.path.join(cfg["save_dir"], cfg["model_name"] + ".jpg")
        plot_results(res, save_path)
        print(f"saved to {save_path}")

        print("=== top_k slow layers ===")
        for (
            latency,
            tops,
            layer_name,
            module,
            input_shape,
            params,
            read_counts,
            write_counts,
            arithmetric_intensity,
            flops,
        ) in zip(
            res["latencies"],
            res["tops_list"],
            res["layer_names"],
            res["modules"],
            res["input_shapes"],
            res["params_list"],
            res["read_counts_list"],
            res["write_counts_list"],
            res["arithmetric_intensity_list"],
            res["flops_list"],
        ):
            mega_params = params * 1e-6
            print(
                f"{latency:.3f} [ms] {tops:.3f}  => {layer_name} : {module} : {input_shape}"
            )
            print(
                f"params={mega_params:.3f} [M], flops={flops}, read_couts={read_counts}, write_counts={write_counts}, arithmetric_intensity={arithmetric_intensity}"
            )
            print()

        latency = get_latency(model, img)
        latency_overhead = (res["total_latency"] - latency) / latency * 100
        print(
            f"latency = {latency:.3f} [ms], profiler latency overhead={latency_overhead:.2f} [%]"
        )
