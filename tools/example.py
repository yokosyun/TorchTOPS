import os
import argparse
import yaml
from itertools import islice
import matplotlib.pyplot as plt
import torch
import torchvision
from torchtops import profile, filter_modules


def plot_results(x, y, main_title: str, sub_title: str, y_label: str, filename: str):
    fig, ax = plt.subplots()
    plt.suptitle(main_title)
    plt.title(sub_title, fontsize=12)
    ax.bar(x, y)
    fig.autofmt_xdate()
    plt.ylabel(y_label)
    fig.savefig(filename)


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

    if cfg["use_cuda"]:
        model = model.cuda()
        img = img.cuda()

    res = profile(model, img)

    res = filter_modules(res, target_modules=cfg["target_modules"])

    sub_title = "shape=(" + ",".join(map(str, img.shape)) + ")"
    tops_list, layer_names, modules, input_shapes, params_list = zip(
        *sorted(
            zip(
                res["tops_list"],
                res["layer_names"],
                res["modules"],
                res["input_shapes"],
                res["params_list"],
            )
        )
    )

    worst_k = min(len(layer_names), cfg["worst_k"])

    save_path = cfg["save_dir"] + "/" + cfg["model_name"] + ".jpg"

    plot_results(
        x=layer_names[:worst_k],
        y=tops_list[:worst_k],
        main_title=cfg["model_name"],
        sub_title=sub_title,
        y_label="TOPS",
        filename=save_path,
    )

    print(f"saved to {save_path}")

    print("=== low TOPS layers ===")
    for tops, layer_name, module, input_shape, params in islice(
        zip(tops_list, layer_names, modules, input_shapes, params_list), worst_k
    ):
        mega_params = params * 1e-6
        print(
            f"{tops:.3f}  => {layer_name} : {module} : {input_shape} ,params={mega_params:.3f} [M]"
        )
