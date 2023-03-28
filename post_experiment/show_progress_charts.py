#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from typing import Sequence, Tuple, List, Optional, NamedTuple, Union
from cycler import cycler

from misc import ProgressElement
from misc import get_file_name_stat, get_file_name_stat_img

from post_common import NetInfo


def load_results(file_path: str) -> List[ProgressElement]:
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_legend_long(net_info: Union[Tuple, NetInfo]) -> str:
    if not isinstance(net_info, NetInfo):
        net_info = NetInfo(*net_info)

    af_name_ffn = net_info.af_name

    if net_info.af_name_cnn is None:
        af_name_cnn = af_name_ffn
    else:
        af_name_cnn = net_info.af_name_cnn

    if net_info.net_type == "base":
        legend = "{} CNN, {} FFN".format(af_name_cnn, af_name_ffn)
    elif net_info.net_type == "ahaf":
        legend = "{}-like AHAF CNN, {}-like AHAF FFN".format(
            af_name_cnn, af_name_ffn
        )
    elif net_info.net_type == "fuzzy_ffn":
        legend = "{} CNN, {}-like Fuzzy FFN".format(
            af_name_cnn, af_name_ffn
        )
    else:
        raise ValueError("Network type is not supported")

    if net_info.fine_tuned:
        legend = legend + ", fine-tuned"

    return legend


def get_short_af_name(orig: str) -> str:
    if orig == "Tanh":
        return "tanh"
    elif orig == "Sigmoid":
        return "σ-fn"
    else:
        return orig


def get_legend_short(net_info: Union[Tuple, NetInfo]) -> str:
    if not isinstance(net_info, NetInfo):
        net_info = NetInfo(*net_info)

    af_name_ffn = net_info.af_name

    if net_info.af_name_cnn is None:
        af_name_cnn = af_name_ffn
    else:
        af_name_cnn = net_info.af_name_cnn

    af_name_cnn = get_short_af_name(af_name_cnn)
    af_name_ffn = get_short_af_name(af_name_ffn)

    if net_info.net_type == "base":
        legend = "Base, {}, {}".format(af_name_cnn, af_name_ffn)
    elif net_info.net_type == "ahaf":
        legend = "AHAF, {}, {}".format(
            af_name_cnn, af_name_ffn
        )
    elif net_info.net_type == "fuzzy_ffn":
        legend = "Fuzzy, {}, {}".format(
            af_name_cnn, af_name_ffn
        )
    else:
        raise ValueError("Network type is not supported")

    if net_info.fine_tuned:
        legend = legend + ", tuned"

    return legend


def analyze_network(net_info: Tuple):
    file_path = get_file_name_stat(*net_info)
    results = load_results(file_path)
    base_legend = get_legend_short(net_info)

    acc = []
    loss = []

    for r in results:
        acc.append(float(r["test_acc"]) * 100.0)
        loss.append(float(r["train_loss_mean"]))

    return base_legend, acc, loss


def plot_networks(fig, nets: Sequence[Union[Tuple, NetInfo]], bw=False):
    acc_legends = []
    loss_legends = []

    monochrome = (
            cycler('color', ['black', 'grey', 'dimgray'])
            * cycler('linestyle', ['--', ':', '-.', '-'])
            * cycler('marker', ['None'])
    )

    gs = plt.GridSpec(1, 2)

    acc_fig = fig.add_subplot(gs[0, 0])
    #acc_loc = plticker.LinearLocator(numticks=10)
    #acc_fig.yaxis.set_major_locator(acc_loc)
    acc_fig.set_xlabel('epoch')
    acc_fig.set_ylabel('test accuracy, %')
    acc_fig.grid()
    if bw:
        acc_fig.set_prop_cycle(monochrome)

    loss_fig = fig.add_subplot(gs[0, 1])
    #loss_loc = plticker.LinearLocator(numticks=10)
    #loss_fig.yaxis.set_major_locator(loss_loc)
    loss_fig.set_xlabel('epoch')
    loss_fig.set_ylabel('training loss')
    loss_fig.grid()
    if bw:
        loss_fig.set_prop_cycle(monochrome)

    for net in nets:
        try:
            base_legend, acc, loss = analyze_network(net)
        except Exception as e:
            print("Exception: {}, skipped".format(e))
            continue

        n_epochs = len(acc)
        end_ep = net.epoch
        start_ep = end_ep - n_epochs

        x = tuple(range(start_ep, end_ep))

        acc_legends.append(
            base_legend
        )
        loss_legends.append(
            base_legend
        )
        acc_fig.plot(x, acc)
        loss_fig.plot(x, loss)

    acc_fig.legend(acc_legends)
    loss_fig.legend(loss_legends)


def visualize(
        net_group: str, nets: Sequence[Union[Tuple, NetInfo]], base_title=None,
        bw: bool = False
):
    fig = plt.figure(tight_layout=True, figsize=(7, 3.5))
    if base_title is not None:
        title = "{}, test accuracy and training loss".format(base_title)
        fig.suptitle(title)

    plot_networks(fig, nets, bw)

    if nets:
        #plt.show()
        plt.savefig(get_file_name_stat_img(
            net_group, nets[0].epoch, nets[0].patched, nets[0].fine_tuned
        ))


def main():
    nets_vs_ahaf = [
        NetInfo("base", "ReLU", 100, patched=False, fine_tuned=False),
        NetInfo("ahaf", "ReLU", 100, patched=False, fine_tuned=False),
        NetInfo("ahaf", "ReLU", 150, patched=True, fine_tuned=True),
        NetInfo("base", "SiLU", 100, patched=False, fine_tuned=False),
        NetInfo("ahaf", "SiLU", 100, patched=False, fine_tuned=False),
        NetInfo("ahaf", "SiLU", 150, patched=True, fine_tuned=True),
    ]

    visualize("vs_ahaf", nets_vs_ahaf, "Base VS AHAF", bw=False)

    nets_vs_fuzzy = [
        NetInfo("base", "Tanh", 100, patched=False, fine_tuned=False),
        NetInfo("fuzzy_ffn", "Tanh", 100, patched=False, fine_tuned=False),
        NetInfo("fuzzy_ffn", "Tanh", 150, patched=True, fine_tuned=True),
        NetInfo("base", "Sigmoid", 100, patched=False, fine_tuned=False),
        NetInfo("fuzzy_ffn", "Sigmoid", 100, patched=False, fine_tuned=False),
        NetInfo("fuzzy_ffn", "Sigmoid", 150, patched=True, fine_tuned=True)
    ]

    visualize("vs_fuzzy", nets_vs_fuzzy, "Base VS Fuzzy", bw=False)


if __name__ == "__main__":
    main()
