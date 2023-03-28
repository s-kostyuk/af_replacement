#!/usr/bin/env python3

import csv

from typing import Sequence, Tuple, List, NamedTuple, Union, Iterable, Generator
from decimal import Decimal

from misc import ProgressElement
from misc import get_file_name_stat, get_file_name_stat_table

from post_common import NetInfo


def load_results(file_path: str) -> List[ProgressElement]:
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze_network(net_info: NetInfo) -> Tuple[int, float, float]:
    """
    TBD

    :param net_info: TBD
    :return: best_epoch, best_accuracy, best_duration_m
    """
    file_path = get_file_name_stat(*net_info)
    results = load_results(file_path)

    max_acc = -1.0
    max_pos = -1
    pos = 0
    total_duration = 0
    best_duration = -1.0
    pos_offset = net_info.epoch - len(results)

    for el in results:
        pos += 1
        acc = Decimal(el["test_acc"]) * 100
        total_duration += Decimal(el["duration"])

        if acc > max_acc:
            max_acc = acc
            max_pos = pos + pos_offset
            best_duration = total_duration

    return max_pos, max_acc, round(best_duration / 60, 2)


class SummaryItem(NamedTuple):
    net_type: str
    af_cnn: str
    af_ffn: str
    best_acc: float
    best_ep: int
    best_duration_m: float
    tuned: bool


def gather_results(nets: Sequence[Union[Tuple, NetInfo]]) -> List[SummaryItem]:
    results = []

    for net in nets:
        if not isinstance(net, NetInfo):
            net = NetInfo(*net)
        try:
            best_ep, best_acc, duration = analyze_network(net)
        except Exception as e:
            print("Exception: {}, skipped".format(e))
            continue

        net_af_cnn = net.af_name_cnn if net.af_name_cnn else net.af_name
        net_af_ffn = net.af_name

        results.append(
            SummaryItem(net.net_type, net_af_cnn, net_af_ffn, best_acc, best_ep,
                        duration, net.fine_tuned)
        )

    return results


def prettify_net_type_short(net_type: str, fine_tuned: bool = False) -> str:
    if net_type == "base":
        net_type = "Base"
    elif net_type == "ahaf":
        net_type = "AHAF"
    elif net_type == "fuzzy_ffn":
        net_type = "Fuzzy"
    else:
        raise ValueError("Network type is not supported")

    if fine_tuned:
        net_type = net_type + " tuned"

    return net_type


def prettify_net_type_long(net_type: str, fine_tuned: bool = False) -> str:
    if net_type == "base":
        net_type = "Base KerasNet"
    elif net_type == "ahaf":
        net_type = "KerasNet w/ AHAF"
    elif net_type == "fuzzy_ffn":
        net_type = "KerasNet w/ Fuzzy FFN"
    else:
        raise ValueError("Network type is not supported")

    if fine_tuned:
        net_type = net_type + " fine-tuned"

    return net_type


def prettify_result(el: SummaryItem) -> Tuple:
    net_type = prettify_net_type_short(el.net_type, el.tuned)

    return (
        net_type, el.af_cnn, el.af_ffn, el.best_acc, el.best_ep,
        el.best_duration_m
    )


def prettify_results(
        results: Sequence[SummaryItem]
) -> Generator[Tuple, None, None]:
    for el in results:
        yield prettify_result(el)


def save_results_as_csv(results: List[SummaryItem], path: str):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ("Type", "CNN AF", "FFN AF", "Accuracy, %", "Epoch", "Time, m")
        )
        writer.writerows(prettify_results(results))


def summarize(
        net_group: str, nets: Sequence[Union[Tuple, NetInfo]]
):
    results = gather_results(nets)

    if nets:
        save_results_as_csv(results, get_file_name_stat_table(
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

    summarize("vs_ahaf", nets_vs_ahaf)

    nets_vs_fuzzy = [
        NetInfo("base", "Tanh", 100, patched=False, fine_tuned=False),
        NetInfo("fuzzy_ffn", "Tanh", 100, patched=False, fine_tuned=False),
        NetInfo("fuzzy_ffn", "Tanh", 150, patched=True, fine_tuned=True),
        NetInfo("base", "Sigmoid", 100, patched=False, fine_tuned=False),
        NetInfo("fuzzy_ffn", "Sigmoid", 100, patched=False, fine_tuned=False),
        NetInfo("fuzzy_ffn", "Sigmoid", 150, patched=True, fine_tuned=True),
    ]

    summarize("vs_fuzzy", nets_vs_fuzzy)


if __name__ == "__main__":
    main()
