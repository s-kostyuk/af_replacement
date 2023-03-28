#!/usr/bin/env python3

import torchinfo

from nns_aaf import KerasNetAaf, AfDefinition


def print_act_functions(net: KerasNetAaf):
    print(net.activations)


def main():
    nn_defs = [
        (None, None),  # expected: all ReLU
        (AfDefinition("SiLU"), AfDefinition("SiLU")),  # expected: all SiLU

        # expected: SiLU in CNN, adaptive SiLU in FFN, range: -10...+10
        (
            AfDefinition("SiLU"),
            AfDefinition(
                "SiLU", AfDefinition.AfType.ADA_CONT
            )
        ),

        # expected: SiLU in CNN, HardTanh in FFN, range: -10...+10
        (
            AfDefinition("SiLU"),
            AfDefinition(
                "HardTanh", AfDefinition.AfType.TRAD,
                AfDefinition.AfInterval(-10.0, +10.0)
            )
        ),

        # expected: SiLU in CNN, Fuzzy HardTanh in FFN, range: -10...+10
        (
            AfDefinition("SiLU"),
            AfDefinition(
                "HardTanh", AfDefinition.AfType.ADA_FUZZ,
                AfDefinition.AfInterval(-10.0, +10.0, n_segments=12)
            )
        ),

        # expected:
        # AHAF as SiLU in CNN,
        # Fuzzy Sigmoid in FFN, range: -3...+3,
        (
            AfDefinition("SiLU", AfDefinition.AfType.ADA_CONT),
            AfDefinition(
                "Sigmoid", AfDefinition.AfType.ADA_FUZZ,
                AfDefinition.AfInterval(-3.0, +3.0, n_segments=12)
            )
        ),

        # expected: SiLU in CNN, AHAF as SiLU in FFN
        (
            AfDefinition("SiLU", AfDefinition.AfType.TRAD),
            AfDefinition("SiLU", AfDefinition.AfType.ADA_CONT),
        )
    ]

    batch_size = 64
    image_dim = (3, 32, 32)
    input_size = (batch_size, *image_dim)

    for d in nn_defs:
        net = KerasNetAaf(flavor='CIFAR10', af_conv=d[0], af_fc=d[1])
        torchinfo.summary(net, input_size=input_size)
        print_act_functions(net)


if __name__ == "__main__":
    main()
