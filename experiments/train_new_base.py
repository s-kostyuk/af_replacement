#!/usr/bin/env python3

from train_common import train_variant


def main():
    af_names = ("ReLU", "SiLU", "Tanh", "Sigmoid")
    for af in af_names:
        train_variant("base", af_name=af, end_epoch=100)


if __name__ == "__main__":
    main()
