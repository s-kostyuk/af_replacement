#!/usr/bin/env python3

from eval_pretrained_common import eval_variant


def main():
    af_names = ("Tanh", "Sigmoid")
    for af in af_names:
        eval_variant("fuzzy_ffn", af_name=af, start_ep=100, patched=True)


if __name__ == "__main__":
    main()
