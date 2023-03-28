#!/usr/bin/env python3

from eval_pretrained_common import eval_variant


def main():
    af_names = ("ReLU", "SiLU")
    for af in af_names:
        eval_variant("ahaf", af_name=af, start_ep=100, patched=True)


if __name__ == "__main__":
    main()
