from typing import Optional

from nns_aaf import KerasNetAaf, AfDefinition


AF_FUZZY_DEFAULT_INTERVAL = AfDefinition.AfInterval(
    start=-4.0, end=+4.0, n_segments=16
)


def create_net(
        net_type: str, af_name: str, *, af_name_cnn: Optional[str] = None
) -> KerasNetAaf:

    if af_name_cnn is None:
        af_name_cnn = af_name

    af_name_ffn = af_name

    if net_type == "base":
        af_type_cnn = AfDefinition.AfType.TRAD
        af_type_ffn = AfDefinition.AfType.TRAD
        af_interval_ffn = None
    elif net_type == "ahaf":
        af_type_cnn = AfDefinition.AfType.ADA_CONT
        af_type_ffn = AfDefinition.AfType.ADA_CONT
        af_interval_ffn = None
    elif net_type == "fuzzy_ffn":
        af_type_cnn = AfDefinition.AfType.TRAD
        af_type_ffn = AfDefinition.AfType.ADA_FUZZ
        af_interval_ffn = AF_FUZZY_DEFAULT_INTERVAL
    else:
        raise ValueError("Network type is not supported")

    cnn_af = AfDefinition(
        af_base=af_name_cnn, af_type=af_type_cnn
    )

    ffn_af = AfDefinition(
        af_base=af_name_ffn, af_type=af_type_ffn,
        af_interval=af_interval_ffn
    )

    net = KerasNetAaf(
        flavor='CIFAR10', af_conv=cnn_af, af_fc=ffn_af
    )

    return net
