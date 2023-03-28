from typing import Optional


def _get_file_name_net_base(
        file_type: str,
        net_type: str, af_name: str, epoch: int,
        patched: bool, fine_tuned: bool,
        af_name_cnn: Optional[str]
) -> str:
    patched_str = "patched_" if patched else ""
    fine_tuned_str = "tuned_" if fine_tuned else ""

    if file_type == "stat":
        extension = "csv"
    elif file_type == "aaf_img":
        extension = "svg"
    else:
        extension = "bin"

    if af_name_cnn is not None and af_name != af_name_cnn:
        af_name = "{}_{}".format(af_name, af_name_cnn)

    return "runs/kerasnet_{}_{}{}{}_{}ep_{}.{}".format(
        af_name, patched_str, fine_tuned_str, net_type, epoch, file_type,
        extension
    )


def get_file_name_net(
        net_type: str, af_name: str, epoch: int, patched: bool = False,
        fine_tuned: bool = False, af_name_cnn: Optional[str] = None
) -> str:
    return _get_file_name_net_base(
        "net", net_type, af_name, epoch, patched, fine_tuned, af_name_cnn
    )


def get_file_name_opt(
        net_type: str, af_name: str, epoch: int, patched: bool = False,
        fine_tuned: bool = False, af_name_cnn: Optional[str] = None
) -> str:
    return _get_file_name_net_base(
        "opt", net_type, af_name, epoch, patched, fine_tuned, af_name_cnn
    )


def get_file_name_stat(
        net_type: str, af_name: str, epoch: int, patched: bool = False,
        fine_tuned: bool = False, af_name_cnn: Optional[str] = None
) -> str:
    return _get_file_name_net_base(
        "stat", net_type, af_name, epoch, patched, fine_tuned, af_name_cnn
    )


def get_file_name_stat_img(
        net_group: str, epoch: int, patched: bool = False,
        fine_tuned: bool = False
) -> str:
    patched_str = "patched_" if patched else ""
    fine_tuned_str = "tuned_" if fine_tuned else ""

    return "runs/kerasnet_{}{}summary_{}_{}ep_stat_img.svg".format(
        patched_str, fine_tuned_str, net_group, epoch
    )


def get_file_name_stat_table(
        net_group: str, epoch: int, patched: bool = False,
        fine_tuned: bool = False
) -> str:
    patched_str = "patched_" if patched else ""
    fine_tuned_str = "tuned_" if fine_tuned else ""

    return "runs/kerasnet_{}{}summary_{}_{}ep_stat_table.csv".format(
        patched_str, fine_tuned_str, net_group, epoch
    )


def get_file_name_aaf_img(
        net_type: str, af_name: str, epoch: int, patched: bool = False,
        fine_tuned: bool = False, af_name_cnn: Optional[str] = None
) -> str:
    return _get_file_name_net_base(
        "aaf_img", net_type, af_name, epoch, patched, fine_tuned, af_name_cnn
    )
