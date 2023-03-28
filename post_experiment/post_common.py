from typing import NamedTuple, Optional


class NetInfo(NamedTuple):
    net_type: str
    af_name: str
    epoch: int
    patched: bool = False,
    fine_tuned: bool = False
    af_name_cnn: Optional[str] = None
