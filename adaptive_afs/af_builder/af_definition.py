from enum import Enum
from typing import Optional, Tuple


class AfDefinition:
    class AfType(Enum):
        TRAD = 0
        ADA_CONT = 1
        ADA_FUZZ = 2

    class AfInterval:
        def __init__(self, start: float, end: float, n_segments: int = 0):
            self.start = start
            self.end = end
            self.n_segments = n_segments

    def __init__(
            self, af_base: str = "ReLU", af_type: AfType = AfType.TRAD,
            af_interval: Optional[AfInterval] = None
    ):
        self.af_base = af_base
        self.af_type = af_type
        self.interval = af_interval
