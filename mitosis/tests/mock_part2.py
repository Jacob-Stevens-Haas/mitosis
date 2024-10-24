from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import NBitBase
from numpy.typing import NDArray

from mitosis._typing import ExpResults


def fit_and_score(
    data: NDArray[np.floating[NBitBase]], metric: Literal["len"] | Literal["zero"]
) -> ExpResults:
    if metric == "len":
        return {"main": len(data)}
    elif metric == "zero":
        return {"main": 0}


def bad_runnable(*args: Any, **kwargs: Any) -> int:
    return 1  # not a dict with key "main"
