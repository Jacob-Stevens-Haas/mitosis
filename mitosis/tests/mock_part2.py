from typing import Literal

import numpy as np
from numpy.typing import NBitBase
from numpy.typing import NDArray


def fit_and_score(
    data: NDArray[np.floating[NBitBase]], metric: Literal["len"] | Literal["zero"]
) -> dict[str, float]:
    if metric == "len":
        return {"main": len(data)}
    elif metric == "zero":
        return {"main": 0}
