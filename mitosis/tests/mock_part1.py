from logging import getLogger

import numpy as np

from mitosis._typing import ExpResults


class Klass:
    @staticmethod
    def gen_data(length: int, extra: bool = False) -> ExpResults:
        getLogger(__name__).info("This is run every time")
        getLogger(__name__).debug("This is run in debug mode only")

        return {
            "data": np.ones(length, dtype=np.float_),
            "extra": extra,
            "main": None,
        }  # type: ignore


def do_nothing(*args, **kwargs) -> ExpResults:
    """An experiment step that accepts anything and produces nothing"""
    return {"main": None}
