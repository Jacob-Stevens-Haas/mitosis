from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import NamedTuple
from typing import ParamSpec
from typing import TypedDict


class ExpResults(TypedDict):
    main: object


P = ParamSpec("P")
ExpRun = Callable[P, ExpResults]


@dataclass
class Parameter:
    """An experimental parameter

    Arguments:
        var_name: short name for the variant (particular values) across use cases
        arg_name: name of arg known to experiment
        vals: value of the parameter
        eval: whether variant name should be evaluated or looked up.
    """

    var_name: str
    arg_name: str
    vals: object
    # > 3.10 only: https://stackoverflow.com/a/49911616/534674
    evaluate: bool = field(default=False, kw_only=True)


class ExpStep(NamedTuple):
    name: str
    action: ExpRun
    action_ref: str
    lookup: Mapping[str, Mapping[str, object]]
    lookup_ref: str
    group: str | None
    args: list[Parameter]
    untracked_args: list[str]
