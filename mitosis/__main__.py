import argparse
from importlib import import_module
from pathlib import Path
from typing import Any
from typing import cast
from typing import Optional
from typing import TypedDict

from . import _disk
from . import _resolve_param
from . import _split_param_str
from . import Experiment
from . import Parameter
from . import run


class StepDef(TypedDict):
    name: str
    module: str
    lookup: str
    group: str
    eval_params: list[str]
    lookup_params: list[str]


class ExpStep(TypedDict):
    name: str
    module: Experiment
    lookup: dict[str, Any]
    group: str
    eval_args: list[Parameter]
    lookup_args: list[Parameter]
    untracked_args: list[str]


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mitosis",
        description=("Orchestrate experiment to ensure reproducibility."),
    )
    parser.add_argument(
        "experiment",
        nargs="*",
        help="Name to identify the experiment step(s), specified in pyproject.toml",
    )
    parser.add_argument(
        "-m",
        "--module",
        nargs="?",
        help="Load complete experiment from a module",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help=(
            "Run in debug mode, allowing one to use uncommitted code changes and not"
            " recording results"
        ),
    )
    parser.add_argument(
        "--group",
        "-g",
        type=str,
        nargs="?",
        default=None,
        help="Group of experiment.  This tells mitosis to store all results for a group"
        " separately.",
    )
    parser.add_argument(
        "--folder",
        "-F",
        type=str,
        default=None,
        help="Where to save trials, relative to the experiment module folder",
    )
    parser.add_argument(
        "--eval-param",
        "-e",
        type=str,
        action="append",
        help="Parameters directly passed on command line",
    )
    parser.add_argument(
        "--param",
        "-p",
        action="append",
        help=(
            "Name of parameters to use with this trial, in format 'key=value'\ne.g."
            "--param solver=solver_1\nKeys must be understood by the experiment"
            " being run.  Values reference variables\nstored by the same name in the"
            " experiment's config dict.\n\n To skip tracking and locking a parameter, "
            "precede the key with a plus sign ('+')."
        ),
    )
    return parser


def _normalize_params(
    ep_strs: Optional[list[str]],
    lp_strs: Optional[list[str]],
    lookup_dict: dict[str, Any],
) -> tuple[list[Parameter], list[str]]:
    params = []

    untracked_args: list[str] = []

    for param in lp_strs if lp_strs else ():
        track, arg_name, var_name = _split_param_str(param)
        if not track:
            untracked_args.append(arg_name)
        params += [_resolve_param(arg_name, var_name, lookup_dict)]

    for ep in ep_strs if ep_strs else ():
        track, arg_name, var_name = _split_param_str(ep)
        if not track:
            untracked_args.append(arg_name)
        params.append(Parameter(var_name, arg_name, var_name, evaluate=True))

    return params, untracked_args


def _process_cl_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.experiment:
        exps = cast(list[str], args.experiment)
        if len(args.experiment) > 1:
            raise RuntimeError(
                "Multi-step experiments not supported yet, check back tomorrow"
            )
        if args.module:
            raise RuntimeError("Cannot use -m option if also passing experiment steps.")
    elif args.module:
        exps = [cast(str, args.module)]
    else:
        raise RuntimeError(
            "Must set either pass a list of experiment steps "
            "(defined in pyproject.toml) or use the -m flag to pass a single"
            "installed experiment module"
        )
    repo = _disk.get_repo()
    pyproj = Path(repo.working_dir) / "pyproject.toml"
    all_steps = _disk.parse_steps(_disk.load_mitosis_steps(pyproj))
    exp_steps: list[StepDef] = []
    for ex in exps:
        exp_steps.append(
            {
                "name": ex,
                "module": all_steps[ex][0],
                "lookup": all_steps[ex][1],
                "group": assign_group(ex, args.group),
                "eval_params": assign_params(ex, args.eval_param),
                "lookup_params": assign_params(ex, args.eval_param),
            }
        )
    # begin clusterfuck
    for ex in exps:
        ex_mod = cast(Experiment, import_module(ex))
    params, untracked_args = _normalize_params(  # here
        args.eval_param, args.param, ex_mod.lookup_dict
    )

    if args.folder is None:
        folder = Path(repo.working_dir) / "trials"
    else:
        folder = Path(args.folder)
    return {
        "ex": exps,  # and ehre
        "debug": args.debug,
        "group": args.group,  # also here
        "dbfile": args.experiment,  # here
        "params": params,
        "trials_folder": folder,
        "untracked_params": untracked_args,
    }


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()
    kwargs = _process_cl_args(args)
    run(**kwargs)


if __name__ == "__main__":
    main()
