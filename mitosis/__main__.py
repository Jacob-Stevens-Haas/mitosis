import argparse
from collections.abc import Sequence
from importlib import import_module
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import cast
from typing import NamedTuple

from . import _disk
from . import _lookup_param
from . import Experiment
from . import ExpRun
from . import Parameter
from . import parse_steps
from . import run


class ExpStep(NamedTuple):
    name: str
    module: ExpRun
    lookup: dict[str, Any]
    group: str | None
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
        nargs="*",
        default=None,
        action="append",
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


class CLIParam(NamedTuple):
    step_key: str
    track: bool
    arg_name: str
    var_name: str


def _split_param_str(paramstr: str) -> CLIParam:
    arg_name, var_name = paramstr.split("=")
    track = True
    if arg_name[0] == "+":
        track = False
        arg_name = arg_name[1:]
    ex_step, _, arg_name = arg_name.rpartition(".")
    return CLIParam(ex_step, track, arg_name, var_name)


def _normalize_params(
    lookup_dict: dict[str, Any],
    ep_strs: Sequence[str] = (),
    lp_strs: Sequence[str] = (),
) -> tuple[list[Parameter], list[str]]:
    params = []

    untracked_args: list[str] = []

    for param in lp_strs:
        track, arg_name, var_name = _split_param_str(param)
        if not track:
            untracked_args.append(arg_name)
        params += [_lookup_param(arg_name, var_name, lookup_dict)]

    for ep in ep_strs:
        track, arg_name, var_name = _split_param_str(ep)
        if not track:
            untracked_args.append(arg_name)
        params.append(Parameter(var_name, arg_name, var_name, evaluate=True))

    return params, untracked_args


def _process_cl_args(args: argparse.Namespace) -> dict[str, Any]:
    ep_tups = [_split_param_str(epstr) for epstr in args.eval_param]
    lp_tups = [_split_param_str(lpstr) for lpstr in args.param]

    if args.experiment:
        if len(args.experiment) > 1:
            raise RuntimeError(
                "Multi-step experiments not supported yet, check back tomorrow"
            )
        if args.module:
            raise RuntimeError("Cannot use -m option if also passing experiment steps.")
        all_steps = parse_steps(args.experiment, _disk.load_mitosis_steps())
        grp_dict = dict(grp.split(".") for grp in args.group)
    elif args.module:
        mod = cast(str, args.module)
        all_steps = parse_steps([mod], normalize_modinput(args.module))
        ep_tups = [CLIParam(mod, track, name, val) for _, track, name, val in ep_tups]
        lp_tups = [CLIParam(mod, track, name, val) for _, track, name, val in lp_tups]
        grp_dict = {mod: args.group[0]} if args.group else {}
    else:
        raise RuntimeError(
            "Must set either pass a list of experiment steps "
            "(defined in pyproject.toml) or use the -m flag to pass a single"
            "installed experiment module"
        )

    def group_and_pop(
        list_of_clargs: list[CLIParam],
    ) -> dict[str, list[tuple[bool, str, str]]]:
        new_list = sorted(list_of_clargs, key=lambda tup: tup.step_key)
        arg_groups = {k: v for k, v in groupby(new_list, key=lambda tup: tup.step_key)}
        popped = {k: [tup[1:] for tup in l] for k, l in arg_groups.items()}
        return popped

    ep_dict = group_and_pop(ep_tups)
    lp_dict = group_and_pop(lp_tups)

    unassigned = set(ep_dict.keys()).union(lp_dict.keys()) - set(all_steps.keys())
    if unassigned:
        raise RuntimeError(
            f"Steps {unassigned} not in experiment, but arguments assigned to them"
        )
    unknown_groups = set(grp_dict.keys()) - set(all_steps.keys())
    if unknown_groups:
        raise RuntimeError(
            f"Steps {unknown_groups} not in experiment, but groups assigned to them"
        )

    def create_step(
        name: str,
        ex: ExpRun,
        lookup_dict: dict[str, Any],
        eps: tuple[bool, str, str],
        lps: tuple[bool, str, str],
    ) -> ExpStep:
        pass

    # groups[step] could be empty...
    exp_steps = [
        ExpStep(
            step,
            runnable,
            lookup,
            grp_dict.get(step),  # deal with this
            [
                Parameter(val, arg, eval(val), evaluate=True)
                for _, arg, val in ep_dict[step]
            ],
            [
                Parameter(val, arg, lookup[arg][val], evaluate=False)
                for _, arg, val in lp_dict[step]
            ],
            [arg for track, arg, _ in ep_dict[step] + lp_dict[step] if track],
        )
        for step, (runnable, lookup) in all_steps.items()
    ]

    exps = []
    # begin clusterfuck
    for ex in exps:
        ex_mod = cast(Experiment, import_module(ex))
    params, untracked_args = _normalize_params(  # here
        ex_mod.lookup_dict, args.eval_param, args.param
    )

    if args.folder is None:
        folder = Path(_disk.get_repo().working_dir) / "trials"
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


def normalize_modinput(obj_ref: str) -> dict[str, tuple[str, str]]:
    modname, _, qualname = obj_ref.partition(":")
    if qualname:
        sep = ":" + qualname + "."
    else:
        sep = ":"
    return {obj_ref: (modname + sep + "run", modname + sep + "lookup_dict")}
