import argparse
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import cast
from typing import NamedTuple

from . import _disk
from . import ExpRun
from . import Parameter
from . import parse_steps
from . import run


class ExpStep(NamedTuple):
    name: str
    module: ExpRun
    lookup: dict[str, Any]
    group: str | None
    args: list[Parameter]
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
        help=(
            "Parameters directly passed on command line.  Make sure to quote if you"
            " want argument to evaluate as a string"
        ),
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


def _process_cl_args(args: argparse.Namespace) -> dict[str, Any]:
    ep_tups = [_split_param_str(epstr) for epstr in args.eval_param]
    lp_tups = [_split_param_str(lpstr) for lpstr in args.param]
    grp_dict: defaultdict[str, None | str] = defaultdict(lambda: None)

    if args.experiment:
        if len(args.experiment) > 1:
            raise RuntimeError(
                "Multi-step experiments not supported yet, check back tomorrow"
            )
        if args.module:
            raise RuntimeError("Cannot use -m option if also passing experiment steps.")
        all_steps = parse_steps(args.experiment, _disk.load_mitosis_steps())
        grp_dict.update(tuple(grp.split(".", 1)) for grp in args.group)
    elif args.module:
        mod = cast(str, args.module)
        all_steps = parse_steps([mod], normalize_modinput(args.module))
        ep_tups = [CLIParam(mod, track, name, val) for _, track, name, val in ep_tups]
        lp_tups = [CLIParam(mod, track, name, val) for _, track, name, val in lp_tups]
        if args.group:
            grp_dict[mod] = args.group[0]
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
        part_step: tuple[ExpRun, dict[str, Any]],
        group: str | None,
        eps: list[tuple[bool, str, str]],
        lps: list[tuple[bool, str, str]],
    ) -> ExpStep:
        runnable, lookup = part_step
        params = []
        untracked_args = []
        for track, arg_name, var_name in eps:
            params.append(Parameter(var_name, arg_name, var_name, evaluate=True))
            if not track:
                untracked_args.append(arg_name)
        for track, arg_name, var_name in lps:
            params.append(Parameter(var_name, arg_name, lookup[arg_name][var_name]))
            if not track:
                untracked_args.append(arg_name)
        return ExpStep(name, runnable, lookup, group, params, untracked_args)

    # groups[step] could be empty...
    exp_steps = [
        create_step(
            step_name,
            all_steps[step_name],
            grp_dict[step_name],
            ep_dict[step_name],
            lp_dict[step_name],
        )
        for step_name in all_steps.keys()
    ]

    if args.folder is None:
        folder = Path(_disk.get_repo().working_dir) / "trials"
    else:
        folder = Path(args.folder)
    return {
        "ex": exp_steps,
        "debug": args.debug,
        "trials_folder": folder,
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
