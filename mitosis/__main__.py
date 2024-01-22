import argparse
from importlib import import_module
from pathlib import Path
from typing import Any
from typing import cast

from . import _resolve_param
from . import _split_param_str
from . import Experiment
from . import Parameter
from . import run


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mitosis",
        description=("Orchestrate experiment to ensure reproducibility."),
    )
    parser.add_argument("experiment", help="Name to identify the experiment")
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
            "precede the key with a minus sign ('-')."
        ),
    )
    return parser


def _process_cl_args(args: argparse.Namespace) -> dict[str, Any]:
    ex = cast(Experiment, import_module(args.experiment))
    params = []

    untracked_args: list[str] = []

    if args.eval_param is None:
        args.eval_param = ()
    for ep in args.eval_param:
        track, arg_name, var_name = _split_param_str(ep)
        if not track:
            untracked_args.append(arg_name)
        arg_val = eval(var_name)
        params.append(Parameter(str(arg_val), arg_name, arg_val, evaluate=True))

    if args.param is None:
        args.param = ()
    for param in args.param:
        track, arg_name, var_name = _split_param_str(param)
        if not track:
            untracked_args.append(arg_name)
        params += [_resolve_param(arg_name, var_name, ex.lookup_dict)]

    if args.folder is None:
        trials_folder = Path(".").resolve()
    else:
        trials_folder = Path(str(ex.__file__)).parent / args.folder
    if not trials_folder.exists():
        trials_folder.mkdir(parents=True)
    return {
        "ex": ex,
        "debug": args.debug,
        "group": args.group,
        "logfile": f"trials_{args.experiment}.db",
        "params": params,
        "trials_folder": trials_folder,
        "untracked_params": untracked_args,
    }


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()
    kwargs = _process_cl_args(args)
    run(**kwargs)


if __name__ == "__main__":
    main()
