import logging
import os
import pprint
import sys
from datetime import datetime
from datetime import timezone
from glob import glob
from importlib import import_module
from importlib.metadata import packages_distributions
from importlib.metadata import version
from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from logging import Logger
from pathlib import Path
from random import choices
from tempfile import NamedTemporaryFile
from time import process_time
from typing import Any
from typing import cast
from typing import Collection
from typing import Optional
from typing import Sequence

import dill  # type: ignore
import nbformat
from nbclient.exceptions import CellExecutionError
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.writers.files import FilesWriter
from nbformat import NotebookNode

from . import _disk
from ._db import _id_variant_iteration
from ._db import _init_variant_table
from ._db import create_trials_db_eng
from ._db import record_finish_in_db
from ._db import record_start_in_db
from ._disk import locate_trial_folder
from ._typing import ExpStep
from ._typing import Parameter
from ._version import version as __version__  # noqa: F401
from mitosis._db import _verify_variant_name


def load_trial_data(hexstr: str, *, trials_folder: Optional[Path | str] = None):
    trial = locate_trial_folder(hexstr, trials_folder=trials_folder)
    all_files = sorted(glob("results*.dill", root_dir=trial))
    results = []
    for file in all_files:
        with open(trial / file, "rb") as fh:
            results.append(dill.load(fh))
    return results


def _load_trial_params(
    hexstr: str, *, step: int = 0, trials_folder: Optional[Path | str] = None
) -> dict[str, Any]:
    """Reload the parameters of a particular step of a trial (unstable)

    Does not do any environment validation, which can cause failures.  E.g.
    if the experiment code is a different version that used in the trial,
    or if the python version is different, or if the experiment database file
    has been reset and configuration file/lookup dictionary changed, no
    guarantee can be made about the true value of the parameters returned.
    """
    metadata_dir = locate_trial_folder(hexstr, trials_folder=trials_folder)
    src_file = metadata_dir / "source.py"
    # Once on 3.12, change delete to "delete_on_close"
    tempfile = NamedTemporaryFile(
        "w+t", suffix=".py", prefix="mod_src", dir=metadata_dir, delete=False
    )
    src_text = ""
    with open(src_file, "r") as f_src:
        for line in f_src.readlines():
            # hackiness: stop copying file before experiment execution code
            if "mitosis._prettyprint_config" in line:
                break
            src_text += line
    with tempfile as f_tgt:
        f_tgt.write(src_text)
        f_tgt.close()
        src_spec = spec_from_file_location("_source", tempfile.name)
        if src_spec is None:
            raise RuntimeError(f"Unable to spec source file for trial {hexstr}")
        src = module_from_spec(src_spec)
        if src_spec.loader is None:
            raise RuntimeError(f"Failed to load source file for trial {hexstr}")
        src_spec.loader.exec_module(src)
        os.unlink(tempfile.name)
    try:
        args = getattr(src, f"resolved_args_{step}")
        sys.modules.pop("_source", None)
        del src
        return args
    except AttributeError:
        raise ValueError(f"Trial {hexstr} does not have a step {step}")


def _lookup_param(
    arg_name: str, var_name: str, lookup_dict: dict[str, Any]
) -> Parameter:
    stored = lookup_dict[arg_name][var_name]
    if isinstance(stored, Parameter):
        return Parameter(var_name, arg_name, stored.vals, evaluate=False)
    else:
        return Parameter(var_name, arg_name, stored, evaluate=False)


def _init_logger() -> Logger:
    """Create a Trials logger with a database handler"""
    exp_logger = logging.Logger("experiments")
    exp_logger.setLevel(20)
    exp_logger.addHandler(logging.StreamHandler())
    return exp_logger


def _lock_in_variant(
    step: str,
    params: Sequence[Parameter],
    untracked_params: Collection[str],
    trial_db: Path,
    debug: bool,
) -> str:
    """Calculate the unique variant name combining all variants of parameters"""
    for param in params:
        if debug or param.arg_name in untracked_params:
            continue
        _init_variant_table(trial_db, step, param)
        _verify_variant_name(trial_db, step, param)
    var_names = [param.var_name for param in params]
    arg_names = [param.arg_name for param in params]
    if not arg_names:
        return "noparams"
    return f"{step}-" + "-".join(
        [x for _, x in sorted(zip(arg_names, var_names), key=lambda pair: pair[0])]
    )


def _get_commit_and_project_root(debug: bool) -> tuple[str, Path]:
    repo = _disk.get_repo()
    commit = "0000000" if debug else repo.head.commit.hexsha
    if not debug and repo.is_dirty():
        raise RuntimeError(
            "Git Repo is dirty.  For repeatable tests,"
            " clean the repo by committing or stashing all changes and "
            "untracked files."
        )
    return commit, Path(repo.working_dir)


def run(
    steps: list[ExpStep],
    debug: bool = False,
    *,
    trials_folder: Path,
    output_extension: str = "html",
    matplotlib_dpi: int = 72,
) -> str:
    """Run the selected experiment.

    Arguments:
        exps: The experiment steps to run
        debug (bool): Whether to run in debugging mode or not.
        trials_folder: The folder to store output, database, log, and metadata.
        output_extension: what output type to produce using nbconvert.
            Either 'html' or 'ipynb'.
        matplotlib_dpi: dpi for matplotlib images.  Not yet
            functional.

    Returns:
        The pseudorandom key to this experiment
    """

    commit, _ = _get_commit_and_project_root(debug)
    trials_folder = Path(trials_folder).absolute()
    if not trials_folder.exists():
        trials_folder.mkdir(parents=True)
    exp_name = "_".join(
        step.name + (f"_{step.group}" if step.group else "") for step in steps
    )
    dbfile = exp_name + ".db"
    trial_db = trials_folder / dbfile
    master_variant = "+".join(
        _lock_in_variant(step.name, step.args, step.untracked_args, trial_db, debug)
        for step in steps
    )
    experiments_table = f"trials_{exp_name}"
    for step in steps:
        if step.group is not None:
            step.args.append(
                Parameter(f"'{step.group}'", "group", step.group, evaluate=True)
            )
    exp_logger = _init_logger()
    eng, trials_tb = create_trials_db_eng(trial_db, experiments_table)
    iteration = 0 if debug else _id_variant_iteration(eng, trials_tb, master_variant)
    rand_key = "".join(choices(list("0123456789abcde"), k=6))

    out_filename = _create_filename(
        master_variant, debug, iteration, rand_key, output_extension
    )
    exp_metadata_folder = _make_metadata_folder(trials_folder, rand_key)
    _write_freezefile(exp_metadata_folder)

    if not debug:
        record_start_in_db(trials_tb, eng, master_variant, iteration, commit)
    start_time = _log_start_experiment(
        exp_logger, master_variant, iteration, commit, debug
    )
    nb, metric, exc = _run_in_notebook(
        steps,
        exp_metadata_folder,
        matplotlib_dpi,
        debug=debug,
    )
    _save_notebook(nb, out_filename, trials_folder, output_extension)
    (exp_metadata_folder / "experiment").symlink_to(trials_folder / out_filename)
    total_time = _log_finish_experiment(
        exp_logger, master_variant, iteration, commit, metric, out_filename, start_time
    )
    if not debug:
        record_finish_in_db(
            trials_tb, eng, master_variant, iteration, metric, out_filename, total_time
        )

    if exc is not None:
        raise exc
    return rand_key


def _run_in_notebook(
    steps: list[ExpStep],
    trials_folder: Path,
    matplotlib_dpi=72,
    debug: bool = False,
) -> tuple[nbformat.NotebookNode, Optional[str], Optional[Exception]]:
    logfile = trials_folder / "experiment.log"
    logset_command = (
        "logger.setLevel(logging.DEBUG)\n"
        if debug
        else "logger.setLevel(logging.INFO)\n"
    )
    code = (
        "import logging\n"
        "from pathlib import Path\n\n"
        "import matplotlib as mpl\n"
        "import dill\n"
        "import mitosis\n\n"
        f"mpl.rcParams['figure.dpi'] = {matplotlib_dpi}\n"
        f"mpl.rcParams['savefig.dpi'] = {matplotlib_dpi}\n"
        "inputs = None\n"
        "\n"
        "logger = logging.getLogger()\n"
        f"{logset_command}\n"
        f"handler = logging.FileHandler('{str(logfile)}', delay=True)\n"
        "handler.setFormatter(logging.Formatter('{levelname}:{asctime}:{module}:{lineno}:{message}', style='{'))\n"  # noqa E501
        "logger.addHandler(handler)\n"
        "logger.info('Initialized experiment logger')\n"
    )
    nb = nbformat.v4.new_notebook()
    setup_cell = nbformat.v4.new_code_cell(source=code)
    step_loader_cells: list[NotebookNode] = []
    step_runner_cells: list[NotebookNode] = []
    for order, step in enumerate(steps):
        lookup_params = {a.arg_name: a.var_name for a in step.args if not a.evaluate}
        eval_params = {a.arg_name: a.var_name for a in step.args if a.evaluate}
        code = (
            f"step_{order} = mitosis.unpack('{step.action_ref}')\n"
            f"lookup_{order} = mitosis.unpack('{step.lookup_ref}')\n"
            f"resolved_args_{order} = {{}}\n"
            f'print("Loaded step {order} as {step.action_ref}")\n'
            f'print("Loaded lookup {order} as {step.lookup_ref}")\n'
            f"for arg_name, var_name in {lookup_params}.items():\n"
            f"    val = mitosis._lookup_param(arg_name, var_name, lookup_{order}).vals\n"  # noqa E501
            f"    resolved_args_{order}.update({{arg_name: val}}) \n"
            f"    print(arg_name,'=',resolved_args_{order}[arg_name])\n\n"
            f"for arg_name, var_name in {eval_params}.items():\n"
            f"    val = eval(var_name)\n"
            f"    resolved_args_{order}.update({{arg_name: val}}) \n"
            f"    print(arg_name,'=',resolved_args_{order}[arg_name])\n\n"
            f"mitosis._prettyprint_config(Path('{trials_folder}'), resolved_args_{order})\n"  # noqa E501
            f"print('Saving metadata to {trials_folder}')\n"
        )
        step_loader_cells.append(nbformat.v4.new_code_cell(source=code))

        code = (
            f"if inputs is not None:\n"
            f"    curr_result = step_{order}(inputs, **resolved_args_{order})\n"
            f"else:\n"
            f"    curr_result = step_{order}(**resolved_args_{order})\n"
            f"with open(r'{trials_folder / (f'results_{order}.dill')}', 'wb') as f:\n"  # noqa E501
            f"    dill.dump(curr_result, f)\n"
            f"try:\n"
            f"    print(curr_result['metrics'])\n"
            f"except KeyError:\n"
            f"    pass\n"
            f"inputs = curr_result.get('data', None)\n"
        )
        step_runner_cells.append(nbformat.v4.new_code_cell(source=code))

    result_cell = nbformat.v4.new_code_cell(source=("print(curr_result['main'])"))
    nb["cells"] = [setup_cell] + step_loader_cells + step_runner_cells + [result_cell]
    with open(trials_folder / "source.py", "w") as fh:
        fh.write("".join(cell["source"] for cell in nb.cells))
    ep = ExecutePreprocessor(timeout=-1)

    exception = None
    metrics = None
    if debug:
        allowed = cast(tuple[type[Exception], ...], ())
    else:
        allowed = (CellExecutionError,)
    try:
        ep.preprocess(nb, {"metadata": {"path": trials_folder}})
        metrics = nb["cells"][-1]["outputs"][0]["text"][:-1]  # last char is newline
    except allowed as exc:
        exception = exc
    return nb, metrics, exception


def _save_notebook(nb, filename, trials_folder, extension):
    if extension == "html":
        html_exporter = HTMLExporter({"template_file": "lab"})
        (body, resources) = html_exporter.from_notebook_node(nb)
        base_filename = filename[:-5]  # FilesWriter adds a .html
        file_writer = FilesWriter()
        file_writer.build_directory = str(trials_folder)
        file_writer.write(body, resources, notebook_name=base_filename)
    elif extension == "ipynb":
        with open(str(trials_folder / filename), "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    else:
        raise ValueError


def _make_metadata_folder(trials_folder: Path, rand_key: str) -> Path:
    metadata_key = datetime.now().astimezone().strftime(r"%Y-%m-%d") + f"_{rand_key}"
    metadata_folder = trials_folder / metadata_key
    metadata_folder.mkdir()
    return metadata_folder


def _create_filename(
    variant: str,
    debug: bool,
    iteration: int,
    suffix: Optional[str],
    extension: str,
) -> str:
    new_filename = f"trial_{variant}_{iteration}_{suffix}"
    if debug:
        new_filename += "debug"
    if extension == "html":
        new_filename += ".html"
    elif extension == "ipynb":
        new_filename += ".ipynb"
    return new_filename


def _log_start_experiment(
    exp_logger: logging.Logger,
    variant: str,
    iteration: int,
    commit: str,
    debug: bool,
) -> float:
    exp_logger.info(f"trial entry: insert--{variant}--{iteration}--{commit}--------")
    utc_now = datetime.now(timezone.utc)
    cpu_now = process_time()
    log_msg = (
        f"Running experiment {variant}, "
        f"iteration {iteration} at time: "
        + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
        + f".  Current repo hash: {commit}"
    )
    if debug:
        log_msg += ".  In debugging mode."
    exp_logger.info(log_msg)
    return cpu_now


def _log_finish_experiment(
    exp_logger: logging.Logger,
    variant: str,
    iteration: int,
    commit: str,
    metric: Optional[str],
    filename: str,
    start_time: float,
) -> float:
    utc_now = datetime.now(timezone.utc)
    exp_logger.info(
        "Finished experiment at time: "
        + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
        + f".  Results: {metric}"
    )
    total_time = process_time() - start_time
    exp_logger.info(
        "trial entry: update"
        + f"--{variant}"
        + f"--{iteration}"
        + f"--{commit}"
        + f"--{total_time}"
        + f"--{metric}"
        + f"--{filename}"
    )
    return total_time


def _write_freezefile(folder: Path):
    installed = {pkg for pkgs in packages_distributions().values() for pkg in pkgs}
    req_str = f"# {sys.version}\n"
    req_str += "\n".join(f"{pkg}=='{version(pkg)}'" for pkg in installed)
    with open(folder / "requirements.txt", "w") as f:
        f.write(req_str)


def _prettyprint_config(folder: Path, params: Collection[Parameter]):
    pretty = pprint.pformat(params)
    with open(folder / "config.txt", "a") as f:
        f.write(pretty)
        f.write("\n")


def unpack(obj_ref: str) -> Any:
    modname, _, qualname = obj_ref.partition(":")
    obj = import_module(modname)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj
