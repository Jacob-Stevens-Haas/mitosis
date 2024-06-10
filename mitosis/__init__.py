import logging
import pprint
import sys
from collections import OrderedDict
from datetime import datetime
from datetime import timezone
from glob import glob
from importlib import import_module
from importlib.metadata import packages_distributions
from importlib.metadata import version
from logging import Logger
from pathlib import Path
from random import choices
from time import process_time
from types import BuiltinFunctionType
from types import BuiltinMethodType
from types import FunctionType
from types import MethodType
from typing import Any
from typing import cast
from typing import Collection
from typing import Hashable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence

import dill  # type: ignore
import nbformat
import pandas as pd
import sqlalchemy as sql
from nbclient.exceptions import CellExecutionError
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.writers.files import FilesWriter
from nbformat import NotebookNode
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import Float
from sqlalchemy import insert
from sqlalchemy import inspection
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import update

from . import _disk
from ._disk import locate_trial_folder
from ._typing import ExpStep
from ._typing import Parameter
from ._version import version as __version__  # noqa: F401


def trials_columns():
    return [
        Column("variant", String, primary_key=True),
        Column("iteration", Integer, primary_key=True),
        Column("commit", String, nullable=False),
        Column("cpu_time", Float),
        Column("results", String),
        Column("filename", String),
    ]


def variant_types():
    return [
        Column("name", String, primary_key=True),
        Column("params", String, unique=False),
    ]


def load_trial_data(hexstr: str, *, trials_folder: Optional[Path | str] = None):
    trial = locate_trial_folder(hexstr, trials_folder=trials_folder)
    all_files = sorted(glob("results*.dill", root_dir=trial))
    results = []
    for file in all_files:
        with open(trial / file, "rb") as fh:
            results.append(dill.load(fh))
    return results


def _lookup_param(
    arg_name: str, var_name: str, lookup_dict: dict[str, Any]
) -> Parameter:
    stored = lookup_dict[arg_name][var_name]
    if isinstance(stored, Parameter):
        return Parameter(var_name, arg_name, stored.vals, evaluate=False)
    else:
        return Parameter(var_name, arg_name, stored, evaluate=False)


class DBHandler(logging.Handler):
    def __init__(
        self,
        filename: Path | str,
        table_name: str,
        cols: List[Column],
        separator: str = "--",
    ):
        self.separator = separator
        if Path(filename).is_absolute():
            self.db = Path(filename)
        else:
            self.db = Path(__file__).resolve().parent / filename

        md = MetaData()
        self.log_table = Table(table_name, md, *cols)
        url = "sqlite:///" + str(self.db)
        self.eng = create_engine(url)
        with self.eng.connect() as conn:
            if not inspection.inspect(conn).has_table(table_name):
                md.create_all(conn)

        super().__init__()
        self.addFilter(lambda rec: self.separator in rec.getMessage())

    def emit(self, record: logging.LogRecord):
        vals = self.parse_record(record.getMessage())
        stmt: sql.Insert | sql.Update
        if "insert" in vals[0]:
            stmt = insert(self.log_table)
            for i, col in enumerate(self.log_table.columns):
                if vals[i + 1]:
                    stmt = stmt.values({col: vals[i + 1]})
        elif "update" in vals[0]:
            stmt = update(self.log_table)
            for i, col in enumerate(self.log_table.columns):
                if col.primary_key:
                    stmt = stmt.where(col == vals[i + 1])
                else:
                    if vals[i + 1]:
                        stmt = stmt.values({col: vals[i + 1]})
        else:
            raise ValueError("Cannot parse db message")
        with self.eng.connect() as conn:
            conn.execute(stmt)

    def parse_record(self, msg: str) -> List[str]:
        return msg.split(self.separator)


def _init_logger(trial_log: Path, table_name: str, debug: bool) -> tuple[Logger, Table]:
    """Create a Trials logger with a database handler"""
    exp_logger = logging.Logger("experiments")
    exp_logger.setLevel(20)
    exp_logger.addHandler(logging.StreamHandler())
    db_h = DBHandler(trial_log, table_name, trials_columns())
    if len(exp_logger.handlers) < 2 and not debug:  # A weird error requires this
        exp_logger.addHandler(db_h)
    return exp_logger, db_h.log_table


def _init_variant_table(trial_db: Path, step: str, param: Parameter) -> Table:
    eng = create_engine("sqlite:///" + str(trial_db))
    md = MetaData()
    var_table = Table(f"{step}_variant_{param.arg_name}", md, *variant_types())
    inspector = inspection.inspect(eng)
    if not inspector.has_table(f"{step}_variant_{param.arg_name}"):
        md.create_all(eng)
    return var_table


def _verify_variant_name(trial_db: Path, step: str, param: Parameter) -> None:
    """Check for conflicts between variant names in prior trials

    Side effects:
        - If trial_db does not exist, will create it
        - If variant name has not been used before, will insert it
    """
    eng = create_engine("sqlite:///" + str(trial_db))
    md = MetaData()
    tb = Table(f"{step}_variant_{param.arg_name}", md, *variant_types())
    vals: Collection[Any]
    if isinstance(param.vals, Mapping):
        vals = StrictlyReproduceableDict({k: v for k, v in sorted(param.vals.items())})
    elif isinstance(param.vals, Collection) and not isinstance(param.vals, str):
        try:
            vals = StrictlyReproduceableList(sorted(param.vals))
        except (ValueError, TypeError):
            vals = param.vals
    else:
        vals = param.vals
    df = pd.read_sql(select(tb), eng)
    ind_equal = df.loc[:, "name"] == param.var_name
    if ind_equal.sum() == 0:
        stmt = tb.insert().values({"name": param.var_name, "params": str(vals)})
        with eng.connect() as conn:
            conn.execute(stmt)
    elif df.loc[ind_equal, "params"].iloc[0] != str(vals):
        raise RuntimeError(
            f"Parameter '{param.arg_name}' variant '{param.var_name}' "
            f"is stored with different values in {trial_db}, table '{tb}'. "
            f"(Stored: {df.loc[ind_equal, 'params'].iloc[0]}), attmpeted: {str(vals)}."
        )
    # Otherwise, parameter has already been registered and no conflicts


def _id_variant_iteration(
    trial_db: Path, trials_table: Table, master_variant: str
) -> int:
    """Identify the iteration for this exact variant of the trial

    Args:
        trial_log (path-like): location of the trial log database
        trials_table (sqlalchemy.Table): the main record of each
            trial/variant
        var_table (sqlalchemy.Table): the lookup table for simulation
            variants
        sim_params (dict): parameters used in simulated experimental
            data
        id_table (sqlalchemy.Table): the lookup table for trial ids
        prob_params (dict): Parameters used to create the problem/solver
            in the experiment
    """
    eng = create_engine("sqlite:///" + str(trial_db))
    stmt = select(trials_table).where(trials_table.c.variant == master_variant)
    df = pd.read_sql(stmt, eng)
    if df.empty:
        return 1
    else:
        return df["iteration"].max() + 1


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
    exp_logger, trials_table = _init_logger(trial_db, experiments_table, debug)
    iteration = (
        0 if debug else _id_variant_iteration(trial_db, trials_table, master_variant)
    )
    rand_key = "".join(choices(list("0123456789abcde"), k=6))

    out_filename = _create_filename(
        master_variant, debug, iteration, rand_key, output_extension
    )
    exp_metadata_folder = _make_metadata_folder(trials_folder, rand_key)
    _write_freezefile(exp_metadata_folder)

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
    _log_finish_experiment(
        exp_logger, master_variant, iteration, commit, metric, out_filename, start_time
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
    code = (
        "import logging\n"
        "from pathlib import Path\n\n"
        "import matplotlib as mpl\n"
        "import dill\n"
        "import mitosis\n\n"
        f"mpl.rcParams['figure.dpi'] = {matplotlib_dpi}\n"
        f"mpl.rcParams['savefig.dpi'] = {matplotlib_dpi}\n"
        "inputs = None\n"
    )
    nb = nbformat.v4.new_notebook()
    setup_cell = nbformat.v4.new_code_cell(source=code)
    step_loader_cells: list[NotebookNode] = []
    step_runner_cells: list[NotebookNode] = []
    for order, step in enumerate(steps):
        lookup_params = {a.arg_name: a.var_name for a in step.args if not a.evaluate}
        eval_params = {a.arg_name: a.var_name for a in step.args if a.evaluate}
        logfile = trials_folder / "experiment.log"
        code = (
            (
                f"step_{order} = mitosis.unpack('{step.action_ref}')\n"
                f"lookup_{order} = mitosis.unpack('{step.lookup_ref}')\n"
                f"resolved_args_{order} = {{}}\n"
                f"logger = logging.getLogger('{step.action.__module__}')\n"
            )
            + (
                "logger.setLevel(logging.DEBUG)\n"
                if debug
                else "logger.setLevel(logging.INFO)\n"
            )
            + (
                f"logger.addHandler(logging.FileHandler('{str(logfile)}', delay=True))\n"  # noqa E501
                f"logger.info('Initialized experiment logger')\n"
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


def cleanstr(obj):
    if (
        isinstance(obj, FunctionType)
        or isinstance(obj, MethodType)
        or isinstance(obj, BuiltinFunctionType)
        or isinstance(obj, BuiltinMethodType)
    ):
        if obj.__name__ == "<lambda>":
            raise ValueError("Cannot use lambda functions in this context")
        import_error = ImportError(
            "Other modules must be able to import stored functions and modules:"
            f"function named {obj.__qualname__} stored in {obj.__module__}"
        )
        if obj.__module__ == "__main__":
            raise import_error
        if "<locals>" in obj.__qualname__:
            raise import_error
        try:
            mod = sys.modules[obj.__module__]
        except KeyError:
            raise import_error
        if not hasattr(mod, obj.__qualname__) or getattr(mod, obj.__qualname__) != obj:
            raise import_error
        return f"<{type(obj).__name__} {obj.__module__}.{obj.__qualname__}>"
    else:
        if isinstance(obj, str):
            return f"'{str(obj)}'"
        return str(obj)


class StrictlyReproduceableDict(OrderedDict):
    """A Dict that enforces reproduceable string representations

    The standard function.__str__ includes a memory location, which means the
    string representation of a function changes every time a program is run.
    In order to provide some stability in logs indicating that a function was
    run, the following dictionary will reject creating a string from functions
    that are difficult or impossible to reproduce in experiments.  It will also
    produce a reasonable string representation without the memory address for
    functions that are reproduceable.
    """

    def __str__(self):
        string = "{"
        for k, v in self.items():
            string += f"{cleanstr(k)}: "
            if isinstance(v, Mapping):
                string += str(StrictlyReproduceableDict(**v)) + ", "
            elif isinstance(v, Collection) and not isinstance(v, Hashable):
                string += str(StrictlyReproduceableList(v)) + ", "
            else:
                string += f"{cleanstr(v)}, "
        else:
            string = string[:-2] + "}"
        return string


class StrictlyReproduceableList(List):
    def __str__(self):
        string = "["
        for item in iter(self):
            if isinstance(item, Mapping):
                string += str(StrictlyReproduceableDict(**item)) + ", "
            elif isinstance(item, Collection) and not isinstance(item, Hashable):
                string += str(StrictlyReproduceableList(item)) + ", "
            else:
                string += f"{cleanstr(item)}, "
        else:
            string = string[:-2] + "]"
        return string


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
) -> None:
    utc_now = datetime.now(timezone.utc)
    exp_logger.info(
        "Finished experiment at time: "
        + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
        + f".  Results: {metric}"
    )
    exp_logger.info(
        "trial entry: update"
        + f"--{variant}"
        + f"--{iteration}"
        + f"--{commit}"
        + f"--{process_time() - start_time}"
        + f"--{metric}"
        + f"--{filename}"
    )


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
