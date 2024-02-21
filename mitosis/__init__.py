import logging
import pprint
import sys
from abc import ABCMeta
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from importlib.metadata import packages_distributions
from importlib.metadata import version
from pathlib import Path
from random import choices
from time import process_time
from types import BuiltinFunctionType
from types import BuiltinMethodType
from types import FunctionType
from types import MethodType
from types import ModuleType
from typing import Any
from typing import Collection
from typing import Hashable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Protocol
from typing import Sequence

import dill  # type: ignore
import nbclient.exceptions
import nbformat
import pandas as pd
import sqlalchemy as sql
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.writers.files import FilesWriter
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


class _ExpRun(Protocol):  # Can't handle Varargs
    def __call__(self, *args: Any) -> dict:
        ...


class Experiment(ModuleType, metaclass=ABCMeta):
    __name__: str
    __file__: str
    name: str
    lookup_dict: dict[str, dict[str, Any]]
    run: _ExpRun


def trials_columns():
    return [
        Column("variant", Integer, primary_key=True),
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
    vals: Any
    # > 3.10 only: https://stackoverflow.com/a/49911616/534674
    evaluate: bool = field(default=False, kw_only=True)


def load_trial_data(hexstr: str, *, trials_folder: Optional[Path | str] = None):
    trial = _locate_trial_folder(hexstr, trials_folder=trials_folder)
    with open(trial / "results.dill", "rb") as fh:
        return dill.load(fh)


def _locate_trial_folder(
    hexstr: str, *, trials_folder: Optional[Path | str] = None
) -> Path:
    if trials_folder is None:
        trials_folder = Path().absolute()
    else:
        trials_folder = Path(trials_folder).resolve()
    matches = trials_folder.glob(f"*{hexstr}")
    try:
        first = next(matches)
    except StopIteration:
        raise FileNotFoundError(f"Could not find a trial that matched {hexstr}")
    try:
        next(matches)
    except StopIteration:
        return first
    raise RuntimeError(f"Two or more matches found for {hexstr}")


def _split_param_str(paramstr: str) -> tuple[bool, str, str]:
    arg_name, var_name = paramstr.split("=")
    track = True
    if arg_name[0] == "+":
        track = False
        arg_name = arg_name[1:]
    return track, arg_name, var_name


def _resolve_param(
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
        filename: str,
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


def _init_logger(trial_log, table_name, debug):
    """Create a Trials logger with a database handler"""
    exp_logger = logging.Logger("experiments")
    exp_logger.setLevel(20)
    exp_logger.addHandler(logging.StreamHandler())
    db_h = DBHandler(trial_log, table_name, trials_columns())
    if len(exp_logger.handlers) < 2 and not debug:  # A weird error requires this
        exp_logger.addHandler(db_h)
    return exp_logger, db_h.log_table


def _init_variant_table(trial_log, param: Parameter):
    eng = create_engine("sqlite:///" + str(trial_log))
    md = MetaData()
    var_table = Table(f"variant_{param.arg_name}", md, *variant_types())
    inspector = inspection.inspect(eng)
    if not inspector.has_table(f"variant_{param.arg_name}"):
        md.create_all(eng)
    return var_table


def _verify_variant_name(trial_db: Path, param: Parameter) -> None:
    """Check for conflicts between variant names in prior trials

    Side effects:
        - If trial_db does not exist, will create it
        - If variant name has not been used before, will insert it
    """
    eng = create_engine("sqlite:///" + str(trial_db))
    md = MetaData()
    tb = Table(f"variant_{param.arg_name}", md, *variant_types())
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


def _id_variant_iteration(trial_log, trials_table, master_variant: str) -> int:
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
    eng = create_engine("sqlite:///" + str(trial_log))
    stmt = select(trials_table).where(trials_table.c.variant == master_variant)
    df = pd.read_sql(stmt, eng)
    if df.empty:
        return 1
    else:
        return df["iteration"].max() + 1


def _lock_in_variant(
    params: Sequence[Parameter],
    untracked_params: Collection[str],
    trial_db: Path,
    debug: bool,
) -> str:
    """Calculate the unique variant name combining all variants of parameters"""
    for param in params:
        if debug or param.arg_name in untracked_params:
            continue
        _init_variant_table(trial_db, param)
        _verify_variant_name(trial_db, param)
    var_names = [param.var_name for param in params]
    arg_names = [param.arg_name for param in params]
    if not arg_names:
        return "noparams"
    return "-".join(
        [x for _, x in sorted(zip(arg_names, var_names), key=lambda pair: pair[0])]
    )


def run(
    exps: list[Experiment],
    debug: bool = False,
    *,
    group: str | None = None,
    params: Sequence[Parameter] = (),
    trials_folder: Path,
    output_extension: str = "html",
    untracked_params: Collection[str] = (),
    matplotlib_dpi: int = 72,
) -> str:
    """Run the selected experiment.

    Arguments:
        ex: The experiment steps to run
        debug (bool): Whether to run in debugging mode or not.
        group (str): Trial grouping.  Name a group if desiring to
            segregate trials using the same experiment code.  ex.run()
            must take a "group" argument.
        params: The assigned parameter dictionaries to generate and
            solve the problem.
        trials_folder: The folder to store output, database, log, and metadata.
        output_extension: what output type to produce using nbconvert.
            Either 'html' or 'ipynb'.
        untracked_params: names of parameters to not track in database
        matplotlib_dpi: dpi for matplotlib images.  Not yet
            functional.

    Returns:
        The pseudorandom key to this experiment
    """

    repo = _disk.get_repo()
    if debug:
        commit = "0000000"
    else:
        if repo.is_dirty():
            raise RuntimeError(
                "Git Repo is dirty.  For repeatable tests,"
                " clean the repo by committing or stashing all changes and "
                "untracked files."
            )
        commit = repo.head.commit.hexsha

    trials_folder = Path(trials_folder).absolute()
    if not trials_folder.exists():
        trials_folder.mkdir(parents=True)
    dbfile = "_".join(ex.__name__ for ex in exps) + ".db"
    trial_db = trials_folder / dbfile
    master_variant = _lock_in_variant(params, untracked_params, trial_db, debug)
    exps = exps[0]
    experiments_table = f"trials_{exps.name}"
    params = list(params)
    if group is not None:
        experiments_table += f" {group}"
        params.append(Parameter(f"'{group}'", "group", group, evaluate=True))
    exp_logger, trials_table = _init_logger(trial_db, experiments_table, debug)
    iteration = (
        0 if debug else _id_variant_iteration(trial_db, trials_table, master_variant)
    )
    rand_key = "".join(choices(list("0123456789abcde"), k=6))

    out_filename = _create_filename(
        exps.name, group, debug, master_variant, iteration, rand_key, output_extension
    )
    exp_metadata_folder = _make_metadata_folder(trials_folder, rand_key)
    _write_freezefile(exp_metadata_folder)

    start_time = _log_start_experiment(
        exps.name, exp_logger, master_variant, iteration, commit, debug
    )
    nb, metric, exc = _run_in_notebook(
        exps,
        {p.arg_name: p.var_name for p in params if not p.evaluate},
        {p.arg_name: p.var_name for p in params if p.evaluate},
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
    ex: Experiment,
    lookup_params: dict[str, str],
    eval_params: dict[str, str],
    trials_folder: Path,
    matplotlib_dpi=72,
    debug: bool = False,
) -> tuple[nbformat.NotebookNode, Optional[str], Optional[Exception]]:
    ex_module = ex.__name__
    ex_file = ex.__file__
    code = (
        "import importlib\n"
        "import logging\n"
        "import matplotlib as mpl\n"
        "import dill\n"
        "import sys\n\n"
        f"ex = importlib.import_module('{ex_module}', '{ex_file}')\n\n"
        f"mpl.rcParams['figure.dpi'] = {matplotlib_dpi}\n"
        f"mpl.rcParams['savefig.dpi'] = {matplotlib_dpi}\n"
        f"logger = logging.getLogger('{ex_module}')\n"
        'print(f"Running {ex.name}.run()")\n'
    )
    code += (
        "logger.setLevel(logging.DEBUG)\n"
        if debug
        else "logger.setLevel(logging.INFO)\n"
    )
    logfile = trials_folder / f"{ex_module}.log"
    code += f"logger.addHandler(logging.FileHandler('{logfile}', delay=True))\n"

    nb = nbformat.v4.new_notebook()
    setup_cell = nbformat.v4.new_code_cell(source=code)
    resolve_code = (
        "import mitosis\n"
        "from pathlib import Path\n"
        "resolved_args = {}\n"
        f"for arg_name, var_name in {lookup_params}.items():\n"
        "    val = mitosis._resolve_param(arg_name, var_name, ex.lookup_dict).vals\n"
        "    resolved_args.update({arg_name: val}) \n"
        "    print(arg_name,'=',resolved_args[arg_name])\n\n"
        f"for arg_name, var_name in {eval_params}.items():\n"
        "    val = eval(var_name)\n"
        "    resolved_args.update({arg_name: val}) \n"
        "    print(arg_name,'=',resolved_args[arg_name])\n\n"
        f"mitosis._prettyprint_config(Path('{trials_folder}'), resolved_args)\n"
        f"print('Saving metadata to {trials_folder}')\n"
    )
    resolve_cell = nbformat.v4.new_code_cell(source=resolve_code)
    run_cell = nbformat.v4.new_code_cell(source="results = ex.run(**resolved_args)")
    result_cell = nbformat.v4.new_code_cell(
        source=""
        f"with open(r'{trials_folder / ('results.dill')}', 'wb') as f:\n"  # noqa E501
        "  dill.dump(results, f)\n"
        "print(repr(results))\n"
    )
    metrics_cell = nbformat.v4.new_code_cell(source="print(results['main'])")
    nb["cells"] = [setup_cell, resolve_cell, run_cell, result_cell, metrics_cell]

    kernel_name = _create_kernel()
    ep = ExecutePreprocessor(timeout=-1, kernel=kernel_name)
    exception = None
    metrics = None
    try:
        ep.preprocess(nb, {"metadata": {"path": trials_folder}})
        metrics = nb["cells"][-1]["outputs"][0]["text"][:-1]
    except nbclient.exceptions.CellExecutionError as exc:
        exception = exc
    return nb, metrics, exception


def _create_kernel():
    from ipykernel import kernelapp as app

    kernel_name = "".join(choices(list("0123456789"), k=6)) + str(
        hash(Path(sys.executable))
    )
    app.launch_new_instance(argv=["install", "--user", "--name", kernel_name])
    return kernel_name


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
    ex_name: str,
    group: Optional[str],
    debug: bool,
    variant: str,
    iteration: int,
    suffix: Optional[str],
    extension: str,
) -> str:
    new_filename = f"trial_{ex_name}"
    if group is not None:
        new_filename += f"_{group}"
    new_filename += f"_{variant}_{iteration}_{suffix}"
    if debug:
        new_filename += "debug"
    elif extension == "html":
        new_filename += ".html"
    elif extension == "ipynb":
        new_filename += ".ipynb"
    return new_filename


def _log_start_experiment(
    name: str,
    exp_logger: logging.Logger,
    variant: str,
    iteration: int,
    commit: str,
    debug: bool,
) -> float:
    exp_logger.info(f"trial entry: insert--{variant}----{iteration}--{commit}------")
    utc_now = datetime.now(timezone.utc)
    cpu_now = process_time()
    log_msg = (
        f"Running experiment {name}, simulation variant {variant}, "
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
    with open(folder / "config.txt", "w") as f:
        f.write(pretty)
