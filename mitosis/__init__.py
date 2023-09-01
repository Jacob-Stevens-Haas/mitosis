import logging
import pickle
import re
import sys
import warnings
from collections import namedtuple
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timezone
from pathlib import Path
from time import process_time
from types import ModuleType
from typing import List, Collection, Mapping, Any, Optional

import git
import nbclient
import nbformat
import numpy as np
import pandas as pd
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.writers import FilesWriter
from numpy import array  # noqa - used in an eval() in _parse_results()
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

REPO = git.Repo(Path.cwd(), search_parent_directories=True)

ModuleInfo = list[tuple[ModuleType, Optional[Collection[str]]]]


def trials_columns():
    return [
        Column("variant", Integer, primary_key=True),
        Column("iteration", Integer, primary_key=True),
        Column("seed", Integer, primary_key=True),
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
        id_name: short name for the variant (particular values) across use cases
        arg_name: name of arg known to experiment
        vals: value of the parameter
        modules: module names required in order to use the values.  Since arguments
            can only be passed to notebooks as strings, any argument that cannot be
            simply recreated from its repr will need to be pickled and the containing
            module imported.
    """

    id_name: str
    arg_name: str
    vals: Any
    modules: List[str] = field(default_factory=list)


def _finalize_param(param: Parameter, folder: Path | str):
    filename = "arg" + "".join(np.random.choice(list("0123456789abcde"), 9)) + ".pickle"
    location = Path(folder) / filename
    with open(location, "wb") as fh:
        pickle.dump(param.vals, fh)
    return location


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


class Experiment:
    def run():
        raise NotImplementedError


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
    if isinstance(param.vals, Mapping):
        vals = OrderedDict({k: v for k, v in sorted(param.vals.items())})
    elif isinstance(param.vals, Collection) and not isinstance(param.vals, str):
        try:
            vals = sorted(param.vals)
        except (ValueError, TypeError):
            vals = param.vals
    else:
        vals = param.vals
    df = pd.read_sql(select(tb), eng)
    ind_equal = df.loc[:, "name"] == param.id_name
    if ind_equal.sum() == 0:
        stmt = insert(tb, values={"name": param.id_name, "params": str(vals)})
        eng.execute(stmt)
    elif df.loc[ind_equal, "params"].iloc[0] != str(vals):
        raise RuntimeError(
            f"Parameter name {param.id_name} "
            f"is stored with different values in {trial_db}, {tb}"
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


def run(
    ex: Experiment,
    debug=False,
    seed=1,
    *,
    group=None,
    logfile="trials.db",
    params: List[Parameter] = None,
    trials_folder=Path(__file__).absolute().parent / "trials",
    output_extension: str = "html",
    addl_mods_and_names: ModuleInfo = None,
    untracked_params: Collection[str]=None,
    matplotlib_dpi: int = 72,
):
    """Run the selected experiment.

    Arguments:
        ex: The experiment class to run
        debug (bool): Whether to run in debugging mode or not.
        group (str): Trial grouping.  Name a group if desiring to
            segregate trials using the same experiment code.  ex.run()
            must take a "group" argument.
        logfile (str): the database log for trial results
        params: The assigned parameter dictionaries to generate and
            solve the problem.
        trials_folder (path-like): The folder to store both output and
            logfile.
        output_extension: what output type to produce using nbconvert.
        addl_mods_and_names: Additional modules names required to
            run experiment as well as names from those modules.
        untracked_params: names of parameters to not track in database
        matplotlib_resolution: dpi for matplotlib images.  Not yet
            functional.
    """
    if not debug and REPO.is_dirty():
        raise RuntimeError(
            "Git Repo is dirty.  For repeatable tests,"
            " clean the repo by committing or stashing all changes and "
            "untracked files."
        )
    trial_db = Path(trials_folder).absolute() / logfile
    table_name = f"trials_{ex.name}"
    if group is not None:
        table_name += f" {group}"
    exp_logger, trials_table = _init_logger(trial_db, f"trials_{ex.name}", debug)
    for param in params:
        if param.arg_name in untracked_params:
            continue
        _init_variant_table(trial_db, param)
        _verify_variant_name(trial_db, param)
    id_names = [param.id_name for param in params]
    arg_names = [param.arg_name for param in params]
    master_variant = "-".join(
        [x for _, x in sorted(zip(arg_names, id_names), key=lambda pair: pair[0])]
    )

    iteration = _id_variant_iteration(trial_db, trials_table, master_variant)
    debug_suffix = "_" + "".join(np.random.choice(list("0123456789abcde"), 6))
    new_filename = f"trial_{ex.name}"
    if group is not None:
        new_filename += f"_{group}"
    new_filename += f"_{master_variant}_{iteration}"
    if debug:
        new_filename += debug_suffix
    if output_extension is None:
        new_filename = None
    elif output_extension == "html":
        new_filename += ".html"
    elif output_extension == "ipynb":
        new_filename += ".ipynb"
    commit = REPO.head.commit.hexsha
    exp_logger.info(
        "trial entry: insert"
        + f"--{master_variant}"
        + f"--{iteration}"
        + f"--{seed}"
        + f"--{commit}"
        + "--"
        + "--"
        + "--None"
    )
    utc_now = datetime.now(timezone.utc)
    cpu_now = process_time()
    log_msg = (
        f"Running experiment {ex.name}, simulation variant {master_variant}, "
        f"iteration {iteration} at time: "
        + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
        + f".  Current repo hash: {commit}"
    )
    if debug:
        log_msg += ".  In debugging mode."
    exp_logger.info(log_msg)

    nb, metrics, exc = _run_in_notebook(
        ex, seed, group, params, trials_folder, addl_mods_and_names, debug_suffix, matplotlib_dpi
    )

    utc_now = datetime.now(timezone.utc)
    exp_logger.info(
        "Finished experiment at time: "
        + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
        + f".  Results: {metrics}"
    )
    cpu_time = process_time() - cpu_now

    if new_filename is not None:
        _save_notebook(nb, new_filename, trials_folder, output_extension)
    else:
        warnings.warn("Logging trial and mock filename, but no file created")

    exp_logger.info(
        "trial entry: update"
        + f"--{master_variant}"
        + f"--{iteration}"
        + f"--{seed}"
        + f"--{commit}"
        + f"--{cpu_time}"
        + f"--{metrics}"
        + f"--{new_filename}"
    )
    if exc is not None:
        raise exc


def _run_in_notebook(
    ex: type,
    seed,
    group,
    params,
    trials_folder,
    addl_mods_and_names: ModuleInfo,
    results_suffix: str,
    matplotlib_dpi=72,
):
    run_args = {param.arg_name: param.vals for param in params if not param.modules}
    run_args["seed"] = seed
    if group is not None:
        run_args["group"] = group

    pickles = {
        param.arg_name: str(_finalize_param(param, trials_folder))
        for param in params
        if param.modules
    }
    mod_names_and_paths = [
        (mod.__name__, mod.__file__, []) for param in params for mod in param.modules
    ]
    mod_names_and_paths += [
        (mod.__name__, mod.__file__, names) for mod, names in addl_mods_and_names
    ]
    code = (
        "import importlib\n"
        "import matplotlib as mpl\n"
        "import numpy as np\n"
        "import pickle\n"
        "import sys\n\n"
        f"mods = {mod_names_and_paths}\n"
        "for modname, mod_path, names in mods:\n"
        "  mod = importlib.import_module(modname, str(mod_path))\n"
        "  for name in names:\n"
        "    globals()[name] = vars(mod)[name]\n"
        f'module = importlib.import_module("{ex.__name__}")\n\n'
        "def unpickle(file):\n"
        "  with open(file, 'rb') as fh:\n"
        "    obj = pickle.load(fh)\n"
        "  return obj\n\n"
        f"args = {run_args}\n"
        f"pickles = {pickles}\n"
        "for a_name, a_pickle in pickles.items():\n"
        f"  args[a_name] = unpickle(a_pickle)\n\n"
        f"mpl.rcParams['figure.dpi'] = {matplotlib_dpi}\n"
        f"mpl.rcParams['savefig.dpi'] = {matplotlib_dpi}\n"
        f"print('Imported {ex.name} from {ex.__file__}')\n"
        'print(f"Running {module.__name__}.run() with parameters {args}")\n'
        'seed = args.pop("seed")\n'
    )

    nb = nbformat.v4.new_notebook()
    setup_cell = nbformat.v4.new_code_cell(source=code)
    run_cell = nbformat.v4.new_code_cell(source="results = module.run(seed, **args)")
    final_cell = nbformat.v4.new_code_cell(
        source=""
        f"with open('results{results_suffix}.npy', 'wb') as f: np.save(f, results)\n"
        "print(repr(results))\n"
    )
    nb["cells"] = [setup_cell, run_cell, final_cell]

    kernel_name = _create_kernel()
    ep = ExecutePreprocessor(timeout=-1, kernel=kernel_name)
    exception = None
    try:
        ep.preprocess(nb, {"metadata": {"path": trials_folder}})
    except nbclient.client.CellExecutionError as exc:
        exception = exc
    try:
        result_string = nb["cells"][2]["outputs"][0]["text"][:-1]
        metrics = _parse_results(result_string)
    except (IndexError, KeyError):
        metrics = None
    return nb, metrics, exception


def _create_kernel():
    from ipykernel import kernelapp as app

    kernel_name = sys.executable.replace("/", ".").replace("\\", ".").replace(":", ".")
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


def _parse_results(result_string):
    match = re.search(r"'main': (.*)}", result_string, re.DOTALL)
    return match.group(1)
