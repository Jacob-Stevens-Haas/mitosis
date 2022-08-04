import logging
import re
import sys
import warnings
from collections import namedtuple
from collections import OrderedDict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from time import process_time
from typing import List

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
        Column("params", String, unique=True),
    ]


Parameter = namedtuple("Parameter", ["id_name", "arg_name", "vals"])


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


def _verify_variant_name(trial_db, param: Parameter):
    """Check for conflicts and register variant name in parameter table"""
    eng = create_engine("sqlite:///" + str(trial_db))
    md = MetaData()
    tb = Table(f"variant_{param.arg_name}", md, *variant_types())
    vals = OrderedDict({k: v for k, v in sorted(param.vals.items())})
    df = pd.read_sql(select(tb), eng)
    ind_equal = df.loc[:, "params"] == str(vals)
    if ind_equal.sum() == 0:
        stmt = insert(tb, values={"name": param.id_name, "params": str(vals)})
        eng.execute(stmt)
    elif df.loc[ind_equal, "name"].iloc[0] != param.id_name:
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
    logfile="trials.db",
    params: List[Parameter] = None,
    trials_folder=Path(__file__).absolute().parent / "trials",
    output_extension: str = "html",
    matplotlib_dpi: int = 72,
):
    """Run the selected experiment.

    Arguments:
        ex: The experiment class to run
        debug (bool): Whether to run in debugging mode or not.
        logfile (str): the database log for trial results
        params: The assigned parameter dictionaries to generate and
            solve the problem.
        trials_folder (path-like): The folder to store both output and
            logfile.
        output_extension: what output type to produce using nbconvert.
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
    exp_logger, trials_table = _init_logger(trial_db, f"trials_{ex.name}", debug)
    for param in params:
        _init_variant_table(trial_db, param)
        _verify_variant_name(trial_db, param)
    id_names = [param.id_name for param in params]
    arg_names = [param.arg_name for param in params]
    master_variant = "-".join(
        [x for _, x in sorted(zip(arg_names, id_names), key=lambda pair: pair[0])]
    )

    iteration = _id_variant_iteration(trial_db, trials_table, master_variant)
    debug_suffix = "_" + "".join(np.random.choice(list("0123456789abcde"), 6))
    new_filename = f"trial{ex.name}_{master_variant}_{iteration}"
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

    run_args = {param.arg_name: param.vals for param in params}
    run_args["seed"] = seed
    nb, metrics, exc = _run_in_notebook(ex, run_args, trials_folder, matplotlib_dpi)

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


def _run_in_notebook(ex: type, run_args, trials_folder, matplotlib_dpi=72):
    code = (
        "import importlib\n"
        "from pathlib import Path\n"
        "import matplotlib as mpl\n"
        "from numpy import array\n"
        "import importlib\n"
        "import sys\n"
        f'mod_path = Path("{ex.__file__}")\n'
        f'spec = importlib.util.spec_from_file_location("{ex.__name__}", mod_path)\n'
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        f"mpl.rcParams['figure.dpi'] = {matplotlib_dpi}\n"
        f"mpl.rcParams['savefig.dpi'] = {matplotlib_dpi}\n"
        f"args = {run_args}\n"
        f"print('Imported {ex.name} from {ex.__file__}')\n"
        'print(f"Running {module.__name__}.run() with parameters {args}")\n'
        'seed = args.pop("seed")\n'
    )

    nb = nbformat.v4.new_notebook()
    setup_cell = nbformat.v4.new_code_cell(source=code)
    run_cell = nbformat.v4.new_code_cell(source="results = module.run(seed, **args)")
    final_cell = nbformat.v4.new_code_cell(source="print(repr(results))")
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
