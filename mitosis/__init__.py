import logging
import re
import sys
import warnings
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

REPO = git.Repo(Path(__file__).parent.parent.parent)


def trials_columns():
    return [
        Column("id", Integer, primary_key=True),
        Column("variant", Integer, primary_key=True),
        Column("iteration", Integer, primary_key=True),
        Column("commit", String, nullable=False),
        Column("cpu_time", Float),
        Column("results", String),
        Column("filename", String),
    ]


def trial_types():
    return [
        Column("id", Integer, primary_key=True),
        Column("short_name", String, unique=True),
        Column("prob_params", String, unique=True),
    ]


def variant_types():
    return [
        Column("variant", Integer, primary_key=True),
        Column("short_name", String, unique=True),
        Column("sim_params", String, unique=True),
    ]


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


def _init_id_variant_tables(trial_log):
    eng = create_engine("sqlite:///" + str(trial_log))
    md = MetaData()
    id_table = Table("trial_types", md, *trial_types())
    var_table = Table("variant_types", md, *variant_types())
    inspector = inspection.inspect(eng)
    if not inspector.has_table("trial_types") and not inspector.has_table(
        "variant_types"
    ):
        md.create_all(eng)
    return id_table, var_table


def _id_variant_iteration(
    trial_log,
    trials_table,
    *,
    var_table,
    sim_params,
    id_table,
    prob_params,
):
    """Identify, from the db_log, which trial id and variant the current
    problem matches, then give the iteration.  If no matches are found,
    increment id or variant appropriately.

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
    sim_params = dict(sim_params)
    prob_params = dict(prob_params)

    def lookup_or_add_params(tb, cols, params, index, lookup_col):
        params = OrderedDict({k: v for k, v in sorted(dict(params).items())})
        df = pd.read_sql(select(tb), eng)
        ind_equal = df.loc[:, lookup_col] == str(params)
        if ind_equal.sum() == 0:
            new_val = 1 if df.empty else df[index].max() + 1
            stmt = insert(tb, values={index: int(new_val), lookup_col: str(params)})
            eng.execute(stmt)
            return new_val, True
        else:
            return df.loc[ind_equal, index].iloc[0], False

    trial_id, new_id = lookup_or_add_params(
        id_table, trial_types(), prob_params, "id", "prob_params"
    )
    variant, new_var = lookup_or_add_params(
        var_table, variant_types(), sim_params, "variant", "sim_params"
    )
    if new_var or new_id:
        iteration = 1
    else:
        stmt = select(trials_table).where(
            (trials_table.c.id == int(trial_id))
            & (trials_table.c.variant == int(variant))
        )
        df = pd.read_sql(stmt, eng)
        if df.empty:
            # an interruption must have occurred in a previous trial
            # after trial_id and variant were created, but before
            # actual results were logged
            iteration = 1
        else:
            iteration = df["iteration"].max() + 1

    return trial_id, variant, iteration


def run(
    ex: Experiment,
    debug=False,
    *,
    logfile="trials.db",
    prob_params=None,
    sim_params=None,
    trials_folder=Path(__file__).absolute().parent / "trials",
    output_extension: str = "html",
    matplotlib_dpi: int = 72,
):
    """Run the selected experiment.

    Arguments:
        ex: The experiment class to run
        debug (bool): Whether to run in debugging mode or not.
        logfile (str): the database log for trial results
        prob_params: The parameters ex uses to solve the problem
        sim_params: The parameters ex uses to generate a problem
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
    exp_logger, trials_table = _init_logger(trial_db, "trials", debug)
    id_table, var_table = _init_id_variant_tables(trial_db)
    trial, variant, iteration = _id_variant_iteration(
        trial_db,
        trials_table,
        sim_params=sim_params,
        var_table=var_table,
        prob_params=prob_params,
        id_table=id_table,
    )
    debug_suffix = "_" + "".join(np.random.choice(list("0123456789abcde"), 6))
    new_filename = f"trial{trial}_{variant}_{iteration}"
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
        + f"--{trial}"
        + f"--{variant}"
        + f"--{iteration}"
        + f"--{commit}"
        + "--"
        + "--"
        + "--None"
    )
    utc_now = datetime.now(timezone.utc)
    cpu_now = process_time()
    if isinstance(ex, type):
        log_msg = (
            f"Running experiment {ex.__name__}, trial {trial}, simulation type"
            f" {variant} at time: "
            + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
            + f".  Current repo hash: {commit}"
        )
    else:
        log_msg = (
            f"Running experiment {ex.name}, trial {trial}, simulation type"
            f" {variant} at time: "
            + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
            + f".  Current repo hash: {commit}"
        )
    if debug:
        log_msg += ".  In debugging mode."
    exp_logger.info(log_msg)

    if isinstance(ex, type):
        nb, metrics = _run_in_notebook_if_possible(
            ex, sim_params, prob_params, trials_folder, matplotlib_dpi
        )
    else:
        warnings.warn(
            "Passing an experiment object is deprecated.  Pass an experiment"
            " class, with sim_params, and prob_params separately"
        )
        nb = None
        metrics = ex.run()["metrics"]

    utc_now = datetime.now(timezone.utc)
    exp_logger.info(
        "Finished experiment at time: "
        + utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")
        + f".  Results: {metrics}"
    )
    cpu_time = process_time() - cpu_now

    if isinstance(ex, type) and new_filename is not None:
        _save_notebook(nb, new_filename, trials_folder, output_extension)
    else:
        warnings.warn("Logging trial and mock filename, but no file created")
    exp_logger.info(
        "trial entry: update"
        + f"--{trial}"
        + f"--{variant}"
        + f"--{iteration}"
        + f"--{commit}"
        + f"--{cpu_time}"
        + f"--{metrics}"
        + f"--{new_filename}"
    )
    return None


def _run_in_notebook_if_possible(
    ex: type, sim_params, prob_params, trials_folder, matplotlib_dpi=72
):
    mod_name = ex.__module__
    code = (
        "import importlib\n"
        "import matplotlib as mpl\n"
        "from numpy import array\n"
        f"mpl.rcParams['figure.dpi'] = {matplotlib_dpi}\n"
        f"mpl.rcParams['savefig.dpi'] = {matplotlib_dpi}\n"
        f"experiment_module = importlib.import_module('{mod_name}')\n"
        f"experiment_class = experiment_module.{ex.__name__}\n"
        f"sim_params = {sim_params}\n"
        f"prob_params = {prob_params}\n"
        "ex = experiment_class(**sim_params, **prob_params)\n"
        "print('Imported ' + experiment_class.__name__ +"
        "' from ' + experiment_module.__name__)"
    )

    nb = nbformat.v4.new_notebook()
    setup_cell = nbformat.v4.new_code_cell(source=code)
    run_cell = nbformat.v4.new_code_cell(source="results = ex.run()")
    final_cell = nbformat.v4.new_code_cell(source="print(repr(results))")
    nb["cells"] = [setup_cell, run_cell, final_cell]

    kernel_name = _create_kernel()
    ep = ExecutePreprocessor(timeout=-1, kernel=kernel_name)
    try:
        ep.preprocess(nb, {"metadata": {"path": trials_folder}})
    except nbclient.client.CellExecutionError:
        pass
    try:
        result_string = nb["cells"][2]["outputs"][0]["text"][:-1]
        metrics = _parse_results(result_string)
    except (IndexError, KeyError):
        metrics = None
    return nb, metrics


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
    match = re.search(r"'metrics': (.*)}", result_string, re.DOTALL)
    return list(eval(match.group(1)))
