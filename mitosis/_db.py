from pathlib import Path
from typing import Optional

from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import Engine
from sqlalchemy import Float
from sqlalchemy import Insert
from sqlalchemy import insert
from sqlalchemy import inspection
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import Update
from sqlalchemy import update

from mitosis._typing import Parameter


def trials_columns() -> list[Column]:
    return [
        Column("variant", String, primary_key=True),
        Column("iteration", Integer, primary_key=True),
        Column("commit", String, nullable=False),
        Column("cpu_time", Float),
        Column("results", String),
        Column("filename", String),
    ]


def variant_types() -> list[Column]:
    return [
        Column("name", String, primary_key=True),
        Column("params", String, unique=False),
    ]


def create_trials_db_eng(filename: Path | str, table_name: str) -> tuple[Engine, Table]:
    if Path(filename).is_absolute():
        db = Path(filename)
    else:
        db = Path(__file__).resolve().parent / filename

    md = MetaData()
    table = Table(table_name, md, *trials_columns())
    url = "sqlite:///" + str(db)
    eng = create_engine(url)
    with eng.begin() as conn:
        if not inspection.inspect(conn).has_table(table_name):
            md.create_all(conn)
    return eng, table


def record_start_in_db(
    tb: Table, eng: Engine, variant: str, iteration: int, commit: str
) -> Insert:
    stmt = insert(tb)
    stmt = stmt.values({"variant": variant, "iteration": iteration, "commit": commit})
    with eng.begin() as conn:
        conn.execute(stmt)
    return stmt


def record_finish_in_db(
    tb: Table,
    eng: Engine,
    variant: str,
    iteration: int,
    metric: Optional[str],
    filename: str,
    total_time: float,
) -> Update:
    stmt = update(tb)
    # primary keys... type ignore is only needed in mypy, not pyright
    stmt = stmt.where(tb.columns.get("variant") == variant)  # type: ignore
    stmt = stmt.where(tb.columns.get("iteration") == iteration)  # type: ignore
    stmt = stmt.values(
        {
            tb.columns.get("cpu_time"): total_time,
            tb.columns.get("results"): metric,
            tb.columns.get("filename"): filename,
        }
    )
    with eng.begin() as conn:
        conn.execute(stmt)
    return stmt


def _init_variant_table(trial_db: Path, step: str, param: Parameter) -> Table:
    eng = create_engine("sqlite:///" + str(trial_db))
    md = MetaData()
    var_table = Table(f"{step}_variant_{param.arg_name}", md, *variant_types())
    inspector = inspection.inspect(eng)
    if not inspector.has_table(f"{step}_variant_{param.arg_name}"):
        md.create_all(eng)
    return var_table


def _id_variant_iteration(eng: Engine, trials_table: Table, master_variant: str) -> int:
    """Identify the iteration for this exact variant of an experiment

    Args:
        trial_db: location of the trial log database
        trials_table: the main record of each
            trial/variant
        master_variant: a string that uniquely identifies a combination of parameters.
    """
    stmt = select(trials_table.c.iteration).where(
        trials_table.c.variant == master_variant
    )
    with eng.connect() as conn:
        rows = list(conn.execute(stmt))
    if len(rows) == 0:
        return 0
    else:
        return max(row[0] for row in rows) + 1
