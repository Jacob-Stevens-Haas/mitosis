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
    # primary keys
    stmt = stmt.where(tb.columns.get("variant") == variant)
    stmt = stmt.where(tb.columns.get("iteration") == iteration)
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
