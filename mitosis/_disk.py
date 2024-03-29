from functools import lru_cache
from pathlib import Path

import git
import toml


@lru_cache
def load_mitosis_steps(
    proj_file: Path | str = "pyproject.toml",
) -> dict[str, list[str]]:
    """Load experiment steps saved in pyproject.toml (or other config)

    Args:
        pyproj_file:
            Local or absolute path to pyproject file.  default is pyproject.toml
            if local path or default, use current working directory
    """
    proj_file = _choose_toml(proj_file)
    with open(proj_file, "r") as f:
        config = toml.load(f)
    try:
        return config["tool"]["mitosis"]["steps"]
    except KeyError:
        raise RuntimeError(
            f"{proj_file.absolute()} does not have a tools.mitosis.steps table"
        )


@lru_cache
def get_repo() -> git.Repo:
    repo = git.Repo(Path.cwd(), search_parent_directories=True)
    return repo


def _choose_toml(filename: Path | str) -> Path:
    repo = get_repo()
    directory = Path(repo.working_dir)
    if filename is None:
        filename = directory / "pyproject.toml"
    else:
        if not Path(filename).is_absolute():
            filename = directory / filename
    return Path(filename)
