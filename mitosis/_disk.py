from functools import lru_cache
from pathlib import Path
from typing import Optional

import git
import toml


@lru_cache
def load_mitosis_steps(
    proj_file: Path | str = "pyproject.toml",
) -> dict[str, tuple[str, str]]:
    """Load experiment steps saved in pyproject.toml (or other config)

    Args:
        pyproj_file:
            Local or absolute path to pyproject file.  default is pyproject.toml
            if local path or default, use current git repo's top level directory
    Raises:
        Runtime error if cannot load steps, or if they are badly formed.
    """
    proj_file = _choose_toml(proj_file)
    with open(proj_file, "r") as f:
        config = toml.load(f)
    try:
        result = config["tool"]["mitosis"]["steps"]
    except KeyError:
        raise RuntimeError(
            f"{proj_file.absolute()} does not have a tools.mitosis.steps table"
        )
    if any(not isinstance(vals, list) or len(vals) != 2 for vals in result.values()):
        raise RuntimeError("tool.mitosis.steps table is malformed")
    return {k: tuple(v) for k, v in result.items()}


@lru_cache
def get_repo() -> git.Repo:
    repo = git.Repo(Path.cwd(), search_parent_directories=True)
    return repo


def _choose_toml(filename: Path | str | None) -> Path:
    repo = get_repo()
    directory = Path(repo.working_dir)
    if filename is None:
        return directory / "pyproject.toml"
    elif not Path(filename).is_absolute():
        return directory / filename
    return Path(filename)


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
