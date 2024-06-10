from functools import lru_cache
from pathlib import Path
from typing import Any
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
    config = _load_config(proj_file)
    try:
        result = config["steps"]
    except KeyError:
        raise RuntimeError(
            f"{proj_file.absolute()} does not have a tool.mitosis.steps table"
        )
    if any(not isinstance(vals, list) or len(vals) != 2 for vals in result.values()):
        raise RuntimeError("tool.mitosis.steps table is malformed")
    return {k: tuple(v)[:2] for k, v in result.items()}


@lru_cache
def get_repo() -> git.Repo:
    repo = git.Repo(Path.cwd(), search_parent_directories=True)
    return repo


@lru_cache
def _choose_toml(filename: Path | str | None) -> Path:
    """Identify the absolute location of the project file"""
    repo = get_repo()
    directory = Path(repo.working_dir).absolute()
    if filename is None:
        return directory / "pyproject.toml"
    elif not Path(filename).is_absolute():
        return directory / filename
    return Path(filename)


def _load_config(toml_pth: Path) -> dict[str, Any]:
    """Load the mitosis section of the config dictionary

    Raises:
        KeyError if no tool.mitosis table exists
    """
    if not toml_pth.is_absolute():
        raise ValueError("Resolve path prior to final loading step")
    with open(toml_pth, "rt", encoding="utf-8") as f:
        config = toml.load(f)
    return config["tool"]["mitosis"]


def locate_trial_folder(
    hexstr: Optional[str] = None,
    *,
    trials_folder: Optional[Path | str] = None,
    proj_file: str = "pyproject.toml",
) -> Path:
    """Identify where trials are saved.

    Args:
        hexstr: the hexstring of a trial.  Default (None) returns the
            directory of all trials.  If not none, returns the metadata
            folder for that trial
        trials_folder: relative or absolute path of trials folder.  Default
            (None) looks for a location in the project config file, or if
            missing, uses <repository root> / 'trials'.
        proj_file: path to the toml config for this project.
    """
    if trials_folder is None:
        try:
            config = _load_config(_choose_toml(proj_file))
            trials_folder = Path(config["trials-folder"])
        except KeyError:
            trials_folder = Path("trials")
    else:
        trials_folder = Path(trials_folder).resolve()
    if not trials_folder.is_absolute():
        trials_folder = Path(get_repo().working_dir).absolute() / trials_folder
    if not hexstr:
        return trials_folder

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
