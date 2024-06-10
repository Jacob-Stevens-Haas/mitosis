import contextlib
import os
import sys
from argparse import Namespace
from io import StringIO
from typing import Generator

import pytest

from mitosis.__main__ import _create_parser
from mitosis.__main__ import _lookup_step_names
from mitosis.__main__ import _process_cl_args
from mitosis.__main__ import _split_param_str
from mitosis.__main__ import main
from mitosis.__main__ import normalize_modinput
from mitosis.tests.mock_legacy import lookup_dict
from mitosis.tests.mock_legacy import run
from mitosis.tests.mock_paper import meth_config
from mitosis.tests.mock_part1 import Klass


def test_cli_step_ordering():
    # GH 48
    expected = ["fit_eval", "data"]
    args = Namespace(
        experiment=expected,
        module=None,
        debug=True,
        config="mitosis/tests/test_pyproject.toml",
        trials_folder=None,
        eval_param=[],
        param=[],
    )
    result = [step.name for step in _process_cl_args(args)["steps"]]
    assert result == expected


def test_lookup_step_names_ordering():
    # GH 48
    expected = ["fit_eval", "data"]
    steps = _lookup_step_names(expected, "mitosis/tests/test_pyproject.toml")
    result = list(steps.keys())
    assert result == expected


@pytest.mark.parametrize(
    ("params", "eval_params"),
    (([], []), (["foo=test"], []), ([], ["seed=1"])),
    ids=("no args", "lookup", "eval"),
)
def test_legacy_module(params, eval_params):
    args = Namespace(
        experiment=[],
        module="mitosis.tests.mock_legacy",
        debug=True,
        config="pyproject.toml",
        trials_folder=None,
        eval_param=eval_params,
        param=params,
    )
    result = _process_cl_args(args)
    assert len(result["steps"]) == 1
    assert result["steps"][0].name == "mitosis.tests.mock_legacy"
    assert result["steps"][0].action == run
    assert id(result["steps"][0].lookup) == id(lookup_dict)


def test_experiment_arg():
    args = Namespace(
        experiment=["data", "fit_eval"],
        module=None,
        debug=True,
        config="mitosis/tests/test_pyproject.toml",
        trials_folder=None,
        eval_param=["data.extra=True"],
        param=["data.length=test", "fit_eval.metric=test"],
    )
    result = _process_cl_args(args)
    assert len(result["steps"]) == 2
    assert [step.name for step in result["steps"]] == ["data", "fit_eval"]
    assert result["steps"][0].action == Klass.gen_data
    assert id(result["steps"][1].lookup) == id(meth_config)


def test_folder_arg(tmp_path):
    parser = _create_parser()
    args = parser.parse_args(["-m", "mitosis.tests.mock_legacy", "-F", "foo"])

    @contextlib.contextmanager
    def change_cwd(new_pth: os.PathLike) -> Generator[None, None, None]:
        temp = os.getcwd()
        try:
            os.chdir(new_pth)
            yield
        finally:
            os.chdir(temp)

    with change_cwd(tmp_path):
        result = _process_cl_args(args)["trials_folder"]
    expected = tmp_path / "foo"
    assert result == expected


def test_argparse_options():
    parser = _create_parser()
    args = parser.parse_args(
        ["-m", "mod", "-d", "--config", "foo.toml", "-F", "foo/bar"]
    )
    assert args.experiment == []
    assert args.module == "mod"
    assert args.debug is True
    assert args.config == "foo.toml"
    assert args.trials_folder == "foo/bar"
    assert args.eval_param == []
    assert args.param == []


def test_argparse_main():
    parser = _create_parser()
    args = parser.parse_args(
        [
            "step1", "step2", "-e", "a=1", "-e", "b=2", "-p", "c=d", "-p", "e=f", "-p", "g=h"  # fmt: skip; # noqa: E501
        ]
    )
    assert args.experiment == ["step1", "step2"]
    assert args.module is None
    assert args.debug is False
    assert args.config == "pyproject.toml"
    assert args.trials_folder is None
    assert len(args.eval_param) == 2
    assert len(args.param) == 3


def test_split_param_str():
    result = _split_param_str("+a=b")
    assert result == ("", False, "a", "b")
    result = _split_param_str("a.b=c")
    assert result == ("a", True, "b", "c")


def test_normalize_modinput():
    modinput = "mitosis.tests.mock_experiment"
    result = normalize_modinput(modinput)
    assert result == {
        "mitosis.tests.mock_experiment": (
            "mitosis.tests.mock_experiment:run",
            "mitosis.tests.mock_experiment:lookup_dict",
        )
    }
    # if modinput is an object, connect to run and lookup_dict with . not :
    modinput = "mitosis.tests.mock_experiment:MockExp.MockExpInner"
    result = normalize_modinput(modinput)
    assert result == {
        "mitosis.tests.mock_experiment:MockExp.MockExpInner": (
            "mitosis.tests.mock_experiment:MockExp.MockExpInner.run",
            "mitosis.tests.mock_experiment:MockExp.MockExpInner.lookup_dict",
        )
    }


def test_version():
    @contextlib.contextmanager
    def set_argv(*args: str) -> Generator[None, None, None]:
        temp = sys.argv
        try:
            sys.argv = list(args)
            yield
        finally:
            sys.argv = temp

    @contextlib.contextmanager
    def capture_stdout() -> Generator[StringIO, None, None]:
        stdout = sys.stdout
        new_out = StringIO()
        try:
            sys.stdout = new_out
            yield new_out
        finally:
            sys.stdout = stdout

    with set_argv("mitosis", "--version"):
        with capture_stdout() as out:
            main()
            result = out.getvalue()
    assert result.split()[0] == "mitosis"
