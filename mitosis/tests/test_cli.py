from argparse import Namespace

import pytest

from mitosis.__main__ import _create_parser
from mitosis.__main__ import _process_cl_args
from mitosis.tests.mock_legacy import lookup_dict
from mitosis.tests.mock_legacy import run
from mitosis.tests.mock_paper import meth_config
from mitosis.tests.mock_part1 import Klass


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
        folder=None,
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
        folder=None,
        eval_param=["data.extra=True"],
        param=["data.length=test", "fit_eval.metric=test"],
    )
    result = _process_cl_args(args)
    assert len(result["steps"]) == 2
    assert [step.name for step in result["steps"]] == ["data", "fit_eval"]
    assert result["steps"][0].action == Klass.gen_data
    assert id(result["steps"][1].lookup) == id(meth_config)


def test_argparse_options():
    parser = _create_parser()
    args = parser.parse_args(
        ["-m", "mod", "-d", "--config", "foo.toml", "-F", "foo/bar"]
    )
    assert args.experiment == []
    assert args.module == "mod"
    assert args.debug is True
    assert args.config == "foo.toml"
    assert args.folder == "foo/bar"
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
    assert args.folder is None
    assert len(args.eval_param) == 2
    assert len(args.param) == 3
