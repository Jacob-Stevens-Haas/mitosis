[![Documentation Status](https://readthedocs.org/projects/mitosis/badge/?version=latest)](https://mitosis.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mitosis.svg)](https://badge.fury.io/py/mitosis)
[![Downloads](https://pepy.tech/badge/mitosis)](https://pepy.tech/project/mitosis)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# `mitosis`
_Manage (and reproduce) computational experiments_

Philosophically, an experiment is any time we run code with an aim to convince someone
of something.  As code, an experiment is a callable, registered with the mitosis
tool in pyproject.toml
The value of mitosis is
automatically recording commit and parameterization of each trial, as well as html
visuals.  It helpfully refuses to run if the git repo is dirty (and thus unreproducible)
or if parameter names have changed from a prior trial (e.g. the meaning of "low noise").
The virtuous consequence of these checks and organization is reduced mental overhead of collaboration.


## Trivial Example

Hypothesis: the maximum value of a sine wave is equal to its amplitude.

*sine_experiment/\_\_init\_\_.py*


    import numpy as np
    import matplotlib.pyplot as plt

    name = "sine-exp"
    lookup_dict = {"frequency": {"fast": 10, "slow": 1}}

    def run(amplitude, frequency):
        """Deterimne if the maximum value of the sine function equals ``amplitude``"""
        x = np.arange(0, 10, .05)
        y = amplitude * np.sin(frequency * x)
        err = np.abs(max(y) - amplitude)
        plt.title("What's the maximum value of a sine wave?")
        plt.plot(x, y, label="trial data")
        plt.plot(x, amplitude * np.ones_like(x), label="expected")
        plt.legend()
        return {"main": err, "data": y}


*pyproject.toml*

    [tool.mitosis.steps]
    my_exp = ["sine_experiment:run", "sine_experiment:lookup_dict"]


Commit these changes to a repository.  After installing sine_experiment as a python package, in CLI, run:

    mitosis my_exp --param my_exp.frequency=slow --eval-param my_exp.amplitude=4

Mitosis will run `sin_experiment.run()`, saving
all output as an html file in a subdirectory.  It will also
track the parameters and results.
If you later change the variant named "slow" to set frequency=2, mitosis will
raise a `RuntimeError`, preventing you from running a trial.  If you want to run
`sine_experiment` with a different parameter value, you need to name that variant
something new.  Eval parameters, like "amplitude" in the example, behave differently.
Rather than being specified by `lookup_dict`, they are evaluated directly.


# Use

Using mitosis involves registering experiments in pyproject.toml, enumerating parameter
values in a lookup dictionary, running experiments on the command line, and browsing
results.

## Registration

mitosis uses the `tool.mitosis.steps` table of pyproject.toml to learn what python
callables are named experiment steps and where to lookup named parameter values.  It
uses a syntax evocative of entry points.

    [tool.mitosis.steps]
    my_exp = ["sine_experiment:run", "sine_experiment:lookup_dict"]

Experiment steps must be callables with a dictionary return type.  The returned
dictionary is required to have a key "main".  All but the final step in an experiment
must also have a key "data" that gets passed to the first argument of the subsequent
step.

_Developer note: Building an experiment step static type at_ `mitosis._typing.ExpRun`

## CLI

The basic invocation lists the steps along with the values of any parameters for each
step.

    mitosis [OPTION...] step [steps...] [[-p step.lookup_param=key...]
        [-e step.eval_param=val...]]...

Some nuance:
* `--debug` can be used to waive a lot of the reproducibility checks mitosis does.
    This arg allows you to run experiments in a dirty git repository (or no repository)
    and will neither save results in the experimental database, nor increment the trials
    counter, nor verify/lock in the definitions of any variants.  It will, however,
    create the output notebook.  It also changes the experiment log level  from INFO
    to DEBUG.
* lookup parameters can be nearly any python object that is pickleable.  Tracking
    parameter values can be turned off for parameters either for something that isn't
    pickleable (e.g. a lambda function) or isn't important to track
    (e.g. which GPU to run on).  This can be done with eval or lookup parameters
    by adding a `+` to the parameter, e.g. `-e +jax_playground.gpu_id=1`.
* Eval parameters which are strings will need quotation marks that escape the shell
    (e.g. `-e smoothing.kernel=\"rbf\"`)
* `-e` and `-p` are short form for `--eval-param` and `--param` (lookup param).

## Results

Trials are saved in `trials/` (or whatever is passed after `-F`).  Each trial has a
pseudorandom bytes key, postpended to a metadata folder and an html output filename.

There are two obviously useful things to do after an experiment:
* view the html file.  `python -m http.server` is helpful to browse results
* load the data with `mitosis.load_trial_data()`

Beyond this, the metadata mitosis keeps to disk is useful for troubleshooting or reproducing experiments, but no facility yet exists to browse or compare experiments.

## API

Mitosis is primarily intended as a command line program, so `mitosis --help` has the syntax documentation.
There is only one intentionally public part of the api: `mitosis.load_trial_data()`.
