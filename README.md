[![Documentation Status](https://readthedocs.org/projects/mitosis/badge/?version=latest)](https://mitosis.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mitosis.svg)](https://badge.fury.io/py/mitosis)
[![Downloads](https://pepy.tech/badge/mitosis)](https://pepy.tech/project/mitosis)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# `mitosis`
_Reproduce Machine Learning experiments easily_

A package designed to manage and visualize experiments, tracking changes across
different commits, parameterizations, and random seed.

## Trivial Example

*sine_experiment.py*


    import numpy as np

    name = "sine-exp"
    lookup_dict = {"frequency": {"fast": 10, "slow": 1}}

    def run(seed, amplitude, frequency):
        """Deterimne if the maximum value of the sine function equals ``amplitude``"""
        x = np.arange(0, 10, .05)
        y = amplitude * np.sin(frequency * x)
        err = np.abs(max(y) - amplitude)
        return {"main": err}

*in interpreter or script*


    import mitosis
    import sine_experiment
    from pathlib import Path

    folder = Path(".").resolve()
    params = [
        mitosis.Parameter("4", "amplitude", 4, evaluate=True)
        mitosis.Parameter("slow", "frequency", 4, evaluate=False)
    ]

    mitosis.run(sine_experiment, params=params, trials_folder=folder)

Commit these changes to a repository.  Mitosis will run `sin_experiment.run()`, saving
all output as an html file in the current directory.  It will also
track the parameters and results.
If you later change the variant named "slow" to set frequency=.1, mitosis will
raise a RuntimeError, preventing you from running a trial.  If you want to run
`sine_experiment` with a different parameter value, you need to name that variant
something new.  Parameters like "amplitude", on the other hand, behave differently.
Rather than being specified by `lookup_dict`, they are evaluated directly.

# How it Works

## Behind the scenes:

The first time `mitosis.run()` is passed a new experiment, it takes several actions:
1. Create a database for your experiment in `trials_folder`
2. Add a table to track all of the experiment trials
3. Add tables to track all of the different variants of your experiment.
4. Create and run a jupyter notebook in memory, saving the result as an HTML file
5. Updating the database of all trials with the results of the experiment.
6. Save experiment config of parameters actually created by jupyter notebook
7. Save a freezefile of python packages installed.

In step 3, `mitosis` attempts to create a unique and reproduceable string from each
parameter value.  This is tricky, since most python objects are mutable and/or have
their `id()` stored in their string representation.  So lists must be sorted, dicts must
be ordered, and function and method arguments must be wrapped in a class with and a
custom `__str__()` attribute.  This is imperfectly done, see the **Reproduceability **
section for comments on the edge cases where `mitosis` will either treat the same
parameter as a new variant, or treat two different parameters as having the same value.

In step 4, `mitosis` needs to start the jupyter notebook with the appropriate variables.
Instead of sending the variables to the notebook, the notebook re-evaluates eval
parameters and re-looks up lookup parameters.  Previously, parameters were sent
to the notebook via pickle; that proved fragile.


The next time `mitosis.run()` is given the same experiment, it will
1. Determine whether parameter names and values match parameters in a previously established
variant.  If they do not, it will either:
   1. Reject the experiment trial if the passed parameter names match existing variants
   but with different values.
   2. Create a new variant for the parameter.
1. do steps 4 to 7 above.


## Abstractions

**Experiment** :the definition of a procedure that will test a hypothesis.
As a python object, an experiment must have a `Callable` attribute named "run"
that takes any number of arguments and returns a dict with at least a key named
"main".  It also requires a `name` attribute

In its current form, `mitosis` does not require a hypothesis, but it does
require experiments to define the "main" metric worth evaluating (though a
user can always define an experiment that merely returns a constant).

**Parameter**: An argument to an experiment.  These are the axes by which an experiment
may vary, e.g. `sim_params`, `data_params`, `solver_params`... etc.  When this argument
is a `Collection`, sometimes the singular (parameter) and plural (parameters) are used
interchangeably.  Parameters can either be lookup parameters (which require the
experiment to have an attribute `lookup_dict`) or eval parameters (which are typically
simple evaluations, e.g. setting the random seed to a given integer).  Eval parameters
which are strings need quotes.

**Variant**: An experimental parameter assigned to specific values and given a name.

**Trial**: a single run of an experiment with all variants specified.  Within `mitosis`,
the name of a trial is the experiment name, concatenated with variant names for each
argument, and suffixed by the number of times that particular trial has been run.  E.g.
If an experiment has three arguments, the first trial run could be called
"trial_sine-arg1a-arg2test-arg3foo-1"  If a bugfix is committed to the experiment and
the trial re-run with the same parameters, the new trial would be named
"trial_sine-arg1a-arg2test-arg3foo-2".

Within `mitosis`, the trial is used to name the resulting html file and is stored in
the "variant" and "iteration" columns in the experiment's sqlite database.

# CLI

See [an example](https://github.com/Jacob-Stevens-Haas/gen-experiments).

## Untracked parameters

If there are certain parameters that are not worth tracking, e.g. plotting flags
that do not change the mathematical results, prepend the argument name with "-".
An example:

```
mitosis project_pkg.exp1 -e -plot=True -p -plot_axes=dense
```

## Fast iterations: Debug

Debug is straightforwards: `mitosis project_pkg.exp1 -d ...` runs in debug mode.
This arg allows you to run experiments in a dirty git repository (or no repository)
and will neither save results in the experimental database, nor increment the trials
counter, nor verify/lock in the definitions of any variants.  It will, however,
create the output notebook.

Early experiment prototyping involves quickly iterating on parameter lists and
complexity.  `mitosis` will still lock in definitions of variants, which means
that you will likely go through variant names quickly.  This disambiguation is
somewhat intentional, but you can free up names by deleting or renaming the
experiment database or deleting records in the `variant_param_name` table.

## Sharing code between experiments: Group

If your experimental code is intended to be used for multiple dissimilar
experiments and want to track results separately, assign a group at the command
line.  The string is passed as an argument "group" to the experiment's run()
function.  It is treated as a special eval parameter, so if passed, there must
be no other params named group

Group is more complex, but a simple example will help:
```
mitosis project_pkg.pdes -g heat -p initial_condition=origin-bump
mitosis project_pkg.pdes -g heat2d -p initial_condition=origin-bump2d
mitosis project_pkg.gridsearch -g pdes-heat -p initial_condition=origin-bump
```

# A More Advanced Workflow

As soon as a research project can define a single `run()` function that specifies
an experiment, the axes/parameters by which trials differ, and the output to
measure, `mitosis` can be useful.
I have found the following structure useful:

    project_dir/
    |-- .git                As well as all other project files, e.g. tests/
    |                       pyproject.toml, .pre-commit-config.yaml...
    |-- project_pkg/
        |-- __init__.py     The definitions of variant names that are common
        |                   to multiple experiments and referenced in each
        |                   experiment's lookup_dict
        |-- exp1.py         One experiment to run
        |-- exp2.py         Another experiment to run
        |-- _common.py      or _utils.py.py, basically anything needed by
        |                   other expermints such as common plotting functions
        |-- trials/         The folder passed to mitosis.run() to store results

Most of this is common across all packages and is basic engineering discipline.
If project_pkg is installed, it allows mitosis's CLI to be called as:

```
mitosis project_pkg.exp1 --eval-param seed=2 --param exp_params=var_a
```
It is also common to have one experiment wrap another, e.g. if exp2 is a gridsearch
around exp1.

I typically run experiments on servers, so `nohup ... &> exp.log &` frees up the
terminal and lets me disconnect ssh, returning later and reading exp.log to see
that the experiment worked or what errors occurred
(if error occurs inside the experiment and not inside `mitosis`, they will also
be visible in the generated html notebook).
If I have a variety of experiments that I want to run, I can copy and paste a
lot of experiments all at once into the terminal, and they will all execute in
parallel.

# Reproduceability Thoughts

The goal of the package, experimental reproduceability, poses a few fun challenges.
Here are my thoughts on reproduceable desiderata.

## Raison d'être
I designed `mitosis` for the primary purpose of stopping my confusion when I tried to
reproduce plots for my advisor after a small code change.  Without an automatic
record of the parameters in each run, I could not be sure whether the difference was due
to the code change (committed or not), a mistake in setting parameters, or the effect
of a new random seed.  `mitosis` prevents this confusion and many other faux-pas.


There's also a broader reason for more rigor around reproduceability.
While papers are published about parameterless methods or methods
where the user only needs to specify a single parameter, that data that proves the
method's efficacy comes from a heavily parametrized distribution (e.g. number of
timesteps, noise level, type of noise, initial conditions, etc).  Building the method
requires even more (e.g. network width, iteration and convergence controls).  Setting up
the experiment requires more (e.g. number of trials, n_folds).  While most of these are
reported in a paper, I have found it critical and difficult to keep track of these
details when developing a method and convincing with collaborators.

## Desiderata
Not all we could wish for is possible.  `mitosis` aspires to items four to nine
in the list below, making compromises along the way:

1. No neutrinos or gamma rays messing with bits
2. Same implementation of floating point arithmetic
3. Using the same versions of binary libraries
4. Using the same versions of python packages and python executable
5. Same git commit of all experimental code
6. Only run experiments with hashable parameters
7. Ability to freeze/reproduce mutable arguments
8. Ability to recreate arguments from either their `__repr__` string or their serialization
9. Don't run experiments without specifying a hypothesis first
10. For experiments that require randomness, only use a single, reproduceable generator.
