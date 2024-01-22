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

Commit these changes to a repository.  After installing sine_experiment as a python package, in CLI, run:

    mitosis sine_experiment --param frequency=slow --eval-param amplitude=4

Mitosis will run `sin_experiment.run()`, saving
all output as an html file in the current directory.  It will also
track the parameters and results.
If you later change the variant named "slow" to set frequency=2, mitosis will
raise a `RuntimeError`, preventing you from running a trial.  If you want to run
`sine_experiment` with a different parameter value, you need to name that variant
something new.  Eval parameters, like "amplitude" in the example, behave differently.
Rather than being specified by `lookup_dict`, they are evaluated directly.

# How it Works

## Behind the scenes:

The first time `mitosis.run()` is passed a new experiment, it takes several actions:
1. Create a database for your experiment in `trials_folder`
2. Add a table to track all of the experiment trials
3. Add tables to track all of the different variants of your experiment.
4. Create a folder for the trial to store metadata.
5. Create and run a jupyter notebook in memory, saving the result as an HTML file
6. Updating the database of all trials with the results of the experiment.
7. Save experiment config of parameters actually created by jupyter notebook (in metadata folder)
8. Save a freezefile of python packages installed (in metadata folder)
9. Save the experiments results (in metadata folder)

In step 3, `mitosis` attempts to create a unique and reproduceable string from each
parameter value.  This is tricky, since most python objects are mutable and/or have
their `id()` stored in their string representation.  So lists must be sorted, dicts must
be ordered, and function and method arguments must be wrapped in a class with and a
custom `__str__()` attribute.  This is imperfectly done, see the **Reproduceability **
section for comments on the edge cases where `mitosis` will either treat the same
parameter as a new variant, or treat two different parameters as having the same value.

In step 5, `mitosis` needs to start the jupyter notebook with the appropriate variables.
Instead of sending the variables to the notebook, the notebook re-evaluates eval
parameters and re-looks up lookup parameters.  Previously, parameters were sent
to the notebook via pickle; that proved fragile.

The next time `mitosis.run()` is given the same experiment, it will
1. Determine whether parameter names and values match parameters in a previously established
variant.  If they do not, it will either:
   1. Reject the trial if the passed variant names match existing variants
   but with different values.
   2. Create a new variant for the parameter.
1. do steps 4 to 9 above.


## Abstractions

**Experiment:** the definition of a procedure that will test a hypothesis.
As a python object, an experiment must have a `Callable` attribute named "run"
that takes any number of arguments and returns a dict with at least a key named
"main".  It also requires a `name` and `lookup_dict` attribute.

In its current form, `mitosis` does not require a hypothesis, but it does
require experiments to define the "main" metric worth evaluating (though a
user can always define an experiment that sets the main metric to a constant).

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

See [an example](https://github.com/Jacob-Stevens-Haas/gen-experiments/blob/57877df35a9775db15719e16396fe8b06df5e3fa/run_exps.sh).

## Untracked parameters

If there are certain parameters that are not worth tracking, e.g. plotting flags
that do not change the mathematical results, prepend the argument name with "+".
An example:

```
mitosis project_pkg.exp1 -e +plot=True -p +plot_axes=dense
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

Mitosis also sets the log level of the experiment module to INFO and gives it
a FileHandler to the metadata directory.  In Debug mode, mitosis sets the log
level of the experiment to DEBUG.

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

# Some more advanced usage.

It should be noted that mitosis only works on installed packages - modules that you can run using `python -m pkgname`.

 When you want two modules share the same, long lookup_dict, I have found creating a
module with multiple dictionaries works well, e.g.

    project_pkg/
        |-- __init__.py     # Should look like
        |                   param_1 = {"var1": 1, "var2": 2}
        |                   param_2 = {"foo": "hello", "bar": "world"}
        |
        |                   # Each experiment gets same lookup dict
        |-- exp1.py         lookup_dict = vars(project_pkg)
        |-- exp2.py         lookup_dict = vars(project_pkg)

This way, the same variants can be used for different experiemnts:

```
mitosis project_pkg.exp1 -e seed=2 -p param_1=var_1 -p param_2=foo
mitosis project_pkg.exp2 -e seed=2 -p param_1=var_1 -p param_2=foo
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
