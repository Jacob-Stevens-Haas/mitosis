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

Hypothesis: the maximum value of a sine wave is equal to its amplitude.

*sine_experiment/\_\_init\_\_.py*


    import numpy as np

    name = "sine-exp"
    lookup_dict = {"frequency": {"fast": 10, "slow": 1}}

    def run(amplitude, frequency):
        """Deterimne if the maximum value of the sine function equals ``amplitude``"""
        x = np.arange(0, 10, .05)
        y = amplitude * np.sin(frequency * x)
        err = np.abs(max(y) - amplitude)
        return {"main": err}

Commit these changes to a repository.  After installing sine_experiment as a python package, in CLI, run:

    mitosis -m sine_experiment --param frequency=slow --eval-param amplitude=4

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
parameter value.
This is essential, since it strikes at the heart of what reproducibility means when talking about experiments.
It's also hard, since most python objects are mutable and/or have
their `id()` stored in their string representation.
So lists must be sorted, dicts must
be ordered, and function and method arguments must have their `__str__()` attribute replaced.
This is imperfectly done, see the **Reproduceability **
section for comments on the edge cases where `mitosis` will either treat the same
parameter as a new variant, or treat two different parameters as having the same value.

In step 5, `mitosis` needs to start the jupyter notebook with the appropriate variables.
Instead of sending the variables to the notebook, the notebook re-evaluates eval
parameters and re-looks up lookup parameters.  Previously, parameters were sent
to the notebook via pickle; that proved fragile.

The next time `mitosis.run()` is given the same experiment, it will
1. Determine whether parameter names and values match parameters in a previously established
variant.  It will either:
   * Reject the trial if the passed variant names match existing variants
   but with different values.
   * Create a new variant for the parameter.
2. Proceeds through steps 4 to 9 above.


## Abstractions

**Experiment:** the definition of a procedure that will test a hypothesis.
As a python object, an experiment is a series of steps, each of which is
a tuple of a `name`, a lookup dictionary, and a `Callable`
that takes any number of arguments and returns a dict with at least a key named
"main".  All but the last also need a key "data" to pass to the next step.  All
but the first step need to accept an argument named data.

In its current form, `mitosis` does not require a hypothesis, but consider
the "main" metric to stand in for a more formal hypothesis (though a
user can always define an experiment that sets the main metric to a constant).

When running in module mode (`-m` on command line), the lookup dictionary, name, and
callable are all loaded from the module's `lookup_dict`, `__qualname__`, and `run`
variables.  Otherwise, they're configured in pyproject.toml.

**Parameter**: An argument to an experiment.  These are the axes by which an experiment
step may vary, e.g. `sim_params`, `data_params`, `solver_params`... etc.  When this argument
is a `Collection`, sometimes the singular (parameter) and plural (parameters) are used
interchangeably.  Parameters can either be lookup parameters (which use the
step's' `lookup_dict`) or eval parameters (which are typically
simple evaluations, e.g. setting the random seed to a given integer).  Eval parameters
which are strings need quotes.

When running in module mode (`-m` on command line), experiments only have a single step,
so mitosis associates all arguments with that step.  In normal mode, the arguments for a
step must be prefixed with that step name, e.g. `step1.noise_level`

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
the pseudorandom key that is attached to filename serves as an effective primary key
over all variants and trials.

# API

Mitosis is primarily intended as a command line program, so `mitosis --help` has the syntax documentation.   There is only one intentionally publi part of the api: `mitosis.load_trial_data()`.

Here's a [pre-0.5.0 example](https://github.com/Jacob-Stevens-Haas/gen-experiments/blob/57877df35a9775db15719e16396fe8b06df5e3fa/run_exps.sh), when the `-m` flag was assumed.  For 0.5.0 usage, see the section on "More advanced usage"



## Untracked parameters

If there are certain parameters that are not worth tracking, e.g. plotting flags
that do not change the mathematical results, prepend the argument name with "+".
An example:

```
mitosis -m project_pkg.exp1 -e +plot=True -p +plot_axes=dense
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

# More advanced usage.

## Module-style experiments

It should be noted that mitosis only works on installed packages - modules that you can run using `python -m pkgname.modname`.

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
mitosis -m project_pkg.exp1 -e seed=2 -p param_1=var_1 -p param_2=foo
mitosis -m project_pkg.exp2 -e seed=2 -p param_1=var_1 -p param_2=foo
```
It is also common to have one experiment wrap another, e.g. if exp2 is a gridsearch
around exp1.

## Recommended: multi-step experiments

Sometimes it's useful to combine steps from different packages, or share steps across projects,
or distribute your experiment without the clutter of a lookup dictionary holding every variant
you tried out.
In these cases, the _project_ is responsible for naming and connecting the callable <-> lookup-dictionary association.
These experiment steps are specified in pyproject.toml, in the `[tools.mitosis.steps]` table using
object reference notation (à la plugins).

Let's say your project folder is called `my_paper` and contains an eponymous python package.  You have an experiment defined in the package `first_exp`, broken down into functions `make_data()` and `linear_pipeline`, but you want to be able to swap out the first step for real data in another module `geospatial.datasets`, you could have the following table in your project's pyproject.toml (likely within the `my_paper` repo):

    [tool.mitosis.steps]
    real_data = ["geospatial.datasets:load_data", "my_paper:data_config"]
    sim_data = ["first_exp:make_data", "my_paper:data_config"]
    fit_eval = ["first_exp:linear_pipeline", "my_paper:meth_config"]

This also says that the lookup dicts for each step are all imported from `my_paper`.  You would invoke experiments like:

```
mitosis sim_data fit_eval -p sim_data.noise=low -e fit_eval.alpha=.5
mitosis real_data fit_eval -p real_data.file="oct-2024.hd5" -e fit_eval.alpha=.5
```

Needless to say, all of the callables need to pass data compatibly, e.g.
`first_exp.linear_pipeline(first_exp.make_data(...)["data"], ...)` must make sense, as must
`first_exp.linear_pipeline(geospatial.datasets.load_data(...)["data"], ...)`.  Some caution here is advised - mitosis does not yet check all editably-installed packages for being git-clean.

You could then have code in `my_paper` that loads the data from these trials and builds comparison plots, or you could rely on the plots each experiment creates.  You could also have a shell/batch script that spawns the main experiments of your paper.

I'm often on a server and want to disconnect while the experiment is running, so I wrap my experiments in `nohup ... &> exp1.log &`.


## pyproject.toml config
The `[tool.mitosis]` table can be used to set `trials-folder`.  If relative,
it is relative to repository root.


## Using persistent data

There are two obviously useful things to do after an experiment:
* view the html file.  `python -m http.server` is helpful to browse results
* load the data with `load_trial_data()`

Beyond this, the metadata mitosis keeps to disk is useful for troubleshooting or reproducing experiments, but no facility yet exists to browse or compare experiments.


<!-- # Reproduceability Thoughts

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
10. For experiments that require randomness, only use a single, reproduceable generator. -->
