[![Documentation Status](https://readthedocs.org/projects/mitosis/badge/?version=latest)](https://mitosis.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mitosis.svg)](https://badge.fury.io/py/mitosis)
[![Downloads](https://pepy.tech/badge/mitosis)](https://pepy.tech/project/mitosis)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# `mitosis`
_Reproduce Machine Learning experiments easily_

A package designed to manage the numerical experiments and track when results change due to different parameters versus when they change due to different code, input data, or random seed.  It works by verifying the git repository of the experiment is clean, recording the commit hash in a local sqlite database.  The database also tracks all the different parameter sets you send to the experiment over time.  It creates a Jupyter notebook of the experiment, runs in the notebook, and saves the notebook as an HTML to allow the user to later extract figures.  It records the performance metrics and filename in the database as well.

I designed `mitosis` for the primary purpose of stopping my confusion when I tried to reproduce experiments for my advisor after a small code change.  Without an automatic record of the parameters in each run, I could not be sure whether the difference was due to the code change (committed or not) a mistake in setting parameters, or the effect of random data.  Visualizations from notebooks were unreliable if the cells were out of order.  `mitosis` prevents these faux-pas.

## Use

A typical project will create experiments as a CLI, which calls `mitosis.run` as a wrapper around the project's experiments.  For instance, you can write a CLI to read parameters and run the named experiment through `mitosis`.  See [an example](https://github.com/Jacob-Stevens-Haas/gen-experiments), especially the `__main__.py`.

## `Experiment` api
An experiment is any module or class with a function obeying the signature:

```python
def run(seed: int, **kwargs) --> dict:
    ...
```

- __`kwargs`__: These are the axes by which an experiment may vary, e.g. `sim_params`, `data_params`, `solver_params`... etc.  If changing the number or name of kwargs between runs, `mitosis` will refuse:  every subsequent variant of a trial needs to specify the same parameters (even if they're empty).  If the need for new kwargs changes as the experiment evolves, either rename the experiment or rename the database that `mitosis` saves for that experiment.
- __metrics__: Typically, the return `dict` are the experiment metrics.  The metrics must have a key `"main"` in order for `mitosis` to know which metric to store in the database.
