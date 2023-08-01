## Issues

Bug reports and requests for features should be made via the [issues](https://github.com/mackelab/py_ephys/issues).

## Code contributions

If you want to contribute to this package yourself, please fork, create a feature branch and then make a PR from your feature branch to the upstream `py_ephys` ([see here for details](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)).

### Development environment

You can install via `setup.py` using `pip install -e ".[dev]"` (the dev flag installs development and testing dependencies). However at the moment the development environment is incomplete. 

### Style conventions

For docstrings and comments, please adhere to [Google Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

Code needs to pass through the following tools, which are installed alongside `py_ephys`:

**[black](https://github.com/psf/black)**: Automatic code formatting for Python. You can
run black manually from the console using `black .` in the top directory of the
repository, which will format all files.

**[isort](https://github.com/timothycrosley/isort)**: Used to consistently order
imports. You can run isort manually from the console using `isort` in the top
directory.

**[pyright](https://github.com/Microsoft/pyright)**: Used for static type checking. (This is not currently enforced, but planned for a future release).

`black` and `isort` and `pyright` are checked as part of our CI actions. If these
checks fail please make sure you have installed the latest versions for each of them
and run them locally.