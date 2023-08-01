from setuptools import setup, find_packages
import os

NAME = "py_ephys"

here = os.path.abspath(os.path.dirname(__file__))

about = {}
project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
with open(os.path.join(here, project_slug, "__version__.py")) as f:
    exec(f.read(), about)

setup(
    name = NAME,
    version = "0.0.1",
    version=about["__version__"],
    packages = find_packages(include=["py_ephys", "py_ephys.*"], exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    description = "Package to extract summary statistics from electrophysiological data.",
    author = "jnsbck",
    python_requires = ">=3.8",
    install_requires = [
    "pandas>=1.5.3",
    "numpy>=1.23.5",
    "matplotlib>=3.4.2",
    "scipy>=1.9.1",
    "scikit-learn>=1.1.1", # might get rid of this in the future.
    ],
    extras_require = {
        "dev": [
        "black", 
        "isort", 
        "pyright", 
        "flake8", 
        "autoflake",
        "pre-commit",
        ]
    },
    tests_require = ["pytest"],
    include_package_data=True,
)