from setuptools import setup, find_packages
import os

NAME = "ephyspy"

EXTRAS = {
    "dev": [
        "black",
        "isort",
        "pyright",
        "flake8",
        "autoflake",
        "pre-commit",
    ]
}

REQUIRES = [
    "pandas>=1.5.3",
    "numpy>=1.23.5",
    "matplotlib>=3.4.2",
    "scipy>=1.9.1",
    "scikit-learn>=1.1.1",  # might get rid of this in the future.
]

here = os.path.abspath(os.path.dirname(__file__))

about = {}
project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
with open(os.path.join(here, project_slug, "__version__.py")) as f:
    exec(f.read(), about)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=about["__version__"],
    packages=find_packages(
        include=["ephyspy", "ephyspy.*"],
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"],
    ),
    description="Package to extract summary statistics from electrophysiological data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jonas Beck",
    python_requires=">=3.8",
    install_requires=REQUIRES,
    extras_require=EXTRAS,
    tests_require=["pytest"],
    include_package_data=True,
    license="GPLv3",
)
