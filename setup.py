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
        "pytest",
        "pytest-cov",
    ]
}

with open("requirements.txt", "r") as f:
    REQUIRES = f.read().splitlines()

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
