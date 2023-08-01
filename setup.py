from setuptools import setup, find_packages

about = {}

setup(
    name = 'py_ephys',
    version = '0.0.1',
    version=about["__version__"],
    packages = find_packages(include=['py_ephys', 'py_ephys.*'], exclude=['tests', '*.tests', '*.tests.*', 'tests.*']),
    description = 'Package to extract summary statistics from electrophysiological data.',
    author = 'jnsbck',
    python_requires = '>=3.8',
    install_requires = [
    'pandas>=1.5.3',
    'numpy>=1.23.5',
    'matplotlib>=3.4.2',
    'scipy>=1.9.1',
    'scikit-learn>=1.1.1', # might get rid of this in the future.
    ],
    extras_require = [
    'black', 
    'isort', 
    'pyright', 
    'flake8', 
    'autoflake',
    ],
    tests_require = ['pytest'],
)