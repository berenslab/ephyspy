from setuptools import setup, find_packages

setup(
    name='py_ephys',
    version='0.0.1',
    packages=find_packages(include=['py_ephys', 'py_ephys.*']),
    description='Package to extract summary statistics from electrophysiological data.',
    author='jnsbck',
    python_requires='>=3.8',
    install_requires=[
    'pandas>=1.5.3',
    'numpy>=1.23.5',
    'matplotlib>=3.4.2',
    'scipy>=1.9.1',
    'sklearn>=0.0', # might get rid of this in the future.
    ],
    setup_requires=['black', 'isort', 'pyright', 'flake8', 'autoflake'],
    tests_require=['pytest'],
)