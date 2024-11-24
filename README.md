[![Documentation Status](https://readthedocs.org/projects/adopt-net0/badge/?version=latest)](https://adopt-net0.readthedocs.io/en/latest/?badge=latest)
![Testing](https://github.com/UU-ER/AdOpT-NET0/actions/workflows/00publish_tests.yml/badge.svg?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/adopt-net0.svg)](https://pypi.org/project/adopt-net0/)
[![status](https://joss.theoj.org/papers/12578885161d419241e50c5e745b7a11/status.svg)](https://joss.theoj.org/papers/12578885161d419241e50c5e745b7a11)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13384688.svg)](https://doi.org/10.5281/zenodo.13384688)

# AdOpT-NET0 - Advanced Optimization Tool for Networks and Energy

This is a python package to simulate and optimize multi energy systems. It can 
model conversion technologies and networks for any carrier and optimize the 
design and operation of a multi energy system.

## Installation
You can use the standard utility for installing Python packages by executing the
following in a shell:

```pip install adopt_net0```

Additionally, you need a [solver installed, that is supported by pyomo](https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers)
(we recommend gurobi, which has a free academic licence).

Note for mac users: The export of the optimization results require a working
[hdf5 library](https://www.hdfgroup.org/solutions/hdf5/). On windows this should be
installed by default. On mac, you can install it with homebrew:

```brew install hdf5```

## Usage and documentation
The documentation and minimal examples of how to use the package can be found 
[here](https://adopt-net0.readthedocs.io/en/latest/index.html). We also provide a 
[visualization tool](https://resultvisualization.streamlit.app/) that is compatible 
with AdOpT-NET0.

## Dependencies
The package relies heavily on other python packages. Among others this package uses:

- [pyomo](https://github.com/Pyomo/pyomo) for compiling and constructing the model
- [pvlib](https://github.com/pvlib/pvlib-python) for converting climate data into 
  electricity output
- [tsam](https://github.com/FZJ-IEK3-VSA/tsam) for the aggregation of time series

## Credits
This tool was developed at Utrecht University.

This is some new text