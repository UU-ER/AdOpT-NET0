import pytest
import pandas as pd

from src.test.utilities import make_data_handle


def test_full_model_pipeline():
    """
    Tests the full modelling pipeline with a small case study

    Topology:
    - Nodes: node1, node2
    - Investment Periods: period1
    - Technologies:
        - node1: existing gas power plant
        - node2: new electric boiler
    - Networks:
        - new electricity
    - Timeframe: 1 timestep

    Data:
    - Demand:
        - node1: electricity=1
        - node2: heat=1
    - Import:
        - node1: gas
    - Import price:
        - node1: gas: 1

    The following is checked:
    - network size >=1
    - electric boiler size >= 1
    - output electric boiler = 1
    - total cost
    - total emissions

    """
    nr_timesteps = 1

    dh = make_data_handle(nr_timesteps)
