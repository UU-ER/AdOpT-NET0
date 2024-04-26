import pytest
from pathlib import Path
from pyomo.environ import ConcreteModel, Set

from src.test.utilities import get_technology_data, make_climate_data


@pytest.mark.technologies
def test_res_pv():
    """
    tests res technology
    :return:
    :rtype:
    """
    # Technology Class Creation
    load_path = Path("./src/test/technology_data")
    technology = "TestTec_ResPhotovoltaic"
    tec = get_technology_data(technology, load_path)

    # Technology fitting
    climate_data = make_climate_data("2022-01-01 00:00", "2022-01-01 00:00")
    location = {}
    location["lon"] = 10
    location["lat"] = 52
    location["alt"] = 0
    tec.fit_technology_performance(climate_data, location)

    #
    block = ConcreteModel()
    set_t = Set(initialize=[1])
    set_t_full = Set(initialize=[1])

    tec.construct_tech_model(block, data, set_t, set_t_full)
