from adopt_net0.data_management.utilities import create_technology_class
from adopt_net0.data_preprocessing.template_creation import initialize_configuration_templates
from adopt_net0.data_preprocessing.data_loading import import_jrc_climate_data

import pyomo.environ as pyo
from pyomo.environ import SolverFactory
from pyomo.environ import Suffix

import pandas as pd
import os
from pathlib import Path
import h5py


def main():
    data_path = Path(r"C:\Users\jwiegner\PycharmProjects\AdOpT-NET0\mock_data")
    periods = 1
    tech_name = "HeatPump_AirSourced"

    # Get climate data
    # climate_data = import_jrc_climate_data(4.9, 52, 2015, 10)
    # climate_data["dataframe"].to_csv(data_path / "climate_data.csv")
    climate_data = pd.read_csv(data_path / "climate_data.csv")
    climate_data = climate_data[:periods]

    # Preprocessing
    technology = create_technology_class(tech_name, data_path)
    technology.fit_technology_performance(climate_data, None)

    data = {"config": {}, "topology": {}}
    data["config"] = initialize_configuration_templates()
    data["topology"]["fraction_of_year_modelled"] = 1
    data["hour_factors"] = [1] * 8760
    data["nr_timesteps_averaged"] = 1

    m = pyo.ConcreteModel()
    # m.dual = Suffix(direction=Suffix.IMPORT)  # for constraints

    m.set_t = pyo.RangeSet(1, periods, 1)
    m.set_t_clustered = pyo.RangeSet(1, periods, 1)

    # Build technology model
    def init_technology_block(b_tec):
        return technology.construct_tech_model(b_tec, data, m.set_t, m.set_t_clustered, cost_targeting=True)
    m.tech = pyo.Block(rule=init_technology_block)

    # Fix size
    m.const_size = pyo.Constraint(expr=m.tech.var_size == 1)

    # Formulate demand
    def init_demand(const, t):
        return m.tech.var_output[t, "heat"] == 0.9
    m.const_demand = pyo.Constraint(m.set_t, rule=init_demand)

    # Formulate profit == 0
    def init_demand(const, t):
        return 0 == sum(m.tech.var_output[t, "heat"] * 2 for t in m.set_t) - sum(m.tech.var_input[t, "electricity"] * 0 for t in m.set_t) - m.tech.var_capex + 0
    m.const_profit = pyo.Constraint(m.set_t, rule=init_demand)

    # Initialize Objective
    def init_cost_objective(obj):
        return m.tech.var_capex
    m.objective = pyo.Objective(rule=init_cost_objective, sense=pyo.maximize)


    solver = SolverFactory("gurobi")

    solution = solver.solve(
        m,
        tee=True,
        warmstart=True,
        keepfiles=True,
    )

    result_path = Path("./mock_data/results")
    h5_file_path = os.path.join(result_path, tech_name + "_optimization_results.h5")

    with h5py.File(h5_file_path, mode="w") as f:
        tec_operation = f.create_group("operation")
        technology.write_results_tec_operation(tec_operation, m.tech)
        tec_design = f.create_group("design")
        technology.write_results_tec_design(tec_design, m.tech)




    # print(m.tech.var_capex.value)

    # for c in m.component_objects(pyo.Constraint, active=True):
    #     for index in c:
    #         if c[index].active:
    #             print("Constraint", c.name, index, "dual =", m.dual[c[index]])



    # m.pprint()



    # m = TechnologyModel(Path("mock_data/mock_tec.json"))
    # m.read_time_series(Path("mock_data/heat_demand.csv"), ("output", "heat"), "demand")
    # m.read_time_series(Path("mock_data/heat_price.csv"), ("output", "heat"), "price")
    # m.read_time_series(Path("mock_data/electricity_price.csv"), ("input", "electricity"), "price")
    #
    # m.build_model()

if __name__ == "__main__":
    main()

    # Visualize
    # Do this for all HPs and electric boiler