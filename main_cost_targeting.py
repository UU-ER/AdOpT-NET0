from adopt_net0.data_management.utilities import create_technology_class
from adopt_net0.data_preprocessing.template_creation import initialize_configuration_templates
from adopt_net0.data_preprocessing.data_loading import import_jrc_climate_data
from hp_cost_targeting.utilities import get_electricity_prices

import pyomo.environ as pyo
from pyomo.environ import SolverFactory
from pyomo.environ import Suffix
import pandas as pd
import os
from pathlib import Path
import h5py
import time

def main():
    data_path = Path(r".\hp_cost_targeting")
    bidding_zone = "CH"
    heat_price = 100

    el_prices = get_electricity_prices()
    el_prices = el_prices[0:8760]
    climate_data = pd.read_csv(data_path / "time_series" / "climate_data" / "climate_data.csv")

    t_start = time.time()

    periods = 1

    summary = pd.DataFrame()

    # for tech_name in ["HeatPump_AirSourced", "Boiler_El", "HeatPump_GroundSourced", "HeatPump_WaterSourced"]:
    for tech_name in ["HeatPump_Industrial"]:

        # Get climate data
        # climate_data = import_jrc_climate_data(4.9, 52, 2024, 10, year_idx=2023)
        # climate_data["dataframe"].to_csv(data_path / "time_series" / "climate_data" / "climate_data.csv")
        climate_data = climate_data[:periods]

        # Preprocessing
        technology = create_technology_class(tech_name, data_path)
        technology.performance_data["T_out"] = 150
        technology.fit_technology_performance(climate_data, None)

        data = {"config": {}, "topology": {}}
        data["config"] = initialize_configuration_templates()
        data["topology"]["fraction_of_year_modelled"] = 1
        data["hour_factors"] = [1] * 8760
        data["nr_timesteps_averaged"] = 1

        print("fitting technology took:", time.time() - t_start)
        t_1 = time.time()

        m = pyo.ConcreteModel()
        # m.dual = Suffix(direction=Suffix.IMPORT)  # for constraints

        m.set_t = pyo.RangeSet(1, periods, 1)
        m.set_t_clustered = pyo.RangeSet(1, periods, 1)

        # Build technology model
        def init_technology_block(b_tec):
            return technology.construct_tech_model(b_tec, data, m.set_t, m.set_t_clustered, cost_targeting=True)
        m.tech = pyo.Block(rule=init_technology_block)

        print("Constructing technology model took:", time.time() - t_1)
        t_1 = time.time()

        # Fix size
        m.const_size = pyo.Constraint(expr=m.tech.var_size == 1)

        # Formulate demand
        def init_demand(const, t):
            return m.tech.var_output[t, "heat"] == 0.5
        m.const_demand = pyo.Constraint(m.set_t, rule=init_demand)

        # Formulate profit == 0
        def init_profit(const):
            return 0 == sum(m.tech.var_output[t, "heat"] * heat_price for t in m.set_t) - sum(m.tech.var_input[t, "electricity"] * el_prices.iloc[t-1][bidding_zone] for t in m.set_t) - m.tech.var_capex
        m.const_profit = pyo.Constraint(rule=init_profit)

        print("Adding constraints took:", time.time() - t_1)
        t_1 = time.time()

        # Initialize Objective
        def init_cost_objective(obj):
            return m.tech.var_capex
        m.objective = pyo.Objective(rule=init_cost_objective, sense=pyo.maximize)

        print("Objective took:", time.time() - t_1)
        t_1 = time.time()

        m.pprint()

        solver = SolverFactory("gurobi")

        solution = solver.solve(
            m,
            tee=True,
            warmstart=True,
            keepfiles=True,
        )

        print("Solving took:", time.time() - t_1)
        t_1 = time.time()

        result_path = Path("hp_cost_targeting/results")
        h5_file_path = os.path.join(result_path, tech_name + "_optimization_results.h5")

        with h5py.File(h5_file_path, mode="w") as f:
            tec_operation = f.create_group("operation")
            technology.write_results_tec_operation(tec_operation, m.tech)
            tec_design = f.create_group("design")
            technology.write_results_tec_design(tec_design, m.tech)

        summary.loc[tech_name, "max_capex"] = m.tech.var_capex.value
        summary.loc[tech_name, "total_electricity_consumption"] = sum(m.tech.var_input[t, "electricity"].value for t in m.set_t)

        summary.to_excel("./hp_cost_targeting/results/summary.xlsx")


if __name__ == "__main__":
    main()

    # Adapt size respectively