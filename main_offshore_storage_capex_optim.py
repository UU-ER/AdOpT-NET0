import os
import time

import adopt_net0 as adopt
from adopt_net0.case_offshore_storage.modelhub import ModelHubCapexOptimization
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pyomo.environ as pyo

"""
TODOS
- Add loop for technologies
- Add all technology datasheets
"""

#setup model (only required once)
input_data_path = Path("./offshore_storage/model_input")
# input_data_path = Path("./offshore_storage/model_input_test")
# adopt.create_optimization_templates(input_data_path)
# adopt.create_input_data_folder_template(input_data_path)
# adopt.copy_technology_data(input_data_path)
# adopt.copy_network_data(input_data_path)

test = 0
test_periods = 72
climate_year = 2000
# all_technologies = [
#     ('offshore', "Storage_OceanBattery_CapexOptimization")
# ]
run_baseline = 0

all_technologies = [
    # ('onshore', "Storage_Battery_CapexOptimization"),
    # ('onshore', "Storage_CAES_CapexOptimization"),
    # ('onshore', "Electrolyzer"),
    # ('offshore', "Storage_Battery_CapexOptimization"),
    # ('offshore', "Storage_CAES_CapexOptimization"),
    ('offshore', "Storage_OceanBattery_CapexOptimization"),
    ('offshore', "Electrolyzer"),
]
# Write generic production
def determine_time_series(f_demand, f_offshore, f_self_sufficiency, cy):
    cap_pv = 215002
    cap_wind_onshore = 115000
    s_pv = cap_pv / (cap_pv + cap_wind_onshore)
    s_wind = cap_wind_onshore / (cap_pv + cap_wind_onshore)

    cf_pv = pd.read_csv("./offshore_storage/data/capacity_factors/DE_Solar2030.csv",
                        sep=",")
    cf_wind_onshore = pd.read_csv(
        "./offshore_storage/data/capacity_factors/DE_Wind_onshore2030.csv",
        sep=",")
    cf_wind_offshore = pd.read_csv(
        "./offshore_storage/data/capacity_factors/DE_Wind_offshore2030.csv",
        sep=",")


    demand = pd.read_csv("./offshore_storage/data/demand/DE_Demand2030.csv", sep=";")
    demand_cy = demand[str(climate_year)] * f_demand
    cf_pv = cf_pv[str(climate_year)]
    cf_wind_onshore = cf_wind_onshore[str(climate_year)]
    cf_wind_offshore = cf_wind_offshore[str(climate_year)]

    annual_demand = sum(demand_cy)

    e_offshore = sum(cf_wind_offshore)
    e_onshore = sum(cf_wind_onshore) * s_wind + sum(cf_pv) * s_pv

    # capacity required for 1MWh annual generation onshore/offshore
    c_offshore = 1 / e_offshore * annual_demand * f_offshore * f_self_sufficiency
    c_onshore = 1 / e_onshore * annual_demand * (1 - f_offshore) * f_self_sufficiency

    # generation profiles
    p_offshore = c_offshore * cf_wind_offshore
    p_onshore = c_onshore * (cf_wind_onshore * s_wind + cf_pv * s_pv)

    return demand_cy, p_onshore, p_offshore

def set_data(climate_year, technology, f_demand, f_offshore,
                               f_self_sufficiency,
                               test):
    gas_price = 26.78  # ERAA 2023
    co2_price = 113  # ERAA 2023
    hydrogen_price = gas_price + co2_price * 0.18

    # DEMAND AND RE GENERATION
    demand, p_onshore, p_offshore = determine_time_series(f_demand, f_offshore,
                                                          f_self_sufficiency, climate_year)

    adopt.fill_carrier_data(input_data_path, p_onshore, ["Generic production"],
                            ["electricity"], ["onshore"], ["period1"])
    adopt.fill_carrier_data(input_data_path, p_offshore, ["Generic production"],
                            ["electricity"], ["offshore"], ["period1"])
    adopt.fill_carrier_data(input_data_path, demand, ["Demand"],
                            ["electricity"], ["onshore"], ["period1"])

    # POWER PLANT SIZE
    with open(
            input_data_path / "period1" / "node_data" / "onshore" / "Technologies.json",
            "r") as json_file:
        technologies = json.load(json_file)
    technologies["existing"] = {"GasTurbine_simple": max(demand) * 1.5,
                                "Storage_H2Cavern": 3000}
    technologies["new"] = []
    with open(
            input_data_path / "period1" / "node_data" / "onshore" / "Technologies.json",
            "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # other tecs
    if technology:
        with open(
                input_data_path / "period1" / "node_data" / technology[0] /
                "Technologies.json",
                "r") as json_file:
            technologies = json.load(json_file)
        technologies["new"] = [technology[1]]
        with open(
                input_data_path / "period1" / "node_data" / "onshore" / "Technologies.json",
                "w") as json_file:
            json.dump(technologies, json_file, indent=4)

    # NETWORK SIZE
    size = pd.read_csv(
        input_data_path / "period1" / "network_topology" / "existing" /
        "electricityOffshore" / "size.csv",
        sep=";", index_col=0)
    size.loc["onshore", "offshore"] = max(p_offshore) * 1.1
    size.loc["offshore", "onshore"] = max(p_offshore) * 1.1
    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" /
        "electricityOffshore" / "size.csv",
        sep=";")
    size.to_csv(
        input_data_path / "period1" / "network_topology" / "existing" /
        "hydrogenOffshore" / "size.csv",
        sep=";")

    # GAS IMPORTS
    adopt.fill_carrier_data(input_data_path, max(demand) * 2, ["Import limit"],
                            ["gas"], ["onshore"], ["period1"])
    adopt.fill_carrier_data(input_data_path, gas_price, ["Import price"],
                            ["gas"], ["onshore"], ["period1"])

    # Hydrogen Exports
    adopt.fill_carrier_data(input_data_path, 10000, ["Export Limit"],
                            ["hydrogen"], ["onshore"], ["period1"])
    adopt.fill_carrier_data(input_data_path, hydrogen_price, ["Export price"],
                            ["hydrogen"], ["onshore"], ["period1"])

    # CO2 COST
    carbon_cost_path = (input_data_path / "period1" / "node_data" / "onshore" /
                        "CarbonCost.csv")
    carbon_cost_template = pd.read_csv(carbon_cost_path, sep=';', index_col=0, header=0)
    carbon_cost_template['price'] = co2_price
    carbon_cost_template = carbon_cost_template.reset_index()
    carbon_cost_template.to_csv(carbon_cost_path, sep=';', index=False)


def adapt_model(m, p_onshore, p_offshore):
    model = m.model[m.info_solving_algorithms["aggregation_model"]].periods["period1"]
    # Adapt network size
    network_size = max(p_offshore)
    set_t_full = model.set_t_full
    b_netw = m.model[m.info_solving_algorithms[
        "aggregation_model"]].periods["period1"].network_block[
        "electricityOffshore_existing"]
    for arc in b_netw.set_arcs:
        b_arc = b_netw.arc_block[arc]
        b_arc.var_flow.setub(
            network_size)  # Set upper bound

        b_arc.del_component('const_flow_size_high')

        def init_size_const_high(const, t):
            return b_arc.var_flow[
                t] <= network_size

        b_arc.const_flow_size_high = pyo.Constraint(set_t_full,
                                                    rule=init_size_const_high)

    b_netw.del_component('const_cut_bidirectional')
    b_netw.del_component('const_cut_bidirectional_index')

    def init_cut_bidirectional(const, t, node_from, node_to):
        return b_netw.arc_block[node_from, node_to].var_flow[t] + \
            b_netw.arc_block[node_to, node_from].var_flow[t] \
            <= network_size

    b_netw.const_cut_bidirectional = pyo.Constraint(set_t_full,
                                                    b_netw.set_arcs_unique,
                                                    rule=init_cut_bidirectional)

    # Adapt production profiles (onshore)
    time_steps = len(m.data.time_series["full"][("period1", "onshore", "CarrierData",
                                                 "electricity", "Generic production")])

    m.data.time_series["full"][("period1", "onshore", "CarrierData",
                                "electricity", "Generic production")] = (
        p_onshore.to_list()[0:time_steps])
    m.data.time_series["full"][("period1", "offshore", "CarrierData",
                                "electricity", "Generic production")] = (
        p_offshore.to_list()[0:time_steps])

    def create_carrier_parameter(node, key, par_mutable=False):
        # Convert to dict/list for performance
        ts = {}
        for car in b_node.set_carriers:
            ts[car] = {}
            ts[car][key] = m.data.time_series["full"]["period1"][node][
                "CarrierData"][car][
                key].to_list()

        def init_carrier_parameter(para, t, car):
            """Rule initiating a carrier parameter"""
            return ts[car][key][t - 1]

        parameter = pyo.Param(
            set_t_full, b_node.set_carriers, rule=init_carrier_parameter,
            mutable=par_mutable
        )
        return parameter

    for nodename in ['onshore', 'offshore']:
        b_node = model.node_blocks[nodename]
        b_node.del_component('const_generic_production_index')
        b_node.del_component('const_generic_production')
        b_node.del_component('para_production_profile')
        b_node.del_component('para_production_profile_index')
        node_data = m.data.time_series["full"]

        b_node.para_production_profile = create_carrier_parameter(nodename,
                                                                  "Generic production")

        def init_generic_production(const, t, car):
            return b_node.para_production_profile[t, car] >= \
                b_node.var_generic_production[t, car]

        b_node.const_generic_production = pyo.Constraint(set_t_full,
                                                         b_node.set_carriers,
                                                         rule=init_generic_production)
    return m

total_cost_limit = {}

factors = {}
factors['demand'] = 0.2
if test == 1:
    factors['offshore'] = [0.25, 0.5]
    factors['self_sufficiency'] = [1, 2]
else:
    factors['offshore'] = [0.25, 0.5, 0.75, 1]
    factors['self_sufficiency'] = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

# BASELINE
if run_baseline:
    for f_offshore in factors['offshore']:
        total_cost_limit[f_offshore] = {}
        for f_self_sufficiency in factors['self_sufficiency']:
            total_cost_limit[f_offshore][f_self_sufficiency] = -1

    with open(input_data_path / "cost_limits.json", "w") as outfile:
        json.dump(total_cost_limit, outfile)

    for f_offshore in factors['offshore']:
        for f_self_sufficiency in factors['self_sufficiency']:
            case_name_bl = ("BL OS_" +
                            str(f_offshore) + " SS_" + str(
                        f_self_sufficiency))
            # SOLVE BASELINE
            if "m_baseline" not in locals():
                set_data(climate_year, None, factors['demand'], f_offshore,
                         f_self_sufficiency, test)

                # read data from files and construct baseline model
                m_baseline = adopt.ModelHub()
                if test:
                    m_baseline.read_data(input_data_path, start_period=0,
                                         end_period=test_periods)
                else:
                    m_baseline.read_data(input_data_path)

                m_baseline.construct_model()
                m_baseline.construct_balances()

            else:
                demand, p_onshore, p_offshore = determine_time_series(factors['demand'],
                                                                      f_offshore,
                                                                      f_self_sufficiency,
                                                                      climate_year)
                m_baseline = adapt_model(m_baseline, p_onshore, p_offshore)

            if test:
                m_baseline.data.model_config["reporting"]["case_name"]["value"] = "TEST " + case_name_bl
            else:
                m_baseline.data.model_config["reporting"]["case_name"]["value"] = case_name_bl
            m_baseline.solve()
            total_cost_limit[f_offshore][f_self_sufficiency] = m_baseline.model[m_baseline.info_solving_algorithms[
                "aggregation_model"]].var_npv.value
            with open(input_data_path / "cost_limits.json", "w") as outfile:
                json.dump(total_cost_limit, outfile)

for technology in all_technologies:
    # INPUT
    idx = 1
    for f_offshore in factors['offshore']:
        for f_self_sufficiency in factors['self_sufficiency']:

            while True:
                if os.path.exists(input_data_path / "cost_limits.json"):
                    with open(input_data_path / "cost_limits.json", "r") as outfile:
                        total_cost_limit = json.load(outfile)
                    while total_cost_limit[str(f_offshore)][str(f_self_sufficiency)] == -1:
                        print("BL not finished yet")
                        time.sleep(10)
                        with open(input_data_path / "cost_limits.json", "r") as outfile:
                            total_cost_limit = json.load(outfile)
                    break  # Exit the loop after successfully reading the file
                else:
                    print("File not found. Waiting 10 seconds before retrying...")
                    time.sleep(10)

            case_name_tec = (technology[0] + "_" + technology[1] + " OS_" +
                         str(f_offshore) + " SS_" + str(
                        f_self_sufficiency))

            if idx == 1:
                set_data(climate_year, None, factors['demand'], f_offshore,
                         f_self_sufficiency, test)
                m_storage = ModelHubCapexOptimization(technology, total_cost_limit[str(f_offshore)][str(f_self_sufficiency)])

                if test:
                    m_storage.read_data(input_data_path, start_period=0,
                                        end_period=test_periods)
                else:
                    m_storage.read_data(input_data_path)

                m_storage.construct_model()
                m_storage.construct_balances()

            else:
                demand, p_onshore, p_offshore = determine_time_series(factors['demand'],
                                                                      f_offshore,
                                                                      f_self_sufficiency,
                                                                      climate_year)
                m_storage = adapt_model(m_storage, p_onshore, p_offshore)
                m_storage.total_cost_limit = total_cost_limit[str(f_offshore)][str(f_self_sufficiency)]

            if test:
                m_storage.data.model_config["reporting"]["case_name"]["value"] = "TEST " + case_name_tec
            else:
                m_storage.data.model_config["reporting"]["case_name"]["value"] = case_name_tec

            m_storage.solve()

            idx = idx + 1




















