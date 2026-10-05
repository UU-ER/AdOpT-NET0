import random
from mes_north_sea.optimization.utilities import *
import os
import argparse

" CHANGE TEST, BASELINE EMISSIONS AND MAX NEGATIVE EMISSIONS FOR CORRECT SOLVE "

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=str, required=True,
                    choices=['RE_only', 'Onshore_DAC_only', 'Offshore_DAC_only'])
parser.add_argument('--fractions', type=str, required=True,
                    help='Comma-separated list of fractions, e.g. "0.4,0.6,0.8"')
args = parser.parse_args()

test = 0
settings = Settings(test=test)
settings.demand_factor = 1
settings.year = 2040
settings.variable_h2_demand = 0
cys = [2009]
co2_tax = [100]
c_permutation = 0.01

max_neg_emissions = 133608932 #full run: 133608932 test: choose
neg_fractions = [float(x) for x in args.fractions.split(',')]
baseline_emissions_pos = 76184504 # test: 180,945.17 full: 76184504

# avg cap factor
# total_production = pd.Series()
# for cy in [1995, 2008, 2009]:
#     time_series = pd.read_csv("./mes_north_sea/clean_data/production_profiles_re/production_profiles_re" + str(cy) + ".csv", index_col=0, header=[0, 1])
#     time_series.loc[:, (slice(None), 'total')].sum().sum()
#     total_production[str(cy)] = time_series.loc[:, (slice(None), 'total')].sum().sum()/1000000

project_root = Path(__file__).resolve().parent
data_path = str(project_root / ("mes_north_sea/data_" + str(settings.year)))

write_to_network_data(settings)

write_to_technology_data(settings)
if os.environ.get('HOSTNAME', '').startswith('sd26') or Path('/data/8051917').exists():
    save_path = "/data/8051917/results"
else:
    save_path = "results"

scenarios = {
    'RE_only': 'RE only',
    'Onshore_DAC_only': 'Onshore DAC only',
    'Offshore_DAC_only': 'Offshore DAC only',
             }

stage = args.stage
description = scenarios[stage]

if stage in ['RE_only', 'Onshore_DAC_only', 'Offshore_DAC_only']:
    settings.model_h2 = 0
else:
    settings.model_h2 = 1

for cy in cys:
    input_data_path = Path(data_path + "_" + str(cy) + "_" + stage)
    input_data_path.mkdir(parents=True, exist_ok=True)

    for tax in co2_tax:
        settings.co2_tax = tax
        settings.climate_year = cy
        settings.new_technologies_stage = stage

        adopt.create_optimization_templates(input_data_path)

        nodes = read_nodes(settings)
        define_topology(settings, input_data_path, nodes)
        define_configuration(input_data_path, settings)

        adopt.create_input_data_folder_template(input_data_path)

        define_node_locations(input_data_path, nodes)
        define_installed_capacities(input_data_path, settings, nodes)
        define_new_technologies(input_data_path, settings, nodes)
        define_storage(input_data_path, settings, nodes)
        adopt.copy_technology_data(input_data_path, Path(settings.data_path / "technology_data"))

        manual_offshore_override = [
            'BE_A', 'BE_B', 'DE_A', 'DE_B', 'DE_C', 'DE_D', 'DE_E', 'DE_F',
            'DK_A', 'DK_B', 'NL_A', 'NL_B', 'NL_C', 'NL_D', 'NL_E',
            'UK_B', 'UK_C', 'UK_D', 'UK_E', 'UK_F', 'UK_G', 'UK_H',
            'UK_L', 'UK_K', 'UK_J', 'UK_I',
            'NO_D', 'NO_C', 'NO_B', 'NO_A',
            'NL_G', 'NL_F', 'DK_C', 'DE_I', 'DE_H', 'DE_G',
            'DK_ST1', 'NO_ST2', 'NO_ST1'
        ]
        dac_size_max = 4000 if stage == 'RE_only' else 50000
        for node in list(set(nodes.offshore_nodes + nodes.onshore_nodes + manual_offshore_override)):
            for dac_name in ['DAC_Adsorption_offshore.json', 'DAC_Adsorption_onshore.json']:
                dac_path = input_data_path / "period1" / "node_data" / str(node) / "technology_data" / dac_name
                if dac_path.exists():
                    with open(dac_path, "r") as f:
                        dac_data = json.load(f)
                    if stage == 'RE_only' and dac_name == 'DAC_Adsorption_offshore.json':
                        dac_data["size_max"] = 4000
                    else:
                        dac_data["size_max"] = 50000
                    with open(dac_path, "w") as f:
                        json.dump(dac_data, f, indent=2)

        define_networks(input_data_path, settings)
        define_network_topology(input_data_path, settings, nodes)
        adopt.copy_network_data(input_data_path, Path(settings.data_path / "network_data"))

        define_demand(input_data_path, settings, nodes)
        define_generic_production(input_data_path, settings, nodes)
        define_hydro_inflow(input_data_path, settings)
        define_capacity_factors(input_data_path, settings, nodes)
        define_max_renewable_capacities(input_data_path, settings)
        define_imports_exports(input_data_path, settings, nodes)

        m = adopt.ModelHub()
        m.read_data(input_data_path)

        for node in m.data.technology_data["period1"]:
            for tec in m.data.technology_data["period1"][node]:
                m.data.technology_data["period1"][node][tec].economics['unit_capex'] = \
                    m.data.technology_data["period1"][node][tec].economics['unit_capex'] * random.uniform(
                        1 - c_permutation, 1 + c_permutation)

        if settings.test:
            final_save_path = save_path + "/2040_test/"
        else:
            final_save_path = save_path + "/2040/" + str(settings.climate_year)

        Path(final_save_path).mkdir(parents=True, exist_ok=True)

        m.data.model_config["reporting"]["save_summary_path"]["value"] = final_save_path
        m.data.model_config["reporting"]["save_path"]["value"] = final_save_path

        m = define_charging_efficiencies(settings, nodes, m)

        m.construct_model()
        m.construct_balances()
        m._define_solver_settings()

        for neg_fraction in neg_fractions:
            m.data.model_config["optimization"]["neg_emission_limit"] = {"value": max_neg_emissions * neg_fraction}
            m.data.model_config["optimization"]["pos_emission_limit"] = {"value": baseline_emissions_pos}
            m.data.model_config["reporting"]["case_name"]["value"] = \
                f"{stage}_neg_E_{int(neg_fraction*100)}pct_cy{settings.climate_year}_co2_tax{tax}"
            m._optimize_north_sea_dac()
