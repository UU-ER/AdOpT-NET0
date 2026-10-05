from pathlib import Path
import pandas as pd
from types import SimpleNamespace
import json
import os

import adopt_net0 as adopt
from dac_offshore.global_vars import CO2_PRICE, SCENARIOS, CLIMATE_YEARS, GAS_PRICE


def _read_nodes(node_data_path):
    nodes = SimpleNamespace()

    node_list = pd.read_excel(node_data_path, sheet_name=str(2040))
    nodes.offshore_existing = node_list[node_list['Type'] == 'offshore_existing']['Node'].values.tolist()
    nodes.offshore_new = node_list[node_list['Type'] == 'offshore_new']['Node'].values.tolist()
    nodes.co2storage = node_list[node_list['Type'] == 'co2storage']['Node'].values.tolist()
    nodes.onshore = node_list[node_list['Type'] == 'onshore']['Node'].values.tolist()
    nodes.all = {}
    for row in node_list.iterrows():
        node_data = {}
        node_data['lon'] = row[1]['x']
        node_data['lat'] = row[1]['y']
        nodes.all[row[1]['Node']] = node_data

    return nodes

class InputDataCreator:
    # clean data path
    clean_data_path = Path(__file__).resolve().parent / 'clean_data'
    model_year = 2040
    nodes = _read_nodes(Path(clean_data_path / "nodes" / "nodes.xlsx"))

    def create_input_data(self):

        self._write_to_technology_data()
        self._write_to_network_data()
        self._define_co2_pipeline_costs()

        for cy in CLIMATE_YEARS:
            for scenario in SCENARIOS:

                # create folder if it doesn't exist yet
                input_data_path = Path(__file__).resolve().parent / 'input_data' / (str(self.model_year) + "_" + str(cy) + "_" + str(scenario))
                input_data_path.mkdir(parents=True, exist_ok=True)

                # create data folder
                self._create_optimization_templates(input_data_path)
                self._define_topology(input_data_path, scenario)
                self._define_configuration(input_data_path)
                self._create_input_data_folder_template(input_data_path)
                self._define_node_locations(input_data_path)

                # Technologies
                self._define_installed_capacities(input_data_path)
                self._define_new_technologies(input_data_path, scenario)
                adopt.copy_technology_data(input_data_path, Path(self.clean_data_path / "technology_data"))
                self._define_max_renewable_capacities(input_data_path)
                self._define_max_dac_capacities(input_data_path, scenario)

                # Networks
                self._define_networks(input_data_path, scenario)
                self._define_network_topology(input_data_path, scenario)
                adopt.copy_network_data(input_data_path, Path(self.clean_data_path / "network_data"))
                self._copy_network_arc_cost_data(input_data_path, scenario)


                # Time series
                self._define_demand(input_data_path, cy)
                self._define_generic_production(input_data_path, cy)
                self._define_hydro_inflow(input_data_path, cy)
                self._define_capacity_factors(input_data_path, cy)
                self._define_weather_data(input_data_path, cy)

                # Imports exports
                self._define_imports_exports(input_data_path)

    @classmethod
    def define_charging_efficiencies(cls, m):
        new_tecs = pd.read_csv(cls.clean_data_path / "technologies_existing" / "capacities_node.csv",
                               index_col=0)

        for node in cls.nodes.onshore:
            new_at_node = \
                new_tecs[new_tecs['Node'] == node][['Technology', 'Capacity our work']].set_index(
                    'Technology').to_dict()[
                    'Capacity our work']

            charging = {
                'Storage_PumpedHydro_Closed': round(new_at_node.get('Hydro - Pump Storage Closed Loop (Pumping)', 0),
                                                    0),
                'Storage_PumpedHydro_Open': round(new_at_node.get('Hydro - Pump Storage Open Loop (Pumping)', 0), 0),
                'Storage_PumpedHydro_Reservoir': 0,
            }

            discharging = {
                'Storage_PumpedHydro_Closed': round(new_at_node.get('Hydro - Pump Storage Closed Loop (Turbine)', 0),
                                                    0),
                'Storage_PumpedHydro_Open': round(new_at_node.get('Hydro - Pump Storage Open Loop (Turbine)', 0), 0),
                'Storage_PumpedHydro_Reservoir': round(new_at_node.get('Hydro - Reservoir (Turbine)', 0), 0),
            }

            capacity = {
                'Storage_PumpedHydro_Closed': round(new_at_node.get('Hydro - Pump Storage Closed Loop (Energy)', 0), 0),
                'Storage_PumpedHydro_Open': round(new_at_node.get('Hydro - Pump Storage Open Loop (Energy)', 0), 0),
                'Storage_PumpedHydro_Reservoir': round(new_at_node.get('Hydro - Reservoir (Energy)', 0), 0),
            }

            storage_tecs_at_node = {k: v for k, v in capacity.items() if v > 0}

            for storage in storage_tecs_at_node:
                if storage == 'Storage_PumpedHydro_Closed':
                    m.data.technology_data["period1"][node][storage + '_existing'].processed_coeff.time_independent[
                        'charge_rate'] = \
                        -charging[storage] / capacity[storage]

                    m.data.technology_data["period1"][node][storage + '_existing'].processed_coeff.time_independent[
                        'discharge_rate'] = \
                        discharging[storage] / capacity[storage]
                else:
                    m.data.technology_data["period1"][node][storage + '_existing'].processed_coeff.time_independent[
                        'charge_max'] = \
                        -charging[storage] / capacity[storage]

                    m.data.technology_data["period1"][node][storage + '_existing'].processed_coeff.time_independent[
                        'discharge_max'] = \
                        discharging[storage] / capacity[storage]

    def _create_optimization_templates(self, input_data_path):
        adopt.create_optimization_templates(input_data_path)

    def _create_input_data_folder_template(self, input_data_path):
        adopt.create_input_data_folder_template(input_data_path)

    def _define_topology(self, input_data_path, scenario):
        with open(input_data_path / "Topology.json", "r") as json_file:
            topology = json.load(json_file)

        # Nodes
        topology["nodes"] = list(self.nodes.all.keys())

        # Carriers:
        if scenario == "RE_only":
            topology["carriers"] = ['electricity', 'gas']
        else:
            topology["carriers"] = ['electricity', 'gas', 'CO2captured', 'heat']

        if self.model_year == 2040:
            model_year = 2041
        else:
            model_year = self.model_year

        topology["start_date"] = str(model_year) + "-" + '01-01 00:00'
        topology["end_date"] = str(model_year) + "-" + '12-31 23:00'

        # Investment periods:
        topology["investment_periods"] = ["period1"]
        # Save json template
        with open(input_data_path / "Topology.json", "w") as json_file:
            json.dump(topology, json_file, indent=4)

    def _define_configuration(self, input_data_path):
        # Configuration
        with open(input_data_path / "ConfigModel.json", "r") as json_file:
            configuration = json.load(json_file)

        configuration["optimization"]["typicaldays"]["N"]["value"] = 0
        configuration["optimization"]["typicaldays"]["method"]["value"] = 2

        configuration["solveroptions"]["solver"]["value"] = 'gurobi'
        configuration["solveroptions"]["mipgap"]["value"] = 0.02
        configuration["solveroptions"]["lpwarmstart"]["value"] = 0
        configuration["solveroptions"]["numericfocus"]["value"] = 3
        configuration["solveroptions"]["timelim"]["value"] = 7 * 24
        configuration["solveroptions"]["method"]["value"] = 2
        configuration["solveroptions"]["threads"]["value"] = 30
        configuration["solveroptions"]["crossover"] = {}
        configuration["solveroptions"]["crossover"]["value"] = 0
        configuration["solveroptions"]["nodemethod"] = {}
        configuration["solveroptions"]["nodemethod"]["value"] = -1
        configuration["solveroptions"]["intfeastol"]["value"] = 1e-3
        configuration["solveroptions"]["feastol"]["value"] = 1e-3

        configuration["scaling"]["scaling_on"]["value"] = 1
        configuration["scaling"]["scaling_factors"]["energy_vars"]["value"] = 1e-2
        configuration["scaling"]["scaling_factors"]["cost_vars"]["value"] = 1e-2
        configuration["scaling"]["scaling_factors"]["objective"]["value"] = 1

        with open(input_data_path / "ConfigModel.json", "w") as json_file:
            json.dump(configuration, json_file, indent=4)

    def _define_node_locations(self, input_data_path):
        node_location = pd.read_csv(input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)
        for _, node in node_location.iterrows():
            node.loc["lon"] = self.nodes.all[node.name]["lon"]
            node.loc["lat"] = self.nodes.all[node.name]["lat"]
            node.loc["alt"] = 10
        node_location = node_location.reset_index()
        node_location.to_csv(input_data_path / "NodeLocations.csv", sep=';', index=False)

    def _define_installed_capacities(self, input_data_path):

        new_tecs = pd.read_csv(self.clean_data_path / 'technologies_existing/capacities_node.csv',
                               index_col=0)
        for node in self.nodes.onshore:
            with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "r") as json_file:
                technologies = json.load(json_file)

            new_at_node = \
                new_tecs[new_tecs['Node'] == node][['Technology', 'Capacity our work']].set_index(
                    'Technology').to_dict()[
                    'Capacity our work']

            tecs_at_node = {
                'PowerPlant_Gas': round(new_at_node.get('Gas', 0), 0),
                'PowerPlant_Nuclear': round(new_at_node.get('Nuclear', 0), 0),
                'PowerPlant_Oil': round(new_at_node.get('Oil', 0), 0),
                'PowerPlant_Coal': round(new_at_node.get('Coal & Lignite', 0), 0),
                'Storage_PumpedHydro_Closed': round(
                    new_at_node.get('Hydro - Pump Storage Closed Loop (Energy)', 0),
                    0),
                'Storage_PumpedHydro_Open': round(new_at_node.get('Hydro - Pump Storage Open Loop (Energy)', 0),
                                                  0),
                'Storage_PumpedHydro_Reservoir': round(new_at_node.get('Hydro - Reservoir (Energy)', 0), 0),
            }

            technologies["existing"] = {k: v for k, v in tecs_at_node.items() if v > 0}

            with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "w") as json_file:
                json.dump(technologies, json_file, indent=4)

    def _define_new_technologies(self, input_data_path, scenario):
        new_tecs = pd.read_excel(self.clean_data_path / 'technologies_new/NewTechnologies.xlsx', index_col=0,
                                     sheet_name='NewTechnologies')
        new_tecs.fillna("", inplace=True)

        if scenario == 'No_DAC':
            columns = ['No_DAC']
        elif scenario ==  'DAC':
            columns = ['No_DAC', 'Onshore_DAC_only', 'Offshore_DAC_only']
        elif scenario ==  'Onshore_DAC_only':
            columns = ['No_DAC', 'Onshore_DAC_only']
        elif scenario ==  'Offshore_DAC_only':
            columns = ['No_DAC', 'Offshore_DAC_only']
        else:
            raise Exception('Scenario not known')


        for node in self.nodes.all.keys():
            with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "r") as json_file:
                technologies = json.load(json_file)

            technology_string =  new_tecs.loc[node, columns]
            technology_string = [item for item in technology_string if item]
            technology_string = [tech.strip() for item in technology_string for tech in item.split(',')]
            technologies["new"] = technology_string

            with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "w") as json_file:
                json.dump(technologies, json_file, indent=4)

    def _write_to_technology_data(self):
        year = 2030

        financial_data = pd.read_excel(self.clean_data_path / 'technologies_cost' / 'TechnologyCost.xlsx', sheet_name='ToModel',
                                       skiprows=1)
        financial_data = financial_data[financial_data['Year'] == year]

        tec_data_path = self.clean_data_path / 'technology_data'

        for filename in os.listdir(tec_data_path):
            with open(os.path.join(tec_data_path, filename), 'r') as openfile:
                # Reading from json file
                tec_data = json.load(openfile)

            tec = filename.replace('.json', '')
            tec = tec.replace('_noh2', '')

            new_financial_data = financial_data[financial_data['Technology'] == tec]
            tec_data['Economics']['unit_capex'] = float(round(new_financial_data['Investment Cost'].values[0], 2))
            tec_data['Economics']['opex_variable'] = float(round(new_financial_data['OPEX Variable'].values[0], 3))
            tec_data['Economics']['opex_fixed'] = float(round(new_financial_data['OPEX Fixed'].values[0], 3))
            tec_data['Economics']['lifetime'] = float(round(new_financial_data['Lifetime'].values[0], 0))
            tec_data['Performance']['emission_factor'] = float(
                round(new_financial_data['Emission factor'].values[0], 3))
            if 'performance' in tec_data['Performance']:
                performance_parameters = {'eta_in': 'Charging Efficiency', 'eta_out': 'Discharging Efficiency',
                                          'lambda': 'Lambda'}
                for para in performance_parameters.keys():
                    if para in tec_data['Performance']['performance']:
                        tec_data['Performance']['performance'][para] = float(
                            new_financial_data[performance_parameters[para]].values[0])
                if 'out' in tec_data['Performance']['performance']:
                    if tec_data['tec_type'] == 'CONV1':
                        tec_data['Performance']['performance']['out'] = [0, float(
                            round(new_financial_data['Conv. Efficiency'].values[0], 3))]
                    else:
                        for car in tec_data['Performance']['performance']['out']:
                            tec_data['Performance']['performance']['out'][car] = [0, float(
                                round(new_financial_data['Conv. Efficiency'].values[0], 3))]

            with open(os.path.join(tec_data_path, filename), 'w') as outfile:
                json.dump(tec_data, outfile, indent=2)

    def _write_to_network_data(self):
        year = self.model_year

        netw_data_path = self.clean_data_path / 'network_data'

        financial_data = pd.read_excel(self.clean_data_path / 'networks_cost' / 'NetworkCost.xlsx', sheet_name='ToModel', skiprows=1)
        financial_data = financial_data[financial_data['Year'] == year]

        for filename in os.listdir(netw_data_path):
            if filename.replace('.json', '') in financial_data['Network'].to_list():
                with open(os.path.join(netw_data_path, filename), 'r') as openfile:
                    # Reading from json file
                    netw_data = json.load(openfile)

                new_financial_data = financial_data[financial_data['Network'] == filename.replace('.json', '')]
                netw_data['Economics']['gamma1'] = float(round(new_financial_data['gamma1'].values[0], 2))
                netw_data['Economics']['gamma2'] = float(round(new_financial_data['gamma2'].values[0], 2))
                netw_data['Economics']['gamma3'] = float(round(new_financial_data['gamma3'].values[0], 2))
                netw_data['Economics']['gamma4'] = float(round(new_financial_data['gamma4'].values[0], 2))
                netw_data['Economics']['opex_variable'] = float(round(new_financial_data['OPEX Variable'].values[0], 3))
                netw_data['Economics']['opex_fixed'] = float(round(new_financial_data['OPEX Fixed'].values[0], 3))
                netw_data['Economics']['lifetime'] = float(round(new_financial_data['Lifetime'].values[0], 0))
                netw_data['Performance']['loss'] = float(round(new_financial_data['loss'].values[0], 8))
                netw_data['Performance']['rated_capacity'] = float(
                    round(new_financial_data['rated power'].values[0], 0))

                with open(os.path.join(netw_data_path, filename), 'w') as outfile:
                    json.dump(netw_data, outfile, indent=2)

    def _define_networks(self, input_data_path, scenario):
        """
        Defines the networks
        """

        # CO2 network
        if scenario != 'No_DAC':
            new_co2_networks = ["CO2_Pipeline"]
        else:
            new_co2_networks = []

        # El networks
        new_el_networks = ["electricityAC", "electricityDC"]

        with open(input_data_path / "period1" / "Networks.json", "r") as json_file:
            networks = json.load(json_file)
        networks["new"] = new_el_networks + new_co2_networks
        networks["existing"] = ["electricityAC", "electricityDC"]

        with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
            json.dump(networks, json_file, indent=4)

    def _define_network_topology(self, input_data_path, scenario):

        netw_data_path = self.clean_data_path / 'networks_topology'

        def get_network_data(file_path, nodes):
            network = pd.read_csv(file_path, sep=None, engine='python')

            network_data = {}
            network_data['size_matrix'] = pd.read_csv(
                input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                index_col=0).astype(float)
            network_data['distance_matrix'] = pd.read_csv(
                input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                index_col=0).astype(float)
            network_data['max_size_matrix'] = pd.read_csv(
                input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                index_col=0).astype(float)
            network_data['connection_matrix'] = pd.read_csv(
                input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                index_col=0).astype(float)
            for idx, row in network.iterrows():
                if (row.node0 in nodes.all.keys()) & (row.node1 in nodes.all.keys()):
                    network_data['size_matrix'].loc[row['node0'], row['node1']] = row['s_nom'] * 1000
                    network_data['size_matrix'].loc[row['node1'], row['node0']] = row['s_nom'] * 1000
                    network_data['distance_matrix'].loc[row['node0'], row['node1']] = row['length']
                    network_data['distance_matrix'].loc[row['node1'], row['node0']] = row['length']
                    network_data['max_size_matrix'].loc[row['node1'], row['node0']] = row['s_nom_max'] * 1000 - row[
                        's_nom'] * 1000
                    network_data['max_size_matrix'].loc[row['node0'], row['node1']] = row['s_nom_max'] * 1000 - row[
                        's_nom'] * 1000
                    if row['s_nom_max'] > 0:
                        network_data['connection_matrix'].loc[row['node1'], row['node0']] = 1
                        network_data['connection_matrix'].loc[row['node0'], row['node1']] = 1

            return network_data

        # Electricity grids existing
        for grid_type in ["AC", "DC"]:
            file_name = f'electricity{grid_type}.csv'
            data = get_network_data(netw_data_path / file_name, self.nodes)
            os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / f"electricity{grid_type}", exist_ok=True)
            data['connection_matrix'].to_csv(
                input_data_path / "period1" / "network_topology" / "existing" / f"electricity{grid_type}" / "connection.csv",
                sep=";")
            data['distance_matrix'].to_csv(
                input_data_path / "period1" / "network_topology" / "existing" / f"electricity{grid_type}" / "distance.csv",
                sep=";")
            data['size_matrix'].to_csv(
                input_data_path / "period1" / "network_topology" / "existing" / f"electricity{grid_type}" / "size.csv",
                sep=";")

        # Electricity grids new
        for grid_type in ["AC", "DC"]:
            file_name = f'electricity{grid_type}.csv'
            data = get_network_data(netw_data_path / file_name, self.nodes)
            os.makedirs(input_data_path / "period1" / "network_topology" / "new" / f"electricity{grid_type}", exist_ok=True)
            data['connection_matrix'].to_csv(
                input_data_path / "period1" / "network_topology" / "new" / f"electricity{grid_type}" / "connection.csv",
                sep=";")
            data['distance_matrix'].to_csv(
                input_data_path / "period1" / "network_topology" / "new" / f"electricity{grid_type}" / "distance.csv",
                sep=";")
            data['size_matrix'].to_csv(
                input_data_path / "period1" / "network_topology" / "new" / f"electricity{grid_type}" / "size_max_arcs.csv",
                sep=";")

        # CO2 networks onshore and offshore
        data = get_network_data(netw_data_path / 'CO2_Pipeline.csv', self.nodes)

        netw_name = "CO2_Pipeline"
        os.makedirs(input_data_path / "period1" / "network_topology" / "new" / netw_name, exist_ok=True)
        data['connection_matrix'].to_csv(
            input_data_path / "period1" / "network_topology" / "new" / netw_name / "connection.csv",
            sep=";"
        )
        data['distance_matrix'].to_csv(
            input_data_path / "period1" / "network_topology" / "new" / netw_name / "distance.csv",
            sep=";"
        )

    def _define_demand(self, input_data_path, climate_year):

        demand_el = pd.read_csv(
            self.clean_data_path / 'demand' / f"TotalDemand_NT_{str(self.model_year)}_{str(climate_year)}.csv", index_col=0)
        for node in self.nodes.onshore:
            demand = pd.DataFrame()
            demand["Demand"] = demand_el[node]
            adopt.fill_carrier_data(input_data_path, value_or_data=demand, columns=['Demand'], carriers=['electricity'],
                                    nodes=[node])

    def _define_generic_production(self, input_data_path, climate_year):

        generic_production = pd.read_csv(
            self.clean_data_path / "production_profiles_re" / f"production_profiles_re{str(climate_year)}.csv",
            index_col=0, header=[0, 1])

        nodes_with_generic_production = self.nodes.offshore_existing + self.nodes.onshore
        for node in nodes_with_generic_production:
            profile = pd.DataFrame()
            profile["Generic production"] = generic_production.loc[:, (node, 'total')].to_numpy().round(1)
            adopt.fill_carrier_data(input_data_path, value_or_data=profile, columns=['Generic production'],
                                    carriers=['electricity'], nodes=[node])

    def _define_hydro_inflow(self, input_data_path, climate_year):

        inflows = pd.read_csv(self.clean_data_path / "hydro_inflows" / f"hydro_inflows{str(climate_year)}.csv",
                              index_col=0, header=[0, 1])

        for col in inflows.columns:
            node = col[0]
            tec = col[1]
            climate_data = pd.read_csv(input_data_path / "period1" / "node_data" / node / "ClimateData.csv", sep=";",
                                       index_col=0)

            climate_data[tec] = inflows[col].tolist()[:len(climate_data)]

            if tec == 'Hydro - Reservoir (Energy)':
                climate_data["Storage_PumpedHydro_Reservoir_existing_inflow"] = inflows[col].tolist()[
                    :len(climate_data)]
            elif tec == 'Hydro - Pump Storage Open Loop (Energy)':
                climate_data["Storage_PumpedHydro_Open_existing_inflow"] = inflows[col].tolist()[:len(climate_data)]

            climate_data.to_csv(input_data_path / "period1" / "node_data" / node / "ClimateData.csv", sep=";")

    def _define_capacity_factors(self, input_data_path, climate_year):

        cfs = {}

        cfs["offshore_wind"] = pd.read_csv(self.clean_data_path /'capacity_factors' / f"wind_offshore{str(climate_year)}.csv", index_col=0)
        cfs["onshore_wind"] = pd.read_csv(self.clean_data_path /'capacity_factors' / f"wind_onshore{str(climate_year)}.csv", index_col=0)
        cfs["pv"] = pd.read_csv(self.clean_data_path /'capacity_factors' / f"pv{str(climate_year)}.csv", index_col=0)

        for profile in cfs.keys():
            for node in cfs[profile].columns:
                climate_data = pd.read_csv(input_data_path / "period1" / "node_data" / node / "ClimateData.csv",
                                           sep=";", index_col=0)
                climate_data[profile] = cfs[profile][node].to_numpy()[:len(climate_data)]
                climate_data.to_csv(input_data_path / "period1" / "node_data" / node / "ClimateData.csv", sep=";")

    def _define_weather_data(self, input_data_path, climate_year):

        weather_data = pd.read_csv(self.clean_data_path / "weather_data" / f"weather_{climate_year}.csv", index_col=0)

        for node in self.nodes.all.keys():
            climate_csv = input_data_path / "period1" / "node_data" / node / "ClimateData.csv"
            climate_data = pd.read_csv(climate_csv, sep=";", index_col=0)
            climate_data['temp_air'] = weather_data[f"{node}_temp"].to_numpy()[:len(climate_data)]
            climate_data['rh'] = weather_data[f"{node}_rh"].to_numpy()[:len(climate_data)]
            climate_data.to_csv(climate_csv, sep=";")

    def _define_max_renewable_capacities(self, input_data_path):

        max_caps = pd.read_csv(self.clean_data_path / "max_re_cap" / "max_re_nodes.csv", index_col=0)

        for node in self.nodes.offshore_new:
            tec_data_path = input_data_path / "period1" / "node_data" / node / "technology_data"

            with open(os.path.join(tec_data_path, "Offshore_Wind.json"), 'r') as openfile:
                tec_data = json.load(openfile)

            tec_data["size_max"] = float(min(max_caps.loc[node, 'RemainingPotential_Wind_off'], 250000 / 2))

            with open(os.path.join(tec_data_path, "Offshore_Wind.json"), 'w') as outfile:
                json.dump(tec_data, outfile, indent=2)

        for node in self.nodes.onshore:
            tec_data_path = input_data_path / "period1" / "node_data" / node / "technology_data"

            with open(os.path.join(tec_data_path, "Onshore_Wind.json"), 'r') as openfile:
                tec_data = json.load(openfile)

            tec_data["size_max"] = float(min(max_caps.loc[node, 'RemainingPotential_Wind_on'], 100000))

            with open(os.path.join(tec_data_path, "Onshore_Wind.json"), 'w') as outfile:
                json.dump(tec_data, outfile, indent=2)

            with open(os.path.join(tec_data_path, "PV.json"), 'r') as openfile:
                tec_data = json.load(openfile)

            tec_data["size_max"] = float(min(max_caps.loc[node, 'RemainingPotential_PV'], 400000))

            with open(os.path.join(tec_data_path, "PV.json"), 'w') as outfile:
                json.dump(tec_data, outfile, indent=2)

    def _define_max_dac_capacities(self, input_data_path, scenario):
        if scenario != 'No_DAC':
            limits = pd.read_csv(self.clean_data_path / "co2_storage_limits" / "CO2_storage_limits_2040.csv", sep=',', thousands=',')
            limits.columns = limits.columns.str.strip()
            limits = limits.set_index("Node")

            for node in self.nodes.all.keys():
                tec_data_path = (input_data_path / "period1" / "node_data" / node / "technology_data" / "PermanentStorage_CO2_simple.json")
                if tec_data_path.exists():
                    with open(tec_data_path, "r") as f:
                        tec_data = json.load(f)
                    if node in limits.index:
                        tec_data["size_max"] = float(limits.loc[node, "size_max"])
                        if "Flexibility" not in tec_data:
                            tec_data["Flexibility"] = {}
                        tec_data["Flexibility"]["injection_rate_max"] = float(
                            limits.loc[node, "injection_rate_max"])

                    with open(tec_data_path, "w") as f:
                        json.dump(tec_data, f, indent=2)

    def _define_imports_exports(self, input_data_path):

        data_path = self.clean_data_path / "import_export" / "ImportExport_noimport.xlsx"

        import_export = pd.read_excel(data_path, index_col=0)

        # Gas import prices
        import_carrier_price = {'gas': GAS_PRICE}

        for node in self.nodes.onshore:
            for car in import_carrier_price:
                adopt.fill_carrier_data(input_data_path, value_or_data=import_carrier_price[car],
                                        columns=['Import price'],
                                        carriers=[car], nodes=[node])
                adopt.fill_carrier_data(input_data_path, value_or_data=import_export['Import_' + car][node],
                                        columns=['Import limit'],
                                        carriers=[car], nodes=[node])

        # Emission Price
        for node in self.nodes.all.keys():
            carbon_cost_path = input_data_path / "period1" / "node_data" / node / "CarbonCost.csv"
            carbon_cost_template = pd.read_csv(carbon_cost_path, sep=';', index_col=0, header=0)
            carbon_cost_template['price'] = CO2_PRICE
            carbon_cost_template = carbon_cost_template.reset_index()
            carbon_cost_template.to_csv(carbon_cost_path, sep=';', index=False)


    def _define_co2_pipeline_costs(self):
        """
        Calculate CO2 pipeline costs for onshore and offshore arcs.
        
        - Reads network topologies for CO2 Pipeline onshore and offshore
        - Calculates the cost for each arc using the database functionalities
        - Merges the two tables and removes duplicates
        - Writes the table to clean_data/networks_cost
        """
        netw_data_path = self.clean_data_path / 'networks_topology'
        output_path = self.clean_data_path / 'networks_cost'
        
        # Common cost model options
        base_options = {
            "currency_out": "EUR",
            "financial_year_out": 2025,
            "discount_rate": 0.1,
            "source": "Oeuvray",
            "timeframe": "mid-term",
            "massflow_min_kg_per_s": 5, #Todo to specify!
            "massflow_max_kg_per_s": 10, #Todo to specify!
            "massflow_evaluation_points": 2,
            "p_inlet_bar": 100, #Todo to specify!
            "p_outlet_bar": 100, #Todo to specify!
            "no_intercept": True,
        }
        
        all_costs = []
        
        # Process both onshore and offshore CO2 pipelines
        file_path = netw_data_path / f'CO2_Pipeline.csv'

        # Read the CSV file (handle both comma and semicolon delimiters)
        network_df = pd.read_csv(file_path, sep=None, engine='python')

        # Process each arc
        for idx, row in network_df.iterrows():
            node0 = row['node0']
            node1 = row['node1']
            length_km = row['length']
            terrain = row['Type']

            print(node0, node1, length_km, terrain)

            # Calculate costs using the database cost model
            options = base_options.copy()
            options["length_km"] = length_km
            if terrain == "onshore":
                options["terrain"] = "Onshore"
            elif terrain == "offshore":
                options["terrain"] = "Offshore"

            cost_indicators = adopt.database.calculate_indicators("CO2_Pipeline", options)

            # Store the cost information
            cost_data = {
                'node0': node0,
                'node1': node1,
                'length_km': length_km,
                'terrain': terrain,
                'gamma1': cost_indicators['financial_indicators'].get('gamma1', 0),
                'gamma2': cost_indicators['financial_indicators'].get('gamma2', 0),
                'gamma3': cost_indicators['financial_indicators'].get('gamma3', 0),
                'gamma4': cost_indicators['financial_indicators'].get('gamma4', 0),
                'opex_fixed': cost_indicators['financial_indicators'].get('opex_fixed', 0),
                'opex_variable': cost_indicators['financial_indicators'].get('opex_variable', 0),
                'lifetime': cost_indicators['financial_indicators'].get('lifetime', 0),
                'levelized_cost': cost_indicators['financial_indicators'].get('levelized_cost', 0),
            }
            all_costs.append(cost_data)
        
        costs_df = pd.DataFrame(all_costs)
        
        # Remove duplicates - keep the first occurrence of each unique arc
        # Create a normalized key for each arc (sorted node order)
        # costs_df['arc_key'] = costs_df.apply(
        #     lambda row: tuple(sorted([row['node0'], row['node1']])),
        #     axis=1
        # )
        # costs_df = costs_df.drop_duplicates(subset=['arc_key'], keep='first')
        # costs_df = costs_df.drop('arc_key', axis=1)
        
        # Write to CSV file
        output_file = output_path / 'CO2_Pipeline_costs_per_arc.csv'
        costs_df.to_csv(output_file, index=False)

    def _copy_network_arc_cost_data(self, input_data_path, scenario):
        """
        Copy CO2 pipeline cost data per arc to the network_data folder.
        
        This method copies the CO2_Pipeline_costs_per_arc.csv file from the clean_data
        folder to the respective input_data network_data folder for the given scenario.
        
        :param input_data_path: Path to the input data folder for the scenario
        :param scenario: The scenario name (No_DAC, DAC, Onshore_DAC_only, Offshore_DAC_only)
        """
        # Only copy cost data if CO2 pipeline exists in this scenario
        if scenario == 'No_DAC':
            return
        
        # Read the cost data from clean_data
        source_file = self.clean_data_path / 'networks_cost' / 'CO2_Pipeline_costs_per_arc.csv'
        cost_data = pd.read_csv(source_file)
        
        # Get all unique nodes to create the matrix structure
        all_nodes = sorted(set(list(cost_data['node0'].unique()) + list(cost_data['node1'].unique())))
        
        # Create destination folder for topology matrices
        topology_folder = input_data_path / "period1" / "network_topology" / "new" / "CO2_Pipeline"

        # Create matrix for each gamma parameter
        for gamma in ['gamma1', 'gamma2', 'gamma3', 'gamma4']:
            # Initialize matrix with zeros
            gamma_matrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
            
            # Fill in the values from cost_data
            for _, row in cost_data.iterrows():
                node0 = row['node0']
                node1 = row['node1']
                value = row[gamma]
                gamma_matrix.loc[node0, node1] = value
            
            # Save to CSV with semicolon separator and index
            output_file = topology_folder / f"{gamma}.csv"
            gamma_matrix.to_csv(output_file, sep=";", index=True)

