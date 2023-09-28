from types import SimpleNamespace
import pandas as pd
from pathlib import Path
import numpy as np
from .utilities import create_save_folder
import os
import shutil


class ResultsHandle:
    """
    Class to handle optimization results from all runs
    """
    def __init__(self, configuration):
        self.pareto = 0
        self.monte_carlo = 0
        self.timestaging = 0
        self.save_detail = configuration.reporting.save_detailed
        self.save_path = Path(configuration.reporting.save_path)

        self.summary = pd.DataFrame(columns=[
            'Objective',
            'Solver_Status',
            'Pareto_Point',
            'Monte_Carlo_Run',
            'Time_stage',
            'Total_Cost',
            'Emission_Cost',
            'Technology_Cost',
            'Network_Cost',
            'Import_Cost',
            'Export_Revenue',
            'Violation_Cost',
            'Net_Emissions',
            'Positive_Emissions',
            'Negative_Emissions',
            'Net_From_Technologies_Emissions',
            'Net_From_Networks_Emissions',
            'Net_From_Carriers_Emissions',
            'Time_Total',
            'lb',
            'up',
            'gap'
            ])

    def report_optimization_result(self, energyhub, timestamp):
        """
        Adds an optimization result to the ResultHandle
        :param energyhub:
        :return:
        """
        # Optimization info
        results = OptimizationResults(self.save_detail)
        results.read_results(energyhub)
        objective = energyhub.configuration.optimization.objective
        pareto_point = energyhub.model_information.pareto_point
        monte_carlo_run = energyhub.model_information.monte_carlo_run
        if self.timestaging:
            time_stage = energyhub.model_information.averaged_data_specs.stage + 1
        else:
            time_stage = 0

        # Summary
        summary = results.summary
        summary['Objective'] = objective
        summary['Solver_Status'] = energyhub.solution.solver.termination_condition
        summary['Pareto_Point'] = pareto_point
        summary['Monte_Carlo_Run'] = monte_carlo_run
        summary['Time_stage'] = time_stage
        summary['Time_stamp'] = timestamp
        self.summary = pd.concat([self.summary, summary])

        if self.save_detail:
            self.save_path = create_save_folder(self.save_path, timestamp)
            results.write_detailed_results(self.save_path)

        if energyhub.model_information.testing:
            shutil.rmtree(self.save_path)

    def write_excel(self, file_name):
        """
        Writes results to excel
        :param str save_path: folder save path
        :param str file_name: file save name
        :return:
        """

        save_path = Path(self.save_path)
        path = save_path / (file_name + '.xlsx')

        with pd.ExcelWriter(path) as writer:
            self.summary.to_excel(writer, sheet_name='Summary')


class OptimizationResults:
    """
    Class to handle optimization results from a single run
    """
    def __init__(self, detail):
        """
        Reads results to ResultHandle for viewing or export

        :param str detail: 0, or 1, basic excludes energy balance, technology and network operation
        :return: self
        """
        self.summary = pd.DataFrame(columns=['Total_Cost',
                                               'Emission_Cost',
                                               'Emission_Revenues',
                                               'Technology_Cost',
                                               'Network_Cost',
                                               'Import_Cost',
                                               'Export_Revenue',
                                               'Violation_Cost',
                                               'Net_Emissions',
                                               'Positive_Emissions',
                                               'Negative_Emissions',
                                               'Net_From_Technologies_Emissions',
                                               'Net_From_Networks_Emissions',
                                               'Net_From_Carriers_Emissions',
                                               'Time_Total',
                                               'lb',
                                               'up',
                                               'gap'
                                               ])
        self.technologies = pd.DataFrame(columns=['node',
                                                  'technology',
                                                  'size',
                                                  'existing',
                                                  'capex',
                                                  'opex_variable',
                                                  'opex_fixed',
                                                  'emissions_pos',
                                                  'emissions_neg'
                                                  ])
        self.networks = pd.DataFrame(columns=['Network',
                                              'fromNode',
                                              'toNode',
                                              'Size',
                                              'capex',
                                              'opex_fixed',
                                              'opex_variable',
                                              'total_flow',
                                              'total_emissions'
                                              ])
        self.energybalance = {}

        self.detail = detail

        if self.detail:
            self.detailed_results = SimpleNamespace(key1='nodes', key2='networks')
            self.detailed_results.nodes = {}
            self.detailed_results.networks = {}

    def read_results(self, energyhub):

        if energyhub.solution.solver.termination_condition == 'optimal':
            model = energyhub.model

            # Solver status
            total_time = energyhub.solution.solver(0).wallclock_time
            lb = energyhub.solution.problem(0).lower_bound
            ub = energyhub.solution.problem(0).upper_bound
            gap = ub - lb

            # Economics
            total_cost = model.var_total_cost.value
            carbon_costs = model.var_carbon_cost.value
            carbon_revenues = model.var_carbon_revenue.value
            set_t = model.set_t_full
            nr_timesteps_averaged = energyhub.model_information.averaged_data_specs.nr_timesteps_averaged

            tec_capex = sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_capex.value
                                for tec in model.node_blocks[node].set_tecsAtNode)
                            for node in model.set_nodes)
            tec_opex_variable = sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_opex_variable[t].value *
                                            nr_timesteps_averaged
                                            for tec in model.node_blocks[node].set_tecsAtNode)
                                        for t in set_t)
                                    for node in model.set_nodes)
            tec_opex_fixed = sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_opex_fixed.value
                                     for tec in model.node_blocks[node].set_tecsAtNode)
                                 for node in model.set_nodes)
            tec_cost = tec_capex + tec_opex_variable + tec_opex_fixed
            import_cost = sum(sum(sum(model.node_blocks[node].var_import_flow[t, car].value *
                                      model.node_blocks[node].para_import_price[t, car].value *
                                            nr_timesteps_averaged
                                      for car in model.node_blocks[node].set_carriers)
                                  for t in set_t)
                              for node in model.set_nodes)
            export_revenue = sum(sum(sum(model.node_blocks[node].var_export_flow[t, car].value *
                                         model.node_blocks[node].para_export_price[t, car].value *
                                            nr_timesteps_averaged
                                         for car in model.node_blocks[node].set_carriers)
                                     for t in set_t)
                                 for node in model.set_nodes)
            if hasattr(model, 'var_violation_cost'):
                violation_cost = model.var_violation_cost.value
            else:
                violation_cost = 0
            netw_cost = model.var_netw_cost.value

            # Emissions
            net_emissions = model.var_emissions_net.value
            positive_emissions = model.var_emissions_pos.value
            negative_emissions = model.var_emissions_neg.value
            from_technologies = sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_tec_emissions_pos[t].value *
                                            nr_timesteps_averaged
                                            for t in set_t)
                                        for tec in model.node_blocks[node].set_tecsAtNode)
                                    for node in model.set_nodes) - \
                                sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_tec_emissions_neg[t].value *
                                            nr_timesteps_averaged
                                            for t in set_t)
                                        for tec in model.node_blocks[node].set_tecsAtNode)
                                    for node in model.set_nodes)
            from_carriers = sum(sum(model.node_blocks[node].var_car_emissions_pos[t].value *
                                            nr_timesteps_averaged
                                    for t in set_t)
                                for node in model.set_nodes) - \
                            sum(sum(model.node_blocks[node].var_car_emissions_neg[t].value *
                                            nr_timesteps_averaged
                                    for t in set_t)
                                for node in model.set_nodes)
            if not energyhub.configuration.energybalance.copperplate:
                from_networks = sum(sum(model.network_block[netw].var_netw_emissions_pos[t].value *
                                                nr_timesteps_averaged
                                        for t in set_t)
                                    for netw in model.set_networks)
            else:
                from_networks = 0

            self.summary.loc[len(self.summary.index)] = \
                [total_cost,
                 carbon_costs, carbon_revenues,
                 tec_cost, netw_cost,
                 import_cost, export_revenue,
                 violation_cost,
                 net_emissions,
                 positive_emissions, negative_emissions,
                 from_technologies, from_networks, from_carriers,
                 total_time, lb, ub, gap]

            # Technology Results
            for node_name in model.set_nodes:
                node_data = model.node_blocks[node_name]
                if self.detail:
                    self.detailed_results.nodes[node_name] = {}

                for tec_name in node_data.set_tecsAtNode:
                    b_tec = node_data.tech_blocks_active[tec_name]
                    tec_results = energyhub.data.technology_data[node_name][tec_name].report_results(b_tec)
                    time_independent = tec_results['time_independent']
                    time_independent['node'] = node_name

                    self.technologies = pd.concat([self.technologies, time_independent], ignore_index=True)
                    if self.detail:
                        self.detailed_results.nodes[node_name][tec_name] = tec_results['time_dependent']

            # Network Results
            if not energyhub.configuration.energybalance.copperplate:
                for netw_name in model.set_networks:
                    b_netw = model.network_block[netw_name]
                    netw_results = energyhub.data.network_data[netw_name].report_results(b_netw)

                    self.networks = pd.concat([self.networks, netw_results['time_independent']], ignore_index=True)
                    if self.detail:
                        self.detailed_results.networks[netw_name] = netw_results['time_dependent']

            # Energy Balance at each node
            if self.detail:
                for node_name in model.set_nodes:
                    self.energybalance[node_name] = {}
                    for car in model.node_blocks[node_name].set_carriers:
                        self.energybalance[node_name][car] = pd.DataFrame(columns=[
                                                                                    'Technology_inputs',
                                                                                    'Technology_outputs',
                                                                                    'Generic_production',
                                                                                    'Network_inflow',
                                                                                    'Network_outflow',
                                                                                    'Network_consumption',
                                                                                    'Import',
                                                                                    'Export',
                                                                                    'Demand'
                                                                                    ])
                        node_data = model.node_blocks[node_name]
                        self.energybalance[node_name][car]['Technology_inputs'] = \
                            [sum(node_data.tech_blocks_active[tec].var_input[t, car].value
                                 for tec in node_data.set_tecsAtNode
                                 if car in node_data.tech_blocks_active[tec].set_input_carriers)
                             for t in set_t]
                        self.energybalance[node_name][car]['Technology_outputs'] = \
                            [sum(node_data.tech_blocks_active[tec].var_output[t, car].value
                                 for tec in node_data.set_tecsAtNode
                                 if car in node_data.tech_blocks_active[tec].set_output_carriers)
                             for t in set_t]
                        self.energybalance[node_name][car]['Generic_production'] = \
                            [node_data.var_generic_production[t, car].value for t in set_t]
                        self.energybalance[node_name][car]['Network_inflow'] = \
                            [node_data.var_netw_inflow[t, car].value for t in set_t]
                        self.energybalance[node_name][car]['Network_outflow'] = \
                            [node_data.var_netw_outflow[t, car].value for t in set_t]
                        if hasattr(node_data, 'var_netw_consumption'):
                            self.energybalance[node_name][car]['Network_consumption'] = \
                                [node_data.var_netw_consumption[t, car].value for t in set_t]
                        self.energybalance[node_name][car]['Import'] = \
                            [node_data.var_import_flow[t, car].value for t in set_t]
                        self.energybalance[node_name][car]['Export'] = \
                            [node_data.var_export_flow[t, car].value for t in set_t]
                        self.energybalance[node_name][car]['Demand'] = \
                            [node_data.para_demand[t, car].value for t in set_t]


    def write_detailed_results(self, save_path):
        """
        Writes results to excel table

        :param Path save_path: path to write excel to
        """
        # Write Summary
        save_summary_path = Path.joinpath(save_path, 'Summary.xlsx')
        with pd.ExcelWriter(save_summary_path) as writer:
            self.summary.to_excel(writer, sheet_name='Summary')
            self.technologies.to_excel(writer, sheet_name='TechnologySizes')
            self.networks.to_excel(writer, sheet_name='Networks')

        if self.detail:
            save_networks_path = Path.joinpath(save_path, 'Networks')
            os.makedirs(save_networks_path)
            save_nodes_path = Path.joinpath(save_path, 'Nodes')
            os.makedirs(save_nodes_path)

            # Networks
            for netw_name in self.detailed_results.networks:
                save_network_path = Path.joinpath(save_networks_path, netw_name + '.xlsx')
                with pd.ExcelWriter(save_network_path) as writer:
                    self.detailed_results.networks[netw_name].to_excel(writer, sheet_name='Time_dependent_vars')

            # Nodes
            for node in self.energybalance:
                save_node_path = Path.joinpath(save_nodes_path, node)
                os.makedirs(save_node_path)

                # Energy balance
                save_energybalance_path = Path.joinpath(save_node_path, 'Energybalance.xlsx')
                with pd.ExcelWriter(save_energybalance_path) as writer:
                    for car in self.energybalance[node]:
                        self.energybalance[node][car].to_excel(writer, sheet_name=car)

                # Technology operation
                save_technologies_path = Path.joinpath(save_node_path, 'TechnologyOperation.xlsx')
                if self.detailed_results.nodes[node]:
                    with pd.ExcelWriter(save_technologies_path) as writer:
                        for tec_name in self.detailed_results.nodes[node]:
                            self.detailed_results.nodes[node][tec_name].to_excel(writer, sheet_name=tec_name)
