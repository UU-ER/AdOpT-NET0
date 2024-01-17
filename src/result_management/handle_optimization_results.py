from types import SimpleNamespace
import pandas as pd
from pathlib import Path
import numpy as np
import os
import shutil
import tables as tb
import numpy


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

        self.summary = {}
        self.time_independent_results = {'nodes': pd.DataFrame(), 'networks': pd.DataFrame()}
        # self.time_independent_results['nodes']['technologies'] = pd.DataFrame()

        self.time_dependent_results = {'nodes': {}, 'networks': {}}
        self.time_dependent_results['nodes']['technologies'] = pd.DataFrame()
        self.time_dependent_results['nodes']['energybalance'] = {}

    def collect_optimization_result(self, energyhub, timestamp):
        """
        Adds an optimization result to the ResultHandle
        :param energyhub:
        :return:
        """
        # Optimization info
        results = ResultsHandle(configuration=energyhub.configuration)
        # results.read_results(energyhub)

        # Save path
        result_folder_path = Path.joinpath(self.save_path, timestamp)

        if energyhub.solution.solver.termination_condition == 'optimal':
            model = energyhub.model

            # SUMMARY: CREATING DICTIONARY WITH COLLECTED AND CALCULATED PARAMETERS
            summary = self.summary

            # SUMMARY: COSTS
            summary["Total_Costs"] = model.var_total_cost.value
            summary["Carbon_Costs"] = model.var_carbon_cost.value
            summary["Carbon_Revenues"] = model.var_carbon_revenue.value

            # variables needed to calculate tec_cost
            set_t = model.set_t_full
            nr_timesteps_averaged = energyhub.model_information.averaged_data_specs.nr_timesteps_averaged
            tec_capex = sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_capex.value
                                for tec in model.node_blocks[node].set_tecsAtNode)
                            for node in model.set_nodes)
            tec_opex_variable = sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_opex_variable[t].value *
                                            nr_timesteps_averaged for tec in model.node_blocks[node].set_tecsAtNode)
                                        for t in set_t)
                                    for node in model.set_nodes)
            tec_opex_fixed = sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_opex_fixed.value
                                     for tec in model.node_blocks[node].set_tecsAtNode) for node in model.set_nodes)

            summary["Technology_Cost"] = tec_capex + tec_opex_variable + tec_opex_fixed

            summary["Network_Cost"] = model.var_netw_cost.value

            summary["Import_Cost"] = sum(sum(sum(model.node_blocks[node].var_import_flow[t, car].value *
                                                 model.node_blocks[node].para_import_price[t, car].value *
                                                 nr_timesteps_averaged
                                                 for car in model.node_blocks[node].set_carriers)
                                             for t in set_t)
                                         for node in model.set_nodes)
            summary["Export_Revenue"] = sum(sum(sum(model.node_blocks[node].var_export_flow[t, car].value *
                                                    model.node_blocks[node].para_export_price[t, car].value *
                                                    nr_timesteps_averaged
                                                    for car in model.node_blocks[node].set_carriers)
                                                for t in set_t)
                                            for node in model.set_nodes)
            if hasattr(model, 'var_violation_cost'):
                violation_cost = model.var_violation_cost.value
            else:
                violation_cost = 0
            summary["Violation_Cost"] = violation_cost

            # SUMMARY: EMISSIONS
            summary["Net_Emissions"] = model.var_emissions_net.value
            summary["Positive_Emissions"] = model.var_emissions_pos.value
            summary["Negative_Emissions"] = model.var_emissions_neg.value
            summary["Net_Emissions_From_Technologies"] = (
                    sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_tec_emissions_pos[t].value *
                                nr_timesteps_averaged for t in set_t)
                            for tec in model.node_blocks[node].set_tecsAtNode)
                        for node in model.set_nodes) -
                    sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_tec_emissions_neg[t].value *
                                nr_timesteps_averaged for t in set_t)
                            for tec in model.node_blocks[node].set_tecsAtNode)
                        for node in model.set_nodes))
            summary["Net_Emissions_From_Carriers"] = (sum(sum(model.node_blocks[node].var_car_emissions_pos[t].value *
                                                              nr_timesteps_averaged for t in set_t)
                                                          for node in model.set_nodes) -
                                                      sum(sum(model.node_blocks[node].var_car_emissions_neg[t].value *
                                                              nr_timesteps_averaged for t in set_t)
                                                          for node in model.set_nodes))
            if not energyhub.configuration.energybalance.copperplate:
                from_networks = sum(sum(model.network_block[netw].var_netw_emissions_pos[t].value *
                                        nr_timesteps_averaged for t in set_t) for netw in model.set_networks)
            else:
                from_networks = 0
            summary["Net_Emissions_From_Networks"] = from_networks

            # SUMMARY: SOLVER STATUS
            summary["Time_Total"] = energyhub.solution.solver(0).wallclock_time
            summary["lb"] = energyhub.solution.problem(0).lower_bound
            summary["ub"] = energyhub.solution.problem(0).upper_bound
            summary["gap"] = summary["lb"] - summary["ub"]

            # SUMMARY: RUN SPECS
            summary['Objective'] = energyhub.configuration.optimization.objective
            summary['Solver_Status'] = energyhub.solution.solver.termination_condition
            summary['Pareto_Point'] = energyhub.model_information.pareto_point
            summary['Monte_Carlo_Run'] = energyhub.model_information.monte_carlo_run

            if self.timestaging:
                time_stage = energyhub.model_information.averaged_data_specs.stage + 1
            else:
                time_stage = 0
            summary['Time_stage'] = time_stage
            summary['Time_stamp'] = timestamp

            self.summary = pd.DataFrame(data=summary, index=[0])

            # TECHNOLOGY RESULTS
            for node_name in model.set_nodes:
                node_data = model.node_blocks[node_name]
                self.time_dependent_results['nodes'][node_name] = {}

                for tec_name in node_data.set_tecsAtNode:
                    b_tec = node_data.tech_blocks_active[tec_name]
                    tec_results = energyhub.data.technology_data[node_name][tec_name].report_results(b_tec)
                    time_independent = tec_results['time_independent']
                    time_independent['node'] = node_name

                    self.time_independent_results['nodes'] = (
                        self.time_independent_results['nodes'].reindex(columns=time_independent.columns,
                                                                       fill_value=None))
                    self.time_independent_results['nodes'] = pd.concat([self.time_independent_results['nodes'],
                                                                        time_independent], ignore_index=True)

                    self.time_dependent_results['nodes'][node_name][tec_name] = tec_results['time_dependent']

            # NETWORK RESULTS
            if not energyhub.configuration.energybalance.copperplate:
                for netw_name in model.set_networks:
                    b_netw = model.network_block[netw_name]
                    netw_results = energyhub.data.network_data[netw_name].report_results(b_netw)

                    self.time_independent_results['networks'] = (self.time_independent_results['networks'].
                                                                 reindex(
                        columns=netw_results['time_independent'].columns,
                        fill_value=None))
                    self.time_independent_results['networks'] = pd.concat(
                        [self.time_independent_results['networks'], netw_results['time_independent']],
                        ignore_index=True)
                    self.time_dependent_results['networks'][netw_name] = netw_results['time_dependent']

            # Energy Balance at each node
            for node_name in model.set_nodes:
                energybalance = self.time_dependent_results['nodes']['energybalance']
                energybalance[node_name] = {}

                for car in model.node_blocks[node_name].set_carriers:
                    energybalance[node_name][car] = pd.DataFrame(columns=['Technology_inputs',
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
                    energybalance[node_name][car]['Technology_inputs'] = \
                        [sum(node_data.tech_blocks_active[tec].var_input[t, car].value
                             for tec in node_data.set_tecsAtNode
                             if car in node_data.tech_blocks_active[tec].set_input_carriers)
                         for t in set_t]
                    energybalance[node_name][car]['Technology_outputs'] = \
                        [sum(node_data.tech_blocks_active[tec].var_output[t, car].value
                             for tec in node_data.set_tecsAtNode
                             if car in node_data.tech_blocks_active[tec].set_output_carriers)
                         for t in set_t]
                    energybalance[node_name][car]['Generic_production'] = \
                        [node_data.var_generic_production[t, car].value for t in set_t]
                    energybalance[node_name][car]['Network_inflow'] = \
                        [node_data.var_netw_inflow[t, car].value for t in set_t]
                    energybalance[node_name][car]['Network_outflow'] = \
                        [node_data.var_netw_outflow[t, car].value for t in set_t]
                    if hasattr(node_data, 'var_netw_consumption'):
                        energybalance[node_name][car]['Network_consumption'] = \
                            [node_data.var_netw_consumption[t, car].value for t in set_t]
                    energybalance[node_name][car]['Import'] = \
                        [node_data.var_import_flow[t, car].value for t in set_t]
                    energybalance[node_name][car]['Export'] = \
                        [node_data.var_export_flow[t, car].value for t in set_t]
                    energybalance[node_name][car]['Demand'] = \
                        [node_data.para_demand[t, car].value for t in set_t]

        # SAVING RESULTS
        self.save_summary_to_excel(result_folder_path)
        self.save_results_to_h5(result_folder_path)

        if energyhub.model_information.testing:
            shutil.rmtree(Path.joinpath(self.save_path, timestamp))

        return results

    def save_summary_to_excel(self, result_folder_path):
        """
        Writes a summary of the results (containing time independent system results) to excel
        :param Path result_folder_path: folder save path
        :return:
        """

        save_summary_path = Path.joinpath(result_folder_path, 'Summary.xlsx')

        if not os.path.isdir(result_folder_path):
            os.makedirs(result_folder_path)

        with pd.ExcelWriter(save_summary_path) as writer:
            self.summary.to_excel(writer, sheet_name='Summary')

    def save_results_to_h5(self, result_folder_path):
        """
        Writes all results to a h5 file.

        :param Path save_path: path to write excel to
        """

        h5_file_path = os.path.join(result_folder_path, 'optimization_results.h5')

        with pd.HDFStore(h5_file_path, mode='w') as store:
            store['Summary'] = self.summary

            store['Design/Technologies'] = self.time_independent_results['nodes']
            store['Design/Networks'] = self.time_independent_results['networks']

            for node_name in self.time_dependent_results['nodes']['energybalance']:

                for car in self.time_dependent_results['nodes']['energybalance'][node_name]:
                    data = self.time_dependent_results['nodes']['energybalance'][node_name][car]
                    store[f'Operation/Nodes/{node_name}/Energy_balance_{car}'] = data

                for tec_name in self.time_dependent_results['nodes'][node_name]:
                    data = self.time_dependent_results['nodes'][node_name][tec_name]
                    store[f'Operation/Nodes/{node_name}/Technology_operation_{tec_name}'] = data

            for netw_name in self.time_dependent_results['networks']:
                data = self.time_dependent_results['networks'][netw_name]
                store[f'Operation/Networks/{netw_name}'] = data

        # with tb.open_file(h5_file_path, mode='w') as results_file:
        #     # summary dataframe (leave)
        #     summary_data = self.summary
        #     results_file.create_table('/', 'Summary', obj=summary_data.to_records(index=False), title='Summary data')
        #     # summary_data.to_hdf(h5_file_path, 'Summary', format='table', data_columns=True)
        #
        #     # design group containing one table for the technologies (at all nodes) and one for the networks
        #     design = results_file.create_group('/', "time_independent")
        #
        #     # Convert DataFrames to tables and store in the 'time_independent' group
        #     tec_design = self.time_independent_results['nodes']
        #     # tec_design.to_hdf(design, 'Technologies', format='table', data_columns=True)
        #     # data = self.time_independent_results['nodes'].to_records(index=False)
        #     # results_file.create_table(design, 'Technologies', description=data.dtype,
        #     #                           title='Overview of the technologies at all nodes in the system')
        #     netw_design = self.time_independent_results['networks'].to_records()
        #     netw_design.to_hdf(design, 'Networks', format='table', data_columns=True)
        #
        #     # operation group containing two subgroups: nodes and networks.
        #     operation = results_file.create_group('/', "time_dependent")
        #
        #     # node subgroup containing another group for each node
        #     node_operation = results_file.create_group(operation, 'Nodes')
        #
        #     for node_name in self.time_dependent_results['nodes']:
        #         node_group = results_file.create_group(node_operation, node_name)
        #
        #         # within each node group two tables 1) E-balance 2) Tec Operation
        #         energy_balance = results_file.create_group(node_group, 'Energy_balance')
        #         tec_operation = results_file.create_group(node_group, 'Technology_operation')
        #
        #         # Energy balance
        #         for car in self.time_dependent_results['nodes']['energybalance'][node_name]:
        #             data = self.time_dependent_results['nodes']['energybalance'][node_name][car]
        #             results_file.create_table(energy_balance, car, data, 'Energybalance of carrier x')
        #
        #         # Technology operation
        #         for tec_name in self.time_dependent_results['nodes'][node_name]:
        #             data = self.time_dependent_results['nodes'][tec_name]
        #             results_file.create_table(tec_operation, tec_name, data, 'Operation of technology x')
        #
        #     # network subgroup containing one table for each network
        #     network_operation = results_file.create_group(operation, 'Networks')
        #
        #     for netw_name in self.time_dependent_results['networks']:
        #         data = self.time_dependent_results['networks'][netw_name]
        #         results_file.create_table(network_operation, netw_name, data, 'Network Operation')


            # design group containing one dataframe for the technologies (at all nodes) and one for the networks
            # design = results_file.create_group('/', "time_independent")
            #
            # self.time_independent_results['nodes'].to_hdf(h5_file_path, 'time_independent', 'Technologies',
            #                                               format='table', data_columns=True)
            #
            # results_file.create_array(design, 'Technologies', self.time_independent_results['nodes'],
            #                           'Overview of the technologies at all nodes in the system')
            # results_file.create_array(design, 'Networks', self.time_independent_results['networks'],
            #                           'Overview of all networks in the system')
            #
            # # operation group containing two subgroups: nodes and networks.
            #
            # operation = results_file.create_group('/', "time_dependent")
            #
            # # node subgroup containing another group for each node
            # node_operation = results_file.create_group(operation, 'Nodes')
            #
            # for node_name in self.time_dependent_results['nodes']:
            #     results_file.create_group(node_operation, node_name)
            #
            #     # within each node group two groups 1) E-balance 2) Tec Operation
            #     energy_balance = results_file.create_group(node_name, 'Energy_balance')
            #     tec_operation = results_file.create_group(node_name, 'Technology_operation')
            #
            #     # Energy balance
            #     for car in self.time_dependent_results['nodes']['energybalance'][node_name]:
            #         data = self.time_dependent_results['nodes']['energybalance'][node_name][car]
            #         results_file.create_array(energy_balance, car, data, 'Energybalance of carrier x')
            #
            #     # Technology operation
            #     for tec_name in self.time_dependent_results['nodes'][node_name]:
            #         data = self.time_dependent_results['nodes'][tec_name]
            #         results_file.create_array(tec_operation, tec_name, data, 'Operation of technology x')
            #
            # # network subgroup containing one dataframe for each network
            # network_operation = results_file.create_group(operation, 'Networks')
            #
            # for netw_name in self.time_dependent_results['networks']:
            #     data = self.time_dependent_results['networks'][netw_name]
            #     results_file.create_array(network_operation, netw_name, data, 'Network Operation')
