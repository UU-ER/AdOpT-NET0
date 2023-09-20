from types import SimpleNamespace
import pandas as pd
import src.global_variables as global_variables
import numpy as np

class ResultsHandle:
    """
    Class to handle optimization results from all runs
    """
    def __init__(self, configuration):
        self.pareto = 0
        self.monte_carlo = 0
        self.timestaging = 0
        self.save_detail = configuration.optimization.save_detail

        self.summary = pd.DataFrame(columns=[
            'Objective',
            'Solver_Status',
            'Pareto_Point',
            'Monte_Carlo_Run',
            'Time_stage',
            'Total_Cost',
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

        self.detailed_results = []

    def add_optimization_result(self, energyhub, timestamp):
        """
        Adds an optimization result to the ResultHandle
        :param energyhub:
        :return:
        """
        # Optimization info
        optimization_result = OptimizationResults(energyhub, self.save_detail)
        objective = energyhub.configuration.optimization.objective
        pareto_point = global_variables.pareto_point
        monte_carlo_run = global_variables.monte_carlo_run
        if self.timestaging:
            time_stage = global_variables.averaged_data_specs.stage +1
        else:
            time_stage = 0

        # Summary
        summary = optimization_result.summary
        summary['Objective'] = objective
        summary['Solver_Status'] = energyhub.solution.solver.termination_condition
        summary['Pareto_Point'] = pareto_point
        summary['Monte_Carlo_Run'] = monte_carlo_run
        summary['Time_stage'] = time_stage
        summary['Time_stamp'] = timestamp
        self.summary = pd.concat([self.summary, summary])

        self.detailed_results.append(optimization_result)

    def write_excel(self, path):
        """
        Writes results to excel
        :param path: save path
        :return:
        """
        file_name = path + '.xlsx'
        with pd.ExcelWriter(file_name) as writer:
            self.summary.to_excel(writer, sheet_name='Summary')

        i = 1
        if not self.save_detail == 'minimal':
            for result in self.detailed_results:
                result.write_excel(path + '_detailed_' + str(i))
                i += 1


class OptimizationResults:
    """
    Class to handle optimization results from a single run
    """
    def __init__(self, energyhub, detail = 'full'):
        """
        Reads results to ResultHandle for viewing or export

        :param EnergyHub energyhub: instance the EnergyHub Class
        :param str detail: 'full', or 'basic', basic excludes energy balance, technology and network operation
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
        self.technologies = pd.DataFrame(columns=['Node',
                                                  'Technology',
                                                  'Size',
                                                  'capex',
                                                  'opex_fixed',
                                                  'opex_variable'
                                                  ])
        self.networks = pd.DataFrame(columns=['Network',
                                              'fromNode',
                                              'toNode',
                                              'Size',
                                              'capex',
                                              'opex_fixed',
                                              'opex_variable',
                                              'total_flow'
                                              ])
        self.energybalance = {}
        self.detailed_results = SimpleNamespace(key1='nodes', key2='networks')
        self.detailed_results.nodes = {}
        self.detailed_results.networks = {}

        if energyhub.solution.solver.termination_condition == 'optimal':
            self.read_results(energyhub, detail)

    def read_results(self, energyhub, detail):
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
        nr_timesteps_averaged = global_variables.averaged_data_specs.nr_timesteps_averaged

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

        # Technology Sizes
        for node_name in model.set_nodes:
            node_data = model.node_blocks[node_name]
            for tec_name in node_data.set_tecsAtNode:
                tec_data = node_data.tech_blocks_active[tec_name]
                s = tec_data.var_size.value
                capex = tec_data.var_capex.value
                opex_fix = tec_data.var_opex_fixed.value
                opex_var = sum(tec_data.var_opex_variable[t].value
                               for t in set_t)
                self.technologies.loc[len(self.technologies.index)] = \
                    [node_name, tec_name, s, capex, opex_fix, opex_var]

        # Network Sizes
        if not energyhub.configuration.energybalance.copperplate:
            for netw_name in model.set_networks:
                netw_data = model.network_block[netw_name]
                for arc in netw_data.set_arcs:
                    arc_data = netw_data.arc_block[arc]
                    fromNode = arc[0]
                    toNode = arc[1]
                    s = arc_data.var_size.value
                    capex = arc_data.var_capex.value
                    if global_variables.clustered_data:
                        sequence = energyhub.data.k_means_specs.full_resolution['sequence']
                        opex_var = sum(arc_data.var_opex_variable[sequence[t - 1]].value
                                       for t in set_t)
                        total_flow = sum(arc_data.var_flow[sequence[t - 1]].value
                                         for t in set_t)
                    else:
                        opex_var = sum(arc_data.var_opex_variable[t].value
                                       for t in set_t)
                        total_flow = sum(arc_data.var_flow[t].value
                                         for t in set_t)
                    opex_fix = netw_data.para_opex_fixed.value * arc_data.var_capex_aux.value

                    self.networks.loc[len(self.networks.index)] = \
                        [netw_name, fromNode, toNode, s, capex, opex_fix, opex_var, total_flow]

        if detail == 'full':
            # Energy Balance @ each node
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

            # Detailed results for technologies
            for node_name in model.set_nodes:
                node_data = model.node_blocks[node_name]
                self.detailed_results.nodes[node_name] = {}
                for tec_name in node_data.set_tecsAtNode:
                    tec_data = node_data.tech_blocks_active[tec_name]
                    technology_model = energyhub.data.technology_data[node_name][tec_name].technology_model

                    if technology_model == 'STOR':
                        if global_variables.clustered_data:
                            time_set = model.set_t_full
                        else:
                            time_set = model.set_t_full
                        if tec_data.find_component('var_input'):
                            input = tec_data.var_input
                            output = tec_data.var_output

                    else:
                        time_set = set_t
                        if tec_data.find_component('var_input'):
                            input = tec_data.var_input
                        output = tec_data.var_output

                    df = pd.DataFrame()

                    for car in tec_data.set_input_carriers:
                        if tec_data.find_component('var_input'):
                            df['input_' + car] = [input[t, car].value for t in time_set]

                    for car in tec_data.set_output_carriers:
                        df['output_' + car] = [output[t, car].value for t in time_set]

                    if tec_data.find_component('var_storage_level'):
                        for car in tec_data.set_input_carriers:
                            df['storage_level_' + car] = [tec_data.var_storage_level[t, car].value for t in time_set]

                    if tec_data.find_component('var_spilling'):
                        df['spilling'] = [tec_data.var_spilling[t].value for t in time_set]

                    self.detailed_results.nodes[node_name][tec_name] = df

            # Detailed results for networks
            if not energyhub.configuration.energybalance.copperplate:

                for netw_name in model.set_networks:
                    netw_data = model.network_block[netw_name]
                    self.detailed_results.networks[netw_name] = {}
                    for arc in netw_data.set_arcs:
                        arc_data = netw_data.arc_block[arc]
                        df = pd.DataFrame()

                        if global_variables.clustered_data:
                            sequence = energyhub.data.k_means_specs.full_resolution['sequence']
                            df['flow'] = [arc_data.var_flow[sequence[t - 1]].value for t in set_t]
                            df['losses'] = [arc_data.var_losses[sequence[t - 1]].value for t in set_t]
                            if netw_data.find_component('var_consumption_send'):
                                for car in netw_data.set_consumed_carriers:
                                    df['consumption_send' + car] = \
                                        [arc_data.var_consumption_send[sequence[t - 1], car].value for t in set_t]
                                    df['consumption_receive' + car] = \
                                        [arc_data.var_consumption_receive[sequence[t - 1], car].value for t in set_t]
                        else:
                            df['flow'] = [arc_data.var_flow[t].value for t in set_t]
                            df['losses'] = [arc_data.var_losses[t].value for t in set_t]
                            if netw_data.find_component('var_consumption_send'):
                                for car in netw_data.set_consumed_carriers:
                                    df['consumption_send' + car] = \
                                        [arc_data.var_consumption_send[t, car].value for t in set_t]
                                    df['consumption_receive' + car] = \
                                        [arc_data.var_consumption_receive[t, car].value for t in set_t]

                        self.detailed_results.networks[netw_name]['_'.join(arc)] = df

    def write_excel(self, path):
        """
        Writes results to excel table

        :param str path: path to write excel to
        """
        def shorten_string(str, length):
            if len(str) > length:
                str = str[0:length-1]
            return str

        file_name = path + '.xlsx'

        with pd.ExcelWriter(file_name) as writer:
            self.summary.to_excel(writer, sheet_name='Summary')
            self.technologies.to_excel(writer, sheet_name='TechnologySizes')
            self.networks.to_excel(writer, sheet_name='Networks')
            for node in self.energybalance:
                for car in self.energybalance[node]:
                    sheet_name = shorten_string(node + '_' + car, 30)
                    self.energybalance[node][car].to_excel(writer, sheet_name=sheet_name)
            for node in self.detailed_results.nodes:
                for tec_name in self.detailed_results.nodes[node]:
                    sheet_name = shorten_string(node + '_' + tec_name, 30)
                    self.detailed_results.nodes[node][tec_name].to_excel(writer, sheet_name=sheet_name)