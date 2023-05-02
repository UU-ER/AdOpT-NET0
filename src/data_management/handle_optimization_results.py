from types import SimpleNamespace
import pandas as pd
import src.global_variables as global_variables
import numpy as np

class ResultsHandle:
    """
    Class to handle optimization results
    """
    def __init__(self):
        self.economics = pd.DataFrame(columns=['Total_Cost',
                                               'Emission_Cost',
                                               'Technology_Cost',
                                               'Network_Cost',
                                               'Import_Cost',
                                               'Export_Revenue'
                                               ])
        self.emissions = pd.DataFrame(columns=['Net',
                                               'Positive',
                                               'Negative',
                                               'Net_From_Technologies',
                                               'Net_From_Networks',
                                               'Net_From_Carriers'
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

    def read_results(self, energyhub, detail = 'full'):
        """
        Reads results to ResultHandle for viewing or export

        :param EnergyHub energyhub: instance the EnergyHub Class
        :param str detail: 'full', or 'basic', basic excludes energy balance, technology and network operation
        :return: self
        """
        model = energyhub.model

        # Economics
        total_cost = model.var_total_cost.value
        # emission_cost = model.var_emission_cost.value
        # Todo: Add this here, if it is done
        emission_cost = 0
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
        netw_cost = model.var_netw_cost.value
        self.economics.loc[len(self.economics.index)] = \
            [total_cost, emission_cost, tec_cost, netw_cost, import_cost, export_revenue]

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
        from_networks = sum(sum(model.network_block[netw].var_netw_emissions_pos[t].value *
                                        nr_timesteps_averaged
                                for t in set_t)
                            for netw in model.set_networks)
        self.emissions.loc[len(self.emissions.index)] = \
            [net_emissions, positive_emissions, negative_emissions, from_technologies,
             from_networks, from_carriers]

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
                    opex_var = sum(arc_data.var_opex_variable[t]
                                   for t in set_t)
                    total_flow = sum(arc_data.var_flow[t].value
                                     for t in set_t)
                opex_fix = capex * netw_data.para_opex_fixed.value

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

                    self.detailed_results.nodes[node_name][tec_name] = df

            # Detailed results for networks
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
        file_name = path + '.xlsx'

        with pd.ExcelWriter(file_name) as writer:
            self.economics.to_excel(writer, sheet_name='Economics')
            self.emissions.to_excel(writer, sheet_name='Emissions')
            self.technologies.to_excel(writer, sheet_name='TechnologySizes')
            self.networks.to_excel(writer, sheet_name='Networks')
            for node in self.energybalance:
                for car in self.energybalance[node]:
                    self.energybalance[node][car].to_excel(writer, sheet_name='Balance_' + node + '_' + car)
            for node in self.detailed_results.nodes:
                for tec_name in self.detailed_results.nodes[node]:
                    self.detailed_results.nodes[node][tec_name].to_excel(writer, sheet_name=
                                                                              'Tec_' + node + '_' + tec_name)