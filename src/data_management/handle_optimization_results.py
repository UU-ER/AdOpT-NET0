from types import SimpleNamespace
import pandas as pd
import numpy as np

import src.config_model as m_config


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
                                                  'CAPEX',
                                                  'OPEX_fixed',
                                                  'OPEX_variable'
                                                  ])
        self.networks = pd.DataFrame(columns=['Network',
                                              'fromNode',
                                              'toNode',
                                              'Size',
                                              'CAPEX',
                                              'OPEX_fixed',
                                              'OPEX_variable',
                                              'total_flow'
                                              ])
        self.energybalance = {}
        self.detailed_results = SimpleNamespace(key1='nodes', key2='networks')
        self.detailed_results.nodes = {}
        self.detailed_results.networks = {}

    def read_results(self, energyhub):
        """
        Reads results to ResultHandle for viewing or export

        :param EnergyHub energyhub: instance the EnergyHub Class
        :return: self
        """
        m = energyhub.model

        if m_config.presolve.clustered_data == 1:
            occurrence_hour = energyhub.data.specifications_time_resolution['factors']['factor'].to_numpy()
        else:
            occurrence_hour = np.ones(len(m.set_t))


        # Economics
        total_cost = m.var_total_cost.value
        # emission_cost = m.var_emission_cost.value
        # Todo: Add this here, if it is done
        emission_cost = 0
        tec_CAPEX = sum(sum(m.node_blocks[node].tech_blocks_active[tec].var_CAPEX.value
                            for tec in m.node_blocks[node].set_tecsAtNode)
                        for node in m.set_nodes)
        tec_OPEX_variable = sum(sum(sum(m.node_blocks[node].tech_blocks_active[tec].var_OPEX_variable[t].value *
                                        occurrence_hour[t - 1]
                                        for tec in m.node_blocks[node].set_tecsAtNode)
                                    for t in m.set_t)
                                for node in m.set_nodes)
        tec_OPEX_fixed = sum(sum(m.node_blocks[node].tech_blocks_active[tec].var_OPEX_fixed.value
                                 for tec in m.node_blocks[node].set_tecsAtNode)
                             for node in m.set_nodes)
        tec_cost = tec_CAPEX + tec_OPEX_variable + tec_OPEX_fixed
        import_cost = sum(sum(sum(m.node_blocks[node].var_import_flow[t, car].value *
                                    m.node_blocks[node].para_import_price[t, car].value *
                                    occurrence_hour[t - 1]
                                  for car in m.set_carriers)
                              for t in m.set_t)
                          for node in m.set_nodes)
        export_revenue = sum(sum(sum(m.node_blocks[node].var_export_flow[t, car].value *
                                     m.node_blocks[node].para_export_price[t, car].value *
                                     occurrence_hour[t - 1]
                                    for car in m.set_carriers)
                                 for t in m.set_t)
                             for node in m.set_nodes)
        netw_cost = m.var_netw_cost.value
        self.economics.loc[len(self.economics.index)] = \
            [total_cost, emission_cost, tec_cost, netw_cost, import_cost, export_revenue]

        # Emissions
        net_emissions = m.var_emissions_net.value
        positive_emissions = m.var_emissions_pos.value
        negative_emissions = m.var_emissions_neg.value
        from_technologies = sum(sum(sum(m.node_blocks[node].tech_blocks_active[tec].var_tec_emissions[t].value *
                                        occurrence_hour[t - 1]
                                        for t in m.set_t)
                                    for tec in m.node_blocks[node].set_tecsAtNode)
                                for node in m.set_nodes) - \
                            sum(sum(sum(m.node_blocks[node].tech_blocks_active[tec].var_tec_emissions_neg[t].value *
                                        occurrence_hour[t - 1]
                                        for t in m.set_t)
                                    for tec in m.node_blocks[node].set_tecsAtNode)
                                for node in m.set_nodes)
        from_carriers = sum(sum(m.node_blocks[node].var_car_emissions[t].value * occurrence_hour[t - 1]
                                for t in m.set_t)
                            for node in m.set_nodes) - \
                        sum(sum(m.node_blocks[node].var_car_emissions_neg[t].value * occurrence_hour[t - 1]
                                for t in m.set_t)
                            for node in m.set_nodes)
        from_networks = sum(sum(m.network_block[netw].var_netw_emissions[t].value * occurrence_hour[t - 1]
                                for t in m.set_t)
                            for netw in m.set_networks)
        self.emissions.loc[len(self.emissions.index)] = \
            [net_emissions, positive_emissions, negative_emissions, from_technologies,
             from_networks, from_carriers]

        # Technology Sizes
        for node_name in m.set_nodes:
            node_data = m.node_blocks[node_name]
            for tec_name in node_data.set_tecsAtNode:
                tec_data = node_data.tech_blocks_active[tec_name]
                s = tec_data.var_size.value
                capex = tec_data.var_CAPEX.value
                opex_fix = tec_data.var_OPEX_fixed.value
                opex_var = sum(tec_data.var_OPEX_variable[t].value *
                                occurrence_hour[t - 1]
                            for t in m.set_t)
                self.technologies.loc[len(self.technologies.index)] = \
                    [node_name, tec_name, s, capex, opex_fix, opex_var]

        # Network Sizes
        for netw_name in m.set_networks:
            netw_data = m.network_block[netw_name]
            for arc in netw_data.set_arcs:
                arc_data = netw_data.arc_block[arc]
                fromNode = arc[0]
                toNode = arc[1]
                s = arc_data.var_size.value
                capex = arc_data.var_CAPEX.value
                opex_var = sum(arc_data.var_OPEX_variable[t] *
                                        occurrence_hour[t - 1]
                                 for t in m.set_t)
                opex_fix = capex * netw_data.para_OPEX_fixed.value
                total_flow = sum(arc_data.var_flow[t].value *
                                occurrence_hour[t - 1]
                             for t in m.set_t)
                self.networks.loc[len(self.networks.index)] = \
                    [netw_name, fromNode, toNode, s, capex, opex_fix, opex_var, total_flow]

        # Energy Balance @ each node
        for car in m.set_carriers:
            self.energybalance[car] = {}
            for node_name in m.set_nodes:
                self.energybalance[car][node_name] = pd.DataFrame(columns=[
                                                                            'Technology_inputs',
                                                                            'Technology_outputs',
                                                                            'Network_inflow',
                                                                            'Network_outflow',
                                                                            'Network_consumption',
                                                                            'Import',
                                                                            'Export',
                                                                            'Demand'
                                                                            ])
                node_data = m.node_blocks[node_name]
                self.energybalance[car][node_name]['Technology_inputs'] = \
                    [sum(node_data.tech_blocks_active[tec].var_output[t, car].value
                         for tec in node_data.set_tecsAtNode
                         if car in node_data.tech_blocks_active[tec].set_output_carriers)
                     for t in m.set_t]
                self.energybalance[car][node_name]['Technology_outputs'] = \
                    [sum(node_data.tech_blocks_active[tec].var_input[t, car].value
                         for tec in node_data.set_tecsAtNode
                         if car in node_data.tech_blocks_active[tec].set_input_carriers)
                     for t in m.set_t]
                self.energybalance[car][node_name]['Network_inflow'] = \
                    [node_data.var_netw_inflow[t, car].value for t in m.set_t]
                self.energybalance[car][node_name]['Network_outflow'] = \
                    [node_data.var_netw_outflow[t, car].value for t in m.set_t]
                self.energybalance[car][node_name]['Network_consumption'] = \
                    [node_data.var_netw_consumption[t, car].value for t in m.set_t]
                self.energybalance[car][node_name]['Import'] = \
                    [node_data.var_import_flow[t, car].value for t in m.set_t]
                self.energybalance[car][node_name]['Export'] = \
                    [node_data.var_export_flow[t, car].value for t in m.set_t]
                self.energybalance[car][node_name]['Demand'] = \
                    [node_data.para_demand[t, car].value for t in m.set_t]

        # Detailed results for technologies
        for node_name in m.set_nodes:
            node_data = m.node_blocks[node_name]
            self.detailed_results.nodes[node_name] = {}
            for tec_name in node_data.set_tecsAtNode:
                tec_data = node_data.tech_blocks_active[tec_name]
                tec_type = energyhub.data.technology_data[node_name][tec_name]['TechnologyPerf']['tec_type']

                if tec_type == 'STOR':
                    time_set = m.set_t_full
                    if tec_data.find_component('var_input'):
                        input = tec_data.var_input_full_resolution
                        output = tec_data.var_output_full_resolution

                else:
                    time_set = m.set_t
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
        for netw_name in m.set_networks:
            netw_data = m.network_block[netw_name]
            self.detailed_results.networks[netw_name] = {}
            for arc in netw_data.set_arcs:
                arc_data = netw_data.arc_block[arc]
                df = pd.DataFrame()

                df['flow'] = [arc_data.var_flow[t].value for t in m.set_t]
                df['losses'] = [arc_data.var_losses[t].value for t in m.set_t]
                if tec_data.find_component('var_consumption_send'):
                    for car in netw_data.set_consumed_carriers:
                        df['consumption_send' + car] = \
                            [arc_data.var_consumption_send[t, car].value for t in m.set_t]
                        df['consumption_receive' + car] = \
                            [arc_data.var_consumption_receive[t, car].value for t in m.set_t]

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
            for car in self.energybalance:
                for node_name in self.energybalance[car]:
                    self.energybalance[car][node_name].to_excel(writer, sheet_name='Balance_' + node_name + '_' + car)
            for node_name in self.detailed_results.nodes:
                for tec_name in self.detailed_results.nodes[node_name]:
                    self.detailed_results.nodes[node_name][tec_name].to_excel(writer, sheet_name=
                                                                              'DetTec_' + node_name + '_' + tec_name)
