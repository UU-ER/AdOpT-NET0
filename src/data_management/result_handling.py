from types import SimpleNamespace
import pandas as pd

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
        self.emissions = pd.DataFrame(columns=['Negative',
                                               'Positive',
                                               'From_Technologies',
                                               'From_Networks',
                                               'From_Import',
                                               'From_Export'
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

        # Economics
        total_cost = m.var_total_cost.value
        # emission_cost = m.var_emission_cost.value
        # Todo: Add this here, if it is done
        emission_cost = 0
        tec_cost = sum(sum(m.node_blocks[node].tech_blocks_active[tec].var_CAPEX.value
                           for tec in m.node_blocks[node].set_tecsAtNode) + \
                       sum(sum(m.node_blocks[node].tech_blocks_active[tec].var_OPEX_variable[t].value
                               for tec in m.node_blocks[node].set_tecsAtNode)
                           for t in m.set_t) + \
                       sum(m.node_blocks[node].tech_blocks_active[tec].var_OPEX_fixed.value
                           for tec in m.node_blocks[node].set_tecsAtNode) \
                       for node in m.set_nodes)
        netw_cost = m.var_netw_cost
        import_cost = sum(sum(sum(m.node_blocks[node].var_import_flow[t, car].value * \
                                   m.node_blocks[node].para_import_price[t, car].value
                                for car in m.set_carriers)
                              for t in m.set_t) \
                            for node in m.set_nodes)
        export_revenue = sum(sum(sum(m.node_blocks[node].var_export_flow[t, car].value * \
                                   m.node_blocks[node].para_export_price[t, car].value
                                for car in m.set_carriers)
                              for t in m.set_t) \
                            for node in m.set_nodes)
        self.economics.loc[len(self.economics.index)] = \
            [total_cost, emission_cost, tec_cost, netw_cost, import_cost, export_revenue]

        # Emissions
        # Todo: Add this here, if it is done

        # Technology Sizes
        for node_name in m.set_nodes:
            node_data = m.node_blocks[node_name]
            for tec_name in node_data.set_tecsAtNode:
                tec_data = node_data.tech_blocks_active[tec_name]
                s = tec_data.var_size.value
                capex = tec_data.var_CAPEX.value
                opex_fix = tec_data.var_OPEX_fixed.value
                opex_var = sum(tec_data.var_OPEX_variable[t].value for t in m.set_t)
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
                opex_var = arc_data.var_OPEX_variable.value
                opex_fix = capex * netw_data.para_OPEX_fixed.value
                total_flow = sum(arc_data.var_flow[t].value for t in m.set_t)
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

                df = pd.DataFrame()

                for car in tec_data.set_input_carriers:
                    if tec_data.find_component('var_input'):
                        df['input_' + car] = [tec_data.var_input[t, car].value for t in m.set_t]

                for car in tec_data.set_output_carriers:
                    df['output_' + car] = [tec_data.var_output[t, car].value for t in m.set_t]

                if tec_data.find_component('var_storage_level'):
                    for car in tec_data.set_input_carriers:
                        df['storage_level_' + car] = [tec_data.var_storage_level[t, car].value for t in m.set_t]

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
