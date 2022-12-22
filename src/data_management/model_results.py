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
        self.detailed_results = SimpleNamespace(key1='Nodes', key2='Networks')

    def read_results(self, energyhub):
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
                                                                           'Import'
                                                                           'Export'
                                                                            ])
                node_data = m.node_blocks[node_name]
                tec_output = [sum(node_data.tech_blocks_active[tec].var_output[t, car] for tec in node_data.set_tecsAtNode if
                    car in node_data.tech_blocks_active[tec].set_output_carriers) for t in m.set_t]
                tec_input = [sum(node_data.tech_blocks_active[tec].var_input[t, car] for tec in node_data.set_tecsAtNode if
                    car in node_data.tech_blocks_active[tec].set_input_carriers) for t in m.set_t]
                netw_inflow = [node_data.var_netw_inflow[t, car] for t in m.set_t]
                netw_outflow = [node_data.var_netw_outflow[t, car] for t in m.set_t]
                netw_consumption = [node_data.var_netw_consumption[t, car] for t in m.set_t]
                import_flow = [node_data.var_import_flow[t, car] for t in m.set_t]
                export_flow = [node_data.var_export_flow[t, car] for t in m.set_t]





