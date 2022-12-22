from types import SimpleNamespace
import pandas as pd

class ResultsHandle:
    """
    Class to handle optimization results
    """
    def __init__(self):
        self.economics = {}
        self.emissions = {}
        self.technologies = pd.DataFrame(columns=['Node', 'Technology', 'Size', 'CAPEX', 'OPEX_fixed', 'OPEX_variable'])
        self.networks = pd.DataFrame(columns=['Network', 'fromNode', 'toNode', 'Size', 'CAPEX', 'OPEX_fixed', 'OPEX_variable'])
        self.energybalance = {}
        self.detailed_results = SimpleNamespace(key1='Nodes', key2='Networks')

    def read_results(self, energyhub):
        m = energyhub.model

        # Technology Sizes
        for node_name in m.set_nodes:
            node_data = m.node_blocks[node_name]
            for car in m.set_carriers:
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
            for car in m.set_carriers:
                for tec_name in node_data.set_tecsAtNode:
                    tec_data = node_data.tech_blocks_active[tec_name]
                    s = tec_data.var_size.value
                    capex = tec_data.var_CAPEX.value
                    opex_fix = tec_data.var_OPEX_fixed.value
                    opex_var = sum(tec_data.var_OPEX_variable[t].value for t in m.set_t)
                    self.technologies.loc[len(self.technologies.index)] = \
                        [node_name, tec_name, s, capex, opex_fix, opex_var]





