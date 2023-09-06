import networkx as nx
import pandas as pd

data_path = r'./cases/NorthSea_v4/Nodes/Nodes.xlsx'

network = pd.read_excel(data_path,
                         sheet_name='Nodes')

G = nx.from_pandas_edgelist(network, source='fromNode', target='toNode', edge_attr='Size')
nx.draw_networkx(G, arrows=True, nodelist = [x for x in network['fromNode'] if x.startswith('onNL')])