import streamlit as st
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta

from energybalance import energybalance
from utilities import determine_graph_boundaries
from technology_operation import tec_operation
from technology_sizes import tec_sizes

# Read data from
path = Path('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20231201/20231201133820_Baseline')
node_path = Path.joinpath(path, 'nodes')
nodes = [f.name for f in os.scandir(node_path) if f.is_dir()]

# Sidebar navigation
page_options = ["Energy Balance at Node", "Technology Operation", "Technologies"]
selected_page = st.sidebar.selectbox("Select graph", page_options)

if selected_page in ["Energy Balance at Node", "Technology Operation"]:
    selected_node = st.sidebar.selectbox('Select a node:', nodes)

# Render the selected page
if selected_page == "Energy Balance at Node":
    energybalance_data = pd.read_excel(Path.joinpath(node_path, selected_node, 'Energybalance.xlsx'), sheet_name=None,
                             index_col=0)
    for carrier in energybalance_data:
        energybalance_data[carrier]['Timestep'] = [datetime(2030, 1, 1, 0, 0, 0) + timedelta(hours=hour) for hour in energybalance_data[carrier].index]
    x_min, x_max = determine_graph_boundaries(energybalance_data)
    energybalance(energybalance_data, x_min, x_max)

elif selected_page == "Technology Operation":
    tec_operation_data = pd.read_excel(Path.joinpath(node_path, selected_node, 'TechnologyOperation.xlsx'), sheet_name=None,
                                       index_col=0)
    for tec in tec_operation_data:
        tec_operation_data[tec]['Timestep'] = [datetime(2030, 1, 1, 0, 0, 0) + timedelta(hours=hour) for hour in tec_operation_data[tec].index]
    x_min, x_max = determine_graph_boundaries(tec_operation_data)
    tec_operation(tec_operation_data, x_min, x_max)

elif selected_page == "Technologies":
    tec_size_data = pd.read_excel(Path.joinpath(path, 'Summary.xlsx'), sheet_name='TechnologySizes',
                                       index_col=0)
    tec_sizes(tec_size_data)
