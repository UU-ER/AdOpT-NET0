import streamlit as st
import numpy as np
import pandas as pd
import h5py
import os
from pathlib import Path


path = Path('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20231201/20231201133820_Baseline/Nodes')
nodes = [f.name for f in os.scandir(path) if f.is_dir()]

selected_node = st.selectbox('Select a node:', nodes)
# Specify the path to the selected folder
carriers = pd.read_excel(Path.joinpath(path, selected_node, 'Energybalance.xlsx'), sheet_name=None, index_col=0)

selected_carrier = st.selectbox('Select a carrier:', carriers.keys())

st.line_chart(carriers[selected_carrier][['Import', 'Demand', 'Technology_outputs']])
