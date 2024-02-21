import streamlit as st
from pathlib import Path
import h5py
import pandas as pd
from utilities import *
from networks import plot_nodes
import folium
from streamlit_folium import st_folium
from branca.colormap import linear
from folium.plugins import PolyLineTextPath, PolyLineOffset
import sys


# Initialize Session States
if 'path' not in st.session_state:
    st.session_state['path'] = {}
if 'network_keys' not in st.session_state:
    st.session_state['network_keys'] = None
if 'nodes' not in st.session_state:
    st.session_state['nodes'] = {}
if 'carriers' not in st.session_state:
    st.session_state['carriers'] = set()
if 'path_loaded' not in st.session_state:
    st.session_state['path_loaded'] = 0
if 'pages' not in st.session_state:
    st.session_state['pages'] = []


if st.sidebar.button('Reset Data'):
    st.session_state['path'] = {}
    st.session_state['nodes'] = {}
    st.session_state['carriers'] = set()
    st.session_state['path_loaded'] = 0
    st.session_state['pages'] = []

# Select Option
selected_option = st.sidebar.selectbox("Select an option", ['Show single result', 'Compare two Results'])

# Load Data
if st.session_state['path_loaded'] == 0:
    if selected_option == 'Show single result':
        nr_pages = [1]
        st.session_state['path'][1] = Path(
            st.sidebar.text_input("Enter path to results:", key="folder_key1") + '/optimization_results.h5')
        st.session_state['pages'] = ["Energy Balance at Node", "Technology Operation", "Technologies", "Networks"]

    else:
        nr_pages = [1, 2]
        st.session_state['path'][1] = Path(
            st.sidebar.text_input("Enter path to results 1:", key="folder_key1") + '/optimization_results.h5')
        st.session_state['path'][2] = Path(
            st.sidebar.text_input("Enter path to results 2:", key="folder_key2") + '/optimization_results.h5')
        st.session_state['pages'] = ["Summary", "Energy Balance at Node", "Technology Operation", "Technologies"]

    if st.sidebar.button('Load Data'):
        st.sidebar.markdown("---")

        # Get info from h5 files
        try:
            for i in st.session_state['path'].keys():
                with h5py.File(st.session_state['path'][i], 'r') as hdf_file:
                    st.session_state['nodes'][i] = extract_datasets_from_h5_dataset(hdf_file["topology/nodes"])
                    st.session_state['carriers'].update(extract_datasets_from_h5_dataset(hdf_file["topology/carriers"]))
            st.session_state['path_loaded'] = 1
        except FileNotFoundError:
            st.text('File not found, the path you entered is invalid!')
            sys.exit(1)

# If Path is loaded do this
if st.session_state['path_loaded'] == 1:

    selected_page = st.sidebar.selectbox("Select graph", st.session_state['pages'])

    st.sidebar.markdown("---")

    # Render side-bar
    if (selected_page == "Energy Balance at Node") or (selected_page == "Technology Operation"):

        # Select a node
        selected_node = {}
        selected_node[1] = st.sidebar.selectbox('Select a node:', st.session_state['nodes'][1], key="node_key1")
        if len(st.session_state['path']) == 2:
            selected_node[2] = st.sidebar.selectbox('Select a node for second result:', st.session_state['nodes'][2], key="node_key2")


    # Energybalance at Node
    if selected_page == "Energy Balance at Node":

        # Select a carrier
        selected_carrier = st.sidebar.selectbox('Select a carrier:', st.session_state['carriers'])

        # Read data
        energybalance = {}
        for i in st.session_state['path'].keys():
            with h5py.File(st.session_state['path'][i], 'r') as hdf_file:
                df = extract_datasets_from_h5_group(hdf_file["operation/energy_balance"])
                df = df.loc[:, (df.columns.get_level_values(0) == selected_node[i]) & (
                        df.columns.get_level_values(1) == selected_carrier)]
            energybalance[i] = df

        # Export to csv
        if selected_option == 'Show single result':
            export_csv(energybalance[1], 'Download Energybalance as CSV', 'energybalance.csv')

        # Determine graph boundaries
        x_min, x_max = determine_graph_boundaries(energybalance[1].index)

        st.title("Energy Balance per Node")

        st.header("Supply")
        # Multi-select box for filtering series
        series_supply = ['generic_production',
                         'technology_outputs',
                         'network_inflow',
                         'import']
        selected_supply_series = st.multiselect('Select Series to Filter', series_supply,
                                                default=series_supply)
        for i in st.session_state['path'].keys():
            plot_data = energybalance[i].loc[:, energybalance[i].columns.get_level_values(2).isin(selected_supply_series)]
            plot_data.columns = plot_data.columns.get_level_values(2)
            chart = plot_area_chart(plot_data, x_min, x_max)
            st.altair_chart(chart, theme="streamlit", use_container_width=True)

        st.header("Demand")
        # Multi-select box for filtering series
        series_demand = ['demand',
                         'technology_inputs',
                         'network_outflow',
                         'export']

        if 'network_consumption' in energybalance[1].columns.get_level_values(2):
            series_demand.append('network_consumption')

        selected_demand_series = st.multiselect('Select Series to Filter', series_demand,
                                                default=series_demand)
        for i in st.session_state['path'].keys():
            plot_data = energybalance[i].loc[:, energybalance[i].columns.get_level_values(2).isin(selected_demand_series)]
            plot_data.columns = plot_data.columns.get_level_values(2)
            chart = plot_area_chart(plot_data, x_min, x_max)
            st.altair_chart(chart, theme="streamlit", use_container_width=True)


    # Technology Operation
    elif selected_page == "Technology Operation":

        # Read data
        tec_operation = {}
        for i in st.session_state['path'].keys():
            with h5py.File(st.session_state['path'][i], 'r') as hdf_file:
                df = extract_datasets_from_h5_group(hdf_file["operation/technology_operation"])
                df = df.loc[:, (df.columns.get_level_values(0) == selected_node[i])]
            tec_operation[i] = df

        # Select a technology
        all_tecs = tec_operation[1].columns.get_level_values(1).unique()
        selected_tec = st.sidebar.selectbox('Select a technology:', all_tecs)

        for i in st.session_state['path'].keys():
            tec_operation[i] = tec_operation[i].loc[:, (tec_operation[i].columns.get_level_values(1) == selected_tec)]
            tec_operation[i].columns = tec_operation[i].columns.get_level_values(2)

        # Export to csv
        if selected_option == 'Show single result':
            export_csv(tec_operation[1], 'Download Technology Operation as CSV', 'technology_operation.csv')

        x_min, x_max = determine_graph_boundaries(tec_operation[1].index)

        # Input
        st.header("Input")
        variables_in = [col for col in tec_operation[1].columns if col.endswith('input')]
        for i in st.session_state['path'].keys():
            plot_data = tec_operation[i][variables_in]
            chart = plot_area_chart(plot_data, x_min, x_max)
            st.altair_chart(chart, theme="streamlit", use_container_width=True)

        # Output
        st.header("Output")
        variables_out = [col for col in tec_operation[1].columns if col.endswith('output')]
        for i in st.session_state['path'].keys():
            plot_data = tec_operation[i][variables_out]
            chart = plot_area_chart(plot_data, x_min, x_max)
            st.altair_chart(chart, theme="streamlit", use_container_width=True)

        st.header("Other Variables")
        variables_other = [x for x in tec_operation[1].columns if ((x not in variables_in) & (x not in variables_out))]
        selected_series = st.multiselect('Select Series to Filter', variables_other,
                                                default=variables_other)
        for i in st.session_state['path'].keys():
            plot_data = tec_operation[i][selected_series]
            chart = plot_line_chart(plot_data, x_min, x_max)
            st.altair_chart(chart, theme="streamlit", use_container_width=True)


    # Technology Sizes
    elif selected_page == "Technologies":
        technology_data = {}

        for i in st.session_state['path'].keys():
            with h5py.File(st.session_state['path'][i], 'r') as hdf_file:
                df = extract_datasets_from_h5_group(hdf_file["design/nodes"])
                df = pd.melt(df)
                df.columns = ['Node', 'Technology', 'Variable', 'Value']
            technology_data[i] = df

        # Select a variable
        all_vars = technology_data[1]['Variable'].unique()
        selected_var = st.sidebar.selectbox('Select a variable:', all_vars)

        # Export to csv
        if selected_option == 'Show single result':
            export_csv(df, 'Download Technology Design as CSV', 'technology_design.csv')

        technology_data_filtered = {}
        for i in st.session_state['path'].keys():
            technology_data_filtered[i] = technology_data[i][technology_data[i]['Variable'] == selected_var]


        # Multi-select box for filtering technologies
        technologies = technology_data[1]['Technology'].unique()
        selected_technologies = st.multiselect('Select Technologies to Filter', technologies,
                                               default=technologies)

        # Filter the DataFrame based on selected technologies
        for i in st.session_state['path'].keys():
            technology_data_filtered[i] = technology_data_filtered[i][technology_data_filtered[i]['Technology'].isin(selected_technologies)]

        st.header(selected_var)
        for i in st.session_state['path'].keys():
            chart = alt.Chart(technology_data_filtered[i]).mark_bar().encode(
                y='Node:N',
                x='Value:Q',
                color='Technology:N'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

    # Networks
    elif selected_page == "Networks":

        # Select graph to show
        graphs = ['Network Size', 'Network Capex', 'Total Flow', 'Flow at time slice']
        selected_graph = st.sidebar.selectbox('Select a figure to show:', graphs)

        # Load Nodes
        if st.session_state['network_keys'] is None:
            st.text('Enter a path to a csv file that maps the node names to longitude and latitude.')
            network_key_path = Path(st.text_input("Enter file path to location keys of nodes:", key="network"))
            if st.button("Load Node Locations"):
                st.session_state['network_keys'] = pd.read_csv(network_key_path, sep=';', index_col=0)

        if st.session_state['network_keys'] is not None:
            # Load network design data
            with h5py.File(st.session_state['path'][1], 'r') as hdf_file:
                network_design = extract_datasets_from_h5_group(hdf_file["design/networks"])

            network_design = network_design.melt()
            network_design.columns = ['Network', 'Arc_ID', 'Variable', 'Value']
            network_design = network_design.pivot(columns='Variable', index=['Arc_ID', 'Network'], values='Value')
            network_design['FromNode'] = network_design['fromNode'].str.decode('utf-8')
            network_design['ToNode'] = network_design['toNode'].str.decode('utf-8')
            network_design.drop(columns=['fromNode', 'toNode', 'network'], inplace=True)
            network_design = network_design.reset_index()

            # Select a network
            networks_available = list(network_design['Network'].unique())
            selected_netw = st.sidebar.multiselect('Select a network:', networks_available)
            network_df_filtered = network_design[network_design['Network'].isin(selected_netw)]

            arc_ids = network_df_filtered[['Arc_ID', 'FromNode', 'ToNode']]
            network_df_filtered = network_df_filtered.groupby('Arc_ID').sum()
            network_df_filtered.drop(columns=['FromNode', 'ToNode', 'Network'], inplace=True)
            network_df_filtered = network_df_filtered.merge(arc_ids, on='Arc_ID')

            # Init map
            node_data = st.session_state['network_keys']
            map_center = [node_data['lat'].mean(), node_data['lon'].mean()]
            map = folium.Map(location=map_center, zoom_start=5)

            # Plot nodes
            plot_nodes(map, node_data)

            # Plot edges
            if selected_graph in ['Network Size', 'Network Capex', 'Total Flow']:
                variables = {'Network Size': 'size', 'Network Capex': 'capex', 'Total Flow': 'total_flow'}

                # Determine color scale:
                max_value = max(network_df_filtered[variables[selected_graph]])
                color_scale = linear.OrRd_09.scale(0, 1)


                for _, edge_data in network_df_filtered.iterrows():
                    from_node_data = node_data.loc[edge_data.FromNode]
                    to_node_data = node_data.loc[edge_data.ToNode]

                    # Normalize edge value to be within [0, 1]
                    normalized_value = (edge_data[variables[selected_graph]]) / max_value

                    # Determine color based on the color scale
                    color = color_scale(normalized_value)
                    if normalized_value > 0.001:
                        line = folium.plugins.PolyLineOffset([(from_node_data['lat'], from_node_data['lon']),
                                                (to_node_data['lat'], to_node_data['lon'])],
                                               color=color,
                                               weight=3.5,  # Set a default weight
                                               opacity=1,
                                               offset=3).add_to(map)
                        attr = {"font-weight": "bold", "font-size": "13"}

                        folium.plugins.PolyLineTextPath(
                            line, "      >", repeat=True, offset=5, attributes=attr
                        ).add_to(map)


            elif selected_graph == 'Flow at time slice':
                # Load required network data
                with h5py.File(st.session_state['path'][1], 'r') as hdf_file:
                    network_operation = extract_datasets_from_h5_group(hdf_file["operation/networks"])

                network_operation = network_operation.melt(value_vars=network_operation.columns.tolist(), ignore_index=False)
                network_operation = network_operation.reset_index()
                network_operation.columns = ['Timeslice', 'Network', 'Arc_ID', 'Variable', 'Value']
                network_operation = network_operation[network_operation['Variable'] == 'flow']
                network_operation = network_operation.merge(arc_ids, on='Arc_ID')

                # Select a network
                network_operation_filtered = network_operation[network_operation['Network'].isin(selected_netw)]

                # Select type
                select_plotting = st.sidebar.selectbox('Select plotting type:', ['Energy Flow', 'Line Utilization'])

                # Choose timeslice to plot;
                t = st.slider(
                    "Select a timeslice: ",
                    min_value=min(network_operation_filtered['Timeslice']),
                    max_value=max(network_operation_filtered['Timeslice']),
                )

                network_operation_filtered = network_operation_filtered[network_operation_filtered['Timeslice'] == t]
                network_operation_filtered = network_operation_filtered.groupby(['Timeslice', 'Arc_ID', 'FromNode', 'ToNode']).sum().reset_index()

                # Determine color scale:
                max_value = max(network_operation_filtered['Value'])
                color_scale = linear.OrRd_09.scale(0, 1)

                for _, edge_data in network_operation_filtered.iterrows():
                    from_node_data = node_data.loc[edge_data.FromNode]
                    to_node_data = node_data.loc[edge_data.ToNode]

                    # Normalize edge value to be within [0, 1]
                    flow_this_direction = edge_data['Value']

                    flow_other_direction = network_operation_filtered[(network_operation_filtered['FromNode'] == to_node_data.name) &
                                                                        (network_operation_filtered['ToNode'] == from_node_data.name)]

                    uni_flow = flow_this_direction - flow_other_direction.loc[:, 'Value'].values[0]

                    if uni_flow > 0.1:
                        if select_plotting == 'Energy Flow':
                            normalized_value = uni_flow / max_value
                        else:
                            normalized_value = uni_flow / network_df_filtered[(network_df_filtered['FromNode'] == to_node_data.name) &
                                                                        (network_df_filtered['ToNode'] == from_node_data.name)].loc[:, 'size'].values[0]

                        # # Determine color based on the color scale
                        color = color_scale(normalized_value)
                        folium.plugins.AntPath(
                            locations=[(from_node_data['lat'], from_node_data['lon']),
                                                (to_node_data['lat'], to_node_data['lon'])],
                            color=color, dash_array=[10, 20], weight= normalized_value * 10,
                        ).add_to(map)



            st_folium(map, width=725)


