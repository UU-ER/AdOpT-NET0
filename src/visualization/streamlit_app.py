import streamlit as st
from pathlib import Path
import h5py

from utilities import *
from energybalance import *

# Initialize Session States
if 'path' not in st.session_state:
    st.session_state['path'] = {}
if 'nodes' not in st.session_state:
    st.session_state['nodes'] = {}
if 'carriers' not in st.session_state:
    st.session_state['carriers'] = set()

# Select option
selected_option = st.sidebar.selectbox("Select an option", ['Show single result', 'Compare two Results'])

if selected_option == 'Show single result':
    nr_pages = [1]
    st.session_state['path'][1] = Path(st.sidebar.text_input("Enter folder path to results:", key="folder_key1"))
    pages = ["Energy Balance at Node", "Technology Operation", "Technologies", "Networks"]

else:
    nr_pages = [1, 2]
    st.session_state['path'][1] = Path(st.sidebar.text_input("Enter folder path to results 1:", key="folder_key1"))
    st.session_state['path'][2] = Path(st.sidebar.text_input("Enter folder path to results 2:", key="folder_key2"))
    pages = ["Summary", "Energy Balance at Node", "Technology Operation", "Technologies"]

# Select graph
st.sidebar.markdown("---")

selected_page = st.sidebar.selectbox("Select graph", pages)

# Get info from h5 files
for i in st.session_state['path'].keys():
    with h5py.File(st.session_state['path'][i], 'r') as hdf_file:
        st.session_state['nodes'][i] = extract_datasets_from_h5_dataset(hdf_file["topology/nodes"])
        st.session_state['carriers'].update(extract_datasets_from_h5_dataset(hdf_file["topology/carriers"]))

st.sidebar.markdown("---")

# Render side bar
if (selected_page == "Energy Balance at Node") or (selected_page == "Technology Operation"):

    # Select a node
    selected_node = {}
    selected_node[1] = st.sidebar.selectbox('Select a node:', st.session_state['nodes'][1], key="node_key1")
    if len(nr_pages) == 2:
        selected_node[2] = st.sidebar.selectbox('Select a node for second result:', st.session_state['nodes'][2], key="node_key2")


# Energybalance at Node
if selected_page == "Energy Balance at Node":

    plot_energybalance()


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
    for i in st.session_state['path'].keys():
        plot_data = tec_operation[i][variables_other]
        chart = plot_line_chart(plot_data, x_min, x_max)
        st.altair_chart(chart, theme="streamlit", use_container_width=True)

#
# # Technology Sizes
# elif selected_page == "Technologies":
#     tec_size_data = {}
#     technologies = []
#
#     for i in nr_pages:
#         tec_size_data[i] = pd.read_excel(Path.joinpath(path[i], 'Summary.xlsx'), sheet_name='TechnologySizes',
#                                       index_col=0)
#         tec_size_data[i]['total_cost'] = tec_size_data[i]['capex'] + \
#                                          tec_size_data[i]['opex_variable'] + \
#                                          tec_size_data[i]['opex_fixed']
#         technologies.append(list(tec_size_data[i]['technology'].unique()))
#
#     technologies = list(set([item for sublist in technologies for item in sublist]))
#
#     # Multi-select box for filtering technologies
#     selected_technologies = st.multiselect('Select Technologies to Filter', technologies,
#                                            default=technologies)
#
#     # Filter the DataFrame based on selected technologies
#     filtered_df = {}
#     for i in nr_pages:
#         filtered_df[i] = tec_size_data[i][tec_size_data[i]['technology'].isin(selected_technologies)]
#
#     st.header("Technology Size")
#     for i in nr_pages:
#         chart = alt.Chart(filtered_df[i]).mark_bar().encode(
#             x='node:N',
#             y='sum(size):Q',
#             color='technology:N',
#             tooltip=['node', 'technology', 'size']
#         ).interactive()
#         st.altair_chart(chart, use_container_width=True)
#
#
#     st.header("Technology Cost")
#     for i in nr_pages:
#         chart = alt.Chart(filtered_df[i]).mark_bar().encode(
#             x='node:N',
#             y='sum(total_cost):Q',
#             color='technology:N',
#             tooltip=['node', 'technology', 'total_cost']
#         ).interactive()
#         st.altair_chart(chart, use_container_width=True)
#
# # Networks
# elif selected_page == "Networks":
#     network_size_data = {}
#     node_data = {}
#     for i in nr_pages:
#         network_size_data[i] = pd.read_excel(Path.joinpath(path[i], 'Summary.xlsx'), sheet_name='Networks',
#                                           index_col=0)
#         node_data[i] = pd.read_excel(Path.joinpath(path[i], 'Summary.xlsx'), sheet_name='Nodes',
#                                           index_col=0)
#         node_data[i] = node_data[i].T
#
#     network_sizes(network_size_data[1], node_data[1])
#
# # Summary Comparison
# elif selected_page == "Summary":
#     summary_data = pd.DataFrame()
#
#     for i in nr_pages:
#         summary_data = pd.concat([summary_data, pd.read_excel(Path.joinpath(path[i], 'Summary.xlsx'), sheet_name='Summary',
#                                  index_col=0)])
#
#     summary_data['Case'] = ['Case ' + str(i) for i in nr_pages]
#
#     st.header("Total Cost")
#     chart = alt.Chart(summary_data.reset_index()).mark_bar().encode(
#         x='Case',
#         y='Total_Cost:Q',
#     ).interactive()
#     st.altair_chart(chart, use_container_width=True)
#
#     st.header("Total Emissions")
#     chart = alt.Chart(summary_data.reset_index()).mark_bar().encode(
#         x='Case',
#         y='Net_Emissions:Q',
#     ).interactive()
#     st.altair_chart(chart, use_container_width=True)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # if selected_option == 'Show single result':
# #     # Sidebar navigation
# #     path = st.sidebar.text_input("Enter folder path to results:", key="folder_key_single")
# #
# #     if not path == '':
# #         # Read data from
# #         path = Path(path)
# #         node_path = Path.joinpath(path, 'Nodes')
# #         nodes = [f.name for f in os.scandir(node_path) if f.is_dir()]
# #
# #         page_options = ["Energy Balance at Node", "Technology Operation", "Technologies", "Networks", "Metrics for Offshore Storage Study"]
# #         selected_page = st.sidebar.selectbox("Select graph", page_options)
# #
# #         if selected_page in ["Energy Balance at Node", "Technology Operation"]:
# #             selected_node = st.sidebar.selectbox('Select a node:', nodes)
# #
# #         # Render the selected page
# #         if selected_page == "Energy Balance at Node":
# #             energybalance_data = pd.read_excel(Path.joinpath(node_path, selected_node, 'Energybalance.xlsx'), sheet_name=None,
# #                                      index_col=0)
# #             for carrier in energybalance_data:
# #                 energybalance_data[carrier]['Timestep'] = [datetime(2030, 1, 1, 0, 0, 0) + timedelta(hours=hour) for hour in energybalance_data[carrier].index]
# #             x_min, x_max = determine_graph_boundaries(energybalance_data)
# #             energybalance(energybalance_data, x_min, x_max)
# #
# #         elif selected_page == "Technology Operation":
# #             tec_operation_data = pd.read_excel(Path.joinpath(node_path, selected_node, 'TechnologyOperation.xlsx'), sheet_name=None,
# #                                                index_col=0)
# #             for tec in tec_operation_data:
# #                 tec_operation_data[tec]['Timestep'] = [datetime(2030, 1, 1, 0, 0, 0) + timedelta(hours=hour) for hour in tec_operation_data[tec].index]
# #             x_min, x_max = determine_graph_boundaries(tec_operation_data)
# #             tec_operation(tec_operation_data, x_min, x_max)
# #
# #         elif selected_page == "Technologies":
# #             tec_size_data = pd.read_excel(Path.joinpath(path, 'Summary.xlsx'), sheet_name='TechnologySizes',
# #                                                index_col=0)
# #             tec_sizes(tec_size_data)
# #
# #
# #         elif selected_page == "Networks":
# #             network_size_data = pd.read_excel(Path.joinpath(path, 'Summary.xlsx'), sheet_name='Networks',
# #                                                index_col=0)
# #             network_sizes(network_size_data)
# #
# #         elif selected_page == "Metrics for Offshore Storage Study":
# #             summary_data = pd.read_excel(Path.joinpath(path, 'Summary.xlsx'), sheet_name='Summary',
# #                                                index_col=0)
# #             st.text('Emissions: ' + str(round(summary_data['Net_Emissions'].values[0]/1000, 2)) + ' kt')
# #
# # elif selected_option == 'Compare Results':