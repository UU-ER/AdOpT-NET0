import streamlit as st
from input_data import *

from utilities import determine_graph_boundaries, select_node, read_time_series, plot_area_chart

main_pages = ['Show input data']
sub_pages = {
    'Show input data': [
        'Installed Capacities',
        'Annual renewable generation and demand',
        'Node Definition',
        'Renewable generation profiles and demand',
        'Networks'
        ]
    }


# [
#         'Installed Capacities',
#         'Demand (national)',
#         'Demand (per node)',
#         'Time series',
#         'Networks'
#     ]
main_page = st.sidebar.selectbox("Select an option:", main_pages)
sub_page = st.sidebar.selectbox("Select data to show:", sub_pages[main_page])

if main_page == 'Show input data':
        show_page_input_data(sub_page)


#
# # Define paths to results
# path = {}
# if selected_option == 'Compare two Results':
#     nr_pages = [1, 2]
#     path[1] = Path(st.sidebar.text_input("Enter folder path to results 1:", key="folder_key1"))
#     path[2] = Path(st.sidebar.text_input("Enter folder path to results 2:", key="folder_key2"))
#     pages = ["Summary", "Energy Balance at Node", "Technology Operation", "Technologies"]
# else:
#     nr_pages = [1]
#     path[1] = Path(st.sidebar.text_input("Enter folder path to results:", key="folder_key1"))
#     pages = ["Energy Balance at Node", "Technology Operation", "Technologies", "Networks"]
#
# # Select what to show
# selected_page = st.sidebar.selectbox("Select graph", pages)
#
# # Energybalance at Node
# if selected_page == "Energy Balance at Node":
#     selected_node, node_path = select_node(path, nr_pages)
#     energybalance_data = {}
#     for i in nr_pages:
#         energybalance_data[i] = read_time_series(Path.joinpath(node_path[i], selected_node[i], 'Energybalance.xlsx'))
#
#     selected_carrier = st.selectbox('Select a carrier:', energybalance_data[1].keys())
#
#     x_min, x_max = determine_graph_boundaries(energybalance_data)
#     st.title("Energy Balance per Node")
#
#     st.header("Supply")
#     # Multi-select box for filtering series
#     series_supply = ['Timestep',
#                 'Generic_production',
#                 'Technology_outputs',
#                 'Network_inflow',
#                 'Import']
#     selected_supply_series = st.multiselect('Select Series to Filter', series_supply,
#                                            default=series_supply)
#     for i in nr_pages:
#         plot_data = energybalance_data[i][selected_carrier][selected_supply_series]
#         chart = plot_area_chart(plot_data, x_min, x_max)
#         st.altair_chart(chart, theme="streamlit", use_container_width=True)
#
#     st.header("Demand")
#     # Multi-select box for filtering technologies
#     series_demand = ['Timestep',
#                         'Demand',
#                         'Technology_inputs',
#                         'Network_outflow',
#                         'Export']
#     selected_demand_series = st.multiselect('Select Series to Filter', series_demand,
#                                            default=series_demand)
#     for i in nr_pages:
#         plot_data = energybalance_data[i][selected_carrier][selected_demand_series]
#         chart = plot_area_chart(plot_data, x_min, x_max)
#         st.altair_chart(chart, theme="streamlit", use_container_width=True)
#
#
# # Technology Operation
# elif selected_page == "Technology Operation":
#     selected_node, node_path = select_node(path, nr_pages)
#     tec_operation_data = {}
#     try:
#
#         for i in nr_pages:
#                 tec_operation_data[i] = read_time_series(Path.joinpath(node_path[i], selected_node[i], 'TechnologyOperation.xlsx'))
#
#         x_min, x_max = determine_graph_boundaries(tec_operation_data)
#
#         st.title("Technology Operation")
#         tec = st.selectbox('Select a technology:', tec_operation_data[1].keys())
#
#
#         # Input
#         st.header("Input")
#         for i in nr_pages:
#             tec_data = tec_operation_data[i][tec]
#             variables = [col for col in tec_data.columns if col.startswith('input')]
#             variables.append('Timestep')
#             plot_data = tec_data[variables]
#             chart = plot_area_chart(plot_data, x_min, x_max)
#             st.altair_chart(chart, theme="streamlit", use_container_width=True)
#
#         # Output
#         st.header("Output")
#         for i in nr_pages:
#             tec_data = tec_operation_data[i][tec]
#             variables = [col for col in tec_data.columns if col.startswith('output')]
#             variables.append('Timestep')
#             plot_data = tec_data[variables]
#             chart = plot_area_chart(plot_data, x_min, x_max)
#             st.altair_chart(chart, theme="streamlit", use_container_width=True)
#
#         st.header("Other Variables")
#         for i in nr_pages:
#             tec_data = tec_operation_data[i][tec]
#             variables = [col for col in tec_data.columns if col.startswith('storagelevel')]
#             if len(variables) >= 1:
#                 variables.append('Timestep')
#                 plot_data = tec_data[variables]
#                 chart = plot_area_chart(plot_data, x_min, x_max)
#                 st.altair_chart(chart, theme="streamlit", use_container_width=True)
#
#     except FileNotFoundError:
#         st.text("There are no technologies to show for this node.")
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












# if selected_option == 'Show single result':
#     # Sidebar navigation
#     path = st.sidebar.text_input("Enter folder path to results:", key="folder_key_single")
#
#     if not path == '':
#         # Read data from
#         path = Path(path)
#         node_path = Path.joinpath(path, 'Nodes')
#         nodes = [f.name for f in os.scandir(node_path) if f.is_dir()]
#
#         page_options = ["Energy Balance at Node", "Technology Operation", "Technologies", "Networks", "Metrics for Offshore Storage Study"]
#         selected_page = st.sidebar.selectbox("Select graph", page_options)
#
#         if selected_page in ["Energy Balance at Node", "Technology Operation"]:
#             selected_node = st.sidebar.selectbox('Select a node:', nodes)
#
#         # Render the selected page
#         if selected_page == "Energy Balance at Node":
#             energybalance_data = pd.read_excel(Path.joinpath(node_path, selected_node, 'Energybalance.xlsx'), sheet_name=None,
#                                      index_col=0)
#             for carrier in energybalance_data:
#                 energybalance_data[carrier]['Timestep'] = [datetime(2030, 1, 1, 0, 0, 0) + timedelta(hours=hour) for hour in energybalance_data[carrier].index]
#             x_min, x_max = determine_graph_boundaries(energybalance_data)
#             energybalance(energybalance_data, x_min, x_max)
#
#         elif selected_page == "Technology Operation":
#             tec_operation_data = pd.read_excel(Path.joinpath(node_path, selected_node, 'TechnologyOperation.xlsx'), sheet_name=None,
#                                                index_col=0)
#             for tec in tec_operation_data:
#                 tec_operation_data[tec]['Timestep'] = [datetime(2030, 1, 1, 0, 0, 0) + timedelta(hours=hour) for hour in tec_operation_data[tec].index]
#             x_min, x_max = determine_graph_boundaries(tec_operation_data)
#             tec_operation(tec_operation_data, x_min, x_max)
#
#         elif selected_page == "Technologies":
#             tec_size_data = pd.read_excel(Path.joinpath(path, 'Summary.xlsx'), sheet_name='TechnologySizes',
#                                                index_col=0)
#             tec_sizes(tec_size_data)
#
#
#         elif selected_page == "Networks":
#             network_size_data = pd.read_excel(Path.joinpath(path, 'Summary.xlsx'), sheet_name='Networks',
#                                                index_col=0)
#             network_sizes(network_size_data)
#
#         elif selected_page == "Metrics for Offshore Storage Study":
#             summary_data = pd.read_excel(Path.joinpath(path, 'Summary.xlsx'), sheet_name='Summary',
#                                                index_col=0)
#             st.text('Emissions: ' + str(round(summary_data['Net_Emissions'].values[0]/1000, 2)) + ' kt')
#
# elif selected_option == 'Compare Results':