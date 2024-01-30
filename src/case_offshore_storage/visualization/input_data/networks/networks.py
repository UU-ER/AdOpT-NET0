from pathlib import Path
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.cm as cm
from streamlit_folium import st_folium
from folium.features import DivIcon
import folium
from folium import plugins

from ..node_definition.input_data_node_definition import plot_nodes_centroids

def show_networks():
    root_load_path = Path('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/reporting/spatial_data')
    layers = {}
    layers['nodes_ours_centroids'] = gpd.read_file(Path.joinpath(root_load_path, Path('NodesPyHub_centroids.geojson')))

    # Create Map
    map_center = [layers['nodes_ours_centroids']['y'].mean(), layers['nodes_ours_centroids']['x'].mean()]
    map = folium.Map(location=map_center, zoom_start=5)
    plot_nodes_centroids(map, layers['nodes_ours_centroids'])

    # Create Edges
    root_load_path = Path('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/clean_data/networks')
    grid_ac = pd.read_csv(Path.joinpath(root_load_path, Path('pyhub_el_ac.csv')), delimiter=';')
    grid_dc = pd.read_csv(Path.joinpath(root_load_path, Path('pyhub_el_dc.csv')), delimiter=';')

    grid_ac['type'] = 'ac'
    grid_dc['type'] = 'dc'

    grid = pd.concat([grid_ac, grid_dc])

    grids_to_show = st.multiselect('Select grids to show', ['ac', 'dc'])
    type = st.selectbox('Select variable to show', ['starting grid', 'allowable expansion'])
    if type == 'starting grid': variable = 's_nom'
    else: variable = 's_nom_max'

    if grids_to_show:
        grid = grid[grid['type'].isin(grids_to_show)]

        grid[['node0', 'node1']] = grid.apply(lambda row: sorted([row['node0'], row['node1']]), axis=1, result_type='expand')
        grid = grid.set_index(['node0', 'node1'])
        grid = grid.groupby(['node0', 'node1']).sum().reset_index()

        node_data = layers['nodes_ours_centroids'].set_index(['NODE_NAME'])
        for _, edge_data in grid.iterrows():
            from_node_data = node_data.loc[edge_data['node0']]
            to_node_data = node_data.loc[edge_data['node1']]

            e_weight = edge_data[variable]/2

            folium.PolyLine([(from_node_data['y'], from_node_data['x']),
                             (to_node_data['y'], to_node_data['x'])],
                            color='black',  # Set a default color
                            weight=e_weight,  # Set edge size based on 'Size' column
                            opacity=0.9).add_to(map)


    st_folium(map, width=725)

#
#
#
# def plot_nodes(map, node_data):
#     for node_name, data in node_data.iterrows():
#         folium.CircleMarker(
#             location=[data['lat'], data['lon']],
#             radius=5,  # Adjust the radius as needed
#             color='black',  # Marker color
#             fill=True,
#             fill_color='black',  # Fill color
#             fill_opacity=0.7,
#         ).add_to(map)
#         folium.map.Marker(
#             [data['lat'], data['lon']],
#             icon=DivIcon(icon_size=(150, 36),
#                          icon_anchor=(-5, 20),
#                          html='<div style="font-size: 9pt">' + node_name + '</div>')
#         ).add_to(map)
#
# def plot_edges(map, node_data, network_size_data, edge_weight='size'):

#
#
# def network_sizes(network_size_data, node_data):
#     networks_available = ['All']
#     networks_available.extend(list(network_size_data['Network'].unique()))
#     selected_netw = st.sidebar.selectbox('Select a network:', networks_available)
#
#     map_center = [node_data['lat'].mean(), node_data['lon'].mean()]
#     map = folium.Map(location=map_center, zoom_start=5)
#
#     plot_nodes(map, node_data)
#
#     if selected_netw == 'All':
#         for netw in network_size_data['Network'].unique():
#             size_data = network_size_data[network_size_data['Network'] == netw]
#             plot_edges(map, node_data, size_data, edge_weight=2)
#     else:
#         size_data = network_size_data[network_size_data['Network'] == selected_netw]
#         plot_edges(map, node_data, size_data)
#
#     # Plot edges on the map with color and size
#     st_folium(map, width=725)
#     # st_folium(map, width=0, height=0)

    #
    #
    #
    # G = nx.from_pandas_edgelist(network_data, 'fromNode', 'toNode', edge_attr = ['Size'])
    #
    # # Extract edge sizes
    # edge_sizes = [size/1000 for _, _, size in G.edges.data('Size')]
    # # Normalize edge sizes for color mapping
    # norm = plt.Normalize(min(edge_sizes), max(edge_sizes))
    # colormap = cm.plasma_r
    # edge_colors = colormap(norm(edge_sizes))
    #
    # # Draw the graph with edge thickness based on 'Size' parameter
    # fig, ax = plt.subplots(figsize=(8, 8))
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels=True, node_size=20, node_color="blue", font_size=10, font_color="black",
    #         edgelist=G.edges, edge_color=edge_colors, width=2, cmap=plt.cm.Blues, ax=ax)
    #
    # plt.axis('off')
    # cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
    # cbar.set_label('Size (GW)')
    #
    # # Display the graph in the Streamlit app
    # st.pyplot(fig)

