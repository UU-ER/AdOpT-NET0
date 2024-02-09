import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.cm as cm
from streamlit_folium import st_folium
from folium.features import DivIcon

import folium
from folium import plugins


def plot_nodes(map, node_data):
    for node_name, data in node_data.iterrows():
        folium.CircleMarker(
            location=[data['lat'], data['lon']],
            radius=5,  # Adjust the radius as needed
            color='black',  # Marker color
            fill=True,
            fill_color='black',  # Fill color
            fill_opacity=0.7,
        ).add_to(map)
        folium.map.Marker(
            [data['lat'], data['lon']],
            icon=DivIcon(icon_size=(150, 36),
                         icon_anchor=(-5, 20),
                         html='<div style="font-size: 9pt">' + node_name + '</div>')
        ).add_to(map)

def plot_edges(map, node_data, network_size_data, edge_weight='size'):
    for _, edge_data in network_size_data.iterrows():
        from_node_data = node_data.loc[edge_data['fromNode']]
        to_node_data = node_data.loc[edge_data['toNode']]

        if edge_weight == 'size':
            e_weight = edge_data['Size'] / 1000
        else:
            e_weight = edge_weight

        folium.PolyLine([(from_node_data['lat'], from_node_data['lon']),
                         (to_node_data['lat'], to_node_data['lon'])],
                        color='black',  # Set a default color
                        weight=e_weight,  # Set edge size based on 'Size' column
                        opacity=0.5).add_to(map)



def network_sizes(network_size_data, node_data):
    networks_available = ['All']
    networks_available.extend(list(network_size_data['Network'].unique()))
    selected_netw = st.sidebar.selectbox('Select a network:', networks_available)

    map_center = [node_data['lat'].mean(), node_data['lon'].mean()]
    map = folium.Map(location=map_center, zoom_start=5)

    plot_nodes(map, node_data)

    if selected_netw == 'All':
        for netw in network_size_data['Network'].unique():
            size_data = network_size_data[network_size_data['Network'] == netw]
            plot_edges(map, node_data, size_data, edge_weight=2)
    else:
        size_data = network_size_data[network_size_data['Network'] == selected_netw]
        plot_edges(map, node_data, size_data)

    # Plot edges on the map with color and size
    st_folium(map, width=725)
    # st_folium(map, width=0, height=0)

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

