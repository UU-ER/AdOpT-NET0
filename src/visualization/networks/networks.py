import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.cm as cm

def network_sizes(network_size_data):
    selected_netw = st.sidebar.selectbox('Select a network:', network_size_data['Network'].unique())

    network_data = network_size_data[network_size_data['Network'] == selected_netw]

    G = nx.from_pandas_edgelist(network_data, 'fromNode', 'toNode', edge_attr = ['Size'])

    # Extract edge sizes
    edge_sizes = [size/1000 for _, _, size in G.edges.data('Size')]
    # Normalize edge sizes for color mapping
    norm = plt.Normalize(min(edge_sizes), max(edge_sizes))
    colormap = cm.plasma_r
    edge_colors = colormap(norm(edge_sizes))

    # Draw the graph with edge thickness based on 'Size' parameter
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=20, node_color="blue", font_size=10, font_color="black",
            edgelist=G.edges, edge_color=edge_colors, width=2, cmap=plt.cm.Blues, ax=ax)

    plt.axis('off')
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
    cbar.set_label('Size (GW)')

    # Display the graph in the Streamlit app
    st.pyplot(fig)

