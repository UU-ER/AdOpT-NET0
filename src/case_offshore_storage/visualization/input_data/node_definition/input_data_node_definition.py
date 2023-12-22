import streamlit as st
import altair as alt
from pathlib import Path
import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
from folium.features import DivIcon
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import folium

def plot_nodes_centroids(map, node_data):
    for node_name, data in node_data.iterrows():
        folium.CircleMarker(
            location=[data['y'], data['x']],
            radius=5,  # Adjust the radius as needed
            color='black',  # Marker color
            fill=True,
            fill_color='black',  # Fill color
            fill_opacity=0.7,
        ).add_to(map)
        folium.map.Marker(
            [data['y'], data['x']],
            icon=DivIcon(icon_size=(150, 36),
                         icon_anchor=(-5, 20),
                         html='<div style="font-size: 9pt">' + data['NODE_NAME'] + '</div>')
        ).add_to(map)

def plot_nodes_polygons(map, node_data, color_map, unique_nodes):
    for node in unique_nodes:
        plot_data = node_data[node_data.NODE_NAME == node]
        folium.GeoJson(plot_data,
                       style_function=lambda feature, color=color_map['Color'][node]: {
                           'fillColor': color,
                           'color': 'black',
                           'weight': 0.2,
                           'fillOpacity': 0.7
                       }).add_to(map)

def show_node_definition():

    root_load_path = Path('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/reporting/spatial_data')
    layers = {}
    layers['nodes_ours_centroids'] = gpd.read_file(Path.joinpath(root_load_path, Path('NodesPyHub_centroids.geojson')))
    layers['nodes_ours_polygons'] = gpd.read_file(Path.joinpath(root_load_path, Path('NodesPyHub_polygons.geojson')))
    layers['nodes_pypsa'] = gpd.read_file(Path.joinpath(root_load_path, Path('Nodes_PyPsa.geojson')))
    layers['nuts2'] = gpd.read_file(Path.joinpath(root_load_path, Path('NUTS2.geojson')))
    layers['wind_farms'] = gpd.read_file(Path.joinpath(root_load_path, Path('WindFarms.geojson')))
    layers['wind_farms'] = layers['wind_farms'].rename(columns= {'NODE_2': 'NODE_NAME'})
    layers['wind_farms'] = layers['wind_farms'][layers['wind_farms']['NODE_NAME'].notna()]

    layer = st.multiselect('Select a layers', ['Nodes this work', 'Wind Farms Offshore', 'NUTS2', 'PyPSA Nodes'])

    unique_nodes = layers['nodes_ours_centroids']['NODE_NAME'].unique()
    colormap = cm.get_cmap('hsv', len(unique_nodes))
    node_color_mapping = pd.DataFrame({'NODE_NAME': unique_nodes, 'Color': [mcolors.to_hex(colormap(i)) for i in range(len(unique_nodes))]})

    layers['nodes_ours_polygons'] = layers['nodes_ours_polygons'].merge(node_color_mapping, right_on='NODE_NAME', left_on='NODE_NAME')
    layers['nodes_ours_centroids'] = layers['nodes_ours_centroids'].merge(node_color_mapping, right_on='NODE_NAME', left_on='NODE_NAME')
    layers['wind_farms'] = layers['wind_farms'].merge(node_color_mapping, right_on='NODE_NAME', left_on='NODE_NAME')

    node_color_mapping = node_color_mapping.set_index('NODE_NAME').to_dict()

    # Create Map
    map_center = [layers['nodes_ours_centroids']['y'].mean(), layers['nodes_ours_centroids']['x'].mean()]
    map = folium.Map(location=map_center, zoom_start=4)

    if 'Wind Farms Offshore' in layer:
        plot_nodes_polygons(map, layers['wind_farms'], node_color_mapping, unique_nodes)
    if 'Nodes this work' in layer:
        plot_nodes_polygons(map, layers['nodes_ours_polygons'], node_color_mapping, unique_nodes)
        plot_nodes_centroids(map, layers['nodes_ours_centroids'])
    if 'PyPSA Nodes' in layer:
        folium.GeoJson(layers['nodes_pypsa'],
            style_function=lambda feature: {
                'fillColor': None,
                'color': 'blue',       # Change this to the desired border color
                'weight': 1,
                'fillOpacity': 0
            }).add_to(map)
    if 'NUTS2' in layer:
        folium.GeoJson(layers['nuts2'],
            style_function=lambda feature: {
                'fillColor': None,
                'color': 'black',       # Change this to the desired border color
                'weight': 1,
                'fillOpacity': 0
            }).add_to(map)

    st_folium(map, width=725)



