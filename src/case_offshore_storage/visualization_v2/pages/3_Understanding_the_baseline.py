import pandas as pd
from utilities import *
from read_data import *
import streamlit as st
import h5py
import time
import folium
from streamlit_folium import st_folium, folium_static
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from matplotlib.colors import to_hex
import time
from matplotlib.animation import FuncAnimation
import streamlit.components.v1 as components
import networkx as nx

def frame_around_fig(ax, show_frame=True):
    ax.set_frame_on(show_frame)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_node_curtailment(ax, curtailment, show_text, fs):
    ax.barh(1, curtailment['generic_production'] / curtailment.sum(), color='orange')
    ax.barh(1, curtailment['Curtailment'] / curtailment.sum(),
            left=curtailment['generic_production'] / curtailment.sum(), color='moccasin')

    if show_text:
        ax.text(0.05, 1,
                f"{curtailment['generic_production'] / curtailment.sum() * 100:.1f}%",
                ha='left', va='center', color='black', fontsize=fs)

    ax.set_xlim([0, 1])
    ax.axis('off')
    return ax

def plot_node_demand(ax, demand, show_text, fs):
    ax.bar(0, demand, color='lightblue')
    ax.set_ylim(0, max_demand*1.2)

    if show_text:
        value = demand[0]
        ax.text(0, max_demand * 0.05, f"{value/1000000:.0f} TWh", ha='center', va='bottom', fontsize=fs, rotation=90)
    return ax

def plot_node_emissions(ax, emissions, show_text, fs):
    ax.bar(0, emissions, color='red')
    ax.set_ylim(0, max_emissions*1.2)

    if show_text:
        value = emissions[0]
        ax.text(0, max_emissions * 0.05, f"{value/1000000:.1f} Mt", ha='center', va='bottom', fontsize=fs, rotation=90)
    return ax

def plot_node_supply(ax, supply, show_text, radius_scale, fs):
    ax.pie(supply, startangle=90, radius=radius_scale,
                     colors=[color for _, color in item_color_dict.items()])
    if show_text:
        value = supply.sum()
        ax.text(0, 0, f"{value/1000000:.0f} TWh", ha='center', va='center', fontsize=fs)
    return ax

def aggregate_time(df, level):
    df = df.groupby(level=level).sum()
    df.index.names = ['Timeslice']
    return df

st.set_page_config(layout="wide")

# LOAD DATA
# All paths
root = 'src/case_offshore_storage/visualization_v2/'
h5_path = root + 'data/cases/results_baseline.h5'
re_gen_path = root + 'data/production_profiles_re.csv'
node_locations_path = root + 'data/Node_Locations.csv'
country_locations_path = root + 'data/Country_Locations.csv'

# Load node and carriers into cash
carriers = load_carriers_from_h5_results(h5_path)
node_loc = pd.read_csv(node_locations_path, sep=';')
country_loc = pd.read_csv(country_locations_path, sep=';')

# Load data into cash
with st.spinner('Wait for loading data...'):
    tec_operation = read_technology_operation(h5_path, re_gen_path)
    network_operation = read_network(h5_path)

# Determine Aggregation
time_agg_options = {'Annual Totals': 'Year',
                    'Monthly Totals': 'Month',
                    'Weekly Totals': 'Week',
                    'Daily Totals': 'Day',
                    'Hourly Totals': 'Hour'}
time_agg = st.sidebar.selectbox('Time Aggregation', time_agg_options.keys())

spatial_agg_options = ['Node', 'Country']
spatial_agg = st.sidebar.selectbox('Spatial Aggregation', spatial_agg_options)

# PREPROCESS DATA
preprocessed_data = {}

# Filter dfs for time aggregation
balance = aggregate_time(tec_operation, time_agg_options[time_agg])
networks = aggregate_time(network_operation, time_agg_options[time_agg])

# Filter dfs for spatial aggregation
balance = balance.T.reset_index()
balance = balance.groupby([spatial_agg, 'Technology', 'Carrier', 'Variable']).sum()
if spatial_agg == 'Country':
    balance = balance.rename_axis(index={'Country': 'Node'})
balance = balance.reset_index()

networks = networks.T
if spatial_agg == 'Country':
    networks = networks.reset_index()
    networks = networks[networks['FromCountry'] != networks['ToCountry']]
    networks = networks.groupby(['Network', 'FromCountry', 'ToCountry']).sum()
    networks = networks.rename_axis(index={'FromCountry': 'FromNode', 'ToCountry': 'ToNode'})

# Preprocessing - Supply
supply = balance[(balance['Variable'].isin(['generic_production', 'import', 'output', 'input'])) & (balance['Carrier'] == 'electricity')]
supply['Generation'] = np.where(supply['Variable'] == 'output', supply['Technology'], supply['Variable'])

keep_generation = ['generic_production',
                    'import',
                    'PowerPlant_Gas_existing',
                    'PowerPlant_Nuclear_existing',
                    'PowerPlant_Coal_existing',
                    'PowerPlant_Oil_existing',
                    'Storage_PumpedHydro_Open_existing',
                    'Storage_PumpedHydro_Reservoir_existing']

supply = supply[(supply['Generation'].isin(keep_generation))].drop(columns=['Technology', 'Carrier', 'Variable']).set_index(['Node', 'Generation']).T
preprocessed_data['supply'] = supply

# Preprocessing - Demand
demand = balance[(balance['Variable'].isin(['demand'])) & (balance['Carrier'] == 'electricity')].drop(columns=['Technology', 'Carrier', 'Variable'])
demand['Variable'] = 'Demand'
demand = demand.set_index(['Node', 'Variable']).T
preprocessed_data['demand'] = demand

# Preprocessing - Curtailment
curtailment = balance[(balance['Variable'].isin(['generic_production', 'Curtailment'])) & (balance['Carrier'] == 'electricity')]
curtailment = curtailment.drop(columns=['Technology', 'Carrier']).set_index(['Node', 'Variable']).T
preprocessed_data['curtailment'] = curtailment

# Preprocessing - Emissions
emissions = balance[(balance['Variable'].isin(['Emissions']))]
emissions = emissions.drop(columns=['Technology', 'Carrier', 'Variable']).set_index(['Node']).T
preprocessed_data['emissions'] = emissions

# Preprocessing Nodes
nodes = supply.columns.get_level_values(0).unique().to_list()
if spatial_agg == 'Country':
    onshore_nodes = nodes
else:
    onshore_nodes = node_loc[node_loc['nodetype'] == 'onshore']['Node']
    offshore_clusters = node_loc[node_loc['nodetype'] == 'offshore']

# Plotting Options
change_size_pie_chart = st.checkbox('Change pie size with total generation')
clusteres_individual = st.checkbox('Plot wind farm clusters individually')
show_values = st.checkbox('Show values in figure')

item_options = ['Curtailment', 'Generation', 'Emissions', 'Network Flows', 'Demand']
plot_items = st.multiselect('Plot the following items:', item_options, default= item_options)
network_options = networks.index.get_level_values('Network').unique().tolist()
plot_networks = st.multiselect('Select networks to plot', network_options, default=network_options)
plot_onshore_nodes = st.multiselect('Select onshore nodes to plot', onshore_nodes, default=onshore_nodes)
if spatial_agg == 'Node':
    plot_offshore_nodes = st.multiselect('Select offshore nodes to plot', offshore_clusters['Node'], default=offshore_clusters['Node'])
    font_size = 4
else:
    plot_offshore_nodes = []
    font_size = 6

# FILTER FOR TIMESTEPS
filtered_data = {}
if time_agg == 'Animation over time':
    for data in preprocessed_data.keys():
        filtered_data[data] = preprocessed_data[data]
    filtered_data['networks'] = networks

elif time_agg == 'Annual Totals':
    for data in preprocessed_data.keys():
        filtered_data[data] = preprocessed_data[data].T
    filtered_data['networks'] = networks

else:
    plot_timestep = st.sidebar.slider('Plot ' + time_agg_options[time_agg], min(supply.index), max(supply.index), 1)
    for data in preprocessed_data.keys():
        filtered_data[data] = preprocessed_data[data].filter(items=[int(plot_timestep)], axis=0).T
    filtered_data['networks'] = networks.T.filter(items=[int(plot_timestep)], axis=0).T

for data in filtered_data.keys():
    filtered_data[data].columns = ['Value']

# Preprocess data
filtered_data['supply'] = filtered_data['supply'].reset_index().pivot(index='Generation', columns='Node')['Value']
filtered_data['curtailment'] = filtered_data['curtailment'].reset_index().pivot(index='Variable', columns='Node')['Value']
filtered_data['demand'] = filtered_data['demand'].reset_index().pivot(index='Variable', columns='Node')['Value']
filtered_data['emissions'] = filtered_data['emissions'].T

# Create Chart
col1, col2 = st.columns([2, 1])

# LEGEND
item_color_dict = {}
if 'Generation' in plot_items:
    item_color_dict.update({'Coal Power Plant': 'dimgray',
                       'Gas Power Plant': 'lightgray',
                       'Nuclear Power Plant': 'yellow',
                       'Oil Power Plant': 'peru',
                       'Hydro Power Plant (open)': 'steelblue',
                       'Hydro Power Plant (reservoir)': 'darkblue',
                       'Renewable generation': 'palegreen',
                       'Import from outside system boundaries': 'black'})

fig_factor = 15

with col2:
    # Preprocess total data
    totals = {}
    for data in filtered_data.keys():
        totals[data] = filtered_data[data].sum(axis=1)

    # Calculate max
    max_supply = totals['supply'].sum()
    max_demand = totals['demand'].sum()
    max_emissions = totals['emissions'].sum()


    fig_total = plt.figure()
    gs = fig_total.add_gridspec(fig_factor + 1, fig_factor + 1)
    plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, top=1, bottom=0)

    ax_back = fig_total.add_subplot(frameon=True)
    frame_around_fig(ax_back, show_frame=True)

    y_start = 1
    y_end = fig_factor + 1
    x_start = 1
    x_end = fig_factor + 1

    # Create all axis
    ax = {}
    ax['title'] =  fig_total.add_subplot(gs[y_start:y_start+1, x_start:x_end], frameon=True)
    if 'Generation' in plot_items:
        ax['supply'] = fig_total.add_subplot(gs[y_start + 2:y_end - 3, x_start:x_end - 6], frameon=True)
    if 'Emissions' in plot_items:
        if change_size_pie_chart:
            ax['emissions'] = fig_total.add_subplot(gs[y_start + 2:y_end - 4, x_end - 3:x_end - 1], frameon=True)
        else:
            ax['emissions'] = fig_total.add_subplot(gs[y_start + 2:y_end - 4, x_end - 3:x_end - 1], frameon=True)
    if 'Demand' in plot_items:
        ax['demand'] = fig_total.add_subplot(gs[y_start + 2:y_end - 4, x_end - 6:x_end - 4], frameon=True)
    if 'Curtailment' in plot_items:
        ax['curtailment'] = fig_total.add_subplot(gs[y_end - 3:y_end - 1, x_start:x_end - 1], frameon=True)

    ax['title'].text(0.5, 0.5, 'Total System', horizontalalignment='center', verticalalignment='center', fontsize=20)

    if 'Generation' in plot_items:
        if change_size_pie_chart:
            radius_scale = np.power(sum(totals['supply']) / max_supply, 0.5)
        else:
            radius_scale = 1

        plot_node_supply(ax['supply'], totals['supply'], show_values, radius_scale, 12)
        ax['supply'].set_title('Generation', pad=-15, fontsize=12)

    if 'Emissions' in plot_items:
        plot_node_emissions(ax['emissions'], totals['emissions'], show_values, 12)
        ax['emissions'].set_title('Emissions', pad=-15, fontsize=12)

    if 'Demand' in plot_items:
        plot_node_demand(ax['demand'], totals['demand'], show_values, 12)
        ax['demand'].set_title('Demand', pad=-15, fontsize=12)

    if 'Curtailment' in plot_items:
        plot_node_curtailment(ax['curtailment'], totals['curtailment'], show_values, 12)
        ax['curtailment'].set_title('Fraction of used RE generation (remainder is curtailed)', pad=-15, fontsize=12)

    for plot in ax.keys():
        frame_around_fig(ax[plot], show_frame=False)

    st.pyplot(fig_total)

    # SHOW LEGEND
    for item, color in item_color_dict.items():
        hex_color = to_hex(color)
        st.markdown(
            f'<div style="background-color: {hex_color}; width: 50px; height: 20px; display: inline-block;"></div> {item}',
            unsafe_allow_html=True)

# PLOT
with col1:
    # Calculate max
    # max_supply = filtered_data['supply'].sum().max()
    # max_demand = filtered_data['demand'].sum().max()
    # max_emissions = filtered_data['emissions'].sum().max()

    # Create Plot
    fig = plt.figure()
    if spatial_agg == 'Country':
        gs = fig.add_gridspec(4 * fig_factor + 1, 4 * fig_factor + 1)
    else:
        gs = fig.add_gridspec(8 * fig_factor + 1, 9 * fig_factor + 1)
    plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, top=1, bottom=0)

    # Network Axis
    ax_back = fig.add_subplot(frameon=True)
    frame_around_fig(ax_back, show_frame=True)
    plot_positions = {}

    # Onshore Nodes
    for node in plot_onshore_nodes:
        plot_data = {}
        ax = {}
        for data in ['supply', 'curtailment', 'demand', 'emissions']:
            plot_data[data] = filtered_data[data][node].fillna(0)

        if spatial_agg == 'Country':
            location_row = int(country_loc.loc[country_loc['Node'] == node, 'row'].values[0])
            location_col = int(country_loc.loc[country_loc['Node'] == node, 'col'].values[0])
        else:
            location_row = int(node_loc.loc[node_loc['Node'] == node, 'row'].values[0])
            location_col = int(node_loc.loc[node_loc['Node'] == node, 'col'].values[0])

        y_start = location_row*fig_factor + 1
        y_end = location_row*fig_factor+fig_factor-1
        x_start = location_col*fig_factor+1
        x_end = location_col*fig_factor+fig_factor-1

        # Frame & Position
        ax_frame = fig.add_subplot(gs[y_start:y_end,
                                   x_start:x_end],
                                   frameon=True)
        frame_around_fig(ax_frame, show_frame=True)
        ax_frame.set_facecolor("white")

        position = ax_frame.get_position()
        plot_positions[node] = (position.x0 + 0.5 * (position.x1 - position.x0), position.y0 + 0.5 * (position.y1 - position.y0))

        # Create all axis
        ax['title'] = fig.add_subplot(gs[y_start:y_start+2, x_start:x_end], frameon=True)
        if 'Generation' in plot_items:
            ax['supply'] = fig.add_subplot(gs[y_start+2:y_end-3, x_start:x_end-6], frameon=True)
        if 'Emissions' in plot_items:
            if change_size_pie_chart:
                ax['emissions'] = fig.add_subplot(gs[y_start + 2:y_end - 4, x_end - 3:x_end - 1], frameon=True)
            else:
                ax['emissions'] = fig.add_subplot(gs[y_start + 2:y_end - 4, x_end - 3:x_end - 1], frameon=True)
        if 'Demand' in plot_items:
            ax['demand'] = fig.add_subplot(gs[y_start + 2:y_end - 4, x_end - 6:x_end - 4], frameon=True)
        if 'Curtailment' in plot_items:
            ax['curtailment'] = fig.add_subplot(gs[y_end - 3:y_end - 1, x_start + 1:x_end - 1], frameon=True)

        # Plot data
        ax['title'].text(0.5, 0.5, node, horizontalalignment='center', verticalalignment='center', fontsize=font_size)

        if 'Generation' in plot_items:
            if change_size_pie_chart:
                radius_scale = np.power(sum(plot_data['supply']) / max_supply, 0.5)
            else:
                radius_scale = 1

            plot_node_supply(ax['supply'], plot_data['supply'], show_values, radius_scale, font_size)

        if 'Emissions' in plot_items:
            plot_node_emissions(ax['emissions'], plot_data['emissions'], show_values, font_size)

        if 'Demand' in plot_items:
            plot_node_demand(ax['demand'], plot_data['demand'], show_values, font_size)

        if 'Curtailment' in plot_items:
            plot_node_curtailment(ax['curtailment'], plot_data['curtailment'], show_values, font_size)

        for plot in ax.keys():
            frame_around_fig(ax[plot], show_frame=False)

    # Offshore nodes
    if spatial_agg == 'Node':
        for cluster in offshore_clusters['offshore_cluster'].unique():
            idx = 0
            offshore_nodes = node_loc[node_loc['offshore_cluster'] == cluster]
            location_row = int(node_loc.loc[node_loc['offshore_cluster'] == cluster, 'row'].values[0])
            location_col = int(node_loc.loc[node_loc['offshore_cluster'] == cluster, 'col'].values[0])

            y_start = location_row*fig_factor + 1
            y_end = location_row*fig_factor+fig_factor-1
            x_start = location_col*fig_factor+1
            x_end = location_col*fig_factor+fig_factor-1

            ax_frame = fig.add_subplot(gs[y_start:y_end,
                                       x_start:x_end],
                                       frameon=True)
            frame_around_fig(ax_frame, show_frame=False)
            ax_frame.set_facecolor("white")

            c_position = ax_frame.get_position()
            x = c_position.x0 + 0.5 * (c_position.x1 - c_position.x0)
            y = c_position.y0 + 0.5 * (c_position.y1 - c_position.y0)

            plot_positions[cluster] = (x, y)

            for node in offshore_nodes['Node']:
                if node in plot_offshore_nodes:
                    plot_data_curt = filtered_data['curtailment'][node].fillna(0)
                    location_row = int(node_loc.loc[node_loc['Node'] == node, 'row'].values[0])
                    location_col = int(node_loc.loc[node_loc['Node'] == node, 'col'].values[0])

                    ax_node = fig.add_subplot(gs[y_start+1 + idx*2,
                                           x_start+1:x_start+4])
                    ax_node.text(0, 0.5, node, horizontalalignment='left', verticalalignment='center', fontsize=4)
                    ax_node.axis('off')

                    ax['curtailment'] = fig.add_subplot(gs[y_start + 1 + idx * 2,
                                                        x_start + 5:x_end - 1],
                                                        frameon=True)
                    if 'Curtailment' in plot_items:
                        plot_node_curtailment(ax['curtailment'], plot_data_curt, show_values, font_size)

                    y_position = ax['curtailment'].get_position()
                    plot_positions[node] = (
                    x, y_position.y0 + 0.5 * (y_position.y1 - y_position.y0))
                    idx += 1


    # Networks
    if 'Network Flows' in plot_items:
        networks = filtered_data['networks'].reset_index()

        # Filter for right network
        networks = networks[networks['Network'].isin(plot_networks)]

        if not clusteres_individual and spatial_agg == 'Node':
            networks = pd.merge(networks,
                                offshore_clusters[['Node', 'offshore_cluster']],
                                left_on='FromNode', right_on='Node', how='left').drop(columns=['Node'])
            networks = networks.rename(columns={'offshore_cluster': 'FromCluster'})
            networks = pd.merge(networks,
                                offshore_clusters[['Node', 'offshore_cluster']],
                                left_on='ToNode', right_on='Node', how='left').drop(columns=['Node'])
            networks = networks.rename(columns={'offshore_cluster': 'ToCluster'})

            networks['FromNode'] = networks.apply(lambda row: row['FromNode'] if pd.isna(row['FromCluster']) else row['FromCluster'], axis=1)
            networks['ToNode'] = networks.apply(lambda row: row['ToNode'] if pd.isna(row['ToCluster']) else row['ToCluster'], axis=1)

        G = nx.Graph()
        G.add_nodes_from(plot_positions.keys())

        networks = networks.groupby(['FromNode', 'ToNode']).sum().reset_index()
        max_flow = networks['Value'].max()
        for _, edge_data in networks.iterrows():
            # Normalize edge value to be within [0, 1]
            from_node = edge_data['FromNode']
            to_node = edge_data['ToNode']

            if (from_node in plot_onshore_nodes or from_node in plot_offshore_nodes) and (to_node in plot_onshore_nodes or to_node in plot_offshore_nodes):

                flow_this_direction = edge_data['Value']
                flow_other_direction = networks[
                    (networks['FromNode'] == to_node) &
                    (networks['ToNode'] == from_node)]

                uni_flow = flow_this_direction - flow_other_direction.loc[:, 'Value'].values[0]

                if uni_flow > 0.1:
                    G.add_edge(from_node, to_node, weight=uni_flow/max_flow*10)
                    x2 = (plot_positions[from_node][0] + plot_positions[to_node][0]) / 2
                    y2 = (plot_positions[from_node][1] + plot_positions[to_node][1]) / 2
                    ax_back.annotate('',
                                     (x2, y2),
                                     xytext=(plot_positions[from_node][0], plot_positions[from_node][1]),
                                     arrowprops=dict(facecolor='blue', lw=0, width=0, headwidth=np.power(uni_flow/max_flow*200, 0.6), headlength=20),
                                     fontsize=5,
                                     horizontalalignment='right', verticalalignment='top')

        weights = nx.get_edge_attributes(G, 'weight').values()
        nx.draw(G, pos=plot_positions, ax=ax_back, with_labels=False, width=list(weights), edge_color='blue', node_size=0)

        ax_back.set_axis_on()
        ax_back.set_xlim(0, 1)
        ax_back.set_ylim(0, 1)
        ax_back.set_xlim(left=0)

    st.pyplot(fig)


#
# def create_map(col, supply_filtered, curtailment_filtered):
#     supply_filtered.columns = ['Value']
#     curtailment_filtered.columns = ['Value']
#     supply_filtered = supply_filtered.reset_index().pivot(index='Generation', columns='Node')['Value']
#     curtailment_filtered = curtailment_filtered.reset_index().pivot(index='Variable', columns='Node')['Value']
#
#     map_center = [55, 5]
#     map = folium.Map(location=map_center, zoom_start=5)
#
#     # PLOTS
#     for node in nodes:
#         plot_data_sup = supply_filtered[node].fillna(0)
#         plot_data_curt = curtailment_filtered[node].fillna(0)
#
#         fig = plt.figure(figsize=(figure_scaling, figure_scaling))
#         fig.suptitle(node, fontsize=12, fontweight='bold')  # Adjust font size and weight as needed
#         gs = fig.add_gridspec(10, 10)
#
#         if 'Generation' in items:
#             ax1 = fig.add_subplot(gs[0:8, 0:8])
#             ax1.pie(plot_data_sup, startangle=90, colors=[color for _, color in item_color_dict.items()])
#
#         if 'Curtailment' in items:
#             ax2 = fig.add_subplot(gs[8:10, :])
#             ax2.barh(1, plot_data_curt['generic_production'] / plot_data_curt.sum(), color='gray')
#             ax2.barh(1, plot_data_curt['Curtailment'] / plot_data_curt.sum(),
#                      left=plot_data_curt['generic_production'] / plot_data_curt.sum(), color='black')
#             ax2.set_xlim([0, 1])
#             ax2.axis('off')
#
#         if spatial_agg == 'Country':
#             location = [country_loc.loc[country_loc['Node'] == node, 'lat'].values[0] + move_y,
#                         country_loc.loc[country_loc['Node'] == node, 'lon'].values[0] + move_x]
#         else:
#             location = [node_loc.loc[node_loc['Node'] == node, 'lat'].values[0] + move_y,
#                         node_loc.loc[node_loc['Node'] == node, 'lon'].values[0] + move_x]
#
#         marker_feature_group = create_folium_marker(fig, location)
#         marker_feature_group.add_to(map)
#
#     with col:
#         folium_static(map, width=1000, height=900)




  #
        #
        # def update(frame):
        #     for node in nodes:
        #         if node_loc.loc[node_loc['Node'] == node, 'nodetype'].values[0] == 'onshore':
        #             supply_filtered = supply.filter(items=[int(supply.index[frame])], axis=0).T
        #             supply_filtered.columns = ['Value']
        #             supply_filtered = supply_filtered.reset_index().pivot(index='Generation', columns='Node')['Value']
        #             plot_data_sup = supply_filtered[node].fillna(0)
        #
        #             ax[node].clear()
        #             ax[node].pie(plot_data_sup, startangle=90, colors=[color for _, color in item_color_dict.items()])
        #
        #
        # # Create an animation
        # fig = plt.figure()
        # fig_factor = 6
        # gs = fig.add_gridspec(8*fig_factor, 9*fig_factor)
        #
        # ax = {}
        # for node in nodes:
        #     if spatial_agg == 'Country':
        #         location_row = int(country_loc.loc[country_loc['Node'] == node, 'row'].values[0])
        #         location_col = int(country_loc.loc[country_loc['Node'] == node, 'col'].values[0])
        #     else:
        #         location_row = int(node_loc.loc[node_loc['Node'] == node, 'row'].values[0])
        #         location_col = int(node_loc.loc[node_loc['Node'] == node, 'col'].values[0])
        #     if node_loc.loc[node_loc['Node'] == node, 'nodetype'].values[0] == 'onshore':
        #         ax[node] = fig.add_subplot(gs[location_row * fig_factor:location_row * fig_factor + 4,
        #                                    location_col * fig_factor:location_col * fig_factor + 3])
        #         ax[node].set_title(node, fontsize=4, pad=0)
        #
        # ani = FuncAnimation(fig, update, frames=len(supply.index),
        #                     interval=400)  # Change interval as needed (in milliseconds)
        #
        # # Display the animation
        # components.html(ani.to_jshtml(), height=1000)

       #
        # else:
        #     fig_factor = 6
        #     gs = fig.add_gridspec(8 * fig_factor, 9 * fig_factor)
        #     onshore_nodes = node_loc[node_loc['nodetype'] == 'onshore']
        #     offshore_clusters = node_loc[node_loc['nodetype'] == 'offshore']
        #     for node in onshore_nodes['Node']:
        #         plot_data_sup = supply_filtered[node].fillna(0)
        #         plot_data_curt = curtailment_filtered[node].fillna(0)
        #         location_row = int(node_loc.loc[node_loc['Node'] == node, 'row'].values[0])
        #         location_col = int(node_loc.loc[node_loc['Node'] == node, 'col'].values[0])
        #         ax_s = plot_node_supply(node, plot_data_sup, gs[location_row*fig_factor:location_row*fig_factor+4,location_col*fig_factor:location_col*fig_factor+3])
        #         ax_c = plot_node_curtailment(plot_data_curt, gs[location_row*fig_factor+4,location_col*fig_factor:location_col*fig_factor+fig_factor])
        #
        #