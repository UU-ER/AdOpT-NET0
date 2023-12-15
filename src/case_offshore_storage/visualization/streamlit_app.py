import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
import os

from utilities import determine_graph_boundaries, select_node, read_time_series, plot_area_chart
from networks import network_sizes

def find_directory(parent_dir, suffix):
    for root, dirs, files in os.walk(parent_dir):
        for dir_name in dirs:
            if dir_name.endswith(suffix):
                return Path(os.path.join(root, dir_name))

    return None

def convert_to_string(value):
    if isinstance(value, float):
        int_value = int(value)
        return str(int_value) if value == int_value else str(value)
    else:
        return str(value)

# Define paths to results
path = Path('C:/Users/6574114/OneDrive - Universiteit Utrecht/Offshore Storage_Full Paper/Results')
selected_option = st.sidebar.selectbox("Select an option", ['Baseline Results', 'Emission Reduction', 'Storage Maximum Investment'])

# Load Baseline:
baseline_results = pd.read_csv(Path.joinpath(path, Path('Overview_baseline.csv')))
self_sufficiency_values = list(baseline_results['Self Sufficiency'].unique())
offshore_share_values = list(baseline_results['Offshore Share'].unique())

self_sufficiency = st.sidebar.selectbox("Select a self sufficiency ratio", self_sufficiency_values)
offshore_share = st.sidebar.selectbox("Select a share of renewable electricity produced offshore", offshore_share_values)


if selected_option == 'Baseline Results':
    pages = ["Summary", "Energy Balance at Node", "Technology Operation"]
    selected_page = st.sidebar.selectbox("Select graph", pages)

    search_pattern = 'Baseline_SS' + convert_to_string(self_sufficiency) + 'OS' + convert_to_string(offshore_share)
    result_path = find_directory(path.joinpath(path, Path('Baseline')), search_pattern)

    # Energybalance at Node
    if selected_page == "Energy Balance at Node":
        selected_node, node_path = select_node(result_path)

        energybalance_data = read_time_series(
            Path.joinpath(node_path, selected_node, 'Energybalance.xlsx'))

        selected_carrier = st.selectbox('Select a carrier:', energybalance_data.keys())

        x_min, x_max = determine_graph_boundaries(energybalance_data)
        st.title("Energy Balance per Node")

        st.header("Supply")
        # Multi-select box for filtering series
        series_supply = ['Timestep',
                         'Generic_production',
                         'Technology_outputs',
                         'Network_inflow',
                         'Import']
        selected_supply_series = st.multiselect('Select Series to Filter', series_supply,
                                                default=series_supply)
        plot_data = energybalance_data[selected_carrier][selected_supply_series]
        chart = plot_area_chart(plot_data, x_min, x_max)
        st.altair_chart(chart, theme="streamlit", use_container_width=True)

        st.header("Demand")
        # Multi-select box for filtering technologies
        series_demand = ['Timestep',
                         'Demand',
                         'Technology_inputs',
                         'Network_outflow',
                         'Export']
        selected_demand_series = st.multiselect('Select Series to Filter', series_demand,
                                                default=series_demand)
        plot_data = energybalance_data[selected_carrier][selected_demand_series]
        chart = plot_area_chart(plot_data, x_min, x_max)
        st.altair_chart(chart, theme="streamlit", use_container_width=True)


    # Technology Operation
    elif selected_page == "Technology Operation":
        selected_node, node_path = select_node(result_path)

        try:

            tec_operation_data = read_time_series(
                Path.joinpath(node_path, selected_node, 'TechnologyOperation.xlsx'))

            x_min, x_max = determine_graph_boundaries(tec_operation_data)

            st.title("Technology Operation")
            tec = st.selectbox('Select a technology:', tec_operation_data.keys())

            # Input
            st.header("Input")
            tec_data = tec_operation_data[tec]
            variables = [col for col in tec_data.columns if col.startswith('input')]
            variables.append('Timestep')
            plot_data = tec_data[variables]
            chart = plot_area_chart(plot_data, x_min, x_max)
            st.altair_chart(chart, theme="streamlit", use_container_width=True)

            # Output
            st.header("Output")
            tec_data = tec_operation_data[tec]
            variables = [col for col in tec_data.columns if col.startswith('output')]
            variables.append('Timestep')
            plot_data = tec_data[variables]
            chart = plot_area_chart(plot_data, x_min, x_max)
            st.altair_chart(chart, theme="streamlit", use_container_width=True)

            st.header("Other Variables")
            tec_data = tec_operation_data[tec]
            variables = [col for col in tec_data.columns if col.startswith('storagelevel')]
            if len(variables) >= 1:
                variables.append('Timestep')
                plot_data = tec_data[variables]
                chart = plot_area_chart(plot_data, x_min, x_max)
                st.altair_chart(chart, theme="streamlit", use_container_width=True)

        except FileNotFoundError:
            st.text("There are no technologies to show for this node.")

    # Summary Comparison
    elif selected_page == "Summary":

        summary_data = pd.read_excel(Path.joinpath(result_path, 'Summary.xlsx'), sheet_name='Summary',
                                             index_col=0)

        st.header("Total Cost")
        chart = alt.Chart(summary_data.reset_index()).mark_bar().encode(
            y='Total_Cost:Q',
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

        st.header("Total Emissions")
        chart = alt.Chart(summary_data.reset_index()).mark_bar().encode(
            y='Net_Emissions:Q',
        ).interactive()
        st.altair_chart(chart, use_container_width=True)


elif selected_option == 'Emission Reduction':
    emission_results = pd.read_csv(Path.joinpath(path, Path('Overview_emission_reduction.csv')))
    emission_results['Curtailment Total'] = emission_results['Curtailment Onshore'] + emission_results['Curtailment Offshore']
    emission_results_filtered = emission_results[(emission_results['Self Sufficiency'] == self_sufficiency) &
                                                 (emission_results['Offshore Share'] == offshore_share)]
    emissions_baseline = baseline_results[(emission_results['Self Sufficiency'] == self_sufficiency) &
                                                 (emission_results['Offshore Share'] == offshore_share)]['Emissions'].values.item()
    emission_results_filtered['Emission Reduction'] = emission_results_filtered['Emissions'] / emissions_baseline
    emission_results_filtered['Technology'] = pd.concat([emission_results_filtered['Technology'], emission_results_filtered['Node']], axis=1).apply(lambda x: ' '.join(x), axis=1)

    # Emission Reduction
    st.header("Emission Reduction")
    chart = alt.Chart(emission_results_filtered).mark_line(point=True).encode(
        x='Emission Reduction:Q',
        y='Size:Q',
        color='Technology:N',
        tooltip=['Technology:N', 'Size:Q', 'Emission Reduction:Q']
    )
    st.altair_chart(chart, use_container_width=True)

    # Curtailment
    st.header("Curtailment")
    baseline_single_row = baseline_results[(emission_results['Self Sufficiency'] == self_sufficiency) &
                                                 (emission_results['Offshore Share'] == offshore_share)]
    baseline_single_row['Technology'] = 'Baseline'
    baseline_single_row['Emission Reduction'] = 1
    curtailment_baseline = (baseline_single_row['Curtailment Onshore'] + baseline_single_row['Curtailment Offshore']).values.item()
    baseline_single_row['Curtailment Total'] = curtailment_baseline

    merged_df = pd.merge(emission_results_filtered, baseline_single_row, how='outer')
    merged_df['Curtailment Normalized'] = merged_df['Curtailment Total'] / curtailment_baseline

    chart = alt.Chart(merged_df).mark_line(point=True).encode(
        x='Emission Reduction:Q',
        y='Curtailment Normalized:Q',
        color='Technology:N',
        tooltip=['Technology:N', 'Size:Q', 'Curtailment Normalized:Q']
    )
    st.altair_chart(chart, use_container_width=True)

    # Cost
    st.header("Cost")
    st.text('Enter cost parameters for the respective technologies:')
    capex = {}
    lifetime = {}
    interest = {}
    for tech in emission_results_filtered['Technology'].unique():
        st.subheader(tech)
        capex[tech] = st.text_input('Capex for ' + tech, key='Capex' + tech)
        lifetime[tech] = st.text_input('Lifetime for ' + tech, key='T' + tech)
        interest[tech] = st.text_input('Interest rate for ' + tech, key='R' + tech)
        # problem: its all reported included cost for storage!

elif selected_option == 'Storage Maximum Investment':
    max_capex_results = pd.read_csv(Path.joinpath(path, Path('Overview_max_capex.csv')))
    max_capex_results = max_capex_results[(max_capex_results['Self Sufficiency'] == self_sufficiency) &
                                                 (max_capex_results['Offshore Share'] == offshore_share)]
    max_capex_results['Technology'] = pd.concat([max_capex_results['Technology'], max_capex_results['Node']], axis=1).apply(lambda x: ' '.join(x), axis=1)


    st.header("Maximal Allowable Investment Costs (annualized)")
    chart = alt.Chart(max_capex_results).mark_bar().encode(
        y='Technology:N',
        x='Max Capex:Q',
        tooltip=['Technology:N', 'Max Capex:Q']
    )
    st.altair_chart(chart, use_container_width=True)
#
# st.header("Total Cost")
# chart = alt.Chart(summary_data.reset_index()).mark_bar().encode(
#     y='Total_Cost:Q',
# ).interactive()
# st.altair_chart(chart, use_container_width=True)

