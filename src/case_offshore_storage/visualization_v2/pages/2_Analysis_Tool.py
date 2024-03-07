import pandas as pd
import streamlit as st
from utilities import *
from read_data import *
import altair as alt

# Streamlit settings
st.set_page_config(layout="wide")
if 'line_data_energy' not in st.session_state:
    st.session_state['line_data_energy'] = pd.DataFrame()
if 'line_data_emissions' not in st.session_state:
    st.session_state['line_data_emissions'] = pd.DataFrame()

# Plotting options
var_types = {}
var_types['Technology Operation'] = {}
var_types['Emissions'] = {}
var_types['Energy balance at node'] = {'Imports': 'import',
                               'Exports': 'export',
                               'Demand': 'demand',
                               'Technology Outputs': 'technology_outputs',
                               'Technology Inputs': 'technology_inputs',
                               'Network Outflow': 'network_outflow',
                               'Network Inflow': 'network_inflow',
                               'Maximal renewable generation': 'max_re',
                               'Actual renewable generation': 'generic_production',
                               'Curtailment': 'Curtailment'}

carriers = {'Electricity': 'electricity', 'Natural gas': 'gas', 'Hydrogen': 'hydrogen'}


# Required Paths
root = 'src/case_offshore_storage/visualization_v2/'
case_keys = root + 'data/Cases.csv'
re_gen_path = root + 'data/production_profiles_re.csv'
node_locations_path = root + 'data/Node_Locations.csv'
country_locations_path = root + 'data/Country_Locations.csv'

# Load data
cases_available = pd.read_csv(case_keys, sep=';')
node_loc = pd.read_csv(node_locations_path, sep=';')
country_loc = pd.read_csv(country_locations_path, sep=';')

# Select Cases
cases_selected = {}
cases_path = {}
cases_selected[1] = st.sidebar.selectbox('Select the first case: ', cases_available['case'])
cases_selected[2] = st.sidebar.selectbox('Select the second case: ', cases_available['case'])

if cases_selected[1] == cases_selected[2]:
    cases_path[cases_selected[1]] = root + 'data/cases/' + cases_available[cases_available['case'] == cases_selected[1]]['file_name'].values[0]
else:
    for case in cases_selected.keys():
        cases_path[cases_selected[case]] = root + 'data/cases/' + cases_available[cases_available['case'] == cases_selected[case]]['file_name'].values[0]

# Determine Aggregation
st.sidebar.markdown("""---""")
time_agg_options = {'Annual Totals': 'Year',
                    'Monthly Totals': 'Month',
                    'Weekly Totals': 'Week',
                    'Daily Totals': 'Day',
                    'Hourly Totals': 'Hour'}
time_agg = st.sidebar.selectbox('Time Aggregation', time_agg_options.keys())

spatial_agg_options = ['Node', 'Country']
spatial_agg = st.sidebar.selectbox('Spatial Aggregation', spatial_agg_options)

# Load data into cash and aggregate
tec_operation = {}
network_operation = {}
tec_operation_agg = {}
network_operation_agg = {}
with st.spinner('Wait for loading data...'):
    for case in cases_path.keys():
        # Aggregate
        tec_operation_agg_time = aggregate_time(
            read_technology_operation(cases_path[case], re_gen_path),
            time_agg_options[time_agg])
        network_operation_agg_time = aggregate_time(
            read_network(cases_path[case]),
            time_agg_options[time_agg])

        tec_operation_agg[case] = aggregate_spatial_balance(tec_operation_agg_time, spatial_agg)
        network_operation_agg[case] = aggregate_spatial_networks(network_operation_agg_time, spatial_agg)

        tec_operation_agg[case] = tec_operation_agg[case].set_index(['Node', 'Technology', 'Carrier', 'Variable'])

nodes =  tec_operation_agg[cases_selected[1]].index.get_level_values('Node').unique()

# Select Figures
col1, col2 = st.columns(2)

with col1:
    line_chart = st.checkbox('Display line chart')
    st.markdown("""---""")

    if st.button('Reset plot data'):
        st.session_state['line_data_energy'] = pd.DataFrame()
        st.session_state['line_data_emissions'] = pd.DataFrame()

    if line_chart:
        c_seleted = st.selectbox('Select case', cases_path.keys(), key="l_case")
        f_node = st.selectbox('Select node', nodes, key="l_nodes")
        l_var_type_selected = st.selectbox('Select type of variable', var_types.keys(), key="l_var_type")

        add_to_plot_data_energy = pd.DataFrame()
        add_to_plot_data_emissions = pd.DataFrame()

        if l_var_type_selected == 'Energy balance at node':
            carrier_selected = st.selectbox('Select a carrier', carriers.keys(), key="l_car")
            l_var_selected = st.selectbox('Select variable', var_types[l_var_type_selected].keys(), key="l_var")
            f_var = var_types[l_var_type_selected][l_var_selected]
            f_car = carriers[carrier_selected]

            add_to_plot_data_energy = tec_operation_agg[c_seleted].loc[(f_node, slice(None), f_car, f_var)] / 1000
            add_to_plot_data_energy.index = [f_node + ' | ' + f_var + ' | ' + f_car]

        elif l_var_type_selected == 'Technology Operation':
            technologies_available = tec_operation_agg[c_seleted].loc[(f_node, slice(None), slice(None), 'output')].reset_index()['Technology'].unique()
            f_tec = st.selectbox('Select Technology', technologies_available, key="l_tec")
            carriers_available = tec_operation_agg[c_seleted].loc[(f_node, f_tec , slice(None), 'output')].reset_index()['Carrier'].unique()
            if len(carriers_available) > 1:
                f_car = st.selectbox('Select carrier', carriers_available, key="l_car")
                add_to_plot_data_energy = tec_operation_agg[c_seleted].loc[(f_node, f_tec, f_car, 'output')] / 1000
                add_to_plot_data_energy.index = [f_node + ' | ' + f_tec + ' | ' + f_car]
            else:
                add_to_plot_data_energy = tec_operation_agg[c_seleted].loc[(f_node, f_tec, slice(None), 'output')] / 1000
                add_to_plot_data_energy.index = [f_node + ' | ' + f_tec]

        elif l_var_type_selected == 'Emissions':
            add_to_plot_data_emissions = tec_operation_agg[c_seleted].loc[(f_node, slice(None), slice(None), 'Emissions')] / 1000000
            add_to_plot_data_emissions.index = [f_node + ' | Emissions']

        if st.button('Add to plot'):
            if not add_to_plot_data_energy.empty:
                st.session_state['line_data_energy'] = st.session_state['line_data_energy'].append(add_to_plot_data_energy)
            if not add_to_plot_data_emissions.empty:
                st.session_state['line_data_emissions'] = st.session_state['line_data_emissions'].append(add_to_plot_data_emissions)

        # PLOTS
        if st.session_state['line_data_energy'].empty:
            chart_energy = None
        else:
            if time_agg == 'Annual Totals':
                chart_energy = alt.Chart(
                    st.session_state['line_data_energy'].reset_index().melt(id_vars=['index'])).mark_bar().encode(
                    x=alt.X('Timeslice:Q', axis=alt.Axis(format='d')).title(time_agg_options[time_agg]),
                    y=alt.Y('value').title('Energy (GWh)', titleColor='#57A44C'),
                    color='index',
                    tooltip=['Timeslice', 'value']
                ).properties(
                    width=800,
                    height=400
                ).interactive()
            else:
                chart_energy = alt.Chart(
                    st.session_state['line_data_energy'].reset_index().melt(id_vars=['index'])).mark_line().encode(
                    x=alt.X('Timeslice:Q', axis=alt.Axis(format='d')).title(time_agg_options[time_agg]),
                    y=alt.Y('value').title('Energy (MWh)', titleColor='#57A44C'),
                    color='index',
                    tooltip=['Timeslice', 'value']
                ).properties(
                    width=800,
                    height=400
                ).interactive()

            export_csv(st.session_state['line_data_energy'].reset_index(), 'Download data as CSV', 'operation_data.csv')

        if st.session_state['line_data_emissions'].empty:
            chart_emissions = None
        else:
            chart_emissions = alt.Chart(
                st.session_state['line_data_emissions'].reset_index().melt(id_vars=['index'])).mark_bar().encode(
                x=alt.X('Timeslice:Q', axis=alt.Axis(format='d')).title(time_agg_options[time_agg]),
                y=alt.Y('value').title('Emissions (Mt)', titleColor='#5276A7'),
                color='index',
                tooltip=['Timeslice', 'value']
            ).properties(
                width=800,
                height=400
            ).interactive()

            export_csv(st.session_state['line_data_emissions'].reset_index(), 'Download emission data as CSV', 'emission_data.csv')

        if chart_energy and chart_emissions:
            combined_chart = alt.layer(chart_energy, chart_emissions).resolve_scale(
                y='independent'
            )
            chart_set = 1
        elif chart_energy:
            combined_chart = chart_energy
            chart_set = 1
        elif chart_emissions:
            combined_chart = chart_emissions
            chart_set = 1
        else:
            chart_set = 0

        if chart_set:
            st.altair_chart(combined_chart)
#
# with col2:
#     scatter_plot = st.checkbox('Display scatter plot')
#
#     col2_1, col2_2 = st.columns(2)
#
#
#     if scatter_plot:
#         c_seleted = st.selectbox('Select case', cases_path.keys(), key="s1_case")
#         f_node = st.selectbox('Select node', nodes, key="s1_nodes")
#         l_var_type_selected = st.selectbox('Select type of variable', var_types.keys(), key="s1_var_type")
#
#         add_to_plot_data_energy = pd.DataFrame()
#         add_to_plot_data_emissions = pd.DataFrame()
#
#         if l_var_type_selected == 'Energy balance at node':
#             carrier_selected = st.selectbox('Select a carrier', carriers.keys(), key="s1_car")
#             l_var_selected = st.selectbox('Select variable', var_types[l_var_type_selected].keys(), key="s1_var")
#             f_var = var_types[l_var_type_selected][l_var_selected]
#             f_car = carriers[carrier_selected]
#
#             add_to_plot_data_energy = tec_operation_agg[c_seleted].loc[(f_node, slice(None), f_car, f_var)]
#             add_to_plot_data_energy.index = [f_node + ' | ' + f_var + ' | ' + f_car]
#
#         elif l_var_type_selected == 'Technology Operation':
#             technologies_available = tec_operation_agg[c_seleted].loc[(f_node, slice(None), slice(None), 'output')].reset_index()['Technology'].unique()
#             f_tec = st.selectbox('Select Technology', technologies_available, key="s1_tec")
#             carriers_available = tec_operation_agg[c_seleted].loc[(f_node, f_tec , slice(None), 'output')].reset_index()['Carrier'].unique()
#             if len(carriers_available) > 1:
#                 f_car = st.selectbox('Select carrier', carriers_available, key="s1_car")
#                 add_to_plot_data_energy = tec_operation_agg[c_seleted].loc[(f_node, f_tec, f_car, 'output')]
#                 add_to_plot_data_energy.index = [f_node + ' | ' + f_tec + ' | ' + f_car]
#             else:
#                 add_to_plot_data_energy = tec_operation_agg[c_seleted].loc[(f_node, f_tec, slice(None), 'output')]
#                 add_to_plot_data_energy.index = [f_node + ' | ' + f_tec]
#
#         elif l_var_type_selected == 'Emissions':
#             add_to_plot_data_emissions = tec_operation_agg[c_seleted].loc[(f_node, slice(None), slice(None), 'Emissions')]
#             add_to_plot_data_emissions.index = [f_node + ' | Emissions']
#
#         if st.button('Add to plot', key='s_add'):
#             if not add_to_plot_data_energy.empty:
#                 st.session_state['line_data_energy'] = st.session_state['line_data_energy'].append(add_to_plot_data_energy)
#             if not add_to_plot_data_emissions.empty:
#                 st.session_state['line_data_emissions'] = st.session_state['line_data_emissions'].append(add_to_plot_data_emissions)
#
#
#
#
#



