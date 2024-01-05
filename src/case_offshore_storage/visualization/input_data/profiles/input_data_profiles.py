import streamlit as st
import altair as alt
from pathlib import Path
import pandas as pd

from utilities import determine_graph_boundaries

def show_profiles(category):

    climate_year = 2009
    root_load_path = Path('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/clean_data/')
    loadpaths = {}
    loadpaths['re_profiles'] = Path('production_profiles_re/production_profiles_re.csv')
    loadpaths['demand'] = Path('demand/TotalDemand_NT_' + str(climate_year) +'.csv')
    loadpaths['hydro_inflows'] = Path('hydro_inflows/hydro_inflows.csv')

    re_profiles = pd.read_csv(Path.joinpath(root_load_path, loadpaths['re_profiles']), index_col=0, header=[0, 1])
    re_profiles = re_profiles.loc[:, ~re_profiles.columns.get_level_values('Profile').str.contains('total')]
    re_profiles = re_profiles.T.reset_index()
    re_profiles['Country'] = re_profiles.reset_index()['Node'].str[0:2]
    re_profiles = re_profiles.set_index(['Country', 'Node', 'Profile'])

    demand = pd.read_csv(Path.joinpath(root_load_path, loadpaths['demand']), index_col=0)
    demand = demand.T.reset_index()
    demand['Country'] = demand.reset_index()['index'].str[0:2]
    demand = demand.rename(columns={'index': 'Node'}).set_index(['Country', 'Node'])

    hydro_inflows = pd.read_csv(Path.joinpath(root_load_path, loadpaths['hydro_inflows']), index_col=0, header=[0, 1])
    hydro_inflows = hydro_inflows.T.reset_index().rename(columns={'level_0': 'Node', 'level_1': 'Profile'})
    hydro_inflows['Country'] = hydro_inflows.reset_index()['Node'].str[0:2]
    hydro_inflows = hydro_inflows.set_index(['Country', 'Node', 'Profile'])

    if category == 'Aggregated (total)':
        re_profiles_filtered = re_profiles.groupby(['Profile']).sum().T

        hydro_inflows_filtered = hydro_inflows.groupby(['Profile']).sum().T

        demand_filtered = pd.DataFrame(demand.sum(), columns=['value'])
        demand_filtered.index = re_profiles_filtered.index


    elif category == 'Aggregated (per country)':
        country = st.selectbox("Select a country:", re_profiles.index.get_level_values('Country').unique())
        re_profiles_filtered = re_profiles.loc[re_profiles.index.get_level_values('Country') == country].groupby(['Profile']).sum().T

        hydro_inflows_filtered = hydro_inflows.loc[hydro_inflows.index.get_level_values('Country') == country].groupby(['Profile']).sum().T

        demand_filtered = demand.loc[demand.index.get_level_values('Country') == country].groupby(['Country']).sum().T
        demand_filtered.index = re_profiles_filtered.index

    elif category == 'Per node':
        node = st.selectbox("Select a Node:", re_profiles.index.get_level_values('Node').unique())
        re_profiles_filtered = re_profiles.loc[re_profiles.index.get_level_values('Node') == node].groupby(['Profile']).sum().T

        hydro_inflows_filtered = hydro_inflows.loc[hydro_inflows.index.get_level_values('Node') == node].groupby(['Profile']).sum().T

        demand_filtered = demand.loc[demand.index.get_level_values('Node') == node].groupby(['Node']).sum().T
        demand_filtered.index = re_profiles_filtered.index

    technologies = st.multiselect('Select time series to filter', list(re_profiles_filtered.columns) + list(hydro_inflows_filtered.columns) + ['Demand'])

    demand_filtered = demand_filtered.reset_index()
    demand_filtered.columns = ['Timestep', 'value']
    demand_filtered['Timestep'] = pd.to_datetime(demand_filtered['Timestep'])
    demand_filtered['Demand'] = 'Demand'

    re_profiles_filtered = re_profiles_filtered[list(set(technologies) & set(list(re_profiles_filtered.columns)))].reset_index()
    re_profiles_filtered = re_profiles_filtered.rename(columns={'index': 'Timestep'})
    re_profiles_filtered['Timestep'] = pd.to_datetime(re_profiles_filtered['Timestep'])

    hydro_inflows_filtered = hydro_inflows_filtered[list(set(technologies) & set(list(hydro_inflows_filtered.columns)))].reset_index()
    hydro_inflows_filtered = hydro_inflows_filtered.rename(columns={'index': 'Timestep'})
    hydro_inflows_filtered['Timestep'] = pd.to_datetime(re_profiles_filtered['Timestep'])

    x_min, x_max = determine_graph_boundaries(demand_filtered)

    re_profiles_filtered = re_profiles_filtered[(re_profiles_filtered['Timestep'] >= x_min) & (re_profiles_filtered['Timestep'] <= x_max)]
    hydro_inflows_filtered = hydro_inflows_filtered[(hydro_inflows_filtered['Timestep'] >= x_min) & (hydro_inflows_filtered['Timestep'] <= x_max)]
    demand_filtered = demand_filtered[(demand_filtered['Timestep'] >= x_min) & (demand_filtered['Timestep'] <= x_max)]

    re_profiles_melted = re_profiles_filtered.melt(id_vars=['Timestep'])
    hydro_inflows_melted = hydro_inflows_filtered.melt(id_vars=['Timestep'])

    profile_order_mapping = {'Biomass': 1, 'PV': 5, 'Run of River': 2, 'Wind offshore': 4, 'Wind onshore': 3}
    re_profiles_melted['Order'] = re_profiles_melted['Profile'].map(profile_order_mapping)

    re_generation_chart = alt.Chart(re_profiles_melted).mark_area().encode(
        x='Timestep:T',
        y=alt.Y('value:Q', axis=alt.Axis(title='Power in MW')),
        color="Profile:N",
        order=alt.Order(
            'Order',
            sort='ascending'
        )
    )

    hydro_inflow_chart = alt.Chart(hydro_inflows_melted).mark_line().encode(
        x='Timestep:T',
        y=alt.Y('value:Q', axis=alt.Axis(title='Power in MW')),
        color="Profile:N",
    )

    if 'Demand' in technologies:
        demand_chart = alt.Chart(demand_filtered.reset_index()).mark_line().encode(
            x='Timestep:T',
            y=alt.Y('value:Q', axis=alt.Axis(title='Power in MW')),
            color=alt.Color('Demand:N')
        )
        layer_chart = re_generation_chart + demand_chart + hydro_inflow_chart
    else:
        layer_chart = re_generation_chart + hydro_inflow_chart

    layer_chart.configure_legend(orient='bottom')
    st.altair_chart(layer_chart, theme="streamlit", use_container_width=True)
