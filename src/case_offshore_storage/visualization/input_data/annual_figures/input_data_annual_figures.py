import streamlit as st
import altair as alt
from pathlib import Path
import pandas as pd

def show_annual_figures():
    loadpaths = {}
    loadpaths['Country'] = Path('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/reporting/demand_supply_country.csv')
    loadpaths['Nodal'] = Path('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/reporting/demand_supply_node.csv')

    show_what = st.selectbox('Select level: ', ['Nodal', 'Country'])

    if show_what == 'Country':
        plot_data = pd.read_csv(loadpaths[show_what], index_col=0, header=[0,1]).reset_index().rename(columns={'Country': 'index'})
    else:
        plot_data = pd.read_csv(loadpaths[show_what], index_col=0, header=[0,1]).reset_index()

    plot_data = plot_data.melt(id_vars='index')
    plot_data = plot_data[plot_data['value'] >0]

    plot_data_cf = plot_data[plot_data['Type'] == 'CF']
    plot_data_supdem = plot_data[(plot_data['Type'] != 'CF') & (plot_data['Technology'] != 'total')]

    st.subheader('Demand and non-dispatchable supply')
    chart = alt.Chart(plot_data_supdem).mark_bar().encode(
        x=alt.X('value:Q', title='Energy (TWh)'),
        y=alt.Y('Type:N', title=None),
        color='Technology:N',
        row='index:N'
    ).interactive()
    st.altair_chart(chart, theme="streamlit")

    st.subheader('Capacity Factors')
    chart = alt.Chart(plot_data_cf).mark_bar().encode(
        x=alt.X('value:Q', title='Capacity Factor'),
        y=alt.Y('index:N', title=None),
        color='Technology:N',
        row='Technology:N'
    ).interactive()
    st.altair_chart(chart, theme="streamlit")