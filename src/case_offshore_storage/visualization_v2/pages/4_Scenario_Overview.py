import pandas as pd
import altair as alt
import streamlit as st
import numpy as np
all_cases = pd.read_excel('src/case_offshore_storage/visualization_v2/data/Summary_Plotting.xlsx')

all_cases['Log_Value'] = np.sign(all_cases['abatemente_cost']) * np.log10(np.abs(all_cases['abatemente_cost']) + 1)


all_cases = all_cases[all_cases['Case'] != 'Baseline']
all_case_pos = all_cases[all_cases['abatemente_cost'] > 0]
all_case_neg = all_cases[all_cases['abatemente_cost'] < 0]
all_case_neg['abatemente_cost'] = all_case_neg['abatemente_cost'] * -1

chart_pos = alt.Chart(all_cases).mark_bar().encode(
    x=alt.X('abatemente_cost', scale=alt.Scale(type='linear', zero=False)),
    y='Case',
    color = 'Case',
    row='Emission_reduction_case'
)

chart_neg = alt.Chart(all_case_neg).mark_bar().encode(
    x=alt.X('abatemente_cost').scale(type="log"),
    y='Case',
    color='Case',
    row='Emission_reduction_case'
)

st.altair_chart(chart_pos)
st.altair_chart(chart_neg)

# chart_energy = alt.Chart(
#     st.session_state['line_data_energy'].reset_index().melt(id_vars=['index'])).mark_line().encode(
#     x=alt.X('Timeslice:Q', axis=alt.Axis(format='d')).title(time_agg_options[time_agg]),
#     y=alt.Y('value').title('Energy (MWh)', titleColor='#57A44C'),
#     color='index',
#     tooltip=['Timeslice', 'value']
# ).properties(
#     width=800,
#     height=400
# ).interactive()