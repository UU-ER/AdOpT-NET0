import streamlit as st
from .installed_capacities import *
from .node_definition import *
from .profiles import *
from .annual_figures import *

def show_page_input_data(sub_page):

    st.header(sub_page)

    categories = {'Installed Capacities':
                  ['Compare installed capacities per country from sources',
                   'Installed Capacities per node'],
                  'Renewable generation profiles and demand':
                  ['Aggregated (total)', 'Aggregated (per country)', 'Per node']}

    # Installed Capacities
    if sub_page == 'Installed Capacities':
        category = st.selectbox("Select an option:", categories[sub_page])
        if category == 'Compare installed capacities per country from sources':
            compare_national_capacities()
        elif category == 'Installed Capacities per node':
            show_nodal_capacities()

    # Node definitions (map)
    elif sub_page == 'Node Definition and networks':
        show_node_definition()

    # Renewable Profiles and Demand
    elif sub_page == 'Renewable generation profiles and demand':
        category = st.selectbox("Select an option:", categories[sub_page])
        show_profiles(category)

    elif sub_page == 'Annual renewable generation and demand':
        show_annual_figures()





