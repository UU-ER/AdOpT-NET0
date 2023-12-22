import streamlit as st
from .installed_capacities import *
from .node_definition import *
from .profiles import *

def show_page_input_data(sub_page):

    st.header(sub_page)

    categories = {'Installed Capacities':
                  ['Compare installed capacities per country from sources',
                   'Installed Capacities per node'],
                  'Renewable generation profiles and demand':
                  ['Aggregated (total)', 'Aggregated (per country)', 'Per node']}

    if sub_page == 'Installed Capacities':
        category = st.selectbox("Select an option:", categories[sub_page])
        if category == 'Compare installed capacities per country from sources':
            compare_national_capacities()
        elif category == 'Installed Capacities per node':
            show_nodal_capacities()


    elif sub_page == 'Node Definition':
        show_node_definition()

    elif sub_page == 'Renewable generation profiles and demand':
        category = st.selectbox("Select an option:", categories[sub_page])
        show_profiles(category)



