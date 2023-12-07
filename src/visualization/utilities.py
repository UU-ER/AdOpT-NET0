from datetime import datetime
import streamlit as st

def get_boundaries_date(df):
    min_value = None
    max_value = None

    for key, df in df.items():
        # Update min_value and max_value based on the current DataFrame
        current_min = df['Timestep'].min()
        current_max = df['Timestep'].max()

        if min_value is None or current_min < min_value:
            min_value = current_min

        if max_value is None or current_max > max_value:
            max_value = current_max
    return min_value, max_value


def determine_graph_boundaries(df):
    # Determine plotted daterange
    min_date, max_date = get_boundaries_date(df)
    st.sidebar.text("Select x-axis range:")
    x_min = st.sidebar.slider(
        "Starting time: ",
        min_value=datetime.fromtimestamp(min_date.timestamp()),
        max_value=datetime.fromtimestamp(max_date.timestamp()),
        format="DD.MM, HH",
    )
    x_max = st.sidebar.slider(
        "Ending time: ",
        min_value=datetime.fromtimestamp(min_date.timestamp()),
        max_value=datetime.fromtimestamp(max_date.timestamp()),
        format="DD.MM, HH",
    )
    return x_min, x_max
