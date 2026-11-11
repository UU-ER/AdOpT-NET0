import pandas as pd
from pathlib import Path


def load_household_data():
    path = Path(__file__).parent
    path = path / "data/data_household.xlsx"
    return pd.read_excel(path, header=0, nrows=8760)


def load_network_city_data():
    path = Path(__file__).parent
    path = path / "data/data_network_city.xlsx"
    return pd.read_excel(path, header=0, nrows=8760)


def load_network_rural_data():
    path = Path(__file__).parent
    path = path / "data/data_network_rural.xlsx"
    return pd.read_excel(path, header=0, nrows=8760)


def load_multiyear_cluster_data():
    path = Path(__file__).parent
    path = path / "data/multiyear_cluster/data_multiyear_cluster.xlsx"
    return pd.read_excel(path, sheet_name=["timeseries", "carrier_constants"])
