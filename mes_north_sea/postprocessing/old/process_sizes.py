import pandas as pd
from pathlib import  Path
import h5py
from src.result_management.read_results import *

year = 2040


if year == 2030:
    dir = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/baseline_demand_v6/Summary_costs.xlsx")
    dir_processed = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/baseline_demand_v6/Summary_sizes1.xlsx")
    h2_emissions = 29478397.12
    h2_production_cost_smr = 48.64
    h2_cost_total = 1.33E+10
elif year == 2040:
    dir = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/2040_demand_v6_simplifiedgrids/Summary.xlsx")
    dir_processed = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/2040_demand_v6_simplifiedgrids/Summary_sizes1.xlsx")
    h2_emissions = 81796113.3
    h2_production_cost_smr = 48.64
    h2_cost_total = 3.68E+10

scenarios = {
               '20240308120552_Baseline_costs': ['Baseline', 'Baseline'],
              'RE_only': ['Baseline', 'Baseline'],
              'Battery_on': ['Storage', 'onshore only'],
              'Battery_off': ['Storage', 'offshore only'],
              'Battery_all': ['Storage', 'all'],
              'Battery_all_HP': ['Storage', 'all, high power-energy-ratio'],
              'ElectricityGrid_all': ['Grid Expansion', 'all'],
              'ElectricityGrid_on': ['Grid Expansion', 'onshore only'],
              'ElectricityGrid_off': ['Grid Expansion', 'offshore only'],
              'ElectricityGrid_noBorder': ['Grid Expansion', 'no Border'],
              'Hydrogen_Baseline': ['Hydrogen', 'all'],
              'Hydrogen_H1': ['Hydrogen', 'no storage'],
              'Hydrogen_H2': ['Hydrogen', 'no hydrogen offshore'],
              'Hydrogen_H3': ['Hydrogen', 'no hydrogen onshore'],
              'Hydrogen_H4': ['Hydrogen', 'local use only'],
              'All': ['All', 'All']
             }

def map_timestamp(timestamp, idx):
    for key, value in scenarios.items():
        if key in timestamp:
            return value[idx]
    return None  # or some default value if no match is found

def add_prefix_to_keys(dictionary, prefix):
    new_dict = {}
    for key, value in dictionary.items():
        new_key = prefix + key
        new_dict[new_key] = value
    return new_dict


df = pd.read_excel(dir)

df['Case'] = df['time_stamp'].apply(lambda x: map_timestamp(x, 0))
df['Subcase'] = df['time_stamp'].apply(lambda x: map_timestamp(x, 1))

# Normalization
baseline_costs = df.loc[df['Case'] == 'Baseline', 'total_costs'].values[0]
baseline_emissions = df.loc[df['Case'] == 'Baseline', 'net_emissions'].values[0] + h2_emissions

df['normalized_costs'] = df['total_costs'] / baseline_costs
df['normalized_emissions'] = ((df['net_emissions']  + h2_emissions) /
                              baseline_emissions)
df['delta_cost'] = df['total_costs'] - baseline_costs
df['delta_emissions'] = df['net_emissions'] - baseline_emissions
df['abatemente_cost'] = round(df['delta_cost'],0) / round(df['delta_emissions'],0)


# IMPORTS AND CURTAILMENT
max_re = pd.read_csv('/mes_north_sea/clean_data/production_profiles_re/production_profiles_re.csv', index_col=0, header=[0, 1])
max_re = max_re.loc[:, (slice(None), 'total')].sum()
max_re.reset_index(level=1, drop=True, inplace=True)

# Get nodes and carriers
with h5py.File(
    df.loc[df['Case'] == 'Baseline', 'time_stamp'].values[0] + '/optimization_results.h5',
    'r') as hdf_file:
    nodes = extract_dataset_from_h5(hdf_file["topology/nodes"])
    carriers = extract_dataset_from_h5(hdf_file["topology/carriers"])


# Calculate Imports and Curtailment
netw_dict = {}
tec_dict = {}

# Node distances
data_path = ('C:/Users/6574114/PycharmProjects/PyHubProductive/mes_north_sea'
             '/clean_data/networks/')
file_name_ac = 'pyhub_el_ac_all.csv'
if year == 2030:
    file_name_dc = 'pyhub_el_dc_all.csv'
elif year == 2040:
    file_name_dc = 'pyhub_el_dc_all_2040.csv'

network_d_ac = pd.read_csv(data_path + file_name_ac, sep=';')
network_d_dc = pd.read_csv(data_path + file_name_dc, sep=';')
network_d = pd.concat([network_d_ac, network_d_dc])
network_d = network_d.drop_duplicates(subset=["LinePyHub"])
network_d = network_d[["node0", "node1", "length"]]
network_d1 = network_d.copy()
network_d1.columns = ["fromNode", "toNode", "length"]
network_d2 = network_d.copy()
network_d2.columns = ["toNode", "fromNode", "length"]
network_d = pd.concat([network_d1, network_d2])

for idx, row in df.iterrows():
    case_path = df.loc[idx, "time_stamp"]

    print(case_path)
    # Network Sizes
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5group(hdf_file["design/networks"])
    # Convert byte strings to regular strings
    for col in df_case.columns:
        if df_case[col].dtype == object:  # Object dtype can hold byte strings
            df_case[col] = df_case[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    df_case = df_case.T
    df_case.columns = ["Value"]
    df_case = df_case.unstack().reset_index()
    cols = list(df_case.columns.droplevel())
    cols[0] = "Network"
    cols[1] = "ArcID"
    df_case.columns = cols
    df_case = df_case[["Network", "fromNode", "toNode", "size"]]
    df_case = df_case.set_index(["fromNode", "toNode"]).join(network_d.set_index([
        "fromNode", "toNode"]), how="left")
    df_case.loc[df_case['Network'] == 'electricityDC_int', 'size'] *= 1000
    df_case["SizeDistance"] = df_case["length"] * df_case["size"] / 1000
    df_case.reset_index().drop(columns=["fromNode", "toNode", "length", "size"],
                          inplace=True)
    df_sizes = df_case.groupby(["Network"]).sum()

    netw_s = {}
    for netw in df_sizes.index:
        if "electricity" in netw:
            netw_s[netw] = df_sizes.loc[(netw, "SizeDistance")] / 2
        else:
            netw_s[netw] = df_sizes.loc[(netw, "SizeDistance")]
    netw_dict[case_path] = netw_s

    # Technology sizes
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5group(hdf_file["design/nodes"])
    df_case = df_case.T
    df_sizes = df_case.groupby(level=[1, 2]).sum()
    tec_s = {}
    for tec in df_sizes.index.get_level_values(0).unique():
        tec_s[tec + "_size"] = df_sizes.loc[(tec, "size")].values[0]
    tec_dict[case_path] = tec_s

# Merge all
netw_all = pd.DataFrame.from_dict(netw_dict, orient='index')
tec_all = pd.DataFrame.from_dict(tec_dict, orient='index')

df = df.set_index('time_stamp')
df_appended = pd.merge(df, netw_all, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, tec_all, right_index=True, left_index=True)

df_appended = df_appended.set_index(['Case', 'Subcase'])
df_appended.to_excel(dir_processed, merge_cells=False)
# df = df.set_index(['Case', 'Subcase'])
# df.to_excel(dir_processed, merge_cells=False)


