import pandas as pd
from pathlib import  Path
import h5py
from src.result_management.read_results import *

year = 2030


if year == 2030:
    dir = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/baseline_demand_v6/Summary - Copy.xlsx")
    dir_processed = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/baseline_demand_v6/Summary_processed_including_tec_costs"
                         ".xlsx")
    h2_emissions = 29478397.12
    h2_production_cost_smr = 48.64
    h2_cost_total = 1.33E+10
elif year == 2040:
    dir = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/2040_demand_v6_simplifiedgrids/Summary - Copy.xlsx")
    dir_processed = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/2040_demand_v6_simplifiedgrids/Summary_processed_including_tec_costs.xlsx")
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
imports_dict = {}
export_dict = {}
curtailment_dict = {}
generic_production_dict = {}
tec_output_dict = {}
demand_dict = {}
netw_dict_s = {}
netw_dict_c = {}
tec_dict_s = {}
tec_dict_c = {}
for idx, row in df.iterrows():
    case_path = df.loc[idx, "time_stamp"]

    print(case_path)
    # Read data
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5group(hdf_file["operation/energy_balance"])
    df_case = df_case.sum()

    # Imports
    imports_df = df_case.loc[:, 'electricity', 'import']
    prefixed_dict = add_prefix_to_keys(imports_df.to_dict(), 'import_')
    prefixed_dict['import_total'] = imports_df.sum()
    imports_dict[case_path] = prefixed_dict

    # Exports
    export_df = df_case.loc[:, 'hydrogen', 'export']
    prefixed_dict = add_prefix_to_keys(export_df.to_dict(), 'export_')
    prefixed_dict['export_total'] = export_df.sum()
    export_dict[case_path] = prefixed_dict

    # Curtailment
    generic_production_df = df_case.loc[:, 'electricity', 'generic_production']
    curtailment_df = max_re - generic_production_df

    prefixed_dict = add_prefix_to_keys(curtailment_df.to_dict(), 'curtailment_')
    prefixed_dict['curtailment_total'] = curtailment_df.sum()
    curtailment_dict[case_path] = prefixed_dict

    prefixed_dict = add_prefix_to_keys(generic_production_df.to_dict(), 'generic_production_')
    prefixed_dict['generic_production_total'] = generic_production_df.sum()
    generic_production_dict[case_path] = prefixed_dict

    # Demand
    demand_df = df_case.loc[:, 'electricity', 'demand']
    prefixed_dict = add_prefix_to_keys(demand_df.to_dict(), 'demand_')
    prefixed_dict['demand_total'] = demand_df.sum()
    demand_dict[case_path] = prefixed_dict

    # Technology operation
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5group(hdf_file["operation/technology_operation"])
    df_case = df_case.sum()
    tec_output = df_case.loc[:, :, 'electricity_output']
    tec_output = tec_output.groupby(level=1).sum()
    tec_output_dict[case_path] = tec_output.to_dict()

    # Network Sizes
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5group(hdf_file["design/networks"])
    df_case = df_case.T
    df_sizes = df_case.groupby(level=[0, 2]).sum()
    netw_s = {}
    netw_c = {}
    networks = list(set(df_case.index.get_level_values(0)))
    for netw in networks:
        netw_s[netw] = df_sizes.loc[(netw, "size")].values[0]/2
        netw_c[netw +"_cost"] = (df_sizes.loc[(netw, "capex")].values[0] +
                                df_sizes.loc[(netw, "opex_variable")].values[0] +
                                df_sizes.loc[(netw, "opex_fixed")].values[0])/2
    netw_dict_s[case_path] = netw_s
    netw_dict_c[case_path] = netw_c

    # Technology sizes
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5group(hdf_file["design/nodes"])
    df_case = df_case.T
    df_sizes = df_case.groupby(level=[1, 2]).sum()
    tec_s = {}
    tec_c = {}
    technologies = list(set(df_case.index.get_level_values(1)))
    for tec in technologies:
        tec_s[tec + "_size"] = df_sizes.loc[(tec, "size")].values[0]
        tec_c[tec + "_cost"] = (df_sizes.loc[(tec, "capex")].values[0] +
                                df_sizes.loc[(tec, "opex_variable")].values[0] +
                                df_sizes.loc[(tec, "opex_fixed")].values[0])
    tec_dict_s[case_path] = tec_s
    tec_dict_c[case_path] = tec_c

# Merge all
imports_df_all = pd.DataFrame.from_dict(imports_dict, orient='index')
export_df_all = pd.DataFrame.from_dict(export_dict, orient='index')
curtailment_all = pd.DataFrame.from_dict(curtailment_dict, orient='index')
generic_production_all = pd.DataFrame.from_dict(generic_production_dict, orient='index')
tec_output_all = pd.DataFrame.from_dict(tec_output_dict, orient='index')
netw_all_s = pd.DataFrame.from_dict(netw_dict_s, orient='index')
netw_all_c = pd.DataFrame.from_dict(netw_dict_c, orient='index')
tec_all_s = pd.DataFrame.from_dict(tec_dict_s, orient='index')
tec_all_c = pd.DataFrame.from_dict(tec_dict_c, orient='index')
demand_all = pd.DataFrame.from_dict(demand_dict, orient='index')

df = df.set_index('time_stamp')
df_appended = pd.merge(df, curtailment_all, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, imports_df_all, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, export_df_all, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, generic_production_all, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, tec_output_all, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, demand_all, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, netw_all_s, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, netw_all_c, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, tec_all_s, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, tec_all_c, right_index=True, left_index=True)

df_appended['h2_emissions'] = h2_emissions - df_appended['export_total']* 0.108
df_appended['total_emissions'] = (df_appended['positive_emissions'] +
                                  df_appended['h2_emissions'])
df_appended['electricity_emissions'] = (df_appended['total_emissions'] -
                                        df_appended['h2_emissions'])

df_appended['hydrogen_costs_smr'] = (h2_cost_total - h2_production_cost_smr *
                                     df_appended['export_total'])
df_appended['total_costs_with_smr'] = (df_appended['hydrogen_costs_smr'] +
                                       df_appended['total_costs'])
df_appended['electricity_costs'] = df_appended['total_costs_with_smr'] - df_appended['hydrogen_costs_smr']
df_appended = df_appended.set_index(['Case', 'Subcase'])
df_appended.to_excel(dir_processed, merge_cells=False)
# df = df.set_index(['Case', 'Subcase'])
# df.to_excel(dir_processed, merge_cells=False)


