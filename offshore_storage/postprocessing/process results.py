import pandas as pd
from pathlib import  Path
import h5py
from adopt_net0 import extract_datasets_from_h5group, extract_dataset_from_h5

dir = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/StorageOffshore"
           "/CapexOptimization/Summary - Copy.xlsx")
dir_processed = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/StorageOffshore"
           "/CapexOptimization/Summary_processed.xlsx")

all_technologies = {
    "onshore_Storage_Battery_CapexOptimization": ["onshore", "Battery"],
    "onshore_Storage_CAES_CapexOptimization": ["onshore", "CAES"],
    "onshore_Electrolyzer": ["onshore", "Electrolyzer"],
    "offshore_Storage_Battery_CapexOptimization": ["offshore", "Battery"],
    "offshore_Storage_CAES_CapexOptimization": ["offshore", "CAES"],
    "offshore_Storage_OceanBattery_CapexOptimization": ["offshore", "OceanBattery"],
    "offshore_Electrolyzer": ["offshore", "Electrolyzer"],
}

sizes = {
    "Battery": 7,
    "CAES": 3000,
    "Electrolyzer": 1,
    "OceanBattery": 7.5

}

def map_timestamp(timestamp, idx):
    for key, value in all_technologies.items():
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

df['Node'] = df['time_stamp'].apply(lambda x: map_timestamp(x, 0))
df['Technology'] = df['time_stamp'].apply(lambda x: map_timestamp(x, 1))
df['SS'] = df['case'].str.extract(r'SS_(\d+\.?\d*)').astype(float)
df['OS'] = df['case'].str.extract(r'OS_(\d+\.?\d*)').astype(float)
df['Baseline'] = df['case'].str.contains('BL|baseline')
df['max_specific_capex'] = df.apply(lambda row: row['cost_capex_tecs'] / sizes[row['Technology']], axis=1)


# IMPORTS AND CURTAILMENT
# max_re = pd.read_csv('C:/Users/6574114/PycharmProjects/PyHubProductive/mes_north_sea/clean_data/production_profiles_re/production_profiles_re.csv', index_col=0, header=[0,1])
# max_re = max_re.loc[:, (slice(None), 'total')].sum()
# max_re.reset_index(level=1, drop=True, inplace=True)

# # Get nodes and carriers
# with h5py.File(
#     df.loc[df['Case'] == 'Baseline', 'time_stamp'].values[0] + '/optimization_results.h5',
#     'r') as hdf_file:
#     nodes = extract_dataset_from_h5(hdf_file["topology/nodes"])
#     carriers = extract_dataset_from_h5(hdf_file["topology/carriers"])
#

# Calculate Imports and Curtailment
# imports_dict = {}
# export_dict = {}
curtailment_dict_tot = {}
curtailment_dict_frac = {}
generic_production_dict = {}
# tec_output_dict = {}
demand_dict = {}
# netw_dict = {}
# tec_dict = {}
for idx, row in df.iterrows():
    case_path = df.loc[idx, "time_stamp"]

    print(case_path)
    # Read data
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5group(hdf_file["operation/energy_balance"])
    df_case = df_case.sum()

    # Demand
    demand_df = df_case.loc[:, :, 'electricity', 'demand']
    prefixed_dict = {}
    prefixed_dict['demand'] = demand_df.sum()
    demand_dict[case_path] = prefixed_dict

    # Curtailment
    max_re = demand_df.sum() * row["SS"]
    generic_production_df = df_case.loc[:, :, 'electricity', 'generic_production']
    curtailment = max_re - generic_production_df.sum()
    prefixed_dict = {}
    prefixed_dict['curtailment_total'] = curtailment
    curtailment_dict_tot[case_path] = prefixed_dict

    prefixed_dict = {}
    prefixed_dict['curtailment_fraction'] = curtailment / max_re
    curtailment_dict_frac[case_path] = prefixed_dict

    # # Imports
    # imports_df = df_case.loc[:, 'electricity', 'import']
    # prefixed_dict = add_prefix_to_keys(imports_df.to_dict(), 'import_')
    # prefixed_dict['import_total'] = imports_df.sum()
    # imports_dict[case_path] = prefixed_dict
    #
    # # Exports
    # export_df = df_case.loc[:, 'hydrogen', 'export']
    # prefixed_dict = add_prefix_to_keys(export_df.to_dict(), 'export_')
    # prefixed_dict['export_total'] = export_df.sum()
    # export_dict[case_path] = prefixed_dict
    #
    # # Technology operation
    # with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
    #     df_case = extract_datasets_from_h5group(hdf_file["operation/technology_operation"])
    # df_case = df_case.sum()
    # tec_output = df_case.loc[:, :, 'electricity_output']
    # tec_output = tec_output.groupby(level=1).sum()
    # tec_output_dict[case_path] = tec_output.to_dict()
    #
    # # Network Sizes
    # with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
    #     df_case = extract_datasets_from_h5group(hdf_file["design/networks"])
    # df_case = df_case.T
    # df_sizes = df_case.groupby(level=[0, 2]).sum()
    # netw_s = {}
    # networks = list(set(df_case.index.get_level_values(0)))
    # for netw in networks:
    #     netw_s[netw] = df_sizes.loc[(netw, "size")].values[0]/2
    # netw_dict[case_path] = netw_s
    #
    # # Technology sizes
    # with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
    #     df_case = extract_datasets_from_h5group(hdf_file["design/nodes"])
    # df_case = df_case.T
    # df_sizes = df_case.groupby(level=[1, 2]).sum()
    # tec_s = {}
    # technologies = list(set(df_case.index.get_level_values(1)))
    # for tec in technologies:
    #     tec_s[tec + "_size"] = df_sizes.loc[(tec, "size")].values[0]/2
    # tec_dict[case_path] = tec_s

# Merge all
# imports_df_all = pd.DataFrame.from_dict(imports_dict, orient='index')
# export_df_all = pd.DataFrame.from_dict(export_dict, orient='index')
curtailment_tot = pd.DataFrame.from_dict(curtailment_dict_tot, orient='index')
curtailment_frac = pd.DataFrame.from_dict(curtailment_dict_frac, orient='index')
# generic_production_all = pd.DataFrame.from_dict(generic_production_dict, orient='index')
# tec_output_all = pd.DataFrame.from_dict(tec_output_dict, orient='index')
# netw_all = pd.DataFrame.from_dict(netw_dict, orient='index')
# tec_all = pd.DataFrame.from_dict(tec_dict, orient='index')
demand_all = pd.DataFrame.from_dict(demand_dict, orient='index')

df = df.set_index('time_stamp')
df_appended = pd.merge(df, curtailment_tot, right_index=True, left_index=True)
df_appended = pd.merge(df, curtailment_frac, right_index=True, left_index=True)
# df_appended = pd.merge(df_appended, imports_df_all, right_index=True, left_index=True)
# df_appended = pd.merge(df_appended, export_df_all, right_index=True, left_index=True)
# df_appended = pd.merge(df_appended, generic_production_all, right_index=True, left_index=True)
# df_appended = pd.merge(df_appended, tec_output_all, right_index=True, left_index=True)
df_appended = pd.merge(df_appended, demand_all, right_index=True, left_index=True)
# df_appended = pd.merge(df_appended, netw_all, right_index=True, left_index=True)
# df_appended = pd.merge(df_appended, tec_all, right_index=True, left_index=True)
#
# df_appended['h2_emissions'] = h2_emissions - df_appended['export_total']* 0.108
# df_appended['total_emissions'] = (df_appended['positive_emissions'] +
#                                   df_appended['h2_emissions'])
# df_appended['electricity_emissions'] = (df_appended['total_emissions'] -
#                                         df_appended['h2_emissions'])
#
# df_appended['hydrogen_costs_smr'] = (h2_cost_total - h2_production_cost_smr *
#                                      df_appended['export_total'])
# df_appended['total_costs_with_smr'] = (df_appended['hydrogen_costs_smr'] +
#                                        df_appended['total_costs'])
# df_appended['electricity_costs'] = df_appended['total_costs_with_smr'] - df_appended['hydrogen_costs_smr']
# df_appended = df_appended.set_index(['Case', 'Subcase'])
df_appended.to_excel(dir_processed, merge_cells=False)
# df = df.set_index(['Case', 'Subcase'])
# df.to_excel(dir_processed, merge_cells=False)


