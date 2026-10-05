import pandas as pd
from src.result_management.read_results import *


def add_prefix_to_keys(dictionary, prefix):
    new_dict = {}
    for key, value in dictionary.items():
        new_key = prefix + key
        new_dict[new_key] = value
    return new_dict

folder_path = 'baseline_demand'

summary_results = pd.read_excel('./src/case_offshore_storage/visualization_v2/data/Summary_Plotting6.xlsx')

# Normalization
baseline_costs = summary_results.loc[summary_results['Case'] == 'Baseline', 'total_costs'].values[0]

# Get nodes and carriers
with h5py.File(
    summary_results.loc[summary_results['Case'] == 'Baseline', 'time_stamp'].values[0] + '/optimization_results.h5',
    'r') as hdf_file:
    nodes = extract_dataset_from_h5(hdf_file["topology/nodes"])
    carriers = extract_dataset_from_h5(hdf_file["topology/carriers"])

# IMPORTS AND CURTAILMENT
max_re = pd.read_csv(
    'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/clean_data/production_profiles_re/production_profiles_re.csv',
    index_col=0, header=[0, 1])
max_re = max_re.loc[:, (slice(None), 'total')].sum()
max_re.reset_index(level=1, drop=True, inplace=True)

# summary_results = summary_results.iloc[0:3]

for index, row in summary_results.iterrows():
    print(row['time_stamp'])

    # Calculate total costs

    with h5py.File(row['time_stamp'] + '/optimization_results.h5', 'r') as hdf_file:
        networks = extract_datasets_from_h5group(hdf_file["design/networks"])
        technologies = extract_datasets_from_h5group(hdf_file["design/nodes"])
        energybalance = extract_datasets_from_h5group(hdf_file["operation/energy_balance"])
        summary = extract_datasets_from_h5group(hdf_file["summary"])

    # Network - Costs
    networks.columns.names = ['Network', 'Arc', 'Variable']
    networks.index = ['Values']
    networks = networks.T.reset_index()
    networks = networks[networks['Variable'].isin(['capex', 'opex_fixed', 'opex_variable'])].drop(columns=['Arc'])
    networks['Type'] = networks['Network'].str.contains('existing').map({True: 'existing', False: 'new'})
    networks['Bidirectional'] = networks['Network'].str.contains('electricity').map({True: True, False: False})
    networks.loc[networks['Bidirectional'], 'Values'] /= 2
    networks.drop(columns = ['Bidirectional'], inplace=True)
    networks = networks.groupby(['Type', 'Network', 'Variable']).sum()
    networks = networks.groupby('Type').sum()
    if 'new' not in networks.index:
        networks.loc['new', :] = 0

    # Technology - Costs
    technologies.columns.names = ['Node', 'Technology', 'Variable']
    technologies = technologies.T.reset_index()
    technologies_cost = technologies[technologies['Variable'].isin(['capex', 'opex_fixed', 'opex_variable'])].drop(columns=['Node'])
    technologies_cost['Type'] = technologies_cost['Technology'].str.contains('existing').map({True: 'existing', False: 'new'})
    technologies_cost = technologies_cost.groupby(['Type', 'Technology', 'Variable']).sum()
    technologies_cost = technologies_cost.groupby('Type').sum()
    if 'new' not in technologies_cost.index:
        technologies_cost.loc['new', :] = 0

    # Technology - Sizes
    technologies_size = technologies[technologies['Variable'].isin(['size'])].drop(columns=['Node', 'Variable'])
    technologies_size = technologies_size.groupby('Technology').sum()
    technologies_size.columns = ['Size']
    # Import - Costs
    energybalance = energybalance.sum()
    imports_el = energybalance.loc[:, 'electricity', 'import'].sum()
    imports_gas = energybalance.loc[:, 'gas', 'import'].sum()
    export_hydrogen = energybalance.loc[:, 'hydrogen', 'export'].sum()

    # Total costs
    total_costs = summary.loc[0, 'total_costs'].values[0]

    # Imports
    imports_df = energybalance.loc[:, 'electricity', 'import']
    total_imports = imports_df.sum()

    # Curtailment
    generic_production_df = energybalance.loc[:, 'electricity', 'generic_production']
    curtailment_df = max_re - generic_production_df

    curtailment_total = curtailment_df.sum()

    prefixed_dict = add_prefix_to_keys(generic_production_df.to_dict(), 'generic_production_')
    generic_production_total = generic_production_df.sum()

    summary_results.at[index, 'Carbon Costs'] = summary.loc[0, 'carbon_costs'].values[0]
    summary_results.at[index, 'Electricity Import Costs'] = imports_el*1000
    summary_results.at[index, 'Hydrogen Exports'] = export_hydrogen
    summary_results.at[index, 'Hydrogen Export Revenues'] = export_hydrogen*(40+80*0.18)
    summary_results.at[index, 'Network Costs (existing)'] = networks.loc['existing', :].values[0]
    summary_results.at[index, 'Network Costs (new)'] = networks.loc['new', :].values[0]
    summary_results.at[index, 'Technology Costs (existing)'] = technologies_cost.loc['existing', :].values[0] + imports_gas*40
    summary_results.at[index, 'Technology Costs (new)'] = technologies_cost.loc['new', :].values[0]
    summary_results.at[index, 'Total Costs (check)'] = (summary_results.at[index, 'Carbon Costs'] +
                                                        summary_results.at[index, 'Electricity Import Costs'] -
                                                        summary_results.at[index, 'Hydrogen Export Revenues'] +
                                                        summary_results.at[index, 'Network Costs (new)'] + summary_results.at[index, 'Network Costs (existing)']  +
                                                        summary_results.at[index, 'Technology Costs (existing)'] + summary_results.at[index, 'Technology Costs (new)']
                                                        )
    summary_results.at[index, 'Total Imports'] = total_imports
    summary_results.at[index, 'Total Curtailment'] = curtailment_total
    summary_results.at[index, 'Total RE Generation'] = generic_production_total

    with h5py.File(row['time_stamp'] + '/optimization_results.h5', 'r') as hdf_file:
        df = extract_datasets_from_h5group(hdf_file["operation/technology_operation"])
    df = df.sum()
    tec_output = df.loc[:, :, 'electricity_output']
    tec_output = tec_output.groupby(level=1).sum()
    for key, v in tec_output.items():
        summary_results.at[index, key + '_el_output'] = v

    try:
        h2_output = df.loc[:, :, 'hydrogen_output']
        h2_output = h2_output.groupby(level=1).sum()
        for key, v in h2_output.items():
            summary_results.at[index, key + '_h2_output'] = v
    except KeyError:
        pass

    h2_input_to_gt = df.loc[:, 'PowerPlant_Gas_existing', 'hydrogen_input']
    h2_input_to_gt = h2_input_to_gt.sum()
    summary_results.at[index, 'hydrogen_input_gt'] = h2_input_to_gt

    for idx, row in technologies_size.iterrows():
        summary_results.at[index, idx + '_size'] = row['Size']


    summary_results.to_excel('./src/case_offshore_storage/visualization_v2/data/Summary_Plotting6_processed.xlsx')



