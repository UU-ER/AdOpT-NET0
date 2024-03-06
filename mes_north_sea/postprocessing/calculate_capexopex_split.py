import pandas as pd
from src.result_management.read_results import *


def add_prefix_to_keys(dictionary, prefix):
    new_dict = {}
    for key, value in dictionary.items():
        new_key = prefix + key
        new_dict[new_key] = value
    return new_dict

folder_path = 'baseline_demand'

summary_results = pd.read_excel('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/' + folder_path + '/Summary_Plotting.xlsx')

# Normalization
baseline_costs = summary_results.loc[summary_results['Case'] == 'Baseline', 'total_costs'].values[0]

# Get nodes and carriers
with h5py.File(
    summary_results.loc[summary_results['Case'] == 'Baseline', 'time_stamp'].values[0] + '/optimization_results.h5',
    'r') as hdf_file:
    nodes = extract_dataset_from_h5(hdf_file["topology/nodes"])
    carriers = extract_dataset_from_h5(hdf_file["topology/carriers"])

# print_h5_tree(summary_results.loc[summary_results['Case'] == 'Baseline', 'time_stamp'].values[0] + '/optimization_results.h5')

# Get paths to results
paths = {}
for case in summary_results['Case'].unique():
    paths[case] = list(summary_results.loc[summary_results['Case'] == case, 'time_stamp'].values)

# Calculate total costs
costs = {}
for case in paths:
    for point in paths[case]:
        print(point)
        with h5py.File(point + '/optimization_results.h5', 'r') as hdf_file:
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
        technologies = technologies[technologies['Variable'].isin(['capex', 'opex_fixed', 'opex_variable'])].drop(columns=['Node'])
        technologies['Type'] = technologies['Technology'].str.contains('existing').map({True: 'existing', False: 'new'})
        technologies = technologies.groupby(['Type', 'Technology', 'Variable']).sum()
        technologies = technologies.groupby('Type').sum()
        if 'new' not in technologies.index:
            technologies.loc['new', :] = 0

        # Import - Costs
        energybalance = energybalance.sum()
        imports_el = energybalance.loc[:, 'electricity', 'import'].sum()
        imports_gas = energybalance.loc[:, 'gas', 'import'].sum()
        export_hydrogen = energybalance.loc[:, 'hydrogen', 'export'].sum()

        import_cost_el = imports_el*1000
        import_cost_gas = imports_gas*40
        export_revenue_hydrogen = export_hydrogen*(40+80*0.18)

        # Carbon costs
        carbon_costs = summary.loc[0, 'carbon_costs'].values[0]

        # Total costs
        total_costs = summary.loc[0, 'total_costs'].values[0]

        cost = {}
        cost['Total Costs'] = total_costs
        cost['Carbon Costs'] = carbon_costs

        cost['Network Costs (existing)'] = networks.loc['existing', :].values[0]
        cost['Network Costs (new)'] = networks.loc['new', :].values[0]

        cost['Technology Costs (existing)'] = technologies.loc['existing', :].values[0] + import_cost_gas
        cost['Technology Costs (new)'] = technologies.loc['new', :].values[0]
        cost['total_costs_check'] = carbon_costs + \
                                    import_cost_el - \
                                    export_revenue_hydrogen + \
                                    cost['Network Costs (existing)'] + cost['Network Costs (new)'] + \
                                    cost['Technology Costs (existing)'] + cost['Technology Costs (new)']

        cost = pd.Series(cost)

        costs[point] = cost

costs = pd.DataFrame.from_dict(costs, orient='index')
costs.to_excel('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/results/' + folder_path + '_cost_comparison.xlsx')



