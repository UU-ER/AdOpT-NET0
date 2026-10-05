import numpy as np
import pandas as pd
from pathlib import  Path
import h5py
from src.result_management.read_results import *
import matplotlib.pyplot as plt

year = 2030


if year == 2030:
    dir = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/baseline_demand_v6/Summary_Storage_EmissionReduction.xlsx")
    dir_processed = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/baseline_demand_v6"
                         "/Summary_Storage_EmissionReduction_processed"
                         ".xlsx")
    h2_emissions = 29478397.12
    h2_production_cost_smr = 48.64
    h2_cost_total = 1.33E+10
    carbon_tax = 80


elif year == 2040:
    dir = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/2040_demand_v6_simplifiedgrids/Summary - Copy.xlsx")
    dir_processed = Path("//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES "
                "NorthSea/2040_demand_v6_simplifiedgrids/Summary_processed_costs")
    h2_emissions = 81796113.3
    h2_production_cost_smr = 48.64
    h2_cost_total = 3.68E+10
    carbon_tax = 100

car_costs = {'gas': 40,
                 'electricity': 1000,
                 'hydrogen': 40 + carbon_tax * 0.108
                            }


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
baseline_costs = df.loc[df['Case'] == 'Baseline', 'total_costs'].values[0] + h2_cost_total
baseline_emissions = df.loc[df['Case'] == 'Baseline', 'net_emissions'].values[0] + h2_emissions


# Calculate Imports and Curtailment
data_list = []

for idx, row in df.iterrows():
    case_path = df.loc[idx, "time_stamp"]
    print(case_path)
    data_dict = {}

    data_dict[("global", "global", "Case")] = row["Case"]
    data_dict[("global", "global", "Subcase")] = row["Subcase"]
    data_dict[("global", "global", "Path")] = row["time_stamp"]
    data_dict[("global", "global", "total_costs")] = row["total_costs"]
    data_dict[("global", "global", "emissions_net")] = row["net_emissions"]


    # Carbon Costs
    data_dict[("global", "global", "carbon_costs")] = row["carbon_costs"]

    # Network costs
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5group(hdf_file["design/networks"])
    df_case = df_case.T
    df_sizes = df_case.groupby(level=[0, 2]).sum()
    networks = list(set(df_case.index.get_level_values(0)))
    cost_existing_networks = 0
    cost_new_networks = 0
    for netw in networks:
        if "electricity" in netw:
            f = 2
        else:
            f = 1
        data_dict[("netw_cost", netw, "total_cost")] = (
                df_sizes.loc[(netw, "capex")].values[0] / f +
                df_sizes.loc[(netw, "opex_fixed")].values[0] / f +
                df_sizes.loc[(netw, "opex_variable")].values[0] / f
        )
        if "existing" in netw:
            cost_existing_networks += data_dict[("netw_cost", netw, "total_cost")]
        else:
            cost_new_networks += data_dict[("netw_cost", netw, "total_cost")]
    data_dict[("global", "global", "netw_cost_existing")] = cost_existing_networks
    data_dict[("global", "global", "netw_cost_new")] = cost_new_networks

    # Nodal costs
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5group(hdf_file["design/nodes"])
    df_case = df_case.T
    df_sizes = df_case.groupby(level=[1, 2]).sum()
    technologies = list(set(df_case.index.get_level_values(1)))
    cost_existing_tecs = 0
    cost_new_tecs = 0
    for tec in technologies:
        data_dict[("tec_cost", tec, "total_cost")] = (
                df_sizes.loc[(tec, "capex")].values[0] +
                df_sizes.loc[(tec, "opex_fixed")].values[0] +
                df_sizes.loc[(tec, "opex_variable")].values[0]
        )
        if "existing" in tec:
            cost_existing_tecs += data_dict[("tec_cost", tec, "total_cost")]
        else:
            cost_new_tecs += data_dict[("tec_cost", tec, "total_cost")]
    data_dict[("global", "global", "tec_cost_existing")] = cost_existing_tecs
    data_dict[("global", "global", "tec_cost_new")] = cost_new_tecs

    # Import/Export Costs
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5group(hdf_file["operation/energy_balance"])
    df_sum = df_case.sum().groupby(level=[1,2]).sum()

    carriers = list(set(df_sum.index.get_level_values(0)))
    for car in carriers:
        data_dict[("global", car, "import_cost")] = (df_sum.loc[(car, "import")] *
                                                     car_costs[car])
        data_dict[("global", car, "export_cost")] = (df_sum.loc[(car, "export")] *
                                                     car_costs[car])

    data_dict[("global", "final", "hydrogen_costs_smr")] = (h2_cost_total -
                                                             h2_production_cost_smr *
                                                             df_sum.loc[("hydrogen",
                                                                         "export")])

    data_dict[("global", "final", "cost_existing_system")] = (
        data_dict[("global", "electricity", "import_cost")] +
        data_dict[("global", "gas", "import_cost")] +
        data_dict[("global", "global", "tec_cost_existing")] +
        data_dict[("global", "global", "netw_cost_existing")] +
        data_dict[("global", "global", "carbon_costs")]
    )

    data_dict[("global", "final", "cost_new_system")] = (
        data_dict[("global", "global", "tec_cost_new")] +
        data_dict[("global", "global", "netw_cost_new")]
    )

    data_dict[("global", "final", "cost_total")] = (
        data_dict[("global", "final", "hydrogen_costs_smr")] +
        data_dict[("global", "final", "cost_existing_system")] +
        data_dict[("global", "final", "cost_new_system")]
    )

    data_dict[("global", "final", "emissions_total")] = (
        data_dict[("global", "global", "emissions_net")] +
        h2_emissions
    )
    data_dict[("global", "final", "emissions_smr")] = (
        data_dict[("global", "final", "hydrogen_costs_smr")] * 0.108 / h2_production_cost_smr
    )
    data_dict[("global", "final", "emissions_other")] = (
        data_dict[("global", "final", "emissions_total")] -
        data_dict[("global", "final", "emissions_smr")]
    )
    data_dict[("global", "final", "emission_reduction")] = (
        baseline_emissions - data_dict[("global", "final", "emissions_total")]
    )
    data_dict[("global", "final", "cost_reduction")] = (
        baseline_costs - data_dict[("global", "final", "cost_total")]
    )
    data_dict[("global", "final", "abatement_cost")] = (
        round(data_dict[("global", "final", "cost_reduction")],0) /
        round(data_dict[("global", "final", "emission_reduction")],0)
    )


    data_list.append(data_dict)


df_final = pd.DataFrame.from_dict(data_list)
df_final.columns = pd.MultiIndex.from_tuples(df_final.columns)

df_final.to_excel(dir_processed, merge_cells=True)
df_plot = df_final[df_final[("global", "global", "Path")].str.contains(
    "_costs")].set_index([("global", "global", "Case"), ("global", "global", "Subcase")])

# Plot Costs
df_plot = df_plot[("global", "final")]

df_plot_cost = df_plot[['hydrogen_costs_smr', 'cost_existing_system','cost_new_system']]

y_labels = [str(i) for i in df_plot_cost.index]
bottom = np.zeros(len(df_plot_cost))
fig, ax = plt.subplots()
for idx, values in df_plot_cost.items():
    print(values.values)

    p = ax.bar(y_labels, values.values, bottom=bottom, label= idx)
    bottom += values.values

plt.xticks(rotation=90)
plt.ylim([0,8e10])
plt.tight_layout()
plt.legend()
plt.show()

plt.savefig("C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - "
            "HydrogenOffshore/00_Figures/"+str(year)+"_costs_aux_storage.svg")

# Plot Abatement
df_plot_cost = df_plot['abatement_cost']

y_labels = [str(i) for i in df_plot.index]
bottom = np.zeros(len(df_plot_cost))
fig, ax = plt.subplots()
ax.bar(y_labels, df_plot_cost)

plt.xticks(rotation=90)
plt.ylim([0,300])
plt.tight_layout()
plt.legend()
plt.show()

plt.savefig("C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - "
            "HydrogenOffshore/00_Figures/"+str(year)+"_abatement_aux_storage.svg")

# Plot Emissions
df_plot_cost = df_plot[['emissions_smr', "emissions_other"]]/1000000

y_labels = [str(i) for i in df_plot_cost.index]
bottom = np.zeros(len(df_plot_cost))
fig, ax = plt.subplots()
for idx, values in df_plot_cost.items():
    p = ax.bar(y_labels, values.values, bottom=bottom, label= idx)
    bottom += values.values

plt.xticks(rotation=90)
plt.ylim([0,100])
plt.tight_layout()
plt.legend()
plt.show()

plt.savefig("C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - "
            "HydrogenOffshore/00_Figures/"+str(year)+"_emissions_aux_storage.svg")


