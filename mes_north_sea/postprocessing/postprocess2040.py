import warnings
import os
import numpy as np

import pandas as pd
from pathlib import Path
import concurrent.futures
import h5py

from mes_north_sea.postprocessing.utilities import map_timestamp, extract_datasets_from_h5_group


def process_row(idx_row):
    idx, row = idx_row
    case_path = row["time_stamp"]
    print(case_path)

    data_dict = {}

    data_dict[("global", "global", "Case")] = row["Case"]
    data_dict[("global", "global", "Subcase")] = row["Subcase"]
    data_dict[("global", "global", "objective")] = row["objective"]
    data_dict[("global", "global", "cy")] = row["climate_year"]
    data_dict[("global", "global", "Path")] = row["time_stamp"]
    data_dict[("global", "global", "total_costs")] = row["total_npv"]
    data_dict[("global", "global", "emissions_net")] = row["emissions_net"]
    data_dict[("global", "global", "carbon_costs")] = row["carbon_cost"]
    data_dict[("global", "global", "carbon_tax")] = row["carbon_tax"]
    data_dict[("global", "global", "variable_h2_demand")] = row["variable_h2_demand"]


    h2_emissions = row["h2_emissions"]
    h2_production_cost_smr = row["h2_production_cost_smr"]
    h2_cost_total = row["h2_cost_total"]
    car_costs = {'gas': 40,
                     'electricity': 1000,
                     'hydrogen': 40 + row["carbon_tax"] * 0.108
                                }
    baseline_costs = row['baseline_costs']
    baseline_emissions = row['baseline_emissions']

    # Not valid for 2040
    # max_re = pd.read_csv(
    #     'C:/Users/6574114/PycharmProjects/PyHubProductive/mes_north_sea/clean_data/production_profiles_re/production_profiles_re' + str(row["climate_year"]) + '.csv',
    #     index_col=0, header=[0, 1])
    # max_re = max_re.loc[:, (slice(None), 'total')].sum().sum()

    # Networks
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5_group(hdf_file["design/networks"])
    df_case = pd.DataFrame(df_case).T.reset_index()
    df_case.columns = ["period", "network", "arc", "variable", "value"]
    df_case = df_case[["network", "variable", "value"]]
    df_case = df_case[df_case["variable"].isin(["capex", "opex_fixed", "opex_variable", "size"])]

    df_sizes = df_case.groupby(["network", "variable"]).sum()
    networks = list(set(df_sizes.index.get_level_values(0)))
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
        data_dict[("netw_size", netw, "size")] = df_sizes.loc[(netw, "size")].values[0] / f
    data_dict[("global", "global", "netw_cost_existing")] = cost_existing_networks
    data_dict[("global", "global", "netw_cost_new")] = cost_new_networks

    # Technology Design
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5_group(hdf_file["design/nodes"])
    df_case = pd.DataFrame(df_case).T.reset_index()
    df_case.columns = ["period", "node", "technology", "variable", "value"]
    df_case = df_case[["technology", "variable", "value"]]
    df_case = df_case[df_case["variable"] != "technology"]

    df_sizes = df_case.groupby(["technology", "variable"]).sum()
    technologies = list(set(df_sizes.index.get_level_values(0)))
    cost_existing_tecs = 0
    cost_new_tecs = 0
    for tec in technologies:
        data_dict[("tec_cost", tec, "total_cost")] = (
                df_sizes.loc[(tec, "capex_tot")].values[0] +
                df_sizes.loc[(tec, "opex_fixed")].values[0] +
                df_sizes.loc[(tec, "opex_variable")].values[0]
        )
        if "existing" in tec:
            cost_existing_tecs += data_dict[("tec_cost", tec, "total_cost")]
        else:
            cost_new_tecs += data_dict[("tec_cost", tec, "total_cost")]
        data_dict[("tec_sizes", tec, "size")] = df_sizes.loc[(tec, "size")].values[0]
    data_dict[("global", "global", "tec_cost_existing")] = cost_existing_tecs
    data_dict[("global", "global", "tec_cost_new")] = cost_new_tecs

    # Energybalance
    with h5py.File(case_path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5_group(hdf_file["operation/energy_balance"])
    df_sum = pd.DataFrame(df_case).sum().groupby(level=[2,3]).sum()

    data_dict[("global", "electricity", "generic_production")] = (df_sum.loc[("electricity", "generic_production")])
    data_dict[("global", "electricity", "demand")] = (df_sum.loc[("electricity", "demand")])
    # data_dict[("global", "electricity", "curtailment")] = max_re - data_dict[("global", "electricity", "generic_production")]


    #costs
    carriers = list(set(df_sum.index.get_level_values(0)))
    for car in carriers:
        data_dict[("global", car, "import_cost")] = (df_sum.loc[(car, "import")] *
                                                     car_costs[car])
        data_dict[("global", car, "export_cost")] = (df_sum.loc[(car, "export")] *
                                                     car_costs[car])
        data_dict[("global", car, "import")] = (df_sum.loc[(car, "import")])
        data_dict[("global", car, "export")] = (df_sum.loc[(car, "export")])



    if "hydrogen" not in carriers:
        data_dict[("global", "hydrogen", "import_cost")] = 0
        data_dict[("global", "hydrogen", "export_cost")] = 0
        data_dict[("global", "hydrogen", "import")] = 0
        data_dict[("global", "hydrogen", "export")] = 0
        data_dict[("global", "final", "hydrogen_costs_smr")] = h2_cost_total
    else:

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
        h2_emissions - data_dict[("global", "hydrogen", "export")] * 0.108
    )

    #
    #     (
    #     data_dict[("global", "final", "hydrogen_costs_smr")] * 0.108 / h2_production_cost_smr
    # )
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

    return data_dict

if __name__ == "__main__":


    #
    result_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2040")
    # result_path["emissions"] = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2030/emission_reduction")
    save_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2040")
    cys = [1995, 2008, 2009]
    # #
    # # Get summary from all h5 files in dir
    # summary_ls = []
    # for dirpath, dirnames, filenames in os.walk(result_path):
    #     # exclude "old" directories
    #     dirnames[:] = [d for d in dirnames if d != "old"]
    #
    #     for file in filenames:
    #         if file.endswith(".h5"):
    #             full_path = os.path.join(dirpath, file)
    #             print(full_path)
    #             with h5py.File(full_path, 'r') as hdf_file:
    #                 data = extract_datasets_from_h5_group(hdf_file["summary"])
    #
    #
    #                 def flatten_value(val):
    #                     val = val[0]  # take the first element of the list
    #                     if isinstance(val, np.generic):  # NumPy scalar
    #                         return val.item()
    #                     elif isinstance(val, bytes):  # decode bytes
    #                         return val.decode('utf-8')
    #                     else:
    #                         return val  # already string or number
    #
    #                 flattened = {k[0]: flatten_value(v) for k, v in data.items()}
    #
    #                 summary_ls.append(pd.DataFrame(flattened, index=[0]))
    #
    # summary = pd.concat(summary_ls, ignore_index=True)
    # summary["climate_year"] = summary["time_stamp"].str.extract(r"cy(\d+)", expand=False).astype(int)
    # summary["carbon_tax"] = summary["case"].str.extract(r"co2_tax(\d+)", expand=False).astype(int)
    #
    # exclude_cases_first_stage = [
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250630101533_ElectricityGrid_all_costs_cy1995_co2_tax100-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250701211222_ElectricityGrid_all_costs_cy2008_co2_tax100-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250702025404_ElectricityGrid_all_costs_cy2009_co2_tax100-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250702103702_ElectricityGrid_all_costs_cy1995_co2_tax200-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250704070031_ElectricityGrid_all_costs_cy2008_co2_tax200-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250705021311_ElectricityGrid_all_costs_cy2009_co2_tax200-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250715122641_ElectricityGrid_all_costs_cy1995_co2_tax100-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250715233757_ElectricityGrid_all_costs_cy2008_co2_tax100-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250716120540_ElectricityGrid_all_costs_cy2009_co2_tax100-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250623161922_Hydrogen_H4_costs_cy1995_co2_tax100-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250624081939_Hydrogen_H4_costs_cy2008_co2_tax100-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250624182018_Hydrogen_H4_costs_cy2009_co2_tax100-1'
    # ]
    #
    # summary = summary[~summary["time_stamp"].isin(exclude_cases_first_stage)]
    # summary["variable_h2_demand"] = 0
    #
    # cases_var_h2_demand = [
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250715171505_All_costs_cy1995_co2_tax100-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250716051718_All_costs_cy2008_co2_tax100-1',
    #     r'\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250716182638_All_costs_cy2009_co2_tax100-1'
    # ]
    # summary["variable_h2_demand"] = 0
    # summary.loc[summary["time_stamp"].isin(cases_var_h2_demand), "variable_h2_demand"] = 1
    #
    # summary["h2_emissions"] = 81796113.3
    # summary["h2_production_cost_smr"] = 48.64
    # summary["h2_cost_total"] = 3.68E+10
    # summary['Case'] = summary['time_stamp'].apply(lambda x: map_timestamp(x, 0))
    # summary['Subcase'] = summary['time_stamp'].apply(lambda x: map_timestamp(x, 1))
    #
    # for cy in cys:
    #     summary.loc[(summary['climate_year'] == cy) & (summary['carbon_tax'] == 100), 'baseline_costs'] = summary.loc[(summary['Case'] == 'Baseline') & (summary['climate_year'] == cy) & (summary['carbon_tax'] == 100), 'total_npv'].values[0]
    #     summary.loc[(summary['climate_year'] == cy) & (summary['carbon_tax'] == 200), 'baseline_costs'] = summary.loc[(summary['Case'] == 'Baseline') & (summary['climate_year'] == cy) & (summary['carbon_tax'] == 200), 'total_npv'].values[0]
    #
    #     summary.loc[(summary['climate_year'] == cy) & (summary['carbon_tax'] == 100), 'baseline_emissions'] = summary.loc[(summary['Case'] == 'Baseline') & (summary['climate_year'] == cy) & (summary['carbon_tax'] == 100), 'emissions_net'].values[0]
    #     summary.loc[(summary['climate_year'] == cy) & (summary['carbon_tax'] == 200), 'baseline_emissions'] = summary.loc[(summary['Case'] == 'Baseline') & (summary['climate_year'] == cy) & (summary['carbon_tax'] == 200), 'emissions_net'].values[0]
    #
    # summary.to_excel(save_path / "Summary.xlsx")

    summary = pd.read_excel(save_path / "Summary.xlsx", index_col=0, header=[0])

    def classify_text(s):
        if '_minE_' in s:
            return 'min emissions'
        elif '_minCost_at_' in s:
            return 'min cost at emission limit'
        elif '_costs_' in s:
            return 'min costs'
        else:
            return 'other'

    summary["objective"] = summary['case'].apply(classify_text)

    results = []
    # for row in summary.iterrows():
    #     print(row)
    #     results.append(process_row(row))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_row, summary.iterrows()))

    results = pd.DataFrame(results)

    # data = pd.read_excel(result_path_cost / "Summary_processed.xlsx", index_col=0)
    results.columns = pd.MultiIndex.from_tuples(results.columns)

    results.to_excel(save_path  / "Summary_processed.xlsx")

    # results1 = pd.read_excel(save_path  / "Summary_processed.xlsx", index_col=0, header=[0,1,2])


