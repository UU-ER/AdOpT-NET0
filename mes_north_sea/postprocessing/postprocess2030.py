import warnings

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


    h2_emissions = row["h2_emissions"]
    h2_production_cost_smr = row["h2_production_cost_smr"]
    h2_cost_total = row["h2_cost_total"]
    warnings.warn("ONLY WORKS FRO 2030")
    car_costs = {'gas': 40,
                     'electricity': 1000,
                     'hydrogen': 40 + row["carbon_tax"] * 0.108
                                }
    baseline_costs = row['baseline_costs']
    baseline_emissions = row['baseline_emissions']


    max_re = pd.read_csv(
        'C:/Users/6574114/PycharmProjects/PyHubProductive/mes_north_sea/clean_data/production_profiles_re/production_profiles_re' + str(row["climate_year"]) + '.csv',
        index_col=0, header=[0, 1])
    max_re = max_re.loc[:, (slice(None), 'total')].sum().sum()

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
    data_dict[("global", "electricity", "curtailment")] = max_re - data_dict[("global", "electricity", "generic_production")]


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

    result_path = {}
    result_path["cost"] = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2030/cost")
    result_path["emissions"] = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2030/emission_reduction")
    save_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2030")
    cys = [1995, 2008, 2009]

    summary = []
    # load summaries
    for cy in cys:
        df = []
        for r in result_path:
            load_path = result_path[r] / ("00_cy" + str(cy))
            df.append(pd.read_excel(load_path / "Summary.xlsx"))
        s = pd.concat(df)
        s["climate_year"] = cy
        s["h2_emissions"] = 29478397.12
        s["h2_production_cost_smr"] = 48.64
        s["h2_cost_total"] = 1.33E+10
        s["carbon_tax"] = 80
        s['Case'] = s['time_stamp'].apply(lambda x: map_timestamp(x, 0))
        s['Subcase'] = s['time_stamp'].apply(lambda x: map_timestamp(x, 1))
        s['baseline_costs'] = s.loc[s['Case'] == 'Baseline', 'total_npv'].values[0] + s["h2_cost_total"]
        s['baseline_emissions'] = s.loc[s['Case'] == 'Baseline', 'emissions_net'].values[0] + s["h2_emissions"]
        summary.append(s)

    summary = pd.concat(summary)


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
    #     results.append(process_row(row))
    #     print(row)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_row, summary.iterrows()))

    results = pd.DataFrame(results)

    # data = pd.read_excel(result_path_cost / "Summary_processed.xlsx", index_col=0)
    results.columns = pd.MultiIndex.from_tuples(results.columns)

    results.to_excel(save_path  / "Summary_processed_all.xlsx")

    # results1 = pd.read_excel(save_path  / "Summary_processed.xlsx", index_col=0, header=[0,1,2])


