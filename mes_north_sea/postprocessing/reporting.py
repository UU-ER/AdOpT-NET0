import pandas as pd
import h5py
from mes_north_sea.postprocessing.utilities import extract_datasets_from_h5_group
import numpy as np
import matplotlib.pyplot as plt

def get_technology_capacities(path):
    with h5py.File(path + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5_group(hdf_file["design/nodes"])
    df_case = pd.DataFrame(df_case).T.reset_index()
    df_case.columns = ["period", "node", "technology", "variable", "value"]
    df_case = df_case[["technology", "variable", "value"]]
    df_case = df_case[df_case["variable"] != "technology"]

    sizes =  df_case.groupby(["technology", "variable"]).sum()
    sizes = sizes.loc[sizes.index.get_level_values(1) == "size"]
    sizes.index = sizes.index.get_level_values(0)
    return sizes

def get_tec_production(path):
    with h5py.File(path + '/optimization_results.h5', 'r') as hdf_file:
        return extract_datasets_from_h5_group(hdf_file["operation/technology_operation"])

def get_annual_tec_production(path):
    tec_operation_dict = get_tec_production(path)
    return pd.DataFrame(tec_operation_dict).sum().groupby(level=[2, 3]).sum()

def get_storage_capacity(path):
    tec_operation = pd.DataFrame(get_tec_production(path))
    tec_operation_storage = tec_operation.loc[:, ((tec_operation.columns.get_level_values(2).str.startswith(
        "Storage_Battery")) & (tec_operation.columns.get_level_values(3) == "storage_level"))]
    cap = 0
    for col in tec_operation_storage:
        cap = cap + (tec_operation_storage[col].max() - tec_operation_storage[col].min())

    return cap

def aggregate_time(df, level, aggregation = 'sum'):
    if aggregation == 'sum':
        df = df.groupby(level=level).sum()
    elif aggregation == 'mean':
        df = df.groupby(level=level).mean()
    df.index.names = ['Timeslice']
    return df


def add_time_steps_to_df(df):
    """
    Adds time index to a df
    """
    num_rows = len(df)
    hour = np.repeat(np.arange(1, num_rows + 1), 1)[0:num_rows]
    day = np.repeat(np.arange(1, num_rows + 1), 24)[0:num_rows]
    week = np.repeat(np.arange(1, num_rows + 1), 24 * 7)[0:num_rows]
    month = (
        pd.date_range(start="2008-01-01 00:00", end="2008-12-31 00:00", freq="1h")
        .month[0:num_rows]
        .to_list()
    )
    year = np.ones(num_rows)

    df.index = pd.MultiIndex.from_arrays(
        [hour, day, week, month, year], names=["Hour", "Day", "Week", "Month", "Year"]
    )

    return df


results_all = pd.read_excel(
        "//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2030/Summary_processed.xlsx",
        header=[0, 1, 2], index_col=0)

# 2030 curtailment in storage scenarios
# 2030 storage capacities in storage scenarios
cys = [1995, 2008, 2009]

path_storage_all = {}
path_storage_all[1995] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2030\emission_reduction\20250521043736_Battery_all_minCost_at_0.9-1"
path_storage_all[2008] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2030\emission_reduction\20250521084035_Battery_all_minCost_at_0.9-1"
path_storage_all[2009] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2030\emission_reduction\20250521061914_Battery_all_minCost_at_0.9-1"
path_storage_offshore = {}
path_storage_offshore[1995] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2030\emission_reduction\20250520223741_Battery_off_minCost_at_0.9-1"
path_storage_offshore[2008] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2030\emission_reduction\20250521004203_Battery_off_minCost_at_0.9-1"
path_storage_offshore[2009] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2030\emission_reduction\20250520234309_Battery_off_minCost_at_0.9-1"
for cy in cys:
    print(f"{cy}")

    curtailment = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline") & (results_all[("global", "global", "Subcase")] == "Baseline")][
        ("global", "electricity", "curtailment")].values[0]
    generic_production = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline") & (results_all[("global", "global", "Subcase")] == "Baseline")][
        ("global", "electricity", "generic_production")].values[0]
    print(f"Curtailed electricity in Baseline = {round((curtailment)/1000000, 3)}")
    print(f"Curtailed fraction of electricity in Baseline = {round((curtailment)/(generic_production + curtailment), 3)}")

    curtailment = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Storage") & (results_all[("global", "global", "Subcase")] == "all")  & (results_all[("global", "global", "objective")] == "min emissions")][
        ("global", "electricity", "curtailment")].values[0]
    generic_production = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Storage") & (results_all[("global", "global", "Subcase")] == "all")  & (results_all[("global", "global", "objective")] == "min emissions")][
        ("global", "electricity", "generic_production")].values[0]
    print(f"Curtailed electricity in Storage All = {round((curtailment)/1000000, 3)}")
    print(f"Curtailed fraction of electricity in Storage All = {round((curtailment)/(generic_production + curtailment), 3)}")

    cap = get_storage_capacity(path_storage_all[cy])
    print(f"Storage all | Required storage capacity is {cap/1000} GWh")
    cap = get_storage_capacity(path_storage_offshore[cy])
    print(f"Storage offshore only | Required storage capacity is {cap/1000} GWh")

# 2030 exports at countries
for cy in cys:
    print(f"{cy}")

    path_baseline = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline") & (results_all[("global", "global", "Subcase")] == "Baseline")][
        ("global", "global", "Path")].values[0]


    tec_operation_dict = get_tec_production(path_baseline)
    annual_production_baseline = pd.DataFrame(tec_operation_dict).sum().groupby(level=[1, 2, 3]).sum()
    annual_electricity_production = annual_production_baseline.groupby(level=[0,2]).sum()

    countries = annual_electricity_production.index.get_level_values(0).str[:2]
    electricity = annual_electricity_production.loc[annual_electricity_production.index.get_level_values(1) == 'electricity_output']
    electricity.index = electricity.index.get_level_values(0).str[:2]
    electricity_sum = electricity.groupby(electricity.index.get_level_values(0).str[:2]).sum()

    with h5py.File(path_baseline + '/optimization_results.h5', "r") as hdf_file:
        bal = extract_datasets_from_h5_group(hdf_file["operation/energy_balance"])
    balance = pd.DataFrame(bal).sum().groupby(level=[1,3]).sum()
    # Extract all columns whose last level is "generic_production"
    generic_prod = pd.DataFrame(balance).xs('generic_production', axis=0, level=-1)
    countries = [c[:2] for c in generic_prod.index.get_level_values(0)]
    generic_prod_per_country = generic_prod.groupby(countries, axis=0).sum()

    with h5py.File(path_baseline + '/optimization_results.h5', 'r') as hdf_file:
        df_case = extract_datasets_from_h5_group(hdf_file["design/networks"])
    network_design = pd.DataFrame(df_case).melt()
    network_design.columns = ["Period", "Network", "Arc_ID", "Variable", "Value"]
    network_design = network_design.pivot(
        columns="Variable", index=["Period", "Arc_ID", "Network"], values="Value"
    )
    network_design["FromNode"] = network_design["fromNode"].str.decode("utf-8")
    network_design["ToNode"] = network_design["toNode"].str.decode("utf-8")
    network_design["FromCountry"] = network_design["FromNode"].str[0:2]
    network_design["ToCountry"] = network_design["ToNode"].str[0:2]
    network_design.drop(columns=["fromNode", "FromNode", "ToNode", "toNode", "network", 'capex', 'opex_fixed', 'opex_variable', 'para_capex_gamma1',
       'para_capex_gamma2', 'para_capex_gamma3', 'para_capex_gamma4', 'size'], inplace=True)

    imports_exports = network_design[network_design["FromCountry"] != network_design["ToCountry"]]

    countries = network_design["FromCountry"].unique()
    for country in countries:
        imports = imports_exports[imports_exports["ToCountry"] == country]["total_flow"].sum()
        exports = imports_exports[imports_exports["FromCountry"] == country]["total_flow"].sum()
        net_exports = exports - imports
        generation = electricity_sum[country] + generic_prod_per_country.loc[country, :][0]
        print(f"{country} net-exports {np.round(net_exports/generation * 100, 2)}% of its domestic generation")


# 2030 Emission reduction
em_reduction_synergies = []
em_reduction_grids = []
em_reduction_storage = []
em_reduction_hydrogen = []
for cy in cys:
    print(f"{cy}")
    emissions_baseline = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline")][
        ("global", "final", "emissions_total")].values[0]
    emissions_synergies = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "All") & (results_all[("global", "global", "Subcase")] == "All")  & (results_all[("global", "global", "objective")] == "min emissions")][
        ("global", "final", "emissions_total")].values[0]
    em_reduction_synergies.append((emissions_baseline - emissions_synergies) / 1000000)
    print(f"Emission reduction 2030 Reference - Synergies: {(emissions_baseline - emissions_synergies)/1000000}")

    emissions_grids = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Grid Expansion") & (results_all[("global", "global", "Subcase")] == "all")  & (results_all[("global", "global", "objective")] == "min emissions")][
        ("global", "final", "emissions_total")].values[0]
    em_reduction_grids.append((emissions_baseline - emissions_grids) / 1000000)
    print(f"Emission reduction 2030 Reference - Grids: {(emissions_baseline - emissions_grids)/1000000}")

    emissions_storage = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Storage") & (results_all[("global", "global", "Subcase")] == "all")  & (results_all[("global", "global", "objective")] == "min emissions")][
        ("global", "final", "emissions_total")].values[0]
    em_reduction_storage.append((emissions_baseline - emissions_storage) / 1000000)
    print(f"Emission reduction 2030 Reference - Storage: {(emissions_baseline - emissions_storage)/1000000}")

    emissions_hydrogen = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Hydrogen") & (results_all[("global", "global", "Subcase")] == "all")  & (results_all[("global", "global", "objective")] == "min emissions")][
        ("global", "final", "emissions_total")].values[0]
    em_reduction_hydrogen.append((emissions_baseline - emissions_hydrogen) / 1000000)
    print(f"Emission reduction 2030 Reference - Hydrogen: {(emissions_baseline - emissions_hydrogen)/1000000}")

    path_synergies = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "All") & (results_all[("global", "global", "Subcase")] == "All")  & (results_all[("global", "global", "objective")] == "min emissions")][
        ("global", "global", "Path")].values[0]
    annual_production_synergies = get_annual_tec_production(path_synergies)

    print(f"Technology operations: {annual_production_synergies[annual_production_synergies>0]}")

print(f"Average emission reduction 2030 Reference - Synergies: {np.mean(em_reduction_synergies)}")
print(f"Average emission reduction 2030 Reference - Grids: {np.mean(em_reduction_grids)}")
print(f"Average emission reduction 2030 Reference - Storage: {np.mean(em_reduction_storage)}")
print(f"Average emission reduction 2030 Reference - Hydrogen: {np.mean(em_reduction_hydrogen)}")

# 2030 Cost reduction
cost_reduction_synergies = []
cost_reduction_grids = []
cost_reduction_storage = []
cost_reduction_hydrogen = []
for cy in cys:
    print(f"{cy}")
    cost_baseline = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline")][
        ("global", "global", "total_costs")].values[0]
    cost_synergies = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "All") & (results_all[("global", "global", "Subcase")] == "All")  & (results_all[("global", "global", "objective")] == "min costs")][
        ("global", "global", "total_costs")].values[0]
    cost_reduction_synergies.append((cost_baseline - cost_synergies)* 10**-9)
    print(f"Cost reduction 2030 Reference - Synergies: {(cost_baseline - cost_synergies)/10**-9}")

    cost_grids = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Grid Expansion") & (results_all[("global", "global", "Subcase")] == "all")  & (results_all[("global", "global", "objective")] == "min costs")][
        ("global", "global", "total_costs")].values[0]
    cost_reduction_grids.append((cost_baseline - cost_grids)* 10**-9)
    print(f"Cost reduction 2030 Reference - Grids: {(cost_baseline - cost_grids)/10**-9}")

    cost_storage = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Storage") & (results_all[("global", "global", "Subcase")] == "all")  & (results_all[("global", "global", "objective")] == "min costs")][
        ("global", "global", "total_costs")].values[0]
    cost_reduction_storage.append((cost_baseline - cost_storage)* 10**-9)
    print(f"Cost reduction 2030 Reference - Storage: {(cost_baseline - cost_storage)/10**-9}")

    cost_hydrogen = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Hydrogen") & (results_all[("global", "global", "Subcase")] == "all")  & (results_all[("global", "global", "objective")] == "min costs")][
        ("global", "global", "total_costs")].values[0]
    cost_reduction_hydrogen.append((cost_baseline - cost_hydrogen)* 10**-9)
    print(f"Cost reduction 2030 Reference - Hydrogen: {(cost_baseline - cost_hydrogen)/10**-9}")

print(f"Average Cost reduction 2030 Reference - Synergies: {np.mean(cost_reduction_synergies)}")
print(f"Average Cost reduction 2030 Reference - Grids: {np.mean(cost_reduction_grids)}")
print(f"Average Cost reduction 2030 Reference - Storage: {np.mean(cost_reduction_storage)}")
print(f"Average Cost reduction 2030 Reference - Hydrogen: {np.mean(cost_reduction_hydrogen)}")


# 2030 Synergies scenario
for cy in cys:
    print(f"{cy}")

    results_cy = results_all[(results_all[("global", "global", "cy")] == cy)]
    capacities_raw = results_cy.loc[:, ["tec_sizes"]]

    #cf of nuclear for baseline
    path_baseline = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline")][
        ("global", "global", "Path")].values[0]
    path_synergies = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "All")][
        ("global", "global", "Path")].values[0]

    annual_production_baseline = get_annual_tec_production(path_baseline)
    annual_production_synergies = get_annual_tec_production(path_synergies)

    # Nuclear capacity
    c_nuclear = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline")][
        ("tec_sizes", "PowerPlant_Nuclear_existing", "size")].values[0]
    # Nuclear production baseline
    cf_baseline = annual_production_baseline["PowerPlant_Nuclear_existing"]["electricity_output"] / (c_nuclear*8760)
    # Nuclear production synergies
    cf_synergies = annual_production_synergies["PowerPlant_Nuclear_existing"]["electricity_output"] / (c_nuclear*8760)

    print(f"CF_nuclear (Baseline) = {round(cf_baseline,2)}")
    print(f"CF_nuclear (Synergies) = {round(cf_synergies,2)}")

    #export of H2
    # export_h2 = results_all[
    #     (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "All")][
    #     ("global", "hydrogen", "export")].values[0]
    production_h2 = (annual_production_synergies["Electrolyser_PEM_offshore"]["hydrogen_output"] + annual_production_synergies["Electrolyser_PEM"]["hydrogen_output"]) / 1000
    input_h2_gt = annual_production_synergies["PowerPlant_Gas_existing"]["hydrogen_input"] / 1000
    input_h2_storage = annual_production_synergies["Storage_Hydrogen"]["hydrogen_input"] / 1000

    print(f"Fraction H2 stored = {round(input_h2_storage / production_h2,4)}")
    print(f"Fraction H2 reconverted = {round(input_h2_gt / production_h2,4)}")

# 2030 Hydrogen scenario
for cy in cys:
    print(f"{cy}")

    path_baseline = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline")][
        ("global", "global", "Path")].values[0]
    path_h2_all = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Hydrogen") & (results_all[("global", "global", "Subcase")] == "all")][
        ("global", "global", "Path")].values[0]

    # Additional re generation
    generic_production_baseline = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline") & (results_all[("global", "global", "Subcase")] == "Baseline")][
        ("global", "electricity", "generic_production")].values[0]
    curtailment_baseline = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline") & (results_all[("global", "global", "Subcase")] == "Baseline")][
        ("global", "electricity", "curtailment")].values[0]
    generic_production_h2 = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Hydrogen") & (results_all[("global", "global", "Subcase")] == "all")][
        ("global", "electricity", "generic_production")].values[0]
    curtailment_h2 = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline") & (results_all[("global", "global", "Subcase")] == "Baseline")][
        ("global", "electricity", "curtailment")].values[0]
    additional_re_generation = (generic_production_h2 - generic_production_baseline)/1000

    print(f"Curtailed fraction of electricity in Baseline = {round((curtailment_baseline)/(curtailment_baseline + generic_production_baseline), 3)}")
    print(f"Curtailed fraction of electricity in H2 all = {round((curtailment_h2)/(generic_production_h2 + curtailment_h2), 3)}")

    annual_production_baseline = get_annual_tec_production(path_baseline)
    annual_production_h2all = get_annual_tec_production(path_h2_all)

    # Nuclear capacity
    c_nuclear = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Baseline")][
        ("tec_sizes", "PowerPlant_Nuclear_existing", "size")].values[0]
    c_electrolyzer = results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Hydrogen") & (results_all[("global", "global", "Subcase")] == "all")][
        ("tec_sizes", "Electrolyser_PEM_offshore", "size")].values[0] + results_all[
        (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "Hydrogen") & (results_all[("global", "global", "Subcase")] == "all")][
        ("tec_sizes", "Electrolyser_PEM", "size")].values[0]
    # Nuclear production baseline
    cf_baseline = annual_production_baseline["PowerPlant_Nuclear_existing"]["electricity_output"] / (c_nuclear*8760)
    # Nuclear production synergies
    cf_h2_all = annual_production_h2all["PowerPlant_Nuclear_existing"]["electricity_output"] / (c_nuclear*8760)
    additional_nuclear_electricity = (annual_production_h2all["PowerPlant_Nuclear_existing"]["electricity_output"] - annual_production_baseline["PowerPlant_Nuclear_existing"]["electricity_output"])/1000

    #H2 production
    production_h2 = (annual_production_h2all["Electrolyser_PEM_offshore"]["hydrogen_output"] + annual_production_h2all["Electrolyser_PEM"]["hydrogen_output"]) / 1000
    cf_electrolyzers = production_h2*1000 / (c_electrolyzer * 8760)
    input_h2_gt = annual_production_h2all["PowerPlant_Gas_existing"]["hydrogen_input"] / 1000
    input_h2_storage = annual_production_h2all["Storage_Hydrogen"]["hydrogen_input"] / 1000
    input_h2_electrolyser = annual_production_h2all["Electrolyser_PEM"]["electricity_input"] / 1000 + annual_production_h2all["Electrolyser_PEM_offshore"]["electricity_input"] / 1000

    print(f"CF_nuclear (Baseline) = {round(cf_baseline,2)}")
    print(f"CF_nuclear (H2all) = {round(cf_h2_all,2)}")
    print(f"CF_electrolyzer (H2all) = {round(cf_electrolyzers,2)}")
    print(f"C_electrolyzer (H2all) = {round(c_electrolyzer,2)}")

    print(f"Fraction H2 stored = {round(input_h2_storage / production_h2,4)}")
    print(f"Fraction H2 reconverted = {round(input_h2_gt / production_h2,4)}")

    print(f"Additional RE: {additional_re_generation}")
    print(f"Additional Nuclear: {additional_nuclear_electricity}")
    print(f"Sum Nuclear + RE: {additional_nuclear_electricity + additional_re_generation}")
    print(f"Electrolyser input: {input_h2_electrolyser}")
    print(f"Fraction of nuclear supply: {additional_nuclear_electricity / input_h2_electrolyser}")
    print(f"H2 produced: {production_h2}")

# 2030 Operation of storage
for cy in cys:
    print(f"{cy}")

    paths = results_all[
        (results_all[("global", "global", "cy")] == cy) &
        (results_all[("global", "global", "Case")] == "Storage") &
        (results_all[("global", "global", "Subcase")] == "all") &
        (results_all[("global", "global", "objective")] == "min cost at emission limit")][
        ("global", "global", "Path")]

    storage_op_week = {}
    storage_op_month = {}
    storage_op_day = {}


    def plot_storage_operation(dict, plot_name):
        for name, series in dict.items():
            plot_data = series / series.max()
            plt.plot(series.index, plot_data, label=name)

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Operation")
        plt.title("Storage operation")
        plt.tight_layout()
        plt.savefig(f"{plot_name}.png", dpi=300)
        plt.close()

    for path in paths:
        if not "_HP_" in path:
            print(path)
            em_reduction = float(path.split("_at_")[1].split("-")[0])
            tec_op = get_tec_production(path)
            storage_op = pd.DataFrame(tec_op)
            storage_op = storage_op["period1"]
            storage_op = storage_op.T.groupby(level=[1, 2]).sum().T
            storage_op.columns = pd.MultiIndex.from_tuples(
                [('Aggregated_nodes',) + col for col in storage_op.columns]
            )
            storage_op.columns.names = ['Node', 'Technology', 'Variable']

            storage_op = storage_op.loc[:, (slice(None), ("Storage_Battery_new","Storage_Battery_Offshore"), "storage_level")].T.groupby(level=[0]).sum().T
            storage_op = add_time_steps_to_df(storage_op)

            storage_op_day[em_reduction] = aggregate_time(storage_op, 'Day', 'mean')
            storage_op_week[em_reduction] = aggregate_time(storage_op, 'Week', 'mean')
            storage_op_month[em_reduction] = aggregate_time(storage_op, 'Month', 'mean')

    plot_storage_operation(storage_op_week, f"storage_operation_week_{cy}")
    plot_storage_operation(storage_op_month, f"storage_operation_month_{cy}")
    plot_storage_operation(storage_op_day, f"storage_operation_day_{cy}")



# 2040 emission reduction in electricity sector 2030 <> 2040
results_2030 = pd.read_excel(
    "//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2030/Summary_processed.xlsx",
    header=[0, 1, 2], index_col=0)
results_2040 = pd.read_excel(
    "//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2040/Summary_processed.xlsx",
    header=[0, 1, 2], index_col=0)


path_synergies_2040 = {}
path_synergies_2040[1995] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250630132532_All_costs_cy1995_co2_tax100-1"
path_synergies_2040[2008] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250701233648_All_costs_cy2008_co2_tax100-1"
path_synergies_2040[2009] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\20250702054904_All_costs_cy2009_co2_tax100-1"

for cy in cys:
    print(f"{cy}")

    #Emissions
    emissions_baseline_2030 = results_2030[
        (results_2030[("global", "global", "cy")] == cy) & (results_2030[("global", "global", "Case")] == "Baseline")][
        ("global", "final", "emissions_other")].values[0]
    emissions_baseline_2040 = results_2040[
        (results_2040[("global", "global", "cy")] == cy) & (results_2040[("global", "global", "Case")] == "Baseline")][
        ("global", "final", "emissions_other")].values[0]
    demand_2030 = results_2030[
        (results_2030[("global", "global", "cy")] == cy) & (results_2030[("global", "global", "Case")] == "Baseline")][
        ("global", "electricity", "demand")].values[0]
    demand_2040 = results_2040[
        (results_2040[("global", "global", "cy")] == cy) & (results_2040[("global", "global", "Case")] == "Baseline")][
        ("global", "electricity", "demand")].values[0]

    print(f"Emissions in electricity sector in 2030: {emissions_baseline_2030}")
    print(f"Emissionsreduction in electricity sector is: {(emissions_baseline_2040-emissions_baseline_2030)/emissions_baseline_2030}")

    print(f"Specific emissions in electricity sector in 2030: {emissions_baseline_2030/demand_2030}")
    print(f"Specific emissions in electricity sector in 2040: {emissions_baseline_2040/demand_2040}")

    path_baseline = results_2040[
        (results_2040[("global", "global", "cy")] == cy) & (results_2040[("global", "global", "Case")] == "Baseline")][
        ("global", "global", "Path")].values[0]
    path_h2_all = results_2040[
        (results_2040[("global", "global", "cy")] == cy) & (results_2040[("global", "global", "Case")] == "Hydrogen") & (
                    results_2040[("global", "global", "Subcase")] == "all")][
        ("global", "global", "Path")].values[0]
    path_synergies = path_synergies_2040[cy]


    c_baseline = get_technology_capacities(path_baseline)
    vre_c_baseline = c_baseline.loc["Offshore_Wind"].value + c_baseline.loc["Onshore_Wind"].value + c_baseline.loc["PV"].value
    c_h2_all = get_technology_capacities(path_h2_all)
    vre_c_h2_all = c_h2_all.loc["Offshore_Wind"].value + c_h2_all.loc["Onshore_Wind"].value + c_h2_all.loc["PV"].value
    print(f"Increase in vre capacities Ref -> H2All: {(vre_c_h2_all - vre_c_baseline)/1000} GW")

    annual_production_baseline = get_annual_tec_production(path_baseline)
    annual_production_h2all = get_annual_tec_production(path_h2_all)


    # Nuclear capacity
    c_nuclear = results_2040[
        (results_2040[("global", "global", "cy")] == cy) & (results_2040[("global", "global", "Case")] == "Baseline")][
        ("tec_sizes", "PowerPlant_Nuclear_existing", "size")].values[0]
    # Nuclear production baseline
    cf_baseline = annual_production_baseline["PowerPlant_Nuclear_existing"]["electricity_output"] / (c_nuclear * 8760)
    # Nuclear production synergies
    cf_h2_all = annual_production_h2all["PowerPlant_Nuclear_existing"]["electricity_output"] / (c_nuclear * 8760)

    print(f"CF_nuclear (Baseline) = {round(cf_baseline, 2)}")
    print(f"CF_nuclear (Synergies) = {round(cf_h2_all, 2)}")

    # export of H2
    # export_h2 = results_all[
    #     (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "All")][
    #     ("global", "hydrogen", "export")].values[0]
    production_h2 = (annual_production_h2all["Electrolyser_PEM_offshore"]["hydrogen_output"] +
                     annual_production_h2all["Electrolyser_PEM"]["hydrogen_output"]) / 1000
    input_h2_gt = annual_production_h2all["PowerPlant_Gas_existing"]["hydrogen_input"] / 1000
    input_h2_storage = annual_production_h2all["Storage_Hydrogen"]["hydrogen_input"] / 1000

    print(f"Fraction H2 stored (h2 all)= {round(input_h2_storage / production_h2, 4)}")
    print(f"Fraction H2 reconverted (h2 all) = {round(input_h2_gt / production_h2, 4)}")



    annual_production_synergies = get_annual_tec_production(path_synergies)

    # Nuclear production synergies
    cf_synergies = annual_production_synergies["PowerPlant_Nuclear_existing"]["electricity_output"] / (c_nuclear * 8760)

    print(f"CF_nuclear (Baseline) = {round(cf_baseline, 2)}")
    print(f"CF_nuclear (Synergies) = {round(cf_synergies, 2)}")

    # export of H2
    # export_h2 = results_all[
    #     (results_all[("global", "global", "cy")] == cy) & (results_all[("global", "global", "Case")] == "All")][
    #     ("global", "hydrogen", "export")].values[0]
    production_h2 = (annual_production_synergies["Electrolyser_PEM_offshore"]["hydrogen_output"] +
                     annual_production_synergies["Electrolyser_PEM"]["hydrogen_output"]) / 1000
    input_h2_gt = annual_production_synergies["PowerPlant_Gas_existing"]["hydrogen_input"] / 1000
    input_h2_storage = annual_production_synergies["Storage_Hydrogen"]["hydrogen_input"] / 1000

    print(f"Fraction H2 stored (synergies)= {round(input_h2_storage / production_h2, 4)}")
    print(f"Fraction H2 reconverted (synergies) = {round(input_h2_gt / production_h2, 4)}")

# 2040 Cost reduction
cost_reduction_synergies = []
cost_reduction_grids = []
cost_reduction_storage = []
cost_reduction_hydrogen = []
cost_share_vre_synergies = []
for cy in cys:
    print(f"{cy}")
    baseline_col = results_2040[
        (results_2040[("global", "global", "cy")] == cy) & (results_2040[("global", "global", "Case")] == "Baseline")]
    synergies_col = results_2040[
        (results_2040[("global", "global", "cy")] == cy) & (results_2040[("global", "global", "Case")] == "All") & (results_2040[("global", "global", "Subcase")] == "All")  & (results_2040[("global", "global", "objective")] == "min costs")]
    grids_col = results_2040[
        (results_2040[("global", "global", "cy")] == cy) & (results_2040[("global", "global", "Case")] == "Grid Expansion") & (results_2040[("global", "global", "Subcase")] == "all")  & (results_2040[("global", "global", "objective")] == "min costs")]
    storage_col = results_2040[
        (results_2040[("global", "global", "cy")] == cy) & (results_2040[("global", "global", "Case")] == "Storage") & (results_2040[("global", "global", "Subcase")] == "all")  & (results_2040[("global", "global", "objective")] == "min costs")]
    h2_col = results_2040[
        (results_2040[("global", "global", "cy")] == cy) & (results_2040[("global", "global", "Case")] == "Hydrogen") & (results_2040[("global", "global", "Subcase")] == "all")  & (results_2040[("global", "global", "objective")] == "min costs")]

    cost_baseline = baseline_col[("global", "global", "total_costs")].values[0]
    cost_synergies = synergies_col[("global", "global", "total_costs")].values[0]
    cost_reduction_synergies.append((cost_baseline - cost_synergies)* 10**-9)
    print(f"Cost reduction 2030 Reference - Synergies: {(cost_baseline - cost_synergies)/10**-9}")

    cost_grids = grids_col[("global", "global", "total_costs")].values[0]
    cost_reduction_grids.append((cost_baseline - cost_grids)* 10**-9)
    print(f"Cost reduction 2030 Reference - Grids: {(cost_baseline - cost_grids)/10**-9}")

    cost_storage = storage_col[("global", "global", "total_costs")].values[0]
    cost_reduction_storage.append((cost_baseline - cost_storage)* 10**-9)
    print(f"Cost reduction 2030 Reference - Storage: {(cost_baseline - cost_storage)/10**-9}")

    cost_hydrogen = h2_col[("global", "global", "total_costs")].values[0]
    cost_reduction_hydrogen.append((cost_baseline - cost_hydrogen)* 10**-9)
    print(f"Cost reduction 2030 Reference - Hydrogen: {(cost_baseline - cost_hydrogen)/10**-9}")

    cost_vre = synergies_col[("tec_cost", "Onshore_Wind", "total_cost")].values[0] + synergies_col[("tec_cost", "Offshore_Wind", "total_cost")].values[0] + synergies_col[("tec_cost", "PV", "total_cost")].values[0]
    cost_new_tecs = synergies_col[("global", "final", "cost_new_system")].values[0]
    cost_share_vre_synergies.append(cost_vre/cost_new_tecs)
    print(f"Cost share of VRE 2040 Synergies: {cost_vre/cost_new_tecs}")


print(f"Average Cost reduction 2030 Reference - Synergies: {np.mean(cost_reduction_synergies)}")
print(f"Average Cost reduction 2030 Reference - Grids: {np.mean(cost_reduction_grids)}")
print(f"Average Cost reduction 2030 Reference - Storage: {np.mean(cost_reduction_storage)}")
print(f"Average Cost reduction 2030 Reference - Hydrogen: {np.mean(cost_reduction_hydrogen)}")
print(f"Average Cost share of VRE 2040 Synergies: {np.mean(cost_share_vre_synergies)}")

# 2040 min emissions
for cy in cys:
    print(f"{cy}")

    read_path = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2040\00_cy" + str(cy) + "/Summary.xlsx"
    results_2040_min_em = pd.read_excel(read_path)
    results_2040_min_em = results_2040_min_em[results_2040_min_em["objective"] == "emissions_net"]
    results_2040_min_em["emissions_total"] = (results_2040_min_em["emissions_net"] + 8.178793*10**7) * 10**-6
    results_2040_min_em["emissions_net"] = (results_2040_min_em["emissions_net"] ) * 10**-6
    print(f"Emissions electricity sector: {results_2040_min_em[["case", "emissions_net"]]}")
    print(f"Total emissions: {results_2040_min_em[["case", "emissions_total"]]}")

# 2040 higher CO2 tax
for cy in cys:
    print(f"{cy}")

