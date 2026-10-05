import mes_north_sea.postprocessing.compare_results.new as read_new
import mes_north_sea.postprocessing.compare_results.old as read_old

import pandas as pd


path_results = {}
path_results["new"] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250424\2040_cy2009\20250514162440_RE_only_costs-1\optimization_results.h5"
path_results["old"] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\tests\20250514162437_TESTRE_only_costs\optimization_results.h5"
path_export = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250514diff_new_old_DK.xlsx"

# Energybalance
energybalance = {}
energybalance["new"] = pd.DataFrame(read_new.read_energy_balance(path_results["new"]))
energybalance["new"] = energybalance["new"]["period1"]
energybalance["old"] = read_old.read_energy_balance(path_results["old"])
energybalance["old"] = energybalance["old"].reset_index(drop=True)
energybalance["diff"] = pd.DataFrame()

for col in energybalance["new"]:
    if col in energybalance["old"].columns:
        energybalance["diff"][col] = energybalance["new"][col] - energybalance["old"][col]

energybalance["diff_sum"] = energybalance["diff"].sum()


# Technology operations
technology_operation = {}
technology_operation["new"] = pd.DataFrame(read_new.read_technology_operation(path_results["new"]))
technology_operation["new"] = technology_operation["new"]["period1"]
technology_operation["old"] = read_old.read_technology_operation(path_results["old"])
technology_operation["old"] = technology_operation["old"].reset_index(drop=True)
technology_operation["diff"] = pd.DataFrame()

for col in technology_operation["new"]:
    if col in technology_operation["old"].columns:
        technology_operation["diff"][col] = technology_operation["new"][col] - technology_operation["old"][col]

technology_operation["diff_sum"] = technology_operation["diff"].sum()

# Technology design
technology_design = {}
technology_design["new"] = pd.DataFrame(read_new.read_technology_design(path_results["new"]))
technology_design["new"] = technology_design["new"].drop(columns=["Period"])
technology_design["new"]["Variable"] = technology_design["new"]["Variable"].str.replace("capex_tot", "capex")
technology_design["new"] = technology_design["new"].set_index(["Node", "Technology", "Variable"])
technology_design["old"] = read_old.read_technology_design(path_results["old"])
technology_design["old"] = technology_design["old"].set_index(["Node", "Technology", "Variable"])
technology_design["diff"] = {}

for idx, row in technology_design["new"].iterrows():
    try:
        technology_design["diff"][idx] = technology_design["new"].loc[idx].values[0] - technology_design["old"].loc[idx].values[0]
    except:
        pass

technology_design["diff"] = pd.Series(technology_design["diff"])


# Network design
network_design = {}
network_design["new"] = read_new.read_networks(path_results["new"])[0]
network_design["new"] = network_design["new"].drop(columns=["Period", "FromNode", "ToNode", 'para_capex_gamma1', 'para_capex_gamma2', 'para_capex_gamma3',
       'para_capex_gamma4',])
network_design["new"] = network_design["new"].set_index(["Network", "Arc_ID"])
network_design["old"] = read_old.read_networks(path_results["old"])
network_design["old"] = network_design["old"].drop(columns=["FromNode", "ToNode", 'total_emissions'])
network_design["old"] = network_design["old"].set_index(["Network", "Arc_ID"])
network_design["diff"] = network_design["new"] - network_design["old"]




with pd.ExcelWriter(path_export, engine="xlsxwriter") as writer:
    technology_operation["diff_sum"].to_excel(writer, sheet_name="technology_operation_diff")
    energybalance["diff_sum"].to_excel(writer, sheet_name="energybalance_diff")
    technology_design["diff"].reset_index().to_excel(writer, sheet_name="technology_design_diff")
    technology_design["old"].reset_index().to_excel(writer, sheet_name="technology_design_old")
    technology_design["new"].reset_index().to_excel(writer, sheet_name="technology_design_new")
    network_design["diff"].reset_index().to_excel(writer, sheet_name="network_design_diff")
    network_design["old"].reset_index().to_excel(writer, sheet_name="network_design_old")
    network_design["new"].reset_index().to_excel(writer, sheet_name="network_design_new")
