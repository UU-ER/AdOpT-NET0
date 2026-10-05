import mes_north_sea.postprocessing.compare_results.new as read

import pandas as pd


path_results = {}
path_results["res1"] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2030_test\20250515101535_Baseline_costs_cy2008-1\optimization_results.h5"
path_results["res2"] = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2030_test\20250515110014_Baseline_costs_cy1995-1\optimization_results.h5"
path_export = r"\\Soliscom.uu.nl\geo\USERS\StaffUsers\6574114\EhubResults\MES NorthSea\20250515\2030_test\diff_new20081995.xlsx"

# Energybalance
energybalance = {}
energybalance["res1"] = pd.DataFrame(read.read_energy_balance(path_results["res1"]))
energybalance["res1"] = energybalance["res1"]["period1"]
energybalance["res2"] = pd.DataFrame(read.read_energy_balance(path_results["res2"]))
energybalance["res2"] = energybalance["res2"]["period1"]
energybalance["diff"] = pd.DataFrame()


energybalance["diff"] = energybalance["res1"] - energybalance["res2"]

energybalance["diff_sum"] = energybalance["diff"].sum().reset_index()

#
# # Technology operations
# technology_operation = {}
# technology_operation["res1"] = pd.DataFrame(read.read_technology_operation(path_results["res1"]))
# technology_operation["res1"] = technology_operation["res1"]["period1"]
# technology_operation["res2"] = read.read_technology_operation(path_results["res2"])
# technology_operation["res2"] = technology_operation["res2"].reset_index(drop=True)
# technology_operation["diff"] = pd.DataFrame()
#
# for col in technology_operation["res1"]:
#     if col in technology_operation["res2"].columns:
#         technology_operation["diff"][col] = technology_operation["res1"][col] - technology_operation["res2"][col]
#
# technology_operation["diff_sum"] = technology_operation["diff"].sum()
#
# # Technology design
# technology_design = {}
# technology_design["res1"] = pd.DataFrame(read.read_technology_design(path_results["res1"]))
# technology_design["res1"] = technology_design["res1"].drop(columns=["Period"])
# technology_design["res1"]["Variable"] = technology_design["res1"]["Variable"].str.replace("capex_tot", "capex")
# technology_design["res1"] = technology_design["res1"].set_index(["Node", "Technology", "Variable"])
# technology_design["res2"] = read.read_technology_design(path_results["res2"])
# technology_design["res2"] = technology_design["res2"].set_index(["Node", "Technology", "Variable"])
# technology_design["diff"] = {}
#
# for idx, row in technology_design["res1"].iterrows():
#     try:
#         technology_design["diff"][idx] = technology_design["res1"].loc[idx].values[0] - technology_design["res2"].loc[idx].values[0]
#     except:
#         pass
#
# technology_design["diff"] = pd.Series(technology_design["diff"])
#
#
# # Network design
# network_design = {}
# network_design["res1"] = read.read_networks(path_results["res1"])[0]
# network_design["res1"] = network_design["res1"].drop(columns=["Period", "FromNode", "ToNode", 'para_capex_gamma1', 'para_capex_gamma2', 'para_capex_gamma3',
#        'para_capex_gamma4',])
# network_design["res1"] = network_design["res1"].set_index(["Network", "Arc_ID"])
# network_design["res2"] = read.read_networks(path_results["res2"])
# network_design["res2"] = network_design["res2"].drop(columns=["FromNode", "ToNode", 'total_emissions'])
# network_design["res2"] = network_design["res2"].set_index(["Network", "Arc_ID"])
# network_design["diff"] = network_design["res1"] - network_design["res2"]
#



with pd.ExcelWriter(path_export, engine="xlsxwriter") as writer:
    # technology_operation["diff_sum"].to_excel(writer, sheet_name="technology_operation_diff")
    energybalance["diff_sum"].to_excel(writer, sheet_name="energybalance_diff")
    # technology_design["diff"].reset_index().to_excel(writer, sheet_name="technology_design_diff")
    # technology_design["res2"].reset_index().to_excel(writer, sheet_name="technology_design_old")
    # technology_design["res1"].reset_index().to_excel(writer, sheet_name="technology_design_new")
    # network_design["diff"].reset_index().to_excel(writer, sheet_name="network_design_diff")
    # network_design["res2"].reset_index().to_excel(writer, sheet_name="network_design_old")
    # network_design["res1"].reset_index().to_excel(writer, sheet_name="network_design_new")
