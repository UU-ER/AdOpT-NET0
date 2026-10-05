from pathlib import Path
import os
import re
import shutil
import pandas as pd

def copy_files_over(source, target, costs_emissions, yr, emission_reduction=False, results=None, cy=None):
    index_map = {
        'Baseline': 'Reference',
        'RE_only': 'Reference',
        'ElectricityGrid_on': 'GridExpansion_onshore_only',
        'ElectricityGrid_off': 'GridExpansion_offshore_only',
        'ElectricityGrid_noBorder': 'GridExpansion_no_border',
        'ElectricityGrid_all': 'GridExpansion_all',
        'Battery_on': 'Storage_onshore_only',
        'Battery_off': 'Storage_offshore_only',
        'Battery_all': 'Storage_all',
        'Battery_all_HP': 'Storage_all_hp',
        'Hydrogen_H1': 'Hydrogen_onshore_only',
        'Hydrogen_H2': 'Hydrogen_offshore_only',
        'Hydrogen_H3': 'Hydrogen_no_storage',
        'Hydrogen_H4': 'Hydrogen_no_transport',
        'Hydrogen_Baseline': 'Hydrogen_all',
        'All': 'Synergies',
    }

    # ensure target folder exists
    os.makedirs(target_path, exist_ok=True)

    # loop through subdirectories
    if emission_reduction:
        list = results["time_stamp"]
    else:
        list = os.listdir(source)


    for dirname in list:
        if emission_reduction:
            dirpath = dirname
        else:
            dirpath = os.path.join(source, dirname)


        if not os.path.isdir(dirpath):
            continue

        # Match pattern like: 20250515175436_Baseline_costs_cy1995-1
        if emission_reduction:
            match = re.search(r"_(\w+_minCost)_at_([\d.]+)", dirpath)
        elif yr == "2040":
            match = re.match(r"\d{14}_(\w+)_" + costs_emissions + r"_(cy\d+)-?\d*" + r"_co2_tax(\d+)-?\d*", dirname)
        else:
            match = re.match(r"\d{14}_(\w+)_" + costs_emissions + r"_(cy\d+)-?\d*", dirname)

        if not match:
            print(f"Skipping unrecognized name format: {dirname}")
            continue

        scenario = match.group(1).replace("_minCost", "")

        if emission_reduction:
            app_string = f"cy{cy}_red{match.group(2)}"
        elif yr == "2040":
            app_string = "co2tax" + match.group(3) + "_" + match.group(2)
        else:
            app_string = match.group(2)
        new_name = f"{yr}_{costs_emissions}_{index_map[scenario]}_{app_string}"

        print(new_name)

        src_file = os.path.join(dirpath, "optimization_results.h5")
        dst_file = os.path.join(target, f"{new_name}.h5")

        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} → {dst_file}")
        else:
            print(f"File not found: {src_file}")


# 2030 costs
# source_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2030/cost")
# target_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/00_FinalResults/v01/2030_costs")
# copy_files_over(source_path, target_path, "costs", "2030")

# 2030 emissions
# source_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2030/emission_reduction")
# target_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/00_FinalResults/v01/2030_min_emissions")
# copy_files_over(source_path, target_path, "minE", "2030")

# 2030 emission reduction
# source_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2030/emission_reduction")
# target_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/00_FinalResults/v01/2030_emission_target")
#
# for cy in [1995, 2008, 2009]:
#     results = pd.read_excel(source_path/ f"00_cy{str(cy)}"/ "Summary.xlsx")
#
#     copy_files_over(source_path, target_path, "minCost", "2030",True, results, str(cy))

# 2040 costs
source_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2040")
target_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/00_FinalResults/v01/2040_costs")
copy_files_over(source_path, target_path, "costs", "2040")

# 2040 emissions
source_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2040")
target_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/00_FinalResults/v01/2040_emissions")
copy_files_over(source_path, target_path, "emissions", "2040")

# 2040 emissions
source_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/20250515/2040/flexible_hydrogen_demand")
target_path = Path("//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/00_FinalResults/v01/2040_costs_variableh2demand")
copy_files_over(source_path, target_path, "costs", "2040")