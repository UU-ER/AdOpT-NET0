import adopt_net0 as adopt
import pandas as pd

#setup model (only required once)
input_data_path = "./offshore_storage/model_input"
# adopt.create_optimization_templates(input_data_path)
# adopt.create_input_data_folder_template(input_data_path)
# adopt.copy_technology_data(input_data_path)
# adopt.copy_network_data(input_data_path)

climate_year = 2000

# Write generic production
cap_pv = 10000
cap_wind_onshore = 8000
cap_wind_offshore = 2000
scale_demand = 0.001

cf_pv = pd.read_csv("./offshore_storage/data/capacity_factors/DE_Solar2030.csv",
                    sep=",")
cf_wind_onshore = pd.read_csv("./offshore_storage/data/capacity_factors/DE_Wind_onshore2030.csv",
                    sep=",")
cf_wind_offshore = pd.read_csv(
    "./offshore_storage/data/capacity_factors/DE_Wind_offshore2030.csv",
                    sep=",")

prod_pv = cf_pv[str(climate_year)] * cap_pv
prod_wind_onshore = cf_wind_onshore[str(climate_year)] * cap_wind_onshore
prod_wind_offshore = cf_wind_offshore[str(climate_year)] * cap_wind_offshore

prod_onshore = prod_pv + prod_wind_onshore
prod_offshore = prod_wind_offshore

adopt.fill_carrier_data(input_data_path, prod_onshore, ["Generic production"],
                        ["electricity"], ["onshore"], ["period1"])
adopt.fill_carrier_data(input_data_path, prod_offshore, ["Generic production"],
                        ["electricity"], ["offshore"], ["period1"])

# Write demand
demand = pd.read_csv(
    "./offshore_storage/data/demand/DE_Demand2030.csv", sep=";")
demand_cy = demand[str(climate_year)] * scale_demand
adopt.fill_carrier_data(input_data_path, demand_cy, ["Demand"],
                        ["electricity"], ["onshore"], ["period1"])

# Allow for gas import
adopt.fill_carrier_data(input_data_path, 10000, ["Import"],
                        ["gas"], ["onshore"], ["period1"])

m = adopt.ModelHub()
m.read_data(input_data_path, start_period=0, end_period=2)
m.quick_solve()