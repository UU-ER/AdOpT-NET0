from pathlib import Path  # import Path from pathlib module
import os  # import os module
import pandas as pd

# directory = Path(r'C:\Users\6574114\OneDrive - Universiteit Utrecht\PhD Jan\Papers\DOSTA - HydrogenOffshore\00_raw_data\climate_data')  # set directory path
#
# summary = {}
# summary["1995"] = {}
# summary["2008"] = {}
# summary["2009"] = {}
# for root, _, files in os.walk(directory):
#     for filename in files:
#         idx = filename.replace(".csv", "")
#         idx = idx.split("_")
#         nuts_region = idx[0]
#         year = idx[1]
#         summary[year][nuts_region] = pd.read_csv(os.path.join(root, filename), index_col=0).mean().to_dict()
#
#
# records = []
# for year, regions in summary.items():
#     for region, values in regions.items():
#         for key, val in values.items():
#             records.append((year, region, key, val))
#
# df = pd.DataFrame(records, columns=['year', 'region', 'variable', 'value'])
# df.to_excel("climate_data.xlsx")

directory = r'C:\Users\6574114\PycharmProjects\PyHubProductive\mes_north_sea\clean_data\capacity_factors/'

summary_on = pd.DataFrame()
summary_off = pd.DataFrame()
for cy in [1995, 2008, 2009]:
    summary_on[("pv", cy)] = pd.read_csv(directory + "pv" + str(cy) + ".csv", index_col=0).mean()
    summary_on[("wind_on", cy)] = pd.read_csv(directory + "wind_onshore" + str(cy) + ".csv", index_col=0).mean()
    summary_off[("wind_off", cy)] = pd.read_csv(directory + "wind_offshore" + str(cy) + ".csv", index_col=0).mean()

summary_on.columns = pd.MultiIndex.from_tuples(summary_on.columns)
summary_off.columns = pd.MultiIndex.from_tuples(summary_off.columns)

summary_on.mean()
summary_off.mean()

directory = r'C:\Users\6574114\PycharmProjects\PyHubProductive\mes_north_sea\clean_data\production_profiles_re/'

re_prod = pd.DataFrame()
for cy in [1995, 2008, 2009]:
    re_prod[cy] = pd.read_csv(directory + "production_profiles_re" + str(cy) + ".csv", index_col=0, header=[0, 1]).mean()

directory = r'C:\Users\6574114\PycharmProjects\PyHubProductive\mes_north_sea\clean_data\hydro_inflows/'

hydro = pd.DataFrame()
for cy in [1995, 2008, 2009]:
    hydro[cy] = pd.read_csv(directory + "hydro_inflows" + str(cy) + ".csv", index_col=0, header=[0, 1]).mean()

