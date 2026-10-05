import pandas as pd

# CAPACITY FACTORS
def calculate_cap_factors(caps_type, re_profiles, save_path, tec):
    cap_factors = pd.DataFrame()
    for idx, row in caps_type.iterrows():
        node = row['Node']
        re_profile = re_profiles[(node, tec)]

        if tec != 'Wind offshore':
            cap = row['Capacity our work']
        else:
            cap = re_profile.max()

        cap_factors[node] = re_profile/cap

    cap_factors.to_csv(save_path)
    print(cap_factors.max())
    print(cap_factors.mean())

for climate_year in [1995, 2008, 2009]:
    re_profiles = pd.read_csv('./mes_north_sea/clean_data/production_profiles_re/production_profiles_re' + str(climate_year) + '.csv', header=[0,1])
    caps = pd.read_csv('./mes_north_sea/clean_data/installed_capacities/capacities_node.csv', index_col=[0])

    calculate_cap_factors(caps[caps['Technology'] == 'Solar'], re_profiles, './mes_north_sea/clean_data/capacity_factors/pv' + str(climate_year) + '.csv', 'PV')
    calculate_cap_factors(caps[caps['Technology'] == 'Wind Onshore'], re_profiles, './mes_north_sea/clean_data/capacity_factors/wind_onshore' + str(climate_year) + '.csv', 'Wind onshore')

    # NOTE: OFFSHORE WIND CAPACITY FACTORS ARE GENERATED FROM ERA5 DATA AND WIND FARM DATA DIRECTLY
