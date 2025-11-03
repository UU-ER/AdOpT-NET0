import pandas as pd
import os

def get_electricity_prices():

    directory = "C:/Users/jwiegner/PycharmProjects/AdOpT-NET0/hp_cost_targeting/time_series/electricity_prices"

    el_prices_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            el_prices_list.append(
                pd.read_excel(os.path.join(directory, filename), skiprows=4, header=[0,1,2], index_col=0)
            )

    el_prices = pd.concat(el_prices_list, axis=1)
    el_prices.columns = el_prices.columns.get_level_values(0).str.replace("BZN|", "")

    el_prices_summary = pd.DataFrame()
    el_prices_summary["mean"] = el_prices.mean()
    el_prices_summary["min"] = el_prices.min()
    el_prices_summary["max"] = el_prices.max()
    el_prices_summary["std"] = el_prices.std()
    el_prices_summary["nas"] = el_prices.isna().sum()

    print("Electricity price summary:")
    print(el_prices_summary)

    return el_prices