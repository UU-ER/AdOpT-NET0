import pandas as pd
import numpy as np
import json

def get_RES_technologies(technologies):
    """
    Writes all RES technologies to a dict
    """
    tecs_used = {}
    RES_tecs_used = {}
    for nodename in technologies:
        tecs_used[nodename] = technologies[nodename]
        RES_tecs_used[nodename] = {}
        # read in data to Data Handle and fit performance functions
        for tec in tecs_used[nodename]:
            # Read in JSON files
            with open('./data/technology_data/' + tec + '.json') as json_file:
                technology_data = json.load(json_file)
            if technology_data['TechnologyPerf']['tec_type'] == 'RES':
                RES_tecs_used[nodename][tec] = 1
    return RES_tecs_used

def reshape_df(series_to_add, column_names, nr_cols):
    """
    Transform all data to large dataframe with each row being one day
    """
    if not type(series_to_add).__module__ == np.__name__:
        transformed_series = series_to_add.to_numpy()
    else:
        transformed_series = series_to_add
    transformed_series = transformed_series.reshape((-1, nr_cols))
    transformed_series = pd.DataFrame(transformed_series, columns=column_names)
    return transformed_series


def get_day_factors(keys):
    """
    Get factors for each hour

    This function assigns an integer to each hour in the full resolution, specifying how many times
    this hour occurs in the clustered data-set.
    """
    factors = pd.DataFrame(np.unique(keys, return_counts=True))
    factors = factors.transpose()
    factors.columns = ['timestep', 'factor']
    return factors