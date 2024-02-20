import h5py
from src.result_management.read_results import *

path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/baseline_demand/20240220170428_TESTHydrogen_Baseline/optimization_results.h5'
print_h5_tree(path)

with h5py.File(path, 'r') as hdf_file:
    df = extract_datasets_from_h5group(hdf_file["operation/networks"])

