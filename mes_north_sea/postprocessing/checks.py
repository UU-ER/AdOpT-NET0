from pathlib import Path
import os
import pandas as pd

result_path = Path('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20231201/20231206111512_Baseline/')
node_path = Path.joinpath(result_path, 'nodes')
nodes = [f.name for f in os.scandir(node_path) if f.is_dir()]

metrics = {}

# Max Imports
metrics['max_import'] = {}
for node in nodes:
    energybalance = pd.read_excel(Path.joinpath(node_path, node, 'Energybalance.xlsx'), sheet_name='electricity',
                                  index_col=0)
    metrics['max_import'][node] = max(energybalance['Import'])


pd.DataFrame(metrics['max_import'])
