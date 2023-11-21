import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

result_folder = 'C:/Users/6145795/OneDrive - Universiteit Utrecht/ESCAPE_Conference paper_Data exchange/Results/v2/'

result_data = []
for folder in os.listdir(result_folder):
    folder_path = os.path.join(result_folder, folder)

    if os.path.isdir(folder_path):
        folder_name_parts = folder.split('_')

        if len(folder_name_parts) == 4 and folder_name_parts[2].startswith("SD") and folder_name_parts[3].startswith("CAPEX"):
            sd_value = int(folder_name_parts[2][2:])
            capex_value = float(folder_name_parts[3][5:])

            excel_file_path = os.path.join(folder_path, 'Nodes', 'offshore', 'TechnologyDesign.xlsx')

            if os.path.exists(excel_file_path):
                df = pd.read_excel(excel_file_path)

                parameters = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
                parameters['SD'] = sd_value
                parameters['CAPEX'] = capex_value

                # calculate total pump size
                pump_capex_rows = df[df.iloc[:,0].str.match(r'pump_\d+_capex') & (df.iloc[:, 1] > 0.1)]
                nr_pumps_installed = pump_capex_rows.shape[0]
                parameters['total_pump_size'] = parameters['single_pump_designpower'] * nr_pumps_installed

                # calculate total turbine size
                turbine_capex_rows = df[df.iloc[:,0].str.match(r'turbine_\d+_capex') & (df.iloc[:, 1] > 0.1)]
                nr_turbine_installed = turbine_capex_rows.shape[0]
                parameters['total_turbine_size'] = parameters['single_turbine_designpower'] * nr_turbine_installed

                result_data.append(parameters)

result_df = pd.DataFrame(result_data)

### PLOT: CAPEX, SD, RESERVOIR SIZE

x = result_df["SD"]
y = result_df["CAPEX"]
z = result_df["reservoir_size"]

plt.figure()
contour = plt.tricontour(x, y, z, cmap='viridis')
plt.colorbar(contour, label='Reservoir Size [m3]')
plt.xlabel('SD')
plt.ylabel('CAPEX')
plt.title('Isolines of reservoir size')
plt.grid(True)
plt.show()


### PLOT: CAPEX, SD, PUMP SIZE

x = result_df["SD"]
y = result_df["CAPEX"]
z = result_df["total_pump_size"]

plt.figure()
contour = plt.tricontour(x, y, z, cmap='viridis')
plt.colorbar(contour, label='Installed pump capacity [MW]')
plt.xlabel('SD')
plt.ylabel('CAPEX')
plt.title('Isolines of pump capacity installed')
plt.grid(True)
plt.show()

### PLOT: CAPEX, SD, TURBINE SIZE

x = result_df["SD"]
y = result_df["CAPEX"]
z = result_df["total_turbine_size"]

plt.figure()
contour = plt.tricontour(x, y, z, cmap='viridis')
plt.colorbar(contour, label='Installed turbine capacity [MW]')
plt.xlabel('SD')
plt.ylabel('CAPEX')
plt.title('Isolines of turbine capacity installed')
plt.grid(True)
plt.show()