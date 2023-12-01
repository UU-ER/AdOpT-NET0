import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, interpn

result_folder = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/ESCAPE_Conference paper_Data exchange/Results/v3/'
save_path = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/ESCAPE_Conference paper_Data exchange/plots/'

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
                pump_capex_rows = df[df.iloc[:,0].str.match(r'pump_\d+_capex') & (df.iloc[:, 1] > 0.01)]
                nr_pumps_installed = pump_capex_rows.shape[0]
                parameters['total_pump_size'] = parameters['single_pump_designpower'] * nr_pumps_installed

                # calculate total turbine size
                turbine_capex_rows = df[df.iloc[:,0].str.match(r'turbine_\d+_capex') & (df.iloc[:, 1] > 0.01)]
                nr_turbine_installed = turbine_capex_rows.shape[0]
                parameters['total_turbine_size'] = parameters['single_turbine_designpower'] * nr_turbine_installed

                result_data.append(parameters)

result_df = pd.DataFrame(result_data)
result_df.to_excel('C:/Users/6574114/OneDrive - Universiteit Utrecht/ESCAPE_Conference paper_Data exchange/oceanbatteryresults.xlsx')
plt.rcParams.update({'font.size': 14})


def plot_size(x,y,z,z_label,l):
    """
    plots a contourplot on x,y,z with z_label and the levels l
    """
    x_s = [4, 9, 9]
    y_s = [0.1, 0.1, 0.9]
    x_p, y_p = np.meshgrid(np.linspace(x.min(), x.max(), 10),
                           np.linspace(y.min(), y.max(), 10))
    z_p = griddata((x.values, y.values), z, (x_p, y_p), method='linear')
    fig1, ax = plt.subplots()
    CS1 = ax.contourf(x_p, y_p, z_p, 50, cmap='viridis_r')
    CS2 = ax.contour(x_p, y_p, z_p, levels=l, colors='black')
    S = plt.scatter(x_s, y_s, marker='D', facecolors='white', edgecolors='black', s=60)
    plt.clabel(CS2, fontsize=10)
    plt.xlabel('Normalized standard deviation of electricity price')
    plt.ylabel('Normalized capex of reservoir')
    plt.xlim([1, 10])
    plt.ylim([0.05, 1])
    colorbar = plt.colorbar(CS1)
    colorbar.set_label(z_label)


# PLOT: RESERVOIR SIZE
x = result_df["SD"]
y = result_df["CAPEX"]
z = result_df["reservoir_size"] / 1000
l = [50, 100, 150, 200, 250]
z_label = 'Reservoir size in 1000 m³'
plot_size(x,y,z,z_label,l)
plt.savefig(save_path + 'reservoir_size.jpg', dpi=600, bbox_inches='tight')

# PLOT: Turbine Design
x = result_df["SD"]
y = result_df["CAPEX"]
z = result_df["total_turbine_size"]
l = [1.5, 3, 4.5, 6, 7.5, 9]
z_label = 'total turbine design power (MW)'
plot_size(x,y,z,z_label,l)
plt.savefig(save_path + 'turbine_size.jpg', dpi=600, bbox_inches='tight')

# PLOT: Pump Design
x = result_df["SD"]
y = result_df["CAPEX"]
z = result_df["total_pump_size"]
l = [2, 4, 6]
z_label = 'total pump design power (MW)'
plot_size(x,y,z,z_label,l)
plt.savefig(save_path + 'pump_size.jpg', dpi=600, bbox_inches='tight')
