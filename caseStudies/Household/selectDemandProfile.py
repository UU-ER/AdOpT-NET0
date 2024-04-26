import csv
import numpy as np

def select_one_demand_profile(csv_file):
    BE_demand_profile = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > 1:  # Check if the row has at least two columns
                try:
                    value = float(row[1])  # Convert to float
                    BE_demand_profile.append(value)  # Append to list
                except ValueError:
                    pass  # Skip invalid values
    return BE_demand_profile

csv_file = "TotalDemand_NT_2030_1995.csv"  # Replace with the path to your CSV file
BE_demand_profile = select_one_demand_profile(csv_file)

householdAnnualDemand = 3  # MWh
total_demand = np.sum(BE_demand_profile)
rescale_factor = householdAnnualDemand / total_demand

# Rescale the demand profile
household_demand_profile = [rescale_factor * demand for demand in BE_demand_profile]
a =1

