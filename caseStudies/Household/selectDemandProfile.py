import csv
import numpy as np

def select_one_demand_profile(csv_file):
    BE_demand_profile = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > 5:  # Check if the row has at least two columns
                try:
                    value = float(row[5])  # Convert to float
                    BE_demand_profile.append(value)  # Append to list
                except ValueError:
                    pass  # Skip invalid values
    return BE_demand_profile

csv_file = "TotalDemand_NT_2030_1995.csv"  # Replace with the path to your CSV file
BE_demand_profile = select_one_demand_profile(csv_file)

householdAnnualDemand = 4  # MWh
total_demand = np.sum(BE_demand_profile)
rescale_factor = householdAnnualDemand / total_demand

# Rescale the demand profile
household_demand_profile = [rescale_factor * demand for demand in BE_demand_profile]

# Stretch the values
min_demand = np.min(household_demand_profile)
max_demand = np.max(household_demand_profile)
average_demand = np.mean(household_demand_profile)

stretched_demand_profile = 0.0015 * (household_demand_profile - average_demand)/max_demand + average_demand

print(np.min(stretched_demand_profile))
print(np.max(stretched_demand_profile))

# Save the household_demand_profile to a CSV file
output_csv_file = "el_demand.csv"
with open(output_csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Demand"])
    for demand_value in stretched_demand_profile:
        csv_writer.writerow([demand_value])

