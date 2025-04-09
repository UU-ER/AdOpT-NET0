from adopt_net0 import database as td
import json
import pandas as pd
import os

def define_topology(path):
    # Load json template
    with open(path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)
    # Nodes
    topology["nodes"] = ["node1", "node2"]
    # Carriers:
    topology["carriers"] = ["electricity", "gas", "uranium"]
    # Investment periods:
    topology["investment_periods"] = ["period1"]
    # Save json template
    with open(path / "Topology.json", "w") as json_file:
        json.dump(topology, json_file, indent=4)

def define_model_config(path):
    # Load json template
    with open(path / "ConfigModel.json", "r") as json_file:
        configuration = json.load(json_file)
    # Change objective
    configuration["optimization"]["objective"]["value"] = "costs"
    # Set MILP gap
    configuration["solveroptions"]["mipgap"]["value"] = 0.01
    # Turn on dynamics
    configuration["performance"]["dynamics"]["value"] = 1
    # Save json template
    with open(path / "ConfigModel.json", "w") as json_file:
        json.dump(configuration, json_file, indent=4)

def define_node_locations(path):
    # Define node locations (here two exemplary location in the Netherlands)
    node_location = pd.read_csv(path / "NodeLocations.csv", sep=';', index_col=0, header=0)
    node_lon = {'node1': 5.1214, 'node2': 5.24}
    node_lat = {'node1': 52.0907, 'node2': 51.9561}
    node_alt = {'node1': 5, 'node2': 10}
    for node in ['node1', 'node2']:
        node_location.at[node, 'lon'] = node_lon[node]
        node_location.at[node, 'lat'] = node_lat[node]
        node_location.at[node, 'alt'] = node_alt[node]

    node_location = node_location.reset_index()
    node_location.to_csv(path / "NodeLocations.csv", sep=';', index=False)

def define_technologies(path):
    # Add required technologies for node 'node1'
    with open(path / "period1" / "node_data" / "node1" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)
    technologies["new"] = ["Storage_Battery", "Photovoltaic", "WindTurbine_Onshore_4000", "PowerPlant_Gas", "PowerPlant_Nuclear"]

    with open(path / "period1" / "node_data" / "node1" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add required technologies for node 'node2'
    with open(path / "period1" / "node_data" / "node2" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)
    technologies["new"] = ["WindTurbine_Offshore_11000"]

    with open(path / "period1" / "node_data" / "node2" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

def define_networks(path):
    # Add networks
    with open(path / "period1" / "Networks.json", "r") as json_file:
        networks = json.load(json_file)
    networks["new"] = ["electricityOffshore"]
    
    with open(path / "period1" / "Networks.json", "w") as json_file:
        json.dump(networks, json_file, indent=4)

    # Make a new folder for the new network
    os.makedirs(path / "period1" / "network_topology" / "new" / "electricityOffshore", exist_ok=True)

    # max size arc
    arc_size = pd.read_csv(path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)
    arc_size.loc["node1", "node2"] = 4
    arc_size.loc["node2", "node1"] = 4
    arc_size.to_csv(
        path / "period1" / "network_topology" / "new" / "electricityOffshore" / "size_max_arcs.csv", sep=";")
    print("Max size per arc:", arc_size)

    # Use the templates, fill and save them to the respective directory
    # Connection
    connection = pd.read_csv(path / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)
    connection.loc["node1", "node2"] = 1
    connection.loc["node2", "node1"] = 1
    connection.to_csv(
        path / "period1" / "network_topology" / "new" / "electricityOffshore" / "connection.csv", sep=";")
    print("Connection:", connection)

    # Delete the template
    os.remove(path / "period1" / "network_topology" / "new" / "connection.csv")

    # Distance
    distance = pd.read_csv(path / "period1" / "network_topology" / "new" / "distance.csv", sep=";",
                           index_col=0)
    distance.loc["node1", "node2"] = 50
    distance.loc["node2", "node1"] = 50
    distance.to_csv(path / "period1" / "network_topology" / "new" / "electricityOffshore" / "distance.csv",
                    sep=";")
    print("Distance:", distance)

    # Delete the template
    os.remove(path / "period1" / "network_topology" / "new" / "distance.csv")

    # Delete the max_size_arc template
    os.remove(path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")


def write_technology_jsons(path):
    # Adapt data
    currency = "EUR"
    financial_year = 2023
    discount_rate = 0.1

    tec = "WindTurbine_Onshore_4000"
    options = {"currency_out": currency,
               "financial_year_out": financial_year,
               "discount_rate": discount_rate,
               "nameplate_capacity_MW": 4,
               "terrain": "Onshore",
               "source": "DEA",
               "projection_year": 2030,
               }
    td.write_json(tec, path, options)

    tec = "WindTurbine_Offshore_11000"
    options = {"currency_out": currency,
               "financial_year_out": financial_year,
               "discount_rate": discount_rate,
               "nameplate_capacity_MW": 11,
               "terrain": "Offshore",
               "source": "DEA",
               "projection_year": 2030,
               "mounting_type": "fixed"
               }
    td.write_json(tec, path, options)

    tec = "Photovoltaic"
    options = {"currency_out": currency,
               "financial_year_out": financial_year,
               "discount_rate": discount_rate,
               "source": "DEA",
               "projection_year": 2030,
               "pv_type": "utility",
               }
    td.write_json(tec, path, options)
