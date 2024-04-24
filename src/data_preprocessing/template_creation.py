import json
from pathlib import Path

import numpy as np
import pandas as pd


def create_empty_network_matrix(nodes):
    """
    Function creates matrix for defined nodes.

    :param list nodes: list of nodes to create matrices from
    :return: pandas data frame with nodes
    """
    # construct matrix
    matrix = pd.DataFrame(
        data=np.full((len(nodes), len(nodes)), 0), index=nodes, columns=nodes
    )
    return matrix


def create_input_data_folder_template(base_path: Path | str) -> None:
    """
    This function creates the input data folder structure required to organize the input data to the model.
    Note that the folder needs to already exist with a Topology.json file in it that specifies the nodes, carriers,
    timesteps, investment periods and the length of the investment period.

    You can create an examplary json template with the function `create_topology_template`

    :param str/Path base_path: path to folder
    """
    # Convert to Path
    if isinstance(base_path, str):
        base_path = Path(base_path)

    # Read Topology.json
    with open(base_path / "Topology.json") as json_file:
        topology = json.load(json_file)

    timesteps = pd.date_range(
        start=topology["start_date"],
        end=topology["end_date"],
        freq=topology["resolution"],
    )

    # Template jsons:
    networks = {"existing": [], "new": []}
    technologies = {"existing": [], "new": []}
    energy_balance_options = {
        carrier: {"curtailment_possible": 0} for carrier in topology["carriers"]
    }

    # Template csvs
    carrier_data = pd.DataFrame(
        index=timesteps,
        columns=[
            "Demand",
            "Import limit",
            "Export limit",
            "Import price",
            "Export price",
            "Import emission factor",
            "Export emission factor",
            "Generic production",
        ],
    )
    climate_data = pd.DataFrame(
        index=timesteps,
        columns=["ghi", "dni", "dhi", "temp_air", "rh", "TECHNOLOGYNAME_hydro_inflow"],
    )
    carbon_cost = pd.DataFrame(index=timesteps, columns=["price", "subsidy"])
    node_locations = pd.DataFrame(
        index=topology["nodes"], columns=["lon", "lat", "alt"]
    )

    # Make folder structure
    node_locations.to_csv(base_path / "NodeLocations.csv", sep=";")
    for investment_period in topology["investment_periods"]:
        (base_path / investment_period).mkdir(parents=True, exist_ok=True)

        # Networks
        with open(base_path / investment_period / "Networks.json", "w") as f:
            json.dump(networks, f, indent=4)
        (base_path / investment_period / "network_data").mkdir(
            parents=True, exist_ok=True
        )
        (base_path / investment_period / "network_topology").mkdir(
            parents=True, exist_ok=True
        )
        (base_path / investment_period / "network_topology" / "new").mkdir(
            parents=True, exist_ok=True
        )
        (base_path / investment_period / "network_topology" / "existing").mkdir(
            parents=True, exist_ok=True
        )
        empty_network_matrix = create_empty_network_matrix(topology["nodes"])
        empty_network_matrix.to_csv(
            base_path
            / investment_period
            / "network_topology"
            / "new"
            / "connection.csv",
            sep=";",
        )
        empty_network_matrix.to_csv(
            base_path / investment_period / "network_topology" / "new" / "distance.csv",
            sep=";",
        )
        empty_network_matrix.to_csv(
            base_path
            / investment_period
            / "network_topology"
            / "new"
            / "size_max_arcs.csv",
            sep=";",
        )
        empty_network_matrix.to_csv(
            base_path
            / investment_period
            / "network_topology"
            / "existing"
            / "size.csv",
            sep=";",
        )
        empty_network_matrix.to_csv(
            base_path
            / investment_period
            / "network_topology"
            / "existing"
            / "distance.csv",
            sep=";",
        )
        empty_network_matrix.to_csv(
            base_path
            / investment_period
            / "network_topology"
            / "existing"
            / "connection.csv",
            sep=";",
        )
        empty_network_matrix.to_csv(
            base_path
            / investment_period
            / "network_topology"
            / "existing"
            / "size_max_arcs.csv",
            sep=";",
        )

        # Node data
        (base_path / investment_period / "node_data").mkdir(parents=True, exist_ok=True)
        for node in topology["nodes"]:
            (base_path / investment_period / "node_data" / node / "carrier_data").mkdir(
                parents=True, exist_ok=True
            )
            with open(
                base_path
                / investment_period
                / "node_data"
                / node
                / "Technologies.json",
                "w",
            ) as f:
                json.dump(technologies, f, indent=4)
            with open(
                base_path
                / investment_period
                / "node_data"
                / node
                / "carrier_data"
                / "EnergybalanceOptions.json",
                "w",
            ) as f:
                json.dump(energy_balance_options, f, indent=4)
            for carrier in topology["carriers"]:
                carrier_data.to_csv(
                    base_path
                    / investment_period
                    / "node_data"
                    / node
                    / "carrier_data"
                    / f"{carrier}.csv",
                    sep=";",
                )
            climate_data.to_csv(
                base_path / investment_period / "node_data" / node / "ClimateData.csv",
                sep=";",
            )
            carbon_cost.to_csv(
                base_path / investment_period / "node_data" / node / "CarbonCost.csv",
                sep=";",
            )
            (
                base_path / investment_period / "node_data" / node / "technology_data"
            ).mkdir(parents=True, exist_ok=True)


def create_optimization_templates(path: Path | str) -> None:
    """
    Creates an examplary topology json file in the specified path.

    :param str/Path path: path to folder to create Topology.json
    """
    if isinstance(path, str):
        path = Path(path)

    topology_template = {
        "nodes": ["node1", "node2"],
        "carriers": ["electricity", "hydrogen"],
        "investment_periods": ["period1", "period2"],
        "start_date": "2022-01-01 00:00",
        "end_date": "2022-12-31 23:00",
        "resolution": "1h",
        "investment_period_length": 1,
    }

    configuration_template = {
        "optimization": {
            "objective": {
                "description": "String specifying the objective/type of optimization.",
                "options": [
                    "costs",
                    "emissions_pos",
                    "emissions_net",
                    "emissions_minC",
                    "costs_emissionlimit",
                    "pareto",
                ],
                "value": "costs",
            },
            "monte_carlo": {
                "on": {
                    "description": "Turn Monte Carlo simulation on.",
                    "options": [0, 1],
                    "value": 0,
                },
                "sd": {
                    "description": "Value defining the range in which variables are varied in Monte Carlo simulations (defined as the standard deviation of the original value).",
                    "value": 0.2,
                },
                "N": {
                    "description": "Number of Monte Carlo simulations.",
                    "value": 100,
                },
                "on_what": {
                    "description": "List: Defines component to vary.",
                    "options": ["Technologies", "ImportPrices", "ExportPrices"],
                    "value": "Technologies",
                },
            },
            "pareto_points": {"description": "Number of Pareto points.", "value": 5},
            "timestaging": {
                "description": "Defines number of timesteps that are averaged (0 = off).",
                "value": 0,
            },
            "typicaldays": {
                "N": {
                    "description": "Determines number of typical days (0 = off).",
                    "value": 0,
                },
                "method": {
                    "description": "Determine method used for modeling technologies with typical days.",
                    "options": [2],
                    "value": 2,
                },
            },
            "multiyear": {
                "description": "Enable multiyear analysis, if turned off max time horizon is 1 year.",
                "options": [0, 1],
                "value": 0,
            },
        },
        "solveroptions": {
            "solver": {
                "description": "String specifying the solver used.",
                "value": "gurobi",
            },
            "mipgap": {"description": "Value to define MIP gap.", "value": 0.001},
            "timelim": {
                "description": "Value to define time limit in hours.",
                "value": 10,
            },
            "threads": {
                "description": "Value to define number of threads (default is maximum available).",
                "value": 0,
            },
            "mipfocus": {
                "description": "Modifies high level solution strategy.",
                "options": [0, 1, 2, 3],
                "value": 0,
            },
            "nodefilestart": {
                "description": "Parameter to decide when nodes are compressed and written to disk.",
                "value": 60,
            },
            "method": {
                "description": "Defines algorithm used to solve continuous models.",
                "options": [-1, 0, 1, 2, 3, 4, 5],
                "value": -1,
            },
            "heuristics": {
                "description": "Parameter to determine amount of time spent in MIP heuristics.",
                "value": 0.05,
            },
            "presolve": {
                "description": "Controls the presolve level.",
                "options": [-1, 0, 1, 2],
                "value": -1,
            },
            "branchdir": {
                "description": "Determines which child node is explored first in the branch-and-cut.",
                "options": [-1, 0, 1],
                "value": 0,
            },
            "lpwarmstart": {
                "description": "Controls whether and how warm start information is used for LP.",
                "options": [0, 1, 2],
                "value": 0,
            },
            "intfeastol": {
                "description": "Value that determines the integer feasibility tolerance.",
                "value": 1e-05,
            },
            "feastol": {
                "description": "Value that determines feasibility for all constraints.",
                "value": 1e-06,
            },
            "numericfocus": {
                "description": "Degree of which Gurobi tries to detect and manage numeric issues.",
                "options": [0, 1, 2, 3],
                "value": 0,
            },
            "cuts": {
                "description": "Setting defining the aggressiveness of the global cut.",
                "options": [-1, 0, 1, 2, 3],
                "value": -1,
            },
        },
        "reporting": {
            "save_detailed": {
                "description": "Setting to select how the results are saved. When turned off only the summary is saved.",
                "options": [0, 1],
                "value": 1,
            },
            "save_summary_path": {
                "description": "Path to save the summary file path to.",
                "value": "./userData/",
            },
            "save_path": {
                "description": "Option to define the save path.",
                "value": "./userData/",
            },
            "case_name": {
                "description": "Option to define a case study name that is added to the results folder name.",
                "value": -1,
            },
            "write_solution_diagnostics": {
                "description": "If 1, writes solution quality, if 2 also writes pyomo to Gurobi variable map and constraint map to file.",
                "options": [0, 1, 2],
                "value": 0,
            },
        },
        "energybalance": {
            "violation": {
                "description": "Determines the energy balance violation price (-1 is no violation allowed).",
                "value": -1,
            },
            "copperplate": {
                "description": "Determines if a copperplate approach is used.",
                "options": [0, 1],
                "value": 0,
            },
        },
        "economic": {
            "global_discountrate": {
                "description": "Determines if and which global discount rate is used. This holds for the CAPEX of all technologies and networks.",
                "value": -1,
            },
            "global_simple_capex_model": {
                "description": "Determines if the CAPEX model of technologies is set to 1 for all technologies.",
                "options": [0, 1],
                "value": 0,
            },
        },
        "performance": {
            "dynamics": {
                "description": "Determines if dynamics are used.",
                "options": [0, 1],
                "value": 0,
            }
        },
        "scaling": {
            "scaling_on": {
                "description": "Determines if the model is scaled. If 1, it uses global and component specific scaling factors.",
                "options": [0, 1],
                "value": 0,
            },
            "scaling_factors": {
                "energy_vars": {
                    "description": "Scaling factor used for all energy variables.",
                    "value": 0.001,
                },
                "cost_vars": {
                    "description": "Scaling factor used for all cost variables.",
                    "value": 0.001,
                },
                "objective": {
                    "description": "Scaling factor used for the objective function.",
                    "value": 1,
                },
            },
        },
    }

    with open(path / "Topology.json", "w") as f:
        json.dump(topology_template, f, indent=4)
    with open(path / "ConfigModel.json", "w") as f:
        json.dump(configuration_template, f, indent=4)
