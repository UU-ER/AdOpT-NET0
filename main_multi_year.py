# from adopt_net0.model_configuration import ModelConfiguration
# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np


# Specify the path to your input data
# path = Path("../AdOpt_caseStudies/multi_year_casestudy")
path = Path("Z:/PyHub/PyHub_casestudies/Multiyear_test")

create_data = 0

if create_data == 1:
    # Create template files (comment these lines if already defined)
    adopt.create_optimization_templates(path)

    # Load json template
    with open(path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)
    # Nodes
    topology["nodes"] = ["node1"]
    # Carriers:
    topology["carriers"] = ["electricity", "heat", "gas"]
    # Investment periods:
    topology["investment_periods"] = ["period1", "period2"]
    # Investment periods:
    topology["investment_period_length"] = 3
    # Save json template
    with open(path / "Topology.json", "w") as json_file:
        json.dump(topology, json_file, indent=4)

    # Load json template
    with open(path / "ConfigModel.json", "r") as json_file:
        configuration = json.load(json_file)
    # Change objective
    configuration["optimization"]["objective"]["value"] = "costs"
    # Typical days
    configuration["optimization"]["typicaldays"]["N"]["value"] = 0
    # Multi-year
    configuration["optimization"]["multiyear"]["value"] = 1
    # Set MILP gap
    configuration["solveroptions"]["mipgap"]["value"] = 0.02
    # Save json template
    with open(path / "ConfigModel.json", "w") as json_file:
        json.dump(configuration, json_file, indent=4)

    adopt.create_input_data_folder_template(path)

    # Add technologies period1
    with open(
        path / "period1" / "node_data" / "node1" / "Technologies.json", "r"
    ) as json_file:
        technologies = json.load(json_file)
    technologies["new"] = ["Boiler_Small_NG", "Boiler_El"]

    with open(
        path / "period1" / "node_data" / "node1" / "Technologies.json", "w"
    ) as json_file:
        json.dump(technologies, json_file, indent=4)

    # Add technologies period2
    with open(
        path / "period2" / "node_data" / "node1" / "Technologies.json", "r"
    ) as json_file:
        technologies = json.load(json_file)
    technologies["new"] = ["Boiler_Small_NG", "Boiler_El"]

    with open(
        path / "period2" / "node_data" / "node1" / "Technologies.json", "w"
    ) as json_file:
        json.dump(technologies, json_file, indent=4)

    # Copy over technology files
    adopt.copy_technology_data(path)

    # Initialize input data
    adopt.fill_carrier_data(path, value_or_data=0)

    # Set import limits/cost
    adopt.fill_carrier_data(
        path,
        value_or_data=50,
        columns=["Import limit"],
        carriers=["gas"],
        nodes=["node1"],
        investment_periods=["period1", "period2"],
    )
    adopt.fill_carrier_data(
        path,
        value_or_data=2,
        columns=["Demand"],
        carriers=["heat"],
        nodes=["node1"],
        investment_periods=["period1", "period2"],
    )
    adopt.fill_carrier_data(
        path,
        value_or_data=50,
        columns=["Import limit"],
        carriers=["electricity"],
        nodes=["node1"],
    )
    # Different gas and el prices in period 1 and 2
    adopt.fill_carrier_data(
        path,
        value_or_data=50,
        columns=["Import price"],
        carriers=["gas"],
        nodes=["node1"],
        investment_periods=["period1"],
    )
    adopt.fill_carrier_data(
        path,
        value_or_data=100,
        columns=["Import price"],
        carriers=["gas"],
        nodes=["node1"],
        investment_periods=["period2"],
    )
    adopt.fill_carrier_data(
        path,
        value_or_data=50,
        columns=["Import price"],
        carriers=["electricity"],
        nodes=["node1"],
        investment_periods=["period1"],
    )
    adopt.fill_carrier_data(
        path,
        value_or_data=30,
        columns=["Import price"],
        carriers=["electricity"],
        nodes=["node1"],
        investment_periods=["period2"],
    )


run_case = 1

if run_case == 1:
    # Load json template
    with open(path / "ConfigModel.json", "r") as json_file:
        configuration = json.load(json_file)
    # Change objective
    configuration["optimization"]["objective"]["value"] = "costs"
    # Typical days
    configuration["optimization"]["typicaldays"]["N"]["value"] = 0
    # Multi-year
    configuration["optimization"]["multiyear"]["value"] = 1
    # Set MILP gap
    configuration["solveroptions"]["mipgap"]["value"] = 0.02
    # Save json template
    with open(path / "ConfigModel.json", "w") as json_file:
        json.dump(configuration, json_file, indent=4)

    # Construct model
    m = adopt.ModelHub()
    m.read_data(path, start_period=0, end_period=120)

    # run model
    m.quick_solve()
    m.model["full"].periods["period1"].node_blocks["node1"].tech_blocks_active[
        "Boiler_El"
    ].pprint()
