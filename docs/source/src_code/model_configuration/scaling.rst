.. _scaling:

Model Scaling
=====================================
Scaling is a crucial aspect in the context of Mixed-Integer Linear Programming (MILP) models, as it significantly
impacts the numerical stability and efficiency of the solution algorithms. Proper scaling ensures that all variables
and constraints are in comparable ranges, preventing numerical issues such as loss of precision, ill-conditioning,
or convergence problems during the optimization process. When variables and constraints are scaled appropriately,
MILP solvers can navigate the solution space more effectively, leading to faster convergence and reliable results.

For matrices in a MILP model, it is recommended to scale them in a way that the absolute values of the coefficients fall
within the range of 1e-3 to 1e3, The right-hand side values should also be scaled to be within a similar range.

If you have larger ranges, please consider global scaling as well as individual scaling.

Exemplary Usage
^^^^^^^^^^^^^^^^^^
The PyHub offers two options to scale the model: Global options specified in the :func:`.ClusteredDataHandle` class and
scaling factors for each individual constraint and variable. As a general rule of thumb: if you are dealing with a
model on the GW scale, use a scaling factor of 1e-3 to bring down the right hand side. Typically the global options
do not help to reduce the matrix range, there you need to scale constraints individually. As a general rule of thumb,
large matrix coefficients tend to come from high unit-capex values, and you need to scale the respective constraint.
Small matrix coefficients are typically from the performance functions of technologies or the loss function of a
network. Find below a minimal example of how to fix the scaling.

To solve the unscaled model:

.. testcode::

    import src.data_management as dm

    from src.model_configuration import ModelConfiguration
    import src.data_management as dm
    from src.energyhub import EnergyHub
    import numpy as np
    from pathlib import Path

    # TOPOLOGY
    topology = dm.SystemTopology()
    topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
    topology.define_carriers(['electricity', 'gas', 'hydrogen', 'heat'])
    # topology.define_nodes(['onshore'])
    topology.define_nodes(['onshore', 'offshore'])
    topology.define_new_technologies('onshore', ['Photovoltaic', 'Storage_Battery', 'WindTurbine_Onshore_4000', 'GasTurbine_simple'])
    topology.define_new_technologies('offshore', ['WindTurbine_Offshore_6000'])

    distance = dm.create_empty_network_matrix(topology.nodes)
    distance.at['onshore', 'offshore'] = 100
    distance.at['offshore', 'onshore'] = 100

    connection = dm.create_empty_network_matrix(topology.nodes)
    connection.at['onshore', 'offshore'] = 1
    connection.at['offshore', 'onshore'] = 1
    topology.define_new_network('electricitySimple', distance=distance, connections=connection)

    # Initialize instance of DataHandle
    data = dm.DataHandle(topology)

    # CLIMATE DATA
    data.read_climate_data_from_file('onshore', './data/climate_data_onshore.txt')
    data.read_climate_data_from_file('offshore', './data/climate_data_offshore.txt')

    # DEMAND
    electricity_demand = np.ones(len(topology.timesteps)) * 1000
    data.read_demand_data('onshore', 'electricity', electricity_demand)

    # Import Limit
    import_lim = np.ones(len(topology.timesteps)) * 100
    data.read_import_limit_data('onshore', 'electricity', import_lim)
    gas_import = np.ones(len(topology.timesteps)) * 2000
    data.read_import_limit_data('onshore', 'gas', gas_import)

    # Export Limit
    import_lim = np.ones(len(topology.timesteps)) * 10000
    data.read_export_limit_data('onshore', 'heat', import_lim)

    # Import price
    data.read_import_price_data('onshore', 'electricity', np.ones(len(topology.timesteps)) * 60)
    data.read_import_emissionfactor_data('onshore', 'electricity', np.ones(len(data.topology.timesteps)) * 0.1)
    gas_price = np.ones(len(topology.timesteps)) * 70
    data.read_import_price_data('onshore', 'gas', gas_price)

    # Carbon Tax
    carbontax = np.ones(len(topology.timesteps)) * 11
    carbonsubsidy = np.ones(len(topology.timesteps)) * 11
    data.read_carbon_price_data(carbontax, 'tax')
    data.read_carbon_price_data(carbonsubsidy, 'subsidy')

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data(load_path = './scaling_data/technology_data')
    data.read_network_data(load_path = './scaling_data/network_data')

    # SAVING/LOADING DATA FILE
    configuration = ModelConfiguration()

    # Solve unscaled model
    energyhub = EnergyHub(data, configuration)
    energyhub.quick_solve()

The ranges given out by gurobi are:

.. testcode::

    Matrix range     [7e-06, 1e+06]
    Objective range  [1e+00, 1e+00]
    Bounds range     [2e-02, 9e+06]
    RHS range        [1e+00, 1e+03]


To solve a scaled model, we specify global scaling factors in the configuration:

.. testcode::

    # Solve scaled model
    configuration.scaling = 0
    configuration.scaling_factors.energy_vars = 1e-2
    configuration.scaling_factors.cost_vars = 1

    energyhub = EnergyHub(data, configuration)
    energyhub.quick_solve()

And we add scaling factors to some of the individual constraints and variables in component json files.

.. testcode::

    # In electricitySimple.json:
    [...]
      "size_max": 1,
      "size_is_int": 0,
      "decommission": 0,
      "ScalingFactors": {
        "const_flowlosses": 1e4,
        "var_losses": 1e1,
        "const_flow_size_high": 1
      }
    }

    # In GasTurbine_simple.json:
    [...]
            "output_carrier": {
          "electricity": "MW",
          "heat": "MW"
        }
      },
      "ScalingFactors": {
        "const_capex_aux": 1e-1,
        "const_input_output": 1e2,
        "const_max_input": 1e2,
        "const_opex_variable": 1e2,
        "const_opex_fixed": 1e2
      }
    }

    # In Photovoltaic.json:
    [...]
          "electricity": "MW"
        }
      },
      "ScalingFactors": {
        "const_capex_aux": 1e-3,
        "var_capex_aux": 1e-1
      }
    }

    # In Storage_Battery.json:
          "electricity": "MW"
        }
      },
      "ScalingFactors": {
        "const_opex_variable": 1e2,
        "const_opex_fixed": 1e2,
        "const_capex_aux": 1e-3,
        "var_capex_aux": 1e-1
      }
    }

    # In the two wind turbine json files:
        "output_carrier": {
          "electricity": "MW"
        }
      },
      "ScalingFactors": {
        "const_capex_aux": 1e-3,
        "var_capex_aux": 1e-1
      }
    }

The ranges given out by gurobi are now:

.. testcode::

    Matrix range     [1e-03, 1e+03]
    Objective range  [1e+02, 1e+02]
    Bounds range     [2e-04, 9e+04]
    RHS range        [1e-02, 1e+01]
