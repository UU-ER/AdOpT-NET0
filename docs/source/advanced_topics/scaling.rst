..   _scaling:

=====================================
Model Scaling
=====================================
Scaling is a crucial aspect in the context of Mixed-Integer Linear Programming (MILP) models, as it significantly
impacts the numerical stability and efficiency of the solution algorithms. Proper scaling ensures that all variables
and constraints are in comparable ranges, preventing numerical issues such as loss of precision, ill-conditioning,
or convergence problems during the optimization process. When variables and constraints are scaled appropriately,
MILP solvers can navigate the solution space more effectively, leading to faster convergence and reliable results.

The PyHub offers two options to scale the model:

- Global scaling (options specified in the :func:`.ClusteredDataHandle` class): this option is used to bring down the
  right hand side. As a general rule of thumb: if you are dealing with a model on the GW scale, use a scaling factor of
  1e-3 to bring down the right hand side.
- Individual scaling (factors for individual constraints and variables): this option is used to help reduce the matrix range.
  As a general rule of thumb, large matrix coefficients tend to come from high unit-capex values, and you need to scale
  the respective constraint. Small matrix coefficients are typically from the performance functions of technologies or
  the loss function of a network.

For matrices in a MILP model, it is recommended to scale them in a way that the absolute values of the coefficients fall
within the range of 1e-3 to 1e3, The right-hand side values should also be scaled to be within a similar range. If you
have larger ranges, please consider global scaling as well as individual scaling.

Exemplary Usage
^^^^^^^^^^^^^^^^^^
Find below a minimal example of how to fix the scaling.

Let's say you are solving a model that consists of two nodes, connected by an electricity network. The following
technologies are allowed to be newly installed at either of the nodes: 'Photovoltaic', 'Storage_Battery',
'WindTurbine_Onshore_4000', 'GasTurbine_simple', 'WindTurbine_Offshore_6000'.

The ranges given out by gurobi are:

.. testcode::

    Matrix range     [7e-06, 1e+06]
    Objective range  [1e+00, 1e+00]
    Bounds range     [2e-02, 9e+06]
    RHS range        [1e+00, 1e+03]


To scale the model, we specify global scaling factors in ``configuration.json`` as follows:

.. testcode::

    "scaling": {
        "scaling": {
            "description": "Determines if the model is scaled. If 1, it uses global and component specific scaling factors.",
            "options": [
                0,
                1
            ],
            "value": 1
        },
        "scaling_factors": {
            "energy_vars": {
                "description": "Scaling factor used for all energy variables.",
                "value": 1e-2
            },
            "cost_vars": {
                "description": "Scaling factor used for all cost variables.",
                "value": 1
            },
            "objective": {
                "description": "Scaling factor used for the objective function.",
                "value": 1
            }

And we add scaling factors to some of the individual constraints and variables in component json files.

.. testcode::

    # In electricitySimple.json:
    [...]
      "size_max": 1,
      "size_is_int": 0,
      "decommission": 0,
  "decommission_full": 0,
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

When solving this adapted model, the ranges given out by gurobi are:

.. testcode::

    Matrix range     [1e-03, 1e+03]
    Objective range  [1e+02, 1e+02]
    Bounds range     [2e-04, 9e+04]
    RHS range        [1e-02, 1e+01]
