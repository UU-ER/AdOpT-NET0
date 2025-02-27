..  _technologies_list:

List of available technology templates
-------------------------------------------

All technologies that are available as templates are listed below, as well as their respective technology models (i.e.,
types of technologies that follow similar constraints, which are explained :ref:`here<technologies>`). The list
can also be produced by:

.. testcode::

 adopt.show_available_networks()

.. csv-table::
   :file: generated_tech_list.csv
   :header-rows: 1
   :delim: ;


..  _network_list:

List of available network templates
-------------------------------------------

All networks that are available as a template are listed below. The list
can also be produced by:


.. testcode::

 adopt.show_available_technologies()

.. csv-table::
   :file: generated_netw_list.csv
   :header-rows: 1
   :delim: ;


.. _cost_models:

Calculating detailed technology and network costs
--------------------------------------------------

For a number of technologies and networks, there are detailed cost models available. The main functions to generate
them are documented here. Below, you can also find further information about each of the implemented cost models.

Examplary use:

.. testcode::

   from adopt_net0 import database as td

   # Show all available cost models
   td.help()

   # Show help for a specific cost model
   tec = "DAC_Adsorption"
   td.help(component_name=tec)

   # Define options
   options = {"currency_out": "EUR",
           "financial_year_out": 2020,
           "discount_rate": 0.1,
           "cumulative_capacity_installed_t_per_a": 10000}

   # Calculate indicators and print them
   financial_inds = td.calculate_indicators(tec, options)
   print(financial_inds)

   # Write to a json file in specified PATH
   td.write_json(tec, PATH, options)


.. automodule:: adopt_net0.database.technology_database
    :members: help, write_json, calculate_indicators


Detailed technology cost models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Photovoltaic**

.. automodule:: adopt_net0.database.technologies.pv_cost_model

.. autoclass:: PV_CostModel
   :undoc-members:
   :noindex:

**Wind energy**

.. automodule:: adopt_net0.database.technologies.wind_cost_model

.. autoclass:: WindEnergy_CostModel
   :undoc-members:
   :noindex:

**Solid Sorbent Direct Air Capture**

.. automodule:: adopt_net0.database.technologies.dac_adsorption_cost_model

.. autoclass:: Dac_SolidSorbent_CostModel
   :undoc-members:
   :noindex:

**CO2 compression**

.. automodule:: adopt_net0.database.technologies.co2_compression_cost_model

.. autoclass:: CO2_Compression_CostModel
   :undoc-members:
   :noindex:

Detailed network cost models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**CO2 pipeline**

.. automodule:: adopt_net0.database.networks.co2_pipelines_cost_model

.. autoclass:: CO2_Pipeline_CostModel
   :undoc-members:
   :noindex:

