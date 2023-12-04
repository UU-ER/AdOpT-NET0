.. EnergyHub documentation master file, created by
   sphinx-quickstart on Thu Nov 17 10:46:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EnergyHub's documentation!
=====================================
Energyhub is a Python Library for bottom-up multi energy system modeling. It can
model conversion technologies and networks for any carrier.

To define an energy system to optimize, you need to

#. Define a topology, i.e. which carriers, nodes, technologies and networks are part of the system (see
   documentation :ref:`here <data-management-system_topology>`) and the input data, e.g. weather data,
   technology performance, etc. (see documentation :ref:`here <data-management-data-handle>`).

#. Define the modeling configuration, i.e. objective, global optimization settings, solver options, etc. (see documentation :ref:`here <model_configuration>`).

#. construct the model (see documentation :ref:`here <model_constructing>`).

#. solve the model (see documentation :ref:`here <model_solving>`).

#. look at the optimization results (see documentation :ref:`here <result_data_management>`).

Table of Content
==================

.. toctree::
   :maxdepth: 2

   InputDataManagement
   ModelComponents
   ModelConfiguration
   ModelConstruction
   ModelSolve
   ModelDiagnostics
   ResultDataManagement
   ForDevelopers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
