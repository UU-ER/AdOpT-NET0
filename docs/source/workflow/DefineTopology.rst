.. _workflow_define-topology:

Defining System Topology
=====================================

In the ``Topology.JSON`` file, the system topology can be defined. The system topology describes:

#. The nodes: Here you can alter the number and names of nodes you want to model. Node names can be chosen freely.
#. The carriers: Here you can specify the carriers that you want to model in your system, e.g., electricity, hydrogen,
   gas, etc.. In general, carrier names need to follow the same naming convention as in the technology files in
   ``.\data\technology_data``. Note: in theory, carrier names can be chosen freely: for each carrier specified you can
   (later) simply set a demand, production, import and export, and you can adjust conversion technologies to produce or
   use this carrier, and subsequently an energy balance for this carrier will be solved.
#. The investment periods: Here you can specify the number and names of investment periods you want to
   model. (elaborate)
#. The duration of your run: Here you can enter the start and end date of your model run and the resolution. By default,
   the length is one year with a resolution of one hour. NB: the dates you fill in here determine how many time
   steps are generated in the :ref:`templates for input data<workflow_create-data-templates>` (i.e., in the
   default case, 8760 time steps are accounted for, so you would need to define hourly input data for 1 year). You can
   later decide to run the model using only part of your data, by not reading all the data into the model.
#. The investment period length: Here you can specify the length of one investment period (in years).

