.. _workflow_create-model-templates:

Creating Model Templates
=====================================
In order to set up your model, you start by creating a new folder for all input data. In the main module
(``main.py``), you specify the path to this folder and pass it through the ``create_optimization_templates`` method.

.. automodule:: src.data_preprocessing.template_creation
    :members: create_optimization_templates
    :exclude-members:

This creates two templates in the form of json files: one for the model configuration and one for the topology. Before
you can continue with :ref:`retrieving the templates for the input data<workflow_create-data-templates>`,
you will first have to adjust ``Topology.JSON`` to :ref:`define the system topology<workflow_define-topology>`. The
:ref:`configuration <model_configuration>`can be defined anytime before solving the model in ``ModelConfig.JSON``.

Note: all methods related to template creation can be found in the ``template_creation.py`` module in the
``data_preprocessing`` directory.
