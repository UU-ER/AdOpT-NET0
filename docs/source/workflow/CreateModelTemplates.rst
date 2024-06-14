.. _workflow_create-model-templates:

=====================================
1 Creating Model Templates
=====================================
In order to set up your model, you start by creating a new folder for your case study that will all input data. In the
main module (``main.py``), you specify the path to this folder and pass it through the :func:`create_optimization_templates`
method.

.. automodule:: adopt_net0.data_preprocessing.template_creation
    :members: create_optimization_templates
    :exclude-members:

This creates two templates in the form of json files: one for the model configuration and one for the topology. Before
you can continue with :ref:`retrieving the templates for the input data<workflow_create-data-templates>`,
you will first have to adjust ``Topology.JSON`` to :ref:`define the system topology<workflow_define-topology>`. The
:ref:`configuration <src-code_model-configuration>` can be defined anytime before solving the model in ``ModelConfig.JSON``.

Note: See a complete documentation of all template creation functions in the
:ref:`source code documentation<src-code_data-preparation>`.
