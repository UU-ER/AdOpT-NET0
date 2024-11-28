.. _workflow_create-model-templates:

=====================================
Creating Model Templates
=====================================
In order to set up your model, you start by creating a new folder for your case study
that will contain all input data. Specify the path to this folder and pass it to the
:func:`create_optimization_templates` method.

.. testcode::

    import adopt_net0 as adopt
    from pathlib import Path

    path = Path("path_to_your_input_data_folder")

    adopt.create_optimization_templates(path)

This creates two templates in the form of json files: one for the model
configuration and one for the topology. Before you can continue with
:ref:`retrieving the templates for the input data<workflow_create-data-templates>`,
you will first have to adjust ``Topology.JSON`` to
:ref:`define the system topology<workflow_define-topology>`. The
:ref:`configuration <src-code_model-configuration>` can be  defined anytime before
solving the model in ``ModelConfig.JSON``.

Note: See a complete documentation of all template creation functions in the
:ref:`source code documentation<src-code_data-preparation>`. There, you can also view
a tree diagram on how a complete input data directory looks like.

.. automodule:: adopt_net0.data_preprocessing.template_creation
    :members: create_optimization_templates
    :exclude-members:

