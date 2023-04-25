.. _model_configuration:

Model Configuration
====================
The class ``src.model_configuration`` is used to define and change the standard modeling configuration. An example
on how to use the class is given in :ref:`Example Usage <configuration-example-usage>`.

.. automodule:: src.model_configuration
    :members:

..  _configuration-example-usage:

Example Usage
^^^^^^^^^^^^^^^^
The framework includes a standard configuration of the modeling settings. You can change the standard settings as follows:

.. testcode::

    from src.model_configuration import ModelConfiguration

    # Initialize an instance of the model configuration class
    configuration = ModelConfiguration()

    # Change some settings, while maintaining other settings
    configuration.solveroptions.timelim = 10
    configuration.economic.globalinterest = 0.05

    # Configure to cluster for 40 typical days
    configuration.optimization.typicaldays = 40

