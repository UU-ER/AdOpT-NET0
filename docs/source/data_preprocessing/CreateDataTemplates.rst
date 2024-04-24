..  _data-preprocessing_create-data-templates:

Creating Input Data Templates
=====================================
After you have defined your system topology, you can now retrieve the templates for the input data required for the
optimization with this specific topology. For this, you call the ``create_input_data_folder_template`` method,
passing the same folder path as for the :ref:`Model Templates<data-preprocessing_create-model-templates>`.

.. automodule:: src.data_preprocessing.template_creation
    :members: create_input_data_folder_template
    :exclude-members:



Note: all methods related to template creation can be found in the ``template_creation.py`` module in the
``data_preprocessing`` directory.