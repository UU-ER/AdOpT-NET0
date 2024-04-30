.. _src-code_data-processing:

Data Processing
=====================================

All the data and settings that have been defined in your input data folder, have to be read into the model. For this,
``src.data_management.handle_input_data.DataHandle`` provides a class to import and manipulate data consistent with
the model.

.. automodule:: src.data_management.handle_input_data
    :members:
    :exclude-members: define_multiindex, reshape_df, DataHandle_AveragedData,
                      ClusteredDataHandle, flag_tecs_for_clustering