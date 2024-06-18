.. _workflow_manage-results:

=====================================
Manage results
=====================================

After your model has solved, the results are automatically saved as HDF5 files. For an exact overview of the structure of
these files, see the :ref:`Source Code for Result Management<src-code_result_management>`.

Provided an h5 file was saved, the data can be visualized on a visualization platform. You can access this platform
by going to https://resultvisualization.streamlit.app/. Further instructions are on the web page.
Note: from the visualization platform, the results can also be downloaded in csv format.

You can specify a path to save the results and a summary in ``ModelConfig.JSON``: ``save_path`` and ``save_summary_path``,
respectively. Each run will have a separate folder named with a case name, if specified, and a timestamp
of the run. The case name can be defined in ``ModelConfig.JSON``: ``case_name``.

The results folder contains 1) the Gurobi log of your optimization, and 2) the HDF5 file. The Excel file with the summary of each run (one row per run) is created in your specified path: for
each additional run you do an additional row is appended to the summary.

If you want to export more results to Excel, you can do so after the optimization as follows:

.. testcode::

    file_path = 'pathtoh5file/optimization_results.h5'
    save_path = 'pathtosaveresults'
    print_h5_tree(file_path)
    with h5py.File(file_path, 'r') as hdf_file:
        data = extract_datasets_form_h5(hdf_file["operation/energy_balance/offshore"])
        data.to_excel(save_path)
        print(data)
