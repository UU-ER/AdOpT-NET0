.. _workflow_manage-results:

Manage results
=====================================

After your model has solved, the results are automatically saved as HDF5 files. For an exact overview of the structure of
these files, see the :ref:`Source Code for Result Management<src-code_result_management>`.

For the results, it is important to have defined the path to where you want the results folder (named with a timestamp
of the run) and the summary thereof to be saved in ``ModelConfig.JSON``: ``save_path`` and ``save_summary_path``,
respectively.

The results folder is named with a timestamp of the model run and contains 1) the Gurobi log of your optimization, and
2) the HDF5 file. The Excel file with the summary of each run (one row per run) is created in your specified path: for
each additional run you do an additional row is appended to the summary.

If you want to export more results to Excel, you can do so after the optimization as follows:

.. testcode::

    file_path = './userData/20240206140357/optimization_results.h5'
    save_path = 'whereveryouwanttosaveit'
    print_h5_tree(file_path)
    with h5py.File(file_path, 'r') as hdf_file:
        data = extract_datasets_form_h5(hdf_file["operation/energy_balance/offshore"])
        data.to_excel(save_path)
        print(data)