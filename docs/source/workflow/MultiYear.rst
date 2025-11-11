.. _workflow_multi-year:

Multiyear analysis with rolling-horizon
========================================


When performing a **multi-year analysis** using the **rolling-horizon** approach, the model is organized as a dictionary,
where each key corresponds to a modeled investment interval. Each interval has its own case study folder, from which
the model reads the relevant input data.

Each investment interval is solved as a separate optimization problem, while the installed capacities of networks and
technologies are carried over from the previous interval. By appending the interval name to the case name in the model
configuration, each interval automatically generates a separate result folder, ensuring that results are stored
independently for each optimization step.

.. testcode::

    adopthub = {}
    intervals = ["Interval_1", "Interval_2", "Interval_n"]

    for i, interval in enumerate(intervals):
        path_interval = path + "/" + interval

        # Take installed capacities from previous interval
        if i != 0:
            prev_interval = intervals[i - 1]
            installed_capacities_existing(adopthub, interval, prev_interval, path_interval)

        adopthub[interval] = ModelHub()
        adopthub[interval].read_data(path_interval, start_period=None, end_period=None)

        # Add interval name as case name
        adopthub[interval].data.model_config["reporting"]["case_name"]["value"] = interval

        adopthub[interval].quick_solve()