.. _workflow_multi-year:

Multiyear analysis with myopic foresight
========================================


When performing a **multi-year analysis** using the **myopic foresight** approach, the model is organized as a dictionary,
where each key corresponds to a modeled investment interval. Each interval has its own case study folder, from which
the model reads the relevant input data.

Each investment interval is solved as a separate optimization problem, while the installed capacities of networks and
technologies are carried over from the previous interval. By appending the interval name to the case name in the model
configuration, each interval automatically generates a separate result folder, ensuring that results are stored
independently for each optimization step.

In AdOpt, technology lifetimes represent only the economic depreciation period, which is used to compute annuitized
investment costs. Actual technical lifetimes are not explicitly modeled. This is because, in industrial practice, assets
can often remain operational well beyond their nominal lifetime through increased maintenance efforts and retrofits. To
reflect this flexibility, users may directly adjust the input parameters, such as the maintenance-related fixed costs,
to represent the effective condition of each technology. This approach ensures that the model remains adaptable to
real-world asset management strategies without enforcing hard replacement cycles.

.. testcode::

    adopthub = {}
    intervals = ["Interval_1", "Interval_2", "Interval_3"]
    intervals_between_years = [10, 10]

    for i, interval in enumerate(intervals):
        interval_path = Path(casestudy_path) / f"Case_{interval}"

        # Take installed capacities from previous interval
        if i != 0:
            prev_interval = intervals[i - 1]
            installed_capacities_existing(
                adopthub, interval, prev_interval,
                interval_path, intervals_between_years, i,
            )
            del adopthub[prev_interval]  # free memory

        adopthub[interval] = ModelHub()
        adopthub[interval].read_data(interval_path)

        # Add interval name as case name
        adopthub[interval].data.model_config["reporting"]["case_name"]["value"] = interval

        adopthub[interval].quick_solve()

``intervals_between_years`` holds the number of years between each pair of consecutive
intervals and must have length ``len(intervals) - 1``. If it is set to ``None``, all
capacities are carried forward unchanged and no lifetime check is performed (a warning
is raised). If it has the wrong length, a ``ValueError`` is raised.

Lifetime tracking (carry_overs)
----------------------------

When ``intervals_between_years`` is provided, each investment is tracked as a separate
**carry_over**: capacity built in a given interval keeps its own size and remaining
lifetime. At every transition, the years elapsed are subtracted from each carry_over's
remaining lifetime; carry_overs that reach the end of their lifetime are expired and
excluded from the carry-over (a warning reports the expired capacity).

Two lifetime fields in the component JSON files are used:

- ``technical_lifetime`` — drives the physical expiry of a carry_over. If not defined,
  ``lifetime`` is used instead.
- ``lifetime`` — the economic lifetime, used for the annualization of the investment
  costs (CAPEX). It should not exceed ``technical_lifetime`` (a warning is raised
  otherwise).

The tracking state is stored in the case study files and carried from one interval to
the next:

- ``Technologies.json`` (per node) and ``Networks.json`` receive the entries
  ``carry_over_sizes``, ``remaining_lifetime``, ``carry_over_capex`` and
  ``remaining_econ_lifetime``, each keyed by technology/network and by the interval in
  which the carry_over was built.
- For networks, one ``size_{interval}.csv`` per surviving carry_over is written to the
  existing network topology folder, next to the total ``size.csv``.

For example, after two transitions a technology entry can look like:

.. code-block:: json

    {
        "existing": {"TechA": 15.0},
        "carry_over_sizes": {"TechA": {"Interval_1": 10.0, "Interval_2": 5.0}},
        "remaining_lifetime": {"TechA": {"Interval_1": 5, "Interval_2": 15}},
        "carry_over_capex": {"TechA": {"Interval_1": 0.8, "Interval_2": 0.5}},
        "remaining_econ_lifetime": {"TechA": {"Interval_1": 5, "Interval_2": 15}}
    }

Pre-existing (brownfield) capacities in the first interval
----------------------------------------------------------

Capacities that already exist at the start of the pathway are treated as **sunk cost**:
they are tracked as a carry_over named ``{interval}_initial`` but no ``carry_over_capex``
entry is written for them. To account for their (partially elapsed) lifetime, write a
``remaining_lifetime`` entry for these components manually in the first interval's
``Technologies.json`` / ``Networks.json``. If no entry is given, the full lifetime from
the component data is assumed at the first transition.

Costs of carried-over investments
---------------------------------

Within one interval's optimization, existing capacities do not pay investment costs
(they are sunk for the MILP). Over the pathway, however, the annualized investment cost
of a carry_over is still being paid until the end of its economic lifetime. This is
accounted for in post-processing: the annualized CAPEX of each carry_over is frozen at its
build interval (``carry_over_capex``) and carried forward while the economic lifetime is
running. After solving all intervals, it can be added to the summary file:

.. testcode::

    add_values_to_summary(summary_path)
    add_carry_over_annualization_to_summary(summary_path, casestudy_path, intervals)

This adds the columns ``cost_annualization_carry_over_tecs``,
``cost_annualization_carry_over_netw``, ``cost_annualization`` and
``total_cost_with_carry_over_annualization`` to the summary and to each interval's
``optimization_results.h5``. It requires one summary row per interval, in the same
order as ``intervals``.

To compare intervals on a present-value basis, the per-interval costs can be discounted
back to the first interval using the global discount rate:

.. testcode::

    add_discounted_cost_to_summary(
        summary_path, casestudy_path, intervals, intervals_between_years
    )

This adds ``year_offset``, ``discount_factor``, ``discounted_total_cost`` and (if
present) ``discounted_total_cost_with_carry_over_annualization``. A global discount rate
must be set in ``ConfigModel.json`` (``global_discountrate`` different from ``-1``); the
reference interval's rate is applied to the whole horizon.

Decommissioning of carried capacity
-----------------------------------

If existing technologies or networks are allowed to decommission
(``"decommission": "continuous"`` or ``"only_complete"``), the capacity the optimizer
keeps in an interval is reconciled with the tracked carry_overs at the next transition.
The decommissioned amount (tracked total minus the solved ``_existing`` size, computed
**after** the lifetime expiry check) is removed **oldest carry_over first (FIFO)**. The
annualized CAPEX of a partially decommissioned carry_over is **prorated** to its
surviving size, and a fully decommissioned carry_over stops paying CAPEX (the MILP
charges the decommissioning cost separately, within the interval).

.. note::
    Because all carry_overs of a technology share a single ``_existing`` block in the
    optimization, retirement cannot be attributed to a specific carry_over; FIFO keeps
    the youngest, still-CAPEX-paying vintages.

.. note::
    Current limitations of the multiyear approach:

    - All intervals must model the same fraction of the year (e.g. all full-year
      runs): the stored carry_over CAPEX values are scaled with the fraction of the
      year modelled at their build interval.
    - CCS units added to a technology are not carried over between intervals. To
      include CCS in a multiyear analysis, define the plant with CCS as a separate
      technology.
    - Carry_overs of a technology share a single ``_existing`` block per interval, so
      decommissioning is reconciled at the aggregate level (oldest carry_over first)
      and cannot be attributed to a specific build vintage.