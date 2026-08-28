..   _multiyear:

Multiyear analysis with myopic foresight
=============================================

To represent long-term transitions toward net-zero systems, the model supports a multi-period optimization framework
using a myopic foresight approach. Each investment interval (e.g., decade) is formulated as an independent optimization
problem, while the installed capacities of networks and technologies are carried forward to the next interval. Within
each interval, perfect foresight is assumed, but decisions are myopic across intervals. This approach enables sequential
optimization of transition pathways, reflecting dynamic investment and decommissioning decisions under evolving boundary
conditions, while keeping computational complexity manageable.

Each interval has its own case study folder, from which the model reads the relevant input data, and the intervals are
solved one after the other in chronological order. Each interval is given its own result folder, so that the results of
the single steps stay separate.

Between two intervals, the capacities of the previous solution are carried over and tracked per **carry_over**: each
investment keeps its own size and remaining lifetime, and expires when its technical lifetime runs out. The annualized
investment cost of each carry_over is frozen at its build interval and accounted for in post-processing until the end
of its economic lifetime, so that the total cost of the pathway includes the investments of the past intervals.
Capacities that already exist at the start of the pathway are treated as sunk cost. The sections below describe the
pathway loop, what is tracked, and how the costs of the pathway are reported.

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
-------------------------------

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

The two are kept apart because they answer different questions: ``lifetime`` is the
depreciation period over which the investment is paid back, while ``technical_lifetime``
is how long the asset is physically available.

.. note::
    Outside a multiyear analysis only ``lifetime`` is used: within a single optimization
    there is no transition at which a component could expire, so ``technical_lifetime``
    has no effect.

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

A CCS unit is carried over together with the technology it is installed on: its size is
scaled by the share of host capacity that survived and stored under ``existing`` as
``{"size": ..., "ccs_size": ...}``, and its annualized CAPEX is added to the
``carry_over_capex`` of the host, following the same rules.

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

This adds the column ``total_cost_with_carry_over_annualization`` (the interval's
objective value plus the annualized capex of its carry_overs) to the summary and to each
interval's ``optimization_results.h5``, together with the two components it is made of,
``cost_annualization_carry_over_tecs`` and ``cost_annualization_carry_over_netws``.

The undiscounted cost of the pathway is added as ``cumulative_total_cost``: the running
sum of ``total_cost_with_carry_over_annualization`` up to and including each interval, so
that the last row holds the cost of the whole horizon.

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

The net present value is added as ``npv``: as for ``cumulative_total_cost``, it is the
running sum of the discounted costs up to and including each interval, so that the last
row holds the net present value of the whole pathway.

.. note::
    ``npv`` is based on the post-processed cost, i.e. it includes the annualized capex
    of the components carried over from earlier intervals, and therefore requires
    ``add_carry_over_annualization_to_summary`` to have been run first. If it was not,
    a warning is raised and ``npv`` falls back to the running sum of
    ``discounted_total_cost``.

In summary, the pathway results reported are the objective value of each interval
(``total_cost``), the post-processed cost per interval
(``total_cost_with_carry_over_annualization``) and cumulated over the pathway
(``cumulative_total_cost``), both undiscounted, their discounted counterparts
(``discounted_total_cost``,
``discounted_total_cost_with_carry_over_annualization``) and the net present value
(``npv``).

Decommissioning of carried capacity
-----------------------------------

If existing technologies or networks are allowed to decommission
(``"decommission": "continuous"`` or ``"only_complete"``), the capacity the optimizer
keeps in an interval is reconciled with the tracked carry_overs at the next transition.
The decommissioned amount (tracked total minus the solved ``_existing`` size, computed
**after** the lifetime expiry check) is removed **oldest carry_over first (FIFO)**. The
annualized CAPEX of a partially decommissioned carry_over is reduced in proportion to
its remaining size, and a fully decommissioned carry_over stops paying CAPEX (the MILP
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
    - Carry_overs of a technology share a single ``_existing`` block per interval, so
      decommissioning is reconciled at the aggregate level (oldest carry_over first)
      and cannot be attributed to a specific build vintage.
