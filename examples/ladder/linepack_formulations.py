"""Opt-in, integer-equivalent tightenings for the ladder pipeline models.

These experiments modify a *constructed* model, leaving the production component
classes and input files unchanged.  Apply one variant before passing the model to
the solver; use a fresh model (or an unmodified clone) for each comparison.

``hull`` completes the convex hull of ``linepack = installed * k * p_average``.
``signed`` links the two directional pressure drops to the physical pressure
difference once per pipeline, instead of relaxing each direction separately.
``combined`` applies both changes. ``combined_cuts`` additionally tightens
linepack bounds using the sum of the directional pressure drops. None of these
experiments removes SOS2 adjacency, direction decisions, nodal coupling, or the
temporal linepack balance.
"""

# Standalone handoff for the linepack_modelling branch, commit
# 6f3959dff744c71e43dc46fd5f9290559e9da306.
# Combines formulation_experiments.py and magnitude_pipeline_experiments.py;
# the formulation functions are unchanged. No other experimental files needed.
# Use run_linepack_formulations.py for baseline / combined_cuts / magnitude_cuts.

import math

import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables

from adopt_net0.components.networks.specificNetworks.fluidynamic_pipeline import (
    FluidynamicPipeline,
)


VARIANTS = ("baseline", "hull", "signed", "combined", "combined_cuts")


def _pipeline_blocks(model, network_data):
    """Yield ``(network object, model block)`` pairs for dynamic pipelines only."""
    for period, networks in network_data.items():
        for name, network in networks.items():
            if isinstance(network, FluidynamicPipeline):
                yield network, model.periods[period].network_block[name]


def _validate(network, block):
    """Check the bounds and expected components before modifying any model rows."""
    coefficients = network.processed_coeff.time_independent
    p_min = float(coefficients["pressure_min"])
    p_max = float(coefficients["pressure_max"])
    if not math.isfinite(p_min) or not math.isfinite(p_max) or p_min < 0:
        raise ValueError("Pipeline pressure bounds must be finite and nonnegative.")
    if p_max < p_min:
        raise ValueError("Pipeline pressure_max must be at least pressure_min.")
    for name in (
        "const_linepack_high",
        "const_no_flow_pressure_high",
        "const_no_flow_pressure_low",
    ):
        if not hasattr(block, name):
            raise ValueError(f"Pipeline block {block.name} lacks {name}.")
    if hasattr(block, "const_signed_pressure_high"):
        raise ValueError(f"Pipeline block {block.name} has already been tightened.")
    for node_from, node_to in block.set_arcs_unique:
        k = float(coefficients["linepack"].at[node_from, node_to])
        if not math.isfinite(k) or k < 0:
            raise ValueError(
                "Pipeline linepack coefficients must be finite and nonnegative."
            )
    for arc in block.arc_block.values():
        for name in ("const_delta_pressure_high", "const_delta_pressure_low"):
            if not hasattr(arc, name):
                raise ValueError(f"Pipeline arc {arc.name} lacks {name}.")


def _complete_linepack_hull(network, block):
    """Replace the weak upper facet, keeping all integer linepack states feasible.

    The original rows already enforce ``k*p_min*y <= L <= k*p_max*y`` and
    ``L >= k*p_average - k*p_max*(1-y)``.  The missing fourth product-hull
    facet is ``L <= k*p_average - k*p_min*(1-y)``.  It reduces to the original
    equality when built; when unbuilt, ``L=0`` and ``p_average>=p_min`` make it
    valid.  A zero linepack coefficient remains supported.
    """
    coefficients = network.processed_coeff.time_independent
    for t, node_from, node_to in block.const_linepack_high:
        arc = (node_from, node_to)
        k = float(coefficients["linepack"].at[arc])
        installed = block.arc_block[arc].var_installed
        average_pressure = (
            network._get_pressure(block, t, arc, node_from)
            + network._get_pressure(block, t, arc, node_to)
        ) / 2
        block.const_linepack_high[t, node_from, node_to].set_value(
            block.var_linepack[t, node_from, node_to]
            <= k * average_pressure - k * coefficients["pressure_min"] * (1 - installed)
        )
    return len(block.const_linepack_high)


def _link_signed_pressure(network, block):
    """Replace direction-specific pressure gates by one signed pipe relation.

    Let ``d = p_from - p_to``, ``s = delta_forward - delta_reverse``, and
    ``D = p_max - p_min``.  The replacement is ``-D*(1-y) <= d-s <= D*(1-y)``.
    A missing reverse arc contributes zero.  In any built integer solution,
    exactly one direction or neither can carry flow, so the old rows imply
    ``d=s``.  An unbuilt pipe has both interpolation sums zero and its endpoint
    pressures already imply ``|d|<=D``.  Thus all integer states are preserved,
    including idle and unidirectional pipes.  For fractional directions on a
    built pipe the replacement still enforces ``d=s``, strengthening the old
    relaxation without changing its PWL curve.
    """
    coefficients = network.processed_coeff.time_independent
    span = float(coefficients["pressure_max"] - coefficients["pressure_min"])

    def residual(t, node_from, node_to):
        arc = (node_from, node_to)
        signed_drop = block.arc_block[arc].var_delta_pressure[t]
        if (node_to, node_from) in block.set_arcs:
            signed_drop -= block.arc_block[node_to, node_from].var_delta_pressure[t]
        return (
            network._get_pressure(block, t, arc, node_from)
            - network._get_pressure(block, t, arc, node_to)
            - signed_drop
        )

    def high_rule(_, t, node_from, node_to):
        installed = block.arc_block[node_from, node_to].var_installed
        return residual(t, node_from, node_to) <= span * (1 - installed)

    def low_rule(_, t, node_from, node_to):
        installed = block.arc_block[node_from, node_to].var_installed
        return residual(t, node_from, node_to) >= -span * (1 - installed)

    block.const_signed_pressure_high = pyo.Constraint(
        network.set_t, block.set_arcs_unique, rule=high_rule
    )
    block.const_signed_pressure_low = pyo.Constraint(
        network.set_t, block.set_arcs_unique, rule=low_rule
    )

    deactivated = 0
    for arc in block.arc_block.values():
        for component in (arc.const_delta_pressure_high, arc.const_delta_pressure_low):
            deactivated += sum(row.active for row in component.values())
            component.deactivate()
    for component in (
        block.const_no_flow_pressure_high,
        block.const_no_flow_pressure_low,
    ):
        deactivated += sum(row.active for row in component.values())
        component.deactivate()
    added = len(block.const_signed_pressure_high) + len(block.const_signed_pressure_low)
    return deactivated, added


def _add_inventory_drop_cuts(network, block):
    """Tighten inventory bounds without changing interpolation or directions.

    In every integer solution, ``delta_forward + delta_reverse`` equals
    ``abs(p_from-p_to)`` for an installed pipe and is zero for an unbuilt one.
    The endpoint pressure bounds therefore imply
    ``k*p_min*y + k*sum_delta/2 <= linepack <= k*p_max*y-k*sum_delta/2``.
    The inequalities also remain valid with no reverse arc or with ``k=0``.
    """
    coefficients = network.processed_coeff.time_independent

    def terms(t, node_from, node_to):
        forward = block.arc_block[node_from, node_to]
        drop = forward.var_delta_pressure[t]
        if (node_to, node_from) in block.set_arcs:
            drop += block.arc_block[node_to, node_from].var_delta_pressure[t]
        k = float(coefficients["linepack"].at[node_from, node_to])
        return k, forward.var_installed, drop

    def low_rule(_, t, node_from, node_to):
        k, installed, drop = terms(t, node_from, node_to)
        return block.var_linepack[t, node_from, node_to] >= (
            k * coefficients["pressure_min"] * installed + k * drop / 2
        )

    def high_rule(_, t, node_from, node_to):
        k, installed, drop = terms(t, node_from, node_to)
        return block.var_linepack[t, node_from, node_to] <= (
            k * coefficients["pressure_max"] * installed - k * drop / 2
        )

    block.const_linepack_drop_low = pyo.Constraint(
        network.set_t, block.set_arcs_unique, rule=low_rule
    )
    block.const_linepack_drop_high = pyo.Constraint(
        network.set_t, block.set_arcs_unique, rule=high_rule
    )
    return len(block.const_linepack_drop_low) + len(block.const_linepack_drop_high)


def apply_formulation(model, network_data, variant="baseline"):
    """Apply a controlled pipeline formulation experiment and return row counts.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        The fully constructed AdOpT model (before creating its solver instance).
    network_data : mapping
        ``pyhub.data.network_data``, indexed by investment period and network name.
    variant : {"baseline", "hull", "signed", "combined", "combined_cuts"}
        ``baseline`` leaves the model unchanged.  Other variants modify only
        ``FluidynamicPipeline`` blocks; fixed-size reference networks are skipped.
        ``combined_cuts`` adds two valid inventory/drop rows per pipe-hour to
        ``combined``, retaining both original directional SOS2 sets and binaries.

    Returns
    -------
    dict
        Variant, affected network count, replaced linepack-row count, and
        deactivated/added pressure-row counts.  Repeating the same non-baseline
        variant is idempotent.  Switching an already modified model to a different
        variant raises ``ValueError`` to prevent invalid benchmark comparisons.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown formulation {variant!r}; choose one of {VARIANTS}.")
    previous = getattr(model, "_linepack_formulation_experiment", None)
    if previous is not None:
        if previous["variant"] == variant:
            return dict(previous)
        raise ValueError("Use a fresh model when changing formulation variants.")

    pipelines = list(_pipeline_blocks(model, network_data))
    result = {
        "variant": variant,
        "networks": len(pipelines),
        "linepack_rows_replaced": 0,
        "pressure_rows_deactivated": 0,
        "pressure_rows_added": 0,
    }
    if variant == "combined_cuts":
        result["inventory_drop_cuts_added"] = 0
    if variant == "baseline":
        return result

    # Validate every block first, so malformed input cannot cause a partial change.
    for network, block in pipelines:
        _validate(network, block)
    for network, block in pipelines:
        if variant in ("hull", "combined", "combined_cuts"):
            result["linepack_rows_replaced"] += _complete_linepack_hull(network, block)
        if variant in ("signed", "combined", "combined_cuts"):
            deactivated, added = _link_signed_pressure(network, block)
            result["pressure_rows_deactivated"] += deactivated
            result["pressure_rows_added"] += added
        if variant == "combined_cuts":
            result["inventory_drop_cuts_added"] += _add_inventory_drop_cuts(
                network, block
            )

    model._linepack_formulation_experiment = dict(result)
    return result


def _validate_magnitude(network, block):
    coefficients = network.processed_coeff.time_independent
    drops = list(map(float, coefficients["breakpoints_delta_pressure"]))
    if (
        len(drops) < 2
        or drops[0] != 0
        or any(not math.isfinite(value) for value in drops)
        or any(right <= left for left, right in zip(drops, drops[1:]))
    ):
        raise ValueError(
            "Magnitude interpolation requires increasing knots starting at zero."
        )
    if hasattr(block, "magnitude_pipe"):
        raise ValueError("Magnitude formulation has already been applied.")
    for arc in block.set_arcs_unique:
        flow = list(map(float, coefficients["breakpoints_flow"][arc]))
        if (
            len(flow) != len(drops)
            or flow[0] != 0
            or any(not math.isfinite(value) for value in flow)
            or any(right <= left for left, right in zip(flow, flow[1:]))
        ):
            raise ValueError(
                "Magnitude interpolation requires increasing flow knots from zero."
            )
        reverse = (arc[1], arc[0])
        if reverse in block.set_arcs:
            reverse_flow = list(map(float, coefficients["breakpoints_flow"][reverse]))
            if flow != reverse_flow:
                raise ValueError(
                    "Magnitude interpolation requires identical directional curves."
                )
        for direction in (arc, reverse):
            if direction not in block.set_arcs:
                continue
            directed = block.arc_block[direction]
            if any(value.fixed for value in directed.var_lambda.values()):
                raise ValueError(
                    "Apply magnitude formulation before fixing interpolation weights."
                )
            if any(value.fixed for value in directed.var_direction.values()):
                raise ValueError(
                    "Apply magnitude formulation before fixing direction choices."
                )


def _validate_external_references(model, pipelines):
    """Reject extensions depending on replaced interpolation/direction coordinates."""
    auxiliary_ids = set()
    standard_rows = set()
    standard_sos = set()
    for _, block in pipelines:
        for arc in block.arc_block.values():
            for component in (arc.var_direction, arc.var_lambda):
                auxiliary_ids.update(id(variable) for variable in component.values())
            standard_sos.update(id(sos) for sos in arc.const_lambda_sos2.values())
            for name in (
                "const_lambda",
                "const_interpolation_flow",
                "const_interpolation_delta_pressure",
                "const_delta_pressure_high",
                "const_delta_pressure_low",
                "const_flow_direction_in",
                "const_flow_direction_out",
            ):
                standard_rows.update(id(row) for row in getattr(arc, name).values())
        for name in (
            "const_direction_unique",
            "const_no_flow_pressure_high",
            "const_no_flow_pressure_low",
        ):
            standard_rows.update(id(row) for row in getattr(block, name).values())
    for row in model.component_data_objects(pyo.Constraint, active=True):
        if id(row) not in standard_rows and any(
            id(variable) in auxiliary_ids for variable in identify_variables(row.body)
        ):
            raise ValueError(
                f"Custom constraint {row.name} refers to magnitude direction/weight coordinates."
            )
    for objective in model.component_data_objects(pyo.Objective, active=True):
        if any(
            id(variable) in auxiliary_ids
            for variable in identify_variables(objective.expr)
        ):
            raise ValueError(
                f"Custom objective {objective.name} refers to magnitude direction/weight coordinates."
            )
    for sos in model.component_data_objects(pyo.SOSConstraint, active=True):
        if id(sos) not in standard_sos and any(
            id(variable) in auxiliary_ids for variable, _ in sos.get_items()
        ):
            raise ValueError(
                f"Custom SOS {sos.name} refers to magnitude direction/weight coordinates."
            )


def apply_magnitude_formulation(
    model, network_data, inventory_cuts=False, endpoint_hull=False, ratio_cuts=False
):
    """Apply to a fresh model and return formulation metadata.

    Includes the complete inventory product hull and signed endpoint-pressure
    linkage.  With ``inventory_cuts=True``, additionally impose the valid endpoint
    pressure/inventory inequalities conditional on the magnitude pressure drop.
    ``endpoint_hull=True`` implies those cuts and jointly links the two endpoint
    inventories to their pressures, replacing separate average/drop gates.

    For each pipe, ``sum(lambda)=y``, ``q_forward+q_reverse=sum(F*lambda)``, and
    ``delta_forward+delta_reverse=sum(dP*lambda)``.  The sum of directions equals
    ``y``; directional flow gates and ``delta<=max(dP)*direction`` force the
    inactive side to zero.  Thus a built integer pipe uses exactly the same PWL
    segments, endpoint capacities, and permitted flow signs as the original.

    ``ratio_cuts=True`` adds, for each direction, ``r_min*delta <= q <= r_max*delta``
    with ``r_min``/``r_max`` the smallest/largest ``F/B`` over the breakpoints with a
    nonzero drop. Every point of the PWL curve satisfies them, as the curve starts at
    the origin, so they only stop the relaxation from putting the flow in one
    direction and the pressure drop in the other.
    """
    inventory_cuts = bool(inventory_cuts or endpoint_hull)
    endpoint_hull = bool(endpoint_hull)
    ratio_cuts = bool(ratio_cuts)
    previous = getattr(model, "_magnitude_pipeline_experiment", None)
    if previous is not None:
        if (
            previous["inventory_cuts"] == inventory_cuts
            and previous["endpoint_hull"] == endpoint_hull
            and previous["ratio_cuts"] == ratio_cuts
        ):
            return dict(previous)
        raise ValueError("Use a fresh model when changing magnitude variants.")
    pipelines = list(_pipeline_blocks(model, network_data))
    for network, block in pipelines:
        _validate_magnitude(network, block)
    _validate_external_references(model, pipelines)
    base = apply_formulation(model, network_data, "combined")
    result = {
        "variant": (
            "magnitude_hull"
            if endpoint_hull
            else ("magnitude_cuts" if inventory_cuts else "magnitude")
        ),
        "inventory_cuts": inventory_cuts,
        "endpoint_hull": endpoint_hull,
        "ratio_cuts": ratio_cuts,
        "ratio_cut_rows_added": 0,
        "networks": len(pipelines),
        "base_tightening": base,
        "direction_binaries_relaxed": 0,
        "old_sos2_removed": 0,
        "new_sos2_added": 0,
        "old_weights_removed": 0,
        "new_weights_added": 0,
        "endpoint_hull_rows_added": 0,
        "separate_pressure_inventory_rows_deactivated": 0,
        "zero_linepack_pipes_skipped": 0,
    }
    for network, block in pipelines:
        coefficients = network.processed_coeff.time_independent
        drops = list(map(float, coefficients["breakpoints_delta_pressure"]))
        max_drop = max(drops)

        def pipe_rule(pipe, node_from, node_to):
            arc = (node_from, node_to)
            forward = block.arc_block[arc]
            reverse_arc = (node_to, node_from)
            reverse = (
                block.arc_block[reverse_arc] if reverse_arc in block.set_arcs else None
            )
            directions = [forward] if reverse is None else [forward, reverse]
            flow_knots = list(map(float, coefficients["breakpoints_flow"][arc]))
            pipe.knots = pyo.Set(initialize=range(len(drops)))
            pipe.var_lambda = pyo.Var(network.set_t, pipe.knots, bounds=(0, 1))
            pipe.const_sum = pyo.Constraint(
                network.set_t,
                rule=lambda _, t: sum(pipe.var_lambda[t, z] for z in pipe.knots)
                == forward.var_installed,
            )
            pipe.const_flow = pyo.Constraint(
                network.set_t,
                rule=lambda _, t: sum(part.var_flow[t] for part in directions)
                == sum(flow_knots[z] * pipe.var_lambda[t, z] for z in pipe.knots),
            )
            pipe.const_drop = pyo.Constraint(
                network.set_t,
                rule=lambda _, t: sum(part.var_delta_pressure[t] for part in directions)
                == sum(drops[z] * pipe.var_lambda[t, z] for z in pipe.knots),
            )
            pipe.const_sos2 = pyo.SOSConstraint(
                network.set_t,
                var=pipe.var_lambda,
                index={t: [(t, z) for z in pipe.knots] for t in network.set_t},
                weights={(t, z): z + 1 for t in network.set_t for z in pipe.knots},
                sos=2,
            )
            result["new_sos2_added"] += len(pipe.const_sos2)
            result["new_weights_added"] += len(pipe.var_lambda)
            for t in network.set_t:
                block.const_direction_unique[t, node_from, node_to].set_value(
                    sum(part.var_direction[t] for part in directions)
                    == forward.var_installed
                )
                redundant = forward if reverse is None else reverse
                redundant.var_direction[t].domain = pyo.UnitInterval
                result["direction_binaries_relaxed"] += 1
            for directed in directions:
                result["old_sos2_removed"] += len(directed.const_lambda_sos2)
                result["old_weights_removed"] += len(directed.var_lambda)
                for name in (
                    "const_lambda",
                    "const_interpolation_delta_pressure",
                    "const_interpolation_flow",
                    "const_lambda_sos2",
                ):
                    directed.del_component(name)
                # No active row, result writer, or physical variable refers to
                # the old interpolation weights. Remove them so persistent
                # interfaces never transfer unused/fixed columns to the solver.
                directed.del_component("var_lambda")
                directed.del_component("set_breakpoints")
                directed.const_magnitude_drop_gate = pyo.Constraint(
                    network.set_t,
                    rule=lambda _, t, part=directed: part.var_delta_pressure[t]
                    <= max_drop * part.var_direction[t],
                )
            if ratio_cuts:
                if any(drops[z] <= 0 and flow_knots[z] != 0 for z in range(len(drops))):
                    raise ValueError(
                        "Ratio cuts need every zero-drop breakpoint at zero flow."
                    )
                ratios = [
                    flow_knots[z] / drops[z] for z in range(len(drops)) if drops[z] > 0
                ]
                r_min, r_max = min(ratios), max(ratios)
                pipe.const_ratio = pyo.ConstraintList()
                for directed in directions:
                    for t in network.set_t:
                        pipe.const_ratio.add(
                            directed.var_flow[t]
                            >= r_min * directed.var_delta_pressure[t]
                        )
                        pipe.const_ratio.add(
                            directed.var_flow[t]
                            <= r_max * directed.var_delta_pressure[t]
                        )
                result["ratio_cut_rows_added"] += len(pipe.const_ratio)
            if inventory_cuts:
                k = float(coefficients["linepack"].at[arc])
                p_min = float(coefficients["pressure_min"])
                p_max = float(coefficients["pressure_max"])
                pipe.const_inventory_low = pyo.Constraint(
                    network.set_t,
                    rule=lambda _, t: block.var_linepack[t, node_from, node_to]
                    >= k * p_min * forward.var_installed
                    + (k / 2) * sum(part.var_delta_pressure[t] for part in directions),
                )
                pipe.const_inventory_high = pyo.Constraint(
                    network.set_t,
                    rule=lambda _, t: block.var_linepack[t, node_from, node_to]
                    <= k * p_max * forward.var_installed
                    - (k / 2) * sum(part.var_delta_pressure[t] for part in directions),
                )
                if endpoint_hull and k > 0:
                    # H_i = L + k*s/2 and H_j = L - k*s/2, where s is
                    # forward minus reverse drop. In any integer built state,
                    # H_i=k*p_i and H_j=k*p_j; in the off state both are zero.
                    # Inventory/drop cuts already bound each H between
                    # k*p_min*y and k*p_max*y. These four remaining McCormick
                    # facets complete the endpoint products, without 1/k.
                    pipe.const_endpoint_hull = pyo.ConstraintList()
                    for t in network.set_t:
                        signed_drop = forward.var_delta_pressure[t]
                        if reverse is not None:
                            signed_drop -= reverse.var_delta_pressure[t]
                        inventory = block.var_linepack[t, node_from, node_to]
                        for node, sign in ((node_from, 1), (node_to, -1)):
                            endpoint_inventory = (
                                inventory + sign * (k / 2) * signed_drop
                            )
                            pressure = network._get_pressure(block, t, arc, node)
                            pipe.const_endpoint_hull.add(
                                endpoint_inventory
                                >= k * pressure
                                - k * p_max * (1 - forward.var_installed)
                            )
                            pipe.const_endpoint_hull.add(
                                endpoint_inventory
                                <= k * pressure
                                - k * p_min * (1 - forward.var_installed)
                            )
                        # Sum the endpoint links to recover the old inventory
                        # gates; subtract opposing links to recover the signed
                        # pressure gates. Therefore these rows are redundant.
                        for component in (
                            block.const_linepack_high,
                            block.const_linepack_low,
                            block.const_signed_pressure_high,
                            block.const_signed_pressure_low,
                        ):
                            component[t, node_from, node_to].deactivate()
                            result["separate_pressure_inventory_rows_deactivated"] += 1
                    result["endpoint_hull_rows_added"] += len(pipe.const_endpoint_hull)
                elif endpoint_hull:
                    # With k=0 the endpoint equations carry no pressure
                    # information; keep the signed pressure gates unchanged.
                    result["zero_linepack_pipes_skipped"] += 1

        block.magnitude_pipe = pyo.Block(block.set_arcs_unique, rule=pipe_rule)
    model._magnitude_pipeline_experiment = dict(result)
    return result
