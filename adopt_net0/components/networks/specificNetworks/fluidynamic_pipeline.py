import pandas as pd
import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
import numpy as np
from warnings import warn

from .fixed_size_pipeline import FixedSizePipeline
from ...utilities import link_full_resolution_to_clustered


class FluidynamicPipeline(FixedSizePipeline):
    """
    Pipeline network with quasi-dynamic flow and linepack

    This network type extends :class:`FixedSizePipeline` with the quasi-dynamic
    pipeline equation, so that the amount of fluid stored in the pipeline itself
    (the linepack) can be used as a short term storage. The geometry, the fluid
    properties and the capacity of an arc are the ones of the parent class, so the
    two differ only in how the fluid moves: here the flow follows the pipeline
    equation and is coupled to the pressures at the end nodes, there it is bounded
    by the capacity of the arc and nothing else.

    The quasi-dynamic pipeline equation reads

    .. math::
        f * |f| = R * (p^2_{nodeFrom} - p^2_{nodeTo})

    with :math:`f` the average flow in the pipeline, :math:`p` the pressures at the
    end nodes and :math:`R` a coefficient containing the pipe and gas properties.
    It is approximated by a piecewise linear relation between the pressure
    difference and the flow, constructed around a reference pressure.

    **Parameter declarations:**

    - ``para_pressure_min``, ``para_pressure_max``: pressure limits at the nodes

    **Variable declarations:**

    - For each node:

        * ``var_pressure``: pressure at the node

    **Arc Block declaration**

    Each arc represents a connection between two nodes, and is thus indexed by (
    node_from, node_to). For each arc, the following components are defined. Each
    variable is indexed by the timestep :math:`t` (here left out for convenience).

    - Decision Variables:

        * ``var_installed``: 1 if the arc is built, 0 otherwise
        * ``var_direction``: 1 if the arc is the active direction of the pipeline
        * ``var_flow_in``: flow entering the arc at the sending node
        * ``var_flow_out``: flow leaving the arc at the receiving node
        * ``var_delta_pressure``: pressure difference over the arc
        * ``var_lambda``: interpolation variables of the piecewise linear relation

    For each unique arc, i.e. once per pipeline:

    - Decision Variables:

        * ``var_linepack``: amount of fluid stored in the pipeline

    - Constraint definitions:

        * Size, fixed to the capacity of the pipeline type if the arc is built:

          .. math::
            size = installed * size_{max}

        * Average flow in the pipeline:

          .. math::
            flow = (flow_{in} + flow_{out}) / 2

        * Linepack, proportional to the average pressure in the arc:

          .. math::
            linepack = k_{linepack} * (p_{nodeFrom} + p_{nodeTo}) / 2

        * Linepack balance, with the flows summed over both directions:

          .. math::
            linepack_{t} = linepack_{t-1} + dt * (flow_{in} - flow_{out})

        * Only one direction of a pipeline can be active in a timestep:

          .. math::
            direction_{nodeFrom, nodeTo} + direction_{nodeTo, nodeFrom} <= installed

        * Flow only in the active direction:

          .. math::
            flow_{in} <= direction * size_{max}

        * Pressure difference over the arc, in the active direction:

          .. math::
            deltapressure = p_{nodeFrom} - p_{nodeTo}

        * Piecewise linear relation between pressure difference and flow, with the
          breakpoints :math:`(dP_{z}, F_{z})` and an SOS2 condition on lambda:

          .. math::
            deltapressure = sum_{z} dP_{z} * lambda_{z}

          .. math::
            flow = sum_{z} F_{z} * lambda_{z}

          .. math::
            sum_{z} lambda_{z} = direction

    .. note::
        The coefficients are derived in SI units and the conversion factors are
        hardcoded in :func:`fit_network_performance`, so the input data has to be
        given as ``diameter`` and ``roughness`` in m, ``pressure_min``,
        ``pressure_max`` and ``pressure_ref`` in bar, ``temperature`` in K,
        ``molar_mass`` in kg/kmol and the distance of the network topology in km.
        The flow of a pipeline follows from the mass flow in t/h, but the rest of
        the model works in energy terms, so it is converted with
        ``energy_density``, the energy density of the fluid in MWh/t. The flow and
        the size of an arc are then in MW and the linepack is in MWh, consistent
        with the energy balance and with the compressors.

    .. note::
        The breakpoints are calculated before the optimization from the quasi-
        dynamic pipeline equation, with the lower pressure fixed at the reference
        pressure ``pressure_ref``:

        .. math::
            F_{z} = sqrt(R * dP_{z} * (2 * p_{ref} + dP_{z}))

        This gives a one dimensional relation, at the price of neglecting the
        dependency on the absolute pressure level. A two dimensional approximation
        in pressure difference and average pressure would remove this
        simplification.

    .. note::
        The pressure of the pipeline is kept separate from the pressures used by the
        compressors, which are fixed input data. The compressors see the reference
        pressure of the pipeline, while the pressure of the pipeline itself is free
        to move between ``pressure_min`` and ``pressure_max``.

    .. note::
        The linepack balance is cyclic, i.e. no initial linepack is imposed and the
        first time interval is coupled to the last one, as is done for the storage
        technologies. If the network is modelled with typical days (method 1), the
        balance is closed within each typical day instead, as the chronology is only
        preserved there. The linepack can then not be shifted from one typical day
        to the next, which is acceptable as it is an intra-day storage. Note that
        the approach used for storage technologies, i.e. keeping the storage level
        at full resolution and mapping the flows with the sequence of the typical
        days, cannot be used here: the linepack is not a free variable but is fixed
        by the pressures, so it cannot be at a higher resolution than the pressures,
        and the pressures in turn cannot be at a higher resolution than the flows
        they are linked to. (For now: don't use typical days)

    .. note::
        For a bidirectional network, the two directions of a pipeline are two
        separate arc blocks, as is the case for all networks. The flow and the
        pressure difference of a pipeline are thus already split in a positive and a
        negative part, one per arc block, and ``var_direction`` is the binary that
        enforces that only one of the two carries flow in a given timestep. The
        disjunction of the parent class is switched off, as it would enforce the
        same condition a second time. The linepack belongs to the pipeline and not
        to one of its directions, and is therefore defined once per unique arc.

    """

    def __init__(self, netw_data: dict):
        """
        Constructor

        :param dict netw_data: network data
        """
        super().__init__(netw_data)

        self.linepack_on = netw_data["Performance"].get("linepack_on", 1)
        self.pressure_coupled = netw_data["Performance"].get("pressure_coupled", 1)

        if self.bidirectional_network:
            # The direction is enforced per timestep with var_direction, so the
            # disjunction of the parent class is not needed
            self.bidirectional_network_precise = 0

    def construct_netw_model(
        self, b_netw, data: dict, set_nodes, set_t_full, set_t_clustered
    ):
        """
        Constructs a fluid dynamic pipeline as model block.

        :param b_netw: pyomo network block
        :param dict data: dict containing model information
        :param set_nodes: pyomo set containing all nodes
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with network model
        """
        config = data["config"]

        if not self.linepack_on:
            coeff_ti = self.processed_coeff.time_independent
            coeff_ti["linepack"] = coeff_ti["linepack"] * 0

        if config["optimization"]["timestaging"]["value"] != 0:
            self.nr_timesteps_averaged = config["optimization"]["timestaging"]["value"]
        else:
            self.nr_timesteps_averaged = 1

        # If the network is modelled with typical days, the linepack is balanced
        # within each typical day, as the chronology is only preserved there
        if config["optimization"]["typicaldays"]["N"]["value"] == 0:
            self.hours_per_typical_day = 0
        elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
            self.hours_per_typical_day = data["topology"]["hours_per_day"]["full"]
        else:
            self.hours_per_typical_day = 0

        b_netw = super(FluidynamicPipeline, self).construct_netw_model(
            b_netw, data, set_nodes, set_t_full, set_t_clustered
        )

        # The linepack and the direction of the flow are defined once per pipeline,
        # i.e. on the unique arcs, and refer to the arc blocks of both directions
        if b_netw.find_component("set_arcs_unique") is None:
            b_netw = self._define_unique_arcs(b_netw)

        b_netw = self._define_direction_constraints(b_netw)
        b_netw = self._define_linepack_constraints(b_netw)

        return b_netw

    def _define_energyconsumption_parameters(self, b_netw):
        """
        Constructs constraints for pipeline energy consumption

        .. note::
            The pressure at the nodes is defined here, as this is the last network
            wide method that is called before the arc blocks are constructed, and
            the arc constraints refer to it.

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        super(FluidynamicPipeline, self)._define_energyconsumption_parameters(b_netw)

        b_netw = self._define_pressure_vars(b_netw)

        return b_netw

    def _get_pressure_key(self, arc, node):
        """
        Returns the index the pressure of a node is defined on

        With ``pressure_coupled`` the pressure is indexed by the node alone, so every
        arc that touches it reads the same variable. Without it the index also carries
        the pipeline the pressure is seen from, and the two directions of a pipeline
        map to the same index.

        :param arc: the arc the pressure is seen from
        :param str node: the node
        :return: the index of the pressure
        """
        if self.pressure_coupled:
            return node

        node_from, node_to = sorted(arc)
        return f"{node_from}|{node_to}|{node}"

    def _get_pressure(self, b_netw, t, arc, node):
        """
        Returns the pressure of a node as the given arc sees it

        :param b_netw: pyomo network block
        :param t: timestep
        :param arc: the arc the pressure is seen from
        :param str node: the node
        :return: the pressure variable
        """
        return b_netw.var_pressure[t, self._get_pressure_key(arc, node)]

    def _define_pressure_vars(self, b_netw):
        """
        Defines the pressure at each node the network reaches

        .. note::
            The pressure belongs to the network and not to the node, so a node
            reached by two pipeline types has one pressure per type. Only the nodes
            of an arc appear in the pipeline equation and in the linepack, so the
            pressure is defined on those alone: a node the network does not reach
            would get a variable that no constraint touches, which stays unsolved.

        .. note::
            With ``pressure_coupled = 0`` the pressure is defined per pipeline end
            instead of per node, so the arcs that meet at a node no longer share it
            and the pressure does not propagate through the network. Each arc keeps
            its own pipeline equation and its own linepack, but at a pressure level
            of its own, so this is a diagnostic and not a transport model.

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        coeff_ti = self.processed_coeff.time_independent

        b_netw.para_pressure_min = pyo.Param(
            domain=pyo.NonNegativeReals,
            initialize=coeff_ti["pressure_min"],
            mutable=True,
        )
        b_netw.para_pressure_max = pyo.Param(
            domain=pyo.NonNegativeReals,
            initialize=coeff_ti["pressure_max"],
            mutable=True,
        )

        # At each node the pressure of a network is the same for every arc of the
        # network that touches it, which is what makes the pressure propagate. The
        # coupling is what pressure_coupled switches: without it the index carries the
        # pipeline as well, so every arc has a pair of end pressures of its own
        b_netw.set_pressure_nodes = pyo.Set(
            initialize=sorted(
                {
                    self._get_pressure_key(arc, node)
                    for arc in b_netw.set_arcs
                    for node in arc
                }
            )
        )

        b_netw.var_pressure = pyo.Var(
            self.set_t,
            b_netw.set_pressure_nodes,
            domain=pyo.NonNegativeReals,
            bounds=(b_netw.para_pressure_min, b_netw.para_pressure_max),
        )

        return b_netw

    def _define_flow(self, b_arc, b_netw):
        """
        Defines the flow through one arc and the respective losses

        The flow entering the arc differs from the flow leaving it by the linepack
        that is built up or released. The flow inherited from :class:`Network` is
        the average of the two, i.e. the average flow in the pipeline, and is used
        by the emission, opex and size constraints of the parent classes.

        For a bidirectional network the two directions of a pipeline are two
        separate arcs, so the flow of this arc is the flow in one direction only and
        is zero if the other direction is the active one.

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        super(FluidynamicPipeline, self)._define_flow(b_arc, b_netw)

        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]

        b_arc.var_flow_in = pyo.Var(
            self.set_t,
            domain=pyo.NonNegativeReals,
            bounds=(0, b_arc.para_size_max * rated_capacity),
        )
        b_arc.var_flow_out = pyo.Var(
            self.set_t,
            domain=pyo.NonNegativeReals,
            bounds=(0, b_arc.para_size_max * rated_capacity),
        )
        b_arc.var_direction = pyo.Var(self.set_t, within=pyo.Binary)

        def init_flow_average(const, t):
            return (
                b_arc.var_flow[t] == (b_arc.var_flow_in[t] + b_arc.var_flow_out[t]) / 2
            )

        b_arc.const_flow_average = pyo.Constraint(self.set_t, rule=init_flow_average)

        # Flow only in the active direction
        def init_flow_direction_in(const, t):
            return (
                b_arc.var_flow_in[t]
                <= b_arc.var_direction[t] * b_arc.para_size_max * rated_capacity
            )

        b_arc.const_flow_direction_in = pyo.Constraint(
            self.set_t, rule=init_flow_direction_in
        )

        def init_flow_direction_out(const, t):
            return (
                b_arc.var_flow_out[t]
                <= b_arc.var_direction[t] * b_arc.para_size_max * rated_capacity
            )

        b_arc.const_flow_direction_out = pyo.Constraint(
            self.set_t, rule=init_flow_direction_out
        )

        b_arc = self._define_pipeline_equation(b_arc, b_netw)

        return b_arc

    def _define_linepack_constraints(self, b_netw):
        """
        Defines the linepack of each pipeline and its temporal balance

        The linepack belongs to the pipeline and not to a direction of it, so it is
        defined once per unique arc. The flows of both directions enter the balance,
        of which at most one is non zero at each timestep.

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        coeff_ti = self.processed_coeff.time_independent

        def init_linepack_bounds(bounds, t, node_from, node_to):
            return (
                0,
                coeff_ti["linepack"].at[node_from, node_to] * coeff_ti["pressure_max"],
            )

        b_netw.var_linepack = pyo.Var(
            self.set_t,
            b_netw.set_arcs_unique,
            domain=pyo.NonNegativeReals,
            bounds=init_linepack_bounds,
        )

        def get_net_flow(t, node_from, node_to):
            """Net flow into the pipeline, summed over both directions"""
            net_flow = (
                b_netw.arc_block[node_from, node_to].var_flow_in[t]
                - b_netw.arc_block[node_from, node_to].var_flow_out[t]
            )
            if (node_to, node_from) in b_netw.set_arcs:
                net_flow += (
                    b_netw.arc_block[node_to, node_from].var_flow_in[t]
                    - b_netw.arc_block[node_to, node_from].var_flow_out[t]
                )
            return net_flow

        # An arc that is not built holds nothing. Without these two the linepack of
        # such an arc is a free constant: its bounds do not force it to zero, the two
        # constraints below are switched off by their own big-M, and the balance only
        # ties it to itself once the flow is zero. It would then float at no cost
        # whenever the installation variable is fractional, which is most of the
        # relaxation. Tying it to the installation instead also gives the built case
        # a real lower bound, as the pressure of a built pipeline is at least the
        # minimum one.
        def init_linepack_installed_high(const, t, node_from, node_to):
            linepack = coeff_ti["linepack"].at[node_from, node_to]
            return (
                b_netw.var_linepack[t, node_from, node_to]
                <= linepack
                * coeff_ti["pressure_max"]
                * b_netw.arc_block[node_from, node_to].var_installed
            )

        b_netw.const_linepack_installed_high = pyo.Constraint(
            self.set_t, b_netw.set_arcs_unique, rule=init_linepack_installed_high
        )

        def init_linepack_installed_low(const, t, node_from, node_to):
            linepack = coeff_ti["linepack"].at[node_from, node_to]
            return (
                b_netw.var_linepack[t, node_from, node_to]
                >= linepack
                * coeff_ti["pressure_min"]
                * b_netw.arc_block[node_from, node_to].var_installed
            )

        b_netw.const_linepack_installed_low = pyo.Constraint(
            self.set_t, b_netw.set_arcs_unique, rule=init_linepack_installed_low
        )

        # Linepack from the average pressure in the pipeline, only if it is built.
        # The upper one needs no big-M: the linepack is non negative and the average
        # pressure is positive, so it can never be violated by an arc that is not
        # built, whose linepack is zero by the constraint above.
        def init_linepack_high(const, t, node_from, node_to):
            linepack = coeff_ti["linepack"].at[node_from, node_to]
            return (
                b_netw.var_linepack[t, node_from, node_to]
                <= linepack
                * (
                    self._get_pressure(b_netw, t, (node_from, node_to), node_from)
                    + self._get_pressure(b_netw, t, (node_from, node_to), node_to)
                )
                / 2
            )

        b_netw.const_linepack_high = pyo.Constraint(
            self.set_t, b_netw.set_arcs_unique, rule=init_linepack_high
        )

        def init_linepack_low(const, t, node_from, node_to):
            linepack = coeff_ti["linepack"].at[node_from, node_to]
            return b_netw.var_linepack[t, node_from, node_to] >= linepack * (
                self._get_pressure(b_netw, t, (node_from, node_to), node_from)
                + self._get_pressure(b_netw, t, (node_from, node_to), node_to)
            ) / 2 - linepack * coeff_ti["pressure_max"] * (
                1 - b_netw.arc_block[node_from, node_to].var_installed
            )

        b_netw.const_linepack_low = pyo.Constraint(
            self.set_t, b_netw.set_arcs_unique, rule=init_linepack_low
        )

        # Linepack balance
        def init_linepack_balance(const, t, node_from, node_to):
            if self.hours_per_typical_day:
                if (t - 1) % self.hours_per_typical_day == 0:
                    # couple first and last time interval of the typical day
                    linepack_previous = b_netw.var_linepack[
                        t + self.hours_per_typical_day - 1, node_from, node_to
                    ]
                else:  # all other time intervals of the typical day
                    linepack_previous = b_netw.var_linepack[t - 1, node_from, node_to]
            elif t == 1:  # couple first and last time interval
                linepack_previous = b_netw.var_linepack[
                    max(self.set_t), node_from, node_to
                ]
            else:  # all other time intervals
                linepack_previous = b_netw.var_linepack[t - 1, node_from, node_to]

            return (
                b_netw.var_linepack[t, node_from, node_to]
                == linepack_previous
                + get_net_flow(t, node_from, node_to) * self.nr_timesteps_averaged
            )

        b_netw.const_linepack_balance = pyo.Constraint(
            self.set_t, b_netw.set_arcs_unique, rule=init_linepack_balance
        )

        return b_netw

    def _define_direction_constraints(self, b_netw):
        """
        Defines the constraints linking the two directions of a pipeline

        Both directions are built together, and at most one of them can be the
        active one at each timestep.

        :param b_netw: pyomo network block
        :return: pyomo network block
        """

        def init_installed_bidirectional(const, node_from, node_to):
            if (node_to, node_from) not in b_netw.set_arcs:
                return pyo.Constraint.Skip

            return (
                b_netw.arc_block[node_from, node_to].var_installed
                == b_netw.arc_block[node_to, node_from].var_installed
            )

        b_netw.const_installed_bidirectional = pyo.Constraint(
            b_netw.set_arcs_unique, rule=init_installed_bidirectional
        )

        def get_direction(t, node_from, node_to):
            """Sum of the direction variables of both directions of a pipeline"""
            direction = b_netw.arc_block[node_from, node_to].var_direction[t]
            if (node_to, node_from) in b_netw.set_arcs:
                direction += b_netw.arc_block[node_to, node_from].var_direction[t]
            return direction

        def init_direction_unique(const, t, node_from, node_to):
            return (
                get_direction(t, node_from, node_to)
                <= b_netw.arc_block[node_from, node_to].var_installed
            )

        b_netw.const_direction_unique = pyo.Constraint(
            self.set_t, b_netw.set_arcs_unique, rule=init_direction_unique
        )

        # If a built pipeline carries no flow, the pressures at its ends equalize.
        # As in the pipeline equation, an active direction cannot open the pressures
        # further than the largest breakpoint, so that is the coefficient it gets;
        # the whole range is only needed for an arc that is not built.
        coeff_ti = self.processed_coeff.time_independent
        delta_pressure_max = coeff_ti["pressure_max"] - coeff_ti["pressure_min"]
        delta_pressure_active = max(coeff_ti["breakpoints_delta_pressure"])
        delta_pressure_unbuilt = delta_pressure_max - delta_pressure_active

        def get_slack(t, node_from, node_to):
            """Pressure difference the arc may show in this timestep"""
            installed = b_netw.arc_block[node_from, node_to].var_installed
            return delta_pressure_active * (
                get_direction(t, node_from, node_to) + 1 - installed
            ) + delta_pressure_unbuilt * (1 - installed)

        def init_no_flow_pressure_high(const, t, node_from, node_to):
            return self._get_pressure(
                b_netw, t, (node_from, node_to), node_from
            ) - self._get_pressure(
                b_netw, t, (node_from, node_to), node_to
            ) <= get_slack(
                t, node_from, node_to
            )

        b_netw.const_no_flow_pressure_high = pyo.Constraint(
            self.set_t, b_netw.set_arcs_unique, rule=init_no_flow_pressure_high
        )

        def init_no_flow_pressure_low(const, t, node_from, node_to):
            return self._get_pressure(
                b_netw, t, (node_from, node_to), node_from
            ) - self._get_pressure(
                b_netw, t, (node_from, node_to), node_to
            ) >= -get_slack(
                t, node_from, node_to
            )

        b_netw.const_no_flow_pressure_low = pyo.Constraint(
            self.set_t, b_netw.set_arcs_unique, rule=init_no_flow_pressure_low
        )

        return b_netw

    def _define_pipeline_equation(self, b_arc, b_netw):
        """
        Defines the piecewise linear approximation of the quasi-dynamic pipeline
        equation for an arc

        The pressure difference and the average flow are both interpolated between
        the breakpoints with the same interpolation variables, on which an SOS2
        condition is imposed so that only two adjacent ones can be non zero. The
        interpolation variables sum up to the installation variable, so that the
        pressures at the two end nodes are decoupled if the arc is not built.

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        node_from, node_to = b_arc.index()
        coeff_ti = self.processed_coeff.time_independent

        breakpoints_delta_pressure = coeff_ti["breakpoints_delta_pressure"]
        breakpoints_flow = coeff_ti["breakpoints_flow"][(node_from, node_to)]
        delta_pressure_max = coeff_ti["pressure_max"] - coeff_ti["pressure_min"]

        b_arc.set_breakpoints = pyo.Set(
            initialize=range(0, len(breakpoints_delta_pressure))
        )

        b_arc.var_delta_pressure = pyo.Var(
            self.set_t,
            domain=pyo.NonNegativeReals,
            bounds=(0, max(breakpoints_delta_pressure)),
        )
        b_arc.var_lambda = pyo.Var(
            self.set_t,
            b_arc.set_breakpoints,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
        )

        # Pressure difference over the arc, only in the active direction.
        #
        # The big-M is split in two. On a built pipeline the difference cannot leave
        # [-dP, +dP] with dP the largest breakpoint: the active direction ties it to
        # var_delta_pressure, which is bounded by dP, and the inactive one is the
        # active direction of the opposite arc, which forces the difference to have
        # the other sign. Only an arc that is not built has its node pressures free
        # over the whole range.
        delta_pressure_active = max(breakpoints_delta_pressure)
        delta_pressure_unbuilt = delta_pressure_max - delta_pressure_active

        def init_delta_pressure_high(const, t):
            return b_arc.var_delta_pressure[t] <= self._get_pressure(
                b_netw, t, (node_from, node_to), node_from
            ) - self._get_pressure(
                b_netw, t, (node_from, node_to), node_to
            ) + delta_pressure_active * (
                1 - b_arc.var_direction[t]
            ) + delta_pressure_unbuilt * (
                1 - b_arc.var_installed
            )

        b_arc.const_delta_pressure_high = pyo.Constraint(
            self.set_t, rule=init_delta_pressure_high
        )

        def init_delta_pressure_low(const, t):
            return b_arc.var_delta_pressure[t] >= self._get_pressure(
                b_netw, t, (node_from, node_to), node_from
            ) - self._get_pressure(
                b_netw, t, (node_from, node_to), node_to
            ) - delta_pressure_active * (
                1 - b_arc.var_direction[t]
            ) - delta_pressure_unbuilt * (
                1 - b_arc.var_installed
            )

        b_arc.const_delta_pressure_low = pyo.Constraint(
            self.set_t, rule=init_delta_pressure_low
        )

        # Interpolation
        def init_lambda(const, t):
            return (
                sum(b_arc.var_lambda[t, z] for z in b_arc.set_breakpoints)
                == b_arc.var_direction[t]
            )

        b_arc.const_lambda = pyo.Constraint(self.set_t, rule=init_lambda)

        def init_interpolation_delta_pressure(const, t):
            return b_arc.var_delta_pressure[t] == sum(
                breakpoints_delta_pressure[z] * b_arc.var_lambda[t, z]
                for z in b_arc.set_breakpoints
            )

        b_arc.const_interpolation_delta_pressure = pyo.Constraint(
            self.set_t, rule=init_interpolation_delta_pressure
        )

        def init_interpolation_flow(const, t):
            return b_arc.var_flow[t] == sum(
                breakpoints_flow[z] * b_arc.var_lambda[t, z]
                for z in b_arc.set_breakpoints
            )

        b_arc.const_interpolation_flow = pyo.Constraint(
            self.set_t, rule=init_interpolation_flow
        )

        # Only two adjacent interpolation variables can be non zero
        b_arc.const_lambda_sos2 = pyo.SOSConstraint(
            self.set_t,
            var=b_arc.var_lambda,
            index={t: [(t, z) for z in b_arc.set_breakpoints] for t in self.set_t},
            weights={(t, z): z + 1 for t in self.set_t for z in b_arc.set_breakpoints},
            sos=2,
        )

        return b_arc

    def _define_inflow_constraints(self, b_netw):
        """
        Connects the arc flows to inflow at each node

        :param b_netw: pyomo network block
        :return: pyomo network block
        """

        def init_inflow(const, t, car, node):
            return b_netw.var_inflow[t, car, node] == sum(
                b_netw.arc_block[from_node, node].var_flow_out[t]
                - b_netw.arc_block[from_node, node].var_losses[t]
                for from_node in b_netw.set_receives_from[node]
            )

        b_netw.const_inflow = pyo.Constraint(
            self.set_t, b_netw.set_netw_carrier, self.set_nodes, rule=init_inflow
        )
        return b_netw

    def _define_outflow_constraints(self, b_netw):
        """
        Connects the arc flows to outflow at each node

        :param b_netw: pyomo network block
        :return: pyomo network block
        """

        def init_outflow(const, t, car, node):
            return b_netw.var_outflow[t, car, node] == sum(
                b_netw.arc_block[node, to_node].var_flow_in[t]
                for to_node in b_netw.set_sends_to[node]
            )

        b_netw.const_outflow = pyo.Constraint(
            self.set_t, b_netw.set_netw_carrier, self.set_nodes, rule=init_outflow
        )
        return b_netw

    def write_results_netw_operation(self, h5_group, model_block):
        super(FluidynamicPipeline, self).write_results_netw_operation(
            h5_group, model_block
        )

        for arc_name in model_block.set_arcs:
            arc = model_block.arc_block[arc_name]
            str = "".join(arc_name)
            arc_group = h5_group[str]

            arc_group.create_dataset(
                "flow_in", data=[arc.var_flow_in[t].value for t in self.set_t]
            )
            arc_group.create_dataset(
                "flow_out", data=[arc.var_flow_out[t].value for t in self.set_t]
            )
            arc_group.create_dataset(
                "delta_pressure",
                data=[arc.var_delta_pressure[t].value for t in self.set_t],
            )
            arc_group.create_dataset(
                "direction", data=[arc.var_direction[t].value for t in self.set_t]
            )

        for arc_name in model_block.set_arcs_unique:
            str = "".join(arc_name)
            arc_group = h5_group[str]

            arc_group.create_dataset(
                "linepack",
                data=[
                    model_block.var_linepack[(t,) + arc_name].value for t in self.set_t
                ],
            )

        for node in model_block.set_pressure_nodes:
            node_group = h5_group.create_group(node)
            node_group.create_dataset(
                "pressure",
                data=[model_block.var_pressure[t, node].value for t in self.set_t],
            )
