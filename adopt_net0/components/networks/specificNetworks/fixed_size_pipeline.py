import pyomo.environ as pyo
import numpy as np
import pandas as pd

from ..genericNetworks.fluid import Fluid


class FixedSizePipeline(Fluid):
    """
    Pipeline network of a fixed diameter

    This network type extends the fluid network with a pipeline geometry. The
    diameter, roughness and fluid properties are fixed per network, i.e. one json
    file is used per pipeline type, while the length of each arc comes from the
    network topology. An arc is either not built or built with the capacity that
    follows from the geometry, so the size of an arc is not a continuous decision:
    the choice of a capacity is the choice of a pipeline type.

    The capacity of an arc is the flow at the largest pressure difference the
    pipeline may see, i.e. between ``pressure_max`` and ``pressure_ref``, derived
    from the quasi-dynamic pipeline equation. It therefore depends on the length of
    the arc: the same pipeline type carries less over a longer distance.

    The transport itself is the one of the parent class, i.e. the flow of an arc is
    bounded by its capacity and nothing else. :class:`FluidynamicPipeline` derives
    from this class and replaces that with the pipeline equation and the linepack,
    so the two are identical in what they may build and differ only in how the fluid
    moves. That is what makes them a pair worth comparing.

    **Arc Block declaration**

    - Decision Variables:

        * ``var_installed``: 1 if the arc is built, 0 otherwise

    - Constraint definitions:

        * Size, fixed to the capacity of the pipeline type if the arc is built:

          .. math::
            size = installed * size_{max}

    .. note::
        The coefficients are derived in SI units and the conversion factors are
        hardcoded in :func:`fit_network_performance`, so the input data has to be
        given as ``diameter`` and ``roughness`` in m, ``pressure_min``,
        ``pressure_max`` and ``pressure_ref`` in bar, ``temperature`` in K,
        ``molar_mass`` in kg/kmol and the distance of the network topology in km.
        The flow of a pipeline follows from the mass flow in t/h, but the rest of
        the model works in energy terms, so it is converted with
        ``energy_density``, the energy density of the fluid in MWh/t. The flow and
        the size of an arc are then in MW, consistent with the energy balance and
        with the compressors.
    """

    def __init__(self, netw_data: dict):
        """
        Constructor

        :param dict netw_data: network data
        """
        super().__init__(netw_data)

        self.diameter = netw_data["Performance"]["diameter"]
        self.roughness = netw_data["Performance"]["roughness"]
        self.pressure_min = netw_data["Performance"]["pressure_min"]
        self.pressure_max = netw_data["Performance"]["pressure_max"]
        self.pressure_ref = netw_data["Performance"]["pressure_ref"]
        self.temperature = netw_data["Performance"]["temperature"]
        self.compressibility_factor = netw_data["Performance"]["compressibility_factor"]
        self.molar_mass = netw_data["Performance"]["molar_mass"]
        self.energy_density = netw_data["Performance"]["energy_density"]
        self.nr_breakpoints = netw_data["Performance"]["nr_breakpoints"]

        self.nr_timesteps_averaged = 1
        self.hours_per_typical_day = 0

    def fit_network_performance(self):
        """
        Fits network performance for a pipeline of a fixed diameter (bounds and
        coefficients).

        The coefficient of the quasi-dynamic pipeline equation and the linepack
        coefficient are derived from the pipeline geometry and the fluid properties.
        They are defined per arc, as they depend on the length of the arc. The
        breakpoints of the piecewise linear relation are calculated per arc as well,
        and the maximum size of an arc is not read from the json file, but taken as
        the flow at the largest pressure difference.
        """
        super(FixedSizePipeline, self).fit_network_performance()

        time_independent = self.processed_coeff.time_independent

        # Arcs without a connection have a distance of zero and are not constructed
        distance = self.distance.replace(0, np.nan)

        # Fluid and pipeline properties
        r_specific = 8314 / self.molar_mass  # J/kg/K
        friction_factor = (2 * np.log10(3.7 * self.diameter / self.roughness)) ** -2
        volume_per_km = np.pi * self.diameter**2 / 4 * 1000  # m3/km

        # Coefficient of the quasi-dynamic pipeline equation in MW^2/bar^2,
        # f * |f| = R * (p_from^2 - p_to^2), with the distance in km. The mass flow
        # is converted to an energy flow with the energy density of the fluid, so
        # the squared coefficient carries it squared as well
        r_per_km = (
            self.energy_density**2
            * 3.6**2
            * np.pi**2
            / 16
            * self.diameter**5
            * 1e10
            / (
                friction_factor
                * 1000
                * self.compressibility_factor
                * r_specific
                * self.temperature
            )
        )
        time_independent["r_pipeline"] = r_per_km / distance

        # Linepack coefficient in MWh/bar, i.e. density times volume times energy
        # density, with the density given by p / (Z * R_s * T) and the distance in km
        linepack_per_km = (
            self.energy_density
            * volume_per_km
            * 100
            / (self.compressibility_factor * r_specific * self.temperature)
        )
        time_independent["linepack"] = linepack_per_km * self.distance

        # Pressure
        time_independent["pressure_min"] = self.pressure_min
        time_independent["pressure_max"] = self.pressure_max

        # Breakpoints of the piecewise linear relation, with the lower pressure
        # fixed at the reference pressure
        time_independent["breakpoints_delta_pressure"] = np.linspace(
            0, self.pressure_max - self.pressure_ref, self.nr_breakpoints
        )
        time_independent["breakpoints_flow"] = {}
        for node_from in distance.index:
            for node_to in distance.columns:
                if np.isnan(distance.at[node_from, node_to]):
                    continue
                time_independent["breakpoints_flow"][(node_from, node_to)] = np.sqrt(
                    time_independent["r_pipeline"].at[node_from, node_to]
                    * time_independent["breakpoints_delta_pressure"]
                    * (
                        2 * self.pressure_ref
                        + time_independent["breakpoints_delta_pressure"]
                    )
                )

        # Size of an arc, i.e. the flow at the largest pressure difference
        time_independent["size_max_arcs"] = pd.DataFrame(
            np.nan, index=distance.index, columns=distance.columns
        )
        for arc in time_independent["breakpoints_flow"]:
            time_independent["size_max_arcs"].at[arc] = max(
                time_independent["breakpoints_flow"][arc]
            )

        self.processed_coeff.time_independent = time_independent

    def _define_size_arc(self, b_arc, b_netw, node_from: str, node_to: str):
        """
        Defines the size of an arc

        The diameter of the pipeline is fixed, so the size of an arc is either zero
        or the capacity of the pipeline type.

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :param str node_from: node from which arc comes
        :param str node_to: node to which arc goes
        :return: pyomo arc block
        """
        super(FixedSizePipeline, self)._define_size_arc(
            b_arc, b_netw, node_from, node_to
        )

        b_arc.var_installed = pyo.Var(within=pyo.Binary)

        if self.existing:
            b_arc.const_installed = pyo.Constraint(expr=b_arc.var_installed == 1)

        b_arc.const_size_installed = pyo.Constraint(
            expr=b_arc.var_size == b_arc.var_installed * b_arc.para_size_max
        )

        return b_arc

    def _define_capex_constraints_arc(self, b_arc, b_netw, node_from, node_to):
        """
        Defines the capex of an arc and corresponding constraints

        As the size of an arc is the product of a binary variable and a parameter,
        the cost components that are independent of the size can be charged with the
        binary variable directly, and no disjunction is required.

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :param str node_from: node from which arc comes
        :param str node_to: node to which arc goes
        :return: pyomo arc block
        """

        def init_capex(const):
            return (
                b_arc.var_capex_aux
                == b_arc.para_capex_gamma1 * b_arc.var_installed
                + b_arc.para_capex_gamma2 * b_arc.var_size
                + b_arc.para_capex_gamma3 * b_arc.distance * b_arc.var_installed
                + b_arc.para_capex_gamma4 * b_arc.var_size * b_arc.distance
            )

        b_arc.const_capex_aux = pyo.Constraint(rule=init_capex)

        if self.existing:
            b_arc.const_capex = pyo.Constraint(expr=b_arc.var_capex == 0)
        else:
            b_arc.const_capex = pyo.Constraint(
                expr=b_arc.var_capex == b_arc.var_capex_aux
            )

        return b_arc
