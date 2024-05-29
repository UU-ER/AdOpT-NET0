from ..component import ModelComponent
from ..utilities import (
    annualize,
    set_discount_rate,
    perform_disjunct_relaxation,
    determine_variable_scaling,
    determine_constraint_scaling,
)
from ...logger import log_event

import pandas as pd
import copy
import pyomo.environ as pyo
import pyomo.gdp as gdp


class Network(ModelComponent):
    """
    Class to read and manage data for networks

    For each connection between nodes, an arc is created, with its respective cost, flows, losses and \
    consumption at nodes.

    Networks that can be used in two directions (e.g. electricity cables), are called bidirectional and are \
    treated respectively with their size and costs. Other networks, e.g. pipelines, require two installations \
    to be able to transport in two directions. As such their CAPEX is double.

    **Set declarations:**

    - Set of network carrier (i.e. only one carrier, that is transported in the network)
    - Set of all arcs (from_node, to_node)
    - In case the network is bidirectional: Set of unique arcs (i.e. for each pair of arcs, one unique entry)
    - Furthermore for each node:

        * A set of nodes it receives from
        * A set of nodes it sends to

    **Parameter declarations:**

    - Min Size (for each arc)
    - Max Size (for each arc)
    - :math:`{\gamma}_1, {\gamma}_2, {\gamma}_3, {\gamma}_4` for CAPEX calculation  \
      (annualized from given data on up-front CAPEX, lifetime and discount rate)
    - Variable OPEX
    - Fixed OPEX
    - Network losses (in % per km and flow) :math:`{\mu}`
    - Minimum transport (% of rated capacity)
    - Parameters for energy consumption at receiving and sending node

    **Variable declarations:**

    - CAPEX: ``var_capex``
    - Variable OPEX: ``var_opex_variable``
    - Fixed OPEX: ``var_opex_fixed``
    - Furthermore for each node:

        * Inflow to node (as a sum of all inflows from other nodes): ``var_inflow``
        * Outflow from node (as a sum of all outflows toother nodes): ``var_outflow``
        * Consumption of other carriers (e.g. electricity required for compression of a gas): ``var_consumption``

    **Arc Block declaration**

    Each arc represents a connection between two nodes, and is thus indexed by (node_from, node_to). For each arc,
    the following components are defined. Each variable is indexed by the timestep :math:`t` (here left out
    for convinience).

    - Decision Variables:

        * Size :math:`S`
        * Flow :math:`flow`
        * Losses :math:`loss`
        * CAPEX: :math:`CAPEX`
        * Variable :math:`OPEXvariable`
        * Fixed :math:`OPEXfixed`
        * If consumption at nodes exists for network:

          * Consumption at sending node :math:`Consumption_{nodeFrom}`
          * Consumption at receiving node :math:`Consumption_{nodeTo}`

    - Constraint definitions

        * Flow losses:

          .. math::
            loss = flow * {\mu}

        * Flow constraints:

          .. math::
            S * minTransport \leq flow \leq S

        * Consumption at sending and receiving node:

          .. math::
            Consumption_{nodeFrom} = flow * k_{1, send} + flow * distance * k_{2, send}

          .. math::
            Consumption_{nodeTo} = flow * k_{1, receive} + flow * distance * k_{2, receive}

        * CAPEX of respective arc. The CAPEX is calculated as follows:

          .. math::
            CAPEX_{arc} = {\gamma}_1 + {\gamma}_2 * S + {\gamma}_3 * distance + {\gamma}_4 * S * distance

        * Variable OPEX:

          .. math::
            OPEXvariable_{arc} = CAPEX_{arc} * opex_{variable}

    **Constraint declarations**
    This part calculates variables for all respective nodes and enforces constraints for bi-directional networks.

    - If network is bi-directional, the sizes in both directions are equal, and only one direction of flow is
      possible in each time step:

      .. math::
        S_{nodeFrom, nodeTo} = S_{nodeTo, nodeFrom}\forall unique arcs

      .. math::
        flow_{nodeFrom, nodeTo} = 0 \lor flow_{nodeTo, nodeFrom} = 0

    - CAPEX calculation of whole network as a sum of CAPEX of all arcs. For bi-directional networks, each arc
      is only considered once, regardless of the direction of the arc.

    - OPEX fix, as fraction of total CAPEX

    - OPEX variable as a sum of variable OPEX for each arc

    - Total network costs as the sum of OPEX and CAPEX

    - Total inflow and outflow as a sum for each node:

      .. math::
        outflow_{node} = \sum_{nodeTo \in sendsto_{node}} flow_{node, nodeTo}

      .. math::
        inflow_{node} = \sum_{nodeFrom \in receivesFrom_{node}} flow_{nodeFrom, node} - losses_{nodeFrom, node}

    - Energy consumption of other carriers at each node.

    """

    def __init__(self, netw_data: dict):
        """
        Initializes network class from network data

        :param dict netw_data: technology data
        """
        super().__init__(netw_data)

        # General information
        self.connection = []
        self.distance = []
        self.size_max_arcs = []
        self.energy_consumption = {}

        # Technology Performance
        if self.component_options.energyconsumption:
            self._calculate_energy_consumption()

        self.set_nodes = []
        self.set_t = []

        self.scaling_factors = []
        if "ScalingFactors" in netw_data:
            self.scaling_factors = netw_data["ScalingFactors"]

    def fit_network_performance(self):
        """
        Fits network performance (bounds and coefficients).
        """
        input_parameters = self.input_parameters
        time_independent = {}

        # Size
        time_independent["size_min"] = input_parameters.size_min
        if not self.existing:
            time_independent["size_max"] = input_parameters.size_max
        else:
            time_independent["size_max"] = input_parameters.size_initial
            time_independent["size_initial"] = input_parameters.size_initial

        if self.existing == 0:
            if not isinstance(self.size_max_arcs, pd.DataFrame):
                # Use max size
                time_independent["size_max_arcs"] = pd.DataFrame(
                    time_independent["size_max"],
                    index=self.distance.index,
                    columns=self.distance.columns,
                )
            else:
                time_independent["size_max_arcs"] = self.size_max_arcs
        elif self.existing == 1:
            # Use initial size
            time_independent["size_max_arcs"] = time_independent["size_initial"]

        # Emissions
        time_independent["loss2emissions"] = input_parameters.performance_data[
            "loss2emissions"
        ]
        time_independent["emissionfactor"] = input_parameters.performance_data[
            "emissionfactor"
        ]

        # Other
        time_independent["rated_power"] = input_parameters.rated_power
        time_independent["min_transport"] = input_parameters.performance_data[
            "min_transport"
        ]
        time_independent["loss"] = input_parameters.performance_data["loss"]

        # Write to self
        self.processed_coeff.time_independent = time_independent

    def construct_netw_model(
        self, b_netw, data: dict, set_nodes, set_t_full, set_t_clustered
    ):
        """
        Adds a network as model block.

        :param b_netw: pyomo network block
        :param dict data: dict containing model information
        :param set_nodes: pyomo set containing all nodes
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with network model
        """
        # LOG
        log_event(f"\t - Constructing Network {self.name}")

        # NETWORK DATA
        config = data["config"]

        # MODELING TYPICAL DAYS
        self.component_options.modelled_with_full_res = True
        self.component_options.lower_res_than_full = False
        if config["optimization"]["typicaldays"]["method"]["value"] == 1:
            self.set_t = set_t_clustered
        elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
            self.set_t = set_t_full
        else:
            self.set_t = set_t_full

        self.set_nodes = set_nodes

        b_netw = self._define_possible_arcs(b_netw)

        if self.component_options.bidirectional:
            b_netw = self._define_unique_arcs(b_netw)

        b_netw = self._define_size(b_netw)
        b_netw = self._define_capex_parameters(b_netw, data)
        b_netw = self._define_opex_parameters(b_netw)
        b_netw = self._define_emission_vars(b_netw)
        b_netw = self._define_network_characteristics(b_netw)
        b_netw = self._define_inflow_vars(b_netw)
        b_netw = self._define_outflow_vars(b_netw)

        if self.component_options.energyconsumption:
            b_netw = self._define_energyconsumption_parameters(b_netw)

        def arc_block_init(b_arc, node_from, node_to):
            """
            Establish each arc as a block

            INDEXED BY: (from, to)
            - size
            - flow
            - losses
            - consumption at from node
            - consumption at to node
            """

            b_arc.big_m_transformation_required = 0
            b_arc = self._define_size_arc(b_arc, b_netw, node_from, node_to)
            b_arc = self._define_capex_arc(b_arc, b_netw, node_from, node_to)
            b_arc = self._define_flow(b_arc, b_netw)
            b_arc = self._define_opex_arc(b_arc, b_netw)
            b_arc = self._define_emissions_arc(b_arc, b_netw)

            if self.component_options.energyconsumption:
                b_arc = self._define_energyconsumption_arc(b_arc, b_netw)

            if b_arc.big_m_transformation_required:
                b_arc = perform_disjunct_relaxation(b_arc)

            # LOG
            log_event(
                f"\t\t - Constructing Arc {node_from} - {node_to} " f"completed",
                print_it=False,
            )

        b_netw.arc_block = pyo.Block(b_netw.set_arcs, rule=arc_block_init)

        # CONSTRAINTS FOR BIDIRECTIONAL NETWORKS
        if self.component_options.bidirectional:
            b_netw = self._define_bidirectional_constraints(b_netw)

        b_netw = self._define_capex_total(b_netw)
        b_netw = self._define_opex_total(b_netw)
        b_netw = self._define_inflow_constraints(b_netw)
        b_netw = self._define_outflow_constraints(b_netw)
        b_netw = self._define_emission_constraints(b_netw)

        if self.component_options.energyconsumption:
            b_netw = self._define_energyconsumption_total(b_netw)

        # LOG
        log_event(f"\t - Constructing Network {self.name} completed", print_it=False)

        return b_netw

    def _define_possible_arcs(self, b_netw):
        """
        Define all possible arcs that have a connection

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        connection = copy.deepcopy(self.connection[:])

        def init_arcs_set(set):
            for from_node in connection:
                for to_node in connection[from_node].index:
                    if connection.at[from_node, to_node] == 1:
                        yield [from_node, to_node]

        b_netw.set_arcs = pyo.Set(initialize=init_arcs_set)

        def init_nodesIn(set, node):
            for i, j in b_netw.set_arcs:
                if j == node:
                    yield i

        b_netw.set_receives_from = pyo.Set(self.set_nodes, initialize=init_nodesIn)

        def init_nodesOut(set, node):
            for i, j in b_netw.set_arcs:
                if i == node:
                    yield j

        b_netw.set_sends_to = pyo.Set(self.set_nodes, initialize=init_nodesOut)

        return b_netw

    def _define_unique_arcs(self, b_netw):
        """
        Define arcs that are unique (one arc per direction)

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        connection = copy.deepcopy(self.connection[:])

        def init_arcs_all(set):
            for from_node in connection:
                for to_node in connection[from_node].index:
                    if connection.at[from_node, to_node] == 1:
                        connection.at[to_node, from_node] = 0
                        yield [from_node, to_node]

        b_netw.set_arcs_unique = pyo.Set(initialize=init_arcs_all)

        return b_netw

    def _define_size(self, b_netw):
        """
        Defines parameters related to network size.

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        coeff_ti = self.processed_coeff.time_independent

        b_netw.para_size_min = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=coeff_ti["size_min"], mutable=True
        )

        if self.existing:
            # Parameters for initial size
            def init_size_initial(param, node_from, node_to):
                return coeff_ti["size_initial"].at[node_from, node_to]

            b_netw.para_size_initial = pyo.Param(
                b_netw.set_arcs,
                domain=pyo.NonNegativeReals,
                initialize=init_size_initial,
            )
            # Check if sizes in both direction are the same for bidirectional existing networks
            if self.component_options.bidirectional:
                for from_node in coeff_ti["size_initial"]:
                    for to_node in coeff_ti["size_initial"][from_node].index:
                        assert (
                            coeff_ti["size_initial"].at[from_node, to_node]
                            == coeff_ti["size_initial"].at[to_node, from_node]
                        )
        return b_netw

    def _define_capex_parameters(self, b_netw, data: dict):
        """
        Defines variables and parameters related to technology capex.

        :param b_netw: pyomo network block
        :param dict data: dict containing model information
        :return: pyomo network block
        """

        config = data["config"]
        economics = self.economics

        # CHECK FOR GLOBAL ECONOMIC OPTIONS
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]

        # CAPEX
        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        b_netw.para_capex_gamma1 = pyo.Param(
            domain=pyo.Reals,
            mutable=True,
            initialize=economics.capex_data["gamma1"] * annualization_factor,
        )
        b_netw.para_capex_gamma2 = pyo.Param(
            domain=pyo.Reals,
            mutable=True,
            initialize=economics.capex_data["gamma2"] * annualization_factor,
        )
        b_netw.para_capex_gamma3 = pyo.Param(
            domain=pyo.Reals,
            mutable=True,
            initialize=economics.capex_data["gamma3"] * annualization_factor,
        )
        b_netw.para_capex_gamma4 = pyo.Param(
            domain=pyo.Reals,
            mutable=True,
            initialize=economics.capex_data["gamma4"] * annualization_factor,
        )

        b_netw.var_capex = pyo.Var()

        return b_netw

    def _define_opex_parameters(self, b_netw):
        """
        Defines OPEX parameters (fixed and variable)

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        economics = self.economics

        b_netw.para_opex_variable = pyo.Param(
            domain=pyo.Reals, initialize=economics.opex_variable, mutable=True
        )
        b_netw.para_opex_fixed = pyo.Param(
            domain=pyo.Reals, initialize=economics.opex_fixed, mutable=True
        )

        b_netw.var_opex_variable = pyo.Var(self.set_t)
        b_netw.var_opex_fixed = pyo.Var()
        if self.existing:
            b_netw.para_decommissioning_cost = pyo.Param(
                domain=pyo.Reals, initialize=economics.decommission_cost, mutable=True
            )

        return b_netw

    def _define_emission_vars(self, b_netw):
        """
        Defines network emissions

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        b_netw.var_netw_emissions_pos = pyo.Var(
            self.set_t, self.set_nodes, domain=pyo.NonNegativeReals
        )

        return b_netw

    def _define_network_characteristics(self, b_netw):
        """
        Defines transported carrier, losses and minimum transport requirements

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        # Define set of transported carrier
        b_netw.set_netw_carrier = pyo.Set(
            initialize=[self.component_options.transported_carrier]
        )

        return b_netw

    def _define_inflow_vars(self, b_netw):
        """
        Defines network inflow (i.e. sum of inflow to one node)

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        b_netw.var_inflow = pyo.Var(
            self.set_t,
            b_netw.set_netw_carrier,
            self.set_nodes,
            domain=pyo.NonNegativeReals,
        )
        return b_netw

    def _define_outflow_vars(self, b_netw):
        """
        Defines network outflow (i.e. sum of outflow to one node)

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        b_netw.var_outflow = pyo.Var(
            self.set_t,
            b_netw.set_netw_carrier,
            self.set_nodes,
            domain=pyo.NonNegativeReals,
        )
        return b_netw

    def _define_energyconsumption_parameters(self, b_netw):
        """
        Constructs constraints for network energy consumption

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        # Set of consumed carriers
        b_netw.set_consumed_carriers = pyo.Set(
            initialize=list(self.energy_consumption.keys())
        )

        # Parameters
        def init_cons_send1(para, car):
            return self.energy_consumption[car]["send"]["k_flow"]

        b_netw.para_send_kflow = pyo.Param(
            b_netw.set_consumed_carriers, domain=pyo.Reals, initialize=init_cons_send1
        )

        def init_cons_send2(para, car):
            return self.energy_consumption[car]["send"]["k_flowDistance"]

        b_netw.para_send_kflowDistance = pyo.Param(
            b_netw.set_consumed_carriers, domain=pyo.Reals, initialize=init_cons_send2
        )

        def init_cons_receive1(para, car):
            return self.energy_consumption[car]["receive"]["k_flow"]

        b_netw.para_receive_kflow = pyo.Param(
            b_netw.set_consumed_carriers,
            domain=pyo.Reals,
            initialize=init_cons_receive1,
        )

        def init_cons_receive2(para, car):
            return self.energy_consumption[car]["receive"]["k_flowDistance"]

        b_netw.para_receive_kflowDistance = pyo.Param(
            b_netw.set_consumed_carriers,
            domain=pyo.Reals,
            initialize=init_cons_receive2,
        )

        # Consumption at each node
        b_netw.var_consumption = pyo.Var(
            self.set_t,
            b_netw.set_consumed_carriers,
            self.set_nodes,
            domain=pyo.NonNegativeReals,
        )

        return b_netw

    def _calculate_energy_consumption(self):
        """
        Fits the performance parameters for a network, i.e. the consumption at each node.
        """
        # Get energy consumption at nodes form file
        energycons = self.input_parameters.performance_data["energyconsumption"]

        for car in energycons:
            self.energy_consumption[car] = {}
            if energycons[car]["cons_model"] == 1:
                self.energy_consumption[car]["send"] = {}
                self.energy_consumption[car]["send"] = energycons[car]
                self.energy_consumption[car]["send"].pop("cons_model")
                self.energy_consumption[car]["receive"] = {}
                self.energy_consumption[car]["receive"]["k_flow"] = 0
                self.energy_consumption[car]["receive"]["k_flowDistance"] = 0
            elif energycons[car]["cons_model"] == 2:
                temp = energycons[car]
                self.energy_consumption[car]["send"] = {}
                self.energy_consumption[car]["send"]["k_flow"] = round(
                    temp["c"]
                    * temp["T"]
                    / temp["eta"]
                    / temp["LHV"]
                    * ((temp["p"] / 30) ** ((temp["gam"] - 1) / temp["gam"]) - 1),
                    4,
                )
                self.energy_consumption[car]["send"]["k_flowDistance"] = 0
                self.energy_consumption[car]["receive"] = {}
                self.energy_consumption[car]["receive"]["k_flow"] = 0
                self.energy_consumption[car]["receive"]["k_flowDistance"] = 0

    def _define_size_arc(self, b_arc, b_netw, node_from, node_to):
        """
        Defines the size of an arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :param str node_from: node from which arc comes
        :param str node_to: node to which arc goes
        :return: pyomo arc block
        """
        coeff_ti = self.processed_coeff.time_independent

        if self.component_options.size_is_int:
            size_domain = pyo.NonNegativeIntegers
        else:
            size_domain = pyo.NonNegativeReals

        b_arc.para_size_max = pyo.Param(
            domain=size_domain,
            initialize=coeff_ti["size_max_arcs"].at[node_from, node_to],
        )

        b_arc.distance = self.distance.at[node_from, node_to]

        if self.existing:
            # Existing network
            if not self.component_options.decommission:
                # Decommissioning not possible
                b_arc.var_size = pyo.Param(
                    domain=size_domain,
                    initialize=b_netw.para_size_initial[node_from, node_to],
                )
            else:
                # Decommissioning possible
                b_arc.var_size = pyo.Var(
                    domain=size_domain,
                    bounds=(
                        b_netw.para_size_min,
                        b_netw.para_size_initial[node_from, node_to],
                    ),
                )
        else:
            # New network
            b_arc.var_size = pyo.Var(
                domain=size_domain, bounds=(b_netw.para_size_min, b_arc.para_size_max)
            )

        return b_arc

    def _define_capex_arc(self, b_arc, b_netw, node_from, node_to):
        """
        Defines the capex of an arc and corresponding constraints

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :param str node_from: node from which arc comes
        :param str node_to: node to which arc goes
        :return: pyomo arc block
        """

        def calculate_max_capex():
            max_capex = (
                b_netw.para_capex_gamma1
                + b_netw.para_capex_gamma2 * b_arc.para_size_max
                + b_netw.para_capex_gamma3 * b_arc.distance
                + b_netw.para_capex_gamma4 * b_arc.para_size_max * b_arc.distance
            )
            return (0, max_capex)

        # CAPEX auxilliary (used to calculate theoretical CAPEX)
        # For new technologies, this is equal to actual CAPEX
        # For existing technologies it is used to calculate fixed OPEX
        b_arc.var_capex_aux = pyo.Var(bounds=calculate_max_capex())

        def init_capex(const):
            return (
                b_arc.var_capex_aux
                == b_netw.para_capex_gamma1
                + b_netw.para_capex_gamma2 * b_arc.var_size
                + b_netw.para_capex_gamma3 * b_arc.distance
                + b_netw.para_capex_gamma4 * b_arc.var_size * b_arc.distance
            )

        # CAPEX Variable
        if self.existing and not self.component_options.decommission:
            b_arc.var_capex = pyo.Param(domain=pyo.NonNegativeReals, initialize=0)
        else:
            b_arc.var_capex = pyo.Var(bounds=calculate_max_capex())

        # CAPEX aux:
        if self.existing and not self.component_options.decommission:
            b_arc.const_capex_aux = pyo.Constraint(rule=init_capex)
        else:
            b_arc.big_m_transformation_required = 1
            s_indicators = range(0, 2)

            def init_installation(dis, ind):
                if ind == 0:  # network not installed
                    dis.const_capex_aux = pyo.Constraint(expr=b_arc.var_capex_aux == 0)
                    dis.const_not_installed = pyo.Constraint(expr=b_arc.var_size == 0)
                else:  # network installed
                    dis.const_capex_aux = pyo.Constraint(rule=init_capex)

            b_arc.dis_installation = gdp.Disjunct(s_indicators, rule=init_installation)

            def bind_disjunctions(dis):
                return [b_arc.dis_installation[i] for i in s_indicators]

            b_arc.disjunction_installation = gdp.Disjunction(rule=bind_disjunctions)

        # CAPEX and CAPEX aux
        if self.existing and self.component_options.decommission:
            b_arc.const_capex = pyo.Constraint(
                expr=b_arc.var_capex
                == (b_netw.para_size_initial[node_from, node_to] - b_arc.var_size)
                * b_netw.para_decommissioning_cost
            )
        elif not self.existing:
            b_arc.const_capex = pyo.Constraint(
                expr=b_arc.var_capex == b_arc.var_capex_aux
            )

        return b_arc

    def _define_flow(self, b_arc, b_netw):
        """
        Defines the flow through one arc and respective losses

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        rated_capacity = self.input_parameters.rated_power
        coeff_ti = self.processed_coeff.time_independent

        b_arc.var_flow = pyo.Var(
            self.set_t,
            domain=pyo.NonNegativeReals,
            bounds=(
                b_netw.para_size_min * rated_capacity,
                b_arc.para_size_max * rated_capacity,
            ),
        )
        b_arc.var_losses = pyo.Var(
            self.set_t,
            domain=pyo.NonNegativeReals,
            bounds=(
                b_netw.para_size_min * rated_capacity,
                b_arc.para_size_max * rated_capacity,
            ),
        )

        # Losses
        def init_flowlosses(const, t):
            return (
                b_arc.var_losses[t]
                == b_arc.var_flow[t] * coeff_ti["loss"] * b_arc.distance
            )

        b_arc.const_flowlosses = pyo.Constraint(self.set_t, rule=init_flowlosses)

        # Flow-size-constraint
        def init_size_const_high(const, t):
            return b_arc.var_flow[t] <= b_arc.var_size * rated_capacity

        b_arc.const_flow_size_high = pyo.Constraint(
            self.set_t, rule=init_size_const_high
        )

        def init_size_const_low(const, t):
            return (
                b_arc.var_size * rated_capacity * coeff_ti["min_transport"]
                <= b_arc.var_flow[t]
            )

        b_arc.const_flow_size_low = pyo.Constraint(self.set_t, rule=init_size_const_low)
        return b_arc

    def _define_energyconsumption_arc(self, b_arc, b_netw):
        """
        Defines the energyconsumption for an arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        b_arc.var_consumption_send = pyo.Var(
            self.set_t,
            b_netw.set_consumed_carriers,
            domain=pyo.NonNegativeReals,
            bounds=(b_netw.para_size_min, b_arc.para_size_max),
        )
        b_arc.var_consumption_receive = pyo.Var(
            self.set_t,
            b_netw.set_consumed_carriers,
            domain=pyo.NonNegativeReals,
            bounds=(b_netw.para_size_min, b_arc.para_size_max),
        )

        # Sending node
        def init_consumption_send(const, t, car):
            return (
                b_arc.var_consumption_send[t, car]
                == b_arc.var_flow[t] * b_netw.para_send_kflow[car]
                + b_arc.var_flow[t]
                * b_netw.para_send_kflowDistance[car]
                * b_arc.distance
            )

        b_arc.const_consumption_send = pyo.Constraint(
            self.set_t, b_netw.set_consumed_carriers, rule=init_consumption_send
        )

        # Receiving node
        def init_consumption_receive(const, t, car):
            return (
                b_arc.var_consumption_receive[t, car]
                == b_arc.var_flow[t] * b_netw.para_receive_kflow[car]
                + b_arc.var_flow[t]
                * b_netw.para_receive_kflowDistance[car]
                * b_arc.distance
            )

        b_arc.const_consumption_receive = pyo.Constraint(
            self.set_t, b_netw.set_consumed_carriers, rule=init_consumption_receive
        )

        return b_arc

    def _define_opex_arc(self, b_arc, b_netw):
        """
        Defines OPEX per Arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        b_arc.var_opex_variable = pyo.Var(self.set_t)

        def init_opex_variable(const, t):
            return (
                b_arc.var_opex_variable[t]
                == b_arc.var_flow[t] * b_netw.para_opex_variable
            )

        b_arc.const_opex_variable = pyo.Constraint(self.set_t, rule=init_opex_variable)
        return b_arc

    def _define_emissions_arc(self, b_arc, b_netw):
        """
        defines emission per arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        coeff_ti = self.processed_coeff.time_independent

        b_arc.var_emissions = pyo.Var(self.set_t)

        def init_arc_emissions(const, t):
            return (
                b_arc.var_emissions[t]
                == b_arc.var_flow[t] * coeff_ti["emissionfactor"]
                + b_arc.var_losses[t] * coeff_ti["loss2emissions"]
            )

        b_arc.const_arc_emissions = pyo.Constraint(self.set_t, rule=init_arc_emissions)

        return b_arc

    def _define_bidirectional_constraints(self, b_netw):
        """
        Defines constraints necessary, in case one arc can transport in two directions.

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        # Size in both direction is the same
        if self.component_options.decommission or not self.existing:

            def init_size_bidirectional(const, node_from, node_to):
                return (
                    b_netw.arc_block[node_from, node_to].var_size
                    == b_netw.arc_block[node_to, node_from].var_size
                )

            b_netw.const_size_bidirectional = pyo.Constraint(
                b_netw.set_arcs_unique, rule=init_size_bidirectional
            )

        s_indicators = range(0, 2)

        # Cut according to Germans work
        def init_cut_bidirectional(const, t, node_from, node_to):
            return (
                b_netw.arc_block[node_from, node_to].var_flow[t]
                + b_netw.arc_block[node_to, node_from].var_flow[t]
                <= b_netw.arc_block[node_from, node_to].var_size
            )

        b_netw.const_cut_bidirectional = pyo.Constraint(
            self.set_t, b_netw.set_arcs_unique, rule=init_cut_bidirectional
        )

        # Disjunction
        if self.component_options.bidirectional_precise:
            self.big_m_transformation_required = 1

            def init_bidirectional(dis, t, node_from, node_to, ind):
                if ind == 0:

                    def init_bidirectional1(const):
                        return b_netw.arc_block[node_from, node_to].var_flow[t] == 0

                    dis.const_flow_zero = pyo.Constraint(rule=init_bidirectional1)

                else:

                    def init_bidirectional2(const):
                        return b_netw.arc_block[node_to, node_from].var_flow[t] == 0

                    dis.const_flow_zero = pyo.Constraint(rule=init_bidirectional2)

            b_netw.dis_one_direction_only = gdp.Disjunct(
                self.set_t,
                b_netw.set_arcs_unique,
                s_indicators,
                rule=init_bidirectional,
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t, node_from, node_to):
                return [
                    b_netw.dis_one_direction_only[t, node_from, node_to, i]
                    for i in s_indicators
                ]

            b_netw.disjunction_one_direction_only = gdp.Disjunction(
                self.set_t, b_netw.set_arcs_unique, rule=bind_disjunctions
            )

        return b_netw

    def _define_capex_total(self, b_netw):
        """
        Defines total CAPEX of network

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        if self.component_options.bidirectional:
            arc_set = b_netw.set_arcs_unique
        else:
            arc_set = b_netw.set_arcs

        def init_capex(const):
            return (
                sum(b_netw.arc_block[arc].var_capex for arc in arc_set)
                == b_netw.var_capex
            )

        b_netw.const_capex = pyo.Constraint(rule=init_capex)

        return b_netw

    def _define_opex_total(self, b_netw):
        """
        Defines total OPEX of network

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        if self.component_options.bidirectional:
            arc_set = b_netw.set_arcs_unique
        else:
            arc_set = b_netw.set_arcs

        def init_opex_fixed(const):
            return (
                b_netw.para_opex_fixed
                * sum(b_netw.arc_block[arc].var_capex_aux for arc in arc_set)
                == b_netw.var_opex_fixed
            )

        b_netw.const_opex_fixed = pyo.Constraint(rule=init_opex_fixed)

        def init_opex_variable(const, t):
            return (
                sum(
                    b_netw.arc_block[arc].var_opex_variable[t]
                    for arc in b_netw.set_arcs
                )
                == b_netw.var_opex_variable[t]
            )

        b_netw.const_opex_var = pyo.Constraint(self.set_t, rule=init_opex_variable)
        return b_netw

    def _define_inflow_constraints(self, b_netw):
        """
        Connects the arc flows to inflow at each node

        :param b_netw: pyomo network block
        :return: pyomo network block
        """

        def init_inflow(const, t, car, node):
            return b_netw.var_inflow[t, car, node] == sum(
                b_netw.arc_block[from_node, node].var_flow[t]
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
                b_netw.arc_block[node, to_node].var_flow[t]
                for to_node in b_netw.set_sends_to[node]
            )

        b_netw.const_outflow = pyo.Constraint(
            self.set_t, b_netw.set_netw_carrier, self.set_nodes, rule=init_outflow
        )
        return b_netw

    def _define_emission_constraints(self, b_netw):
        """
        Defines Emissions from network

        :param b_netw: pyomo network block
        :return: pyomo network block
        """

        def init_netw_emissions(const, t, node):
            return b_netw.var_netw_emissions_pos[t, node] == sum(
                b_netw.arc_block[from_node, node].var_emissions[t]
                for from_node in b_netw.set_receives_from[node]
            )

        b_netw.const_netw_emissions = pyo.Constraint(
            self.set_t, self.set_nodes, rule=init_netw_emissions
        )
        return b_netw

    def _define_energyconsumption_total(self, b_netw):
        """
        Defines network consumption at each node

        :param b_netw: pyomo network block
        :return: pyomo network block
        """

        def init_network_consumption(const, t, car, node):
            return b_netw.var_consumption[t, car, node] == sum(
                b_netw.arc_block[node, to_node].var_consumption_send[t, car]
                for to_node in b_netw.set_sends_to[node]
            ) + sum(
                b_netw.arc_block[from_node, node].var_consumption_receive[t, car]
                for from_node in b_netw.set_receives_from[node]
            )

        b_netw.const_netw_consumption = pyo.Constraint(
            self.set_t,
            b_netw.set_consumed_carriers,
            self.set_nodes,
            rule=init_network_consumption,
        )

        return b_netw

    def write_results_netw_design(self, h5_group, model_block):
        """
        Function to report network design

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        coeff_ti = self.processed_coeff.time_independent

        for arc_name in model_block.set_arcs:
            arc = model_block.arc_block[arc_name]
            str = "".join(arc_name)
            arc_group = h5_group.create_group(str)

            arc_group.create_dataset("network", data=self.name)
            arc_group.create_dataset("fromNode", data=arc_name[0])
            arc_group.create_dataset("toNode", data=arc_name[1])
            arc_group.create_dataset("size", data=arc.var_size.value)
            arc_group.create_dataset("capex", data=arc.var_capex.value)
            arc_group.create_dataset(
                "opex_fixed",
                data=[model_block.para_opex_fixed.value * arc.var_capex_aux.value],
            )
            arc_group.create_dataset(
                "opex_variable",
                data=sum(arc.var_opex_variable[t].value for t in self.set_t),
            )
            arc_group.create_dataset(
                "total_flow", data=sum(arc.var_flow[t].value for t in self.set_t)
            )
            total_emissions = (
                sum(arc.var_flow[t].value for t in self.set_t)
                * coeff_ti["emissionfactor"]
                + sum(arc.var_losses[t].value for t in self.set_t)
                * coeff_ti["loss2emissions"]
            )
            arc_group.create_dataset("total_emissions", data=total_emissions)

    def write_results_netw_operation(self, h5_group, model_block):
        """
        Function to report network operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        for arc_name in model_block.set_arcs:
            arc = model_block.arc_block[arc_name]
            str = "".join(arc_name)
            arc_group = h5_group.create_group(str)

            arc_group.create_dataset(
                "flow", data=[arc.var_flow[t].value for t in self.set_t]
            )
            arc_group.create_dataset(
                "losses", data=[arc.var_losses[t].value for t in self.set_t]
            )

            if arc.find_component("var_consumption_send"):
                for car in model_block.set_consumed_carriers:

                    arc_group.create_dataset(
                        "consumption_send" + car,
                        data=[
                            arc.var_consumption_send[t, car].value for t in self.set_t
                        ],
                    )
                    arc_group.create_dataset(
                        "consumption_receive" + car,
                        data=[
                            arc.var_consumption_receive[t, car].value
                            for t in self.set_t
                        ],
                    )

    def scale_model(self, b_netw, model, config: dict):
        """
        Scales network model

        :param b_netw: pyomo network block
        :param model: pyomo model
        :param dict config: config dict containing scaling factors
        :return: pyomo model
        """

        f = self.scaling_factors
        f_global = config["scaling_factors"]

        model = determine_variable_scaling(model, b_netw, f, f_global)
        model = determine_constraint_scaling(model, b_netw, f, f_global)

        for arc in b_netw.arc_block:
            b_arc = b_netw.arc_block[arc]

            model = determine_variable_scaling(model, b_arc, f, f_global)
            model = determine_constraint_scaling(model, b_arc, f, f_global)

        return model
