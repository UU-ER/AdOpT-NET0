from ..component import ModelComponent
from ..utilities import (
    annualize,
    set_discount_rate,
    perform_disjunct_relaxation,
    determine_variable_scaling,
    determine_constraint_scaling,
    get_attribute_from_dict,
)

import warnings
import pandas as pd
import copy
import pyomo.environ as pyo
import pyomo.gdp as gdp

import logging

log = logging.getLogger(__name__)


class Network(ModelComponent):
    """
    Class to read and manage data for networks

    For each connection between nodes, an arc is created, with its respective cost,
    flows, losses and  consumption at nodes.

    Networks that can be used in two directions (e.g. electricity cables), are called
    bidirectional and are treated respectively with their size and costs. Other
    networks, e.g. pipelines, require two installations to be able to transport in
    two directions. As such their CAPEX is double and their size in both directions
    can be different. The CAPEX parameters (gamma1,2,3,4) can either be specified in
    the json file of the network or for every arc. In the latter case, the user needs
    to create a csv file for every gamma (gamma1.csv etc) with a matrix-like structure (nodes
    as index and columns, similar to the distance.csv). These files need to be placed in the corresponding folder
    in "network_topology", e.g. my_case_study/period1/network_topology/new(or existing)/my_network_type.
    You can use the capex per arc if capex_defined_per_arc is set to 1 in the corresponding json file.

    **Set declarations:**

    - ``set_netw_carrier``: Set of network carrier (i.e. only one carrier, that is
      transported in the network)
    - ``set_arcs``: Set of all arcs (from_node, to_node)
    - ``set_arcs_unique``: In case the network is bidirectional: Set of unique arcs (i.e.
      for each pair of arcs, one unique entry)
    - Furthermore for each node:

        * ``set_receives_from``: A set of nodes the node receives from
        * ``set_sends_to``: A set of nodes the node sends to

    - ``set_consumed_carriers``: In case the network has an energy consumption

    **Parameter declarations:**

    - ``para_size_min``: Min Size (for each arc)
    - ``para_size_max``: Max Size (for each arc)
    - ``para_size_initial``, var_size, var_capex: for existing networks
    - ``para_capex_gamma``: :math:`{\\gamma}_1, {\\gamma}_2, {\\gamma}_3, {\\gamma}_4` for
      CAPEX calculation (annualized from given data on up-front CAPEX, lifetime and
      discount rate) (either for the network or for each arc)
    - ``para_opex_variable``: Variable OPEX
    - ``para_opex_fixed``: Fixed OPEX in % of up-front CAPEX
    - ``para_decommissioning_cost_annual``: decommissioning costs for existing networks

    **Variable declarations:**

    - ``var_capex``: CAPEX
    - ``var_opex_variable``: Variable OPEX
    - ``var_opex_fixed``: Fixed OPEX
    - Furthermore for each node:

        * ``var_netw_emissions_pos``: positive emissions at node
        * ``var_inflow``: Inflow to node (as a sum of all inflows from other nodes)
        * ``var_outflow``: Outflow from node (as a sum of all outflows to other nodes)
        * ``var_consumption``: Consumption of other carriers (e.g. electricity
          required for compression of a gas)

    **Arc Block declaration**

    Each arc represents a connection between two nodes, and is thus indexed by (
    node_from, node_to). For each arc, the following components are defined. Each
    variable is indexed by the timestep :math:`t` (here left out for convenience).

    - Decision Variables:

        * ``var_size``: Size :math:`S`
        * ``var_flow``: Flow :math:`flow`
        * ``var_losses``: Losses :math:`loss`
        * ``var_capex``, ``var_capex_aux`` CAPEX: :math:`CAPEX`
        * ``var_opex_variable``: Variable :math:`OPEXvariable`

    - Constraint definitions:

        * Flow losses:

          .. math::
            loss = flow * {\\mu} * D

        * Flow constraints:

          .. math::
            S * minTransport \\leq flow \\leq S

        * CAPEX of respective arc. The CAPEX is calculated as follows (for new
          networks). Note that for existing networks, the CAPEX is zero, but the
          fixed OPEX is calculated as a fraction of a hypothetical up-front CAPEX
          based on the existing size.

          .. math::
            CAPEX_{arc} = {\\gamma}_1 + {\\gamma}_2 * S + {\\gamma}_3 * distance + {\\gamma}_4 * S * distance

        * Variable OPEX:

          .. math::
            OPEXvariable_{arc} = CAPEX_{arc} * opex_{variable}


    **Network constraint declarations**
    This part calculates variables for all respective nodes.

    - CAPEX calculation of the whole network as a sum of CAPEX of all arcs. If
      ``bidirectional_network`` is set to 1 for this network, the capex and fixed
      opex for an arc is only counted once.

    - OPEX fix, as fraction of total CAPEX

    - OPEX variable as a sum of variable OPEX for each arc

    - Total emissions as the sum of all arc emissions

    - Total inflow and outflow as a sum for each node:

      .. math::
        outflow_{node} = \\sum_{nodeTo \\in sendsto_{node}} flow_{node, nodeTo}

      .. math::
        inflow_{node} = \\sum_{nodeFrom \\in receivesFrom_{node}} flow_{nodeFrom, node} - losses_{nodeFrom, node}

    - Energy consumption of other carriers at each node.

    - If  ``bidirectional_network`` is set to 1 for this network only additional
      constraints are enforced to ensure that at each time step a flow can only be in
      one direction. For ``bidirectional_network_precise = 0``, only a cut and a
      constraint on the sizes of the two directions of an arc are formulated:

      .. math::
        S_{nodeFrom, nodeTo} = S_{nodeTo, nodeFrom}

      .. math::
        flow_{nodeFrom, nodeTo} + flow_{nodeTo, nodeFrom} \\lor S_{nodeTo, nodeFrom}

      For ``bidirectional_network_precise = 1`` additional disjunctions are
      formulated, thus adding binaries and complexity to the model:

      .. math::
        flow_{nodeFrom, nodeTo} = 0 \\lor flow_{nodeTo, nodeFrom} = 0

    Existing networks, i.e. existing = 1, can be decommissioned (decommission = 'continuous' or decommission =
      'only_complete') or not (decommission = 'impossible').
      For networks that cannot be decommissioned, the size is fixed to the initial size given in the network
      data. For networks that can be decommissioned, the size can be smaller or equal to the initial size. When
      decommission = 'continuous' the size can take any value between the minimum and initial size. When decommission =
      'only_complete' the size is either 0 or the initial size. Reducing the size comes at the decommissioning costs or
      benefits specified in the economics of the network.

    """

    def __init__(self, netw_data: dict):
        """
        Initializes network class from network data

        :param dict netw_data: network data
        """
        super().__init__(netw_data)
        # Options
        self.capex_defined_per_arc = netw_data["capex_defined_per_arc"]
        self.size_max_defined_per_arc = netw_data["size_max_defined_per_arc"]

        # General information
        self.connection = []
        self.distance = []
        self.size_max_arcs = []
        self.energy_consumption = {}
        self.gamma_per_arc = {}

        self.transported_carrier = netw_data["Performance"]["carrier"]

        self.set_nodes = []
        self.set_t = []

        self.scaling_factors = []
        if "ScalingFactors" in netw_data:
            self.scaling_factors = netw_data["ScalingFactors"]

        self.bidirectional_network = None
        self.bidirectional_network_precise = None

    def fit_network_performance(self):
        """
        Fits network performance (bounds and coefficients).
        """
        time_independent = {}

        # Size
        time_independent["size_min"] = self.size_min
        if not self.existing:
            time_independent["size_max"] = self.size_max
        else:
            time_independent["size_max"] = self.size_initial
            time_independent["size_initial"] = self.size_initial

        if self.existing:
            # Use initial size
            time_independent["size_max_arcs"] = time_independent["size_initial"]
        else:
            if self.size_max_defined_per_arc:
                time_independent["size_max_arcs"] = self.size_max_arcs
            else:
                # Use max size
                time_independent["size_max_arcs"] = pd.DataFrame(
                    time_independent["size_max"],
                    index=self.distance.index,
                    columns=self.distance.columns,
                )

        time_independent["rated_capacity"] = get_attribute_from_dict(
            self.performance_data, "rated_capacity", 1
        )

        # Cost parameters
        time_independent["cost_per_arc"] = {}
        if self.capex_defined_per_arc:
            # Use gammas from file (defined per arc)
            time_independent["cost_per_arc"] = self.gamma_per_arc
        else:
            # Use global cost parameters, as defined in the json file
            gammas = ["gamma1", "gamma2", "gamma3", "gamma4"]
            for gamma in gammas:
                time_independent["cost_per_arc"][gamma] = pd.DataFrame(
                    self.economics[gamma],
                    index=self.distance.index,
                    columns=self.distance.columns,
                )

        # Other
        time_independent["min_transport"] = self.performance_data["min_transport"]
        time_independent["loss"] = self.performance_data["loss"]

        # Write to self
        self.processed_coeff.time_independent = time_independent

    def construct_netw_model(
        self, b_netw, data: dict, set_nodes, set_t_full, set_t_clustered
    ):
        """
        Constructs a network as model block.

        :param b_netw: pyomo network block
        :param dict data: dict containing model information
        :param set_nodes: pyomo set containing all nodes
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with network model
        """
        # LOG
        log_msg = f"\t - Constructing Network {self.name}"
        print(log_msg)
        log.info(log_msg)

        # NETWORK DATA
        config = data["config"]

        # MODELING TYPICAL DAYS
        self.modelled_with_full_res = True
        self.lower_res_than_full = False
        if config["optimization"]["typicaldays"]["method"]["value"] == 1:
            self.set_t = set_t_clustered
        elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
            self.set_t = set_t_full
        else:
            self.set_t = set_t_full

        self.set_nodes = set_nodes

        b_netw = self._define_possible_arcs(b_netw)

        if self.bidirectional_network:
            b_netw = self._define_unique_arcs(b_netw)

        b_netw = self._define_size(b_netw)
        b_netw = self._define_capex_variables_netw(b_netw)
        b_netw = self._define_opex_parameters(b_netw)
        b_netw = self._define_emission_vars(b_netw)
        b_netw = self._define_network_carrier(b_netw)
        b_netw = self._define_inflow_vars(b_netw)
        b_netw = self._define_outflow_vars(b_netw)
        b_netw = self._define_energyconsumption_parameters(b_netw)

        def arc_block_init(b_arc, node_from, node_to):
            """
            Constructs each arc as a block
            """

            b_arc.big_m_transformation_required = 0
            b_arc = self._define_size_arc(b_arc, b_netw, node_from, node_to)
            b_arc = self._define_capex_parameters_arc(
                b_arc, b_netw, node_from, node_to, data
            )
            b_arc = self._define_capex_variables_arc(b_arc)
            b_arc = self._define_capex_constraints_arc(
                b_arc, b_netw, node_from, node_to
            )
            b_arc = self._define_flow(b_arc, b_netw)
            b_arc = self._define_opex_arc(b_arc, b_netw, data)
            b_arc = self._define_emissions_arc(b_arc, b_netw)

            b_arc = self._define_energyconsumption_arc(b_arc, b_netw)

            # Decommissioning only complete
            if self.existing and self.decommission == "only_complete":
                b_arc = self._define_decommissioning_at_once_constraints(
                    b_arc, b_netw, node_from, node_to
                )

            if b_arc.big_m_transformation_required:
                b_arc = perform_disjunct_relaxation(b_arc)

            # LOG
            log_msg = f"\t\t - Constructing Arc {node_from} - {node_to} " f"completed"
            print(log_msg)
            log.info(log_msg)

        b_netw.arc_block = pyo.Block(b_netw.set_arcs, rule=arc_block_init)

        # CONSTRAINTS FOR BIDIRECTIONAL NETWORKS
        if self.bidirectional_network:
            b_netw = self._define_bidirectional_constraints(b_netw)

        b_netw = self._define_capex_total(b_netw)
        b_netw = self._define_opex_total(b_netw, data)
        b_netw = self._define_inflow_constraints(b_netw)
        b_netw = self._define_outflow_constraints(b_netw)
        b_netw = self._define_emission_constraints(b_netw)

        b_netw = self._define_energyconsumption_total(b_netw)

        # LOG
        log_msg = f"\t - Constructing Network {self.name} completed"
        print(log_msg)
        log.info(log_msg)

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

        # Check if sizes in both direction are the same for bidirectional existing networks
        if self.existing:
            if self.bidirectional_network:
                for from_node in coeff_ti["size_initial"]:
                    for to_node in coeff_ti["size_initial"][from_node].index:
                        assert (
                            coeff_ti["size_initial"].at[from_node, to_node]
                            == coeff_ti["size_initial"].at[to_node, from_node]
                        )

        return b_netw

    def _define_capex_variables_netw(self, b_netw):
        """
        Defines the capex variable of the network

        :param b_netw: pyomo network bloc
        :return: pyomo network bloc
        """

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
            domain=pyo.Reals, initialize=economics["opex_variable"], mutable=True
        )
        b_netw.para_opex_fixed = pyo.Param(
            domain=pyo.Reals, initialize=economics["opex_fixed"], mutable=True
        )

        b_netw.var_opex_variable = pyo.Var()
        b_netw.var_opex_fixed = pyo.Var()

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

    def _define_network_carrier(self, b_netw):
        """
        Defines transported carrier

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        # Define set of transported carrier
        b_netw.set_netw_carrier = pyo.Set(initialize=[self.transported_carrier])

        return b_netw

    def _define_inflow_vars(self, b_netw):
        """
        Defines network inflow variable (i.e. sum of inflow to one node)

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
        Defines network outflow variable (i.e. sum of outflow to one node)

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

        # Consumption at each node
        b_netw.var_consumption = pyo.Var(
            self.set_t,
            b_netw.set_consumed_carriers,
            self.set_nodes,
            domain=pyo.NonNegativeReals,
        )

        return b_netw

    def _define_size_arc(self, b_arc, b_netw, node_from: str, node_to: str):
        """
        Defines the size of an arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :param str node_from: node from which arc comes
        :param str node_to: node to which arc goes
        :return: pyomo arc block
        """
        coeff_ti = self.processed_coeff.time_independent

        if self.size_is_int:
            size_domain = pyo.NonNegativeIntegers
        else:
            size_domain = pyo.NonNegativeReals

        b_arc.para_size_max = pyo.Param(
            domain=size_domain,
            initialize=coeff_ti["size_max_arcs"].at[node_from, node_to],
        )

        if self.existing:
            b_arc.para_size_initial = pyo.Param(
                domain=size_domain,
                initialize=coeff_ti["size_initial"].at[node_from, node_to],
            )

        b_arc.distance = self.distance.at[node_from, node_to]

        if self.existing and self.decommission == "impossible":
            # Decommissioning is not possible, size fixed
            b_arc.var_size = pyo.Var(
                within=size_domain,
                bounds=(b_arc.para_size_initial, b_arc.para_size_max),
            )
        else:
            # Size is variable
            b_arc.var_size = pyo.Var(
                within=size_domain,
                bounds=(b_netw.para_size_min, b_arc.para_size_max),
            )

        return b_arc

    def _define_capex_parameters_arc(
        self, b_arc, b_netw, node_from: str, node_to: str, data: dict
    ):
        """
        Defines the capex parameters of each arc. If these are not specified with gamma1.csv,..., gamma4.csv
        files in the folder "network_topology/existing(or new)/name_of_network", then the gammas are taken
        from the json file of the network and are assumed to be the same for every arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :param str node_from: node from which arc comes
        :param str node_to: node to which arc goes
        :param dict data: dict containing model information
        :return: pyomo arc block
        """

        config = data["config"]
        coeff_ti = self.processed_coeff.time_independent
        economics = self.economics

        gamma1 = coeff_ti["cost_per_arc"]["gamma1"].at[node_from, node_to]
        gamma2 = coeff_ti["cost_per_arc"]["gamma2"].at[node_from, node_to]
        gamma3 = coeff_ti["cost_per_arc"]["gamma3"].at[node_from, node_to]
        gamma4 = coeff_ti["cost_per_arc"]["gamma4"].at[node_from, node_to]

        # CHECK FOR GLOBAL ECONOMIC OPTIONS
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]

        # CAPEX
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        b_arc.para_capex_gamma1 = pyo.Param(
            domain=pyo.Reals,
            mutable=True,
            initialize=gamma1 * annualization_factor,
        )
        b_arc.para_capex_gamma2 = pyo.Param(
            domain=pyo.Reals,
            mutable=True,
            initialize=gamma2 * annualization_factor,
        )
        b_arc.para_capex_gamma3 = pyo.Param(
            domain=pyo.Reals,
            mutable=True,
            initialize=gamma3 * annualization_factor,
        )
        b_arc.para_capex_gamma4 = pyo.Param(
            domain=pyo.Reals,
            mutable=True,
            initialize=gamma4 * annualization_factor,
        )

        if self.existing:
            b_arc.para_decommissioning_cost_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=economics["decommission_cost"] * annualization_factor,
                mutable=True,
            )

        return b_arc

    def _define_capex_variables_arc(self, b_arc):
        """
        Defines the capex variables of an arc

        :param b_arc: pyomo arc block
        :return: pyomo arc block
        """
        coeff_ti = self.processed_coeff.time_independent

        def calculate_max_capex():
            max_capex = (
                b_arc.para_capex_gamma1
                + b_arc.para_capex_gamma2 * b_arc.para_size_max
                + b_arc.para_capex_gamma3 * b_arc.distance
                + b_arc.para_capex_gamma4 * b_arc.para_size_max * b_arc.distance
            )
            return (0, max_capex)

        # CAPEX auxilliary (used to calculate theoretical CAPEX)
        # For new technologies, this is equal to actual CAPEX
        # For existing technologies it is used to calculate fixed OPEX
        b_arc.var_capex_aux = pyo.Var(bounds=calculate_max_capex())

        b_arc.var_capex = pyo.Var()

        return b_arc

    def _define_capex_constraints_arc(self, b_arc, b_netw, node_from, node_to):
        """
        Defines the capex of an arc and corresponding constraints

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :param str node_from: node from which arc comes
        :param str node_to: node to which arc goes
        :return: pyomo arc block
        """
        coeff_ti = self.processed_coeff.time_independent

        def init_capex(const):
            return (
                b_arc.var_capex_aux
                == b_arc.para_capex_gamma1
                + b_arc.para_capex_gamma2 * b_arc.var_size
                + b_arc.para_capex_gamma3 * b_arc.distance
                + b_arc.para_capex_gamma4 * b_arc.var_size * b_arc.distance
            )

        # CAPEX aux:
        if self.existing and self.decommission == "impossible":
            b_arc.const_capex_aux = pyo.Constraint(rule=init_capex)
        elif (b_arc.para_capex_gamma1.value == 0) and (
            b_arc.para_capex_gamma3.value == 0
        ):
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
        if self.existing:
            if self.decommission == "impossible":
                b_arc.const_capex = pyo.Constraint(expr=b_arc.var_capex == 0)
            else:
                b_arc.const_capex = pyo.Constraint(
                    expr=b_arc.var_capex
                    == (b_arc.para_size_initial - b_arc.var_size)
                    * b_arc.para_decommissioning_cost_annual
                )
        else:
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
        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]

        b_arc.var_flow = pyo.Var(
            self.set_t,
            domain=pyo.NonNegativeReals,
            bounds=(
                0,
                b_arc.para_size_max * rated_capacity,
            ),
        )
        b_arc.var_losses = pyo.Var(
            self.set_t,
            domain=pyo.NonNegativeReals,
            bounds=(
                0,
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
        Defines the energy consumption for an arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """

        return b_arc

    def _define_opex_arc(self, b_arc, b_netw, data):
        """
        Defines OPEX per Arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        b_arc.var_opex_variable = pyo.Var()

        hour_factors = data["hour_factors"]
        nr_timesteps_averaged = data["nr_timesteps_averaged"]

        def init_opex_variable(const):
            return b_arc.var_opex_variable == sum(
                (b_arc.var_flow[t] * nr_timesteps_averaged * hour_factors[t - 1])
                * b_netw.para_opex_variable
                for t in self.set_t
            )

        b_arc.const_opex_variable = pyo.Constraint(rule=init_opex_variable)
        return b_arc

    def _define_emissions_arc(self, b_arc, b_netw):
        """
        defines emission per arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        return b_arc

    def _define_bidirectional_constraints(self, b_netw):
        """
        Defines constraints necessary, in case one arc can transport in two directions.

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]

        # Size in both direction is the same
        if not self.existing or not self.decommission == "impossible":

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
                <= b_netw.arc_block[node_from, node_to].var_size * rated_capacity
            )

        b_netw.const_cut_bidirectional = pyo.Constraint(
            self.set_t, b_netw.set_arcs_unique, rule=init_cut_bidirectional
        )

        # Disjunction
        if self.bidirectional_network_precise:
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
        if self.bidirectional_network:
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

    def _define_opex_total(self, b_netw, data):
        """
        Defines total OPEX of network

        :param b_netw: pyomo network block
        :param dict data: dict containing model information
        :return: pyomo network block
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        if self.bidirectional_network:
            arc_set = b_netw.set_arcs_unique
        else:
            arc_set = b_netw.set_arcs

        def init_opex_fixed(const):
            return (
                b_netw.para_opex_fixed
                * fraction_of_year_modelled
                * (
                    sum(b_netw.arc_block[arc].var_capex_aux for arc in arc_set)
                    / annualization_factor
                )
                == b_netw.var_opex_fixed
            )

        b_netw.const_opex_fixed = pyo.Constraint(rule=init_opex_fixed)

        def init_opex_variable(const):
            return (
                sum(b_netw.arc_block[arc].var_opex_variable for arc in b_netw.set_arcs)
                == b_netw.var_opex_variable
            )

        b_netw.const_opex_var = pyo.Constraint(rule=init_opex_variable)
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
        Defines emissions from network

        :param b_netw: pyomo network block
        :return: pyomo network block
        """

        return b_netw

    def _define_energyconsumption_total(self, b_netw):
        """
        Defines network consumption at each node

        :param b_netw: pyomo network block
        :return: pyomo network block
        """

        return b_netw

    def _define_decommissioning_at_once_constraints(
        self, b_arc, b_netw, node_from: str, node_to: str
    ):
        """
        Defines constraints to ensure that a network connection can only be decommissioned as a whole.

        This function creates a disjunction formulation that enforces
        full decommissioning decisions for a network arc, meaning that either the connection is fully
        installed or fully decommissioned, with no partial decommissioning allowed.

        :param b_arc: The block representing the network arc.
        :param b_netw: The block representing the network.
        :param node_from: The originating node of the arc.
        :param node_to: The destination node of the arc.

        :return: The modified network arc block with added decommissioning constraints.
        :rtype: Pyomo Block
        """

        # Full plant decommissioned only
        self.big_m_transformation_required = 1
        s_indicators = range(0, 2)

        def init_decommission_full(dis, ind):
            if ind == 0:  # tech not installed
                dis.const_decommissioned = pyo.Constraint(expr=b_arc.var_size == 0)
            else:  # tech installed
                dis.const_installed = pyo.Constraint(
                    expr=b_arc.var_size == b_arc.para_size_initial
                )

        b_arc.dis_decommission_full = gdp.Disjunct(
            s_indicators, rule=init_decommission_full
        )

        def bind_disjunctions(dis):
            return [b_arc.dis_decommission_full[i] for i in s_indicators]

        b_arc.disjunction_decommission_full = gdp.Disjunction(rule=bind_disjunctions)

        return b_arc

    def write_results_netw_design(self, h5_group, model_block, config, data):
        """
        Function to report network design

        :param model_block: pyomo network block
        :param dict config: dict containing model configuration
        :param dict data: dict containing model information
        :param h5_group: h5 group to write to
        """
        coeff_ti = self.processed_coeff.time_independent
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data.topology["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        for arc_name in model_block.set_arcs:
            arc = model_block.arc_block[arc_name]
            str = "".join(arc_name)
            arc_group = h5_group.create_group(str)

            arc_group.create_dataset(
                "para_capex_gamma1", data=arc.para_capex_gamma1.value
            )
            arc_group.create_dataset(
                "para_capex_gamma2", data=arc.para_capex_gamma2.value
            )
            arc_group.create_dataset(
                "para_capex_gamma3", data=arc.para_capex_gamma3.value
            )
            arc_group.create_dataset(
                "para_capex_gamma4", data=arc.para_capex_gamma4.value
            )
            arc_group.create_dataset("network", data=self.name)
            arc_group.create_dataset("fromNode", data=arc_name[0])
            arc_group.create_dataset("toNode", data=arc_name[1])
            arc_group.create_dataset("size", data=arc.var_size.value)
            arc_group.create_dataset("capex", data=arc.var_capex.value)
            arc_group.create_dataset(
                "opex_fixed",
                data=[
                    model_block.para_opex_fixed.value
                    * (arc.var_capex_aux.value / annualization_factor)
                ],
            )
            arc_group.create_dataset(
                "opex_variable",
                data=arc.var_opex_variable.value,
            )
            arc_group.create_dataset(
                "total_flow", data=sum(arc.var_flow[t].value for t in self.set_t)
            )

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

    def scale_model(self, b_netw, model, config: dict):
        """
        Scales network model

        :param b_netw: pyomo network block
        :param model: pyomo model
        :param dict config: config dict containing scaling factors
        :return: pyomo model
        """

        f = self.scaling_factors
        f_global = config["scaling"]["scaling_factors"]

        model = determine_variable_scaling(model, b_netw, f, f_global)
        model = determine_constraint_scaling(model, b_netw, f, f_global)

        for arc in b_netw.arc_block:
            b_arc = b_netw.arc_block[arc]

            model = determine_variable_scaling(model, b_arc, f, f_global)
            model = determine_constraint_scaling(model, b_arc, f, f_global)

        return model
