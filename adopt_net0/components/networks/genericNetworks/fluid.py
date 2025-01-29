import pandas as pd
import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
from warnings import warn

# from ..genericNetworks import fitting_classes as f
from ..network import Network
from ...utilities import link_full_resolution_to_clustered


class Fluid(Network):

    def __init__(self, netw_data: dict):
        """
        Constructor

        :param dict netw_data: network data
        """
        super().__init__(netw_data)

        self._calculate_energy_consumption()

    def fit_network_performance(self):
        """
        Fits network performance for fluid network (bounds and coefficients).
        """
        super(Fluid, self).fit_network_performance()

        input_parameters = self.input_parameters
        # time_independent = {}

        # Emissions
        self.processed_coeff.time_independent["loss2emissions"] = (
            input_parameters.performance_data["loss2emissions"]
        )
        self.processed_coeff.time_independent["emissionfactor"] = (
            input_parameters.performance_data["emissionfactor"]
        )

    def _define_energyconsumption_parameters(self, b_netw):
        """
        Constructs constraints for fluid network energy consumption

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        super(Fluid, self)._define_energyconsumption_parameters(b_netw)

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

    def _define_energyconsumption_arc(self, b_arc, b_netw):
        """
        Defines the energy consumption for an arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        super(Fluid, self)._define_energyconsumption_arc(b_arc, b_netw)
        rated_capacity = self.input_parameters.rated_power

        b_arc.var_consumption_send = pyo.Var(
            self.set_t,
            b_netw.set_consumed_carriers,
            domain=pyo.NonNegativeReals,
            bounds=(
                b_netw.para_size_min * rated_capacity,
                b_arc.para_size_max * rated_capacity,
            ),
        )
        b_arc.var_consumption_receive = pyo.Var(
            self.set_t,
            b_netw.set_consumed_carriers,
            domain=pyo.NonNegativeReals,
            bounds=(
                b_netw.para_size_min * rated_capacity,
                b_arc.para_size_max * rated_capacity,
            ),
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

    def _define_emissions_arc(self, b_arc, b_netw):
        """
        Defines emission per arc for Fluid network

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        super(Fluid, self)._define_emissions_arc(b_arc, b_netw)

        coeff_ti = self.processed_coeff.time_independent

        def init_arc_emissions(const, t):
            return (
                b_arc.var_emissions[t]
                == b_arc.var_flow[t] * coeff_ti["emissionfactor"]
                + b_arc.var_losses[t] * coeff_ti["loss2emissions"]
            )

        b_arc.const_arc_emissions = pyo.Constraint(self.set_t, rule=init_arc_emissions)

        return b_arc

    def _define_emission_constraints(self, b_netw):
        """
        Defines emissions from fluid network

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        super(Fluid, self)._define_emission_constraints(b_netw)

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
        Defines network consumption at each node for fluid network

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        super(Fluid, self)._define_energyconsumption_total(b_netw)

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
        super(Fluid, self).write_results_netw_design(h5_group, model_block)

        coeff_ti = self.processed_coeff.time_independent

        for arc_name in model_block.set_arcs:
            arc = model_block.arc_block[arc_name]
            str = "".join(arc_name)
            arc_group = h5_group[str]

            total_emissions = (
                sum(arc.var_flow[t].value for t in self.set_t)
                * coeff_ti["emissionfactor"]
                + sum(arc.var_losses[t].value for t in self.set_t)
                * coeff_ti["loss2emissions"]
            )
            arc_group.create_dataset("total_emissions", data=total_emissions)
