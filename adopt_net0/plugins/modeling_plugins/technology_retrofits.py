"""
Plugin enabling retrofits to technologies.

Each retrofit is implemented as a separate class in ``technology_retrofits_helpers``. To use a retrofit,
add its name to the plugin configuration under the "retrofits" key:

.. code-block:: json

   "modeling_plugins.technology_retrofits": {
        "config": {
            "retrofits": ["Retrofit_A", "Retrofit_B"]
        }
    }


Retrofits implement the following variables of a technology that change the base variable respectively:

"""
import pyomo.environ as pyo

from adopt_net0.core.utilities import get_set_t
from adopt_net0.plugins.base import Plugin as PluginBase
from adopt_net0.plugins.hooks import Hook
from adopt_net0.plugins.modeling_plugins.technology_retrofits_helpers import *

FACTORY = {
    "CCS_Retrofit": CcsRetrofit,
    "Efficiency_Retrofit": EfficiencyRetrofit,
}


class Plugin(PluginBase):
    """
    Plugin enabling custom technologies.
    """
    name = "Technology retrofits"
    hooks = {
        Hook.TECHNOLOGY_CONSTRUCTION_END,
        Hook.ADD_TO_ENERGYBALANCE,
        Hook.ADD_TO_COSTBALANCE_CAPEX,
        Hook.ADD_TO_COSTBALANCE_OPEX,
        Hook.ADD_TO_EMISSIONBALANCE,
    }
    config_template = {
      "retrofits": []
    }

    def on_technology_construction_end(self, technology_constructor, modelhub, b_tec: pyo.Block) -> None:
        if "Efficiency_Retrofit" in self.config["retrofits"]:
            self._define_efficiency_retrofit(technology_constructor, modelhub, b_tec)

        if "CCS_Retrofit" in self.config["retrofits"]:
            self._define_ccs_retrofit(technology_constructor, modelhub, b_tec)



    def on_energybalance_construction(self, b_period: pyo.Block, node:str, t: int, carrier: str) -> pyo.Expression | int:
        """
        Sums over all inputs and outputs of all retrofits at current period, node, and timestep

        :return: Pyomo expression to be added to energy balance
        """
        delta_output = 0
        for tec in b_period.node_blocks[node].set_technologies:
            b_tec = b_period.node_blocks[node].tech_blocks_active[tec]
            
            # Efficiency Improvements
            if hasattr(b_tec, "efficiency_retrofit"):
                if carrier in b_tec.efficiency_retrofit.set_output_carriers:
                    delta_output += b_tec.efficiency_retrofit.var_delta_output[t, carrier]

            # CCS Retrofit
            if hasattr(b_tec, "ccs_retrofit"):
                if carrier in b_tec.ccs_retrofit.set_output_carriers:
                    delta_output += b_tec.ccs_retrofit.var_delta_output[t, carrier]
                if carrier in b_tec.ccs_retrofit.set_input_carriers:
                    delta_output -= b_tec.ccs_retrofit.var_delta_input[t, carrier]


        return delta_output


    def on_costbalance_capex_construction(self, model: pyo.Block, period: str) -> pyo.Expression | int:
        """
        Sums over all var_delta_capex in all retrofits at current period for all technologies

        :return: Pyomo expression to be added to cost balance
        """
        capex_retrofits = 0
        b_period = model.periods[period]
        for node in model.set_nodes:
            for tec in b_period.node_blocks[node].set_technologies:
                b_tec = b_period.node_blocks[node].tech_blocks_active[tec]

                # Efficiency Improvements
                if hasattr(b_tec, "efficiency_retrofit"):
                    capex_retrofits += b_tec.efficiency_retrofit.var_delta_capex

                # CCS Retrofit
                if hasattr(b_tec, "ccs_retrofit"):
                    capex_retrofits += b_tec.ccs_retrofit.var_delta_capex

        return capex_retrofits

    def on_costbalance_opex_construction(self, model: pyo.Block, period: str) -> pyo.Expression | int:
        """
        Sums over all var_delta_opex_var and var_delta_opex_fix in all retrofits at current period for all technologies

        :return: Pyomo expression to be added to cost balance
        """
        opex_var_retrofits = 0
        b_period = model.periods[period]

        for node in model.set_nodes:
            for tec in b_period.node_blocks[node].set_technologies:
                b_tec = b_period.node_blocks[node].tech_blocks_active[tec]

                # Efficiency Improvements
                if hasattr(b_tec, "efficiency_retrofit"):
                    opex_var_retrofits += b_tec.efficiency_retrofit.var_delta_opex_var + b_tec.efficiency_retrofit.var_delta_opex_fix

                # CCS Retrofit
                if hasattr(b_tec, "ccs_retrofit"):
                    opex_var_retrofits += b_tec.ccs_retrofit.var_delta_opex_var + b_tec.ccs_retrofit.var_delta_opex_fix

        return opex_var_retrofits

    def on_emissionbalance_construction(self, modelhub, model, period: pyo.Block) -> pyo.Expression | int:
        """
        Sums over all var_delta_emissions in all retrofits at current period for all technologies

        :return: Pyomo expression to be added to emission balance
        """
        config = modelhub.data["config"]
        hour_factors = modelhub.data["aggregation_info"]["hour_factors"]
        nr_timesteps_averaged = modelhub.data["aggregation_info"]["nr_timesteps_averaged"]

        delta_emissions = 0
        for node in model.set_nodes:
            b_period = model.periods[period]
            set_t = get_set_t(config, b_period)
            for tec in b_period.node_blocks[node].set_technologies:
                b_tec = b_period.node_blocks[node].tech_blocks_active[tec]
                for t in set_t:

                    # Efficiency Improvements
                    if hasattr(b_tec, "efficiency_retrofit"):
                        delta_emissions += b_tec.efficiency_retrofit.var_delta_emissions[t] * nr_timesteps_averaged * hour_factors[t - 1]

                    # CCS Retrofit
                    if hasattr(b_tec, "ccs_retrofit"):
                        delta_emissions += b_tec.ccs_retrofit.var_delta_emissions[t] * nr_timesteps_averaged * hour_factors[t - 1]

        return delta_emissions

    def _define_efficiency_retrofit(self, technology_constructor, modelhub, b_tec: pyo.Block) -> None:
        """
        Defines all variables and constraints for Efficiency_Retrofit

        It creates a block ``b_tec.efficiency_retrofit`` on the technology block.

        :param technology_constructor: Technology constructor
        :param modelhub: Model hub
        :param b_tec: Pyomo technology block
        :return:
        """
        if ("settings_technology_retrofits" in technology_constructor.data.keys() and
                technology_constructor.data["settings_technology_retrofits"].get("enable_efficiency_retrofit", False)):
            constructor = EfficiencyRetrofit()

            def efficiency_block_init(b_retrofit):
                """
                Constructs each arc as a block
                """
                constructor.define_retrofit_variables(technology_constructor, modelhub, b_retrofit)
                constructor.define_retrofit_constraints(technology_constructor, modelhub, b_retrofit, b_tec)

            b_tec.efficiency_retrofit = pyo.Block(rule=efficiency_block_init)

    def _define_ccs_retrofit(self, technology_constructor, modelhub, b_tec: pyo.Block) -> None:
        """
        Defines all variables and constraints for CCS_Retrofit

        It creates a block ``b_tec.ccs_retrofit`` on the technology block.

        :param technology_constructor: Technology constructor
        :param modelhub: Model hub
        :param b_tec: Pyomo technology block
        :return:
        """
        if ("settings_technology_retrofits" in technology_constructor.data.keys() and
                technology_constructor.data["settings_technology_retrofits"].get("enable_ccs_retrofit", False)):
            constructor = CcsRetrofit()

            def efficiency_block_init(b_retrofit):
                """
                Constructs each arc as a block
                """

                constructor.preprocess_parameters(technology_constructor)
                constructor.define_retrofit_variables(technology_constructor, modelhub, b_retrofit)
                constructor.define_retrofit_constraints(technology_constructor, modelhub, b_retrofit, b_tec)

            b_tec.ccs_retrofit = pyo.Block(rule=efficiency_block_init)



    def on_technology_results_writing_design(self, technology_constructor, b_tec: pyo.Block, h5_group):
        """
        Writes design results of retrofits to h5 file

        :param technology_constructor: Technology constructor
        :param b_tec: Pyomo technology block
        :param h5_group: H5 technology design group
        :return:
        """

        if ("settings_technology_retrofits" in technology_constructor.data.keys() and
                technology_constructor.data["settings_technology_retrofits"].get("enable_efficiency_retrofit", False)):
            constructor = EfficiencyRetrofit()
            retrofit_group = h5_group.create_group("Efficiency_Retrofit")
            constructor.write_results_design(b_tec.efficiency_retrofit, retrofit_group)

        if ("settings_technology_retrofits" in technology_constructor.data.keys() and
                technology_constructor.data["settings_technology_retrofits"].get("enable_ccs_retrofit", False)):
            constructor = CcsRetrofit()
            retrofit_group = h5_group.create_group("CCS_Retrofit")
            constructor.write_results_design(b_tec.ccs_retrofit, retrofit_group)



    def on_technology_results_writing_operation(self, technology_constructor, b_tec: pyo.Block, h5_group):
        """
        Writes operational results of retrofits to h5 file

        :param technology_constructor: Technology constructor
        :param b_tec: Pyomo technology block
        :param h5_group: H5 technology operation group
        :return:
        """
        if ("settings_technology_retrofits" in technology_constructor.data.keys() and
                technology_constructor.data["settings_technology_retrofits"].get("enable_efficiency_retrofit", False)):
            constructor = EfficiencyRetrofit()
            retrofit_group = h5_group.create_group("Efficiency_Retrofit")
            constructor.write_results_operation(technology_constructor, b_tec.efficiency_retrofit, retrofit_group)

        if ("settings_technology_retrofits" in technology_constructor.data.keys() and
                technology_constructor.data["settings_technology_retrofits"].get("enable_ccs_retrofit", False)):
            constructor = CcsRetrofit()
            retrofit_group = h5_group.create_group("CCS_Retrofit")
            constructor.write_results_operation(technology_constructor, b_tec.ccs_retrofit, retrofit_group)
