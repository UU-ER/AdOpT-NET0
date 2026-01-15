"""
Plugin enabling operational constraints for technologies.

This plugin adds detailed operational constraints to technologies, allowing for more realistic modeling of their behavior.
The formulation follows the model presented in Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden power system
inflexibilities imposed by traditional unit commitment formulations. Applied Energy, 191, 223–238.
https://doi.org/10.1016/J.APENERGY.2017.01.089.

Supported constraints are:

- Minimum uptime and downtime
- Maximum number of startups
- Ramping rates (up and down)

The plugin needs to be enabled in Plugins.json in the input data folder like this. The technologies listed will
have the operational constraints applied.

.. code-block:: json

    "modeling_plugins.operational_constraints_technologies": {
        "config": {
            "technologies": ["Technology_A", "Technology_B"]
        }
    }

For all technologies listed in the plugin config, the following settings need to be defined in the respective technology data file
under the key ``settings_operational_constraints``:

.. code-block:: json

    "settings_operational_constraints": {
        "enable_operational_constraints": True,
        "max_startups": {
            "enabled": True,
            "value": 10
         },
        "min_uptime": {
            "enabled": True,
             "value": 2
         },
        "min_downtime": {
            "enabled": True,
            "value": 5
        },
        "relative_ramping_rate_up": {
            "enabled": True,
            "on": "output",
            "carriers": {
                "electricity": 0.1,
            },
            "operator": "sum"
        },
        "relative_ramping_rate_down": {
            "enabled": True,
            "on": "output",
            "carriers": {
                "electricity": 0.1,
            },
            "operator": "sum"
        },
    }

Where:

- ``enable_operational_constraints``: Boolean to enable/disable operational constraints for the technology
- ``max_startups``: Maximum number of startups allowed during the modeling horizon
- ``min_uptime``: Minimum number of consecutive timesteps the technology must remain on after startup
- ``min_downtime``: Minimum number of consecutive timesteps the technology must remain off after shutdown
- ``relative_ramping_rate_up``: Ramping up rate constraints defined as a fraction of the technology size. Specify whether the constraint applies to input or output,
  the carriers to which it applies (with their respective rates), and whether the constraint should be applied to the sum of all specified carriers (``"operator" = "sum"``)
  or individually (``"operator" = "individual"``).
- ``relative_ramping_rate_down``: Ramping down rate constraints defined as a fraction of the technology size. Specify whether the constraint applies to input or output,
  the carriers to which it applies (with their respective rates), and whether the constraint should be applied to the sum of all specified carriers (``"operator" = "sum"``)
  or individually (``"operator" = "individual"``).

# TODO: Check equations!

**Variable declarations:**

(These variables are not required if only ramping rates are used)

- ``var_on_off``: Binary on/off indicator :math:`t`: :math:`var\_on\_off_{t}`
- ``var_in_startup``: Binary startup indicator :math:`t`: :math:`var\_in\_startup_{t}`
- ``var_in_shutdown``: Binary shutdown indicator :math:`t`: :math:`var\_in\_shutdown_{t}`

**Constraint declarations:**

- **Startup/Shutdown Logic**: The relationship between the on/off state and startup/shutdown variables is defined as:

  .. math::
     var\_on\_off_{t} - var\_on\_off_{t-1} = var\_in\_startup_{t} - var\_in\_shutdown_{t}

- **Minimum Uptime/Downtime**: Constraints to ensure the technology remains on for a minimum duration after starting up, and off for a minimum duration after shutting down.

  .. math::
     var\_in\_startup_{t - min\_uptime + 1} \\leq var\_on\_off_{t} \\\\
     var\_in\_shutdown_{t - min\_downtime + 1} \\leq 1 - var\_on\_off_{t}

- **Maximum Startups**: Limits the total number of startups over the modeling horizon.

  .. math::
     \\sum_{t} var\_in\_startup_{t} \\leq max\_startups

- **Ramping Rates**: Constrains the rate at which the technology's input or output can change between timesteps.

  - **Ramping Up**:

    .. math::
       \\sum(var_{t, car}) - \\sum(var_{t-1, car}) \\leq ramping\_rate_{up} * S

  - **Ramping Down**:

    .. math::
       \\sum(var_{t, car}) - \\sum(var_{t-1, car}) \\geq -ramping\_rate_{down} * S

Where :math:`S` is the size of the technology.

"""

import pyomo.core as pyo
import pyomo.gdp as gdp

from adopt_net0.plugins.base import Plugin as PluginBase
from adopt_net0.plugins.hooks import Hook
from adopt_net0.core.components.utilities import link_full_resolution_to_clustered


class Plugin(PluginBase):
    name = "Operational constraints for technologies"
    hooks = {
        Hook.TECHNOLOGY_CONSTRUCTION_END,
    }
    config_template = {
        "technologies": []
    }

    def on_technology_construction_end(self, technology_constructor, modelhub, b_tec: pyo.Block) -> None:
        if not technology_constructor.name in self.config.get("technologies", []):
            return

        try:
            settings = technology_constructor.data["settings_operational_constraints"]
        except KeyError:
            raise KeyError(f"'settings_operational_constraints' are not defined for technology {technology_constructor.name} but required for operational constraints plugin.")

        # Check if operational constraints are enabled
        if not settings["enable_operational_constraints"]:
            return

        # Check if operational constraints require binary variables
        binary_operational_variables_required = 0
        settings_requiring_binary_vars = [
            "max_startups",
            "min_uptime",
            "min_downtime",
        ]
        for setting in settings_requiring_binary_vars:
            if settings[setting] != -1:
                binary_operational_variables_required = 1

        if binary_operational_variables_required:
            self._define_binary_operational_variables(technology_constructor, b_tec)
            self._define_binary_operational_constraints(technology_constructor, b_tec, settings)
            technology_constructor.big_m_transformation_required = 1

        if settings["max_startups"]["enabled"]:
            self._define_max_startup_constraints(technology_constructor, b_tec, settings)

        if settings["relative_ramping_rate_up"]["enabled"]:
            self._define_relative_ramping_up_constraints(technology_constructor, b_tec, settings)

        if settings["relative_ramping_rate_down"]["enabled"]:
            self._define_relative_ramping_down_constraints(technology_constructor, b_tec, settings)


    def _define_binary_operational_variables(self, technology_constructor, b_tec):

        data_clustered = False
        #Todo: change this later

        # New variables
        b_tec.var_on_off = pyo.Var(
            technology_constructor.set_t_performance, domain=pyo.Binary
        )
        b_tec.var_in_startup = pyo.Var(
            technology_constructor.set_t_performance, domain=pyo.Binary
        )
        b_tec.var_in_shutdown = pyo.Var(
            technology_constructor.set_t_performance, domain=pyo.Binary
        )
        if data_clustered:
            sequence = []
            b_tec.var_on_off_full = pyo.Var(
                technology_constructor.set_t_full, domain=pyo.Binary
            )
            b_tec.var_in_startup_full = pyo.Var(
                technology_constructor.set_t_full, domain=pyo.Binary
            )
            b_tec.var_in_shutdown_full = pyo.Var(
                technology_constructor.set_t_full, domain=pyo.Binary
            )
            b_tec.const_link_full_resolution_on_off = link_full_resolution_to_clustered(
                b_tec.var_on_off,
                b_tec.var_on_off_full,
                technology_constructor.set_t_full,
                sequence,
                b_tec.set_input_carriers,
            )
            b_tec.const_link_full_resolution_in_startup = link_full_resolution_to_clustered(
                b_tec.var_in_startup,
                b_tec.var_in_startup_full,
                technology_constructor.set_t_full,
                sequence,
                b_tec.set_input_carriers,
            )
            b_tec.const_link_full_resolution_in_shutdown = link_full_resolution_to_clustered(
                b_tec.var_in_shutdown,
                b_tec.var_in_shutdown_full,
                technology_constructor.set_t_full,
                sequence,
                b_tec.set_input_carriers,
            )

    def _define_binary_operational_constraints(self, technology_constructor, b_tec, settings):
        data_clustered = False
        #Todo: change this later

        min_uptime = settings["min_uptime"]["value"] if settings["min_uptime"]["enabled"] else 1
        min_downtime = settings["min_downtime"]["value"] if settings["min_downtime"]["enabled"] else 1


        var_input = technology_constructor.input
        var_output = technology_constructor.output
        if not data_clustered:
            var_on_off = b_tec.var_on_off
            var_in_startup = b_tec.var_in_startup
            var_in_shutdown = b_tec.var_in_shutdown
            set_t_used = technology_constructor.set_t_performance
        else:
            var_on_off = b_tec.var_on_off_full
            var_in_startup = b_tec.var_in_startup_full
            var_in_shutdown = b_tec.var_in_shutdown_full
            set_t_used = technology_constructor.set_t_full


        s_indicators = range(0, 2)

        # Input/output on reduced resolution
        def init_on_off(dis, t, ind):
            if ind == 0:  # technology off

                dis.const_x_off = pyo.Constraint(expr=b_tec.var_on_off[t] == 0)

                def init_input_off(const, car_input):
                    return var_input[t, car_input] == 0

                dis.const_input = pyo.Constraint(
                    b_tec.set_input_carriers, rule=init_input_off
                )

                def init_output_off(const, car_output):
                    return var_output[t, car_output] == 0

                dis.const_output_off = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_off
                )

            else:  # technology on
                dis.const_x_on = pyo.Constraint(expr=b_tec.var_on_off[t] == 1)

        b_tec.dis_on_off = gdp.Disjunct(
            technology_constructor.set_t_performance, s_indicators, rule=init_on_off
        )
        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_on_off[t, i] for i in s_indicators]

        b_tec.disjunction_on_off = gdp.Disjunction(
            technology_constructor.set_t_performance, rule=bind_disjunctions
        )


        # Enforce startup/shutdown logic on full resolution
        def init_SUSD_logic1(const, t):
            if t == 1:
                return (
                        var_on_off[t] - 0
                        == var_in_startup[t] - var_in_shutdown[t]
                )
            else:
                return (
                        var_on_off[t] - var_on_off[t - 1]
                        == var_in_startup[t] - var_in_shutdown[t]
                )

        b_tec.const_SUSD_logic1 = pyo.Constraint(
            set_t_used, rule=init_SUSD_logic1
        )

        def init_SUSD_logic2(const, t):
            if t >= min_uptime:
                return var_in_startup[t - min_uptime + 1] <= var_on_off[t]
            else:
                return (
                    var_in_startup[max(set_t_used) + (t - min_uptime + 1)]
                    <= var_on_off[t]
                )

        b_tec.const_SUSD_logic2 = pyo.Constraint(
            set_t_used, rule=init_SUSD_logic2
        )

        def init_SUSD_logic3(const, t):
            if t >= min_downtime:
                return var_in_shutdown[t - min_downtime + 1] <= 1 - var_on_off[t]
            else:
                return (
                    var_in_shutdown[max(set_t_used) + (t - min_downtime + 1)]
                    <= 1 - var_on_off[t]
                )

        b_tec.const_SUSD_logic3 = pyo.Constraint(
            set_t_used, rule=init_SUSD_logic3
        )

    def _define_max_startup_constraints(self, technology_constructor, b_tec, settings):

        max_startups = settings["max_startups"]["value"]

        data_clustered = False

        if not data_clustered:
            var_in_startup = b_tec.var_in_startup
            set_t_used = technology_constructor.set_t_performance
        else:
            var_in_startup = b_tec.var_in_startup_full
            set_t_used = technology_constructor.set_t_full

        def init_max_startups(const):
            return (
                    sum(var_in_startup[t] for t in set_t_used) <= max_startups
            )

        b_tec.const_max_startups = pyo.Constraint(rule=init_max_startups)

    def _define_relative_ramping_up_constraints(self, technology_constructor, b_tec, settings):
        relative_ramping_rate_up = settings["relative_ramping_rate_up"]["carriers"]

        if settings["relative_ramping_rate_up"]["on"] == "input":
            var_rr = b_tec.var_input
        else:
            var_rr = b_tec.var_output

        def init_ramping_up_rate_sum(const, t):
            if t > 1:
                return sum(
                    var_rr[t, carrier] - var_rr[t - 1, carrier]
                    for carrier in relative_ramping_rate_up.keys()
                ) <= sum(relative_ramping_rate_up[key] for key in relative_ramping_rate_up.keys()) * b_tec.var_size
            else:
                return pyo.Constraint.Skip

        def init_ramping_up_rate_per_carrier(const, t, carrier):
            if t > 1:
                return var_rr[t, carrier] - var_rr[t - 1, carrier] <= relative_ramping_rate_up[carrier]* b_tec.var_size
            else:
                return pyo.Constraint.Skip


        if settings["relative_ramping_rate_up"]["operator"] == "sum":
            b_tec.const_ramping_up_rate = pyo.Constraint(
                technology_constructor.set_t_full, rule=init_ramping_up_rate_sum
            )
        else:
            b_tec.const_ramping_up_rate = pyo.Constraint(
                technology_constructor.set_t_full, relative_ramping_rate_up.keys(), rule=init_ramping_up_rate_per_carrier
            )

    def _define_relative_ramping_down_constraints(self, technology_constructor, b_tec, settings):
        relative_ramping_rate_down = settings["relative_ramping_rate_down"]["carriers"]

        if settings["relative_ramping_rate_down"]["on"] == "input":
            var_rr = b_tec.var_input
        else:
            var_rr = b_tec.var_output

        def init_ramping_down_rate_sum(const, t):
            if t > 1:
                return sum(
                    var_rr[t, carrier] - var_rr[t - 1, carrier]
                    for carrier in relative_ramping_rate_down.keys()
                ) >= - sum(relative_ramping_rate_down[key] for key in relative_ramping_rate_down.keys()) * b_tec.var_size
            else:
                return pyo.Constraint.Skip

        def init_ramping_down_rate_per_carrier(const, t, carrier):
            if t > 1:
                return var_rr[t, carrier] - var_rr[t - 1, carrier] >= -relative_ramping_rate_down[carrier]* b_tec.var_size
            else:
                return pyo.Constraint.Skip

        if settings["relative_ramping_rate_down"]["operator"] == "sum":
            b_tec.const_ramping_up_rate = pyo.Constraint(
                technology_constructor.set_t_full, rule=init_ramping_down_rate_sum
            )
        else:
            b_tec.const_ramping_up_rate = pyo.Constraint(
                technology_constructor.set_t_full, relative_ramping_rate_down.keys(),
                rule=init_ramping_down_rate_per_carrier
            )
