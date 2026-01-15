"""
Resembles gas turbines of different sizes.
Hydrogen and Natural Gas Turbines are possible at four different sizes,
as indicated by the file names of the data. Performance data and the model is
taken from Weimann, L., Ellerker, M., Kramer, G. J., & Gazzani, M. (2019).
Modeling gas turbines in multi-energy systems: A linear model accounting for
part-load operation, fuel, temperature, and sizing effects. International
Conference on Applied Energy. https://doi.org/10.46855/energy-proceedings-5280

A small adaption is made: Natural gas turbines can co-fire hydrogen up to 5% of
the energy content

**Variable declarations:**

- Total fuel input in :math:`t`: :math:`Input_{tot, t}`

- Number of turbines on in :math:`t`: :math:`N_{on,t}`

**Constraint declarations:**

The following constants are used:

- :math:`Input_{min}`: Minimal input per turbine

- :math:`Input_{max}`: Maximal input per turbine

- :math:`in_{H2max}`: Maximal H2 admixture to fuel (only for natural gas turbines, default is 0.05)

- :math:`{\\alpha}`: Performance parameter for electricity output

- :math:`{\\beta}`: Performance parameter for electricity output

- :math:`{\\epsilon}`: Performance parameter for heat output

- :math:`f({\\Theta})`: Ambient temperature correction factor

- Input calculation (For hydrogen turbines, :math:`Input_{NG, t}` is zero, and the second constraint is removed):

 .. math::
   Input_{H2, t} + Input_{NG, t} = Input_{tot, t}

 .. math::
   Input_{H2, t} \\leq in_{H2max} Input_{tot, t}

- Turbines on:

 .. math::
   N_{on, t} \\leq S

- If technology is on:

 .. math::
   Output_{el,t} = ({\\alpha} Input_{tot, t} + {\\beta} * N_{on, t}) *f({\\Theta})

 .. math::
   Output_{th,t} = {\\epsilon} Input_{tot, t} - Output_{el,t}

 .. math::
   Input_{min} * N_{on, t} \\leq Input_{tot, t} \\leq Input_{max} * N_{on, t}

- If the technology is off, input and output is set to 0:

 .. math::
    \\sum(Output_{t, car}) = 0

 .. math::
    \\sum(Input_{t, car}) = 0

"""

import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
import numpy as np

from adopt_net0.core.components.technology import Technology
from adopt_net0.core.components.utilities import link_full_resolution_to_clustered


class GasTurbine(Technology):

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.emissions_based_on = "input"
        self.size_based_on = "output"
        self.main_input_carrier = tec_data["Performance"]["main_input_carrier"]

    def fit_performance(self, modelhub: object, component_id: tuple, **kwargs):
        """
        Fits technology performance and writes it to self.

        :param modelhub: model hub
        :param tuple component_id: component id containing (period, node, component name)
        """
        super(GasTurbine, self).fit_performance(modelhub, component_id)
        period = component_id[0]
        node = component_id[1]
        technology_time_series = modelhub.data["time_series_data"]["full_resolution"][(period, node, "TechnologyTimeSeries", self.name)]

        # Climate data & Number of timesteps
        time_steps = len(technology_data)

        # Ambient air temperature
        T = copy.deepcopy(technology_data["temp_air"])

        # Temperature correction factors
        f = np.empty(shape=(time_steps))
        f[T <= 6] = (
            self.performance_data["gamma"][0]
            * (T[T <= 6] / self.performance_data["T_iso"])
            + self.performance_data["delta"][0]
        )
        f[T > 6] = (
            self.performance_data["gamma"][1]
            * (T[T > 6] / self.performance_data["T_iso"])
            + self.performance_data["delta"][1]
        )

        # Derive return
        fit = {}
        fit["td"] = {}
        fit["td"]["temperature_correction"] = f.round(5)

        fit["ti"] = {}
        fit["ti"]["alpha"] = round(self.performance_data["alpha"], 5)
        fit["ti"]["beta"] = round(self.performance_data["beta"], 5)
        fit["ti"]["epsilon"] = round(self.performance_data["epsilon"], 5)
        fit["ti"]["in_min"] = round(self.performance_data["in_min"], 5)
        fit["ti"]["in_max"] = round(self.performance_data["in_max"], 5)
        if len(self.input_carrier) == 2:
            fit["ti"]["max_H2_admixture"] = self.performance_data["max_H2_admixture"]
        else:
            fit["ti"]["max_H2_admixture"] = 1

        # Coefficients
        for par in fit["td"]:
            self.processed_coeff.time_dependent_full[par] = fit["td"][par]
        for par in fit["ti"]:
            self.processed_coeff.time_independent[par] = fit["ti"][par]

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(GasTurbine, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        bounds = {}

        # Input bounds
        bounds["input_bounds"] = {}
        for c in self.input_carrier:
            if c == "hydrogen":
                bounds["input_bounds"][c] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps))
                        * self.performance_data["in_max"]
                        * self.processed_coeff.time_independent["max_H2_admixture"],
                    )
                )
            else:
                bounds["input_bounds"][c] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps)) * self.performance_data["in_max"],
                    )
                )

        # Output bounds
        bounds["output_bounds"] = {}
        bounds["output_bounds"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                self.processed_coeff.time_dependent_used["temperature_correction"]
                * (
                    self.performance_data["in_max"]
                    * self.processed_coeff.time_independent["alpha"]
                    + self.processed_coeff.time_independent["beta"]
                ),
            )
        )
        bounds["output_bounds"]["heat"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                self.processed_coeff.time_independent["epsilon"]
                * self.processed_coeff.time_independent["in_max"]
                - self.processed_coeff.time_dependent_used["temperature_correction"]
                * (
                    self.performance_data["in_max"]
                    * self.processed_coeff.time_independent["alpha"]
                    + self.processed_coeff.time_independent["beta"]
                ),
            )
        )

        # Output Bounds
        self.bounds["output"] = bounds["output_bounds"]
        # Input Bounds
        for car in self.input_carrier:
            self.bounds["input"][car] = np.column_stack(
                (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
            )

    def construct_model(self, b_tec, modelhub, set_t_full, set_t_clustered, **kwargs):
        """
        Adds constraints to technology blocks for gas turbines

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(GasTurbine, self).construct_model(
            b_tec, modelhub, set_t_full, set_t_clustered, **kwargs
        )

        # Transformation required
        self.big_m_transformation_required = 1

        # DATA OF TECHNOLOGY
        bounds = self.bounds
        coeff_td = self.processed_coeff.time_dependent_used
        coeff_ti = self.processed_coeff.time_independent
        dynamics = self.processed_coeff.dynamics

        # Parameter declaration
        in_min = coeff_ti["in_min"]
        in_max = coeff_ti["in_max"]
        max_H2_admixture = coeff_ti["max_H2_admixture"]
        alpha = coeff_ti["alpha"]
        beta = coeff_ti["beta"]
        epsilon = coeff_ti["epsilon"]
        temperature_correction = coeff_td["temperature_correction"]

        # Additional decision variables
        size_max = self.size_max

        def init_input_bounds(bds, t):
            if len(self.input_carrier) == 2:
                car = "gas"
            else:
                car = "hydrogen"
            return tuple(bounds["input"][car][t - 1, :] * size_max)

        b_tec.var_total_input = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_input_bounds,
        )

        b_tec.var_units_on = pyo.Var(
            self.set_t_performance, within=pyo.NonNegativeIntegers, bounds=(0, size_max)
        )

        # Calculate total input
        def init_total_input(const, t):
            return b_tec.var_total_input[t] == sum(
                self.input[t, car_input] for car_input in b_tec.set_input_carriers
            )

        b_tec.const_total_input = pyo.Constraint(
            self.set_t_performance, rule=init_total_input
        )

        # Constrain hydrogen input
        if len(self.input_carrier) == 2:

            def init_h2_input(const, t):
                return (
                    self.input[t, "hydrogen"]
                    <= b_tec.var_total_input[t] * max_H2_admixture
                )

            b_tec.const_h2_input = pyo.Constraint(
                self.set_t_performance, rule=init_h2_input
            )

        # LINEAR, MINIMAL PARTLOAD
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off

                def init_input_off(const, car):
                    return self.input[t, car] == 0

                dis.const_input = pyo.Constraint(
                    b_tec.set_input_carriers, rule=init_input_off
                )

                def init_output_off(const, car):
                    return self.output[t, car] == 0

                dis.const_output_off = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_off
                )

            else:  # technology on
                # input-output relation
                def init_input_output_on_el(const):
                    return (
                        self.output[t, "electricity"]
                        == (
                            alpha * b_tec.var_total_input[t]
                            + beta * b_tec.var_units_on[t]
                        )
                        * temperature_correction[t - 1]
                    )

                dis.const_input_output_on_el = pyo.Constraint(
                    rule=init_input_output_on_el
                )

                def init_input_output_on_th(const):
                    return (
                        self.output[t, "heat"]
                        == epsilon * b_tec.var_total_input[t]
                        - self.output[t, "electricity"]
                    )

                dis.const_input_output_on_th = pyo.Constraint(
                    rule=init_input_output_on_th
                )

                # min part load relation
                def init_min_input(const):
                    return b_tec.var_total_input[t] >= in_min * b_tec.var_units_on[t]

                dis.const_min_input = pyo.Constraint(rule=init_min_input)

                def init_max_input(const):
                    return b_tec.var_total_input[t] <= in_max * b_tec.var_units_on[t]

                dis.const_max_input = pyo.Constraint(rule=init_max_input)

        b_tec.dis_input_output = gdp.Disjunct(
            self.set_t_performance, s_indicators, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions
        )

        # Technologies on
        def init_n_on(const, t):
            return b_tec.var_units_on[t] <= b_tec.var_size

        b_tec.const_n_on = pyo.Constraint(self.set_t_performance, rule=init_n_on)

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(GasTurbine, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "modules_on",
            data=[model_block.var_units_on[t].value for t in self.set_t_performance],
        )
