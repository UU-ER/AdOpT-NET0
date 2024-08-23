import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata

from ..technology import Technology
from ...utilities import link_full_resolution_to_clustered
from ..utilities import fit_piecewise_function


import logging

log = logging.getLogger(__name__)


class CCPP(Technology):
    """
    Combined Cycle Power Plant with Steam Production

    Resembles Combined Cycle Power Plant with Steam Production.
    The size of the power plant it fixed and it only possible to model the plant as
    an existing technology (i.e. it cannot be sized). The model is based on Wiegner
    et al. (2024). Optimizing the Use of Limited Amounts of Hydrogen in Existing
    Combined Heat and Power Plants.

    It is possible in the JSON file to specify the following options:

    - Size of an Oxy-fuel hydrogen burner

    - Size of a duct burner that can burn hydrogen or natural gas

    - Max share (energy-based) of hydrogen combustion in the gas turbine ("max_input")

    - How to treat steam production ("steam_production"):

        - "ignore": technology has no additional output other than electricity

        - "as heat": technology has heat output, equivalent to the sum of HP/MP
          generation

        - "as steam": technology has steam output, equivalent to the sum of HP/MP
          generation

        - "as hp/mp steam": technology has HP and MP steam output

    - What additional components in the plant should be considered. Possible options
      are:

        - "None": only CCPP

        - "DB": Duct burner

        - "OHB": Oxyfuel-hydrogen burner

    - Number of segments to use for performance function ("nr_segments").
      Recommended: 2

    **Variable declarations:**

    - Total fuel input in :math:`t`: :math:`Input_{tot, t}`

    - Fuel input to duct burner/ OHB :math:`Input_{OHB/DB, t}`

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
        Input_{H2, t} \leq in_{H2max} Input_{tot, t}

    - Turbines on:

      .. math::
        N_{on, t} \leq S

    - If technology is on:

      .. math::
        Output_{el,t} = ({\\alpha} Input_{tot, t} + {\\beta} * N_{on, t}) *f({\\Theta})

      .. math::
        Output_{th,t} = {\\epsilon} Input_{tot, t} - Output_{el,t}

      .. math::
        Input_{min} * N_{on, t} \leq Input_{tot, t} \leq Input_{max} * N_{on, t}

    - If the technology is off, input and output is set to 0:

      .. math::
         \sum(Output_{t, car}) = 0

      .. math::
         \sum(Input_{t, car}) = 0

    - Additionally, ramping rates of the technology can be constraint.

      .. math::
         -rampingrate \leq \sum(Input_{t, car}) - \sum(Input_{t-1, car}) \leq rampingrate

    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.component_options.emissions_based_on = "input"
        self.component_options.size_based_on = "output"
        self.component_options.main_input_carrier = tec_data["Performance"][
            "main_input_carrier"
        ]

        # Treatment of Steam Production
        self.component_options.other["steam_production"] = tec_data["Performance"][
            "steam_production"
        ]
        steam_production = self.component_options.other["steam_production"]

        if steam_production == "ignore":
            self.component_options.output_carrier = ["electricity"]
            log_msg = "CCPP setting: only electricity output"

        elif steam_production == "as heat":
            self.component_options.output_carrier = ["electricity", "heat"]
            log_msg = "CCPP setting: only electricity, heat output"

        elif steam_production == "as steam":
            self.component_options.output_carrier = ["electricity", "steam"]
            log_msg = "CCPP setting: only electricity, steam output"

        elif steam_production == "as hp/mp steam":
            self.component_options.output_carrier = [
                "electricity",
                "steam_hp",
                "steam_mp",
            ]
            log_msg = "CCPP setting: electricity, steam_hp, steam_mp output"

        else:
            raise Exception("steam_production setting in CCPP incorrectly specified")
        log.info(log_msg)

        # Addtional Components considered
        self.component_options.other["component"] = tec_data["Performance"]["component"]

        # Number of segments
        self.component_options.other["nr_segments"] = tec_data["Performance"][
            "nr_segments"
        ]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Performs fitting for technology type CCPP

        :param tec_data: technology data
        :param climate_data: climate data
        :return:
        """
        super(CCPP, self).fit_technology_performance(climate_data, location)

        # Climate data & Number of timesteps
        time_steps = len(climate_data)
        T = copy.deepcopy(climate_data["temp_air"])

        # Remove outliers
        T[T >= 30] = 30
        T[T <= 0] = 0

        # Determine correct reading paths
        data_path = {}
        performance_data_path = Path(__file__).parent.parent.parent.parent
        performance_data_path = (
            performance_data_path
            / "data/technology_data/PowerGeneration/CombinedCycle_fixed_size_data"
        )
        data_path["GT"] = Path("GT_fitting_data.csv")
        data_path["HP"] = Path("HP_fitting_data.csv")
        # data_path["MP"] = Path("MP_fitting_data.csv")

        # Fit GT performance
        perf_data = {}
        perf_data["GT"] = pd.read_csv(performance_data_path / data_path["GT"], sep=";")
        perf_data["HP"] = pd.read_csv(
            performance_data_path / data_path["HP"], sep=";", index_col=0, header=[0, 1]
        )
        # perf_data["MP"] = pd.read_csv(performance_data_path / data_path["GT"], sep=";")

        # Fit to temperature
        igv_positions = list(perf_data["GT"]["IGV"].unique())
        nr_igv_positions = len(igv_positions)
        gt_eta_el = np.empty(shape=(len(T), nr_igv_positions))
        gt_p_in = np.empty(shape=(len(T), nr_igv_positions))
        gt_alpha_th = np.empty(shape=(len(T), nr_igv_positions))

        pos = 0
        for igv_position in igv_positions:
            data = perf_data["GT"][perf_data["GT"]["IGV"] == igv_position]
            gt_eta_el[:, pos] = griddata(
                np.array(data["T"]), np.array(data["GT_eta"]), T, method="linear"
            )
            gt_p_in[:, pos] = griddata(
                np.array(data["T"]), np.array(data["GT_P_in"]), T, method="linear"
            )
            data["share_thermal"] = (data["GT_P_el"] + data["GT_P_th"]) / data[
                "GT_P_in"
            ]
            gt_alpha_th[:, pos] = griddata(
                np.array(data["T"]), np.array(data["share_thermal"]), T, method="linear"
            )
            pos += 1

        # Fit performance (piecewise performance GT)
        nr_segments = self.component_options.other["nr_segments"]

        alpha_el = np.empty(shape=(len(T), nr_segments))
        beta_el = np.empty(shape=(len(T), nr_segments))
        bp_el_x = np.empty(shape=(len(T), nr_segments + 1))
        bp_el_y = np.empty(shape=(len(T), nr_segments + 1))
        alpha_th = gt_alpha_th.mean(axis=1)

        log.info("Deriving GT performance for CCPP...")

        for timestep in range(len(T)):
            if timestep % 100 == 1:
                print("\rComplete: ", round(timestep / len(T), 2) * 100, "%", end="")

            # Input-Output relation
            y = {}
            y["out_el"] = gt_p_in[timestep, :] * gt_eta_el[timestep, :]
            time_step_fit = fit_piecewise_function(
                gt_p_in[timestep, :], y, int(nr_segments)
            )
            alpha_el[timestep, :] = time_step_fit["out_el"]["alpha1"]
            beta_el[timestep, :] = time_step_fit["out_el"]["alpha2"]
            bp_el_x[timestep, :] = time_step_fit["out_el"]["bp_x"]
            bp_el_y[timestep, :] = time_step_fit["out_el"]["bp_y"]

        print("Complete: ", 100, "%")

        self.processed_coeff.time_dependent_full["GT"] = {}
        self.processed_coeff.time_dependent_full["GT"]["alpha_el"] = alpha_el
        self.processed_coeff.time_dependent_full["GT"]["beta_el"] = beta_el
        self.processed_coeff.time_dependent_full["GT"]["bp_el_x"] = bp_el_x
        self.processed_coeff.time_dependent_full["GT"]["bp_el_y"] = bp_el_y
        self.processed_coeff.time_dependent_full["GT"]["alpha_th"] = alpha_th

        # Fit HP performance
        bp_hp = [0, 12.92, 100]

        alpha_gt = np.empty(shape=(len(T), nr_segments))
        alpha_hp = np.empty(shape=(len(T), nr_segments))
        alpha_mp = np.empty(shape=(len(T), nr_segments))
        alpha_cst = np.empty(shape=(len(T), nr_segments))
        if self.component_options.other["component"] == "DB":
            alpha_db = np.empty(shape=(len(T), nr_segments))
        elif self.component_options.other["component"] == "OHB":
            alpha_ohb = np.empty(shape=(len(T), nr_segments))

        for par in [1, 2]:
            data = perf_data["HP"]["GT"]["alpha_" + str(par)]
            alpha_gt[:, par - 1] = griddata(
                np.array(data.index), np.array(data), T, method="linear"
            )

            data = perf_data["HP"]["HP"]["alpha_" + str(par)]
            alpha_hp[:, par - 1] = griddata(
                np.array(data.index), np.array(data), T, method="linear"
            )

            data = perf_data["HP"]["MP"]["alpha_" + str(par)]
            alpha_mp[:, par - 1] = griddata(
                np.array(data.index), np.array(data), T, method="linear"
            )

            data = perf_data["HP"]["cst"]["alpha_" + str(par)]
            alpha_cst[:, par - 1] = griddata(
                np.array(data.index), np.array(data), T, method="linear"
            )

            if self.component_options.other["component"] == "DB":
                data = perf_data["HP"]["DB"]["alpha_" + str(par)]
                alpha_db[:, par - 1] = griddata(
                    np.array(data.index), np.array(data), T, method="linear"
                )

            elif self.component_options.other["component"] == "OHB":
                data = perf_data["HP"]["OHB"]["alpha_" + str(par)]
                alpha_ohb[:, par - 1] = griddata(
                    np.array(data.index), np.array(data), T, method="linear"
                )

        # Temperature correction factors
        f = np.empty(shape=(time_steps))
        f[T <= 6] = (
            self.input_parameters.performance_data["gamma"][0]
            * (T[T <= 6] / self.input_parameters.performance_data["T_iso"])
            + self.input_parameters.performance_data["delta"][0]
        )
        f[T > 6] = (
            self.input_parameters.performance_data["gamma"][1]
            * (T[T > 6] / self.input_parameters.performance_data["T_iso"])
            + self.input_parameters.performance_data["delta"][1]
        )

        # Derive return
        fit = {}
        fit["td"] = {}
        fit["td"]["temperature_correction"] = f.round(5)

        fit["ti"] = {}
        fit["ti"]["alpha"] = round(self.input_parameters.performance_data["alpha"], 5)
        fit["ti"]["beta"] = round(self.input_parameters.performance_data["beta"], 5)
        fit["ti"]["epsilon"] = round(
            self.input_parameters.performance_data["epsilon"], 5
        )
        fit["ti"]["in_min"] = round(self.input_parameters.performance_data["in_min"], 5)
        fit["ti"]["in_max"] = round(self.input_parameters.performance_data["in_max"], 5)
        if len(self.component_options.input_carrier) == 2:
            fit["ti"]["max_H2_admixture"] = self.input_parameters.performance_data[
                "max_H2_admixture"
            ]
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
        super(CCPP, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        bounds = {}

        # Input bounds
        bounds["input_bounds"] = {}
        for c in self.component_options.input_carrier:
            if c == "hydrogen":
                bounds["input_bounds"][c] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps))
                        * self.input_parameters.performance_data["in_max"]
                        * self.processed_coeff.time_independent["max_H2_admixture"],
                    )
                )
            else:
                bounds["input_bounds"][c] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps))
                        * self.input_parameters.performance_data["in_max"],
                    )
                )

        # Output bounds
        bounds["output_bounds"] = {}
        bounds["output_bounds"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                self.processed_coeff.time_dependent_used["temperature_correction"]
                * (
                    self.input_parameters.performance_data["in_max"]
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
                    self.input_parameters.performance_data["in_max"]
                    * self.processed_coeff.time_independent["alpha"]
                    + self.processed_coeff.time_independent["beta"]
                ),
            )
        )

        # Output Bounds
        self.bounds["output"] = bounds["output_bounds"]
        # Input Bounds
        for car in self.component_options.input_carrier:
            self.bounds["input"][car] = np.column_stack(
                (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
            )

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for gas turbines

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(CCPP, self).construct_tech_model(b_tec, data, set_t_full, set_t_clustered)

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
        size_max = self.input_parameters.size_max

        def init_input_bounds(bds, t):
            if len(self.component_options.input_carrier) == 2:
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
        if len(self.component_options.input_carrier) == 2:

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

        # RAMPING RATES
        if "ramping_time" in dynamics:
            if not dynamics["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec, data)

        return b_tec

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(CCPP, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "modules_on",
            data=[model_block.var_units_on[t].value for t in self.set_t_performance],
        )

    def _define_ramping_rates(self, b_tec, data):
        """
        Constraints the inputs for a ramping rate

        :param b_tec: technology model block
        :return:
        """
        dynamics = self.processed_coeff.dynamics

        ramping_time = dynamics["ramping_time"]

        # Calculate ramping rates
        if "ref_size" in dynamics and not dynamics["ref_size"] == -1:
            ramping_rate = dynamics["ref_size"] / ramping_time
        else:
            ramping_rate = b_tec.var_size / ramping_time

        # Constraints ramping rates
        if "ramping_const_int" in dynamics and dynamics["ramping_const_int"] == 1:

            s_indicators = range(0, 3)

            def init_ramping_operation_on(dis, t, ind):
                if t > 1:
                    if ind == 0:  # ramping constrained
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 0
                        )

                        def init_ramping_down_rate_operation(const):
                            return -ramping_rate <= sum(
                                self.input[t, car_input] - self.input[t - 1, car_input]
                                for car_input in b_tec.set_input_carriers
                            )

                        dis.const_ramping_down_rate = pyo.Constraint(
                            rule=init_ramping_down_rate_operation
                        )

                        def init_ramping_up_rate_operation(const):
                            return (
                                sum(
                                    self.input[t, car_input]
                                    - self.input[t - 1, car_input]
                                    for car_input in b_tec.set_input_carriers
                                )
                                <= ramping_rate
                            )

                        dis.const_ramping_up_rate = pyo.Constraint(
                            rule=init_ramping_up_rate_operation
                        )

                    elif ind == 1:  # startup, no ramping constraint
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 1
                        )

                    else:  # shutdown, no ramping constraint
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == -1
                        )

            b_tec.dis_ramping_operation_on = gdp.Disjunct(
                self.set_t_performance, s_indicators, rule=init_ramping_operation_on
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_ramping_operation_on[t, i] for i in s_indicators]

            b_tec.disjunction_ramping_operation_on = gdp.Disjunction(
                self.set_t_performance, rule=bind_disjunctions
            )

        else:
            if data["config"]["optimization"]["typicaldays"]["N"]["value"] == 0:
                input_aux_rr = self.input
                set_t_rr = self.set_t_performance
            else:
                if (
                    data["config"]["optimization"]["typicaldays"]["method"]["value"]
                    == 1
                ):
                    sequence = data["k_means_specs"]["sequence"]
                elif (
                    data["config"]["optimization"]["typicaldays"]["method"]["value"]
                    == 2
                ):
                    sequence = self.sequence

                # init bounds at full res
                bounds_rr_full = {
                    "input": self.fitting_class.calculate_input_bounds(
                        self.component_options.size_based_on, len(self.set_t_full)
                    )
                }

                # create input variable for full res
                def init_input_bounds(bounds, t, car):
                    return tuple(
                        bounds_rr_full["input"][car][t - 1, :]
                        * self.processed_coeff.time_independent["size_max"]
                        * self.processed_coeff.time_independent["rated_power"]
                    )

                b_tec.var_input_rr_full = pyo.Var(
                    self.set_t_full,
                    b_tec.set_input_carriers,
                    within=pyo.NonNegativeReals,
                    bounds=init_input_bounds,
                )

                b_tec.const_link_full_resolution_rr = link_full_resolution_to_clustered(
                    self.input,
                    b_tec.var_input_rr_full,
                    self.set_t_full,
                    sequence,
                    b_tec.set_input_carriers,
                )

                input_aux_rr = b_tec.var_input_rr_full
                set_t_rr = self.set_t_full

            # Ramping constraint without integers
            def init_ramping_down_rate(const, t):
                if t > 1:
                    return -ramping_rate <= sum(
                        input_aux_rr[t, car_input] - input_aux_rr[t - 1, car_input]
                        for car_input in b_tec.set_input_carriers
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_down_rate = pyo.Constraint(
                set_t_rr, rule=init_ramping_down_rate
            )

            def init_ramping_up_rate(const, t):
                if t > 1:
                    return (
                        sum(
                            input_aux_rr[t, car_input] - input_aux_rr[t - 1, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        <= ramping_rate
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_up_rate = pyo.Constraint(
                set_t_rr, rule=init_ramping_up_rate
            )

        return b_tec
