import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

from ..utilities import fit_piecewise_function
from ...utilities import Parameters
from ..technology import Technology


class DacAdsorption(Technology):
    """
    The model resembles as Direct Air Capture technology with a modular setup. It has a heat and electricity input
    and CO2 as an output. The performance is based on data for a generic solid sorbent, as described in the
    article (see below). The performance data is fitted to the ambient temperature and humidity at the respective
    node.

    The model is based on Wiegner et al. (2022). Optimal Design and Operation of Solid Sorbent Direct Air Capture
    Processes at Varying Ambient Conditions. Industrial and Engineering Chemistry Research, 2022,
    12649–12667. https://doi.org/10.1021/acs.iecr.2c00681. It resembles operation configuration 1 without water
    spraying.
    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.options.emissions_based_on = "output"
        self.info.main_output_carrier = "CO2captured"

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits the technology performance

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(DacAdsorption, self).fit_technology_performance(climate_data, location)

        # Climate data & Number of timesteps
        time_steps = len(climate_data)

        # Number of segments
        nr_segments = self.parameters.unfitted_data["nr_segments"]

        # Read performance data from file
        performance_data = pd.read_csv(
            Path(
                "./data/technology_data/CO2Capture/DAC_adsorption_data/dac_adsorption_performance.txt"
            ),
            sep=",",
        )
        performance_data = performance_data.rename(
            columns={"T": "temp_air", "RH": "humidity"}
        )

        # Unit Conversion of input data
        performance_data.E_tot = performance_data.E_tot.multiply(
            performance_data.CO2_Out / 3600
        )  # in MWh / h
        performance_data.E_el = performance_data.E_el.multiply(
            performance_data.CO2_Out / 3600
        )  # in MWh / h
        performance_data.E_th = performance_data.E_th.multiply(
            performance_data.CO2_Out / 3600
        )  # in MWh / h
        performance_data.CO2_Out = performance_data.CO2_Out / 1000  # in t / h

        # Get humidity and temperature
        RH = copy.deepcopy(climate_data["rh"])
        T = copy.deepcopy(climate_data["temp_air"])

        # Set minimum temperature
        T.loc[T < min(performance_data.temp_air)] = min(performance_data.temp_air)

        # Derive performance points for each timestep
        def interpolate_performance_point(t, rh, point_data, var):
            zi = griddata(
                (point_data.temp_air, point_data.humidity),
                point_data[var],
                (T, RH),
                method="linear",
            )
            return zi

        CO2_Out = np.empty(shape=(len(T), len(performance_data.Point.unique())))
        E_tot = np.empty(shape=(len(T), len(performance_data.Point.unique())))
        E_el = np.empty(shape=(len(T), len(performance_data.Point.unique())))
        for point in performance_data.Point.unique():
            CO2_Out[:, point - 1] = interpolate_performance_point(
                T, RH, performance_data.loc[performance_data.Point == point], "CO2_Out"
            )
            E_tot[:, point - 1] = interpolate_performance_point(
                T, RH, performance_data.loc[performance_data.Point == point], "E_tot"
            )
            E_el[:, point - 1] = interpolate_performance_point(
                T, RH, performance_data.loc[performance_data.Point == point], "E_el"
            )

        # Derive piecewise definition
        alpha = np.empty(shape=(len(T), nr_segments))
        beta = np.empty(shape=(len(T), nr_segments))
        b = np.empty(shape=(len(T), nr_segments + 1))
        gamma = np.empty(shape=(len(T), nr_segments))
        delta = np.empty(shape=(len(T), nr_segments))
        a = np.empty(shape=(len(T), nr_segments + 1))
        el_in_max = np.empty(shape=(len(T)))
        th_in_max = np.empty(shape=(len(T)))
        out_max = np.empty(shape=(len(T)))
        total_in_max = np.empty(shape=(len(T)))

        print("Deriving performance data for DAC...")

        for timestep in range(len(T)):
            if timestep % 100 == 1:
                print("\rComplete: ", round(timestep / len(T), 2) * 100, "%", end="")
            # Input-Output relation
            y = {}
            y["CO2_Out"] = CO2_Out[timestep, :]
            time_step_fit = fit_piecewise_function(
                E_tot[timestep, :], y, int(nr_segments)
            )
            alpha[timestep, :] = time_step_fit["CO2_Out"]["alpha1"]
            beta[timestep, :] = time_step_fit["CO2_Out"]["alpha2"]
            b[timestep, :] = time_step_fit["CO2_Out"]["bp_x"]
            out_max[timestep] = max(time_step_fit["CO2_Out"]["bp_y"])
            total_in_max[timestep] = max(time_step_fit["CO2_Out"]["bp_x"])

            # Input-Input relation
            y = {}
            y["E_el"] = E_el[timestep, :]
            time_step_fit = fit_piecewise_function(
                E_tot[timestep, :], y, int(nr_segments)
            )
            gamma[timestep, :] = time_step_fit["E_el"]["alpha1"]
            delta[timestep, :] = time_step_fit["E_el"]["alpha2"]
            a[timestep, :] = time_step_fit["E_el"]["bp_x"]
            el_in_max[timestep] = max(time_step_fit["E_el"]["bp_y"])
            th_in_max[timestep] = max(time_step_fit["E_el"]["bp_x"])

        print("Complete: ", 100, "%")

        # Coefficients
        self.coeff.time_dependent_full["alpha"] = alpha
        self.coeff.time_dependent_full["beta"] = beta
        self.coeff.time_dependent_full["b"] = b
        self.coeff.time_dependent_full["gamma"] = gamma
        self.coeff.time_dependent_full["delta"] = delta
        self.coeff.time_dependent_full["a"] = a
        self.coeff.time_dependent_full["out_max"] = out_max
        self.coeff.time_dependent_full["el_in_max"] = el_in_max
        self.coeff.time_dependent_full["th_in_max"] = th_in_max
        self.coeff.time_dependent_full["total_in_max"] = total_in_max

        self.coeff.time_independent["eta_elth"] = self.parameters.unfitted_data[
            "performance"
        ]["eta_elth"]

        # Options
        self.options.other["nr_segments"] = self.parameters.unfitted_data["nr_segments"]
        self.options.other["ohmic_heating"] = self.parameters.unfitted_data[
            "ohmic_heating"
        ]

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(DacAdsorption, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        # Output Bounds
        self.bounds["output"]["CO2captured"] = np.column_stack(
            (np.zeros(shape=(time_steps)), self.coeff.time_dependent_used["out_max"])
        )

        # Input Bounds
        self.bounds["input"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                self.coeff.time_dependent_used["el_in_max"]
                + self.coeff.time_dependent_used["th_in_max"]
                / self.parameters.unfitted_data["performance"]["eta_elth"],
            )
        )
        self.bounds["input"]["heat"] = np.column_stack(
            (np.zeros(shape=(time_steps)), self.coeff.time_dependent_used["th_in_max"])
        )
        self.bounds["input"]["total"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                self.coeff.time_dependent_used["total_in_max"],
            )
        )

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type DAC_adsorption

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(DacAdsorption, self).construct_tech_model(
            b_tec, data, set_t_full, set_t_clustered
        )

        # Comments on the equations refer to the equation numbers in the paper. All equations can be looked up there.

        # Transformation required
        self.big_m_transformation_required = 1

        # DATA OF TECHNOLOGY
        nr_segments = self.options.other["nr_segments"]
        ohmic_heating = self.options.other["ohmic_heating"]

        bounds = self.bounds
        c_td = self.coeff.time_dependent_used
        c_ti = self.coeff.time_independent

        alpha = c_td["alpha"]
        beta = c_td["beta"]
        b_point = c_td["b"]
        gamma = c_td["gamma"]
        delta = c_td["delta"]
        a_point = c_td["a"]
        eta_elth = c_ti["eta_elth"]

        # Additional sets
        b_tec.set_pieces = pyo.RangeSet(1, nr_segments)

        # Additional decision variables
        b_tec.var_modules_on = pyo.Var(
            self.set_t_performance,
            domain=pyo.NonNegativeIntegers,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )

        def init_input_total_bounds(bds, t):
            return tuple(bounds["input"]["total"][t - 1] * b_tec.para_size_max)

        b_tec.var_input_total = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_input_total_bounds,
        )

        def init_input_el_bounds(bds, t):
            return tuple(bounds["input"]["electricity"][t - 1] * b_tec.para_size_max)

        b_tec.var_input_el = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_input_el_bounds,
        )

        def init_input_th_bounds(bds, t):
            return tuple(bounds["input"]["heat"][t - 1] * b_tec.para_size_max)

        b_tec.var_input_th = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_input_th_bounds,
        )

        def init_input_ohmic_bounds(bds, t):
            return tuple(
                (bounds["input"]["heat"][t - 1] / eta_elth * b_tec.para_size_max)
            )

        b_tec.var_input_ohmic = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_input_ohmic_bounds,
        )

        # Input-Output relationship (eq. 1-5)
        def init_input_output(dis, t, ind):
            # Input-output (eq. 2)
            def init_output(const):
                return (
                    self.output[t, "CO2captured"]
                    == alpha[t - 1, ind - 1] * b_tec.var_input_total[t]
                    + beta[t - 1, ind - 1] * b_tec.var_modules_on[t]
                )

            dis.const_output = pyo.Constraint(rule=init_output)

            # Lower bound on the energy input (eq. 5)
            def init_input_low_bound(const):
                return (
                    b_point[t - 1, ind - 1] * b_tec.var_modules_on[t]
                    <= b_tec.var_input_total[t]
                )

            dis.const_input_on1 = pyo.Constraint(rule=init_input_low_bound)

            # Upper bound on the energy input (eq. 5)
            def init_input_up_bound(const):
                return (
                    b_tec.var_input_total[t]
                    <= b_point[t - 1, ind] * b_tec.var_modules_on[t]
                )

            dis.const_input_on2 = pyo.Constraint(rule=init_input_up_bound)

        b_tec.dis_input_output = gdp.Disjunct(
            self.set_t_performance, b_tec.set_pieces, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in b_tec.set_pieces]

        b_tec.disjunction_input_output = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions
        )

        # Electricity-Heat relationship (eq. 7-10)
        def init_input_input(dis, t, ind):
            # Input-output (eq. 7)
            def init_input(const):
                return (
                    b_tec.var_input_el[t]
                    == gamma[t - 1, ind - 1] * b_tec.var_input_total[t]
                    + delta[t - 1, ind - 1] * b_tec.var_modules_on[t]
                )

            dis.const_output = pyo.Constraint(rule=init_input)

            # Lower bound on the energy input (eq. 10)
            def init_input_low_bound(const):
                return (
                    a_point[t - 1, ind - 1] * b_tec.var_modules_on[t]
                    <= b_tec.var_input_total[t]
                )

            dis.const_input_on1 = pyo.Constraint(rule=init_input_low_bound)

            # Upper bound on the energy input (eq. 10)
            def init_input_up_bound(const):
                return (
                    b_tec.var_input_total[t]
                    <= a_point[t - 1, ind] * b_tec.var_modules_on[t]
                )

            dis.const_input_on2 = pyo.Constraint(rule=init_input_up_bound)

        b_tec.dis_input_input = gdp.Disjunct(
            self.set_t_performance, b_tec.set_pieces, rule=init_input_input
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_input[t, i] for i in b_tec.set_pieces]

        b_tec.disjunction_input_input = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions
        )

        # Constraint of number of working modules (eq. 6)
        def init_modules_on(const, t):
            return b_tec.var_modules_on[t] <= b_tec.var_size

        b_tec.const_var_modules_on = pyo.Constraint(
            self.set_t_performance, rule=init_modules_on
        )

        # Connection thermal and electric energy demand (eq. 11)
        def init_thermal_energy(const, t):
            return (
                b_tec.var_input_th[t]
                == b_tec.var_input_total[t] - b_tec.var_input_el[t]
            )

        b_tec.const_thermal_energy = pyo.Constraint(
            self.set_t_performance, rule=init_thermal_energy
        )

        # Account for ohmic heating (eq. 12)
        def init_input_el(const, t):
            return (
                self.input[t, "electricity"]
                == b_tec.var_input_ohmic[t] + b_tec.var_input_el[t]
            )

        b_tec.const_input_el = pyo.Constraint(
            self.set_t_performance, rule=init_input_el
        )

        def init_input_th(const, t):
            return (
                self.input[t, "heat"]
                == b_tec.var_input_th[t] - b_tec.var_input_ohmic[t] * eta_elth
            )

        b_tec.const_input_th = pyo.Constraint(
            self.set_t_performance, rule=init_input_th
        )

        # If ohmic heating not allowed, set to zero
        if not ohmic_heating:

            def init_ohmic_heating(const, t):
                return b_tec.var_input_ohmic[t] == 0

            b_tec.const_ohmic_heating = pyo.Constraint(
                self.set_t_performance, rule=init_ohmic_heating
            )

        return b_tec

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report technology operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(DacAdsorption, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "modules_on",
            data=[
                model_block.var_modules_on[self.sequence[t - 1]].value
                for t in self.set_t_performance
            ],
        )
        h5_group.create_dataset(
            "ohmic_heating",
            data=[
                model_block.var_input_ohmic[self.sequence[t - 1]].value
                for t in self.set_t_performance
            ],
        )
