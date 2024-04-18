from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

from ..utilities import FittedPerformance, fit_piecewise_function
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

    def __init__(self, tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()

    def fit_technology_performance(self, node_data):
        """
        Fits conversion technology type 1 and returns fitted parameters as a dict

        :param performance_data: contains X and y data of technology performance
        :param performance_function_type: options for type of performance function (linear, piecewise,...)
        :param nr_seg: number of segments on piecewise defined function
        """

        # Climate data & Number of timesteps
        climate_data = node_data.data["climate_data"]
        time_steps = len(climate_data)

        # Number of segments
        nr_segments = self.performance_data["nr_segments"]

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

        # Output Bounds
        self.fitted_performance.bounds["output"]["CO2"] = np.column_stack(
            (np.zeros(shape=(time_steps)), out_max)
        )
        # Input Bounds
        self.fitted_performance.bounds["input"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                el_in_max
                + th_in_max / self.performance_data["performance"]["eta_elth"],
            )
        )
        self.fitted_performance.bounds["input"]["heat"] = np.column_stack(
            (np.zeros(shape=(time_steps)), th_in_max)
        )
        self.fitted_performance.bounds["input"]["total"] = [
            sum(x)
            for x in zip(
                self.fitted_performance.bounds["input"]["heat"],
                self.fitted_performance.bounds["input"]["electricity"],
            )
        ]
        # Coefficients
        self.fitted_performance.coefficients["alpha"] = alpha
        self.fitted_performance.coefficients["beta"] = beta
        self.fitted_performance.coefficients["b"] = b
        self.fitted_performance.coefficients["gamma"] = gamma
        self.fitted_performance.coefficients["delta"] = delta
        self.fitted_performance.coefficients["a"] = a

        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1

    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type DAC_adsorption

        :param obj b_tec: technology block
        :param Energyhub energyhub: energyhub instance
        :return: technology block
        """
        super(DacAdsorption, self).construct_tech_model(b_tec, energyhub)

        # Comments on the equations refer to the equation numbers in the paper. All equations can be looked up there.

        # Transformation required
        self.big_m_transformation_required = 1

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients
        bounds = self.fitted_performance.bounds
        nr_segments = performance_data["nr_segments"]
        ohmic_heating = performance_data["ohmic_heating"]

        # Additional sets
        b_tec.set_pieces = RangeSet(1, nr_segments)

        # Additional decision variables
        b_tec.var_modules_on = Var(
            self.set_t,
            domain=NonNegativeIntegers,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )

        def init_input_total_bounds(bds, t):
            return tuple(bounds["input"]["total"][t - 1] * b_tec.para_size_max)

        b_tec.var_input_total = Var(
            self.set_t, within=NonNegativeReals, bounds=init_input_total_bounds
        )

        def init_input_el_bounds(bds, t):
            return tuple(bounds["input"]["electricity"][t - 1] * b_tec.para_size_max)

        b_tec.var_input_el = Var(
            self.set_t, within=NonNegativeReals, bounds=init_input_el_bounds
        )

        def init_input_th_bounds(bds, t):
            return tuple(bounds["input"]["heat"][t - 1] * b_tec.para_size_max)

        b_tec.var_input_th = Var(
            self.set_t, within=NonNegativeReals, bounds=init_input_th_bounds
        )

        def init_input_ohmic_bounds(bds, t):
            return tuple(
                (
                    el - th
                    for el, th in zip(
                        bounds["input"]["electricity"][t - 1] * b_tec.para_size_max,
                        bounds["input"]["heat"][t - 1] * b_tec.para_size_max,
                    )
                )
            )

        b_tec.var_input_ohmic = Var(
            self.set_t, within=NonNegativeReals, bounds=init_input_ohmic_bounds
        )

        # Additional parameters
        alpha = coeff["alpha"]
        beta = coeff["beta"]
        b_point = coeff["b"]
        gamma = coeff["gamma"]
        delta = coeff["delta"]
        a_point = coeff["a"]
        eta_elth = performance_data["performance"]["eta_elth"]

        # Input-Output relationship (eq. 1-5)
        def init_input_output(dis, t, ind):
            # Input-output (eq. 2)
            def init_output(const):
                return (
                    self.output[t, "CO2"]
                    == alpha[t - 1, ind - 1] * b_tec.var_input_total[t]
                    + beta[t - 1, ind - 1] * b_tec.var_modules_on[t]
                )

            dis.const_output = Constraint(rule=init_output)

            # Lower bound on the energy input (eq. 5)
            def init_input_low_bound(const):
                return (
                    b_point[t - 1, ind - 1] * b_tec.var_modules_on[t]
                    <= b_tec.var_input_total[t]
                )

            dis.const_input_on1 = Constraint(rule=init_input_low_bound)

            # Upper bound on the energy input (eq. 5)
            def init_input_up_bound(const):
                return (
                    b_tec.var_input_total[t]
                    <= b_point[t - 1, ind] * b_tec.var_modules_on[t]
                )

            dis.const_input_on2 = Constraint(rule=init_input_up_bound)

        b_tec.dis_input_output = Disjunct(
            self.set_t, b_tec.set_pieces, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in b_tec.set_pieces]

        b_tec.disjunction_input_output = Disjunction(self.set_t, rule=bind_disjunctions)

        # Electricity-Heat relationship (eq. 7-10)
        def init_input_input(dis, t, ind):
            # Input-output (eq. 7)
            def init_input(const):
                return (
                    b_tec.var_input_el[t]
                    == gamma[t - 1, ind - 1] * b_tec.var_input_total[t]
                    + delta[t - 1, ind - 1] * b_tec.var_modules_on[t]
                )

            dis.const_output = Constraint(rule=init_input)

            # Lower bound on the energy input (eq. 10)
            def init_input_low_bound(const):
                return (
                    a_point[t - 1, ind - 1] * b_tec.var_modules_on[t]
                    <= b_tec.var_input_total[t]
                )

            dis.const_input_on1 = Constraint(rule=init_input_low_bound)

            # Upper bound on the energy input (eq. 10)
            def init_input_up_bound(const):
                return (
                    b_tec.var_input_total[t]
                    <= a_point[t - 1, ind] * b_tec.var_modules_on[t]
                )

            dis.const_input_on2 = Constraint(rule=init_input_up_bound)

        b_tec.dis_input_input = Disjunct(
            self.set_t, b_tec.set_pieces, rule=init_input_input
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_input[t, i] for i in b_tec.set_pieces]

        b_tec.disjunction_input_input = Disjunction(self.set_t, rule=bind_disjunctions)

        # Constraint of number of working modules (eq. 6)
        def init_modules_on(const, t):
            return b_tec.var_modules_on[t] <= b_tec.var_size

        b_tec.const_var_modules_on = Constraint(self.set_t, rule=init_modules_on)

        # Connection thermal and electric energy demand (eq. 11)
        def init_thermal_energy(const, t):
            return (
                b_tec.var_input_th[t]
                == b_tec.var_input_total[t] - b_tec.var_input_el[t]
            )

        b_tec.const_thermal_energy = Constraint(self.set_t, rule=init_thermal_energy)

        # Account for ohmic heating (eq. 12)
        def init_input_el(const, t):
            return (
                self.input[t, "electricity"]
                == b_tec.var_input_ohmic[t] + b_tec.var_input_el[t]
            )

        b_tec.const_input_el = Constraint(self.set_t, rule=init_input_el)

        def init_input_th(const, t):
            return (
                self.input[t, "heat"]
                == b_tec.var_input_th[t] - b_tec.var_input_ohmic[t] * eta_elth
            )

        b_tec.const_input_th = Constraint(self.set_t, rule=init_input_th)

        # If ohmic heating not allowed, set to zero
        if not ohmic_heating:

            def init_ohmic_heating(const, t):
                return b_tec.var_input_ohmic[t] == 0

            b_tec.const_ohmic_heating = Constraint(self.set_t, rule=init_ohmic_heating)

        return b_tec

    def write_tec_operation_results_to_group(self, h5_group, model_block):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(DacAdsorption, self).write_tec_operation_results_to_group(
            h5_group, model_block
        )

        h5_group.create_dataset(
            "modules_on",
            data=[
                model_block.var_modules_on[self.sequence[t - 1]].value
                for t in self.set_t_full
            ],
        )
        h5_group.create_dataset(
            "ohmic_heating",
            data=[
                model_block.var_input_ohmic[self.sequence[t - 1]].value
                for t in self.set_t_full
            ],
        )
