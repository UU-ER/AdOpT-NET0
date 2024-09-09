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
        data_path["MP"] = Path("MP_fitting_data.csv")

        # Fit GT performance
        perf_data = {}
        perf_data["GT"] = pd.read_csv(performance_data_path / data_path["GT"], sep=";")
        for turbine in ["HP", "MP"]:
            perf_data[turbine] = pd.read_csv(
                performance_data_path / data_path[turbine],
                sep=";",
                index_col=0,
                header=[0, 1],
            )
            perf_data[turbine] = pd.read_csv(
                performance_data_path / data_path[turbine],
                sep=";",
                index_col=0,
                header=[0, 1],
            )

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

        # Fit HP/MP performance
        bp = {}
        bp["HP"] = [0, 12.92, 100]
        bp["MP"] = [0, 42.50, 100]
        for p in ["HP", "MP"]:

            alpha_gt = np.empty(shape=(len(T), nr_segments))
            alpha_hp = np.empty(shape=(len(T), nr_segments))
            alpha_mp = np.empty(shape=(len(T), nr_segments))
            alpha_cst = np.empty(shape=(len(T), nr_segments))
            if self.component_options.other["component"] == "DB":
                alpha_db = np.empty(shape=(len(T), nr_segments))
            elif self.component_options.other["component"] == "OHB":
                alpha_ohb = np.empty(shape=(len(T), nr_segments))

            for par in [1, 2]:
                data = perf_data[p]["GT"]["alpha_" + str(par)]
                alpha_gt[:, par - 1] = griddata(
                    np.array(data.index), np.array(data), T, method="linear"
                )

                data = perf_data[p]["HP"]["alpha_" + str(par)]
                alpha_hp[:, par - 1] = griddata(
                    np.array(data.index), np.array(data), T, method="linear"
                )

                data = perf_data[p]["MP"]["alpha_" + str(par)]
                alpha_mp[:, par - 1] = griddata(
                    np.array(data.index), np.array(data), T, method="linear"
                )

                data = perf_data[p]["cst"]["alpha_" + str(par)]
                alpha_cst[:, par - 1] = griddata(
                    np.array(data.index), np.array(data), T, method="linear"
                )

                if self.component_options.other["component"] == "DB":
                    data = perf_data[p]["DB"]["alpha_" + str(par)]
                    alpha_db[:, par - 1] = griddata(
                        np.array(data.index), np.array(data), T, method="linear"
                    )

                elif self.component_options.other["component"] == "OHB":
                    data = perf_data[p]["OHB"]["alpha_" + str(par)]
                    alpha_ohb[:, par - 1] = griddata(
                        np.array(data.index), np.array(data), T, method="linear"
                    )

            self.processed_coeff.time_independent[p] = {}
            self.processed_coeff.time_independent[p]["alpha_gt"] = alpha_gt
            self.processed_coeff.time_independent[p]["alpha_hp"] = alpha_hp
            self.processed_coeff.time_independent[p]["alpha_mp"] = alpha_mp
            self.processed_coeff.time_independent[p]["alpha_cst"] = alpha_cst
            self.processed_coeff.time_independent[p]["bp_" + p.lower()] = bp[p]
            if self.component_options.other["component"] == "DB":
                self.processed_coeff.time_independent[p]["alpha_db"] = alpha_db
            elif self.component_options.other["component"] == "OHB":
                self.processed_coeff.time_independent[p]["alpha_ohb"] = alpha_ohb

        data = self.input_parameters.performance_data
        self.processed_coeff.time_independent["eta_stg"] = data[
            "steam_turbine_generator_efficiency"
        ]
        self.processed_coeff.time_independent["hp_steam_max"] = data[
            "max_steam_extract_HP"
        ]
        self.processed_coeff.time_independent["mp_steam_max"] = data[
            "max_steam_extract_MP"
        ]
        self.processed_coeff.time_independent["kappa_steam"] = data[
            "max_steam_extract_total"
        ]
        self.processed_coeff.time_independent["size_db"] = data["size_db"]
        self.processed_coeff.time_independent["size_ohb"] = data["size_ohb"]
        self.processed_coeff.time_independent["max_h2_in"] = data["max_h2_in"]
        self.processed_coeff.time_independent["max_h2_in_gt"] = data["max_h2_in_gt"]
        self.processed_coeff.time_independent["hp_max_p"] = 27
        self.processed_coeff.time_independent["hp_min_p"] = 3
        self.processed_coeff.time_independent["mp_max_p"] = 90
        self.processed_coeff.time_independent["mp_min_p"] = 12

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(CCPP, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        # Input bounds
        min_in = np.zeros(shape=(time_steps))

        # H2
        max_h2 = (
            np.ones(shape=(time_steps))
            * self.processed_coeff.time_independent["max_h2_in"]
        )

        # NG
        if self.component_options.other["component"] == "DB":
            additional_ng_in = self.processed_coeff.time_independent["size_db"]
        else:
            additional_ng_in = 0
        max_ng = (
            self.processed_coeff.time_dependent_full["GT"]["bp_el_x"][:, -1]
            + additional_ng_in
        )

        self.bounds["input"]["hydrogen"] = np.column_stack((min_in, max_h2))
        self.bounds["input"]["gas"] = np.column_stack((min_in, max_ng))
        self.bounds["input"]["total"] = np.column_stack((min_in, max_ng + max_h2))

        # Output bounds
        min_el = np.zeros(shape=(time_steps))
        max_el = np.ones(shape=(time_steps)) * 3000
        self.bounds["output"]["electricity"] = np.column_stack((min_el, max_el))

        # TODO: Calculate input bounds of el, heat, mp steam, hp steam

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

        b_tec = self._define_additional_vars(b_tec)
        b_tec = self._define_tec_global_balances(b_tec)
        b_tec = self._define_performance(b_tec)

        self.big_m_transformation_required = 1

        # RAMPING RATES
        dynamics = self.processed_coeff.dynamics
        if "ramping_time" in dynamics:
            if not dynamics["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec, data)

        b_tec.pprint()

        return b_tec

    def _define_additional_vars(self, b_tec):

        # DATA OF TECHNOLOGY
        coeff_td = self.processed_coeff.time_dependent_used
        coeff_ti = self.processed_coeff.time_independent

        # GT - total fuel input
        def init_input_total_gt_bounds(bds, t):
            return tuple((0, coeff_td["GT"]["bp_el_x"][t - 1, -1]))

        b_tec.var_gt_input = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_input_total_gt_bounds,
        )

        # GT - ng fuel input
        def init_input_ng_gt_bounds(bds, t):
            return tuple((0, coeff_td["GT"]["bp_el_x"][t - 1, -1]))

        b_tec.var_gt_ng_input = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_input_ng_gt_bounds,
        )

        # GT - h2 fuel input
        def init_input_h2_gt_bounds(bds, t):
            min_h2 = min(
                coeff_td["GT"]["bp_el_x"][t - 1, -1] * coeff_ti["max_h2_in_gt"],
                coeff_ti["max_h2_in"],
            )
            return tuple((0, min_h2))

        b_tec.var_gt_h2_input = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_input_h2_gt_bounds,
        )

        # GT - P electric
        def init_gt_p_el_bounds(bds, t):
            return tuple((0, coeff_td["GT"]["bp_el_y"][t - 1, -1]))

        b_tec.var_gt_p_el = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_gt_p_el_bounds,
        )

        # GT - P thermal
        # TODO: calculate bounds
        def init_gt_p_th_bounds(bds, t):
            return tuple((0, 1000))

        b_tec.var_gt_p_th = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_gt_p_th_bounds,
        )

        # Duct Burner - input
        if self.component_options.other["component"] == "DB":

            def init_db_input_bounds(bds, t):
                return tuple((0, coeff_ti["size_db"]))

            b_tec.var_db_h2_input = pyo.Var(
                self.set_t_performance,
                within=pyo.NonNegativeReals,
                bounds=init_db_input_bounds,
            )
            b_tec.var_db_ng_input = pyo.Var(
                self.set_t_performance,
                within=pyo.NonNegativeReals,
                bounds=init_db_input_bounds,
            )
            b_tec.var_db_input = pyo.Var(
                self.set_t_performance,
                within=pyo.NonNegativeReals,
                bounds=init_db_input_bounds,
            )
        else:
            b_tec.var_db_h2_input = pyo.Param(
                self.set_t_performance, domain=pyo.Reals, initialize=0
            )
            b_tec.var_db_ng_input = pyo.Param(
                self.set_t_performance, domain=pyo.Reals, initialize=0
            )
            b_tec.var_db_input = pyo.Param(
                self.set_t_performance, domain=pyo.Reals, initialize=0
            )

        # OHB - input
        if self.component_options.other["component"] == "OHB":

            def init_ohb_input_bounds(bds, t):
                return tuple((0, coeff_ti["size_ohb"]))

            b_tec.var_ohb_h2_input = pyo.Var(
                self.set_t_performance,
                within=pyo.NonNegativeReals,
                bounds=init_ohb_input_bounds,
            )
        else:
            b_tec.var_ohb_h2_input = pyo.Param(
                self.set_t_performance, domain=pyo.Reals, initialize=0
            )

        # ST - HP
        def init_hp_output_bounds(bds, t):
            return tuple((0, coeff_ti["hp_max_p"]))

        b_tec.var_hp_p_el = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_hp_output_bounds,
        )

        # ST - MP
        def init_mp_output_bounds(bds, t):
            return tuple((0, coeff_ti["mp_max_p"]))

        b_tec.var_mp_p_el = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_mp_output_bounds,
        )

        # Steam Output
        # TODO: Calculate bounds
        def init_mp_hp_bounds(bds, t):
            return tuple((0, 1000))

        b_tec.var_mp_p_hp = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_mp_hp_bounds,
        )

        def init_mp_mp_bounds(bds, t):
            return tuple((0, 1000))

        b_tec.var_mp_p_mp = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_mp_mp_bounds,
        )

        def init_hp_hp_bounds(bds, t):
            return tuple((0, 1000))

        b_tec.var_hp_p_hp = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_hp_hp_bounds,
        )

        def init_hp_mp_bounds(bds, t):
            return tuple((0, 1000))

        b_tec.var_hp_p_mp = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=init_hp_mp_bounds,
        )

        if not b_tec.find_component("var_x"):
            b_tec.var_x = pyo.Var(
                self.set_t_performance, domain=pyo.NonNegativeIntegers, bounds=(0, 1)
            )

        return b_tec

    def _define_tec_global_balances(self, b_tec):

        # DATA OF TECHNOLOGY
        coeff_td = self.processed_coeff.time_dependent_used
        coeff_ti = self.processed_coeff.time_independent

        # Calculate total hydrogen input
        def init_total_input_h2(const, t):
            return (
                b_tec.var_input[t, "hydrogen"]
                == b_tec.var_gt_h2_input[t]
                + b_tec.var_db_h2_input[t]
                + b_tec.var_ohb_h2_input[t]
            )

        b_tec.const_total_input_h2 = pyo.Constraint(
            self.set_t_performance,
            rule=init_total_input_h2,
        )

        # Calculate total gas input
        def init_total_input_ng(const, t):
            return (
                b_tec.var_input[t, "gas"]
                == b_tec.var_gt_ng_input[t] + b_tec.var_db_ng_input[t]
            )

        b_tec.const_total_input_ng = pyo.Constraint(
            self.set_t_performance,
            rule=init_total_input_ng,
        )

        # Calculate total electric output
        def init_total_output_el(const, t):
            return b_tec.var_output[t, "electricity"] == b_tec.var_hp_p_el[
                t
            ] + coeff_ti["eta_stg"] * (b_tec.var_mp_p_el[t] + b_tec.var_gt_p_el[t])

        b_tec.const_total_output_el = pyo.Constraint(
            self.set_t_performance,
            rule=init_total_output_el,
        )
        # TODO: Calculate total heat, hp, mp steam output

        # Constrain total H2 input
        def init_total_input_h2_max(const, t):
            return b_tec.var_input[t, "hydrogen"] <= coeff_ti["max_h2_in"]

        b_tec.const_total_input_h2_max = pyo.Constraint(
            self.set_t_performance,
            rule=init_total_input_h2_max,
        )

        return b_tec

    def _define_performance(self, b_tec):
        coeff_td = self.processed_coeff.time_dependent_used
        coeff_ti = self.processed_coeff.time_independent
        nr_segments = self.component_options.other["nr_segments"]
        gt_alpha_th = coeff_td["GT"]["alpha_th"]
        gt_alpha = coeff_td["GT"]["alpha_el"]
        gt_beta = coeff_td["GT"]["beta_el"]
        gt_bp_x = coeff_td["GT"]["bp_el_x"]

        # Total input to GT
        def init_total_input_gt(const, t):
            return (
                b_tec.var_gt_input[t]
                == b_tec.var_gt_ng_input[t] + b_tec.var_gt_h2_input[t]
            )

        b_tec.const_total_input_gt = pyo.Constraint(
            self.set_t_performance,
            rule=init_total_input_gt,
        )

        # Max H2 in
        def init_max_h2_input_gt(const, t):
            return (
                b_tec.var_gt_h2_input[t]
                <= coeff_ti["max_h2_in_gt"] * b_tec.var_gt_input[t]
            )

        b_tec.const_max_h2_input_gt = pyo.Constraint(
            self.set_t_performance,
            rule=init_max_h2_input_gt,
        )

        s_indicators = range(0, nr_segments + 1)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off

                dis.const_x_off = pyo.Constraint(expr=b_tec.var_x[t] == 0)

                def init_input_gt_ng_off(const):
                    return b_tec.var_gt_ng_input[t] == 0

                dis.const_input_gt_ng_off = pyo.Constraint(rule=init_input_gt_ng_off)

                def init_input_gt_h2_off(const):
                    return b_tec.var_gt_h2_input[t] == 0

                dis.const_input_gt_h2_off = pyo.Constraint(rule=init_input_gt_h2_off)

                def init_output_gt_el_off(const):
                    return b_tec.var_gt_p_el[t] == 0

                dis.const_output_gt_el_off = pyo.Constraint(rule=init_output_gt_el_off)

                def init_output_mp_off(const):
                    return b_tec.var_mp_p_el[t] == 0

                dis.const_output_mp_off = pyo.Constraint(rule=init_output_mp_off)

                def init_output_hp_off(const):
                    return b_tec.var_hp_p_el[t] == 0

                dis.const_output_hp_off = pyo.Constraint(rule=init_output_hp_off)

                def init_input_off(const, car_input):
                    return self.input[t, car_input] == 0

                dis.const_input = pyo.Constraint(
                    b_tec.set_input_carriers, rule=init_input_off
                )

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_off
                )

                if self.component_options.other["component"] == "DB":

                    def init_input_db_off(const):
                        return b_tec.var_db_input[t] == 0

                    dis.const_input_db_off = pyo.Constraint(rule=init_input_db_off)

                    def init_input_db_ng_off(const):
                        return b_tec.var_db_ng_input[t] == 0

                    dis.const_input_db_ng_off = pyo.Constraint(
                        rule=init_input_db_ng_off
                    )

                    def init_input_db_h2_off(const):
                        return b_tec.var_db_h2_input[t] == 0

                    dis.const_input_db_h2_off = pyo.Constraint(
                        rule=init_input_db_h2_off
                    )

                if self.component_options.other["component"] == "OHB":

                    def init_input_ohb_h2_off(const):
                        return b_tec.var_ohb_h2_input[t] == 0

                    dis.const_input_ohb_h2_off = pyo.Constraint(
                        rule=init_input_ohb_h2_off
                    )

            else:  # piecewise definition

                dis.const_x_on = pyo.Constraint(expr=b_tec.var_x[t] == 1)

                def init_input_on1(const):
                    return gt_bp_x[t - 1, ind - 1] <= b_tec.var_gt_input[t]

                dis.const_input_on1 = pyo.Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return b_tec.var_gt_input[t] <= gt_bp_x[t - 1, ind]

                dis.const_input_on2 = pyo.Constraint(rule=init_input_on2)

                def init_output_gt_on(const):
                    return (
                        b_tec.var_gt_p_el[t]
                        == gt_alpha[t - 1, ind - 1] * b_tec.var_gt_input[t]
                        + gt_beta[t - 1, ind - 1]
                    )

                dis.const_input_output_gt_on = pyo.Constraint(rule=init_output_gt_on)

        b_tec.dis_input_output = gdp.Disjunct(
            self.set_t_performance, s_indicators, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions
        )

        def init_output_gt_th(const, t):
            return b_tec.var_gt_p_th[t] == gt_alpha_th[t - 1] * (
                b_tec.var_gt_input[t] - b_tec.var_gt_p_el[t]
            )

        b_tec.const_output_gt_th = pyo.Constraint(
            self.set_t_performance, rule=init_output_gt_th
        )

        st_hp_alpha_gt = coeff_ti["HP"]["alpha_gt"]
        st_hp_alpha_hp = coeff_ti["HP"]["alpha_hp"]
        st_hp_alpha_mp = coeff_ti["HP"]["alpha_mp"]
        st_hp_alpha_cst = coeff_ti["HP"]["alpha_cst"]
        st_hp_bp_x = coeff_ti["HP"]["bp_hp"]
        if self.component_options.other["component"] == "DB":
            st_hp_alpha_db = coeff_ti["HP"]["alpha_db"]
        elif self.component_options.other["component"] == "OHB":
            st_hp_alpha_ohb = coeff_ti["HP"]["alpha_ohb"]
        s_indicators = range(0, nr_segments + 1)

        def init_perf_hp(dis, t, ind):
            if ind == 0:  # technology off
                dis.const_x_off = pyo.Constraint(expr=b_tec.var_x[t] == 0)

            else:  # technology on
                dis.const_x_on = pyo.Constraint(expr=b_tec.var_x[t] == 1)

                # HP bounds
                def init_hp_on1(const):
                    return st_hp_bp_x[ind - 1] <= b_tec.var_hp_p_el[t]

                dis.const_hp_on1 = pyo.Constraint(rule=init_hp_on1)

                def init_hp_on2(const):
                    return b_tec.var_hp_p_el[t] <= st_hp_bp_x[ind]

                dis.const_hp_on2 = pyo.Constraint(rule=init_hp_on2)

                # HP performance
                def init_hp_performance(const):
                    if self.component_options.other["component"] == "DB":
                        add = st_hp_alpha_db[t - 1, ind - 1] * b_tec.var_db_input[t]
                    elif self.component_options.other["component"] == "OHB":
                        add = (
                            st_hp_alpha_ohb[t - 1, ind - 1] * b_tec.var_ohb_h2_input[t]
                        )
                    else:
                        add = 0

                    return (
                        b_tec.var_hp_p_el[t]
                        == st_hp_alpha_gt[t - 1, ind - 1] * b_tec.var_gt_p_th[t]
                        + st_hp_alpha_hp[t - 1, ind - 1] * b_tec.var_hp_p_hp[t]
                        + st_hp_alpha_mp[t - 1, ind - 1] * b_tec.var_hp_p_mp[t]
                        + add
                        + st_hp_alpha_cst[t - 1, ind - 1]
                    )

                dis.const_hp_performance = pyo.Constraint(rule=init_hp_performance)

        b_tec.dis_performance_hp = gdp.Disjunct(
            self.set_t_performance, s_indicators, rule=init_perf_hp
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_performance_hp[t, i] for i in s_indicators]

        b_tec.disjunction_performance_hp = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions
        )

        st_mp_alpha_gt = coeff_ti["MP"]["alpha_gt"]
        st_mp_alpha_hp = coeff_ti["MP"]["alpha_hp"]
        st_mp_alpha_mp = coeff_ti["MP"]["alpha_mp"]
        st_mp_alpha_cst = coeff_ti["MP"]["alpha_cst"]
        st_mp_bp_x = coeff_ti["MP"]["bp_mp"]

        if self.component_options.other["component"] == "DB":
            st_mp_alpha_db = coeff_ti["MP"]["alpha_db"]
        elif self.component_options.other["component"] == "OHB":
            st_mp_alpha_ohb = coeff_ti["MP"]["alpha_ohb"]

        s_indicators = range(0, nr_segments + 1)

        def init_perf_mp(dis, t, ind):
            if ind == 0:  # technology off
                dis.const_x_off = pyo.Constraint(expr=b_tec.var_x[t] == 0)

            else:  # technology on
                dis.const_x_on = pyo.Constraint(expr=b_tec.var_x[t] == 1)

                # MP bounds
                def init_mp_on1(const):
                    return st_mp_bp_x[ind - 1] <= b_tec.var_mp_p_el[t]

                dis.const_mp_on1 = pyo.Constraint(rule=init_mp_on1)

                def init_mp_on2(const):
                    return b_tec.var_mp_p_el[t] <= st_mp_bp_x[ind]

                dis.const_mp_on2 = pyo.Constraint(rule=init_mp_on2)

                # MP performance
                def init_mp_performance(const):
                    if self.component_options.other["component"] == "DB":
                        add = st_mp_alpha_db[t - 1, ind - 1] * b_tec.var_db_input[t]
                    elif self.component_options.other["component"] == "OHB":
                        add = (
                            st_mp_alpha_ohb[t - 1, ind - 1] * b_tec.var_ohb_h2_input[t]
                        )
                    else:
                        add = 0
                    return (
                        b_tec.var_mp_p_el[t]
                        == st_mp_alpha_gt[t - 1, ind - 1] * b_tec.var_gt_p_th[t]
                        + st_mp_alpha_hp[t - 1, ind - 1] * b_tec.var_mp_p_hp[t]
                        + st_mp_alpha_mp[t - 1, ind - 1] * b_tec.var_mp_p_mp[t]
                        + add
                        + st_mp_alpha_cst[t - 1, ind - 1]
                    )

                dis.const_mp_performance = pyo.Constraint(rule=init_mp_performance)

        b_tec.dis_performance_mp = gdp.Disjunct(
            self.set_t_performance, s_indicators, rule=init_perf_mp
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_performance_mp[t, i] for i in s_indicators]

        b_tec.disjunction_performance_mp = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions
        )

        return b_tec

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(CCPP, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "gt_input",
            data=[model_block.var_gt_input[t].value for t in self.set_t_performance],
        )
        h5_group.create_dataset(
            "gt_ng_input",
            data=[model_block.var_gt_ng_input[t].value for t in self.set_t_performance],
        )
        h5_group.create_dataset(
            "gt_h2_input",
            data=[model_block.var_gt_h2_input[t].value for t in self.set_t_performance],
        )
        h5_group.create_dataset(
            "gt_p_el",
            data=[model_block.var_gt_p_el[t].value for t in self.set_t_performance],
        )
        h5_group.create_dataset(
            "gt_p_th",
            data=[model_block.var_gt_p_th[t].value for t in self.set_t_performance],
        )
        h5_group.create_dataset(
            "hp_p_el",
            data=[model_block.var_hp_p_el[t].value for t in self.set_t_performance],
        )
        h5_group.create_dataset(
            "mp_p_el",
            data=[model_block.var_mp_p_el[t].value for t in self.set_t_performance],
        )
        if self.component_options.other["component"] == "DB":
            h5_group.create_dataset(
                "db_input",
                data=[
                    model_block.var_db_input[t].value for t in self.set_t_performance
                ],
            )
            h5_group.create_dataset(
                "db_h2_input",
                data=[
                    model_block.var_db_h2_input[t].value for t in self.set_t_performance
                ],
            )
            h5_group.create_dataset(
                "db_ng_input",
                data=[
                    model_block.var_db_ng_input[t].value for t in self.set_t_performance
                ],
            )
        if self.component_options.other["component"] == "OHB":
            h5_group.create_dataset(
                "ohb_h2_input",
                data=[
                    model_block.var_ohb_h2_input[t].value
                    for t in self.set_t_performance
                ],
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
