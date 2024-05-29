import pyomo.gdp as gdp
import pyomo.environ as pyo
from warnings import warn
import numpy as np
import pandas as pd

from ..component import ModelComponent, ProcessedCoefficients
from ..utilities import (
    annualize,
    set_discount_rate,
    link_full_resolution_to_clustered,
    determine_variable_scaling,
    determine_constraint_scaling,
)
from .utilities import set_capex_model
from ...logger import log_event
from .ccs import fit_ccs_coeff


class Technology(ModelComponent):
    """
    Class to read and manage data for technologies

    This class is parent class to all generic and specific technologies. It creates the variables, parameters,
    constraints and sets of a technology.

    This function is extented in the generic/specific technology classes. It adds Sets, Parameters, Variables and
    Constraints that are common for all technologies
    The following description is true for new technologies. For existing technologies a few adaptions are made
    (see below).
    When CCS is available, we add heat and electricity to the input carriers Set and CO2captured to the output
    carriers Set. Moreover, we create extra Parameters and Variables equivalent to the ones created for the
    technology, but specific for CCS. In addition, we create Variables that are the sum of the input, output,
    CAPEX and OPEX of the technology and of CCS. We calculate the emissions of the
    technology discounting already
    what is being captured by the CCS.

    **Set declarations:**

    - set_input_carriers: Set of input carriers
    - set_output_carriers: Set of output carriers

    If ccs is possible:

    - set_input_carriers_ccs: Set of ccs input carriers
    - set_output_carriers_ccs: Set of ccs output carriers

    ** Set declarations for time aggregation

    Three sets are declared for each technology. These are required for time
    averaging algorithms:

    - set_t_full: set of all time steps before clustering
    - set_t_performance: set of time steps, on which the technology performance is
      based on
    - set_t_global: set of time steps, on which the energy balance is based on

    **Parameter declarations:**

    The following is a list of declared pyomo parameters.

    - para_size_min: minimal possible size
    - para_size_max: maximal possible size
    - para_unit_capex: investment costs per unit
    - para_unit_capex_annual: Unit CAPEX annualized (annualized from given data on
      up-front CAPEX, lifetime and discount rate)
    - para_fix_capex: fixed costs independent of size
    - para_fix_capex_annual: Fixed CAPEX annualized (annualized from given data on
      up-front CAPEX, lifetime and discount rate)
    - para_opex_variable: operational cost EUR/output or input
    - para_opex_fixed: fixed opex as fraction of annualized capex
    - para_tec_emissionfactor: emission factor per output or input

    If ccs is possible:

    - para_size_min_ccs: minimal possible size
    - para_size_max_ccs: maximal possible size
    - para_unit_capex_annual_ccs: investment costs per unit (annualized from given data on up-front CAPEX, lifetime
      and discount rate)
    - para_fix_capex_annual_ccs: Fixed CAPEX annualized (annualized from given data on
      up-front CAPEX, lifetime and discount rate)
    - para_opex_variable_ccs: operational cost EUR/output or input
    - para_opex_fixed_ccs: fixed opex as fraction of annualized capex

    For existing technologies:

    - para_size_initial: initial size
    - para_decommissioning_cost: Decommissioning cost

    **Variable declarations:**

    - var_size: Size of the technology, can be integer or continuous
    - var_input: input to the technology, defined for each input carrier and time slice
    - var_output: output of the technology, defined for each output carrier and time
      slice
    - var_input_tot: input aggregation of technology and ccs input
    - var_output_tot: output aggregation of technology and ccs output
    - var_capex: annualized investment of the technology
    - var_opex_variable: variable operation costs, defined for each time slice
    - var_opex_fixed: fixed operational costs
    - var_capex_tot: aggregation of technology and ccs capex
    - var_capex_aux: auxiliary variable to calculate the fixed opex of existing technologies
    - var_opex_variable_tot: aggregation of technology and ccs opex variable, defined for
      each time slice
    - var_opex_fixed_tot: aggregation of technology and ccs opex fixed
    - var_tec_emissions_pos: positive emissions, defined per time slice
    - var_tec_emissions_neg: negative emissions, defined per time slice

    If ccs is possible:

    - var_size_ccs: Size of ccs
    - var_input_ccs: input to the ccs component, defined for each ccs input carrier and
      time slice
    - var_output_ccs: output from the ccs component, defined for each ccs output carrier
      and time slice
    - var_capex_ccs: annualized investment of ccs
    - var_capex_aux_ccs: auxiliary variable to calculate the fixed opex of existing ccs
    - var_opex_variable_ccs: variable operation costs, defined for each time slice
    - var_opex_fixed_ccs: fixed operational costs

    **Constraint declarations**

    - For new technologies, CAPEX, can be linear (for ``capex_model == 1``), piecewise
      linear (for ``capex_model == 2``) or linear with a fixed cost when the
      technology is installed (for ``capex_model == 3``). The capex model can
      also be overwritten in the children technology classes  (for ``capex_model ==
      4``). Linear is defined as:

        .. math::
            capex_{aux} = size * capex_{unitannual}

      while linear with fixed installation costs is defined as. Note that capex_aux is
      zero if the technology is not installed:

        .. math::
            capex_{aux} = size * capex_{unitannual} + capex_{fixed}

      Existing technologies, i.e. existing = 1, can be decommissioned (decommission = 1) or not (decommission = 0).
      For technologies that cannot be decommissioned, the size is fixed to the size
      given in the technology data.
      For technologies that can be decommissioned, the size can be smaller or equal to the initial size. Reducing the
      size comes at the decommissioning costs specified in the economics of the technology.
      The fixed opex is calculated by determining the capex that the technology would have costed if newly build and
      then taking the respective opex_fixed share of this. This is done with the auxiliary variable capex_aux.

    - For existing technologies that can be decommissioned, the CAPEX equal to
      the decommissioning costs:

        .. math::
            capex = (size_{initial} - size) * decommissioningcost

    - Variable OPEX: variable opex is defined in terms of the input, with the
      exception of DAC_Adsorption, RES and CONV4, where it is defined per unit of
      output:

        .. math::
            opexvar_{t} = Input_{t, maincarrier} * opex_{var}

    - Fixed OPEX: defined as a fraction of annual CAPEX:

        .. math::
            opexfix = capex * opex_{fix}

    - Input aggregation: aggregates total input from technology
      and ccs. In case there is no ccs, input_ccs is zero:

        .. math::
            input_{t, car} + input_ccs_{t, car} = input_tot_{t, car}

    - Output aggregation: aggregates total output from technology
      and ccs. In case there is no ccs, output_ccs is zero:

        .. math::
            output_{t, car} + output_ccs_{t, car} = output_tot_{t, car}

    - Capex aggregation: aggregates capex of technology
      and ccs. In case there is no ccs, capex_ccs is zero:

        .. math::
            capex + capex_{ccs} = capex_{tot}

    - Opex variable aggregation: aggregates opex variable of technology
      and ccs. In case there is no ccs, var_opex_variable_ccs is zero:

        .. math::
            opex_{variable, t} + opex_{variable,ccs, t} =  opex_{variable,tot, t}

    - Opex fixed aggregation: aggregates opex fixed of technology
      and ccs. In case there is no ccs, opex_fixed_ccs is zero:

        .. math::
            opex_{fixed} + opex_{fixed,ccs} =  opex_{fixed,tot}

    - Emissions: depending if they are based on input or output and depending if
      emission factor is negative or positive

        .. math::
            emissions_{pos/neg,t} = output_{maincarrier, t} * emissionfactor_{pos}

    If CCS is possible the following constraints apply:

    - Input carriers are given by:

    .. math::
        input_CCS_{car} <= inputRatio_{carrier} * output_CCS/captureRate
        input_tot_{car} = inputTec_{car} + input_CCS_{car}

    - CO2 captured output is constrained by:

    .. math::
        output_CCS <= input(output)_{tec} * emissionFactor * captureRate

    - The total output are given by:

    .. math::
        output_tot_{car} = outputTec_{car} + output_CCS_{car}

    - Emissions of the technolgy are:

    .. math::
        emissions_{tec} = input(output)_{tec} * emissionFactor - output_CCS

    - CAPEX is given by

    .. math::
        CAPEX_CCS = Size_CCS * UnitCost_CCS + FixCost_CCS
        CAPEX_tot = CAPEX_CCS + CAPEX_{tec}

    - Fixed OPEX: defined as a fraction of annual CAPEX:

    .. math::
        OPEXfix_CCS = CAPEX_CCS * opex_CCS
        OPEX_tot = OPEX_CCS + OPEX_{tec}
    """

    def __init__(self, tec_data: dict):
        """
        Initializes technology class from technology data

        The technology name needs to correspond to the name of a JSON file in ./data/technology_data.

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        # Technology data
        self.ccs_data = None
        self.ccs_component = None

        # Modelling attributes
        self.input = None
        self.output = None
        self.set_t_full = None
        self.set_t_performance = None
        self.set_t_global = None
        self.sequence = None

        # Scaling factors
        self.scaling_factors = None
        if "ScalingFactors" in tec_data:
            self.scaling_factors = tec_data["ScalingFactors"]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits technology performance (bounds and coefficients).

        This function is overwritten in the technology child classes

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        unfitted_coeff = self.input_parameters
        time_independent = {}

        # Size
        time_independent["size_min"] = unfitted_coeff.size_min
        if not self.existing:
            time_independent["size_max"] = unfitted_coeff.size_max
        else:
            time_independent["size_max"] = unfitted_coeff.size_initial
            time_independent["size_initial"] = unfitted_coeff.size_initial

        # Emissions
        time_independent["emission_factor"] = unfitted_coeff.performance_data[
            "emission_factor"
        ]

        # Other
        time_independent["rated_power"] = unfitted_coeff.rated_power
        time_independent["min_part_load"] = unfitted_coeff.min_part_load
        time_independent["standby_power"] = unfitted_coeff.standby_power

        # Dynamics
        dynamics = {}
        dynamics_parameter = [
            "ramping_time",
            "ref_size",
            "ramping_const_int",
            "standby_power",
            "min_uptime",
            "min_downtime",
            "SU_time",
            "SD_time",
            "SU_load",
            "SD_load",
            "max_startups",
        ]
        for p in dynamics_parameter:
            if p in unfitted_coeff.performance_data:
                dynamics[p] = unfitted_coeff.performance_data[p]

        # Write to self
        self.processed_coeff.time_independent = time_independent
        self.processed_coeff.dynamics = dynamics

        # CCS
        if self.component_options.ccs_possible:
            co2_concentration = self.input_parameters.performance_data["ccs"][
                "co2_concentration"
            ]
            self.ccs_data["name"] = "CCS"
            self.ccs_data["tec_type"] = self.component_options.ccs_type
            self.ccs_component = fit_ccs_coeff(
                co2_concentration, self.ccs_data, climate_data
            )

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used. Overwritten in child classes
        """
        pass

    def _calculate_ccs_bounds(self):
        """
        Calculates bounds of ccs
        """
        time_steps = len(self.set_t_performance)

        # Calculate input and output bounds
        for car in self.ccs_component.component_options.input_carrier:
            self.ccs_component.bounds["input"][car] = np.column_stack(
                (
                    np.zeros(shape=(time_steps)),
                    np.ones(shape=(time_steps))
                    * self.ccs_component.processed_coeff.time_independent[
                        "input_ratios"
                    ][car],
                )
            )
        for car in self.ccs_component.component_options.output_carrier:
            self.ccs_component.bounds["output"][car] = np.column_stack(
                (
                    np.zeros(shape=(time_steps)),
                    np.ones(shape=(time_steps))
                    * self.ccs_component.processed_coeff.time_independent[
                        "capture_rate"
                    ],
                )
            )

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Construct the technology model with all required parameters, variable, sets,...

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        # LOG
        log_event(f"\t - Adding Technology {self.name}")

        # TECHNOLOGY DATA
        config = data["config"]

        # SET T
        self.set_t_full = set_t_full

        # MODELING TYPICAL DAYS
        technologies_modelled_with_full_res = config["optimization"]["typicaldays"][
            "technologies_with_full_res"
        ]["value"]

        if config["optimization"]["typicaldays"]["N"]["value"] == 0:
            # everything with full resolution
            self.component_options.modelled_with_full_res = True
            self.set_t_performance = set_t_full
            self.set_t_global = set_t_full
            self.sequence = list(self.set_t_performance)

        elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
            # everything with reduced resolution
            self.component_options.modelled_with_full_res = False
            self.set_t_performance = set_t_clustered
            self.set_t_global = set_t_clustered
            self.sequence = list(self.set_t_performance)

        elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
            # resolution of balances is full, so interactions with them also need to
            # be full resolution
            self.set_t_global = set_t_full

            if (
                self.component_options.technology_model
                in technologies_modelled_with_full_res
            ):
                # technologies modelled with full resolution
                self.component_options.modelled_with_full_res = True
                self.component_options.lower_res_than_full = False
                self.set_t_performance = self.set_t_full
                self.sequence = list(self.set_t_performance)
            else:
                # technologies modelled with reduced resolution
                self.component_options.modelled_with_full_res = False
                self.component_options.lower_res_than_full = True
                self.set_t_performance = set_t_clustered
                self.sequence = data["k_means_specs"]["sequence"]

        # Coefficients
        if self.component_options.modelled_with_full_res:
            if config["optimization"]["timestaging"]["value"] == 0:
                self.processed_coeff.time_dependent_used = (
                    self.processed_coeff.time_dependent_full
                )
            else:
                self.processed_coeff.time_dependent_used = (
                    self.processed_coeff.time_dependent_averaged
                )
        else:
            self.processed_coeff.time_dependent_used = (
                self.processed_coeff.time_dependent_clustered
            )

        # CALCULATE BOUNDS
        self._calculate_bounds()

        # GENERAL TECHNOLOGY CONSTRAINTS
        b_tec = self._define_input_carriers(b_tec)
        b_tec = self._define_output_carriers(b_tec)
        b_tec = self._define_size(b_tec)
        b_tec = self._define_capex(b_tec, data)
        b_tec = self._define_input(b_tec, data)
        b_tec = self._define_output(b_tec, data)

        # CLUSTERED DATA
        if (config["optimization"]["typicaldays"]["N"]["value"] == 0) or (
            config["optimization"]["typicaldays"]["method"]["value"] == 1
        ):
            # input/output to calculate performance is the same as var_input
            if b_tec.find_component("var_input"):
                self.input = b_tec.var_input
            if b_tec.find_component("var_output"):
                self.output = b_tec.var_output
        elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
            if (
                self.component_options.technology_model
                in technologies_modelled_with_full_res
            ):
                # input/output to calculate performance is the same as var_input
                if b_tec.find_component("var_input"):
                    self.input = b_tec.var_input
                if b_tec.find_component("var_output"):
                    self.output = b_tec.var_output
            else:
                # input/output to calculate performance has lower resolution
                b_tec = self._define_auxiliary_vars(b_tec, data)
                if b_tec.find_component("var_input"):
                    self.input = b_tec.var_input_aux
                if b_tec.find_component("var_output"):
                    self.output = b_tec.var_output_aux

        b_tec = self._define_opex(b_tec)

        # CCS and Emissions
        if self.component_options.ccs_possible:
            log_event(f"\t - Adding CCS to Technology {self.name}")
            self._calculate_ccs_bounds()
            if self.component_options.modelled_with_full_res:
                self.ccs_component.processed_coeff.time_dependent_used = (
                    self.ccs_component.processed_coeff.time_dependent_full
                )
            else:
                self.ccs_component.processed_coeff.time_dependent_used = (
                    self.ccs_component.processed_coeff.time_dependent_clustered
                )
            b_tec = self._define_ccs_performance(b_tec, data)
            b_tec = self._define_ccs_emissions(b_tec)
            b_tec = self._define_ccs_costs(b_tec, data)
            log_event(
                f"\t - Adding CCS to Technology {self.name} completed", print_it=False
            )
        else:
            b_tec = self._define_emissions(b_tec)

        # DYNAMICS
        if config["performance"]["dynamics"]["value"]:
            technologies_modelled_with_dynamics = ["CONV1", "CONV2", "CONV3"]
            if (
                self.component_options.technology_model
                in technologies_modelled_with_dynamics
            ):
                b_tec = self._define_dynamics(b_tec, data)
            else:
                warn(
                    "Modeling dynamic constraints not enabled for technology type"
                    + self.name
                )
        else:
            if self.component_options.performance_function_type == 4:
                self.component_options.performance_function_type = 3
                warn(
                    "Switching dynamics off for performance function type 4, type changed to 3 for "
                    + self.name
                )

        # AGGREGATE ALL VARIABLES
        self._aggregate_input(b_tec)
        self._aggregate_output(b_tec)
        self._aggregate_cost(b_tec)

        return b_tec

    def _define_input_carriers(self, b_tec):
        """
        Defines the input carriers

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        b_tec.set_input_carriers = pyo.Set(
            initialize=self.component_options.input_carrier
        )

        if self.component_options.ccs_possible:
            b_tec.set_input_carriers_ccs = pyo.Set(
                initialize=self.ccs_component.component_options.input_carrier
            )
        else:
            b_tec.set_input_carriers_ccs = pyo.Set(initialize=[])

        b_tec.set_input_carriers_all = (
            b_tec.set_input_carriers_ccs | b_tec.set_input_carriers
        )

        return b_tec

    def _define_output_carriers(self, b_tec):
        """
        Defines the output carriers

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        b_tec.set_output_carriers = pyo.Set(
            initialize=self.component_options.output_carrier
        )

        if self.component_options.ccs_possible:
            b_tec.set_output_carriers_ccs = pyo.Set(
                initialize=self.ccs_component.component_options.output_carrier
            )
        else:
            b_tec.set_output_carriers_ccs = pyo.Set(initialize=[])

        b_tec.set_output_carriers_all = (
            b_tec.set_output_carriers_ccs | b_tec.set_output_carriers
        )

        return b_tec

    def _define_size(self, b_tec):
        """
        Defines variables and parameters related to technology size.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        coeff_ti = self.processed_coeff.time_independent

        if self.component_options.size_is_int:
            size_domain = pyo.NonNegativeIntegers
        else:
            size_domain = pyo.NonNegativeReals

        b_tec.para_size_min = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=coeff_ti["size_min"], mutable=True
        )
        b_tec.para_size_max = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=coeff_ti["size_max"], mutable=True
        )

        if self.existing:
            b_tec.para_size_initial = pyo.Param(
                within=size_domain, initialize=coeff_ti["size_initial"]
            )

        if self.existing and not self.component_options.decommission:
            # Decommissioning is not possible, size fixed
            b_tec.var_size = pyo.Param(
                within=size_domain, initialize=b_tec.para_size_initial
            )
        else:
            # Decommissioning is possible, size variable
            b_tec.var_size = pyo.Var(
                within=size_domain, bounds=(b_tec.para_size_min, b_tec.para_size_max)
            )

        return b_tec

    def _define_capex(self, b_tec, data: dict):
        """
        Defines variables and parameters related to technology capex.

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        config = data["config"]

        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        capex_model = set_capex_model(config, economics)

        def calculate_max_capex():
            if self.economics.capex_model == 1:
                max_capex = (
                    b_tec.para_size_max
                    * economics.capex_data["unit_capex"]
                    * annualization_factor
                )
                bounds = (0, max_capex)
            elif self.economics.capex_model == 2:
                max_capex = (
                    b_tec.para_size_max
                    * max(economics.capex_data["piecewise_capex"]["bp_y"])
                    * annualization_factor
                )
                bounds = (0, max_capex)
            elif self.economics.capex_model == 3:
                max_capex = (
                    b_tec.para_size_max * economics.capex_data["unit_capex"]
                    + economics.capex_data["fix_capex"]
                ) * annualization_factor
                bounds = (0, max_capex)
            else:
                bounds = None
            return bounds

        # CAPEX auxilliary (used to calculate theoretical CAPEX)
        # For new technologies, this is equal to actual CAPEX
        # For existing technologies it is used to calculate fixed OPEX
        b_tec.var_capex_aux = pyo.Var(bounds=calculate_max_capex())

        if capex_model == 1:
            b_tec.para_unit_capex = pyo.Param(
                domain=pyo.Reals,
                initialize=economics.capex_data["unit_capex"],
                mutable=True,
            )
            b_tec.para_unit_capex_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=annualization_factor * economics.capex_data["unit_capex"],
                mutable=True,
            )

            """capex_aux = size * capex_unit_annual"""
            b_tec.const_capex_aux = pyo.Constraint(
                expr=b_tec.var_size * b_tec.para_unit_capex_annual
                == b_tec.var_capex_aux
            )

        elif capex_model == 2:
            bp_x = economics.capex_data["piecewise_capex"]["bp_x"]
            bp_y_annual = [
                y * annualization_factor
                for y in economics.capex_data["piecewise_capex"]["bp_y"]
            ]
            self.big_m_transformation_required = 1
            b_tec.const_capex_aux = pyo.Piecewise(
                b_tec.var_capex_aux,
                b_tec.var_size,
                pw_pts=bp_x,
                pw_constr_type="EQ",
                f_rule=bp_y_annual,
                pw_repn="SOS2",
            )
        elif capex_model == 3:
            b_tec.para_unit_capex = pyo.Param(
                domain=pyo.Reals,
                initialize=economics.capex_data["unit_capex"],
                mutable=True,
            )
            b_tec.para_fix_capex = pyo.Param(
                domain=pyo.Reals,
                initialize=economics.capex_data["fix_capex"],
                mutable=True,
            )
            b_tec.para_unit_capex_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=annualization_factor * economics.capex_data["unit_capex"],
                mutable=True,
            )
            b_tec.para_fix_capex_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=annualization_factor * economics.capex_data["fix_capex"],
                mutable=True,
            )

            # capex unit commitment constraint
            self.big_m_transformation_required = 1
            s_indicators = range(0, 2)

            def init_installation(dis, ind):
                if ind == 0:  # tech not installed
                    dis.const_capex_aux = pyo.Constraint(expr=b_tec.var_capex_aux == 0)
                    dis.const_not_installed = pyo.Constraint(expr=b_tec.var_size == 0)
                else:  # tech installed
                    dis.const_capex_aux = pyo.Constraint(
                        expr=b_tec.var_size * b_tec.para_unit_capex_annual
                        + b_tec.para_fix_capex_annual
                        == b_tec.var_capex_aux
                    )

            b_tec.dis_installation = gdp.Disjunct(s_indicators, rule=init_installation)

            def bind_disjunctions(dis):
                return [b_tec.dis_installation[i] for i in s_indicators]

            b_tec.disjunction_installation = gdp.Disjunction(rule=bind_disjunctions)

        else:
            # Defined in the technology subclass
            pass

        # CAPEX
        if self.existing and not self.component_options.decommission:
            b_tec.var_capex = pyo.Param(domain=pyo.Reals, initialize=0)
        else:
            b_tec.var_capex = pyo.Var()
            if self.existing:
                b_tec.para_decommissioning_cost = pyo.Param(
                    domain=pyo.Reals,
                    initialize=economics.decommission_cost,
                    mutable=True,
                )
                b_tec.const_capex = pyo.Constraint(
                    expr=b_tec.var_capex
                    == (b_tec.para_size_initial - b_tec.var_size)
                    * b_tec.para_decommissioning_cost
                )
            else:
                b_tec.const_capex = pyo.Constraint(
                    expr=b_tec.var_capex == b_tec.var_capex_aux
                )

        return b_tec

    def _define_input(self, b_tec, data: dict):
        """
        Defines input to a technology

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        # Technology related data
        c = self.processed_coeff.time_independent

        def init_input_bounds(bounds, t, car):
            return tuple(
                self.bounds["input"][car][self.sequence[t - 1] - 1, :]
                * c["size_max"]
                * c["rated_power"]
            )

        b_tec.var_input = pyo.Var(
            self.set_t_global,
            b_tec.set_input_carriers,
            within=pyo.NonNegativeReals,
            bounds=init_input_bounds,
        )

        return b_tec

    def _define_output(self, b_tec, data: dict):
        """
        Defines output to a technology

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        # Technology related data
        c = self.processed_coeff.time_independent

        def init_output_bounds(bounds, t, car):
            return tuple(
                self.bounds["output"][car][self.sequence[t - 1] - 1, :]
                * c["size_max"]
                * c["rated_power"]
            )

        b_tec.var_output = pyo.Var(
            self.set_t_global,
            b_tec.set_output_carriers,
            within=pyo.NonNegativeReals,
            bounds=init_output_bounds,
        )
        return b_tec

    def _define_opex(self, b_tec):
        """
        Defines variable and fixed OPEX

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        economics = self.economics

        # VARIABLE OPEX
        b_tec.para_opex_variable = pyo.Param(
            domain=pyo.Reals, initialize=economics.opex_variable, mutable=True
        )
        b_tec.var_opex_variable = pyo.Var(self.set_t_global)

        def init_opex_variable(const, t):
            """opexvar_{t} = Input_{t, maincarrier} * opex_{var}"""
            if (
                (self.component_options.technology_model == "RES")
                or (self.component_options.technology_model == "CONV4")
                or (self.component_options.technology_model == "DAC_Adsorption")
            ):
                opex_variable_based_on = b_tec.var_output[
                    t, b_tec.set_output_carriers[1]
                ]
            else:
                opex_variable_based_on = b_tec.var_input[
                    t, self.component_options.main_input_carrier
                ]
            return (
                opex_variable_based_on * b_tec.para_opex_variable
                == b_tec.var_opex_variable[t]
            )

        b_tec.const_opex_variable = pyo.Constraint(
            self.set_t_global, rule=init_opex_variable
        )

        # FIXED OPEX
        b_tec.para_opex_fixed = pyo.Param(
            domain=pyo.Reals, initialize=economics.opex_fixed, mutable=True
        )
        b_tec.var_opex_fixed = pyo.Var()
        b_tec.const_opex_fixed = pyo.Constraint(
            expr=b_tec.var_capex_aux * b_tec.para_opex_fixed == b_tec.var_opex_fixed
        )
        return b_tec

    def _define_emissions(self, b_tec):
        """
        Defines Emissions

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        c = self.processed_coeff.time_independent
        technology_model = self.component_options.technology_model
        emissions_based_on = self.component_options.emissions_based_on

        b_tec.para_tec_emissionfactor = pyo.Param(
            domain=pyo.Reals, initialize=c["emission_factor"]
        )
        b_tec.var_tec_emissions_pos = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )
        b_tec.var_tec_emissions_neg = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )

        if technology_model == "RES":
            # Set emissions to zero
            def init_tec_emissions_pos(const, t):
                return b_tec.var_tec_emissions_pos[t] == 0

            b_tec.const_tec_emissions_pos = pyo.Constraint(
                self.set_t_global, rule=init_tec_emissions_pos
            )

            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = pyo.Constraint(
                self.set_t_global, rule=init_tec_emissions_neg
            )

        else:

            if emissions_based_on == "output":

                def init_tec_emissions_pos(const, t):
                    """emissions_pos = output * emissionfactor"""
                    if c["emission_factor"] >= 0:
                        return (
                            b_tec.var_output[
                                t, self.component_options.main_output_carrier
                            ]
                            * b_tec.para_tec_emissionfactor
                            == b_tec.var_tec_emissions_pos[t]
                        )
                    else:
                        return b_tec.var_tec_emissions_pos[t] == 0

                b_tec.const_tec_emissions_pos = pyo.Constraint(
                    self.set_t_global, rule=init_tec_emissions_pos
                )

                def init_tec_emissions_neg(const, t):
                    if c["emission_factor"] < 0:
                        return (
                            b_tec.var_output[
                                t, self.component_options.main_output_carrier
                            ]
                            * (-b_tec.para_tec_emissionfactor)
                            == b_tec.var_tec_emissions_neg[t]
                        )
                    else:
                        return b_tec.var_tec_emissions_neg[t] == 0

                b_tec.const_tec_emissions_neg = pyo.Constraint(
                    self.set_t_global, rule=init_tec_emissions_neg
                )

            elif emissions_based_on == "input":

                def init_tec_emissions_pos(const, t):
                    if c["emission_factor"] >= 0:
                        return (
                            b_tec.var_input[
                                t, self.component_options.main_input_carrier
                            ]
                            * b_tec.para_tec_emissionfactor
                            == b_tec.var_tec_emissions_pos[t]
                        )
                    else:
                        return b_tec.var_tec_emissions_pos[t] == 0

                b_tec.const_tec_emissions_pos = pyo.Constraint(
                    self.set_t_global, rule=init_tec_emissions_pos
                )

                def init_tec_emissions_neg(const, t):
                    if c["emission_factor"] < 0:
                        return (
                            b_tec.var_input[
                                t, self.component_options.main_input_carrier
                            ](-b_tec.para_tec_emissionfactor)
                            == b_tec.var_tec_emissions_neg[t]
                        )
                    else:
                        return b_tec.var_tec_emissions_neg[t] == 0

                b_tec.const_tec_emissions_neg = pyo.Constraint(
                    self.set_t_global, rule=init_tec_emissions_neg
                )

        return b_tec

    def _define_auxiliary_vars(self, b_tec, data: dict):
        """
        Defines auxiliary variables, that are required for the modelling of clustered data

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        c = self.processed_coeff.time_independent

        if not (self.component_options.technology_model == "RES") and not (
            self.component_options.technology_model == "CONV4"
        ):

            def init_input_bounds(bounds, t, car):
                return tuple(
                    self.bounds["input"][car][t - 1, :]
                    * c["size_max"]
                    * c["rated_power"]
                )

            b_tec.var_input_aux = pyo.Var(
                self.set_t_performance,
                b_tec.set_input_carriers,
                within=pyo.NonNegativeReals,
                bounds=init_input_bounds,
            )

            b_tec.const_link_full_resolution_input = link_full_resolution_to_clustered(
                b_tec.var_input_aux,
                b_tec.var_input,
                self.set_t_full,
                self.sequence,
                b_tec.set_input_carriers,
            )

        def init_output_bounds(bounds, t, car):
            return tuple(
                self.bounds["output"][car][t - 1, :] * c["size_max"] * c["rated_power"]
            )

        b_tec.var_output_aux = pyo.Var(
            self.set_t_performance,
            b_tec.set_output_carriers,
            within=pyo.NonNegativeReals,
            bounds=init_output_bounds,
        )

        b_tec.const_link_full_resolution_output = link_full_resolution_to_clustered(
            b_tec.var_output_aux,
            b_tec.var_output,
            self.set_t_full,
            self.sequence,
            b_tec.set_output_carriers,
        )

        return b_tec

    def _aggregate_input(self, b_tec):
        """
        Aggregates ccs and technology input

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """

        b_tec.var_input_tot = pyo.Var(
            self.set_t_global,
            b_tec.set_input_carriers_all,
            within=pyo.NonNegativeReals,
        )

        def init_aggregate_input(const, t, car):
            """input_ccs + input = input_tot"""
            input_tec = (
                b_tec.var_input[t, car] if car in b_tec.set_input_carriers else 0
            )
            if self.component_options.ccs_possible:
                input_ccs = (
                    b_tec.var_input_ccs[t, car]
                    if car in b_tec.set_input_carriers_ccs
                    else 0
                )
            else:
                input_ccs = 0
            return input_tec + input_ccs == b_tec.var_input_tot[t, car]

        b_tec.const_input_aggregation = pyo.Constraint(
            self.set_t_global, b_tec.set_input_carriers_all, rule=init_aggregate_input
        )

        return b_tec

    def _aggregate_output(self, b_tec):
        """
        Aggregates ccs and technology output

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        b_tec.var_output_tot = pyo.Var(
            self.set_t_global,
            b_tec.set_output_carriers_all,
            within=pyo.NonNegativeReals,
        )

        def init_aggregate_output(const, t, car):
            """output + output_ccs = output_tot"""
            output_tec = (
                b_tec.var_output[t, car] if car in b_tec.set_output_carriers else 0
            )
            if self.component_options.ccs_possible:
                output_ccs = (
                    b_tec.var_output_ccs[t, car]
                    if car in b_tec.set_output_carriers_ccs
                    else 0
                )
            else:
                output_ccs = 0
            return output_tec + output_ccs == b_tec.var_output_tot[t, car]

        b_tec.const_output_aggregation = pyo.Constraint(
            self.set_t_global, b_tec.set_output_carriers_all, rule=init_aggregate_output
        )

        return b_tec

    def _aggregate_cost(self, b_tec):
        """
        Aggregates ccs and technology cost

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        b_tec.var_capex_tot = pyo.Var()
        b_tec.var_opex_fixed_tot = pyo.Var()
        b_tec.var_opex_variable_tot = pyo.Var(self.set_t_global)

        def init_aggregate_capex(const):
            """capex + capex_ccs = capex_tot"""
            capex_tec = b_tec.var_capex
            if self.component_options.ccs_possible:
                capex_ccs = b_tec.var_capex_ccs
            else:
                capex_ccs = 0
            return b_tec.var_capex_tot == capex_tec + capex_ccs

        b_tec.const_capex_aggregation = pyo.Constraint(rule=init_aggregate_capex)

        def init_aggregate_opex_var(const, t):
            """var_opex_variable + var_opex_variable_ccs = var_opex_variable_tot"""
            opex_var_tec = b_tec.var_opex_variable[t]
            if self.component_options.ccs_possible:
                opex_var_ccs = b_tec.var_opex_variable_ccs[t]
            else:
                opex_var_ccs = 0
            return b_tec.var_opex_variable_tot[t] == opex_var_tec + opex_var_ccs

        b_tec.const_opex_var_aggregation = pyo.Constraint(
            self.set_t_global, rule=init_aggregate_opex_var
        )

        def init_aggregate_opex_fixed(const):
            """var_opex_fixed + var_opex_fixed_ccs = var_opex_fixed_tot"""
            opex_fixed_tec = b_tec.var_opex_fixed
            if self.component_options.ccs_possible:
                opex_fixed_ccs = b_tec.var_opex_fixed_ccs
            else:
                opex_fixed_ccs = 0
            return b_tec.var_opex_fixed_tot == opex_fixed_tec + opex_fixed_ccs

        b_tec.const_opex_fixed_aggregation = pyo.Constraint(
            rule=init_aggregate_opex_fixed
        )

        return b_tec

    def write_results_tec_design(self, h5_group, model_block):
        """
        Function to report technology design

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        h5_group.create_dataset("technology", data=[self.name])
        h5_group.create_dataset("size", data=[model_block.var_size.value])
        h5_group.create_dataset("existing", data=[self.existing])
        h5_group.create_dataset("capex_tot", data=[model_block.var_capex_tot.value])
        h5_group.create_dataset(
            "opex_variable",
            data=[
                sum(model_block.var_opex_variable[t].value for t in self.set_t_global)
            ],
        )
        h5_group.create_dataset(
            "opex_fixed_tot", data=[model_block.var_opex_fixed_tot.value]
        )
        h5_group.create_dataset(
            "emissions_pos",
            data=[
                sum(
                    model_block.var_tec_emissions_pos[t].value
                    for t in self.set_t_global
                )
            ],
        )
        h5_group.create_dataset(
            "emissions_neg",
            data=[
                sum(
                    model_block.var_tec_emissions_neg[t].value
                    for t in self.set_t_global
                )
            ],
        )
        if self.component_options.ccs_possible:
            h5_group.create_dataset("size_ccs", data=[model_block.var_size_ccs.value])
            h5_group.create_dataset("capex_tec", data=[model_block.var_capex.value])
            h5_group.create_dataset("capex_ccs", data=[model_block.var_capex_ccs.value])
            h5_group.create_dataset(
                "opex_fixed_ccs", data=[model_block.var_opex_fixed_ccs.value]
            )

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report technology operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        for car in model_block.set_input_carriers_all:
            if model_block.find_component("var_input"):
                h5_group.create_dataset(
                    f"{car}_input",
                    data=[
                        model_block.var_input_tot[t, car].value
                        for t in self.set_t_global
                    ],
                )
        for car in model_block.set_output_carriers_all:
            h5_group.create_dataset(
                f"{car}_output",
                data=[
                    model_block.var_output_tot[t, car].value for t in self.set_t_global
                ],
            )
        h5_group.create_dataset(
            "emissions_pos",
            data=[
                model_block.var_tec_emissions_pos[t].value for t in self.set_t_global
            ],
        )
        h5_group.create_dataset(
            "emissions_neg",
            data=[
                model_block.var_tec_emissions_neg[t].value for t in self.set_t_global
            ],
        )
        if model_block.find_component("var_x"):
            h5_group.create_dataset(
                "var_x",
                data=[
                    0 if x is None else x
                    for x in [
                        model_block.var_x[t].value for t in self.set_t_performance
                    ]
                ],
            )
        if model_block.find_component("var_y"):
            h5_group.create_dataset(
                "var_y",
                data=[
                    0 if x is None else x
                    for x in [
                        model_block.var_y[t].value for t in self.set_t_performance
                    ]
                ],
            )
        if model_block.find_component("var_z"):
            h5_group.create_dataset(
                "var_z",
                data=[
                    0 if x is None else x
                    for x in [
                        model_block.var_z[t].value for t in self.set_t_performance
                    ]
                ],
            )

        if model_block.find_component("set_input_carriers_ccs"):
            for car in model_block.set_input_carriers_ccs:
                h5_group.create_dataset(
                    f"{car}_var_input_ccs",
                    data=[
                        model_block.var_input_ccs[t, car].value
                        for t in self.set_t_performance
                    ],
                )
            for car in model_block.set_output_carriers_ccs:
                h5_group.create_dataset(
                    f"{car}_var_output_ccs",
                    data=[
                        model_block.var_output_ccs[t, car].value
                        for t in self.set_t_performance
                    ],
                )

    def scale_model(self, b_tec, model, config):
        """
        Scales technology model

        :param b_tec: pyomo network block
        :param model: pyomo model
        :param dict config: config dict containing scaling factors
        :return: pyomo model
        """

        f = self.scaling_factors
        f_global = config["scaling_factors"]

        model = determine_variable_scaling(model, b_tec, f, f_global)
        model = determine_constraint_scaling(model, b_tec, f, f_global)

        return model

    # CCS FUNCTIONS
    def _define_ccs_performance(self, b_tec, data: dict):
        """
        Defines CCS performance. The unit capex parameter is calculated from Eq. 10 of Weimann et al. 2023

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        coeff_ti = self.ccs_component.processed_coeff.time_independent

        emissions_based_on = self.component_options.emissions_based_on
        capture_rate = coeff_ti["capture_rate"]

        # LOG
        log_event(f"\t - Adding CCS to Technology {self.name}")

        # TODO: maybe make the full set of all carriers as a intersection between this set and the others?
        # Emission Factor
        b_tec.para_tec_emissionfactor = pyo.Param(
            domain=pyo.Reals,
            initialize=self.processed_coeff.time_independent["emission_factor"],
        )
        b_tec.var_tec_emissions_pos = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )
        b_tec.var_tec_emissions_neg = pyo.Var(
            self.set_t_global, within=pyo.NonNegativeReals
        )

        def init_input_bounds(bounds, t, car):
            return tuple(
                self.ccs_component.bounds["input"][car][self.sequence[t - 1] - 1, :]
                * coeff_ti["size_max"]
            )

        b_tec.var_input_ccs = pyo.Var(
            self.set_t_global,
            b_tec.set_input_carriers_ccs,
            within=pyo.NonNegativeReals,
            bounds=init_input_bounds,
        )

        def init_output_bounds(bounds, t, car):
            return tuple(
                self.ccs_component.bounds["output"][car][self.sequence[t - 1] - 1, :]
                * coeff_ti["size_max"]
            )

        b_tec.var_output_ccs = pyo.Var(
            self.set_t_global,
            b_tec.set_output_carriers_ccs,
            within=pyo.NonNegativeReals,
            bounds=init_output_bounds,
        )

        # Input-output correlation
        def init_input_output_ccs(const, t):
            if emissions_based_on == "output":
                return (
                    b_tec.var_output_ccs[t, "CO2captured"]
                    <= capture_rate
                    * b_tec.para_tec_emissionfactor
                    * b_tec.var_output[t, self.component_options.main_output_carrier]
                )
            else:
                return (
                    b_tec.var_output_ccs[t, "CO2captured"]
                    <= capture_rate
                    * b_tec.para_tec_emissionfactor
                    * b_tec.var_input[t, self.component_options.main_input_carrier]
                )

        b_tec.const_input_output_ccs = pyo.Constraint(
            self.set_t_global, rule=init_input_output_ccs
        )

        # Electricity and heat demand CCS
        def init_input_ccs(const, t, car):
            return (
                b_tec.var_input_ccs[t, car]
                == coeff_ti["input_ratios"][car]
                * b_tec.var_output_ccs[t, "CO2captured"]
                / capture_rate
            )

        b_tec.const_input_el = pyo.Constraint(
            self.set_t_global, b_tec.set_input_carriers_ccs, rule=init_input_ccs
        )

        return b_tec

    def _define_ccs_emissions(self, b_tec):
        """
        Defines CCS performance. The unit capex parameter is calculated from Eq. 10 of Weimann et al. 2023

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        emissions_based_on = self.component_options.emissions_based_on

        # Emissions
        if emissions_based_on == "output":

            def init_tec_emissions_pos(const, t):
                return (
                    b_tec.var_output[t, self.component_options.main_output_carrier]
                    * b_tec.para_tec_emissionfactor
                    - b_tec.var_output_ccs[t, "CO2captured"]
                    == b_tec.var_tec_emissions_pos[t]
                )

            b_tec.const_tec_emissions_pos = pyo.Constraint(
                self.set_t_global, rule=init_tec_emissions_pos
            )

            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = pyo.Constraint(
                self.set_t_global, rule=init_tec_emissions_neg
            )

        elif emissions_based_on == "input":

            def init_tec_emissions_pos(const, t):
                return (
                    b_tec.var_input[t, self.component_options.main_input_carrier]
                    * b_tec.para_tec_emissionfactor
                    - b_tec.var_output_ccs[t, "CO2captured"]
                    == b_tec.var_tec_emissions_pos[t]
                )

            b_tec.const_tec_emissions_pos = pyo.Constraint(
                self.set_t_global, rule=init_tec_emissions_pos
            )

            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = pyo.Constraint(
                self.set_t_global, rule=init_tec_emissions_neg
            )

        # Initialize the size of CCS as in _define_size (size given in mass flow of CO2 entering the CCS object)
        b_tec.para_size_min_ccs = pyo.Param(
            domain=pyo.NonNegativeReals,
            initialize=self.ccs_component.input_parameters.size_min,
            mutable=True,
        )
        b_tec.para_size_max_ccs = pyo.Param(
            domain=pyo.NonNegativeReals,
            initialize=self.ccs_component.input_parameters.size_max,
            mutable=True,
        )

        # Decommissioning is possible, size variable
        b_tec.var_size_ccs = pyo.Var(
            within=pyo.NonNegativeReals,
            bounds=(0, b_tec.para_size_max_ccs),
        )

        return b_tec

    def _define_ccs_costs(self, b_tec, data: dict):
        """
        Defines ccs costs

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        config = data["config"]

        # Costs
        economics = self.ccs_component.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        b_tec.para_unit_capex_ccs = pyo.Param(
            domain=pyo.Reals,
            initialize=economics.capex_data["unit_capex"],
            mutable=True,
        )
        b_tec.para_unit_capex_annual_ccs = pyo.Param(
            domain=pyo.Reals,
            initialize=annualization_factor * economics.capex_data["unit_capex"],
            mutable=True,
        )

        b_tec.para_fix_capex_annual_ccs = pyo.Param(
            domain=pyo.Reals,
            initialize=annualization_factor * economics.capex_data["fix_capex"],
            mutable=True,
        )

        def calculate_max_capex_ccs():
            max_capex = (
                self.ccs_component.input_parameters.size_max
                * b_tec.para_unit_capex_annual_ccs
                + b_tec.para_fix_capex_annual_ccs
            )
            return (0, max_capex)

        b_tec.var_capex_aux_ccs = pyo.Var(bounds=calculate_max_capex_ccs())

        # capex unit commitment constraint
        self.big_m_transformation_required = 1
        s_indicators = range(0, 2)

        def init_installation(dis, ind):
            if ind == 0:  # tech not installed
                dis.const_capex_aux_ccs = pyo.Constraint(
                    expr=b_tec.var_capex_aux_ccs == 0
                )
                dis.const_not_installed_ccs = pyo.Constraint(
                    expr=b_tec.var_size_ccs == 0
                )
            else:  # tech installed
                dis.const_capex_aux_ccs = pyo.Constraint(
                    expr=b_tec.var_size_ccs * b_tec.para_unit_capex_annual_ccs
                    + b_tec.para_fix_capex_annual_ccs
                    == b_tec.var_capex_aux_ccs
                )
                dis.const_installed_ccs_sizelim_min = pyo.Constraint(
                    expr=b_tec.var_size_ccs >= b_tec.para_size_min_ccs
                )
                dis.const_installed_ccs_sizelim_max = pyo.Constraint(
                    expr=b_tec.var_size_ccs <= b_tec.para_size_max_ccs
                )

        b_tec.dis_installation_ccs = gdp.Disjunct(s_indicators, rule=init_installation)

        def bind_disjunctions(dis):
            return [b_tec.dis_installation_ccs[i] for i in s_indicators]

        b_tec.disjunction_installation_ccs = gdp.Disjunction(rule=bind_disjunctions)

        # CAPEX
        b_tec.var_capex_ccs = pyo.Var()
        b_tec.const_capex_ccs = pyo.Constraint(
            expr=b_tec.var_capex_ccs == b_tec.var_capex_aux_ccs
        )

        # FIXED OPEX
        b_tec.para_opex_fixed_ccs = pyo.Param(
            domain=pyo.Reals, initialize=economics.opex_fixed, mutable=True
        )
        b_tec.var_opex_fixed_ccs = pyo.Var()
        b_tec.const_opex_fixed_ccs = pyo.Constraint(
            expr=b_tec.var_capex_aux_ccs * b_tec.para_opex_fixed_ccs
            == b_tec.var_opex_fixed_ccs
        )

        # VARIABLE OPEX
        b_tec.para_opex_variable_ccs = pyo.Param(
            domain=pyo.Reals, initialize=economics.opex_variable, mutable=True
        )
        b_tec.var_opex_variable_ccs = pyo.Var(self.set_t_global)

        def init_opex_variable_ccs(const, t):
            return (
                b_tec.var_output_ccs[t, b_tec.set_output_carriers_ccs[1]]
                * b_tec.para_opex_variable_ccs
                == b_tec.var_opex_variable_ccs[t]
            )

        b_tec.const_opex_variable_ccs = pyo.Constraint(
            self.set_t_global, rule=init_opex_variable_ccs
        )

        return b_tec

    # DYNAMICS FUNCTIONS
    def _define_dynamics(self, b_tec, data: dict):
        """
        Selects the dynamic constraints that are required based on the technology dynamic performance parameters or the
        performance function type.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        config = data["config"]

        log_event(f"\t \t Adding dynamics to Technology {self.name}")
        if config["optimization"]["typicaldays"]["N"]["value"] != 0:
            raise Exception("time aggregation with dynamics is not implemented")

        dynamics = self.processed_coeff.dynamics
        SU_load = dynamics["SU_load"]
        SD_load = dynamics["SD_load"]
        min_uptime = dynamics["min_uptime"]
        min_downtime = dynamics["min_downtime"]
        max_startups = dynamics["max_startups"]

        if (
            (min_uptime + min_downtime > -2)
            or (max_startups > -1)
            or (SU_load + SD_load > -2)
            or self.component_options.performance_function_type == 4
        ):
            b_tec = self._dynamics_SUSD_logic(b_tec)
        if not (self.component_options.performance_function_type == 4) and (
            SU_load + SD_load > -2
        ):
            b_tec = self._dynamics_fast_SUSD(b_tec)

        log_event(f"\t \t Adding dynamics to Technology {self.name}", print_it=False)

        return b_tec

    def _dynamics_SUSD_logic(self, b_tec):
        """
        Adds the startup and shutdown logic to the technology model and constrains the maximum number of startups.

        Based on Equations 4-5 in Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden power system
        inflexibilities imposed by traditional unit commitment formulations. Applied Energy, 191, 223–238.
        https://doi.org/10.1016/J.APENERGY.2017.01.089

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        dynamics = self.processed_coeff.dynamics

        # New variables
        b_tec.var_x = pyo.Var(
            self.set_t_performance, domain=pyo.NonNegativeReals, bounds=(0, 1)
        )
        b_tec.var_y = pyo.Var(
            self.set_t_performance, domain=pyo.NonNegativeReals, bounds=(0, 1)
        )
        b_tec.var_z = pyo.Var(
            self.set_t_performance, domain=pyo.NonNegativeReals, bounds=(0, 1)
        )

        # Check for default values
        para_names = ["SU_time", "SD_time"]
        for para in para_names:
            if dynamics[para] < 0:
                dynamics[para] = 0
                warn(
                    "Using SU/SD logic constraints, parameter "
                    + str(para)
                    + " set to default value 0"
                )
        para_names = ["min_uptime", "min_downtime"]
        for para in para_names:
            if dynamics[para] < 0:
                dynamics[para] = 1
                warn(
                    "Using SU/SD logic constraints, parameter "
                    + str(para)
                    + " set to default value 1"
                )

        # Collect parameters
        SU_time = dynamics["SU_time"]
        SD_time = dynamics["SD_time"]
        min_uptime = dynamics["min_uptime"]
        min_downtime = dynamics["min_downtime"] + SU_time + SD_time
        max_startups = dynamics["max_startups"]

        # Enforce startup/shutdown logic
        def init_SUSD_logic1(const, t):
            if t == 1:
                return pyo.Constraint.Skip
            else:
                return (
                    b_tec.var_x[t] - b_tec.var_x[t - 1]
                    == b_tec.var_y[t] - b_tec.var_z[t]
                )

        b_tec.const_SUSD_logic1 = pyo.Constraint(
            self.set_t_performance, rule=init_SUSD_logic1
        )

        def init_SUSD_logic2(const, t):
            if t >= min_uptime:
                return b_tec.var_y[t - min_uptime + 1] <= b_tec.var_x[t]
            else:
                return (
                    b_tec.var_y[len(self.set_t_performance) + (t - min_uptime + 1)]
                    <= b_tec.var_x[t]
                )

        b_tec.const_SUSD_logic2 = pyo.Constraint(
            self.set_t_performance, rule=init_SUSD_logic2
        )

        def init_SUSD_logic3(const, t):
            if t >= min_downtime:
                return b_tec.var_z[t - min_downtime + 1] <= 1 - b_tec.var_x[t]
            else:
                return (
                    b_tec.var_z[len(self.set_t_performance) + (t - min_downtime + 1)]
                    <= 1 - b_tec.var_x[t]
                )

        b_tec.const_SUSD_logic3 = pyo.Constraint(
            self.set_t_performance, rule=init_SUSD_logic3
        )

        # Constrain number of startups
        if not max_startups == -1:

            def init_max_startups(const):
                return (
                    sum(b_tec.var_y[t] for t in self.set_t_performance) <= max_startups
                )

            b_tec.const_max_startups = pyo.Constraint(rule=init_max_startups)

        return b_tec

    def _dynamics_fast_SUSD(self, b_tec):
        """
        Adds startup and shutdown load constraints to the model.

        Based on Equations 9-11 and 13 in Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden power
        system inflexibilities imposed by traditional unit commitment formulations. Applied Energy, 191, 223–238.
        https://doi.org/10.1016/J.APENERGY.2017.01.089

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        dynamics = self.processed_coeff.dynamics

        # Check for default values
        para_names = ["SU_load", "SD_load"]
        for para in para_names:
            if dynamics[para] < 0:
                dynamics[para] = 1
                warn(
                    "Using SU/SD load constraints, parameter"
                    + str(para)
                    + "set to default value 1"
                )

        # Collect parameters
        SU_load = dynamics["SU_load"]
        SD_load = dynamics["SD_load"]
        main_car = self.component_options.main_input_carrier
        rated_power = self.input_parameters.rated_power

        # SU load limit
        s_indicators = range(0, 2)

        def init_SU_load(dis, t, ind):
            if ind == 0:  # no startup (y=0)
                dis.const_y_off = pyo.Constraint(expr=b_tec.var_y[t] == 0)

            else:  # tech in startup
                dis.const_y_on = pyo.Constraint(expr=b_tec.var_y[t] == 1)

                def init_SU_load_limit(cons):
                    if self.component_options.technology_model == "CONV3":
                        return (
                            self.input[t, main_car]
                            <= b_tec.var_size * SU_load * rated_power
                        )
                    else:
                        return (
                            sum(
                                self.input[t, car_input]
                                for car_input in b_tec.set_input_carriers
                            )
                            <= b_tec.var_size * SU_load * rated_power
                        )

                dis.const_SU_load_limit = pyo.Constraint(rule=init_SU_load_limit)

        b_tec.dis_SU_load = gdp.Disjunct(
            self.set_t_performance, s_indicators, rule=init_SU_load
        )

        def bind_disjunctions_SU_load(dis, t):
            return [b_tec.dis_SU_load[t, i] for i in s_indicators]

        b_tec.disjunction_SU_load = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions_SU_load
        )

        # SD load limit
        s_indicators = range(0, 2)

        def init_SD_load(dis, t, ind):
            if ind == 0:  # no shutdown (z=0)
                dis.const_z_off = pyo.Constraint(expr=b_tec.var_z[t] == 0)

            else:  # tech in shutdown
                dis.const_z_on = pyo.Constraint(expr=b_tec.var_z[t] == 1)

                def init_SD_load_limit(cons):
                    if t == 1:
                        return pyo.Constraint.Skip
                    else:
                        if self.component_options.technology_model == "CONV3":
                            return (
                                self.input[t - 1, main_car]
                                <= b_tec.var_size * SD_load * rated_power
                            )
                        else:
                            return (
                                sum(
                                    self.input[t - 1, car_input]
                                    for car_input in b_tec.set_input_carriers
                                )
                                <= b_tec.var_size * SD_load * rated_power
                            )

                dis.const_SD_load_limit = pyo.Constraint(rule=init_SD_load_limit)

        b_tec.dis_SD_load = gdp.Disjunct(
            self.set_t_performance, s_indicators, rule=init_SD_load
        )

        def bind_disjunctions_SD_load(dis, t):
            return [b_tec.dis_SD_load[t, i] for i in s_indicators]

        b_tec.disjunction_SD_load = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions_SD_load
        )

        return b_tec
