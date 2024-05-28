import pyomo.gdp as gdp
import pyomo.environ as pyo
from warnings import warn
from types import SimpleNamespace

from ..component import ModelComponent
from ..utilities import (
    annualize,
    set_discount_rate,
    link_full_resolution_to_clustered,
    determine_variable_scaling,
    determine_constraint_scaling,
)
from .utilities import set_capex_model
from ...logger import log_event


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

        self.technology_model = tec_data["tec_type"]
        self.modelled_with_full_res = []

        # Technology Performance
        self.performance_data = tec_data["TechnologyPerf"]

        if "ccs" in self.performance_data and self.performance_data["ccs"]["possible"]:
            self.ccs_data = None
            self.ccs = 1
        else:
            self.ccs = 0

        # Input/output are based on
        if self.technology_model == "CONV1":
            self.performance_data["size_based_on"] = tec_data["size_based_on"]
        else:
            self.performance_data["size_based_on"] = "input"

        # Emissions are based on
        if (self.technology_model == "DAC_Adsorption") or (
            self.technology_model == "CONV4"
        ):
            self.emissions_based_on = "output"
        else:
            self.emissions_based_on = "input"

        self.fitted_performance = None

        # Other attributes
        self.input = []
        self.output = []
        self.set_t = []
        self.set_t_full = []
        self.sequence = []

        # Scaling factors
        self.scaling_factors = []
        if "ScalingFactors" in tec_data:
            self.scaling_factors = tec_data["ScalingFactors"]

    def construct_tech_model(self, b_tec, data: dict, set_t, set_t_clustered):
        """
        Construct the technology model with all required parameters, variable, sets,...

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        # LOG
        log_event(f"\t - Adding Technology {self.name}")

        # TECHNOLOGY DATA
        config = data["config"]

        # SET CAPEX MODEL
        self.economics.capex_model = set_capex_model(config, self.economics)

        # MODELING TYPICAL DAYS
        self.set_t_full = set_t
        if config["optimization"]["typicaldays"]["N"]["value"] != 0:
            if config["optimization"]["typicaldays"]["method"]["value"] == 2:
                technologies_modelled_with_full_res = ["RES", "STOR", "Hydro_Open"]
                if self.technology_model in technologies_modelled_with_full_res:
                    self.modelled_with_full_res = 1
                else:
                    self.modelled_with_full_res = 0
            else:
                raise KeyError(
                    "The clustering method specified in the configuration file does not exist."
                )
        else:
            self.modelled_with_full_res = 1

        # GENERAL TECHNOLOGY CONSTRAINTS
        b_tec = self._define_input_carriers(b_tec)
        b_tec = self._define_output_carriers(b_tec)
        b_tec = self._define_size(b_tec)
        b_tec = self._define_capex_parameters(b_tec, data)
        b_tec = self._define_capex_variables(b_tec, data)
        b_tec = self._define_capex_constraints(b_tec, data)
        b_tec = self._define_input(b_tec, data)
        b_tec = self._define_output(b_tec, data)
        b_tec = self._define_opex(b_tec)

        # CCS and Emissions
        if self.ccs:
            log_event(f"\t - Adding CCS to Technology {self.name}")
            b_tec = self._define_ccs_performance(b_tec, data)
            b_tec = self._define_ccs_emissions(b_tec)
            b_tec = self._define_ccs_costs(b_tec, data)
            log_event(
                f"\t - Adding CCS to Technology {self.name} completed", print_it=False
            )
        else:
            b_tec = self._define_emissions(b_tec)

        # CLUSTERED DATA
        if (
            config["optimization"]["typicaldays"]["N"]["value"] != 0
            and not self.modelled_with_full_res
        ):
            b_tec = self._define_auxiliary_vars(b_tec, data)
        else:
            if not (self.technology_model == "RES") and not (
                self.technology_model == "CONV4"
            ):
                self.input = b_tec.var_input
            self.output = b_tec.var_output
            self.set_t = set_t
            self.sequence = list(self.set_t)

        # DYNAMICS
        if config["performance"]["dynamics"]["value"]:
            technologies_modelled_with_dynamics = ["CONV1", "CONV2", "CONV3"]
            if self.technology_model in technologies_modelled_with_dynamics:
                b_tec = self._define_dynamics(b_tec)
            else:
                warn(
                    "Modeling dynamic constraints not enabled for technology type"
                    + self.name
                )
        else:
            if hasattr(self.performance_data, "performance_function_type"):
                if self.performance_data.performance_function_type == 4:
                    self.performance_data.performance_function_type = 3
                    warn(
                        "Switching dynamics off for performance function type 4, type changed to 3 for "
                        + self.name
                    )

        # AGGREGATE ALL VARIABLES
        self._aggregate_input(b_tec)
        self._aggregate_output(b_tec)
        self._aggregate_cost(b_tec)

        return b_tec

    def _aggregate_input(self, b_tec):

        b_tec.var_input_tot = pyo.Var(
            self.set_t,
            b_tec.set_input_carriers_all,
            within=pyo.NonNegativeReals,
        )

        def init_aggregate_input(const, t, car):
            input_tec = (
                b_tec.var_input[t, car] if car in b_tec.set_input_carriers else 0
            )
            if self.ccs:
                input_ccs = (
                    b_tec.var_input_ccs[t, car]
                    if car in b_tec.set_input_carriers_ccs
                    else 0
                )
            else:
                input_ccs = 0
            return input_tec + input_ccs == b_tec.var_input_tot[t, car]

        b_tec.const_input_aggregation = pyo.Constraint(
            self.set_t, b_tec.set_input_carriers_all, rule=init_aggregate_input
        )

        return b_tec

    def _aggregate_output(self, b_tec):

        b_tec.var_output_tot = pyo.Var(
            self.set_t,
            b_tec.set_output_carriers_all,
            within=pyo.NonNegativeReals,
        )

        def init_aggregate_output(const, t, car):
            output_tec = (
                b_tec.var_output[t, car] if car in b_tec.set_output_carriers else 0
            )
            if self.ccs:
                output_ccs = (
                    b_tec.var_output_ccs[t, car]
                    if car in b_tec.set_output_carriers_ccs
                    else 0
                )
            else:
                output_ccs = 0
            return output_tec + output_ccs == b_tec.var_output_tot[t, car]

        b_tec.const_output_aggregation = pyo.Constraint(
            self.set_t, b_tec.set_output_carriers_all, rule=init_aggregate_output
        )

        return b_tec

    def _aggregate_cost(self, b_tec):

        set_t = self.set_t_full

        b_tec.var_capex_tot = pyo.Var()
        b_tec.var_opex_fixed_tot = pyo.Var()
        b_tec.var_opex_variable_tot = pyo.Var(set_t)

        def init_aggregate_capex(const):
            capex_tec = b_tec.var_capex
            if self.ccs:
                capex_ccs = b_tec.var_capex_ccs
            else:
                capex_ccs = 0
            return b_tec.var_capex_tot == capex_tec + capex_ccs

        b_tec.const_capex_aggregation = pyo.Constraint(rule=init_aggregate_capex)

        def init_aggregate_opex_var(const, t):
            opex_var_tec = b_tec.var_opex_variable[t]
            if self.ccs:
                opex_var_ccs = b_tec.var_opex_variable_ccs[t]
            else:
                opex_var_ccs = 0
            return b_tec.var_opex_variable_tot[t] == opex_var_tec + opex_var_ccs

        b_tec.const_opex_var_aggregation = pyo.Constraint(
            set_t, rule=init_aggregate_opex_var
        )

        def init_aggregate_opex_fixed(const):
            opex_fixed_tec = b_tec.var_opex_fixed
            if self.ccs:
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
        Function to report results of technologies after optimization

        :param model_block: technology model block
        :return: dict results: holds results
        """

        h5_group.create_dataset("technology", data=[self.name])
        h5_group.create_dataset("size", data=[model_block.var_size.value])
        h5_group.create_dataset("existing", data=[self.existing])
        h5_group.create_dataset("capex_tot", data=[model_block.var_capex_tot.value])
        h5_group.create_dataset(
            "opex_variable",
            data=[sum(model_block.var_opex_variable[t].value for t in self.set_t_full)],
        )
        h5_group.create_dataset(
            "opex_fixed_tot", data=[model_block.var_opex_fixed_tot.value]
        )
        h5_group.create_dataset(
            "emissions_pos",
            data=[
                sum(model_block.var_tec_emissions_pos[t].value for t in self.set_t_full)
            ],
        )
        h5_group.create_dataset(
            "emissions_neg",
            data=[
                sum(model_block.var_tec_emissions_neg[t].value for t in self.set_t_full)
            ],
        )
        if "ccs" in self.performance_data and self.performance_data["ccs"]["possible"]:
            h5_group.create_dataset("size_ccs", data=[model_block.var_size_ccs.value])
            h5_group.create_dataset("capex_tec", data=[model_block.var_capex.value])
            h5_group.create_dataset("capex_ccs", data=[model_block.var_capex_ccs.value])
            h5_group.create_dataset(
                "opex_fixed_ccs", data=[model_block.var_opex_fixed_ccs.value]
            )

        h5_group.create_dataset(
            "para_unitCAPEX", data=[model_block.para_unit_capex.value]
        )
        if hasattr(model_block, "para_fix_capex"):
            h5_group.create_dataset(
                "para_fixCAPEX", data=[model_block.para_fix_capex.value]
            )

    def write_results_tec_operation(self, h5_group, model_block):

        for car in model_block.set_input_carriers_all:
            if model_block.find_component("var_input"):
                h5_group.create_dataset(
                    f"{car}_input",
                    data=[
                        model_block.var_input_tot[t, car].value for t in self.set_t_full
                    ],
                )
        for car in model_block.set_output_carriers_all:
            h5_group.create_dataset(
                f"{car}_output",
                data=[
                    model_block.var_output_tot[t, car].value for t in self.set_t_full
                ],
            )
        h5_group.create_dataset(
            "emissions_pos",
            data=[model_block.var_tec_emissions_pos[t].value for t in self.set_t_full],
        )
        h5_group.create_dataset(
            "emissions_neg",
            data=[model_block.var_tec_emissions_neg[t].value for t in self.set_t_full],
        )
        if model_block.find_component("var_x"):
            h5_group.create_dataset(
                "var_x",
                data=[
                    0 if x is None else x
                    for x in [model_block.var_x[t].value for t in self.set_t_full]
                ],
            )
        if model_block.find_component("var_y"):
            h5_group.create_dataset(
                "var_y",
                data=[
                    0 if x is None else x
                    for x in [model_block.var_y[t].value for t in self.set_t_full]
                ],
            )
        if model_block.find_component("var_z"):
            h5_group.create_dataset(
                "var_z",
                data=[
                    0 if x is None else x
                    for x in [model_block.var_z[t].value for t in self.set_t_full]
                ],
            )

        if model_block.find_component("set_input_carriers_ccs"):
            for car in model_block.set_input_carriers_ccs:
                h5_group.create_dataset(
                    f"{car}_var_input_ccs",
                    data=[
                        model_block.var_input_ccs[t, car].value for t in self.set_t_full
                    ],
                )
            for car in model_block.set_output_carriers_ccs:
                h5_group.create_dataset(
                    f"{car}_var_output_ccs",
                    data=[
                        model_block.var_output_ccs[t, car].value
                        for t in self.set_t_full
                    ],
                )

    def scale_model(self, b_tec, model, config):
        """
        Scales technology model
        """

        f = self.scaling_factors
        f_global = config.scaling_factors

        model = determine_variable_scaling(model, b_tec, f, f_global)
        model = determine_constraint_scaling(model, b_tec, f, f_global)

        return model

    def _define_input_carriers(self, b_tec):
        """
        Defines the input carriers

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        if (self.technology_model == "RES") or (self.technology_model == "CONV4"):
            b_tec.set_input_carriers = pyo.Set(initialize=[])
        else:
            b_tec.set_input_carriers = pyo.Set(
                initialize=self.performance_data["input_carrier"]
            )

        if self.ccs:
            b_tec.set_input_carriers_ccs = pyo.Set(
                initialize=self.ccs_data["TechnologyPerf"]["input_carrier"]
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
            initialize=self.performance_data["output_carrier"]
        )

        if self.ccs:
            b_tec.set_output_carriers_ccs = pyo.Set(
                initialize=self.ccs_data["TechnologyPerf"]["output_carrier"]
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
        if self.existing:
            size_max = self.size_initial
        else:
            size_max = self.size_max

        if self.size_is_int:
            size_domain = pyo.NonNegativeIntegers
        else:
            size_domain = pyo.NonNegativeReals

        b_tec.para_size_min = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=self.size_min, mutable=True
        )
        b_tec.para_size_max = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=size_max, mutable=True
        )

        if self.existing:
            b_tec.para_size_initial = pyo.Param(
                within=size_domain, initialize=self.size_initial
            )

        if self.existing and not self.decommission:
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

    def _define_capex_variables(self, b_tec, data):
        """
        Defines variables related to technology capex.

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

        def calculate_max_capex():
            if economics.capex_model == 1:
                max_capex = (
                    b_tec.para_size_max
                    * economics.capex_data["unit_capex"]
                    * annualization_factor
                )
                bounds = (0, max_capex)
            elif economics.capex_model == 2:
                max_capex = (
                    b_tec.para_size_max
                    * max(economics.capex_data["piecewise_capex"]["bp_y"])
                    * annualization_factor
                )
                bounds = (0, max_capex)
            elif economics.capex_model == 3:
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

        if self.existing and not self.decommission:
            b_tec.var_capex = pyo.Param(domain=pyo.Reals, initialize=0)
        else:
            b_tec.var_capex = pyo.Var()

        return b_tec

    def _define_capex_parameters(self, b_tec, data):
        """
        Defines the capex parameters

        For capex model 1:
        - para_unit_capex
        - para_unit_capex_annual

        For capex model 2: defined with constraints
        For capex model 3:
        - para_unit_capex
        - para_fix_capex
        - para_unit_capex_annual
        - para_fix_capex_annual

        :param b_tec:
        :param data:
        :return:
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        if economics.capex_model == 1:
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

        elif economics.capex_model == 2:
            # This is defined in the constraints
            pass
        elif economics.capex_model == 3:
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
        else:
            # Defined in the technology subclass
            pass

        if self.existing and self.decommission:
            b_tec.para_decommissioning_cost = pyo.Param(
                domain=pyo.Reals, initialize=economics.decommission_cost, mutable=True
            )

        return b_tec

    def _define_capex_constraints(self, b_tec, data):
        """
        Defines constraints related to capex.

        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        if economics.capex_model == 1:
            b_tec.const_capex_aux = pyo.Constraint(
                expr=b_tec.var_size * b_tec.para_unit_capex_annual
                == b_tec.var_capex_aux
            )
        elif economics.capex_model == 2:
            self.big_m_transformation_required = 1
            bp_x = economics.capex_data["piecewise_capex"]["bp_x"]
            bp_y_annual = [
                y * annualization_factor
                for y in economics.capex_data["piecewise_capex"]["bp_y"]
            ]
            b_tec.const_capex_aux = pyo.Piecewise(
                b_tec.var_capex_aux,
                b_tec.var_size,
                pw_pts=bp_x,
                pw_constr_type="EQ",
                f_rule=bp_y_annual,
                pw_repn="SOS2",
            )
        elif economics.capex_model == 3:
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
        if self.existing and not self.decommission:
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
        # Technology related options
        existing = self.existing
        fitted_performance = self.fitted_performance
        config = data["config"]

        # set_t and sequence
        set_t = self.set_t_full
        if (
            config["optimization"]["typicaldays"]["N"]["value"] != 0
            and not self.modelled_with_full_res
        ):
            sequence = data.data.k_means_specs.full_resolution["sequence"]

        if existing:
            size_initial = self.size_initial
            size_max = size_initial
        else:
            size_max = self.size_max

        rated_power = fitted_performance.rated_power

        def init_input_bounds(bounds, t, car):
            if (
                config["optimization"]["typicaldays"]["N"]["value"] != 0
                and not self.modelled_with_full_res
            ):
                return tuple(
                    fitted_performance.bounds["input"][car][sequence[t - 1] - 1, :]
                    * size_max
                    * rated_power
                )
            else:
                return tuple(
                    fitted_performance.bounds["input"][car][t - 1, :]
                    * size_max
                    * rated_power
                )

        b_tec.var_input = pyo.Var(
            set_t,
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
        # Technology related options
        existing = self.existing
        fitted_performance = self.fitted_performance
        config = data["config"]

        rated_power = fitted_performance.rated_power

        # set_t
        set_t = self.set_t_full
        if (
            config["optimization"]["typicaldays"]["N"]["value"] != 0
            and not self.modelled_with_full_res
        ):
            sequence = data.data.k_means_specs.full_resolution["sequence"]

        if existing:
            size_initial = self.size_initial
            size_max = size_initial
        else:
            size_max = self.size_max

        def init_output_bounds(bounds, t, car):
            if (
                config["optimization"]["typicaldays"]["N"]["value"] != 0
                and not self.modelled_with_full_res
            ):
                return tuple(
                    fitted_performance.bounds["output"][car][sequence[t - 1] - 1, :]
                    * size_max
                    * rated_power
                )
            else:
                return tuple(
                    fitted_performance.bounds["output"][car][t - 1, :]
                    * size_max
                    * rated_power
                )

        b_tec.var_output = pyo.Var(
            set_t,
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
        set_t = self.set_t_full

        # VARIABLE OPEX
        b_tec.para_opex_variable = pyo.Param(
            domain=pyo.Reals, initialize=economics.opex_variable, mutable=True
        )
        b_tec.var_opex_variable = pyo.Var(set_t)

        def init_opex_variable(const, t):
            """opexvar_{t} = Input_{t, maincarrier} * opex_{var}"""
            if (
                (self.technology_model == "RES")
                or (self.technology_model == "CONV4")
                or (self.technology_model == "DAC_Adsorption")
            ):
                opex_variable_based_on = b_tec.var_output[
                    t, b_tec.set_output_carriers[1]
                ]
            else:
                opex_variable_based_on = b_tec.var_input[t, self.main_car]
            return (
                opex_variable_based_on * b_tec.para_opex_variable
                == b_tec.var_opex_variable[t]
            )

        b_tec.const_opex_variable = pyo.Constraint(set_t, rule=init_opex_variable)

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

        set_t = self.set_t_full
        performance_data = self.performance_data
        technology_model = self.technology_model
        emissions_based_on = self.emissions_based_on

        b_tec.para_tec_emissionfactor = pyo.Param(
            domain=pyo.Reals, initialize=performance_data["emission_factor"]
        )
        b_tec.var_tec_emissions_pos = pyo.Var(set_t, within=pyo.NonNegativeReals)
        b_tec.var_tec_emissions_neg = pyo.Var(set_t, within=pyo.NonNegativeReals)

        if technology_model == "RES":
            # Set emissions to zero
            def init_tec_emissions_pos(const, t):
                return b_tec.var_tec_emissions_pos[t] == 0

            b_tec.const_tec_emissions_pos = pyo.Constraint(
                set_t, rule=init_tec_emissions_pos
            )

            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = pyo.Constraint(
                set_t, rule=init_tec_emissions_neg
            )

        else:

            if emissions_based_on == "output":

                def init_tec_emissions_pos(const, t):
                    """emissions_pos = output * emissionfactor"""
                    if performance_data["emission_factor"] >= 0:
                        return (
                            b_tec.var_output[t, performance_data["main_output_carrier"]]
                            * b_tec.para_tec_emissionfactor
                            == b_tec.var_tec_emissions_pos[t]
                        )
                    else:
                        return b_tec.var_tec_emissions_pos[t] == 0

                b_tec.const_tec_emissions_pos = pyo.Constraint(
                    set_t, rule=init_tec_emissions_pos
                )

                def init_tec_emissions_neg(const, t):
                    if performance_data["emission_factor"] < 0:
                        return (
                            b_tec.var_output[t, performance_data["main_output_carrier"]]
                            * (-b_tec.para_tec_emissionfactor)
                            == b_tec.var_tec_emissions_neg[t]
                        )
                    else:
                        return b_tec.var_tec_emissions_neg[t] == 0

                b_tec.const_tec_emissions_neg = pyo.Constraint(
                    set_t, rule=init_tec_emissions_neg
                )

            elif emissions_based_on == "input":

                def init_tec_emissions_pos(const, t):
                    if performance_data["emission_factor"] >= 0:
                        return (
                            b_tec.var_input[t, performance_data["main_input_carrier"]]
                            * b_tec.para_tec_emissionfactor
                            == b_tec.var_tec_emissions_pos[t]
                        )
                    else:
                        return b_tec.var_tec_emissions_pos[t] == 0

                b_tec.const_tec_emissions_pos = pyo.Constraint(
                    set_t, rule=init_tec_emissions_pos
                )

                def init_tec_emissions_neg(const, t):
                    if performance_data["emission_factor"] < 0:
                        return (
                            b_tec.var_input[t, performance_data["main_input_carrier"]](
                                -b_tec.para_tec_emissionfactor
                            )
                            == b_tec.var_tec_emissions_neg[t]
                        )
                    else:
                        return b_tec.var_tec_emissions_neg[t] == 0

                b_tec.const_tec_emissions_neg = pyo.Constraint(
                    set_t, rule=init_tec_emissions_neg
                )

        return b_tec

    def _define_auxiliary_vars(self, b_tec, data: dict):
        """
        Defines auxiliary variables, that are required for the modelling of clustered data

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        set_t_clustered = data.model.set_t_clustered
        set_t_full = self.set_t_full
        self.set_t = set_t_clustered
        self.sequence = data.data.k_means_specs.full_resolution["sequence"]

        if self.existing:
            size_initial = self.size_initial
            size_max = size_initial
        else:
            size_max = self.size_max

        rated_power = self.fitted_performance.rated_power

        sequence = data.data.k_means_specs.full_resolution["sequence"]

        if not (self.technology_model == "RES") and not (
            self.technology_model == "CONV4"
        ):

            def init_input_bounds(bounds, t, car):
                return tuple(
                    self.fitted_performance.bounds["input"][car][t - 1, :]
                    * size_max
                    * rated_power
                )

            b_tec.var_input_aux = pyo.Var(
                set_t_clustered,
                b_tec.set_input_carriers,
                within=pyo.NonNegativeReals,
                bounds=init_input_bounds,
            )

            b_tec.const_link_full_resolution_input = link_full_resolution_to_clustered(
                b_tec.var_input_aux,
                b_tec.var_input,
                set_t_full,
                sequence,
                b_tec.set_input_carriers,
            )
            self.input = b_tec.var_input_aux

        def init_output_bounds(bounds, t, car):
            return tuple(
                self.fitted_performance.bounds["output"][car][t - 1, :]
                * size_max
                * rated_power
            )

        b_tec.var_output_aux = pyo.Var(
            set_t_clustered,
            b_tec.set_output_carriers,
            within=pyo.NonNegativeReals,
            bounds=init_output_bounds,
        )

        b_tec.const_link_full_resolution_output = link_full_resolution_to_clustered(
            b_tec.var_output_aux,
            b_tec.var_output,
            set_t_full,
            sequence,
            b_tec.set_output_carriers,
        )

        self.output = b_tec.var_output_aux

        return b_tec

    def _aggregate_input(self, b_tec):
        """
        Aggregates ccs and technology input

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """

        b_tec.var_input_tot = pyo.Var(
            self.set_t,
            b_tec.set_input_carriers_all,
            within=pyo.NonNegativeReals,
        )

        def init_aggregate_input(const, t, car):
            """input_ccs + input = input_tot"""
            input_tec = (
                b_tec.var_input[t, car] if car in b_tec.set_input_carriers else 0
            )
            if self.ccs:
                input_ccs = (
                    b_tec.var_input_ccs[t, car]
                    if car in b_tec.set_input_carriers_ccs
                    else 0
                )
            else:
                input_ccs = 0
            return input_tec + input_ccs == b_tec.var_input_tot[t, car]

        b_tec.const_input_aggregation = pyo.Constraint(
            self.set_t, b_tec.set_input_carriers_all, rule=init_aggregate_input
        )

        return b_tec

    def _aggregate_output(self, b_tec):
        """
        Aggregates ccs and technology output

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        b_tec.var_output_tot = pyo.Var(
            self.set_t,
            b_tec.set_output_carriers_all,
            within=pyo.NonNegativeReals,
        )

        def init_aggregate_output(const, t, car):
            """output + output_ccs = output_tot"""
            output_tec = (
                b_tec.var_output[t, car] if car in b_tec.set_output_carriers else 0
            )
            if self.ccs:
                output_ccs = (
                    b_tec.var_output_ccs[t, car]
                    if car in b_tec.set_output_carriers_ccs
                    else 0
                )
            else:
                output_ccs = 0
            return output_tec + output_ccs == b_tec.var_output_tot[t, car]

        b_tec.const_output_aggregation = pyo.Constraint(
            self.set_t, b_tec.set_output_carriers_all, rule=init_aggregate_output
        )

        return b_tec

    def _aggregate_cost(self, b_tec):
        """
        Aggregates ccs and technology cost

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """

        set_t = self.set_t_full

        b_tec.var_capex_tot = pyo.Var()
        b_tec.var_opex_fixed_tot = pyo.Var()
        b_tec.var_opex_variable_tot = pyo.Var(set_t)

        def init_aggregate_capex(const):
            """capex + capex_ccs = capex_tot"""
            capex_tec = b_tec.var_capex
            if self.ccs:
                capex_ccs = b_tec.var_capex_ccs
            else:
                capex_ccs = 0
            return b_tec.var_capex_tot == capex_tec + capex_ccs

        b_tec.const_capex_aggregation = pyo.Constraint(rule=init_aggregate_capex)

        def init_aggregate_opex_var(const, t):
            """var_opex_variable + var_opex_variable_ccs = var_opex_variable_tot"""
            opex_var_tec = b_tec.var_opex_variable[t]
            if self.ccs:
                opex_var_ccs = b_tec.var_opex_variable_ccs[t]
            else:
                opex_var_ccs = 0
            return b_tec.var_opex_variable_tot[t] == opex_var_tec + opex_var_ccs

        b_tec.const_opex_var_aggregation = pyo.Constraint(
            set_t, rule=init_aggregate_opex_var
        )

        def init_aggregate_opex_fixed(const):
            """var_opex_fixed + var_opex_fixed_ccs = var_opex_fixed_tot"""
            opex_fixed_tec = b_tec.var_opex_fixed
            if self.ccs:
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
            data=[sum(model_block.var_opex_variable[t].value for t in self.set_t_full)],
        )
        h5_group.create_dataset(
            "opex_fixed_tot", data=[model_block.var_opex_fixed_tot.value]
        )
        h5_group.create_dataset(
            "emissions_pos",
            data=[
                sum(model_block.var_tec_emissions_pos[t].value for t in self.set_t_full)
            ],
        )
        h5_group.create_dataset(
            "emissions_neg",
            data=[
                sum(model_block.var_tec_emissions_neg[t].value for t in self.set_t_full)
            ],
        )
        if "ccs" in self.performance_data and self.performance_data["ccs"]["possible"]:
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
                        model_block.var_input_tot[t, car].value for t in self.set_t_full
                    ],
                )
        for car in model_block.set_output_carriers_all:
            h5_group.create_dataset(
                f"{car}_output",
                data=[
                    model_block.var_output_tot[t, car].value for t in self.set_t_full
                ],
            )
        h5_group.create_dataset(
            "emissions_pos",
            data=[model_block.var_tec_emissions_pos[t].value for t in self.set_t_full],
        )
        h5_group.create_dataset(
            "emissions_neg",
            data=[model_block.var_tec_emissions_neg[t].value for t in self.set_t_full],
        )
        if model_block.find_component("var_x"):
            h5_group.create_dataset(
                "var_x",
                data=[
                    0 if x is None else x
                    for x in [model_block.var_x[t].value for t in self.set_t_full]
                ],
            )
        if model_block.find_component("var_y"):
            h5_group.create_dataset(
                "var_y",
                data=[
                    0 if x is None else x
                    for x in [model_block.var_y[t].value for t in self.set_t_full]
                ],
            )
        if model_block.find_component("var_z"):
            h5_group.create_dataset(
                "var_z",
                data=[
                    0 if x is None else x
                    for x in [model_block.var_z[t].value for t in self.set_t_full]
                ],
            )

        if model_block.find_component("set_input_carriers_ccs"):
            for car in model_block.set_input_carriers_ccs:
                h5_group.create_dataset(
                    f"{car}_var_input_ccs",
                    data=[
                        model_block.var_input_ccs[t, car].value for t in self.set_t_full
                    ],
                )
            for car in model_block.set_output_carriers_ccs:
                h5_group.create_dataset(
                    f"{car}_var_output_ccs",
                    data=[
                        model_block.var_output_ccs[t, car].value
                        for t in self.set_t_full
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
        size_max = self.ccs_data["size_max"]
        set_t = self.set_t_full
        carbon_capture_rate = self.ccs_data["TechnologyPerf"]["capture_rate"]
        performance_data = self.performance_data
        emissions_based_on = self.emissions_based_on
        config = data["config"]

        # LOG
        log_event(f"\t - Adding CCS to Technology {self.name}")

        # TODO: maybe make the full set of all carriers as a intersection between this set and the others?
        # Emission Factor
        b_tec.para_tec_emissionfactor = pyo.Param(
            domain=pyo.Reals, initialize=performance_data["emission_factor"]
        )
        b_tec.var_tec_emissions_pos = pyo.Var(set_t, within=pyo.NonNegativeReals)
        b_tec.var_tec_emissions_neg = pyo.Var(set_t, within=pyo.NonNegativeReals)

        def init_input_bounds(bounds, t, car):
            if (
                config["optimization"]["typicaldays"]["N"]["value"] != 0
                and not self.modelled_with_full_res
            ):
                return tuple(
                    self.ccs_data["TechnologyPerf"]["bounds"]["input"][car][
                        pyo.pyomo.environ.sequence[t - 1] - 1, :
                    ]
                    * size_max
                )
            else:
                return tuple(
                    self.ccs_data["TechnologyPerf"]["bounds"]["input"][car][t - 1, :]
                    * size_max
                )

        b_tec.var_input_ccs = pyo.Var(
            set_t,
            b_tec.set_input_carriers_ccs,
            within=pyo.NonNegativeReals,
            bounds=init_input_bounds,
        )

        def init_output_bounds(bounds, t, car):
            if (
                config["optimization"]["typicaldays"]["N"]["value"] != 0
                and not self.modelled_with_full_res
            ):
                return tuple(
                    self.ccs_data["TechnologyPerf"]["bounds"]["output"][car][
                        pyo.pyomo.environ.sequence[t - 1] - 1, :
                    ]
                    * size_max
                )
            else:
                return tuple(
                    self.ccs_data["TechnologyPerf"]["bounds"]["output"][car][t - 1, :]
                    * size_max
                )

        b_tec.var_output_ccs = pyo.Var(
            set_t,
            b_tec.set_output_carriers_ccs,
            within=pyo.NonNegativeReals,
            bounds=init_output_bounds,
        )

        # Input-output correlation
        def init_input_output_ccs(const, t):
            if emissions_based_on == "output":
                return (
                    b_tec.var_output_ccs[t, "CO2captured"]
                    <= carbon_capture_rate
                    * b_tec.para_tec_emissionfactor
                    * b_tec.var_output[t, self.main_car]
                )
            else:
                return (
                    b_tec.var_output_ccs[t, "CO2captured"]
                    <= carbon_capture_rate
                    * b_tec.para_tec_emissionfactor
                    * b_tec.var_input[t, self.main_car]
                )

        b_tec.const_input_output_ccs = pyo.Constraint(set_t, rule=init_input_output_ccs)

        # Electricity and heat demand CCS
        def init_input_ccs(const, t, car):
            return (
                b_tec.var_input_ccs[t, car]
                == self.ccs_data["TechnologyPerf"]["input_ratios"][car]
                * b_tec.var_output_ccs[t, "CO2captured"]
                / carbon_capture_rate
            )

        b_tec.const_input_el = pyo.Constraint(
            set_t, b_tec.set_input_carriers_ccs, rule=init_input_ccs
        )

        return b_tec

    def _define_ccs_emissions(self, b_tec):
        """
        Defines CCS performance. The unit capex parameter is calculated from Eq. 10 of Weimann et al. 2023

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """

        set_t = self.set_t_full
        performance_data = self.performance_data
        emissions_based_on = self.emissions_based_on

        # Emissions
        if emissions_based_on == "output":

            def init_tec_emissions_pos(const, t):
                return (
                    b_tec.var_output[t, performance_data["main_output_carrier"]]
                    * b_tec.para_tec_emissionfactor
                    - b_tec.var_output_ccs[t, "CO2captured"]
                    == b_tec.var_tec_emissions_pos[t]
                )

            b_tec.const_tec_emissions_pos = pyo.Constraint(
                set_t, rule=init_tec_emissions_pos
            )

            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = pyo.Constraint(
                set_t, rule=init_tec_emissions_neg
            )

        elif emissions_based_on == "input":

            def init_tec_emissions_pos(const, t):
                return (
                    b_tec.var_input[t, performance_data["main_input_carrier"]]
                    * b_tec.para_tec_emissionfactor
                    - b_tec.var_output_ccs[t, "CO2captured"]
                    == b_tec.var_tec_emissions_pos[t]
                )

            b_tec.const_tec_emissions_pos = pyo.Constraint(
                set_t, rule=init_tec_emissions_pos
            )

            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = pyo.Constraint(
                set_t, rule=init_tec_emissions_neg
            )

        # Initialize the size of CCS as in _define_size (size given in mass flow of CO2 entering the CCS object)
        b_tec.para_size_min_ccs = pyo.Param(
            domain=pyo.NonNegativeReals,
            initialize=self.ccs_data["size_min"],
            mutable=True,
        )
        b_tec.para_size_max_ccs = pyo.Param(
            domain=pyo.NonNegativeReals,
            initialize=self.ccs_data["size_max"],
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
        co2_concentration = self.performance_data["ccs"]["co2_concentration"]
        carbon_capture_rate = self.ccs_data["TechnologyPerf"]["capture_rate"]
        config = data["config"]

        # Costs
        economics = self.ccs_data["Economics"]
        economics = SimpleNamespace(**economics)
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )
        molar_mass_CO2 = 44.01
        carbon_capture_rate = self.ccs_data["TechnologyPerf"]["capture_rate"]
        convert2t_per_h = (
            molar_mass_CO2 * co2_concentration * 3.6
        )  # convert kmol/s of fluegas to ton/h of CO2molar_mass_CO2 = 44.01

        def init_unit_capex_ccs_annualized(self):
            unit_capex = (
                economics.CAPEX_kappa / convert2t_per_h
                + economics.CAPEX_lambda
                * carbon_capture_rate
                * co2_concentration
                / convert2t_per_h
            ) * annualization_factor
            return unit_capex

        b_tec.para_unit_capex_annual_ccs = pyo.Param(
            domain=pyo.Reals, initialize=init_unit_capex_ccs_annualized, mutable=True
        )
        b_tec.para_fix_capex_annual_ccs = pyo.Param(
            domain=pyo.Reals,
            initialize=annualization_factor * economics.CAPEX_zeta,
            mutable=True,
        )

        def calculate_max_capex_ccs():
            max_capex = (
                self.ccs_data["size_max"] * b_tec.para_unit_capex_annual_ccs
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
            domain=pyo.Reals, initialize=economics.OPEX_fixed, mutable=True
        )
        b_tec.var_opex_fixed_ccs = pyo.Var()
        b_tec.const_opex_fixed_ccs = pyo.Constraint(
            expr=b_tec.var_capex_aux_ccs * b_tec.para_opex_fixed_ccs
            == b_tec.var_opex_fixed_ccs
        )

        # VARIABLE OPEX
        set_t = self.set_t_full
        b_tec.para_opex_variable_ccs = pyo.Param(
            domain=pyo.Reals, initialize=economics.OPEX_variable, mutable=True
        )
        b_tec.var_opex_variable_ccs = pyo.Var(set_t)

        def init_opex_variable_ccs(const, t):
            return (
                b_tec.var_output_ccs[t, b_tec.set_output_carriers_ccs[1]]
                * b_tec.para_opex_variable_ccs
                == b_tec.var_opex_variable_ccs[t]
            )

        b_tec.const_opex_variable_ccs = pyo.Constraint(
            set_t, rule=init_opex_variable_ccs
        )

        return b_tec

    # DYNAMICS FUNCTIONS
    def _define_dynamics(self, b_tec):
        """
        Selects the dynamic constraints that are required based on the technology dynamic performance parameters or the
        performance function type.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        log_event(f"\t - Adding dynamics to Technology {self.name}")
        SU_load = self.performance_data["SU_load"]
        SD_load = self.performance_data["SD_load"]
        min_uptime = self.performance_data["min_uptime"]
        min_downtime = self.performance_data["min_downtime"]
        max_startups = self.performance_data["max_startups"]
        performance_function_type4 = (
            "performance_function_type" in self.performance_data
        ) and (self.performance_data["performance_function_type"] == 4)
        if (
            (min_uptime + min_downtime > -2)
            or (max_startups > -1)
            or (SU_load + SD_load > -2)
            or performance_function_type4
        ):
            b_tec = self._dynamics_SUSD_logic(b_tec)
        if not performance_function_type4 and (SU_load + SD_load > -2):
            b_tec = self._dynamics_fast_SUSD(b_tec)

        log_event(f"\t - Adding dynamics to Technology {self.name}", print_it=False)

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
        # New variables
        b_tec.var_x = pyo.Var(
            self.set_t_full, domain=pyo.NonNegativeReals, bounds=(0, 1)
        )
        b_tec.var_y = pyo.Var(
            self.set_t_full, domain=pyo.NonNegativeReals, bounds=(0, 1)
        )
        b_tec.var_z = pyo.Var(
            self.set_t_full, domain=pyo.NonNegativeReals, bounds=(0, 1)
        )

        # Check for default values
        para_names = ["SU_time", "SD_time"]
        for para in para_names:
            if self.performance_data[para] < 0:
                self.performance_data[para] = 0
                warn(
                    "Using SU/SD logic constraints, parameter "
                    + str(para)
                    + " set to default value 0"
                )
        para_names = ["min_uptime", "min_downtime"]
        for para in para_names:
            if self.performance_data[para] < 0:
                self.performance_data[para] = 1
                warn(
                    "Using SU/SD logic constraints, parameter "
                    + str(para)
                    + " set to default value 1"
                )

        # Collect parameters
        SU_time = self.performance_data["SU_time"]
        SD_time = self.performance_data["SD_time"]
        min_uptime = self.performance_data["min_uptime"]
        min_downtime = self.performance_data["min_downtime"] + SU_time + SD_time
        max_startups = self.performance_data["max_startups"]

        # Enforce startup/shutdown logic
        def init_SUSD_logic1(const, t):
            if t == 1:
                return pyo.Constraint.Skip
            else:
                return (
                    b_tec.var_x[t] - b_tec.var_x[t - 1]
                    == b_tec.var_y[t] - b_tec.var_z[t]
                )

        b_tec.const_SUSD_logic1 = pyo.Constraint(self.set_t_full, rule=init_SUSD_logic1)

        def init_SUSD_logic2(const, t):
            if t >= min_uptime:
                return b_tec.var_y[t - min_uptime + 1] <= b_tec.var_x[t]
            else:
                return (
                    b_tec.var_y[len(self.set_t_full) + (t - min_uptime + 1)]
                    <= b_tec.var_x[t]
                )

        b_tec.const_SUSD_logic2 = pyo.Constraint(self.set_t_full, rule=init_SUSD_logic2)

        def init_SUSD_logic3(const, t):
            if t >= min_downtime:
                return b_tec.var_z[t - min_downtime + 1] <= 1 - b_tec.var_x[t]
            else:
                return (
                    b_tec.var_z[len(self.set_t_full) + (t - min_downtime + 1)]
                    <= 1 - b_tec.var_x[t]
                )

        b_tec.const_SUSD_logic3 = pyo.Constraint(self.set_t_full, rule=init_SUSD_logic3)

        # Constrain number of startups
        if not max_startups == -1:

            def init_max_startups(const):
                return sum(b_tec.var_y[t] for t in self.set_t_full) <= max_startups

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

        # Check for default values
        para_names = ["SU_load", "SD_load"]
        for para in para_names:
            if self.performance_data[para] < 0:
                self.performance_data[para] = 1
                warn(
                    "Using SU/SD load constraints, parameter"
                    + str(para)
                    + "set to default value 1"
                )

        # Collect parameters
        SU_load = self.performance_data["SU_load"]
        SD_load = self.performance_data["SD_load"]
        main_car = self.performance_data["main_input_carrier"]
        rated_power = self.fitted_performance.rated_power

        # SU load limit
        s_indicators = range(0, 2)

        def init_SU_load(dis, t, ind):
            if ind == 0:  # no startup (y=0)
                dis.const_y_off = pyo.Constraint(expr=b_tec.var_y[t] == 0)

            else:  # tech in startup
                dis.const_y_on = pyo.Constraint(expr=b_tec.var_y[t] == 1)

                def init_SU_load_limit(cons):
                    if self.technology_model == "CONV3":
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
            self.set_t_full, s_indicators, rule=init_SU_load
        )

        def bind_disjunctions_SU_load(dis, t):
            return [b_tec.dis_SU_load[t, i] for i in s_indicators]

        b_tec.disjunction_SU_load = gdp.Disjunction(
            self.set_t_full, rule=bind_disjunctions_SU_load
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
                        if self.technology_model == "CONV3":
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
            self.set_t_full, s_indicators, rule=init_SD_load
        )

        def bind_disjunctions_SD_load(dis, t):
            return [b_tec.dis_SD_load[t, i] for i in s_indicators]

        b_tec.disjunction_SD_load = gdp.Disjunction(
            self.set_t_full, rule=bind_disjunctions_SD_load
        )

        return b_tec
