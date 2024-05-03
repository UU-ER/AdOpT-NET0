from pyomo.gdp import *
from warnings import warn
from pyomo.environ import *
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

"""
TODO
Suggestions:
- add sources to documentation
- Delete src/model_construction/generic_technology_constraints.py
"""


class Technology(ModelComponent):
    """
    Class to read and manage data for technologies

    This class is parent class to all generic and specific technologies. It creates the variables, parameters,
    constraints and sets of a technology.
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

    def construct_tech_model(
        self, b_tec: Block, data: dict, set_t: Set, set_t_clustered: Set
    ) -> Block:
        r"""
        Construct the technology model

        This function is extented in the generic/specific technology classes. It adds Sets, Parameters, Variables and
        Constraints that are common for all technologies (see below
        for the case when CCS is possible).
        The following description is true for new technologies. For existing technologies a few adaptions are made
        (see below).

        **Set declarations:**

        - Set of input carriers
        - Set of output carriers

        **Parameter declarations:**

        - Min Size
        - Max Size
        - Output max (same as size max)
        - Unit CAPEX (annualized from given data on up-front CAPEX, lifetime and discount rate)
        - Variable OPEX
        - Fixed OPEX

        **Variable declarations:**

        - Size (can be integer or continuous)
        - Input for each input carrier
        - Output for each output carrier
        - CAPEX
        - Variable OPEX
        - Fixed OPEX

        **Constraint declarations**

        - CAPEX, can be linear (for ``capex_model == 1``), piecewise linear (for ``capex_model == 2``) or linear with \
          a fixed cost when the technology is installed (for ``capex_model == 3``). Linear is defined as:

            .. math::
                CAPEX_{tec} = Size_{tec} * UnitCost_{tec}

          while linear with fixed installation costs is defined as:

            .. math::
                CAPEX_{tec} = Size_{tec} * UnitCost_{tec} + FixCost_{tec}

        - Variable OPEX: defined per unit of output for the main carrier:

            .. math::
                OPEXvar_{t, tec} = Output_{t, maincarrier} * opex_{var} \forall t \in T

        - Fixed OPEX: defined as a fraction of annual CAPEX:

            .. math::
                OPEXfix_{tec} = CAPEX_{tec} * opex_{fix}

        Existing technologies, i.e. existing = 1, can be decommissioned (decommission = 1) or not (decommission = 0).
        For technologies that cannot be decommissioned, the size is fixed to the size given in the technology data.
        For technologies that can be decommissioned, the size can be smaller or equal to the initial size. Reducing the
        size comes at the decommissioning costs specified in the economics of the technology.
        The fixed opex is calculated by determining the capex that the technology would have costed if newly build and
        then taking the respective opex_fixed share of this. This is done with the auxiliary variable var_capex_aux.

        :param str nodename: name of node for which technology is installed
        :param set set_tecsToAdd: list of technologies to add
        :param energyhub EnergyHub: instance of the energyhub
        :return: b_node

        When CCS is available, we add heat and electricity to the input carriers Set and CO2captured to the output
        carriers Set. Moreover, we create extra Parameters and Variables equivalent to the ones created for the
        technology, but specific for CCS. In addition, we create Variables that are the sum of the input, output,
        CAPEX and OPEX of the technology and of CCS. We calculate the emissions of the techology discounting already
        what is being captured by the CCS.

        **Parameter declarations:**

        - Min Size CCS
        - Max Size CCS
        - Unit CAPEX CCS (annualized from given data on up-front CAPEX, lifetime and discount rate)
        - Fixed OPEX (fraction of the CAPEX)

        **Variable declarations:**

        - Size CCS (in t/h of CO2 entering capture process)
        - Input for heat and electricity
        - Output of CO2 captured
        - CAPEX CCS
        - Fixed OPEX CCS
        - Total input
        - Total output
        - Total CAPEX
        - Total OPEX fixed

        **Constraint declarations**


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

        print("\t - Adding Technology " + self.name)

        # TECHNOLOGY DATA
        config = data["config"]

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
        b_tec = self._define_capex(b_tec, data)
        b_tec = self._define_input(b_tec, data)
        b_tec = self._define_output(b_tec, data)
        b_tec = self._define_opex(b_tec, data)
        if self.ccs:
            b_tec = self._define_ccs_performance(b_tec, data)
            b_tec = self._define_ccs_emissions(b_tec, data)
            b_tec = self._define_ccs_costs(b_tec, data)
        else:
            b_tec = self._define_emissions(b_tec, data)

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

        self._aggregate_input(b_tec)
        self._aggregate_output(b_tec)
        self._aggregate_cost(b_tec)

        return b_tec

    def _aggregate_input(self, b_tec):

        b_tec.var_input_tot = Var(
            self.set_t,
            b_tec.set_input_carriers_all,
            within=NonNegativeReals,
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

        b_tec.const_input_aggregation = Constraint(
            self.set_t, b_tec.set_input_carriers_all, rule=init_aggregate_input
        )

        return b_tec

    def _aggregate_output(self, b_tec):

        b_tec.var_output_tot = Var(
            self.set_t,
            b_tec.set_output_carriers_all,
            within=NonNegativeReals,
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

        b_tec.const_output_aggregation = Constraint(
            self.set_t, b_tec.set_output_carriers_all, rule=init_aggregate_output
        )

        return b_tec

    def _aggregate_cost(self, b_tec):

        set_t = self.set_t_full

        b_tec.var_capex_tot = Var()
        b_tec.var_opex_fixed_tot = Var()
        b_tec.var_opex_variable_tot = Var(set_t)

        def init_aggregate_capex(const):
            capex_tec = b_tec.var_capex
            if self.ccs:
                capex_ccs = b_tec.var_capex_ccs
            else:
                capex_ccs = 0
            return b_tec.var_capex_tot == capex_tec + capex_ccs

        b_tec.const_capex_aggregation = Constraint(rule=init_aggregate_capex)

        def init_aggregate_opex_var(const, t):
            opex_var_tec = b_tec.var_opex_variable[t]
            if self.ccs:
                opex_var_ccs = b_tec.var_opex_variable_ccs[t]
            else:
                opex_var_ccs = 0
            return b_tec.var_opex_variable_tot[t] == opex_var_tec + opex_var_ccs

        b_tec.const_opex_var_aggregation = Constraint(
            set_t, rule=init_aggregate_opex_var
        )

        def init_aggregate_opex_fixed(const):
            opex_fixed_tec = b_tec.var_opex_fixed
            if self.ccs:
                opex_fixed_ccs = b_tec.var_opex_fixed_ccs
            else:
                opex_fixed_ccs = 0
            return b_tec.var_opex_fixed_tot == opex_fixed_tec + opex_fixed_ccs

        b_tec.const_opex_fixed_aggregation = Constraint(rule=init_aggregate_opex_fixed)

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
        """
        if (self.technology_model == "RES") or (self.technology_model == "CONV4"):
            b_tec.set_input_carriers = Set(initialize=[])
        else:
            b_tec.set_input_carriers = Set(
                initialize=self.performance_data["input_carrier"]
            )

        if self.ccs:
            b_tec.set_input_carriers_ccs = Set(
                initialize=self.ccs_data["TechnologyPerf"]["input_carrier"]
            )
        else:
            b_tec.set_input_carriers_ccs = Set(initialize=[])

        b_tec.set_input_carriers_all = (
            b_tec.set_input_carriers_ccs | b_tec.set_input_carriers
        )

        return b_tec

    def _define_output_carriers(self, b_tec):
        """
        Defines the output carriers
        """
        b_tec.set_output_carriers = Set(
            initialize=self.performance_data["output_carrier"]
        )

        if self.ccs:
            b_tec.set_output_carriers_ccs = Set(
                initialize=self.ccs_data["TechnologyPerf"]["output_carrier"]
            )
        else:
            b_tec.set_output_carriers_ccs = Set(initialize=[])

        b_tec.set_output_carriers_all = (
            b_tec.set_output_carriers_ccs | b_tec.set_output_carriers
        )

        return b_tec

    def _define_size(self, b_tec):
        """
        Defines variables and parameters related to technology size.

        Parameters defined:
        - size min
        - size max
        - size initial (for existing technologies)

        Variables defined:
        - size
        """
        if self.existing:
            size_max = self.size_initial
        else:
            size_max = self.size_max

        if self.size_is_int:
            size_domain = NonNegativeIntegers
        else:
            size_domain = NonNegativeReals

        b_tec.para_size_min = Param(
            domain=NonNegativeReals, initialize=self.size_min, mutable=True
        )
        b_tec.para_size_max = Param(
            domain=NonNegativeReals, initialize=size_max, mutable=True
        )

        if self.existing:
            b_tec.para_size_initial = Param(
                within=size_domain, initialize=self.size_initial
            )

        if self.existing and not self.decommission:
            # Decommissioning is not possible, size fixed
            b_tec.var_size = Param(
                within=size_domain, initialize=b_tec.para_size_initial
            )
        else:
            # Decommissioning is possible, size variable
            b_tec.var_size = Var(
                within=size_domain, bounds=(b_tec.para_size_min, b_tec.para_size_max)
            )

        return b_tec

    def _define_capex(self, b_tec, data):
        """
        Defines variables and parameters related to technology capex.

        Parameters defined:
        - unit capex/ breakpoints for capex function

        Variables defined:
        - capex_aux (theoretical CAPEX for existing technologies)
        - CAPEX (actual CAPEX)
        - Decommissioning Costs (for existing technologies)
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
        b_tec.var_capex_aux = Var(bounds=calculate_max_capex())

        if capex_model == 1:
            b_tec.para_unit_capex = Param(
                domain=Reals,
                initialize=economics.capex_data["unit_capex"],
                mutable=True,
            )
            b_tec.para_unit_capex_annual = Param(
                domain=Reals,
                initialize=annualization_factor * economics.capex_data["unit_capex"],
                mutable=True,
            )
            b_tec.const_capex_aux = Constraint(
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
            b_tec.const_capex_aux = Piecewise(
                b_tec.var_capex_aux,
                b_tec.var_size,
                pw_pts=bp_x,
                pw_constr_type="EQ",
                f_rule=bp_y_annual,
                pw_repn="SOS2",
            )
        elif capex_model == 3:
            b_tec.para_unit_capex = Param(
                domain=Reals,
                initialize=economics.capex_data["unit_capex"],
                mutable=True,
            )
            b_tec.para_fix_capex = Param(
                domain=Reals, initialize=economics.capex_data["fix_capex"], mutable=True
            )
            b_tec.para_unit_capex_annual = Param(
                domain=Reals,
                initialize=annualization_factor * economics.capex_data["unit_capex"],
                mutable=True,
            )
            b_tec.para_fix_capex_annual = Param(
                domain=Reals,
                initialize=annualization_factor * economics.capex_data["fix_capex"],
                mutable=True,
            )

            # capex unit commitment constraint
            self.big_m_transformation_required = 1
            s_indicators = range(0, 2)

            def init_installation(dis, ind):
                if ind == 0:  # tech not installed
                    dis.const_capex_aux = Constraint(expr=b_tec.var_capex_aux == 0)
                    dis.const_not_installed = Constraint(expr=b_tec.var_size == 0)
                else:  # tech installed
                    dis.const_capex_aux = Constraint(
                        expr=b_tec.var_size * b_tec.para_unit_capex_annual
                        + b_tec.para_fix_capex_annual
                        == b_tec.var_capex_aux
                    )

            b_tec.dis_installation = Disjunct(s_indicators, rule=init_installation)

            def bind_disjunctions(dis):
                return [b_tec.dis_installation[i] for i in s_indicators]

            b_tec.disjunction_installation = Disjunction(rule=bind_disjunctions)

        else:
            # Defined in the technology subclass
            pass

        # CAPEX
        if self.existing and not self.decommission:
            b_tec.var_capex = Param(domain=Reals, initialize=0)
        else:
            b_tec.var_capex = Var()
            if self.existing:
                b_tec.para_decommissioning_cost = Param(
                    domain=Reals, initialize=economics.decommission_cost, mutable=True
                )
                b_tec.const_capex = Constraint(
                    expr=b_tec.var_capex
                    == (b_tec.para_size_initial - b_tec.var_size)
                    * b_tec.para_decommissioning_cost
                )
            else:
                b_tec.const_capex = Constraint(
                    expr=b_tec.var_capex == b_tec.var_capex_aux
                )

        return b_tec

    def _define_input(self, b_tec, data):
        """
        Defines input to a technology

        var_input is always in full resolution
        var_input_aux can be in reduced resolution
        """
        # Technology related options
        existing = self.existing
        performance_data = self.performance_data
        fitted_performance = self.fitted_performance
        technology_model = self.technology_model
        modelled_with_full_res = self.modelled_with_full_res
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

        b_tec.var_input = Var(
            set_t,
            b_tec.set_input_carriers,
            within=NonNegativeReals,
            bounds=init_input_bounds,
        )

        return b_tec

    def _define_output(self, b_tec, data):
        """
        Defines output to a technology

        var_output is always in full resolution
        """
        # Technology related options
        existing = self.existing
        performance_data = self.performance_data
        fitted_performance = self.fitted_performance
        modelled_with_full_res = self.modelled_with_full_res
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

        b_tec.var_output = Var(
            set_t,
            b_tec.set_output_carriers,
            within=NonNegativeReals,
            bounds=init_output_bounds,
        )
        return b_tec

    def _define_opex(self, b_tec, data):
        """
        Defines variable and fixed OPEX
        """
        economics = self.economics
        set_t = self.set_t_full

        # VARIABLE OPEX
        b_tec.para_opex_variable = Param(
            domain=Reals, initialize=economics.opex_variable, mutable=True
        )
        b_tec.var_opex_variable = Var(set_t)

        def init_opex_variable(const, t):
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

        b_tec.const_opex_variable = Constraint(set_t, rule=init_opex_variable)

        # FIXED OPEX
        b_tec.para_opex_fixed = Param(
            domain=Reals, initialize=economics.opex_fixed, mutable=True
        )
        b_tec.var_opex_fixed = Var()
        b_tec.const_opex_fixed = Constraint(
            expr=b_tec.var_capex_aux * b_tec.para_opex_fixed == b_tec.var_opex_fixed
        )
        return b_tec

    def _define_emissions(self, b_tec, data):
        """
        Defines Emissions
        """

        set_t = self.set_t_full
        performance_data = self.performance_data
        technology_model = self.technology_model
        emissions_based_on = self.emissions_based_on

        b_tec.para_tec_emissionfactor = Param(
            domain=Reals, initialize=performance_data["emission_factor"]
        )
        b_tec.var_tec_emissions_pos = Var(set_t, within=NonNegativeReals)
        b_tec.var_tec_emissions_neg = Var(set_t, within=NonNegativeReals)

        if technology_model == "RES":
            # Set emissions to zero
            def init_tec_emissions_pos(const, t):
                return b_tec.var_tec_emissions_pos[t] == 0

            b_tec.const_tec_emissions_pos = Constraint(
                set_t, rule=init_tec_emissions_pos
            )

            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = Constraint(
                set_t, rule=init_tec_emissions_neg
            )

        else:

            if emissions_based_on == "output":

                def init_tec_emissions_pos(const, t):
                    if performance_data["emission_factor"] >= 0:
                        return (
                            b_tec.var_output[t, performance_data["main_output_carrier"]]
                            * b_tec.para_tec_emissionfactor
                            == b_tec.var_tec_emissions_pos[t]
                        )
                    else:
                        return b_tec.var_tec_emissions_pos[t] == 0

                b_tec.const_tec_emissions_pos = Constraint(
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

                b_tec.const_tec_emissions_neg = Constraint(
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

                b_tec.const_tec_emissions_pos = Constraint(
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

                b_tec.const_tec_emissions_neg = Constraint(
                    set_t, rule=init_tec_emissions_neg
                )

        return b_tec

    def _define_ccs_performance(self, b_tec, data):
        """
        Defines CCS performance. The unit capex parameter is calculated from Eq. 10 of Weimann et al. 2023
        """
        size_max = self.ccs_data["size_max"]
        set_t = self.set_t_full
        carbon_capture_rate = self.ccs_data["TechnologyPerf"]["capture_rate"]
        performance_data = self.performance_data
        emissions_based_on = self.emissions_based_on
        config = data["config"]

        # TODO: maybe make the full set of all carriers as a intersection between this set and the others?
        # Emission Factor
        b_tec.para_tec_emissionfactor = Param(
            domain=Reals, initialize=performance_data["emission_factor"]
        )
        b_tec.var_tec_emissions_pos = Var(set_t, within=NonNegativeReals)
        b_tec.var_tec_emissions_neg = Var(set_t, within=NonNegativeReals)

        def init_input_bounds(bounds, t, car):
            if (
                config["optimization"]["typicaldays"]["N"]["value"] != 0
                and not self.modelled_with_full_res
            ):
                return tuple(
                    self.ccs_data["TechnologyPerf"]["bounds"]["input"][car][
                        sequence[t - 1] - 1, :
                    ]
                    * size_max
                )
            else:
                return tuple(
                    self.ccs_data["TechnologyPerf"]["bounds"]["input"][car][t - 1, :]
                    * size_max
                )

        b_tec.var_input_ccs = Var(
            set_t,
            b_tec.set_input_carriers_ccs,
            within=NonNegativeReals,
            bounds=init_input_bounds,
        )

        def init_output_bounds(bounds, t, car):
            if (
                config["optimization"]["typicaldays"]["N"]["value"] != 0
                and not self.modelled_with_full_res
            ):
                return tuple(
                    self.ccs_data["TechnologyPerf"]["bounds"]["output"][car][
                        sequence[t - 1] - 1, :
                    ]
                    * size_max
                )
            else:
                return tuple(
                    self.ccs_data["TechnologyPerf"]["bounds"]["output"][car][t - 1, :]
                    * size_max
                )

        b_tec.var_output_ccs = Var(
            set_t,
            b_tec.set_output_carriers_ccs,
            within=NonNegativeReals,
            bounds=init_output_bounds,
        )

        # Input-output correlation
        def init_input_output_ccs(const, t):
            if emissions_based_on == "output":
                print(emissions_based_on)
                return (
                    b_tec.var_output_ccs[t, "CO2captured"]
                    <= carbon_capture_rate
                    * b_tec.para_tec_emissionfactor
                    * b_tec.var_output[t, self.main_car]
                )
            else:
                print(emissions_based_on)
                return (
                    b_tec.var_output_ccs[t, "CO2captured"]
                    <= carbon_capture_rate
                    * b_tec.para_tec_emissionfactor
                    * b_tec.var_input[t, self.main_car]
                )

        b_tec.const_input_output_ccs = Constraint(set_t, rule=init_input_output_ccs)

        # Electricity and heat demand CCS
        def init_input_ccs(const, t, car):
            return (
                b_tec.var_input_ccs[t, car]
                == self.ccs_data["TechnologyPerf"]["input_ratios"][car]
                * b_tec.var_output_ccs[t, "CO2captured"]
                / carbon_capture_rate
            )

        b_tec.const_input_el = Constraint(
            set_t, b_tec.set_input_carriers_ccs, rule=init_input_ccs
        )

        return b_tec

    def _define_ccs_emissions(self, b_tec, data):
        """
        Defines CCS performance. The unit capex parameter is calculated from Eq. 10 of Weimann et al. 2023
        """
        co2_concentration = self.performance_data["ccs"]["co2_concentration"]
        set_t = self.set_t_full
        carbon_capture_rate = self.ccs_data["TechnologyPerf"]["capture_rate"]
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

            b_tec.const_tec_emissions_pos = Constraint(
                set_t, rule=init_tec_emissions_pos
            )

            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = Constraint(
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

            b_tec.const_tec_emissions_pos = Constraint(
                set_t, rule=init_tec_emissions_pos
            )

            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = Constraint(
                set_t, rule=init_tec_emissions_neg
            )

        # Initialize the size of CCS as in _define_size (size given in mass flow of CO2 entering the CCS object)
        b_tec.para_size_min_ccs = Param(
            domain=NonNegativeReals, initialize=self.ccs_data["size_min"], mutable=True
        )
        b_tec.para_size_max_ccs = Param(
            domain=NonNegativeReals, initialize=self.ccs_data["size_max"], mutable=True
        )

        # Decommissioning is possible, size variable
        b_tec.var_size_ccs = Var(
            within=NonNegativeReals,
            bounds=(0, b_tec.para_size_max_ccs),
        )

        return b_tec

    def _define_ccs_costs(self, b_tec, data):
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

        # CAPEX auxilliary (used to calculate theoretical CAPEX)
        # For new technologies, this is equal to actual CAPEX
        # For existing technologies it is used to calculate fixed OPEX
        b_tec.para_kappa_ccs = Param(
            domain=Reals, initialize=economics.CAPEX_kappa, mutable=True
        )
        b_tec.para_lambda_ccs = Param(
            domain=Reals, initialize=economics.CAPEX_lambda, mutable=True
        )
        b_tec.para_zeta_ccs = Param(
            domain=Reals, initialize=economics.CAPEX_zeta, mutable=True
        )

        def init_unit_capex_ccs_annualized(self):
            unit_capex = (
                economics.CAPEX_kappa / convert2t_per_h
                + economics.CAPEX_lambda
                * carbon_capture_rate
                * co2_concentration
                / convert2t_per_h
            ) * annualization_factor
            return unit_capex

        b_tec.para_unit_capex_annual_ccs = Param(
            domain=Reals, initialize=init_unit_capex_ccs_annualized, mutable=True
        )
        b_tec.para_fix_capex_annual_ccs = Param(
            domain=Reals,
            initialize=annualization_factor * economics.CAPEX_zeta,
            mutable=True,
        )

        def calculate_max_capex_ccs():
            max_capex = (
                self.ccs_data["size_max"] * b_tec.para_unit_capex_annual_ccs
                + b_tec.para_fix_capex_annual_ccs
            )
            return (0, max_capex)

        b_tec.var_capex_aux_ccs = Var(bounds=calculate_max_capex_ccs())

        # capex unit commitment constraint
        self.big_m_transformation_required = 1
        s_indicators = range(0, 2)

        def init_installation(dis, ind):
            if ind == 0:  # tech not installed
                dis.const_capex_aux_ccs = Constraint(expr=b_tec.var_capex_aux_ccs == 0)
                dis.const_not_installed_ccs = Constraint(expr=b_tec.var_size_ccs == 0)
            else:  # tech installed
                dis.const_capex_aux_ccs = Constraint(
                    expr=b_tec.var_size_ccs * b_tec.para_unit_capex_annual_ccs
                    + b_tec.para_fix_capex_annual_ccs
                    == b_tec.var_capex_aux_ccs
                )
                dis.const_installed_ccs_sizelim_min = Constraint(
                    expr=b_tec.var_size_ccs >= b_tec.para_size_min_ccs
                )
                dis.const_installed_ccs_sizelim_max = Constraint(
                    expr=b_tec.var_size_ccs <= b_tec.para_size_max_ccs
                )

        b_tec.dis_installation_ccs = Disjunct(s_indicators, rule=init_installation)

        def bind_disjunctions(dis):
            return [b_tec.dis_installation_ccs[i] for i in s_indicators]

        b_tec.disjunction_installation_ccs = Disjunction(rule=bind_disjunctions)

        # CAPEX
        b_tec.var_capex_ccs = Var()
        b_tec.const_capex_ccs = Constraint(
            expr=b_tec.var_capex_ccs == b_tec.var_capex_aux_ccs
        )

        # FIXED OPEX
        b_tec.para_opex_fixed_ccs = Param(
            domain=Reals, initialize=economics.OPEX_fixed, mutable=True
        )
        b_tec.var_opex_fixed_ccs = Var()
        b_tec.const_opex_fixed_ccs = Constraint(
            expr=b_tec.var_capex_aux_ccs * b_tec.para_opex_fixed_ccs
            == b_tec.var_opex_fixed_ccs
        )

        # VARIABLE OPEX
        set_t = self.set_t_full
        b_tec.para_opex_variable_ccs = Param(
            domain=Reals, initialize=economics.OPEX_variable, mutable=True
        )
        b_tec.var_opex_variable_ccs = Var(set_t)

        def init_opex_variable_ccs(const, t):
            return (
                b_tec.var_output_ccs[t, b_tec.set_output_carriers_ccs[1]]
                * b_tec.para_opex_variable_ccs
                == b_tec.var_opex_variable_ccs[t]
            )

        b_tec.const_opex_variable_ccs = Constraint(set_t, rule=init_opex_variable_ccs)

        return b_tec

    def _define_auxiliary_vars(self, b_tec, data):
        """
        Defines auxiliary variables, that are required for the modelling of clustered data
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

            b_tec.var_input_aux = Var(
                set_t_clustered,
                b_tec.set_input_carriers,
                within=NonNegativeReals,
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

        b_tec.var_output_aux = Var(
            set_t_clustered,
            b_tec.set_output_carriers,
            within=NonNegativeReals,
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

    def _define_dynamics(self, b_tec):
        """
        Selects the dynamic constraints that are required based on the technology dynamic performance parameters or the
        performance function type.
        :param b_tec:
        :return:
        """
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

        return b_tec

    def _dynamics_SUSD_logic(self, b_tec):
        """
        Adds the startup and shutdown logic to the technology model and constrains the maximum number of startups.

        Based on Equations 4-5 in Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden power system
        inflexibilities imposed by traditional unit commitment formulations. Applied Energy, 191, 223–238.
        https://doi.org/10.1016/J.APENERGY.2017.01.089

        """
        # New variables
        b_tec.var_x = Var(self.set_t_full, domain=NonNegativeReals, bounds=(0, 1))
        b_tec.var_y = Var(self.set_t_full, domain=NonNegativeReals, bounds=(0, 1))
        b_tec.var_z = Var(self.set_t_full, domain=NonNegativeReals, bounds=(0, 1))

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
                return Constraint.Skip
            else:
                return (
                    b_tec.var_x[t] - b_tec.var_x[t - 1]
                    == b_tec.var_y[t] - b_tec.var_z[t]
                )

        b_tec.const_SUSD_logic1 = Constraint(self.set_t_full, rule=init_SUSD_logic1)

        def init_SUSD_logic2(const, t):
            if t >= min_uptime:
                return b_tec.var_y[t - min_uptime + 1] <= b_tec.var_x[t]
            else:
                return (
                    b_tec.var_y[len(self.set_t_full) + (t - min_uptime + 1)]
                    <= b_tec.var_x[t]
                )

        b_tec.const_SUSD_logic2 = Constraint(self.set_t_full, rule=init_SUSD_logic2)

        def init_SUSD_logic3(const, t):
            if t >= min_downtime:
                return b_tec.var_z[t - min_downtime + 1] <= 1 - b_tec.var_x[t]
            else:
                return (
                    b_tec.var_z[len(self.set_t_full) + (t - min_downtime + 1)]
                    <= 1 - b_tec.var_x[t]
                )

        b_tec.const_SUSD_logic3 = Constraint(self.set_t_full, rule=init_SUSD_logic3)

        # Constrain number of startups
        if not max_startups == -1:

            def init_max_startups(const):
                return sum(b_tec.var_y[t] for t in self.set_t_full) <= max_startups

            b_tec.const_max_startups = Constraint(rule=init_max_startups)

        return b_tec

    def _dynamics_fast_SUSD(self, b_tec):
        """
        Adds startup and shutdown load constraints to the model.

        Based on Equations 9-11 and 13 in Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden power
        system inflexibilities imposed by traditional unit commitment formulations. Applied Energy, 191, 223–238.
        https://doi.org/10.1016/J.APENERGY.2017.01.089

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
                dis.const_y_off = Constraint(expr=b_tec.var_y[t] == 0)

            else:  # tech in startup
                dis.const_y_on = Constraint(expr=b_tec.var_y[t] == 1)

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

                dis.const_SU_load_limit = Constraint(rule=init_SU_load_limit)

        b_tec.dis_SU_load = Disjunct(self.set_t_full, s_indicators, rule=init_SU_load)

        def bind_disjunctions_SU_load(dis, t):
            return [b_tec.dis_SU_load[t, i] for i in s_indicators]

        b_tec.disjunction_SU_load = Disjunction(
            self.set_t_full, rule=bind_disjunctions_SU_load
        )

        # SD load limit
        s_indicators = range(0, 2)

        def init_SD_load(dis, t, ind):
            if ind == 0:  # no shutdown (z=0)
                dis.const_z_off = Constraint(expr=b_tec.var_z[t] == 0)

            else:  # tech in shutdown
                dis.const_z_on = Constraint(expr=b_tec.var_z[t] == 1)

                def init_SD_load_limit(cons):
                    if t == 1:
                        return Constraint.Skip
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

                dis.const_SD_load_limit = Constraint(rule=init_SD_load_limit)

        b_tec.dis_SD_load = Disjunct(self.set_t_full, s_indicators, rule=init_SD_load)

        def bind_disjunctions_SD_load(dis, t):
            return [b_tec.dis_SD_load[t, i] for i in s_indicators]

        b_tec.disjunction_SD_load = Disjunction(
            self.set_t_full, rule=bind_disjunctions_SD_load
        )

        return b_tec
