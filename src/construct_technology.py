from pyomo.environ import *
from pyomo.environ import units as u
from pyomo.gdp import *
import warnings
import numbers
from src.generic_technology_constraints import *

def add_technologies(nodename, b_node, model, data):
    def technology_block_rule(b_tec, tec):
        """" Adds all technologies at considered node
        - Common decision variables and constraints are added first
        - Technology specific variables are added second
        """

        # region Get options from data
        tec_data = data.technology_data[nodename][tec]
        tec_type = tec_data['TechnologyPerf']['tec_type']
        capex_model = tec_data['Economics']['CAPEX_model']
        size_is_integer = tec_data['TechnologyPerf']['size_is_int']
        # endregion

        # region PARAMETERS

        # We need this shit because python does not accept single value in its build-in min function
        if isinstance(tec_data['TechnologyPerf']['size_min'], numbers.Number):
            size_min = tec_data['TechnologyPerf']['size_min']
        else:
            size_min = min(tec_data['TechnologyPerf']['size_min'])
        if isinstance(tec_data['TechnologyPerf']['size_max'], numbers.Number):
            size_max = tec_data['TechnologyPerf']['size_max']
        else:
            size_max = max(tec_data['TechnologyPerf']['size_max'])

        if size_is_integer:
            unit_size = u.dimensionless
        else:
            unit_size = u.MW
        b_tec.para_size_min = Param(domain=NonNegativeReals, initialize=size_min, units=unit_size)
        b_tec.para_size_max = Param(domain=NonNegativeReals, initialize=size_max, units=unit_size)
        b_tec.para_output_max = Param(domain=NonNegativeReals, initialize=size_max, units=u.MW)
        b_tec.para_unit_CAPEX = Param(domain=Reals, initialize=tec_data['Economics']['unit_CAPEX_annual'],
                                      units=u.EUR/unit_size)
        b_tec.para_OPEX_variable = Param(domain=Reals, initialize=tec_data['Economics']['OPEX_variable'],
                                         units=u.EUR/u.MWh)
        b_tec.para_OPEX_fixed = Param(domain=Reals, initialize=tec_data['Economics']['OPEX_fixed'],
                                      units=u.EUR/u.EUR)
        # endregion

        # region SETS
        b_tec.set_input_carriers = Set(initialize=tec_data['TechnologyPerf']['input_carrier'])
        b_tec.set_output_carriers = Set(initialize=tec_data['TechnologyPerf']['output_carrier'])
        # endregion

        # region DECISION VARIABLES
        # Input
        if not tec_type == 1:
            b_tec.var_input = Var(model.set_t, b_tec.set_input_carriers, within=NonNegativeReals,
                                  bounds=(b_tec.para_size_min, b_tec.para_size_max), units=u.MW)
        # Output
        b_tec.var_output = Var(model.set_t, b_tec.set_output_carriers, within=NonNegativeReals,
                               bounds=(0, b_tec.para_output_max), units=u.MW)
        # Size
        if size_is_integer:  # size
            b_tec.var_size = Var(within=NonNegativeIntegers, bounds=(b_tec.para_size_min, b_tec.para_size_max))
        else:
            b_tec.var_size = Var(within=NonNegativeReals, bounds=(b_tec.para_size_min, b_tec.para_size_max),
                                 units=u.MW)
        # Capex/Opex
        b_tec.var_CAPEX = Var(units=u.EUR)  # capex
        b_tec.var_OPEX_variable = Var(model.set_t, units=u.EUR)  # variable opex
        b_tec.var_OPEX_fixed = Var(units=u.EUR)  # fixed opex
        # endregion

        # region GENERAL CONSTRAINTS
        # Capex
        if capex_model == 1:
            b_tec.const_CAPEX = Constraint(expr=b_tec.var_size * b_tec.para_unit_CAPEX == b_tec.var_CAPEX)
        elif capex_model == 2:
            # TODO Implement link between bps and data
            b_tec.const_CAPEX = Piecewise(b_tec.var_CAPEX, b_tec.var_size,
                                          pw_pts=bp_x,
                                          pw_constr_type='EQ',
                                          f_rule=bp_y,
                                          pw_repn='SOS2')
        # fixed Opex
        b_tec.const_OPEX_fixed = Constraint(expr=b_tec.var_CAPEX * b_tec.para_OPEX_fixed == b_tec.var_OPEX_fixed)

        # variable Opex
        def calculate_OPEX_variable(con, t):
            return sum(b_tec.var_output[t, car] for car in b_tec.set_output_carriers) * b_tec.para_OPEX_variable == \
                   b_tec.var_OPEX_variable[t]
        b_tec.const_OPEX_variable = Constraint(model.set_t, rule=calculate_OPEX_variable)

        # Size constraint
        if tec_type == 1: # in terms of output
            def calculate_output_constraint(con, t):
                return sum(b_tec.var_output[t, c_output] for c_output in b_tec.set_output_carriers) \
                       <= b_tec.var_size
            b_tec.const_size = Constraint(model.set_t, rule=calculate_output_constraint)
        elif tec_type == 6: # This is defined in the generic technology constraints
            pass
        else: # in terms of input
            def calculate_output_constraint(con, t):
                return sum(b_tec.var_input[t, c_input] for c_input in b_tec.set_input_carriers) \
                       <= b_tec.var_size
            b_tec.const_size = Constraint(model.set_t, rule=calculate_output_constraint)

        # endregion

        # region TECHNOLOGY TYPES
        if tec_type == 1: # Renewable technology with cap_factor as input
            b_tec = constraints_tec_type_1(model, b_tec,tec_data)

        elif tec_type == 2: # n inputs -> n output, fuel and output substitution
            b_tec = constraints_tec_type_2(model, b_tec, tec_data)

        elif tec_type == 3: # n inputs -> n output, fuel and output substitution
            b_tec = constraints_tec_type_3(model, b_tec, tec_data)

        # elif tectype == 4:  # 1 input -> n outputs, output flexible, linear performance
        # elif tectype == 5:  # 1 input -> n outputs, fixed output ratio, linear performance

        elif tec_type == 6: # Storage technology (1 input -> 1 output)
            b_tec = constraints_tec_type_6(model, b_tec, tec_data)

    b_node.tech_blocks = Block(b_node.s_techs, rule=technology_block_rule)
    return b_node
