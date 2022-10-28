from pyomo.environ import *
from src.technology_performance_fitting import fit_linear_performance


def add_technologies(b_node, model, data):
    def technology_block_rule(b_tec, tec):
        """" Adds all technologies at considered node
        - Common decision variables and constraints are added first
        - Technology specific variables are added second
        """

        # get options from data
        tec_data = data.technology_data[tec]
        capex_model = tec_data['Economics']['CAPEX_model']
        size_is_integer = tec_data['TechnologyPerf']['size_is_int']
        tec_type = tec_data['TechnologyPerf']['tec_type']
        performance = tec_data['TechnologyPerf']['performance']

        # PARAMETERS
        b_tec.para_size_min = Param(initialize=tec_data['TechnologyPerf']['size_min'])
        b_tec.para_size_max = Param(initialize=tec_data['TechnologyPerf']['size_max'])
        b_tec.para_unit_CAPEX = Param(initialize=tec_data['Economics']['unit_CAPEX_annual'])
        b_tec.para_OPEX_variable = Param(initialize=tec_data['Economics']['OPEX_variable'])
        b_tec.para_OPEX_fixed = Param(initialize=tec_data['Economics']['OPEX_fixed'])

        # SETS
        b_tec.set_input_carriers = Set(initialize=tec_data['TechnologyPerf']['input_carrier'])
        b_tec.set_output_carriers = Set(initialize=tec_data['TechnologyPerf']['output_carrier'])

        # DECISION VARIABLES
        # Input
        if b_tec.set_input_carriers in ['wind', 'solar']:
            b_tec.var_input = Param(model.set_t, b_tec.set_input_carriers)
        else:
            b_tec.var_input = Var(model.set_t, b_tec.set_input_carriers, within=NonNegativeReals)
        # Output
        b_tec.var_output = Var(model.set_t, b_tec.set_output_carriers, within=NonNegativeReals)
        # Size
        if size_is_integer:  # size
            b_tec.var_size = Var(within=NonNegativeIntegers, bounds=(b_tec.para_size_min, b_tec.para_size_max))
        else:
            b_tec.var_size = Var(within=NonNegativeReals, bounds=(b_tec.para_size_min, b_tec.para_size_max))
        # Capex/Opex
        b_tec.var_CAPEX = Var()  # capex
        b_tec.var_OPEX_variable = Var(model.set_t)  # variable opex
        b_tec.var_OPEX_fixed = Var()  # fixed opex

        # GENERAL CONSTRAINTS
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
        def calculate_OPEX_variable(b_tec, t):
            return sum(b_tec.var_output[t, car] for car in b_tec.set_output_carriers) * b_tec.para_OPEX_variable == \
                   b_tec.var_OPEX_variable[t]
        b_tec.const_OPEX_variable = Constraint(model.set_t, rule=calculate_OPEX_variable)
        # Size constraint
        def calculate_output_constraint(b_tec, t, input):
            return b_tec.var_input[t, input] <= b_tec.var_size
        b_tec.const_size = Constraint(model.set_t, b_tec.set_input_carriers, rule=calculate_output_constraint)

        # TECHNOLOGY SPECIFIC PARTS
        # TODO: Include ambient performance factor for each technology type
        if tec_type == 1:  # Renewable technology with parameters as input
            print('Not coded yet!')
        elif tec_type == 2: # n inputs -> 1 output, fuel substitution, linear performance
            try:
                alpha = fit_linear_performance(len(b_tec.set_input_carriers), 1,
                                               performance, tec_type)

                # def calculate_input_output(t, car):
                #     return b_tec.var_output[t] = sum()
                # b_tec.const_input_output = Constraint(model.set_t, b_tec.set_input_carriers,
                #                                       rule=calculate_input_output)
            except:
                print('Not coded yet!')
        # elif tectype == 3:  # n inputs -> 1 output, fixed input ratio, linear performance
        # elif tectype == 4:  # 1 input -> n outputs, output flexible, linear performance
        # elif tectype == 5:  # 1 input -> n outputs, fixed output ratio, linear performance
    #
    #         def inout1(b_tec, t, input, output):
    #             return b_tec.d_output[t, output] == data.alpha[tec] * b_tec.d_input[t, input]
    #
    #         b_tec.c_performance = Constraint(m.s_t, b_tec.s_tecin, b_tec.s_tecout, rule=inout1)
    #
    b_node.tech_blocks = Block(b_node.s_techs, rule=technology_block_rule)
    return b_node
