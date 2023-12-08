import time

from pyomo.core import TransformationFactory
from pyomo.environ import *

def annualize(r, t, year_fraction):
    """
    Calculates annualization factor
    :param r: interest rate
    :param t: lifetime
    :return: annualization factor
    """
    if r==0:
        annualization_factor = 1/t
    else:
        annualization_factor = r / (1 - (1 / (1 + r) ** t))
    return annualization_factor * year_fraction


def set_discount_rate(configuration, economics):
    if not configuration.economic.global_discountrate == -1:
        discount_rate = configuration.economic.global_discountrate
    else:
        discount_rate = economics.discount_rate
    return discount_rate


def link_full_resolution_to_clustered(var_clustered, var_full, set_t, sequence, *other_sets):
    """
    Links two variables (clustered and full)
    """
    if not other_sets:
        def init_link_full_resolution(const, t):
            return var_full[t] \
                   == var_clustered[sequence[t - 1]]
        constraint = Constraint(set_t, rule=init_link_full_resolution)
    elif len(other_sets) == 1:
        set1 = other_sets[0]
        def init_link_full_resolution(const, t, set1):
            return var_full[t, set1] \
                   == var_clustered[sequence[t - 1], set1]
        constraint = Constraint(set_t, set1, rule=init_link_full_resolution)
    elif len(other_sets) == 2:
        set1 = other_sets[0]
        set2 = other_sets[1]
        def init_link_full_resolution(const, t, set1, set2):
            return var_full[t, set1, set2] \
                   == var_clustered[sequence[t - 1], set1, set2]
        constraint = Constraint(set_t, set1, set2, rule=init_link_full_resolution)

    return constraint

class Economics:
    """
    Class to manage economic data of technologies and networks
    """
    def __init__(self, economics):
        self.capex_model = economics['CAPEX_model']
        self.capex_data = {}
        if 'unit_CAPEX' in economics:
            self.capex_data['unit_capex'] = economics['unit_CAPEX']
        if 'fix_CAPEX' in economics:
                self.capex_data['fix_capex'] = economics['fix_CAPEX']
        if 'piecewise_CAPEX' in economics:
            self.capex_data['piecewise_capex'] = economics['piecewise_CAPEX']
        if 'gamma1' in economics:
            self.capex_data['gamma1'] = economics['gamma1']
            self.capex_data['gamma2'] = economics['gamma2']
            self.capex_data['gamma3'] = economics['gamma3']
            self.capex_data['gamma4'] = economics['gamma4']
        self.opex_variable = economics['OPEX_variable']
        self.opex_fixed = economics['OPEX_fixed']
        self.discount_rate = economics['discount_rate']
        self.lifetime = economics['lifetime']
        self.decommission_cost = economics['decommission_cost']


def perform_disjunct_relaxation(model_block, method = 'gdp.bigm'):
    """
    Performs big-m transformation for respective component
    :param component: component
    :return: component
    """
    print('\t\t'+ method + ' Transformation...')
    start = time.time()
    xfrm = TransformationFactory(method)
    xfrm.apply_to(model_block)
    print('\t\t'+ method + ' Transformation completed in ' + str(round(time.time() - start)) + ' s')
    return model_block


def read_dict_value(dict, key):
    """
    Reads a value from a dictonary
    :param dict: dictonary
    :param key: dict key
    :return:
    """
    dict_value = 1

    if dict:
        if key in dict:
            dict_value = dict[key]

    return dict_value


def determine_variable_scaling(model, model_block, f, f_global):
    """
    Scale model block variables

    :param model_block: pyomo model block
    :param f: individual scaling factors
    :param f_global: global scaling factors
    :return: model_block
    """
    for var in model_block.component_objects(Var, active=True):
        var_name = var.name.split('.')[-1]

        # check if var is integer
        var_is_integer = any([var[index].is_integer() for index in var.index_set()])

        if not var_is_integer:
            # Determine global scaling factor
            global_scaling_factor = f_global.energy_vars * read_dict_value(f, var_name)
            if 'capex' in var_name or 'opex' in var_name:
                global_scaling_factor = global_scaling_factor * f_global.cost_vars
            model.scaling_factor[var] = global_scaling_factor

    return model


def determine_constraint_scaling(model, model_block, f, f_global):
    """
    Scale model block variables

    :param model_block: pyomo model block
    :param f: individual scaling factors
    :param f_global: global scaling factors
    :return: model_block
    """
    for constr in model_block.component_objects(Constraint, active=True):
        const_name = constr.name.split('.')[-1]

        # Determine global scaling factor
        global_scaling_factor = read_dict_value(f, const_name) * f_global.energy_vars
        if 'capex' in const_name or 'opex' in const_name:
            global_scaling_factor = global_scaling_factor * f_global.cost_vars

        if not const_name.endswith('xor'):
            model.scaling_factor[constr] = global_scaling_factor

    return model