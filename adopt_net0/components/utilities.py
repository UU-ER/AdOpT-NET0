import time
import pyomo.environ as pyo

import logging

log = logging.getLogger(__name__)


def annualize(r: float, t: int, year_fraction: float):
    """
    Calculates annualization factor

    :param float r: interest rate
    :param int t: lifetime
    :param flaot year_fraction: fraction of year modelled
    :return: annualization factor
    :rtype: float
    """
    if r == 0:
        annualization_factor = 1 / t
    else:
        annualization_factor = r / (1 - (1 / (1 + r) ** t))
    return annualization_factor * year_fraction


def set_discount_rate(config: dict, economics):
    """
    Sets the discount rate to either global or technology value

    :param dict config: dict containing model information
    :param economics: Economics class
    :return: CAPEX model
    :rtype: float
    """
    if not config["economic"]["global_discountrate"]["value"] == -1:
        discount_rate = config["economic"]["global_discountrate"]["value"]
    else:
        discount_rate = economics.discount_rate
    return discount_rate


def link_full_resolution_to_clustered(
    var_clustered, var_full, set_t_full, sequence, *other_sets
):
    """
    Links two variables (clustered and full)

    :param var_clustered: pyomo variable with clustered resolution
    :param var_full: pyomo variable with full resolution
    :param set_t_full: pyomo set containing timesteps
    :param sequence: order of typical days
    :param other_sets: other pyomo sets that variables are indexed by
    :return: pyomo constraint linking var_clustered and var_full
    """
    if not other_sets:

        def init_link_full_resolution(const, t):
            return var_full[t] == var_clustered[sequence[t - 1]]

        constraint = pyo.Constraint(set_t_full, rule=init_link_full_resolution)
    elif len(other_sets) == 1:
        set1 = other_sets[0]

        def init_link_full_resolution(const, t, set1):
            return var_full[t, set1] == var_clustered[sequence[t - 1], set1]

        constraint = pyo.Constraint(set_t_full, set1, rule=init_link_full_resolution)
    elif len(other_sets) == 2:
        set1 = other_sets[0]
        set2 = other_sets[1]

        def init_link_full_resolution(const, t, set1, set2):
            return var_full[t, set1, set2] == var_clustered[sequence[t - 1], set1, set2]

        constraint = pyo.Constraint(
            set_t_full, set1, set2, rule=init_link_full_resolution
        )

    return constraint


def perform_disjunct_relaxation(model_block, method: str = "gdp.bigm"):
    """
    Performs big-m transformation for respective component

    :param component: pyomo component
    :param str method: method to make transformation with.
    :return: component
    """
    log_msg = "\t\t\t" + method + " Transformation..."
    log.info(log_msg)
    start = time.time()
    xfrm = pyo.TransformationFactory(method)
    xfrm.apply_to(model_block)
    log_msg = (
        "\t\t\t"
        + method
        + " Transformation completed in "
        + str(round(time.time() - start))
        + " s"
    )

    log.info(log_msg)
    return model_block


def read_dict_value(dict: dict, key: str) -> str | int | float:
    """
    Reads a value from a dictonary or sets it to 1 if key is not in dict

    :param dict: dict
    :param key: dict key to check for
    :return:
    :rtype: str | int | float
    """
    dict_value = 1

    if dict:
        if key in dict:
            dict_value = dict[key]

    return dict_value


def determine_variable_scaling(model, model_block, f: dict, f_global):
    """
    Scale model block variables

    :param model: pyomo model
    :param model_block: pyomo model block
    :param dict f: individual scaling factors
    :param f_global: global scaling factors
    :return: model_block
    """
    for var in model_block.component_objects(pyo.Var, active=True):
        var_name = var.name.split(".")[-1]

        # check if var is integer
        var_is_integer = any([var[index].is_integer() for index in var.keys()])

        if not var_is_integer:
            # Determine global scaling factor
            global_scaling_factor = f_global["energy_vars"]["value"] * read_dict_value(
                f, var_name
            )
            if "capex" in var_name or "opex" in var_name:
                global_scaling_factor = (
                    global_scaling_factor * f_global["cost_vars"]["value"]
                )
            model.scaling_factor[var] = global_scaling_factor

    return model


def determine_constraint_scaling(model, model_block, f: dict, f_global):
    """
    Scale model block constraints

    :param model: pyomo model
    :param model_block: pyomo model block
    :param dict f: individual scaling factors
    :param f_global: global scaling factors
    :return: model_block
    """
    for constr in model_block.component_objects(pyo.Constraint, active=True):
        const_name = constr.name.split(".")[-1]

        # Determine global scaling factor
        global_scaling_factor = (
            read_dict_value(f, const_name) * f_global["energy_vars"]["value"]
        )
        if "capex" in const_name or "opex" in const_name:
            global_scaling_factor = (
                global_scaling_factor * f_global["cost_vars"]["value"]
            )

        if not const_name.endswith("xor"):
            model.scaling_factor[constr] = global_scaling_factor

    return model


def get_attribute_from_dict(d: dict, key: str, value_other) -> str | float:
    """
    Takes an attribute from dict, if it doesnt exist replace with value_other

    :param dict d: dictonary
    :param str key: key to look for in dictonary
    :param value_other: if key is not in dict, return this value
    """
    if key in d:
        return d[key]
    else:
        return value_other
