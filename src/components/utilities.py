from pyomo.environ import *

def annualize(r, t):
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
    return annualization_factor


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