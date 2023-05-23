from pyomo.environ import *
from pyomo.gdp import *
import src.global_variables as global_variables
import src.model_construction as mc


def constraints_tec_hydro_open(b_tec, tec_data, energyhub):
    """
    Adds constraints to technology blocks for tec_type Hydro_Open, resembling a pumped hydro plant with
    additional natural inflows (defined in climate data)

    The performance
    functions are fitted in ``src.model_construction.technology_performance_fitting``.
    Note that this technology only works for one carrier, and thus the carrier index is dropped in the below notation.

    **Parameter declarations:**

    - :math:`{\\eta}_{in}`: Charging efficiency

    - :math:`{\\eta}_{out}`: Discharging efficiency

    - :math:`{\\lambda}`: Self-Discharging coefficient

    - :math:`Input_{max}`: Maximal charging capacity in one time-slice

    - :math:`Output_{max}`: Maximal discharging capacity in one time-slice

    - :math:`Natural_Inflow{t}`: Natural water inflow in time slice (can be negative, i.e. being an outflow)

    **Variable declarations:**

    - Storage level in :math:`t`: :math:`E_t`

    - Charging in :math:`t`: :math:`Input_{t}`

    - Discharging in :math:`t`: :math:`Output_{t}`

    **Constraint declarations:**

    - Maximal charging and discharging:

      .. math::
        Input_{t} \leq Input_{max}

      .. math::
        Output_{t} \leq Output_{max}

    - Size constraint:

      .. math::
        E_{t} \leq S

    - Storage level calculation:

      .. math::
        E_{t} = E_{t-1} * (1 - \\lambda) + {\\eta}_{in} * Input_{t} - 1 / {\\eta}_{out} * Output_{t} + Natural_Inflow_{t}

    - If ``allow_only_one_direction == 1``, then only input or output can be unequal to zero in each respective time
      step (otherwise, simultanous charging and discharging can lead to unwanted 'waste' of energy/material).

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    model = energyhub.model

    # DATA OF TECHNOLOGY
    performance_data = tec_data.performance_data
    coeff = tec_data.fitted_performance.coefficients
    modelled_with_full_res = tec_data.modelled_with_full_res

    # Full resolution
    input = b_tec.var_input
    output = b_tec.var_output
    set_t = model.set_t_full

    if 'allow_only_one_direction' in performance_data:
        allow_only_one_direction = performance_data['allow_only_one_direction']
    else:
        allow_only_one_direction = 0

    can_pump = performance_data['can_pump']
    if performance_data['maximum_discharge_time_discrete']:
        hydro_maximum_discharge = coeff['hydro_maximum_discharge']

    nr_timesteps_averaged = global_variables.averaged_data_specs.nr_timesteps_averaged

    # Additional decision variables
    b_tec.var_storage_level = Var(set_t, b_tec.set_input_carriers,
                                  domain=NonNegativeReals,
                                  bounds=(b_tec.para_size_min, b_tec.para_size_max))
    b_tec.var_spilling = Var(set_t,
                                  domain=NonNegativeReals,
                                  bounds=(b_tec.para_size_min, b_tec.para_size_max))

    # Abdditional parameters
    eta_in = coeff['eta_in']
    eta_out = coeff['eta_out']
    eta_lambda = coeff['lambda']
    charge_max = coeff['charge_max']
    discharge_max = coeff['discharge_max']
    hydro_natural_inflow = coeff['hydro_inflow']
    spilling_max = coeff['spilling_max']


    # Size constraint
    def init_size_constraint(const, t, car):
        return b_tec.var_storage_level[t, car] <= b_tec.var_size
    b_tec.const_size = Constraint(set_t, b_tec.set_input_carriers, rule=init_size_constraint)

    # Storage level calculation
    def init_storage_level(const, t, car):
        if t == 1: # couple first and last time interval
            return b_tec.var_storage_level[t, car] == \
                  b_tec.var_storage_level[max(set_t), car] * (1 - eta_lambda) ** nr_timesteps_averaged + \
                  (eta_in * input[t, car] - 1 / eta_out * output[t, car] - b_tec.var_spilling[t]) * \
                  sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged)) + \
                  hydro_natural_inflow[t-1]
        else: # all other time intervals
            return b_tec.var_storage_level[t, car] == \
                b_tec.var_storage_level[t-1, car] * (1 - eta_lambda) ** nr_timesteps_averaged + \
                (eta_in * input[t, car] - 1/eta_out * output[t, car] - b_tec.var_spilling[t]) * \
                sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged)) + \
                hydro_natural_inflow[t-1]
    b_tec.const_storage_level = Constraint(set_t, b_tec.set_input_carriers, rule=init_storage_level)

    if not can_pump:
        def init_input_zero(const, t, car):
            return input[t, car] == 0
        b_tec.const_input_zero = Constraint(set_t, b_tec.set_input_carriers, rule=init_input_zero)

    # This makes sure that only either input or output is larger zero.
    if allow_only_one_direction == 1:
        global_variables.big_m_transformation_required = 1
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # input only
                def init_output_to_zero(const, car_input):
                    return output[t, car_input] == 0
                dis.const_output_to_zero = Constraint(b_tec.set_input_carriers, rule=init_output_to_zero)

            elif ind == 1:  # output only
                def init_input_to_zero(const, car_input):
                    return input[t, car_input] == 0
                dis.const_input_to_zero = Constraint(b_tec.set_input_carriers, rule=init_input_to_zero)

        b_tec.dis_input_output = Disjunct(set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # Maximal charging and discharging rates
    def init_maximal_charge(const,t,car):
        return input[t, car] <= charge_max * b_tec.var_size
    b_tec.const_max_charge = Constraint(set_t, b_tec.set_input_carriers, rule=init_maximal_charge)

    def init_maximal_discharge(const,t,car):
        return output[t, car] <= discharge_max * b_tec.var_size
    b_tec.const_max_discharge = Constraint(set_t, b_tec.set_input_carriers, rule=init_maximal_discharge)

    if performance_data['maximum_discharge_time_discrete']:
        def init_maximal_discharge2(const, t, car):
            return output[t, car] <= hydro_maximum_discharge[t-1]
        b_tec.const_max_discharge2 = Constraint(set_t, b_tec.set_input_carriers, rule=init_maximal_discharge2)

    # Maximum spilling
    def init_maximal_spilling(const,t):
        return b_tec.var_spilling[t] <= spilling_max * b_tec.var_size
    b_tec.const_max_spilling = Constraint(set_t, rule=init_maximal_spilling)

    return b_tec