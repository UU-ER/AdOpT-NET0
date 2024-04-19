from datetime import datetime
import os
from pathlib import Path


def create_unique_folder_name(path, name):
    """
    Creates a unique folder name, in case the specified name already exists in path
    :param path: path to check
    :param name: folder name
    :return:
    """
    folder_path = Path.joinpath(path, name)
    counter = 1
    while folder_path.is_dir():
        folder_path = folder_path.with_name(
            f"{folder_path.stem}_{counter}{folder_path.suffix}"
        )
        counter += 1
    return folder_path


def create_save_folder(save_path):
    """
    Creates a new folder at save_path

    :param str save_path: path to create folder at
    :return:
    """
    os.makedirs(save_path)


def calculate_tec_cost(energyhub):
    """
    Calculates costs related to technologies
    """
    nr_timesteps_averaged = (
        energyhub.model_information.averaged_data_specs.nr_timesteps_averaged
    )
    model = energyhub.model
    set_t = model.set_t_full
    tec_capex = sum(
        sum(
            model.node_blocks[node].tech_blocks_active[tec].var_capex.value
            for tec in model.node_blocks[node].set_tecsAtNode
        )
        for node in model.set_nodes
    )
    tec_opex_variable = sum(
        sum(
            sum(
                model.node_blocks[node]
                .tech_blocks_active[tec]
                .var_opex_variable[t]
                .value
                * nr_timesteps_averaged
                for tec in model.node_blocks[node].set_tecsAtNode
            )
            for t in set_t
        )
        for node in model.set_nodes
    )
    tec_opex_fixed = sum(
        sum(
            model.node_blocks[node].tech_blocks_active[tec].var_opex_fixed.value
            for tec in model.node_blocks[node].set_tecsAtNode
        )
        for node in model.set_nodes
    )
    tec_costs = tec_capex + tec_opex_variable + tec_opex_fixed
    return tec_costs


def calculate_import_costs(energyhub):
    """
    Calculates costs related to imports
    """
    nr_timesteps_averaged = (
        energyhub.model_information.averaged_data_specs.nr_timesteps_averaged
    )
    model = energyhub.model
    set_t = model.set_t_full
    import_costs = sum(
        sum(
            sum(
                model.node_blocks[node].var_import_flow[t, car].value
                * model.node_blocks[node].para_import_price[t, car].value
                * nr_timesteps_averaged
                for car in model.node_blocks[node].set_carriers
            )
            for t in set_t
        )
        for node in model.set_nodes
    )
    return import_costs


def calculate_export_revenues(energyhub):
    """
    Calculates costs related to exports
    """
    nr_timesteps_averaged = (
        energyhub.model_information.averaged_data_specs.nr_timesteps_averaged
    )
    model = energyhub.model
    set_t = model.set_t_full
    export_revenues = sum(
        sum(
            sum(
                model.node_blocks[node].var_export_flow[t, car].value
                * model.node_blocks[node].para_export_price[t, car].value
                * nr_timesteps_averaged
                for car in model.node_blocks[node].set_carriers
            )
            for t in set_t
        )
        for node in model.set_nodes
    )
    return export_revenues


def calculate_violation_cost(energyhub):
    """
    Calculates costs related to energy balance violations
    """
    model = energyhub.model
    if hasattr(model, "var_violation_cost"):
        violation_cost = model.var_violation_cost.value
    else:
        violation_cost = 0
    return violation_cost


def calculate_net_emissions_from_tecs(energyhub):
    """
    Calculates costs related to emissions from technologies
    """
    nr_timesteps_averaged = (
        energyhub.model_information.averaged_data_specs.nr_timesteps_averaged
    )
    model = energyhub.model
    set_t = model.set_t_full
    net_emissions_from_tecs = sum(
        sum(
            sum(
                model.node_blocks[node]
                .tech_blocks_active[tec]
                .var_tec_emissions_pos[t]
                .value
                * nr_timesteps_averaged
                for t in set_t
            )
            for tec in model.node_blocks[node].set_tecsAtNode
        )
        for node in model.set_nodes
    ) - sum(
        sum(
            sum(
                model.node_blocks[node]
                .tech_blocks_active[tec]
                .var_tec_emissions_neg[t]
                .value
                * nr_timesteps_averaged
                for t in set_t
            )
            for tec in model.node_blocks[node].set_tecsAtNode
        )
        for node in model.set_nodes
    )
    return net_emissions_from_tecs


def calculate_net_emissions_from_carriers(energyhub):
    """
    Calculates costs related to emissions from carriers
    """
    nr_timesteps_averaged = (
        energyhub.model_information.averaged_data_specs.nr_timesteps_averaged
    )
    model = energyhub.model
    set_t = model.set_t_full
    net_emissions_from_cars = sum(
        sum(
            model.node_blocks[node].var_car_emissions_pos[t].value
            * nr_timesteps_averaged
            for t in set_t
        )
        for node in model.set_nodes
    ) - sum(
        sum(
            model.node_blocks[node].var_car_emissions_neg[t].value
            * nr_timesteps_averaged
            for t in set_t
        )
        for node in model.set_nodes
    )
    return net_emissions_from_cars


def calculate_net_emissions_from_netw(energyhub):
    """
    Calculates costs related to emissions from networks
    """
    nr_timesteps_averaged = (
        energyhub.model_information.averaged_data_specs.nr_timesteps_averaged
    )
    model = energyhub.model
    set_t = model.set_t_full
    if not energyhub.configuration.energybalance.copperplate:
        net_emissions_from_netw = sum(
            sum(
                model.network_block[netw].var_netw_emissions_pos[t].value
                * nr_timesteps_averaged
                for t in set_t
            )
            for netw in model.set_networks
        )
    else:
        net_emissions_from_netw = 0
    return net_emissions_from_netw


def get_time_stage(energyhub):
    """
    Gets time stage
    """
    if energyhub.configuration.optimization.timestaging:
        time_stage = energyhub.model_information.averaged_data_specs.stage + 1
    else:
        time_stage = 0
    return time_stage
