from ..data_management import DataHandle


def determine_network_energy_consumption(network_data: dict):
    """
    Determines if there is network consumption for a network
    """
    # Todo: This can be further extended to check if node is connected to network
    network_energy_consumption = 0
    for netw in network_data:
        if network_data[netw].performance_data["energyconsumption"]:
            network_energy_consumption = 1

    return network_energy_consumption


def get_data_for_investment_period(
    data: DataHandle, investment_period: str, aggregation_type: str
):
    """
    Gets data from DataHandle for specific investement_period. Writes it to a dict.
    :param DataHandle data: data to use
    :param str investment_period: investment period
    :param str aggregation_type: aggregation type
    :return dict: data_period
    """
    data_period = {}
    data_period["topology"] = data.topology
    data_period["technology_data"] = data.technology_data[aggregation_type][
        investment_period
    ]
    data_period["time_series"] = data.time_series[aggregation_type].loc[
        :, investment_period
    ]
    data_period["network_data"] = data.network_data[aggregation_type][investment_period]
    data_period["energybalance_options"] = data.energybalance_options[investment_period]
    data_period["config"] = data.model_config

    return data_period


def get_data_for_node(data, node):
    """
    Gets data from a dict for specific node. Writes it to a dict.

    :param dict data: data to use
    :param str node: node
    :return dict: data_node
    """
    data_node = {}
    data_node["topology"] = data["topology"]
    data_node["technology_data"] = data["technology_data"][node]
    data_node["time_series"] = data["time_series"][node]
    data_node["network_data"] = data["network_data"]
    data_node["energybalance_options"] = data["energybalance_options"][node]
    data_node["config"] = data["config"]

    return data_node
