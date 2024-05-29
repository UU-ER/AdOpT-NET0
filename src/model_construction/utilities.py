def determine_network_energy_consumption(network_data: dict) -> bool:
    """
    Determines if there is network consumption for a network

    :param dict network_data: Network data
    :return: Get if network has energyconsumption or not
    :rtype: int
    """
    # Todo: This can be further extended to check if node is connected to network
    network_energy_consumption = False
    for netw in network_data:
        if network_data[netw].component_options.energyconsumption:
            network_energy_consumption = True

    return network_energy_consumption


def get_data_for_investment_period(
    data, investment_period: str, aggregation_type: str
) -> dict:
    """
    Gets data from DataHandle for specific investement_period. Writes it to a dict.

    :param data: data to use
    :param str investment_period: investment period
    :param str aggregation_type: aggregation type
    :return: data of respective investment period
    :rtype: dict
    """
    data_period = {}
    data_period["topology"] = data.topology
    data_period["technology_data"] = data.technology_data[investment_period]
    data_period["time_series"] = data.time_series[aggregation_type].loc[
        :, investment_period
    ]
    data_period["network_data"] = data.network_data[investment_period]
    data_period["energybalance_options"] = data.energybalance_options[investment_period]
    data_period["config"] = data.model_config
    if data.model_config["optimization"]["typicaldays"]["N"]["value"] != 0:
        data_period["k_means_specs"] = data.k_means_specs[investment_period]
        # data_period["averaged_specs"] = data.averaged_specs[investment_period]

    return data_period


def get_data_for_node(data: dict, node: str) -> dict:
    """
    Gets data from a dict for specific node. Writes it to a dict.

    :param dict data: data to use
    :param str node: node
    :return: data of respective node
    :rtype: dict
    """
    data_node = {}
    data_node["topology"] = data["topology"]
    data_node["technology_data"] = data["technology_data"][node]
    data_node["time_series"] = data["time_series"][node]
    data_node["network_data"] = data["network_data"]
    data_node["energybalance_options"] = data["energybalance_options"][node]
    data_node["config"] = data["config"]
    if data["config"]["optimization"]["typicaldays"]["N"]["value"] != 0:
        data_node["k_means_specs"] = data["k_means_specs"]
        # data_node["averaged_specs"] = data["averaged_specs"]

    return data_node
