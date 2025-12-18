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
    if data["config"]["performance"]["pressure"]["pressure_on"]["value"] == 1:
        data_node["compressor_data"] = data["compressor_data"][node]
    if data["config"]["optimization"]["typicaldays"]["N"]["value"] != 0:
        data_node["k_means_specs"] = data["k_means_specs"]
        # data_node["averaged_specs"] = data["averaged_specs"]

    data_node["hour_factors"] = data["hour_factors"]
    data_node["nr_timesteps_averaged"] = data["nr_timesteps_averaged"]

    return data_node
