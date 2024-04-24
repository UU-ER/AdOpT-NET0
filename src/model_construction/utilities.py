def determine_network_energy_consumption(network_data):
    """
    Determines if there is network consumption for a network
    """
    # Todo: This can be further extended to check if node is connected to network
    network_energy_consumption = 0
    for netw in network_data:
        if not network_data[netw].energyconsumption["carrier"]:
            network_energy_consumption = 1

    return network_energy_consumption
