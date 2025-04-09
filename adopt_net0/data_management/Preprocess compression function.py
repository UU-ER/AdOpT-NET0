# architeture to calculate all possible compression (as a function of mass flow rate)
# between al possible connection happening at a certain node
# Se fo each node we know which technology and networks can be present
# each technology/network will have some carriers possible as input or output
# so we are going to scan which ones will have the same combination of carrier for output/input
# and for all possible connection of output/input we look at the pressure difference
# if the pressure difference require compression, we call a function compression_energy
# if the pressure difference does not require compression, we give zero
def calculate_possible_compressions(self):
    connection_data = {}
    network_list_input = {}
    network_list_output = {}
    technology_input = {}
    technology_output = {}
    target_carriers = self.config["Performance"]["pressure"]["compressed_carrier"]

    for carrier_i in target_carriers:
        connection_data[carrier_i] = {}

        for node_i in self.topology["nodes"]:
            connection_data[carrier_i][node_i] = {}
            connection_data[carrier_i][node_i] = {"inputs": {}, "outputs": {}}

            network_list_input = []
            network_list_output = []
            technology_input = []
            technology_output = []

            # here actually we should look at connection.loc [node_i, NODE 2] =1
            for network_i in self.network_data:
                # here there is a matrix in network_topology
                if network_i["Performance"]["carrier"] == carrier_i:
                    if self.network_data.connection.index == 1:
                        # means that there is a network starting in this node
                        network_list_input[carrier_i].add_netw_to_list(
                            network_i, carrier_i, "Input"
                        )

                    if self.network_data.connection.columns == 1:
                        network_list_output[carrier_i].add_netw_to_list(
                            network_i, carrier_i, "Output"
                        )
                    # the function add_network_to_list should not only add the network
                    # but also it should already read the pressure information and add to the list/dictionary

            for technologies_i in self.technology_data:
                # first we look at the one that has hydrogen as input
                if technologies_i["Performance"]["input_carrier"] == ["carrier_i"]:
                    # as done it before we have a function that write the technology and their INPUT pressure
                    technology_input[carrier_i].add_tech_to_list(
                        technologies_i, carrier_i, "Input"
                    )

                if technologies_i["Performance"]["output_carrier"] == ["carrier_i"]:
                    # same as before, but with OUTPUT carrier and pressure
                    technology_output[carrier_i].add_tech_to_list(
                        technologies_i, carrier_i, "Output"
                    )

            connection_data[carrier_i][node_i]["inputs"]["pressure_level"][
                "networks"
            ] = network_list_input
            connection_data[carrier_i][node_i]["inputs"]["pressure_level"][
                "technologies"
            ] = technology_input
            connection_data[carrier_i][node_i]["outputs"]["pressure_level"][
                "networks"
            ] = network_list_output
            connection_data[carrier_i][node_i]["outputs"]["pressure_level"][
                "technologies"
            ] = technology_output

            # now we have four list
            # one for network that will have the carrier as input
            # one for network that will have the carrier as output to the node
            # one for technology that need the carrier for input
            # one for technology that have the carrier as output

            # We need to find all the possible combination of output_to_node -> input_from_node
            # Question: in this way we will only make possible the connection between components
            # The possibility of having a unique compressor for all is not there
            # what do we think about it?

            # all output → input combinations (by carrier)
            for output_component in (
                network_list_output[carrier_i] + technology_output[carrier_i]
            ):
                for input_component in (
                    network_list_input[carrier_i] + technology_input[carrier_i]
                ):

                    if output_component["pressure"] >= input_component["pressure"]:
                        compression_req.energy[input_component, output_component] = (
                            direct_connection(output_component, input_component)
                        )
                    else:
                        compression_req.energy[input_component, output_component] = (
                            calculate_energy(
                                output_component["pressure"],
                                input_component["pressure"],
                                carrier_i,
                            )
                        )


def add_tech_to_list(tech, carrier, str):
    if str == "Input":
        pressure = tech["Performance"]["pressure"][carrier]["inlet"]
    elif str == "Output":
        pressure = tech["Performance"]["pressure"][carrier]["outlet"]
    return tech, pressure


def add_netw_to_list(netw, carrier, str):
    if str == "Input":
        pressure = netw["Performance"]["pressure"][carrier]["inlet"]
    elif str == "Output":
        pressure = netw["Performance"]["pressure"][carrier]["outlet"]
    return netw, pressure
