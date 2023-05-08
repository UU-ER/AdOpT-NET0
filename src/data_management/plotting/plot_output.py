import matplotlib


def plot_balance_at_node(results, car):


    for node in results.energybalance:
        # make one graph per node
        generic_production = results.energybalance[node][car]['Generic_production']
        tec_output = {}
        for tec in results.detailed_results.nodes[node]:
            tec_output[tec]