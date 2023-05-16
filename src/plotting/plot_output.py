import matplotlib.pyplot as plt


class y_series:
    def __init__(self):
        self.values = []
        self.legend_entries = []

    def append(self, new_values, new_legend_entry):
        if any(new_values):
            self.values.append(new_values)
            self.legend_entries.append(new_legend_entry)

class balance_at_node_opts:
    def __init__(self):
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None


def plot_balance_at_node(results, car, opts=None):

    fig, ax = plt.subplots(nrows=len(results.energybalance), ncols=1, facecolor="#F0F0F0")
    if opts == None:
        opts = balance_at_node_opts()

    i = 0
    for node in results.energybalance:
        # make one graph per node
        x = range(0, len(results.energybalance[node][car]))

        # Positive Values
        y_pos = y_series()
        y_pos.append(results.energybalance[node][car]['Generic_production'], 'Generic Production')
        y_pos.append(results.energybalance[node][car]['Import'], 'Import')
        y_pos.append(results.energybalance[node][car]['Network_inflow'], 'Network Inflow')
        for tec in results.detailed_results.nodes[node]:
            tec_results = results.detailed_results.nodes[node][tec]
            y_pos.append(tec_results['output_' + car], 'Output ' + tec)

        if y_pos.values:
            ax[i].stackplot(x, y_pos.values,
                        labels=y_pos.legend_entries)

        # Negative Values
        y_neg = y_series()
        y_neg.append(-results.energybalance[node][car]['Network_consumption'], 'Network Consumption')
        y_neg.append(-results.energybalance[node][car]['Export'], 'Export')
        y_neg.append(-results.energybalance[node][car]['Network_outflow'], 'Network Outflow')
        for tec in results.detailed_results.nodes[node]:
            tec_results = results.detailed_results.nodes[node][tec]
            if 'input_' + car in tec_results:
                y_neg.append(-tec_results['input_' + car], 'Input ' + tec)

        if y_neg.values:
            ax[i].stackplot(x, y_neg.values,
                          labels=y_neg.legend_entries)

        # Demand
        ax[i].plot(x, results.energybalance[node][car]['Demand'])

        ax[i].legend(loc='upper left')

        i += 1

    plt.show()

