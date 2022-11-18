from pyomo.environ import *
from pyomo.environ import units as u
from src.construct_nodes import add_nodes
from src.construct_networks import add_networks
from src.construct_energybalance import add_energybalance
import textwrap

import numpy as np
import dill as pickle
import pandas as pd


class energyhub:

    def __init__(self, sets, data):
        """"
        This function initializes an instance of the energyhub object.
        It (1) creates the sets used in optimization and (2) reads in data
        """
        # INITIALIZE MODEL
        self.model = ConcreteModel()

        # DEFINE SETS
        self.model.set_nodes = Set(initialize=sets['nodes'])  # Nodes
        self.model.set_carriers = Set(initialize=sets['carriers'])  # Carriers
        self.model.set_t = RangeSet(1,sets['timesteps'])  # Timescale
        climate_vars = list(data.climate_data[self.model.set_nodes[1]]['dataframe'].columns.values)
        self.model.set_climate_vars = Set(initialize=climate_vars) # climate variables

        def tec_node(model, node):  # Technologies
            try:
                if node in model.set_nodes:
                    return sets['technologies'][node]
            except (KeyError, ValueError):
                print('The nodes in the technology sets do not match the node names. The node \'', node,
                      '\' does not exist.')
                raise

        self.model.set_technologies = Set(self.model.set_nodes, initialize=tec_node)

        # READ IN DATA
        self.data = data

        # Define currency unit
        u.load_definitions_from_strings(['EUR = [currency]'])

    def construct_model(self):
        """"
        Adds all decision variables and constraints to the model
        """

        self.model = add_networks(self.model, self.data)
        self.model = add_nodes(self.model, self.data)
        self.model = add_energybalance(self.model)

        def cost_objective(obj):
            return sum(self.model.node_blocks[n].cost for n in self.model.set_nodes)
        self.model.objective = Objective(rule=cost_objective, sense=minimize)

    def save_model(self, file_path, file_name):
        """
        Saves the energyhub object to the specified path
        :param file_path: path to save
        :param file_name: filename
        :return: None
        """
        with open(file_path + '/' + file_name, mode='wb') as file:
            pickle.dump(self, file)

    def print_topology(self):
        print('----- SET OF CARRIERS -----')
        for car in self.model.set_carriers:
            print('- ' + car)
        print('----- NODE DATA -----')
        for node in self.model.set_nodes:
            print('\t -----------------------------------------------------')
            print('\t nodename: '+ node)
            print('\t\ttechnologies installed:')
            for tec in self.model.set_technologies[node]:
                print('\t\t - ' + tec)
            print('\t\taverage demand:')
            for car in self.model.set_carriers:
                avg = round(self.data.demand[node][car].mean(), 2)
                print('\t\t - ' + car + ': ' + str(avg))
            print('\t\taverage of climate data:')
            for ser in self.data.climate_data[node]['dataframe']:
                avg = round(self.data.climate_data[node]['dataframe'][ser].mean(),2)
                print('\t\t - ' + ser + ': ' + str(avg))
        print('----- NETWORK DATA -----')
        for car in self.data.topology['networks']:
            print('\t -----------------------------------------------------')
            print('\t carrier: '+ car)
            for netw in self.data.topology['networks'][car]:
                print('\t\t - ' + netw)
                connection = self.data.topology['networks'][car][netw]['connection']
                for from_node in connection:
                    for to_node in connection[from_node].index:
                        if connection.at[from_node, to_node] == 1:
                            print('\t\t\t' + from_node  + '---' +  to_node)
        # for node in self.model.set_nodes:

    def write_results(self, directory):
        for node_name in self.model.set_nodes:
            # TODO: Add import/export here
            file_name = r'./' + directory + '/' + node_name + '.xlsx'

            # get relevant data
            node_data = self.model.node_blocks[node_name]
            n_carriers = len(self.model.set_carriers)
            n_timesteps = len(self.model.set_t)
            demand = self.data.demand[node_name]

            # Get data - input/output
            input_tecs = dict()
            output_tecs = dict()
            size_tecs = dict()
            for car in self.model.set_carriers:
                input_tecs[car] = pd.DataFrame()
                for tec in node_data.s_techs:
                    if car in node_data.tech_blocks[tec].set_input_carriers:
                        temp = np.zeros((n_timesteps), dtype=float)
                        for t in self.model.set_t:
                            temp[t-1] = node_data.tech_blocks[tec].var_input[t, car].value
                        input_tecs[car][tec] = temp

                output_tecs[car] = pd.DataFrame()
                for tec in node_data.s_techs:
                    if car in node_data.tech_blocks[tec].set_output_carriers:
                        temp = np.zeros((n_timesteps), dtype=float)
                        for t in self.model.set_t:
                            temp[t-1] = node_data.tech_blocks[tec].var_output[t, car].value
                        output_tecs[car][tec] = temp

                for tec in node_data.s_techs:
                    size_tecs[tec] = node_data.tech_blocks[tec].var_size.value

            df = pd.DataFrame(data=size_tecs, index=[0])
            with pd.ExcelWriter(file_name) as writer:
                df.to_excel(writer, sheet_name='size')
                for car in self.model.set_carriers:
                    if car in input_tecs:
                        input_tecs[car].to_excel(writer, sheet_name=car + 'in')
                    if car in output_tecs:
                        output_tecs[car].to_excel(writer, sheet_name=car + 'out')
                writer.save()





            # # Create plot
            # fig, axs = plt.subplots(n_carriers, 2)
            # x = range(1, n_timesteps+1)
            #
            # y = input_tecs['electricity']
            # axs[1, 0].stackplot(x, y)
            #
            # counter_i = 0
            # for car in self.model.set_carriers:
            #     y = input_tecs[car]
            #     print(counter_i)
            #     # axs[counter_i, 0].stackplot(x, y)
            #     counter_i = counter_i + 1


