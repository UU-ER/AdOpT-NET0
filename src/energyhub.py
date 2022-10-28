from pyomo.environ import *
from src.compile_demand import compile_demand
from src.node_construction import add_nodes


class energyhub:

    def __init__(self, sets, data):
        """"
        This function initializes an instance of the energyhub object.
        - It (1) creates the sets used in optimization and (2) reads in data
        - It performs a sanity check of the sets defined
        """
        # INITIALIZE MODEL
        self.model = ConcreteModel()

        # DEFINE SETS
        self.model.set_nodes = Set(initialize=sets['nodes'])  # Nodes

        self.model.set_carriers = Set(initialize=sets['carriers'])  # Carriers

        self.model.set_t = Set(initialize=sets['time'])  # Timescale

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
        # self.m = compile_demand(self.m, data.demand)

    def construct_model(self):
        """"
        Adds all decision variables and constraints to the model
        """
        self.model = add_nodes(self.model, self.data)

        # def cost_objective(ehub):
        #     return sum(self.model.node_blocks[n].cost for n in self.model.s_nodes)
        # self.model.objective = Objective(rule=cost_objective, sense=minimize)

    def add_technology_block(self):
        c = 3
