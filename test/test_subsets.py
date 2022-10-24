from pyomo.environ import *

# Initialize Concrete Model
m = ConcreteModel()

# Initialize Technology Set
m.tec = Set(initialize= ['T1', 'T2'])

# Initialize Carrier Set
m.car = Set(initialize= ['C1', 'C2'])

# Initialize Technologies
tecin = []
temptecin = ('C1', 'T1')
tecin.append(temptecin)
temptecin = ('C1', 'T2')
tecin.append(temptecin)
temptecin = ('C2', 'T2')
tecin.append(temptecin)
m.s_tecin = Set(initialize=tecin)
m.s_tecin.pprint()

# Initialize Input to technology T
m.Tec_in = Var(m.s_tecin)
m.Tec_in.pprint()

# Initialize Input-Tec Combos
def in_tec_comb_init(m, car):
    for i, j in m.s_tecin:
        if i == car:
            yield j
m.in_tec_comb = Set(m.car, initialize=in_tec_comb_init)
m.in_tec_comb.pprint()

# Initialize sum of inputs
m.insum = Var(m.car)
m.insum.pprint()

# Formulate constraints
def sum_all_inputs(m, car):
    return sum(m.Tec_in[car,t] for t in m.in_tec_comb[car]) == m.insum[car]
m.sumIn = Constraint(m.car, rule=sum_all_inputs)
m.sumIn.pprint()

# Formulate Objective

# Solve
