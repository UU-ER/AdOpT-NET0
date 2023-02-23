from pyomo.environ import *
from pyomo.gdp import *
import pylab as pl
import pvlib
import numpy as np
import requests
import json
import pandas as pd
from timezonefinder import TimezoneFinder
import src.data_management as dm
from src.energyhub import EnergyHub


from src.data_management.components.fit_technology_performance import fit_piecewise_function

execute = 0
if execute == 1:
    data = dm.load_data_handle(r'./test/test_data/emissionbalance1.p')
    data.technology_data['onshore']['Furnace_NG']['TechnologyPerf']['performance_function_type'] = 1
    data.technology_data['onshore']['Furnace_NG']['fit']['heat']['alpha1'] = 0.9
    data.network_data['electricityTest']['NetworkPerf']['emissionfactor'] = 0.2
    data.network_data['electricityTest']['NetworkPerf']['loss2emissions'] = 1
    energyhub = EnergyHub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    results = energyhub.write_results()


execute = 0
# region: how to k-means cluster
if execute == 1:
    # Load data handle from file
    modeled_year = 2001
    topology = {}
    topology['timesteps'] = pd.date_range(start=str(modeled_year) + '-01-01 00:00',
                                          end=str(modeled_year) + '-12-31 23:00', freq='1h')
    topology['timestep_length_h'] = 1
    topology['carriers'] = ['electricity']
    topology['nodes'] = ['test_node1']
    topology['technologies'] = {}
    # topology['technologies']['test_node1'] = ['PV', 'testSTOR']
    topology['technologies']['test_node1'] = ['PV']

    topology['networks'] = {}

    # Initialize instance of DataHandle
    data = dm.DataHandle(topology)

    # CLIMATE DATA
    data.read_climate_data_from_file('test_node1', r'./test/test_data/climate_data_test.p')

    # DEMAND
    electricity_demand = np.ones(len(topology['timesteps'])) * 1
    data.read_demand_data('test_node1', 'electricity', electricity_demand)

    # IMPORT
    electricity_import = np.ones(len(topology['timesteps'])) * 10
    data.read_import_limit_data('test_node1', 'electricity', electricity_import)

    # IMPORT Prices
    electricity_price = np.ones(len(topology['timesteps'])) * 1000
    data.read_import_price_data('test_node1', 'electricity', electricity_price)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SOLVE WITH CLUSTERED DATA
    clustered_data = dm.ClusteredDataHandle()
    nr_days_cluster = 5
    clustered_data.cluster_data(data, nr_days_cluster)

    energyhub_clustered = EnergyHub(clustered_data)
    energyhub_clustered.construct_model()
    energyhub_clustered.construct_balances()

    # Solve model
    energyhub_clustered.solve_model()
    results1 = energyhub_clustered.write_results()
    results1.write_excel(r'.\userData\results_clustered')

    # SOLVE WITH FULL RESOLUTION
    energyhub = EnergyHub(data)
    energyhub.construct_model()
    energyhub.construct_balances()

    # Solve model
    energyhub.solve_model()
    results2 = energyhub.write_results()
    results2.write_excel(r'.\userData\results_full')



execute = 0
#region How to formulate hierarchical models with blocks
if execute == 1:
    m = ConcreteModel()

    m.t = RangeSet(1,1)
    m.tec = Set(initialize=['Tec1','Tec2'])

    m.tecBlock = Block(m.tec)

    def tec_block_rule(bl):
        bl.var_input = Var()
        bl.var_output = Var()
        def inout(model):
            return bl.var_output == 0.7 * bl.var_input
        bl.c_perf = Constraint(rule=inout)
    m.tecBlock = Block(m.tec, rule=tec_block_rule)

    newtec = ['Tec3']

    m.tec.add(newtec)
    m.tecBlock[newtec] = Block(rule=tec_block_rule)
    # for tec in m.tec:
    #     m.tecBlock2[tec].transfer_attributes_from(m.tecBlock[tec])
    # m.bla = Block()
    # m.bla.transfer_attributes_from(m.tecBlock['Tec1'].clone())

    #
    # m.tecBlock['Tec1'].add_component('bla', RangeSet(1,1))
    # m.pprint()
    # m.pprint()
    # m.tecBlock['Tec3'].pprint()
    # m.cons_balance = Constraint(expr=m.tecBlock['Tec3'].var_output == 3)
    m.pprint()



    # Set definitions
    # m.nodes = Set(initialize=['onshore','offshore'])
    # m.tecs = Set(m.nodes, dimen=1, initialize=['Tec1','Tec2'])

    # m.tecs['onshore'].pprint()
#endregion

execute = 0
#region How to define semi-continuous variables (modelled as a disjunction)
if execute == 1:
    model = ConcreteModel()
    model.t = RangeSet(1, 2)

    # We want to define a constraint 10 < input < Sx with input, S being continous, and x being a binary variable
    model.s = Var(domain=PositiveReals, bounds=(0, 100))
    model.input = Var(model.t, domain=PositiveReals, bounds=(0, 100))
    model.x = Var(domain=Binary)

    def on(dis, t):
        dis.c1 = Constraint(expr=model.input[t] == 0)
    model.d1 = Disjunct(model.t, rule = on)
    def off(dis, t):
        dis.c1 = Constraint(expr=model.input[t] <= model.s)
        dis.c2 = Constraint(expr= 10 <= model.input[t])
    model.d2 = Disjunct(model.t, rule = off)

    def bind_disjunctions(dis, t):
        return [model.d1[t], model.d2[t]]
    model.dj = Disjunction(model.t, rule=bind_disjunctions)

    def inputreq(ex, t):
        return model.input[t] == 11
    model.c3 = Constraint(model.t, rule=inputreq)
    model.obje = Objective(expr=model.s, sense=minimize)

    model.pprint()

    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(model)
    solver = SolverFactory('gurobi')
    solution = solver.solve(model, tee=True)
    solution.write()
    model.display()


#endregion

execute = 0
#region How to model a piecewise affine function with dependent breakpoints
if execute == 1:
    model = ConcreteModel()

    model.t = RangeSet(1, 2)

    model.s = Var(within=NonNegativeReals, bounds=(0, 100))
    model.input = Var(model.t, domain=PositiveReals, bounds=(0, 100))
    model.output = Var(model.t, domain=PositiveReals, bounds=(0, 100))


    # disjunct for technology off
    def calculate_input_output_off(dis, t):
        def calculate_input_off(con, c_input):
            return model.input[t] == 0
        dis.const_input_off = Constraint(rule=calculate_input_off)
        def calculate_output_off(con):
            return model.output[t] == 0
        dis.const_output_off = Constraint(rule=calculate_output_off)
    model.disjunct_input_output_off = Disjunct(model.t, rule=calculate_input_output_off)

    # disjunct for technology on
    s_indicators=range(1,5)
    x_bp = [10, 12]
    def create_disjunctions(dis, t, ind):
        def calculate_input_on(con):
            return model.input[t] == x_bp[ind-1]
        dis.const_input_on = Constraint(rule=calculate_input_on)
    model.disjunct_on = Disjunct(model.t, s_indicators, rule=create_disjunctions)

    def bind_disjunctions(dis, t):
        return [model.disjunct_on[i,t] for i in s_indicators]
    model.dj = Disjunction(model.t, rule=bind_disjunctions)


    def inputreq(ex, t):
        return model.input[t] == 10
    model.c3 = Constraint(model.t, rule=inputreq)
    model.obje = Objective(expr=model.s, sense=minimize)

    model.pprint()

    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(model)
    solver = SolverFactory('gurobi')
    solution = solver.solve(model, tee=True)
    solution.write()
    model.display()

#endregion

execute = 0
#region How to fit a piece-wise linear function
if execute == 1:
    nr_seg = 3
    tec = 'testPWA'
    with open('./data/technology_data/' + tec + '.json') as json_file:
        technology_data = json.load(json_file)

    tec_data = technology_data['TechnologyPerf']
    performance_data = tec_data['performance']
    X = performance_data['in']
    Y = performance_data['out']
    count = nr_seg
    fitting = fit_piecewise_function(X, Y, nr_seg)

    pl.plot(X, Y['electricity'], ".")
    pl.plot(X, Y['heat'], ".")
    pl.plot(fitting['bp_x'], fitting['electricity']['bp_y'], "-or")
    pl.plot(fitting['bp_x'], fitting['heat']['bp_y'], "-or")

    pl.show()

#endregion

execute = 0
#region How to use pvlib
if execute == 1:
    # pass data from external
    temperature = 15
    dni = 800
    ghi = 900
    dhi = 100
    wind_speed = 5

    lat = 52
    lon = 5.16
    alt = 10

    # temperature = pd.Series(climate_data["temperature"], time_index)
    # dni = pd.Series(climate_data['direct_normal_irr'], time_index)
    # ghi = pd.Series(climate_data['global_horizontal_irr'], time_index)
    # dhi = pd.Series(climate_data['diffuse_horizontal_irr'], time_index)
    # wind_speed = pd.Series(climate_data["wind_speed"], time_index)

    # pass as standard vars
    module_name = 'SunPower_SPR_X20_327'
    inverter_eff = 0.96
    tilt = 20
    surface_azimuth = 180

    # Define Module
    # sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    # module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

    module_database = pvlib.pvsystem.retrieve_sam('CECMod')
    module = module_database[module_name]

    # Define temperature losses of module
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # Get location
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng=lon, lat=lat)
    location = pvlib.location.Location(lat, lon, tz=tz, altitude=alt)

    # Make weather data
    weather = pd.DataFrame([[ghi, dni, dhi, temperature, wind_speed]],
                           columns=['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed'],
                           index=[pd.Timestamp('20170401 1200', tz=tz)])

    # Create PV model chain
    inverter_parameters = {'pdc0': 5000, 'eta_inv_nom': inverter_eff}
    system = pvlib.pvsystem.PVSystem(surface_tilt=tilt, surface_azimuth=surface_azimuth,
                      module_parameters=module,
                      inverter_parameters=inverter_parameters,
                      temperature_model_parameters=temperature_model_parameters)

    pv_model = pvlib.modelchain.ModelChain(system, location, spectral_model="no_loss", aoi_model="physical")

    pv_model.run_model(weather)


    power = pv_model.results.ac.p_mp

    # print(power/220)

    capacity_factor = power/module.STC
    specific_area = module.STC / module.A_c / 1000 / 1000

    print(capacity_factor)

    # parameters['capacity_factors'] = cap_factor

#endregion

execute = 0
#region How to make an API request for JRC
if execute == 1:
    lon = 8
    lat = 45
    year = 'typical_year'
    if year == 'typical_year':
        parameters = {
            'lon': lon,
            'lat': lat,
            'outputformat': 'json'
        }
    else:
        parameters = {
            'lon': lon,
            'lat': lat,
            'year': year,
            'outputformat': 'json'
        }

    print('Importing Climate Data...')
    response = requests.get('https://re.jrc.ec.europa.eu/api/tmy?', params=parameters)
    if response.status_code == 200:
        print('Importing Climate Data successful')
    data = response.json()
    climate_data = data['outputs']['tmy_hourly']
    temperature2m = dict()
    relative_humidity = dict()
    global_horizontal_irr = dict()
    direct_normal_irr = dict()
    diffuse_horizontal_irr = dict()
    wind_speed10m = dict()

    for t_interval in climate_data:
        print(t_interval)
        temperature2m[t_interval['time(UTC)']] = t_interval['T2m']
        relative_humidity[t_interval['time(UTC)']] = t_interval['RH']
        global_horizontal_irr[t_interval['time(UTC)']] = t_interval['G(h)']
        direct_normal_irr[t_interval['time(UTC)']] = t_interval['Gb(n)']
        diffuse_horizontal_irr[t_interval['time(UTC)']] = t_interval['Gd(h)']
        wind_speed10m[t_interval['time(UTC)']] = t_interval['WS10m']
#endregion