from cases.NorthSea.preprocessing_simple import create_data
from src.energyhub import *
from src.model_configuration import ModelConfiguration
from src.data_management import load_object
from pympler import asizeof


# Save Data File to file
data_save_path = r'.\user_data\CaseStudyNorthSea_input'
#
# Read in climate data and demand data
data = create_data()

# READ TECHNOLOGY AND NETWORK DATA
data.read_technology_data()
data.read_network_data()

data.save(data_save_path)

# Read data
configuration = ModelConfiguration()
configuration.optimization.objective = 'pareto'
# configuration.optimization.typicaldays = 10

data = load_object(data_save_path)

energyhub = EnergyHub(data, configuration)

# Construct equations
energyhub.quick_solve_model()

if configuration.optimization.objective == 'pareto':
    i = 0
    for result in energyhub.results:
        result.write_excel(r'.\user_data\CaseStudyNorthSea' + str(i))
        i += 1
else:
    results = energyhub.write_results()
    results.write_excel(r'.\user_data\CaseStudyNorthSea_persistent')
# # Add technology to model and solve again
# energyhub.add_technology_to_node('onshore', ['WT_OS_11000'])
# energyhub.construct_balances()
# energyhub.solve_model()
#
# # Write results
# results = energyhub.write_results()

# energyhub.model.display()
#
# # energyhub.model.pprint()
# # # Save model
# # print('Saving Model...')
# # start = time.time()
# # energyhub.save_model('./data/ehub_instances', 'test_non_transformed')
# # print('Saving Model completed in ' + str(time.time()-start) + ' s')
# #
# Big-M transformation
# print('Performing Big-M transformation...')
# start = time.time()
# xfrm = TransformationFactory('gdp.bigm')
# xfrm.apply_to(energyhub.model)
# print('Performing Big-M transformation completed in ' + str(time.time()-start) + ' s')
# Display whole model
# energyhub.model.pprint()

# Save model
# print('Saving Model...')
# start = time.time()
# energyhub.save_model('./data/ehub_instances', 'test_non_transformed')
# print('Saving Model completed in ' + str(time.time()-start) + ' s')