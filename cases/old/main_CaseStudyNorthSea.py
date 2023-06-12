# TODO: Include hplib
# TODO: Implement option for complete linearization
# TODO: Implement time index for set_t
# TODO: Implement length of time step
# TODO: Implement design days (retain extremes)
# TODO: Implement Lukas Algorithm
from CaseStudy_preprocessing import create_data
from src.energyhub import *
from src.model_configuration import ModelConfiguration
from src.data_management import load_object

# Save Data File to file
data_save_path = r'.\user_data\CaseStudyNorthSea_input'
#
# Read in climate data and demand data
data = create_data()

# READ TECHNOLOGY AND NETWORK DATA
data.read_technology_data()
data.read_network_data()

data.save(data_save_path)
data.pprint()



# # Read data
configuration = ModelConfiguration()

data = load_object(data_save_path)

energyhub = EnergyHub(data, configuration)

# Construct equations
energyhub.construct_model()
energyhub.construct_balances()

energyhub.solve_model()
energyhub.construct_balances()

# Solve model
energyhub.solve_model()
results = energyhub.write_results()
results.write_excel(r'.\user_data\CaseStudyNorthSea')
# # Add technology to model and solve again
# energyhub.add_technology_to_node('onshore', ['WT_OS_11000'])
# energyhub.construct_balances()
# energyhub.solve_model()
#
# # Write results
# results = energyhub.write_results()

print('done')
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