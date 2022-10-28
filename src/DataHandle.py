import numpy as np

class DataHandle:
    def __init__(self, topology):
        self.topology = topology

        # Technologies
        self.technology_data  = dict()
        self.technology_data['FurnaceNg'] = dict()
        self.technology_data['FurnaceEl'] = dict()

        # ECONOMICS
        # FurnaceNg
        econ = dict()
        econ['_comment'] = 'CAPEX_annual in EUR/MW/year or as a piecewise function,' \
                           ' OPEX_variable in EUR/MWh,' \
                           ' OPEX_fixed in % of CAPEX_annual'
        econ['CAPEX_model'] = 1
        econ['CAPEX_annual'] = 100
        econ['OPEX_variable'] = 0.1
        econ['OPEX_fixed'] = 0.1
        self.technology_data['FurnaceNg']['Economics'] = econ

        # FurnaceEl
        econ['CAPEX_model'] = 1
        econ['unit_CAPEX_annual'] = 120
        econ['OPEX_variable'] = 0.04
        econ['OPEX_fixed'] = 0.1
        self.technology_data['FurnaceEl']['Economics'] = econ

        # TEC PERFORMANCE
        tec = dict()
        tec['_comment'] = 'contains fitting data one unit of input,' \
                          ' technology types and input/output carriers'
        tec['tec_type'] = 2
        tec['size_min'] = 0
        tec['size_max'] = 100
        tec['size_is_int'] = 0
        tec['input_carrier'] = ['natural_gas']
        tec['input_carrier'] = ['natural_gas']
        tec['output_carrier'] = ['heat']
        tec['performance_x_points'] = [0, 1]
        tec['performance_y_points'] = [0, 0.95]
        # TODO: Implement temperature/irradiance/RH performance changes for technology
        self.technology_data['FurnaceNg']['TechnologyPerf'] = tec

        tec['tec_type'] = 2
        tec['size_min'] = 0
        tec['size_max'] = 100
        tec['size_is_int'] = 0
        tec['input_carrier'] = ['electricity']
        tec['output_carrier'] = ['heat']
        tec['performance'] = dict()
        tec['performance']['electricity'] = dict()
        tec['performance']['electricity']['in'] = [0, 1]
        tec['performance']['electricity']['out'] = [0, 0.99]
        self.technology_data['FurnaceEl']['TechnologyPerf'] = tec

        # Demand
        self.demand = dict()
        self.demand['onshore'] = dict()  # Demands
        self.demand['onshore']['heat'] = np.ones(8760) * 60