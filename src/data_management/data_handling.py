import json
import src.model_construction as mc
import src.data_management as dm
import pickle
import pandas as pd


class DataHandle:
    """
    Data Handle for loading and performing operations on input data.

    The Data Handle class allows data import and modifications of input data to an instance of the energyhub class.
    The constructor of the class takes a topology dictionary with time indices, networks, technologies, nodes as an \
    an input. An empty topology dict can be created by using the function \
     :func:`~src.data_management.create_templates.create_empty_topology`
    """
    def __init__(self, topology):
        """
        Constructor

        Initializes a data handle class and completes demand data for each carrier used (i.e. sets it to zero for all \
        time steps)

        :param dict topology: dictionary with time indices, networks, technologies, nodes. An empty dict can be created \
        by using the function :func:`~src.data_management.create_templates.create_empty_topology`
        """
        self.topology = topology
        self.node_data = {}
        self.technology_data = {}
        self.network_data = {}

        # init. demand, prices, emission factors = 0 for all timesteps, carriers and nodes

        for nodename in self.topology['nodes']:
            self.node_data[nodename] = {}
            self.node_data[nodename]['demand'] = pd.DataFrame(index=self.topology['timesteps'])
            self.node_data[nodename]['import_prices'] = pd.DataFrame(index=self.topology['timesteps'])
            self.node_data[nodename]['import_limit'] = pd.DataFrame(index=self.topology['timesteps'])
            self.node_data[nodename]['export_prices'] = pd.DataFrame(index=self.topology['timesteps'])
            self.node_data[nodename]['export_limit'] = pd.DataFrame(index=self.topology['timesteps'])
            self.node_data[nodename]['emission_factors'] = pd.DataFrame(index=self.topology['timesteps'])
            for carrier in self.topology['carriers']:
                self.node_data[nodename]['demand'][carrier] = 0
                self.node_data[nodename]['import_prices'][carrier] = 0
                self.node_data[nodename]['import_limit'][carrier] = 0
                self.node_data[nodename]['export_prices'][carrier] = 0
                self.node_data[nodename]['export_limit'][carrier] = 0
                self.node_data[nodename]['emission_factors'][carrier] = 0

    def read_climate_data_from_api(self, nodename, lon, lat, alt=10, dataset='JRC', year='typical_year', save_path=0):
        """
        Reads in climate data for a full year

        Reads in climate data for a full year from the specified source \
        (`JRC PVGIS <https://re.jrc.ec.europa.eu/pvg_tools/en/>`_ or \
        `ERA5 <https://cds.climate.copernicus.eu/cdsapp#!/home>`_). For access to the ERA5 api, \
        an api key is required. Refer to `<https://cds.climate.copernicus.eu/api-how-to>`_

        :param str nodename: nodename as specified in the topology
        :param float lon: longitude of node - the api will read data for this location
        :param float lat: latitude of node - the api will read data for this location
        :param str dataset: dataset to import from, can be JRC (only onshore) or ERA5 (global)
        :param int year: optional, needs to be in range of data available. If nothing is specified, a typical year \
        will be loaded
        :param str save_path: Can save climate data for later use to the specified path
        :return: self at ``self.node_data[nodename]['climate_data']``
        """
        if dataset == 'JRC':
            data = dm.import_jrc_climate_data(lon, lat, year, alt)
        elif dataset == 'ERA5':
            data = dm.import_era5_climate_data(lon, lat, year)

        if not save_path==0:
            with open(save_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.node_data[nodename]['climate_data'] = data

    def read_climate_data_from_file(self, nodename, file):
        """
        Reads climate data from file

        Reads previously saved climate data (imported and saved with :func:`~read_climate_data_from_api`) from a file to \
        the respective node. This can save time, if api imports take too long

        :param str nodename: nodename as specified in the topology
        :param str file: path of climate data file
        :return: self at ``self.node_data[nodename]['climate_data']``
        """
        with open(file, 'rb') as handle:
            data = pickle.load(handle)

        self.node_data[nodename]['climate_data'] = data

    def read_demand_data(self, nodename, carrier, demand_data):
        """
        Reads demand data for one carrier to node.

        Note that demand for all carriers not specified is zero.
        
        :param str nodename: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list demand_data: list of demand data. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[nodename]['demand'][carrier]``
        """

        self.node_data[nodename]['demand'][carrier] = demand_data

    def read_import_price_data(self, nodename, carrier, price_data):
        """
        Reads import price data of carrier to node

        Note that prices for all carriers not specified is zero.

        :param str nodename: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list price_data: list of price data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[nodename]['import_prices'][carrier]``
        """

        self.node_data[nodename]['import_prices'][carrier] = price_data

    def read_export_price_data(self, nodename, carrier, price_data):
        """
        Reads export price data of carrier to node

        Note that prices for all carriers not specified is zero.

        :param str nodename: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list price_data: list of price data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[nodename]['export_prices'][carrier]``
        """

        self.node_data[nodename]['export_prices'][carrier] = price_data

    def read_export_limit_data(self, nodename, carrier, export_limit_data):
        """
        Reads export price data of carrier to node

        Note that prices for all carriers not specified is zero.

        :param str nodename: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list price_data: list of price data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[nodename]['export_prices'][carrier]``
        """

        self.node_data[nodename]['export_limit'][carrier] = export_limit_data

    def read_import_limit_data(self, nodename, carrier, import_limit_data):
        """
        Reads export price data of carrier to node

        Note that prices for all carriers not specified is zero.

        :param str nodename: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list price_data: list of price data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[nodename]['export_prices'][carrier]``
        """

        self.node_data[nodename]['import_limit'][carrier] = import_limit_data

    def read_technology_data(self):
        """
        Writes technologies to self and fits performance functions

        Reads in technology data from JSON files located at ``./data/technology_data`` for all technologies specified in \
        the topology.

        :return: self at ``self.technology_data[nodename][tec]``
        """
        # get all used technologies
        tecs_used = dict()
        for nodename in self.topology['technologies']:
            tecs_used[nodename] = self.topology['technologies'][nodename]
            self.technology_data[nodename] = dict()
            # read in data to Data Handle and fit performance functions
            for tec in tecs_used[nodename]:
                # Read in JSON files
                with open('./data/technology_data/' + tec + '.json') as json_file:
                    technology_data = json.load(json_file)
                # Fit performance function
                if (technology_data['TechnologyPerf']['tec_type'] == 1) or \
                        (technology_data['TechnologyPerf']['tec_type'] == 6):
                    technology_data = mc.fit_tec_performance(technology_data, tec=tec,
                                                          climate_data=self.node_data[nodename]['climate_data'])
                else:
                    technology_data = mc.fit_tec_performance(technology_data)

                self.technology_data[nodename][tec] = technology_data

    def read_single_technology_data(self, nodename, technologies):
        """
        Reads technologies to DataHandle after it has been initialized.

        This function is only required if technologies are added to the model after the DataHandle has been initialized.
        """

        for tec in technologies:
            # Read in JSON files
            with open('./data/technology_data/' + tec + '.json') as json_file:
                technology_data = json.load(json_file)
            # Fit performance function
            if (technology_data['TechnologyPerf']['tec_type'] == 1) or \
                    (technology_data['TechnologyPerf']['tec_type'] == 6):
                technology_data = mc.fit_tec_performance(technology_data, tec=tec,
                                                         climate_data=self.node_data[nodename]['climate_data'])
            else:
                technology_data = mc.fit_tec_performance(technology_data)

            self.technology_data[nodename][tec] = technology_data


    def read_network_data(self):
        """
        Writes network to self and fits performance functions

        Reads in network data from JSON files located at ``./data/network_data`` for all technologies specified in \
        the topology.

        :return: self at ``self.technology_data[nodename][tec]``
        """
        for netw in self.topology['networks']:
            with open('./data/network_data/' + netw + '.json') as json_file:
                network_data = json.load(json_file)
            network_data['distance'] = self.topology['networks'][netw]['distance']
            network_data['connection'] = self.topology['networks'][netw]['connection']
            network_data = mc.fit_netw_performance(network_data)
            self.network_data[netw] = network_data

    def pprint(self):
        """
        Prints a summary of the input data (excluding climate data)

        :return: None
        """
        for nodename in self.topology['nodes']:
            print('----- NODE '+ nodename +' -----')
            for inst in self.node_data[nodename]:
                if not inst == 'climate_data':
                    print('\t ' + inst)
                    print('\t\t' + f"{'':<15}{'Mean':>10}{'Min':>10}{'Max':>10}")
                    for carrier in self.topology['carriers']:
                        print('\t\t' + f"{carrier:<15}"
                                       f"{str(round(self.node_data[nodename][inst][carrier].mean(), 2)):>10}"
                                       f"{str(round(self.node_data[nodename][inst][carrier].min(), 2)):>10}"
                                       f"{str(round(self.node_data[nodename][inst][carrier].max(), 2)):>10}")

    def save(self, path):
        """
        Saves instance of DataHandle to path.

        The instance can later be loaded with

        :param str path: path to save to
        :return: None
        """
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data_handle(path):
    """
    Loads instance of DataHandle from path.

    :param str path: path to load from
    :return: instance of :class:`~DataHandle`
    """
    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    return data