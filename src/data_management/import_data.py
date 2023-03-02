import requests
import cdsapi
from timezonefinder import TimezoneFinder
import pandas as pd
import pickle
import numpy as np

def roundPartial (value, resolution):
    """
    Rounds number to decimals

    Rounds to grid level of JRC dataset

    :param float value: value to round
    :param float resolution: resolution of rounding
    :return float rounded value: result value
    """
    return round (value / resolution) * resolution


def import_jrc_climate_data(lon, lat, year, alt):
    """
    Reads in climate data for a full year from `JRC PVGIS <https://re.jrc.ec.europa.eu/pvg_tools/en/>`_.

    The returned dataframe is consistent with the energyhub format requirements.

    :param float lon: longitude of node - the api will read data for this location
    :param float lat: latitude of node - the api will read data for this location
    :param int year: optional, needs to be in range of data available. If nothing is specified, a typical year \
    will be loaded
    :param float alt: altitude of location specified
    :return: dict containing information on the location (altitude, longitude, latitude and a dataframe \
    containing climate data (ghi = global horizontal irradiance, dni = direct normal irradiance, \
    dhi = diffuse horizontal irradiance, rh = relative humidity, temp_air = air temperature, ws = wind speed at \
    specified hight. Wind speed is returned as a dict for different heights.
    """
    # get time zone
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng=lon, lat=lat)

    # Specify year import, lon, lat
    if year == 'typical_year':
        parameters = {
            'lon': lon,
            'lat': lat,
            'outputformat': 'json'
        }
        time_index = pd.date_range(start='2001-01-01 00:00', freq='1h', periods=8760)
    else:
        parameters = {
            'lon': lon,
            'lat': lat,
            'year': year,
            'outputformat': 'json'
        }
        time_index = pd.date_range(start=str(year)+'-01-01 00:00', end=str(year)+'-12-31 23:00', freq='1h')

    # Get data from JRC dataset
    answer = dict()
    print('Importing Climate Data...')
    response = requests.get('https://re.jrc.ec.europa.eu/api/tmy?', params=parameters)
    if response.status_code == 200:
        print('Importing Climate Data successful')
    else:
        print(response)
    data = response.json()
    climate_data = data['outputs']['tmy_hourly']


    # Compile return dict
    answer['longitude'] = lon
    answer['latitude'] = lat
    answer['altitude'] = alt

    ghi=[]
    dni=[]
    dhi=[]
    rh=[]
    temp_air=[]
    wind_speed=dict()
    wind_speed['10'] = []

    for t_interval in climate_data:
        ghi.append(t_interval['G(h)'])
        dni.append(t_interval['Gb(n)'])
        dhi.append(t_interval['Gd(h)'])
        rh.append(t_interval['RH'])
        temp_air.append(t_interval['T2m'])
        wind_speed['10'].append(t_interval['WS10m'])

    answer['dataframe'] = pd.DataFrame(
        np.array([
            ghi,
            dni,
            dhi,
            temp_air,
            rh
        ]).T,
        columns=['ghi', 'dni', 'dhi', 'temp_air', 'rh'],
        index=time_index)
    for ws in wind_speed:
        answer['dataframe']['ws'+str(ws)] = wind_speed[ws]

    return answer

def import_era5_climate_data(lon, lat, year):
    # TODO: Code this properly
    lat = roundPartial(lat, 0.25)
    lon = roundPartial(lon, 0.25)

    area = [lat + 0.1, lon - 0.1, lat - 0.1, lon + 0.1]

    cds_client = cdsapi.Client()

    print('Retrieving ERA5 data, this might take a wile!')
    cds_client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                "100u",  # 100m_u-component_of_wind
                "100v",  # 100m_v-component_of_wind
                "fsr",  # forecast_surface_roughness
                "sp",  # surface_pressure
                "fdir",  # total_sky_direct_solar_radiation_at_surface
                "ssrd",  # surface_solar_radiation_downwards
                "2t",  # 2m_temperature
                "2d", # 2m_dewpoint_temperature
                "10u",  # 10m_u-component_of_wind
                "10v",  # 10m_v-component_of_wind
            ],
            'year': year,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': area,
        },
        'download.grib')

    print(cds_client.retrieve)
