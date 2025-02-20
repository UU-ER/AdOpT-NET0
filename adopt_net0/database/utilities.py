from pathlib import Path
import pandas as pd


def correct_inflation(value, year_in, year_out):

    ppi_path = Path(__file__).parent / Path("./data/producer_price_index_euro.csv")
    ppi = pd.read_csv(ppi_path)
    ppi.index = pd.to_datetime(ppi["TIME_PERIOD"])  # Convert the index to datetime
    ppi_year_in = ppi[ppi.index.year == year_in]["OBS_VALUE"].mean()
    ppi_year_out = ppi[ppi.index.year == year_out]["OBS_VALUE"].mean()

    return value * ppi_year_out / ppi_year_in


def convert_currency(value, year_in, year_out, currency_in, currency_out):
    """
    Convert value to specified currency for given financial_year_in and currency_in

    Uses the average exchange rates for a year as given by the European Central Bank
    :param float value: value to convert
    :return:
    """
    rates_path = Path(__file__).parent / Path("./data/conversion_rates.csv")
    rates = pd.read_csv(rates_path, index_col=0)
    rates.index = pd.to_datetime(rates.index)  # Convert the index to datetime
    rates_year_in = rates[rates.index.year == year_in].mean()
    rates_year_out = rates[rates.index.year == year_out].mean()

    # Convert to EUR
    if currency_in != "EUR":
        rate_eur = rates_year_in[currency_in]
        value = value / rate_eur

    value = correct_inflation(value, year_in, year_out)

    if currency_out != "EUR":
        rate_other = rates_year_out[currency_out]
        value = value * rate_other

    return value
