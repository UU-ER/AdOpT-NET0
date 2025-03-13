from pathlib import Path
import pandas as pd


def correct_inflation(value: float, year_in: int, year_out: int) -> float:
    """
    Corrects a value for inflation using the producer price index of the euro region taken from Producer prices in industry, total - monthly data
    from eurostat (https://ec.europa.eu/eurostat/databrowser/view/sts_inpp_m/default/table?lang=en)

    The function uses the average value of the monthly data provided

    :param float value: value before inflation correction
    :param float year_in: financial year of original value
    :param float year_out: financial year of inflation-corrected value
    :return: value after inflation correction
    :rtype float
    """

    ppi_path = Path(__file__).parent / Path("./data/producer_price_index_euro.csv")
    ppi = pd.read_csv(ppi_path)
    ppi.index = pd.to_datetime(ppi["TIME_PERIOD"])  # Convert the index to datetime
    ppi_year_in = ppi[ppi.index.year == year_in]["OBS_VALUE"].mean()
    ppi_year_out = ppi[ppi.index.year == year_out]["OBS_VALUE"].mean()

    return value * ppi_year_out / ppi_year_in


def convert_currency(
    value: float, year_in: int, year_out: int, currency_in: str, currency_out: str
) -> float:
    """
    Convert value to specified currency for given financial_year_in and currency_in

    Exchange rates are taken from the european central bank (Euro foreign exchange reference rates), available here: Euro foreign exchange reference rates
    Inflation correction is done using the Producer prices in industry, total - monthly data from eurostat (https://ec.europa.eu/eurostat/databrowser/view/sts_inpp_m/default/table?lang=en)

    :param float value: value to convert
    :param float year_in: financial year of original value
    :param float year_out: financial year of inflation-corrected value
    :param str currency_in: currency of input value
    :param str currency_out: currency to convert to
    :return: inflation and exchange rate corrected value
    :rtype float
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
