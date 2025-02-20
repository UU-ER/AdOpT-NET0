class DataComponent:
    """
    Parent class to all technologies/networks calculations

    At least the following options need to be supplied:
    - currency_out: output currency
    - financial_year_out: output financial year (for inflation correction)
    - discount_rate: discount rate for annualizing
    """

    def __init__(self, options):
        # Output units
        self.currency_out = options["currency_out"]
        self.financial_year_out = options["financial_year_out"]
        self.discount_rate = options["discount_rate"]

        # Input units
        self.currency_in = None
        self.financial_year_in = None

        # Financial indicators
        self.financial_indicators = {}

        # Technical indicators
        self.technical_indicators = {}

        # Json data
        self.json_template = None

        # Options
        self.default_options = {}
        self.options = {}

    def calculate_financial_indicators(self):
        """
        Calculates financial indicators

        Overwritten in child classes
        """
        pass

    def calculate_technical_indicators(self):
        """
        Calculates technical indicators

        Overwritten in child classes
        """
        raise NotImplementedError

    def write_json(self, path):
        """
        Write json to specified path

        Overwritten in child classes
        :param str path: path to write to
        """
        pass

    def _set_option_value(self, key, options):
        """
        Sets option value either to default or to provided option
        :param str key: option key
        :param dict options: provided options
        """
        if key in options:
            self.options[key] = options[key]
        else:
            self.options[key] = self.default_options[key]
