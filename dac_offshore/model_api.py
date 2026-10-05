from pathlib import Path
import json

import adopt_net0 as adopt
from dac_offshore.create_input_data import InputDataCreator
from dac_offshore.global_vars import CO2_PRICE

def load_baseline_emissions(test):
    baseline_emissions_path = Path(__file__).resolve().parent / 'emission_data' / f"baseline_emissions_test_{str(test)}.json"
    with open(baseline_emissions_path, 'r') as f:
        data = json.load(f)
    return data["baseline_emissions"]

def load_max_neg_emissions():
    pass

class ModelApi:

    def __init__(self, test, run_id, climate_year= None, scenario=None):

        # top-level settings
        self.test = test
        self.model_year = 2040
        self.climate_year = climate_year
        self.scenario = scenario
        self.run_id = run_id

        self.baseline_emissions_path = Path(__file__).resolve().parent / 'emission_data' / f"baseline_emissions_test_{str(self.test)}.json"
        self.max_negative_emissions_path = Path(__file__).resolve().parent / 'emission_data' / f"max_negative_emissions_test_{str(self.test)}.json"

        if test:
            self.start_period = 50
            self.end_period = 51
        else:
            self.start_period = None
            self.end_period = None

        # save data path
        self.save_data_path = Path("")

        # input folder data path
        self.input_data_path = Path(__file__).resolve().parent / 'input_data' / (str(self.model_year) + "_" + str(self.climate_year) + "_" + str(self.scenario))

        # model settings
        self.new_technologies_stage = None
        self.demand_factor = 1
        self.simplify_networks = 1
        self.c_permutation = 0

        # model
        self.model = None


    def build_model(self):

        self.model = adopt.ModelHub()
        self.model.read_data(self.input_data_path, start_period=self.start_period, end_period=self.end_period)
        InputDataCreator.define_charging_efficiencies(self.model)

    def define_save_paths(self, save_path):
        if self.test:
            run_save_path = save_path / f"2040_test_{self.run_id}"
            summary_save_path = run_save_path
        else:
            run_save_path = save_path / f"2040_{self.run_id}" / str(self.climate_year)
            summary_save_path = save_path / f"2040_{self.run_id}"

        Path(run_save_path).mkdir(parents=True, exist_ok=True)

        self.model.data.model_config["reporting"]["save_summary_path"]["value"] = summary_save_path
        self.model.data.model_config["reporting"]["save_path"]["value"] = run_save_path

    def construct_model(self):
        self.model.construct_model()
        self.model.construct_balances()
        self.model._define_solver_settings()

    def solve_model(self, optimization_type, emission_reduction_percentage=None):

        if optimization_type == "cost":

            self.model.data.model_config["reporting"]["case_name"]["value"] = \
                f"{self.scenario}_baseline_cy{self.climate_year}_co2_tax{CO2_PRICE}"
            self.model._optimize_cost()

            # save baseline emissions
            model = self.model.model[self.model.info_solving_algorithms["aggregation_model"]]
            baseline_emissions = sum(model.periods[p].var_emissions_pos.value for p in model.set_periods)
            
            # Save baseline emissions to file
            with open(self.baseline_emissions_path, 'r') as f:
                data = json.load(f)
                data[str(self.climate_year)] = baseline_emissions

            with open(self.baseline_emissions_path, 'w') as f:
                json.dump(data, f)

        elif optimization_type == "negative_emissions":
            with open(self.baseline_emissions_path, 'r') as f:
                data = json.load(f)

            baseline_emissions_pos = data[str(self.climate_year)]

            self.model.data.model_config["reporting"]["case_name"]["value"] = \
                f"{self.scenario}_max_neg_E_cy{self.climate_year}_co2_tax{CO2_PRICE}"
            self.model._optimize_emissions_neg(baseline_emissions_pos)

            # Save max negative emissions to file
            model = self.model.model[self.model.info_solving_algorithms["aggregation_model"]]
            max_negative_emissions = sum(model.periods[p].var_emissions_neg.value for p in model.set_periods)

            with open(self.max_negative_emissions_path, 'r') as f:
                data = json.load(f)

            if self.scenario not in data:
                data[self.scenario] = {}
            
            data[self.scenario][str(self.climate_year)] = max_negative_emissions

            with open(self.max_negative_emissions_path, 'w') as f:
                json.dump(data, f)

        elif optimization_type == "cost_at_emission_reduction":
            with open(self.baseline_emissions_path, 'r') as f:
                data = json.load(f)
            baseline_emissions_pos = data[str(self.climate_year)]

            with open(self.max_negative_emissions_path, 'r') as f:
                data = json.load(f)
            max_negative_emissions = data[self.scenario][str(self.climate_year)]

            neg_target = max_negative_emissions * emission_reduction_percentage
            pos_limit = baseline_emissions_pos

            self.model.data.model_config["reporting"]["case_name"]["value"] = \
                f"{self.scenario}_neg_E_{int(emission_reduction_percentage*100)}pct_cy{self.climate_year}_co2_tax{CO2_PRICE}"
            self.model._optimize_north_sea_dac(pos_limit=pos_limit, neg_target=neg_target)
