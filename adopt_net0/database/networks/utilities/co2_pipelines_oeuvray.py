import pandas as pd
from pathlib import Path
import numpy as np
import math


class CO2Transport_Oeuvray:
    """
    Calculates cost CO2 transport cost onshore or offshore, based on distance and inlet/outloet pressure.

    Minimizes the levelized cost of CO2 transport taking into account also electricity cost for compression

    Algorithm is based on Pauline Oeuvray, Johannes Burger, Simon Roussanaly, Marco Mazzotti, Viola Becattini (2024):
    Multi-criteria assessment of inland and offshore carbon dioxide transport options, Journal of Cleaner Production,
    """

    def __init__(self):
        super().__init__()

        input_path = Path(__file__).parent.parent.parent / Path(
            "./data/networks/co2_transport_oeuvray/"
        )

        fluid_properties_input_path = input_path / "CO2IsothermalProperties.xlsx"
        universal_data_input_path = input_path / "OtherData.xlsx"

        # input data
        self.length_km = None
        self.timeframe = None
        self.m_kg_per_s = None
        self.force_phase = None
        self.phase = None
        self.electricity_price_eur_per_mw = None
        self.operating_hours_per_a = None
        self.p_initial_mpa = None
        self.terrain = None
        # results
        self.current_best_results = {}
        self.optimal_configuration = None

        # Input units
        self.currency_in = "EUR"
        self.financial_year_in = 2024

        # Fluid Properties
        self.fluid_properties = {}
        self.fluid_properties["277K"] = pd.read_excel(
            fluid_properties_input_path, "277K"
        ).set_index("Pressure (MPa)")
        self.fluid_properties["288K"] = pd.read_excel(
            fluid_properties_input_path, "288K"
        ).set_index("Pressure (MPa)")

        # Universal data
        self.universal_data = pd.read_excel(
            universal_data_input_path, "Universal", index_col=0
        )["Value"].to_dict()

        # Terrain specific data
        self.terrain_specific_data = {}
        self.terrain_specific_data["gas"] = pd.read_excel(
            universal_data_input_path, "Terrain_specific_gas", index_col=0
        ).to_dict()
        self.terrain_specific_data["liquid"] = pd.read_excel(
            universal_data_input_path, "Terrain_specific_liquid", index_col=0
        ).to_dict()
        self.terrain_specific_data["gas"]["Offshore"]["OD_NPS"] = (
            pd.read_excel(universal_data_input_path, "OD_NPS", index_col=0, header=None)
            .loc["Offshore"]
            .to_numpy()
        )
        self.terrain_specific_data["gas"]["Onshore"]["OD_NPS"] = (
            pd.read_excel(universal_data_input_path, "OD_NPS", index_col=0, header=None)
            .loc["Onshore"]
            .to_numpy()
        )
        self.terrain_specific_data["liquid"]["Offshore"]["OD_NPS"] = (
            pd.read_excel(universal_data_input_path, "OD_NPS", index_col=0, header=None)
            .loc["Offshore"]
            .to_numpy()
        )
        self.terrain_specific_data["liquid"]["Onshore"]["OD_NPS"] = (
            pd.read_excel(universal_data_input_path, "OD_NPS", index_col=0, header=None)
            .loc["Onshore"]
            .to_numpy()
        )

        # Steel data
        self.steel_data = pd.read_excel(
            universal_data_input_path, "Steel_data", index_col=0
        )

        # Cost parameters
        self.cost_compressors = pd.read_excel(
            universal_data_input_path, "Compressor_costs", index_col=0
        )["Value"].to_dict()
        self.cost_pumps = pd.read_excel(
            universal_data_input_path, "Pump_costs", index_col=0
        )["Value"].to_dict()

        # Financial indicators
        self.lifetime = None
        self.discount_rate = None
        self.unit_capex = None
        self.opex_fix = None
        self.opex_var = None

    def _preprocess_data(self):
        """
        Preprocesses data based on input
        """
        # density
        self.terrain_specific_data["liquid"]["Offshore"]["rho_kg_per_m3"] = (
            self.fluid_properties["277K"].loc[8, "Density (kg/m3)"]
        )
        self.terrain_specific_data["liquid"]["Onshore"]["rho_kg_per_m3"] = (
            self.fluid_properties["288K"].loc[8, "Density (kg/m3)"]
        )
        self.terrain_specific_data["gas"]["Offshore"]["rho_kg_per_m3"] = (
            self.fluid_properties["277K"].loc[1.5, "Density (kg/m3)"]
        )
        self.terrain_specific_data["gas"]["Onshore"]["rho_kg_per_m3"] = (
            self.fluid_properties["288K"].loc[1.5, "Density (kg/m3)"]
        )

        # viscosity
        self.terrain_specific_data["liquid"]["Offshore"]["mu_Pas"] = (
            self.fluid_properties["277K"].loc[8, "Viscosity (Pa*s)"]
        )
        self.terrain_specific_data["liquid"]["Onshore"]["mu_Pas"] = (
            self.fluid_properties["288K"].loc[8, "Viscosity (Pa*s)"]
        )
        self.terrain_specific_data["gas"]["Offshore"]["mu_Pas"] = self.fluid_properties[
            "277K"
        ].loc[1.5, "Viscosity (Pa*s)"]
        self.terrain_specific_data["gas"]["Onshore"]["mu_Pas"] = self.fluid_properties[
            "288K"
        ].loc[1.5, "Viscosity (Pa*s)"]

        if self.timeframe == "near-term":
            self.steel_data = self.steel_data[
                self.steel_data["available_near_term"] == 1
            ]
        elif self.timeframe == "mid-term":
            self.steel_data = self.steel_data[
                self.steel_data["available_mid_term"] == 1
            ]
        elif self.timeframe == "long-term":
            self.steel_data = self.steel_data[
                self.steel_data["available_long_term"] == 1
            ]
        else:
            ValueError("Time frame not available")

    def _calculate_pipeline_configuration(
        self, pinlet_mpa, poutlet_mpa, id_calc_m, delta_p_inlet
    ):

        # Get data
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        # Calculate v, re, f
        v_m_per_s = self._calculate_velocity(id_calc_m)
        re = self._calculate_reynolds(id_calc_m, v_m_per_s)
        f = self._calculate_darcyweisbach(id_calc_m, re)

        while pinlet_mpa <= terrain_data["PinletMAX_MPa"]:
            n_pump = 0
            while n_pump <= terrain_data["Npump_max"]:
                print(f"Calculating {round(pinlet_mpa,2)} MPA with {n_pump} Pumps")
                delta_p_design_pa_per_m = self._calculate_design_pressure_drop(
                    pinlet_mpa, poutlet_mpa, n_pump
                )

                l_pump_m = self._calculate_max_distance_pumps(
                    pinlet_mpa, poutlet_mpa, delta_p_design_pa_per_m, self.terrain
                )

                id_calc_m = self._calculate_inner_diameter(
                    pinlet_mpa, poutlet_mpa, f, l_pump_m, delta_p_design_pa_per_m
                )

                if (
                    max(
                        terrain_data["OD_NPS"]
                        - 2 * terrain_data["dtRatio"] * terrain_data["OD_NPS"]
                        - id_calc_m
                    )
                    > 0
                ):
                    best_steel_grade = self._find_best_steel_grade(
                        id_calc_m, pinlet_mpa, poutlet_mpa
                    )
                    current_result = self._minimize_levelized_cost(
                        best_steel_grade, pinlet_mpa, poutlet_mpa
                    )

                    if current_result["lc"] < self.current_best_results["lc"]:
                        self.current_best_results = current_result
                        print(
                            f"New best config found with steel grade {best_steel_grade.index[0]}"
                        )

                n_pump = n_pump + 1
            pinlet_mpa = pinlet_mpa + delta_p_inlet

        return self.current_best_results

    def _find_best_steel_grade(self, id_calc_m, pinlet_mpa, poutlet_mpa):
        """
        Calculates min cost for different steel grades

        :param float id_calc_m: starting inner diameter of pipe in m
        :param float p_inlet_mpa: inlet pressure in MPa
        :param float p_outlet_mpa: outlet pressure in MPa
        :return: cost factors for different steel grades
        :rtype: pd.DataFrame
        """

        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        # Max operating pressure
        max_p_mpa = self._calculate_max_operating_pressure(pinlet_mpa)

        pipe_cost_for_different_steel_grades = []
        for idx, steel_grade in self.steel_data.iterrows():
            # Starting value
            id_nps_m = 0

            while id_calc_m > id_nps_m:
                od_nps_m = terrain_data["OD_NPS"][
                    terrain_data["OD_NPS"] - id_calc_m > 0
                ]

                for od in od_nps_m:
                    t_m = self._calculate_pipe_thickness(
                        od, max_p_mpa, steel_grade.S_MPa
                    )
                    id_nps_m = od - 2 * t_m
                    od_nps_chosen = od
                    if id_nps_m > id_calc_m:
                        break

                v_m_per_s = self._calculate_velocity(id_nps_m)
                re = self._calculate_reynolds(
                    id_nps_m,
                    v_m_per_s,
                )
                f = self._calculate_darcyweisbach(id_nps_m, re)
                delta_p_act_pa_m = self._calculate_actual_pressure_drop(f, id_nps_m)
                l_pump_m = self._calculate_max_distance_pumps(
                    pinlet_mpa, poutlet_mpa, delta_p_act_pa_m, terrain="Onshore"
                )

                n_pumps = self._calculate_number_pumps(l_pump_m, terrain="Onshore")
                delta_p_design_pa_m = self._calculate_design_pressure_drop(
                    pinlet_mpa, poutlet_mpa, n_pumps
                )

                id_calc_m = self._calculate_inner_diameter(
                    pinlet_mpa, poutlet_mpa, f, l_pump_m, delta_p_design_pa_m
                )

            pipe_cost = self._calculate_pipe_costs(
                t_m, od_nps_chosen, steel_grade.Csteel_EUR_per_kg
            )
            pipe_cost["id_nps_m"] = id_nps_m
            pipe_cost["t_m"] = t_m
            pipe_cost["od_nps_m"] = od_nps_chosen

            pipe_cost_for_different_steel_grades.append(pipe_cost)

        pipe_costs = pd.DataFrame(
            pipe_cost_for_different_steel_grades, index=self.steel_data.index
        )

        return pipe_costs[pipe_costs["capex_total"] == pipe_costs["capex_total"].min()]

    def _calculate_velocity(self, id_calc_m):
        """
        Calculates velocity through a pipeline

        :param float id_calc_m: inner diameter in m
        :return: flow rate in m/s
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        return (
            4
            * self.m_kg_per_s
            / (id_calc_m**2 * math.pi * terrain_data["rho_kg_per_m3"])
        )

    def _calculate_reynolds(self, IDNPS_m, v_m_per_s):
        """
        Calculates reynolds number

        :param float IDNPS_m: inner diameter in m
        :param float v_m_per_s: flow rate in m/s
        :return: reynolds number
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        return (
            terrain_data["rho_kg_per_m3"] * IDNPS_m * v_m_per_s / terrain_data["mu_Pas"]
        )

    def _calculate_darcyweisbach(self, IDNPS_m, Re):
        """
        Calculates darcy-weisbach friction factor

        :param float IDNPS_m: inner diameter in m
        :param float Re: reynolds number
        :return: friction factor
        :rtype: float
        """
        return (
            1
            / (
                -1.8
                * math.log10(
                    (self.universal_data["epsilon_m"] / IDNPS_m / 3.7) ** 1.11
                    + 6.9 / Re
                )
            )
        ) ** 2

    def _calculate_compressor_outlet(
        self, f, l_pump_m, poutlet_mpa, pinlet_mpa, id_nps_m
    ):
        """
        Caclulates actual outlet pressure of the compressor for gas transport in Pa

        :param float f: Darcy-Weisbach friction factor
        :param float l_pump_m: lmax distance between pumps in m
        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet in MPa
        :return: Actual outlet pressure of the compressor in Pa
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        p_ave_pa = self._calculate_average_pressure(poutlet_mpa * 1e6, pinlet_mpa * 1e6)
        z_ave = self._calculate_compressibility_factor(p_ave_pa, terrain_data["T_degC"])
        t_ave_k = terrain_data["T_degC"] + 273.15

        return (
            16
            * z_ave
            * self.universal_data["R_J_per_mol_per_K"]
            * t_ave_k
            * self.m_kg_per_s**2
            * f
            * l_pump_m
            / (math.pi**2.0 * id_nps_m**5.0 * self.universal_data["M_kg_per_mol"])
            + 2
            * self.universal_data["g_m_per_s2"]
            * p_ave_pa**2.0
            * self.universal_data["M_kg_per_mol"]
            * self.universal_data["z_m"]
            / (z_ave * t_ave_k * self.universal_data["R_J_per_mol_per_K"])
            + (poutlet_mpa * 1e6) ** 2
        ) ** 0.5

    def _calculate_pressure_last_pump(
        self, poutlet_mpa, l_pump_m, n_pumps, delta_p_act_pa_m
    ):
        """
        Calculates outlet pressure of the last pump

        :param float poutlet_mpa: pressure at outlet in MPa
        :param float l_pump_m: lmax distance between pumps in m
        :param float n_pump: number of pumps
        :param float delta_p_act_pa_m: actual pressure drop in Pa/m
        :return: outlet pressure of last pump in pa
        :rtype: float
        """
        return (
            poutlet_mpa * 1e6
            + (self.length_km * 1000 - l_pump_m * n_pumps) * delta_p_act_pa_m
        )

    def _calculate_design_pressure_drop(self, pinlet_mpa, poutlet_mpa, n_pump):
        """
        Calculates design pressure drop

        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet
        :param float n_pump: number of pumps
        :return: pressure drop in Pa
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        pinlet_pa = pinlet_mpa * 1e6
        poutlet_pa = poutlet_mpa * 1e6
        l_m = self.length_km * 1e3
        return (
            (pinlet_pa - poutlet_pa) * (n_pump + 1)
            + self.universal_data["g_m_per_s2"]
            * terrain_data["rho_kg_per_m3"]
            * self.universal_data["z_m"]
        ) / l_m

    def _calculate_compressor_energy(
        self, poutlet_mpa, pinlet_mpa, initial_compression=0
    ):
        """
        Calculates specific energy of a compressor depending on phase

        :param p_out_pa: pressure at outlet in Pa
        :param p_in_pa: pressure at inlet in Pa
        :return: compression energy in MJ/kg
        :rtype: float
        """
        if self.phase == "gas":
            return self._calculate_compressor_energy_gas(
                poutlet_mpa, pinlet_mpa, initial_compression=initial_compression
            )
        else:
            return self._calculate_compressor_energy_liquid(poutlet_mpa, pinlet_mpa)

    def _calculate_compressor_energy_liquid(self, poutlet_mpa, pinlet_mpa):
        """
        Calculates specific energy of a gas compressor

        :param p_out_pa: pressure at outlet in Pa
        :param p_in_pa: pressure at inlet in Pa
        :return: compression energy in MJ/kg
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        return (poutlet_mpa - pinlet_mpa) / (
            self.universal_data["etaPump"] * terrain_data["rho_kg_per_m3"]
        )

    def _calculate_compressor_energy_gas(
        self, poutlet_mpa, pinlet_mpa, initial_compression=0
    ):
        """
        Calculates specific energy of a gas compressor

        :param p_out_pa: pressure at outlet in Pa
        :param p_in_pa: pressure at inlet in Pa
        :return: compression energy in MJ/kg
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        pr = self.universal_data["PR"]
        pinlet_pa = pinlet_mpa * 1e6
        poutlet_pa = poutlet_mpa * 1e6
        if initial_compression:
            z = 0.994799474
            t_comp_k = 303.15
        else:
            # Onshore compression
            z = 0.910912883
            t_comp_k = 15 + 273.15

        if poutlet_pa > 3e6:
            # liquid
            n_stages = math.floor(math.log(poutlet_pa / pinlet_pa) / math.log(pr))
            p1_pa = pinlet_pa * pr**n_stages
        else:
            # gas
            n_stages = math.ceil(math.log(poutlet_pa / pinlet_pa) / math.log(pr))
            p1_pa = poutlet_pa
            pr = (p1_pa / pinlet_pa) ** (1 / n_stages)
        dp_pump = poutlet_pa - p1_pa

        e_comp_J_per_kg = z * self.universal_data[
            "R_J_per_mol_per_K"
        ] * t_comp_k * n_stages * self.universal_data["kappa"] * (
            pr ** ((self.universal_data["kappa"] - 1) / self.universal_data["kappa"])
            - 1
        ) / (
            self.universal_data["M_kg_per_mol"]
            * self.universal_data["etaIso"]
            * self.universal_data["etaMech"]
            * (self.universal_data["kappa"] - 1)
        ) + dp_pump / (
            self.universal_data["etaPump"] * terrain_data["rho_kg_per_m3"]
        )

        return e_comp_J_per_kg / 1e6

    def _calculate_max_distance_pumps(
        self, pinlet_mpa, poutlet_mpa, delta_p_pa_per_m, terrain
    ):
        """
        Calculates distance between pumping stations in m

        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet in MPa
        :param float delta_p_pa_per_m: pressure drop in Pa
        :return: maximum distance between pumping stations in m
        :rtype: float
        """
        pinlet_pa = pinlet_mpa * 1e6
        poutlet_pa = poutlet_mpa * 1e6

        if terrain == "Onshore":
            return (pinlet_pa - poutlet_pa) / delta_p_pa_per_m
        else:
            return self.length_km * 1000

    def _calculate_number_pumps(self, l_pump_m, terrain):
        """
        Calculates number of pumps required

        :param float l_pump_m: maximum distance between pumping stations in m
        :return: number of pumping stations
        :rtype: float
        """

        if terrain == "Onshore":
            return math.floor(self.length_km * 1000.0 / l_pump_m)
        else:
            return 0

    def _calculate_inner_diameter(
        self, pinlet_mpa, poutlet_mpa, f, l_pump_m, delta_p_design_pa_per_m
    ):
        """
        Calculates required inner diameter in m

        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet in MPa
        :param float f: Darcy-Weisbach friction factor
        :param float l_pump_m: lmax distance between pumps in m
        :return: required inner diameter for gaseous transport in m
        :rtype: float
        """
        if self.phase == "gas":
            return self._calculate_inner_diameter_gas(
                pinlet_mpa, poutlet_mpa, f, l_pump_m
            )
        else:
            return self._calculate_inner_diameter_liquid(f, delta_p_design_pa_per_m)

    def _calculate_inner_diameter_liquid(self, f, delta_p_design_pa_per_m):
        """
        Calculates required inner diameter for liquid transport in m

        :param float f: Darcy-Weisbach friction factor
        :param float delta_p_design_pa_per_m: design pressure drop in Pa/m]
        :return: required inner diameter for gaseous transport in m
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        return (
            8
            * f
            * self.m_kg_per_s**2
            / (math.pi**2 * terrain_data["rho_kg_per_m3"] * delta_p_design_pa_per_m)
        ) ** (1 / 5)

    def _calculate_inner_diameter_gas(self, pinlet_mpa, poutlet_mpa, f, l_pump_m):
        """
        Calculates required inner diameter for gaseous transport in m

        :param float pinlet_mpa: pressure at inlet in MPa
        :param float poutlet_mpa: pressure at outlet in MPa
        :param float f: Darcy-Weisbach friction factor
        :param float l_pump_m: lmax distance between pumps in m
        :return: required inner diameter for gaseous transport in m
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]
        pinlet_pa = pinlet_mpa * 1e6
        poutlet_pa = poutlet_mpa * 1e6
        t_ave_k = terrain_data["T_degC"] + 273.15

        p_ave_pa = self._calculate_average_pressure(poutlet_pa, pinlet_pa)
        z_ave = self._calculate_compressibility_factor(p_ave_pa, terrain_data["T_degC"])
        return (
            -16
            * z_ave**2
            * self.universal_data["R_J_per_mol_per_K"] ** 2
            * t_ave_k**2
            * self.m_kg_per_s**2
            * f
            * l_pump_m
            / (
                math.pi**2
                * (
                    self.universal_data["M_kg_per_mol"]
                    * z_ave
                    * t_ave_k
                    * self.universal_data["R_J_per_mol_per_K"]
                    * (poutlet_pa**2 - pinlet_pa**2)
                    + 2
                    * self.universal_data["g_m_per_s2"]
                    * p_ave_pa**2
                    * self.universal_data["M_kg_per_mol"] ** 2.0
                    * self.universal_data["z_m"]
                )
            )
        ) ** (1 / 5)

    def _calculate_pipe_thickness(self, od_nps_m, max_p_mpa, s_mpa):
        """
        Calculates required inner diameter for gaseous transport in m

        :param float od_nps_m: outer diameter of the nominal pipe size in m
        :param float max_p_mpa: maximum allowable operating pressure in MPa
        :param float s_mpa: minimum yield stress in MPa
        :param float e: longitudinal joint factor
        :return: pipe thickness in m
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        t_m = (
            od_nps_m
            * max_p_mpa
            / (2.0 * s_mpa * terrain_data["F"] * self.universal_data["E"])
            + self.universal_data["CA_m"]
        )
        if t_m / od_nps_m < terrain_data["dtRatio"]:
            t_m = od_nps_m * terrain_data["dtRatio"]

        return math.ceil(t_m * 2000) / 2000

    def _calculate_max_operating_pressure(self, p_inlet_mpa):
        """
        Calculates required inner diameter for gaseous transport in m

        :param float p_inlet_mpa: inlet pressure in MPa
        :return: maximum allowable operating pressure in MPa
        :rtype: float
        """
        return math.ceil(p_inlet_mpa * 1.1 * 10) / 10

    def _calculate_actual_pressure_drop(self, f, id_nps_m):
        """
        Calculates design pressure drop

        :param float f: Darcy-Weisbach friction factor
        :param float id_nps_m: inner diameter of the nominal pipe size in m
        :return: pressure drop in Pa/m
        :rtype: float
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        return (
            8
            * f
            * self.m_kg_per_s**2
            / (math.pi**2 * terrain_data["rho_kg_per_m3"] * id_nps_m**5)
        )

    def _calculate_average_pressure(self, poutlet_pa, pinlet_pa):
        """
        Calculates average pressure in pa

        :param float pinlet_pa: pressure at inlet in Pa
        :param float poutlet_pa: pressure at outlet in Pa
        :return: average pressure in Pa
        :rtype: float
        """

        return (
            2
            * (
                poutlet_pa
                + pinlet_pa
                - poutlet_pa * pinlet_pa / (poutlet_pa + pinlet_pa)
            )
            / 3
        )

    def _calculate_compressibility_factor(self, p_ave_pa, t_ave_c):
        """
        Calculates average compressability factor

        :param float p_ave_pa: average pressure in Pa
        :param float t_ave_c: average temperature in C
        :return: compressibility factor
        :rtype: float
        """
        if t_ave_c == 4:
            fluid_properties = self.fluid_properties["277K"]
        elif t_ave_c == 15:
            fluid_properties = self.fluid_properties["288K"]
        else:
            raise ValueError("Temperature is not allowed")

        return np.interp(
            p_ave_pa * 1e-6,
            fluid_properties.index,
            fluid_properties["Compressibility factor Z"],
        )

    def _calculate_pipe_costs(self, t_m, od_nps_m, c_steel_eur_per_kg):
        """
        Calculates pipeline costs

        :param float t_m: pipe thickness in m
        :param float od_nps_m: outer diameter in m
        :param float c_steel_eur_per_kg: cost of steel in eur/kg
        :return:
        """
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        cost_factors = {}
        cost_factors["material"] = float(
            t_m
            * math.pi
            * (od_nps_m - t_m)
            * self.length_km
            * 1000
            * self.universal_data["rhoSteel_kg_per_m3"]
            * c_steel_eur_per_kg
            * self.universal_data["SteelFactor"]
        )
        cost_factors["labor"] = float(
            od_nps_m * self.length_km * 1000 * self.universal_data["Clab_EUR_per_m2"]
        )
        cost_factors["row"] = float(
            self.length_km * 1000 * terrain_data["CROW_EUR_per_m"]
        )
        cost_factors["misc"] = float(
            self.universal_data["mu_misc"]
            * (cost_factors["material"] + cost_factors["labor"])
        )
        cost_factors["capex_total"] = float(sum(cost_factors.values()))
        cost_factors["opex_fix"] = float(
            cost_factors["capex_total"] * self.universal_data["muOMpipe"]
        )

        return cost_factors

    def _calculate_compressor_cost(self, w_mw, phase):
        """
        Investment costs for pumps of compressors

        :param float w_mw: pump capacity in MW
        :return: pump cost in eur
        :rtype: float
        """
        if phase == "gas":
            return self._calculate_compressor_cost_gas(w_mw)
        else:
            return self._calculate_compressor_cost_liquid(w_mw)

    def _calculate_compressor_cost_liquid(self, w_mw):
        """
        Investment costs of pump in eur

        :param float w_mw: pump capacity in MW
        :return: pump cost in eur
        :rtype: float
        """
        n = math.ceil(w_mw / self.cost_pumps["WpumpMAX_MW"])

        cost = self.cost_pumps["Ipump0_EUR"] * ((w_mw * 1e3) ** 0.58) * n**0.32
        return cost

    def _calculate_compressor_cost_gas(self, w_mw):
        """
        Investment costs of compressor in eur

        :param float w_mw: compressor capacity in MW
        :return: compressor cost in eur
        :rtype: float
        """

        n = math.ceil(w_mw / self.cost_compressors["WcompMAX_MW"])
        if n == 0:
            w1 = 0
        else:
            w1 = w_mw / n

        cost = (
            self.cost_compressors["Icomp0_EUR"]
            * ((w1 / self.cost_compressors["Wcomp0_MW"]) ** self.cost_compressors["y"])
            * n ** self.cost_compressors["me"]
        )

        if cost < 0:
            cost = 0

        return cost

    def _calculate_recompression_energy_cost(self, w_comp_mw_total):
        """
        Calculates total recompression energy costs

        :param float w_comp_mw_total: total recompression capacity
        :return: total recompression energy cost
        :rtype: float
        """
        return (
            w_comp_mw_total
            * self.operating_hours_per_a
            * self.electricity_price_eur_per_mw
        )

    def _calculate_levelized_cost(self, result):

        cr_pipe = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.universal_data["z_pipe"]
            / ((1 + self.discount_rate) ** self.universal_data["z_pipe"] - 1)
        )
        cr_pump_compressions = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.universal_data["z_pumpcomp"]
            / ((1 + self.discount_rate) ** self.universal_data["z_pumpcomp"] - 1)
        )

        return (
            cr_pipe * result["capex_pipe"]
            + cr_pump_compressions
            * (result["capex_recompression"] + result["capex_initial_compression"])
            + result["opex_pipe"]
            + result["opex_fix_compression"]
            + result["opex_energy_recompression"]
            + result["opex_energy_initial_compression"]
        ) / (self.m_kg_per_s * self.operating_hours_per_a * 3.6)

    def _minimize_levelized_cost(
        self, best_steel_grade_config, pinlet_mpa, poutlet_mpa
    ):
        terrain_data = self.terrain_specific_data[self.phase][self.terrain]

        id_nps_m = best_steel_grade_config["id_nps_m"].iloc[0]
        v_m_per_s = self._calculate_velocity(id_nps_m)
        current_result = {}
        current_result["lc"] = 1e10

        if (v_m_per_s >= terrain_data["vRange_min"]) & (
            v_m_per_s <= terrain_data["vRange_max"]
        ):
            re = self._calculate_reynolds(id_nps_m, v_m_per_s)
            f = self._calculate_darcyweisbach(id_nps_m, re)
            delta_p_act_pa_m = self._calculate_actual_pressure_drop(f, id_nps_m)

            l_pump_m = self._calculate_max_distance_pumps(
                pinlet_mpa, poutlet_mpa, delta_p_act_pa_m, self.terrain
            )
            n_pumps = self._calculate_number_pumps(l_pump_m, self.terrain)

            # RECOMPRESSION COST AND ENERGY
            if n_pumps == 0:
                # No recompression stations
                # e_comp_MJ_per_kg_all_but_last = 0
                # e_comp_MJ_per_kg_last = 0
                # e_comp_kJ_per_kg_total = 0
                capex_recompression_eur = 0
                opex_energy_recompression_eur_per_y = 0
                w_recompression_mw_total = 0
            else:
                # Recompression stations
                p_outlet_last_pump_pa = self._calculate_pressure_last_pump(
                    poutlet_mpa, l_pump_m, n_pumps, delta_p_act_pa_m
                )
                e_comp_MJ_per_kg_all_but_last = self._calculate_compressor_energy(
                    pinlet_mpa, poutlet_mpa
                )
                e_comp_MJ_per_kg_last = self._calculate_compressor_energy(
                    p_outlet_last_pump_pa * 1e-6, poutlet_mpa
                )
                e_comp_MJ_per_kg_total = (
                    e_comp_MJ_per_kg_last + e_comp_MJ_per_kg_all_but_last * n_pumps
                )
                w_recompressor_all_but_last_MW = (
                    e_comp_MJ_per_kg_all_but_last * self.m_kg_per_s
                )
                w_recompressor_last_MW = e_comp_MJ_per_kg_last * self.m_kg_per_s
                w_recompression_mw_total = e_comp_MJ_per_kg_total * self.m_kg_per_s

                capex_recompression_eur = self._calculate_compressor_cost(
                    w_recompressor_all_but_last_MW, self.phase
                ) * (n_pumps - 1) + self._calculate_compressor_cost(
                    w_recompressor_last_MW, self.phase
                )
                opex_energy_recompression_eur_per_y = (
                    self._calculate_recompression_energy_cost(
                        w_recompressor_all_but_last_MW
                    )
                    * (n_pumps - 1)
                    + self._calculate_recompression_energy_cost(w_recompressor_last_MW)
                )

            if self.phase == "gas":
                p_2 = (
                    self._calculate_compressor_outlet(
                        f, l_pump_m, poutlet_mpa, pinlet_mpa, id_nps_m
                    )
                    * 1e-6
                )
            else:
                if self.terrain == "Offshore":
                    p_2 = (
                        self._calculate_pressure_last_pump(
                            poutlet_mpa, l_pump_m, 0, delta_p_act_pa_m
                        )
                        * 1e-6
                    )
                else:
                    p_2 = pinlet_mpa

            # INITIAL COMPRESSION COST AND ENERGY
            e_initial_compression_Mj_per_kg = self._calculate_compressor_energy_gas(
                p_2, self.p_initial_mpa, initial_compression=1
            )
            w_initial_compression_mw = e_initial_compression_Mj_per_kg * self.m_kg_per_s
            capex_initial_compression_eur = self._calculate_compressor_cost(
                w_initial_compression_mw, phase="gas"
            )
            opex_energy_initial_compression_eur_per_y = (
                self._calculate_recompression_energy_cost(w_initial_compression_mw)
            )

            # CAPEX
            current_result["capex_pipe"] = best_steel_grade_config["capex_total"].iloc[
                0
            ]
            current_result["capex_recompression"] = capex_recompression_eur
            current_result["capex_initial_compression"] = capex_initial_compression_eur
            current_result["capex_total"] = (
                current_result["capex_pipe"]
                + current_result["capex_recompression"]
                + current_result["capex_initial_compression"]
            )

            # OPEX
            current_result["opex_pipe"] = best_steel_grade_config["opex_fix"].iloc[0]
            current_result["opex_energy_recompression"] = (
                opex_energy_recompression_eur_per_y
            )
            current_result["opex_energy_initial_compression"] = (
                opex_energy_initial_compression_eur_per_y
            )
            current_result["opex_fix_compression"] = (
                capex_recompression_eur + capex_initial_compression_eur
            ) * self.universal_data["muOMpumpcomp"]
            current_result["opex_fix"] = current_result["opex_fix_compression"]
            current_result["opex_var_energy"] = (
                current_result["opex_energy_recompression"]
                + current_result["opex_energy_initial_compression"]
            )

            # Energy consumption
            current_result["energy_compression_specific_MWh_per_kg"] = (
                w_initial_compression_mw + w_recompression_mw_total
            ) / self.m_kg_per_s

            # Design
            current_result["steel_grade"] = best_steel_grade_config.index[0]
            current_result["n_pumps"] = n_pumps
            current_result["id_nps_m"] = id_nps_m
            current_result["l_pump_km"] = l_pump_m / 1000
            current_result["poutlet_mpa"] = poutlet_mpa
            current_result["pinlet_mpa"] = pinlet_mpa
            current_result["p_initial_mpa"] = self.p_initial_mpa

            current_result["lc"] = self._calculate_levelized_cost(current_result)

        return current_result


class CO2Chain_Oeuvray(CO2Transport_Oeuvray):
    def _preprocess_data(self):
        super()._preprocess_data()
        # max number of pumps
        for phase in ["gas", "liquid"]:
            self.terrain_specific_data[phase]["Offshore"]["Npump_max"] = 0
            self.terrain_specific_data[phase]["Onshore"]["Npump_max"] = (
                self.length_km / 40
            )

    def calculate_cost(
        self,
        currency,
        year,
        discount_rate,
        timeframe,
        length_km,
        m_kg_per_s,
        terrain,
        electricity_price_eur_per_mw,
        operating_hours_per_a,
        p_inlet_bar,
        poutlet_bar=None,
    ):
        """
        Calculates the transport cost of CO2, including pipelines, compression and recompression

        :param str currency: currency of cost
        :param int year: year to use
        :param float discount_rate: discount rate
        :param str timeframe: determines which steel grades are available, can be 'short-term', 'mid-term', or 'long-term'
        :param float length_km: distance to cover in km
        :param float m_kg_per_s: mass flow rate of CO2 in kg/s
        :param str terrain: 'Offshore' or 'Onshore'
        :param float electricity_price_eur_per_mw: used to minimize levelized cost (EUR/MWh)
        :param int operating_hours_per_a: number of operating hours per year
        :param float p_inlet_bar: inlet pressure in bar (beginning of pipeline)
        :param float poutlet_bar: outlet pressure in bar (end of pipeline)
        :return: dictonary of cost and energy comsumption indicators of lowest levelized cost configuration
        :rtype: dict
        """
        self.currency_out = currency
        self.financial_year_out = year
        self.length_km = length_km
        self.timeframe = timeframe
        self.m_kg_per_s = m_kg_per_s
        self.terrain = terrain
        self.electricity_price_eur_per_mw = electricity_price_eur_per_mw
        self.operating_hours_per_a = operating_hours_per_a
        self.p_initial_mpa = p_inlet_bar / 10
        self.discount_rate = discount_rate

        self._preprocess_data()

        self.current_best_results["lc"] = 1e3

        if poutlet_bar is None:
            self.force_phase = None
            poutlet_mpa = 1.5
        elif poutlet_bar / 10 < 3e6:
            self.force_phase = "gas"
            poutlet_mpa = poutlet_bar / 10
        elif poutlet_bar / 10 >= 3e6:
            self.force_phase = "liquid"
            poutlet_mpa = poutlet_bar / 10
        else:
            raise ValueError("Given outlet pressure not supported")

        if self.force_phase == "gas":
            # Starting values
            self.phase = "gas"
            pinlet_mpa = 1.6
            id_calc_m = 0.5
            delta_p_inlet = 0.1
            optimal_configuration = self._calculate_pipeline_configuration(
                pinlet_mpa, poutlet_mpa, id_calc_m, delta_p_inlet
            )
        elif self.force_phase == "liquid":
            # Starting values
            self.phase = "liquid"
            pinlet_mpa = 9
            id_calc_m = 0.5
            delta_p_inlet = 1
            optimal_configuration = self._calculate_pipeline_configuration(
                pinlet_mpa, poutlet_mpa, id_calc_m, delta_p_inlet
            )
        else:
            # Starting values
            self.phase = "gas"
            pinlet_mpa = 1.6
            poutlet_mpa = 1.5
            id_calc_m = 0.5
            delta_p_inlet = 0.1
            self._calculate_pipeline_configuration(
                pinlet_mpa, poutlet_mpa, id_calc_m, delta_p_inlet
            )
            # Starting values
            self.phase = "liquid"
            pinlet_mpa = 9
            poutlet_mpa = 8
            id_calc_m = 0.5
            delta_p_inlet = 1
            optimal_configuration = self._calculate_pipeline_configuration(
                pinlet_mpa, poutlet_mpa, id_calc_m, delta_p_inlet
            )

        # Financial indicators
        self.optimal_configuration = optimal_configuration
        self.lifetime = min(self.universal_data)
        self.unit_capex = None
        self.opex_fix = None
        self.opex_var = None

        cr_pipe = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.universal_data["z_pipe"]
            / ((1 + self.discount_rate) ** self.universal_data["z_pipe"] - 1)
        )
        cr_pump_compressions = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.universal_data["z_pumpcomp"]
            / ((1 + self.discount_rate) ** self.universal_data["z_pumpcomp"] - 1)
        )

        cost_pipeline = {}
        cost_pipeline["unit_capex"] = self._convert_currency(
            self.optimal_configuration["capex_pipe"] / self.m_kg_per_s
        )
        cost_pipeline["opex_var"] = 0
        cost_pipeline["opex_fix"] = self.optimal_configuration["opex_pipe"] / (
            cost_pipeline["unit_capex"] * cr_pipe
        )
        cost_pipeline["lifetime"] = self.universal_data["z_pipe"]

        cost_compression = {}
        cost_compression["unit_capex"] = self._convert_currency(
            (
                self.optimal_configuration["capex_total"]
                - self.optimal_configuration["capex_pipe"]
            )
            / self.m_kg_per_s
        )
        cost_compression["opex_var"] = 0
        cost_compression["opex_fix"] = self.optimal_configuration[
            "opex_fix_compression"
        ] / (cost_compression["unit_capex"] * cr_pump_compressions)
        cost_compression["lifetime"] = self.universal_data["z_pumpcomp"]

        energy_requirements = {}
        energy_requirements["specific_compression_energy"] = self.optimal_configuration[
            "energy_compression_specific_MWh_per_kg"
        ]

        results = {}
        results["cost_pipeline"] = cost_pipeline
        results["cost_compression"] = cost_compression
        results["energy_requirements"] = energy_requirements
        results["configuration"] = self.optimal_configuration

        return results


class CO2Pipeline_Oeuvray(CO2Transport_Oeuvray):
    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.p_outlet_mpa = None
        self.p_inlet_mpa = None
        self.id_nps_m

    def calculate_cost(
        self,
        currency,
        year,
        discount_rate,
        timeframe,
        length_km,
        m_kg_per_s,
        terrain,
        p_inlet_bar,
        p_outlet_bar,
        id_nps_m,
    ):
        self.currency_out = currency
        self.financial_year_out = year
        self.m_kg_per_s = m_kg_per_s
        self.p_inlet_mpa = p_inlet_bar / 10
        self.p_outlet_mpa = p_outlet_bar / 10
        self.discount_rate = discount_rate
        self.terrain = terrain

        self._preprocess_data()

        # Determine phase
        if self.p_outlet_mpa < 3e6:
            self.phase = "gas"
        elif self.p_outlet_mpa >= 3e6:
            self.phase = "liquid"


class CO2Compression_Oeuvray(CO2Transport_Oeuvray):
    """
    Calculates the compressor costs and specific compression energy isolated from the CO2 transport chain

    - unit_capex is in [selected currency] / kg(CO2) / s
    - opex_fix is in % of annualized capex using the given discount rate and lifetime
    - opex_var is 0
    - specific_compression_energy_mwh_per_t is in MWh/t(CO2)
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.p_outlet_mpa = None
        self.p_inlet_mpa = None

    def calculate_cost(
        self,
        currency,
        year,
        discount_rate,
        m_kg_per_s,
        terrain,
        p_inlet_bar,
        p_outlet_bar,
    ):
        """
        Calculates cost and compression energy


        :param str currency: currency of cost
        :param int year: year to use
        :param float discount_rate: discount rate
        :param float m_kg_per_s: mass flow rate of CO2 in kg/s
        :param float p_inlet_bar: inlet pressure in bar (beginning of pipeline)
        :param float poutlet_bar: outlet pressure in bar (end of pipeline)
        :return: dictonary of cost and energy comsumption indicators
        :rtype: dict
        """
        self.currency_out = currency
        self.financial_year_out = year
        self.m_kg_per_s = m_kg_per_s
        self.p_inlet_mpa = p_inlet_bar / 10
        self.p_outlet_mpa = p_outlet_bar / 10
        self.discount_rate = discount_rate
        self.terrain = terrain

        self._preprocess_data()

        # Determine phase
        if self.p_outlet_mpa < 3e6:
            self.phase = "gas"
        elif self.p_outlet_mpa >= 3e6:
            self.phase = "liquid"

        # COMPRESSION COST AND ENERGY
        e_initial_compression_Mj_per_kg = self._calculate_compressor_energy_gas(
            self.p_outlet_mpa, self.p_inlet_mpa, initial_compression=1
        )
        w_compression_mw = e_initial_compression_Mj_per_kg * self.m_kg_per_s
        w_compression_mwh_per_t = w_compression_mw / (self.m_kg_per_s * 3.6)

        capex_compression_eur = self._calculate_compressor_cost(
            w_compression_mw, phase="gas"
        )

        opex_fix = capex_compression_eur * self.universal_data["muOMpumpcomp"]

        cr_pump_compressions = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.universal_data["z_pumpcomp"]
            / ((1 + self.discount_rate) ** self.universal_data["z_pumpcomp"] - 1)
        )

        self.lifetime = min(self.universal_data)
        self.unit_capex = self._convert_currency(capex_compression_eur)
        self.opex_fix = opex_fix / (capex_compression_eur * cr_pump_compressions)
        self.opex_var = 0

        return {
            "unit_capex": self.unit_capex,
            "opex_fix": self.opex_fix,
            "opex_var": self.opex_var,
            "specific_compression_energy_mwh_per_t": w_compression_mwh_per_t,
        }
