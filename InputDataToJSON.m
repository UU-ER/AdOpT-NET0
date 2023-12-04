load('Input_Template.mat')
%% Export each network to one JSON
% save_path = 'C:\EHubversions\EHUB-Py\data\network_data\';
save_path = 'C:\EHubversions\';


all_netw = fieldnames(Data.network_data);
for netw_name = 1:size(all_netw,1)
    display(all_netw{netw_name})
    netw = Data.network_data.(all_netw{netw_name});

    % Economics
    econ_in = netw.economic;
    %     a = econ_in.r / (1 - (1 / (1 + econ_in.r)^econ_in.lifetime));
    a = 1;
    econ_out = [];
    econ_out.comment = 'CAPEX coefficients are in EUR, EUR/MW or EUR/MW/km, OPEX_variable in EUR/MWh total output, OPEX_fixed in % of CAPEX_annual';
    econ_out.CAPEX_model = econ_in.investmentapprox;
    if econ_out.CAPEX_model == 1
        econ_out.gamma1 = round(econ_in.alpha1/1000 * a,5);
        econ_out.gamma2 = round(econ_in.alpha2 * a,5);
    elseif econ_out.CAPEX_model == 2
        econ_out.gamma1 = round(econ_in.alpha1/1000 * a,5);
        econ_out.gamma2 = round(econ_in.alpha2 * a,5);
    elseif econ_out.CAPEX_model == 3
        econ_out.gamma1 = round(econ_in.alpha1(1)/1000 * a,5);
        econ_out.gamma2 = round(econ_in.alpha2(1)/1000 * a,5);
        econ_out.gamma3 = round(econ_in.alpha3(1) * a,5);
    end
    econ_out.OPEX_variable = round(econ_in.c_u * 1000,8);
    econ_out.OPEX_fixed = round(econ_in.c_m,8);

    econ_out.discount_rate = econ_in.r;
    econ_out.lifetime = econ_in.lifetime;
    econ_out.decommission_cost = 0;

    % Performance
    tec_in = netw.tech;
    performance = [];
    performance.carrier = netw.carrier;

    if contains(all_netw{netw_name}, 'electricity')
        performance.bidirectional = 1;
    else
        performance.bidirectional = 0;
    end

    performance.loss = tec_in.loss;
    performance.min_transport = tec_in.phi;
    performance.loss2emissions = tec_in.loss2emissions;
    performance.emissionfactor = 0;

    e_cons = netw.energyrequirements
    performance.energyconsumption = [];
    if ~isempty(e_cons.type_cons) && ~strcmp(all_netw{netw_name},'heat')
        for car = 1: length(e_cons.type_cons)
            if e_cons.networkConsModel == 2
                performance.energyconsumption.(e_cons.type_cons{car}).cons_model = 1;
                performance.energyconsumption.(e_cons.type_cons{car}).k_flow = e_cons.consFlow(car);
                performance.energyconsumption.(e_cons.type_cons{car}).k_flowDistance = e_cons.consFlowKm(car);
            else
                performance.energyconsumption.(e_cons.type_cons{car}).cons_model = 2;
                performance.energyconsumption.(e_cons.type_cons{car}).p = e_cons.p;
                performance.energyconsumption.(e_cons.type_cons{car}).c = e_cons.c;
                performance.energyconsumption.(e_cons.type_cons{car}).T = e_cons.T;
                performance.energyconsumption.(e_cons.type_cons{car}).eta = e_cons.eta;
                performance.energyconsumption.(e_cons.type_cons{car}).gam = e_cons.gam;
                performance.energyconsumption.(e_cons.type_cons{car}).LHV = e_cons.LHV;
            end
        end
    end

    netw_out.Economics = econ_out;
    netw_out.NetworkPerf = performance;
    netw_out.size_min = 0;
    netw_out.size_max = min(netw.Smax(end) / 1000, 10000);
    netw_out.size_is_int = 0;
    netw_out.decommission = 0;

    %     netw_out.TechnologyPerf = performance;
    encoded = jsonencode(netw_out, "PrettyPrint",true)
    JSONFILE_name= strcat(save_path, all_netw{netw_name}, '.json');
    fid=fopen(JSONFILE_name,'w');
    fprintf(fid,'%s',encoded);
    fclose(fid);
end




%% Export each technology to one JSON
% save_path = 'C:\EHubversions\EHUB-Py\data\technology_data\';
save_path = 'C:\EHubversions\';

% TechnologyType_SubType_Carrier_Size

% Do something to every technology
all_tecs = fieldnames(Data.technology_data);
remaining_tecs = all_tecs;
for tec_name = 1:size(all_tecs,1)
    set = 0;

    display(all_tecs{tec_name})
    tec = Data.technology_data.(all_tecs{tec_name});
    % Economics
    econ_in = tec.economic;
    a = 1;
    tec_out = [];

    switch all_tecs{tec_name}
        case {'GT'}
            capacity = {10, 100, 250, 400};
            fuel = {'NG', 'H2'};
            for cap = 1:4
                for fl = 1:2
                    name = strcat('GasTurbine_', fuel{fl}, '_', num2str(capacity{cap}));

                    tec_out.tec_type = name;

                    econ_out = [];
                    econ_out.comment = 'CAPEX in EUR/unit or as a piecewise function, OPEX_variable in EUR/MWh total output, OPEX_fixed in % of CAPEX_annual';
                    econ_out.CAPEX_model = 1;
                    econ_out.unit_CAPEX = round(Data.technology_data.GT.economic.c_i(cap,fl),8);
                    econ_out.OPEX_variable = round(econ_in.c_u * 1000,6);
                    econ_out.OPEX_fixed = round(econ_in.c_m,6);
                    econ_out.discount_rate = econ_in.r;
                    econ_out.lifetime = econ_in.lifetime;
                    econ_out.decommission_cost = 0;


                    performance = [];
                    tec_out.size_min = 0;
                    tec_out.size_max = 100;
                    tec_out.size_is_int = tec.tech.disc_size;
                    performance.comment = 'contains fitting data on unit of input, technology types and input/output carriers';
                    if fl == 1
                        performance.input_carrier = tec.type_in;
                        performance.max_H2_admixture = 0.05;
                    else
                        performance.input_carrier = {'hydrogen'};
                    end
                    performance.main_input_carrier = tec.type_in{1};
                    performance.output_carrier = tec.type_out;
                    performance.rated_power = capacity{cap};
                    performance.in_min = Data.technology_data.GT.Smin(cap,fl) /1000;
                    performance.in_max = Data.technology_data.GT.Smax(cap,fl) /1000;
                    performance.alpha = Data.technology_data.GT.tech.alpha(cap,fl);
                    performance.beta = Data.technology_data.GT.tech.beta(cap,fl)/ 1000;
                    performance.gamma = Data.technology_data.GT.tech.gamma;
                    performance.delta = Data.technology_data.GT.tech.delta;
                    performance.T_iso = Data.technology_data.GT.tech.T_0;
                    t = Data.technology_data.GT;
                    performance.epsilon = 1 - t.tech.mr(fl) * (1/t.tech.LHV(fl)) * t.tech.Cp(fl) * t.tech.deltaT;

                    performance.emission_factor = 0.185;
                    
                    %Units
                    units = [];
                    units.size = "MW";
                    for i = 1:length(performance.input_carrier)
                        units.input_carrier.(performance.input_carrier{i}) = "MW";
                    end
                    for i = 1:length(performance.output_carrier)
                        units.output_carrier.(performance.output_carrier{i}) = "MW";
                    end

                    
                    set = 1;

                    tec_out.decommission = 0;
                    tec_out.Economics = econ_out;
                    tec_out.TechnologyPerf = performance;
                    tec_out.Units = units;
        
                    folder = 'PowerGeneration';
                    save_data(tec_out, name, folder)
                end
            end


        case {'HP'}
            econ_out = [];
            econ_out.comment = 'CAPEX in EUR/unit or as a piecewise function, OPEX_variable in EUR/MWh total output, OPEX_fixed in % of CAPEX_annual';
            econ_out.CAPEX_model = 1;
            econ_out.unit_CAPEX =round(econ_in.c_i * a * 1000,8);
            if length(tec.Smin)>1
                econ_out.piecewise_CAPEX.bp_x = round([tec.Smin; tec.Smax(end)] ./ 1000,8);
                econ_out.piecewise_CAPEX.bp_y = round([tec.Smin .* econ_in.zeta1 + econ_in.zeta2; tec.Smax(end) .* econ_in.zeta1(end) + econ_in.zeta2(end)] * a,8);
            else
                econ_out.piecewise_CAPEX.bp_x = 'not defined';
                econ_out.piecewise_CAPEX.bp_y = 'not defined';
            end
            econ_out.OPEX_variable = round(econ_in.c_u * 1000,6);
            econ_out.OPEX_fixed = round(econ_in.c_m,6);

            econ_out.discount_rate = econ_in.r;
            econ_out.lifetime = econ_in.lifetime;
            econ_out.decommission_cost = 0;

            performance = [];
            tec_out.tec_type = [];
            tec_out.size_min = min(tec.Smin / 1000);
            tec_out.size_max = max(tec.Smax / 1000);
            tec_out.size_is_int = tec.tech.disc_size;
            performance.comment = 'contains fitting data on unit of input, technology types and input/output carriers';
            performance.performance_function_type = 3;
            performance.input_carrier = tec.type_in;
            performance.main_input_carrier = tec.type_in{1};
            performance.output_carrier = tec.type_out;
            performance.emission_factor = 0;
            performance.min_part_load = 0.2;
            performance.application = 'radiator_heating';
            performance.min_part_load = 0.2;
            performance.T_out = 39;
            
            %Units
            units = [];
            units.size = "MW";
            for i = 1:length(performance.input_carrier)
                units.input_carrier.(performance.input_carrier{i}) = "MW";
            end
            for i = 1:length(performance.output_carrier)
                units.output_carrier.(performance.output_carrier{i}) = "MW";
            end


            tec_out.decommission = 0;
            tec_out.Economics = econ_out;
            tec_out.TechnologyPerf = performance;
            tec_out.Units = units;

            folder = 'HeatGeneration';
            
            tec_out.tec_type = "HeatPump_AirSourced";
            save_data(tec_out, tec_out.tec_type, folder)


            tec_out.TechnologyPerf.performance.T_in = 10;
            tec_out.tec_type = "HeatPump_GroundSourced";
            save_data(tec_out, tec_out.tec_type, folder)


            tec_out.tec_type = "HeatPump_WaterSourced";
            save_data(tec_out, tec_out.tec_type, folder)

            set = 1;

        case {'DAC'}
            econ_out = [];
            econ_out.comment = 'CAPEX in EUR/unit or as a piecewise function, OPEX_variable in EUR/MWh total output, OPEX_fixed in % of CAPEX_annual';
            econ_out.CAPEX_model = 1;
            econ_out.unit_CAPEX =round(econ_in.c_i * a * 1000,8);
            if length(tec.Smin)>1
                econ_out.piecewise_CAPEX.bp_x = round([tec.Smin; tec.Smax(end)] ./ 1000,8);
                econ_out.piecewise_CAPEX.bp_y = round([tec.Smin .* econ_in.zeta1 + econ_in.zeta2; tec.Smax(end) .* econ_in.zeta1(end) + econ_in.zeta2(end)] * a,8);
            else
                econ_out.piecewise_CAPEX.bp_x = 'not defined';
                econ_out.piecewise_CAPEX.bp_y = 'not defined';
            end
            econ_out.OPEX_variable = round(econ_in.c_u * 1000,6);
            econ_out.OPEX_fixed = round(econ_in.c_m,6);

            econ_out.discount_rate = econ_in.r;
            econ_out.lifetime = econ_in.lifetime;
            econ_out.decommission_cost = 0;

            performance = [];
            tec_out.tec_type = "DAC_Adsorption";
            tec_out.size_min = min(tec.Smin / 1000);
            tec_out.size_max = max(tec.Smax / 1000);
            tec_out.size_is_int = tec.tech.disc_size;
            performance.comment = 'contains fitting data on unit of input, technology types and input/output carriers';
            performance.input_carrier = tec.type_in;
            performance.main_input_carrier = tec.type_in{1};
            performance.output_carrier = tec.type_out;
            performance.nr_segments = 3;
            performance.ohmic_heating = 1;

            performance.emission_factor = -1;
            performance.performance.eta_elth = tec.tech.eta_elth;
            
            %Units
            units = [];
            units.size = "MW";
            for i = 1:length(performance.input_carrier)
                units.input_carrier.(performance.input_carrier{i}) = "MW";
            end
            for i = 1:length(performance.output_carrier)
                units.output_carrier.(performance.output_carrier{i}) = "tonne/hr";
            end


            tec_out.decommission = 0;
            tec_out.Economics = econ_out;
            tec_out.TechnologyPerf = performance;
            tec_out.Units = units;
            
            folder = 'CO2Capture';
            save_data(tec_out, 'DAC_Adsorption', folder)

            set = 1;



            % STORAGE OBJECTS
        case {'battery', 'HWTS','PCM', 'NGS', 'CO2S', 'H2S', 'storageEthy'}

            names = {'Storage_Battery', 'Storage_HotWater','Storage_PCM', 'Storage_NG', 'Storage_CO2', 'Storage_H2', 'Storage_Ethylene'};
            econ_out = [];
            econ_out.comment = 'CAPEX in EUR/MW or as a piecewise function, OPEX_variable in EUR/MWh total output, OPEX_fixed in % of CAPEX_annual';
            econ_out.CAPEX_model = 1;
            econ_out.unit_CAPEX =round(econ_in.c_i * a * 1000,8);
            if length(tec.Smin)>1
                econ_out.piecewise_CAPEX.bp_x = round([tec.Smin; tec.Smax(end)] ./ 1000,8);
                econ_out.piecewise_CAPEX.bp_y = round([tec.Smin .* econ_in.zeta1 + econ_in.zeta2; tec.Smax(end) .* econ_in.zeta1(end) + econ_in.zeta2(end)] * a,8);
            else
                econ_out.piecewise_CAPEX.bp_x = 'not defined';
                econ_out.piecewise_CAPEX.bp_y = 'not defined';
            end
            econ_out.OPEX_variable = round(econ_in.c_u * 1000,6);
            econ_out.OPEX_fixed = round(econ_in.c_m,6);

            econ_out.discount_rate = econ_in.r;
            econ_out.lifetime = econ_in.lifetime;
            econ_out.decommission_cost = 0;

            performance = [];
            tec_out.tec_type = "STOR";
            performance.comment = 'contains fitting data on unit of input, technology types and input/output carriers';
            performance.input_carrier = tec.type_in;
            performance.main_input_carrier = tec.type_in{1};
            performance.output_carrier = tec.type_out;
            tec_out.size_min = min(tec.Smin / 1000);
            tec_out.size_max = max(tec.Smax / 1000);
            tec_out.size_is_int = tec.tech.disc_size;
            performance.emission_factor = 0;
            performance.allow_only_one_direction = 1;

            performance.performance.eta_in = tec.tech.eta_in;
            performance.performance.eta_out = tec.tech.eta_out;
            performance.performance.lambda = tec.tech.lambda;
            performance.performance.theta = tec.tech.theta;
            performance.performance.charge_max = 1/tec.tech.qmax(1);
            if length(tec.tech.qmax) == 1
                performance.performance.discharge_max = 1/tec.tech.qmax;
            else
                performance.performance.discharge_max = 1/tec.tech.qmax(2);
            end
            
            %Units
            units = [];
            
            if sum(strcmp(performance.input_carrier{1}, ["electricity", "hydrogen", "heat", "gas"]))
                units.size = "MWh";
                units.input_carrier.(performance.input_carrier{1}) = "MW";
                units.output_carrier.(performance.input_carrier{1}) = "MW";
            elseif sum(strcmp(performance.input_carrier{1}, ["CO2", "Ethylene"]))
                units.size = "tonne";
                units.input_carrier.(performance.input_carrier{1}) = "tonne/hr";
                units.output_carrier.(performance.input_carrier{1}) = "tonne/hr";               
            end
            
            tec_out.decommission = 0;
            tec_out.Economics = econ_out;
            tec_out.TechnologyPerf = performance;
            tec_out.Units = units;
            
            idx = strcmp(all_tecs{tec_name}, {'battery', 'HWTS','PCM', 'NGS', 'CO2S', 'H2S', 'storageEthy'});

            folder = 'Storage';
            save_data(tec_out, names{idx}, folder)

            set = 1;

            % RE OBJECTS
        case {'PV', 'ST', 'WT_1500', 'WT_2500', 'WT_4000', 'WT_OS_6000', 'WT_OS_9500', 'WT_OS_11000'}

            names = {'Photovoltaic', 'SolarThermal', 'WindTurbine_Onshore_1500', 'WindTurbine_Onshore_2500', 'WindTurbine_Onshore_4000', 'WindTurbine_Offshore_6000', 'WindTurbine_Offshore_9500', 'WindTurbine_Offshore_11000'};

            econ_out = [];
            if strcmp(all_tecs{tec_name}, ['PV', 'ST'])
                econ_out.comment = 'CAPEX in EUR/MW, OPEX_variable in EUR/MWh total output, OPEX_fixed in % of CAPEX_annual';
                econ_out.CAPEX_model = 1;
                econ_out.unit_CAPEX =round(econ_in.c_i * a *1000,8);
            else
                econ_out.comment = 'CAPEX in EUR/module, OPEX_variable in EUR/MWh total output, OPEX_fixed in % of CAPEX_annual';
                econ_out.CAPEX_model = 1;
                econ_out.unit_CAPEX = round(econ_in.c_i * a, 2);
            end
            econ_out.OPEX_variable = round(econ_in.c_u * 1000,6);
            econ_out.OPEX_fixed = round(econ_in.c_m,6);

            econ_out.discount_rate = econ_in.r;
            econ_out.lifetime = econ_in.lifetime;
            econ_out.decommission_cost = 0;

            % Technology Performance
            performance = [];
            tec_out.tec_type = "RES";
            performance.comment = 'contains fitting data on unit of input, technology types and input/output carriers';
            performance.output_carrier = tec.type_out;
            if sum(strcmp(all_tecs{tec_name}, {'PV', 'ST'}))
                tec_out.size_min = 0;
                tec_out.size_max = 1000;
                performance.curtailment = 1;
            else
                tec_out.size_min = tec.Smin;
                tec_out.size_max = tec.Smax/1000;
                performance.curtailment = 2;
            end
            tec_out.size_is_int = tec.tech.disc_size;
            performance.emission_factor = 0;
            
            %Units
            units = [];
            units.size = "MW";
            units.output_carrier.(performance.output_carrier{1}) = "MW";

            tec_out.decommission = 0;
            tec_out.Economics = econ_out;
            tec_out.TechnologyPerf = performance;
            tec_out.Units = units;

            idx = strcmp(all_tecs{tec_name}, {'PV', 'ST', 'WT_1500', 'WT_2500', 'WT_4000', 'WT_OS_6000', 'WT_OS_9500', 'WT_OS_11000'});

            folder = 'RES';
            save_data(tec_out, names{idx}, folder)
            set = 1;

            %         BOILER OBJECT
        case{'boiler', 'boilerH2', 'boilerIND', ...
                'Biomass Gasification', 'STurbine', 'Furnace_NG', ...
                'Furnace_H2', 'boilerE', 'HEX', 'dummyCCS', 'DACdummy'}

            names = {'Boiler_Small_NG', 'Boiler_Small_H2', 'Boiler_Industrial_NG', ...
                'BiomassGasification', 'SteamTurbine', 'Furnace_NG', ...
                'Furnace_H2',  'Boiler_El', 'HeatExchanger', 'dummyCCS', 'DACdummy'};
            
            econ_out = [];
            econ_out.comment = 'CAPEX in EUR/MW or as a piecewise function, OPEX_variable in EUR/MWh total output, OPEX_fixed in % of CAPEX_annual';
            econ_out.CAPEX_model = 1;
            econ_out.unit_CAPEX =round(econ_in.c_i * a * 1000,8);
            if length(tec.Smin)>1
                econ_out.piecewise_CAPEX.bp_x = round([tec.Smin; tec.Smax(end)] ./ 1000,8);
                econ_out.piecewise_CAPEX.bp_y = round([tec.Smin .* econ_in.zeta1 + econ_in.zeta2; tec.Smax(end) .* econ_in.zeta1(end) + econ_in.zeta2(end)] * a,8);
            else
                econ_out.piecewise_CAPEX.bp_x = 'not defined';
                econ_out.piecewise_CAPEX.bp_y = 'not defined';
            end
            econ_out.OPEX_variable = round(econ_in.c_u * 1000,6);
            econ_out.OPEX_fixed = round(econ_in.c_m,6);

            econ_out.discount_rate = econ_in.r;
            econ_out.lifetime = econ_in.lifetime;
            econ_out.decommission_cost = 0;

            % Technology Performance
            performance = [];
            tec_out.tec_type = "CONV2";
            performance.comment = 'contains fitting data on unit of input, technology types and input/output carriers';
            if tec.tech.delta == 0
                performance.performance_function_type = 1;
            else
                performance.performance_function_type = 2;
            end
            performance.input_carrier = tec.type_in;
            performance.main_input_carrier = [tec.type_in{1}];
            performance.output_carrier = tec.type_out;
            tec_out.size_min = min(tec.Smin / 1000);
            tec_out.size_max = max(tec.Smax / 1000);
            tec_out.size_is_int = tec.tech.disc_size;
            if strcmp(tec.type_in, 'gas')
                performance.emission_factor = 0.185;
            else
                performance.emission_factor = 0;
            end
            performance.min_part_load = tec.tech.delta;
            performance.performance.in = [0, 1];
            for out = 1:length(tec.type_out)
                performance.performance.out.(tec.type_out{out}) = [0, tec.tech.eta(out)];
            end
            
            %Units
            units = [];
            units.size = "MW";
            for i = 1:length(performance.input_carrier)
                units.input_carrier.(performance.input_carrier{i}) = "MW";
            end
            for i = 1:length(performance.output_carrier)
                units.output_carrier.(performance.output_carrier{i}) = "MW";
            end


            tec_out.decommission = 0;
            tec_out.Economics = econ_out;
            tec_out.TechnologyPerf = performance;
            tec_out.Units = units;
            idx = strcmp(all_tecs{tec_name}, {'boiler', 'boilerH2', 'boilerIND', ...
                'Biomass Gasification', 'STurbine', 'Furnace_NG', ...
                'Furnace_H2', 'boilerE', 'HEX', 'dummyCCS', 'DACdummy'});

            folder = 'HeatGeneration';
            save_data(tec_out, names{idx}, folder)
            set = 1;
            
            
            % INDUSTRY OBJECTS
        case{'steamcracker','steamcracker_E', 'steamcracker_compression', 'steamcracker_compressionE',  ...
                'steamcracker_separation', 'SMR', 'SMRCCS', 'HaberBosch'}
            
            names = {'NaphthaCracker', 'NapthaCracker_Electric', 'EthyleneCompression', ...
                'EthyleneCompression_Electric', 'EthyleneSeparation', ...
                'SteamReformer', 'SteamReformer_CCS', 'HaberBosch'};
            
            econ_out = [];
            econ_out.comment = 'CAPEX in EUR/unit or as a piecewise function, OPEX_variable in EUR/unit total output, OPEX_fixed in % of CAPEX_annual';
            econ_out.CAPEX_model = 1;
            econ_out.unit_CAPEX =round(econ_in.c_i * a * 1000,8);
            if length(tec.Smin)>1
                econ_out.piecewise_CAPEX.bp_x = round([tec.Smin; tec.Smax(end)] ./ 1000,8);
                econ_out.piecewise_CAPEX.bp_y = round([tec.Smin .* econ_in.zeta1 + econ_in.zeta2; tec.Smax(end) .* econ_in.zeta1(end) + econ_in.zeta2(end)] * a,8);
            else
                econ_out.piecewise_CAPEX.bp_x = 'not defined';
                econ_out.piecewise_CAPEX.bp_y = 'not defined';
            end
            econ_out.OPEX_variable = round(econ_in.c_u * 1000,6);
            econ_out.OPEX_fixed = round(econ_in.c_m,6);

            econ_out.discount_rate = econ_in.r;
            econ_out.lifetime = econ_in.lifetime;
            econ_out.decommission_cost = 0;

            % Technology Performance
            performance = [];
            if length(tec.type_in) == 1
                tec_out.tec_type = "CONV2";
            else
                tec_out.tec_type = "CONV3";
            end
            performance.comment = 'contains fitting data on unit of input, technology types and input/output carriers';
            if tec.tech.delta == 0
                performance.performance_function_type = 1;
            else
                performance.performance_function_type = 2;
            end
            performance.input_carrier = tec.type_in;
            performance.main_input_carrier = [tec.type_in{1}];
            performance.output_carrier = tec.type_out;
            tec_out.size_min = min(tec.Smin / 1000);
            tec_out.size_max = max(tec.Smax / 1000);
            tec_out.size_is_int = tec.tech.disc_size;

            performance.emission_factor = tec.tech.emissionfactor*1000;
            performance.min_part_load = tec.tech.delta;
            performance.performance.in = [0, 1];
            for out = 1:length(tec.type_out)
                performance.performance.out.(tec.type_out{out}) = [0, tec.tech.eta(out)];
            end
            if strcmp(tec_out.tec_type, "CONV3")
                performance.input_ratios.(tec.type_in{1}) = 1;
                for in = 2:length(tec.type_in)
                    performance.input_ratios.(tec.type_in{in}) = tec.tech.theta(in-1);
                end
            end
          
            
            %Units
            units = [];
            if sum(strcmp(all_tecs{tec_name}, ["HaberBosch", "SMR", "SMRCCS"]))
                units.size = "MW";
                for i = 1:length(performance.input_carrier)
                    units.input_carrier.(performance.input_carrier{i}) = "MW";
                end
                for i = 1:length(performance.output_carrier)
                    if contains(performance.output_carrier{i}, 'CO2')
                        units.output_carrier.(performance.output_carrier{i}) = "tonne";
                    else
                        units.output_carrier.(performance.output_carrier{i}) = "MW";
                    end
                end
            else
                units.size = "tonne/hr";
                for i = 1:length(performance.input_carrier)
                    units.input_carrier.(performance.input_carrier{i}) = "";
                end
                for i = 1:length(performance.output_carrier)
                    units.output_carrier.(performance.output_carrier{i}) = "";
                end
            end

            tec_out.decommission = 0;
            tec_out.Economics = econ_out;
            tec_out.TechnologyPerf = performance;
            tec_out.Units = units;
            idx = strcmp(all_tecs{tec_name}, {'steamcracker','steamcracker_E', ...
                'steamcracker_compression', 'steamcracker_compressionE',  ...
                'steamcracker_separation', 'SMR', 'SMRCCS', 'HaberBosch'});

            folder = 'Industrial';
            save_data(tec_out, names{idx}, folder)
            set = 1;
            
    end
    
   
    if set

        remaining_tecs(strcmp(remaining_tecs, all_tecs{tec_name})) = [];
    end
end

remaining_tecs(strcmp(remaining_tecs, 'boilerWAG')) = [];
remaining_tecs(strcmp(remaining_tecs, 'HOS')) = [];
remaining_tecs(strcmp(remaining_tecs, 'WT_spec')) = [];
remaining_tecs(strcmp(remaining_tecs, 'StCycle')) = [];


remaining_tecs

function save_data(data, tec_name, folder)
    save_path = 'C:\EHubversions\EHUB-Py\data\technology_data\';
    encoded = jsonencode(data, "PrettyPrint",true);
    JSONFILE_name= strcat(save_path, folder, '\', tec_name, '.json')
    fid=fopen(JSONFILE_name,'w');
    fprintf(fid,'%s',encoded);
    fclose(fid);
end