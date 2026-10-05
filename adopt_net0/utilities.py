from pyomo.environ import SolverFactory
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

# Gurobi status code -> legacy Pyomo status, as pyomo's gurobi_direct maps them
_GUROBI_STATUS = {
    1: (SolverStatus.aborted, TerminationCondition.error),
    2: (SolverStatus.ok, TerminationCondition.optimal),
    3: (SolverStatus.warning, TerminationCondition.infeasible),
    4: (SolverStatus.warning, TerminationCondition.infeasibleOrUnbounded),
    5: (SolverStatus.warning, TerminationCondition.unbounded),
    6: (SolverStatus.aborted, TerminationCondition.minFunctionValue),
    7: (SolverStatus.aborted, TerminationCondition.maxIterations),
    8: (SolverStatus.aborted, TerminationCondition.maxEvaluations),
    9: (SolverStatus.aborted, TerminationCondition.maxTimeLimit),
    10: (SolverStatus.aborted, TerminationCondition.unknown),
    11: (SolverStatus.aborted, TerminationCondition.error),
    12: (SolverStatus.error, TerminationCondition.error),
    13: (SolverStatus.warning, TerminationCondition.other),
    15: (SolverStatus.aborted, TerminationCondition.other),
}


class GurobiMatrix:
    """
    Gurobi fed with the model as one constraint matrix

    Pyomo's standard-form compiler builds the matrix in one pass and gurobipy takes it
    in one call, several times faster than gurobi_direct, which adds the constraints
    one at a time. Pyomo's newer gurobi_direct (pyomo.contrib.solver) works the same
    way; solve() takes the arguments AdOpT passes to gurobi_direct and returns the
    same legacy results.
    """

    def __init__(self, options: dict):
        self.options = options
        self._solver_model = None

    def solve(self, model, tee=False, warmstart=False, logfile=None, keepfiles=False):
        """
        Solves the model and loads the solution into it

        :param model: pyomo model
        :param bool tee: show the solver log
        :param bool warmstart: start from the current variable values
        :param str logfile: file to write the solver log to
        :param bool keepfiles: not used, no files are written
        :return: legacy pyomo results
        """
        import gurobipy as gp

        repn = LinearStandardFormCompiler().write(
            model, mixed_form=True, set_sense=None
        )
        if len(repn.objectives) > 1:
            raise ValueError("Gurobi takes one objective, the model has several")
        columns = repn.columns
        # Pyomo before 6.10 can return fixed variables as columns (when other indices
        # of the same variable are free); they keep their value, not their bounds
        bounds = [(v.value, v.value) if v.fixed else v.bounds for v in columns]
        grb = gp.Model()
        grb.Params.LogToConsole = int(tee)
        if logfile:
            grb.Params.LogFile = logfile
        for key, value in self.options.items():
            grb.setParam(key, value)
        x = grb.addMVar(
            len(columns),
            lb=[-gp.GRB.INFINITY if lb is None else lb for lb, _ in bounds],
            ub=[gp.GRB.INFINITY if ub is None else ub for _, ub in bounds],
            obj=repn.c.toarray()[0] if repn.c.shape[0] else 0.0,
            vtype=[
                (
                    gp.GRB.BINARY
                    if v.is_binary()
                    else gp.GRB.INTEGER if v.is_integer() else gp.GRB.CONTINUOUS
                )
                for v in columns
            ],
        )
        # bound type of a row: 0 equality, 1 upper bound, -1 lower bound
        grb.addMConstr(
            repn.A, x, ["=<>"[row.bound_type] for row in repn.rows], repn.rhs
        )
        if repn.c.shape[0]:
            grb.ObjCon = repn.c_offset[0]
            grb.ModelSense = int(repn.objectives[0].sense)
        if warmstart:
            x.Start = [
                gp.GRB.UNDEFINED if v.value is None else v.value for v in columns
            ]

        grb.optimize()

        results = SolverResults()
        results.solver.status, results.solver.termination_condition = (
            _GUROBI_STATUS.get(
                grb.Status, (SolverStatus.error, TerminationCondition.error)
            )
        )
        results.solver.wallclock_time = grb.Runtime
        if grb.SolCount > 0:
            for v, value in zip(columns, x.X.tolist()):
                v.set_value(value, skip_validation=True)
            bound = grb.ObjBound if grb.IsMIP else grb.ObjVal
            if grb.ModelSense == 1:
                results.problem.lower_bound, results.problem.upper_bound = (
                    bound,
                    grb.ObjVal,
                )
            else:
                results.problem.lower_bound, results.problem.upper_bound = (
                    grb.ObjVal,
                    bound,
                )
        self._solver_model = grb
        return results


def get_gurobi_parameters(solveroptions: dict, matrix_handover: bool = True):
    """
    Initiates the gurobi solver and defines solver parameters

    Solver "gurobi" gets the model as one matrix (GurobiMatrix) unless
    matrix_handover is False; "gurobi_persistent" keeps pyomo's persistent interface.

    :param dict solveroptions: dict with solver parameters
    :param bool matrix_handover: use GurobiMatrix for solver "gurobi"
    :return: Gurobi Solver
    """
    options = {
        "TimeLimit": solveroptions["timelim"]["value"] * 3600,
        "MIPGap": solveroptions["mipgap"]["value"],
        "MIPFocus": solveroptions["mipfocus"]["value"],
        "Threads": solveroptions["threads"]["value"],
        "NodefileStart": solveroptions["nodefilestart"]["value"],
        "Method": solveroptions["method"]["value"],
        "Heuristics": solveroptions["heuristics"]["value"],
        "Presolve": solveroptions["presolve"]["value"],
        "BranchDir": solveroptions["branchdir"]["value"],
        "LPWarmStart": solveroptions["lpwarmstart"]["value"],
        "IntFeasTol": solveroptions["intfeastol"]["value"],
        "FeasibilityTol": solveroptions["feastol"]["value"],
        "Cuts": solveroptions["cuts"]["value"],
        "NumericFocus": solveroptions["numericfocus"]["value"],
    }
    if matrix_handover and solveroptions["solver"]["value"] == "gurobi":
        return GurobiMatrix(options)
    solver = SolverFactory(solveroptions["solver"]["value"], solver_io="python")
    solver.options.update(options)

    return solver


def get_glpk_parameters(solveroptions: dict):
    """
    Initiates the glpk solver and defines solver parameters

    :param dict solveroptions: dict with solver parameters
    :return: Gurobi Solver
    """
    solver = SolverFactory("glpk")

    return solver


def get_set_t(config: dict, model_block):
    """
    Returns the correct set_t for different clustering options

    :param dict config: config dict
    :param model_block: pyomo block holding set_t_full and set_t_clustered
    :return: set_t
    """
    if config["optimization"]["typicaldays"]["N"]["value"] == 0:
        return model_block.set_t_full
    elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
        return model_block.set_t_clustered
    elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
        return model_block.set_t_full


def get_data_for_investment_period(
    data, investment_period: str, aggregation_model: str
) -> dict:
    """
    Gets data from DataHandle for specific investement_period. Writes it to a dict.

    :param data: data to use
    :param str investment_period: investment period
    :param str aggregation_model: aggregation type
    :return: data of respective investment period
    :rtype: dict
    """
    data_period = {}
    data_period["period_name"] = investment_period
    data_period["topology"] = data.topology
    data_period["technology_data"] = data.technology_data[investment_period]
    data_period["time_series"] = data.time_series[aggregation_model].loc[
        :, investment_period
    ]
    data_period["network_data"] = data.network_data[investment_period]
    data_period["energybalance_options"] = data.energybalance_options[investment_period]
    data_period["config"] = data.model_config
    if data.model_config["optimization"]["typicaldays"]["N"]["value"] != 0:
        data_period["k_means_specs"] = data.k_means_specs[investment_period]
        # data_period["averaged_specs"] = data.averaged_specs[investment_period]
    if data.model_config["performance"]["pressure"]["pressure_on"]["value"] == 1:
        data_period["compressor_data"] = data.compressor_data[investment_period]

    # Hour multiplication factors
    if data.model_config["optimization"]["typicaldays"]["N"]["value"] == 0:
        data_period["hour_factors"] = [1] * len(
            data_period["topology"]["time_index"]["full"]
        )
    elif data.model_config["optimization"]["typicaldays"]["method"]["value"] == 1:
        data_period["hour_factors"] = data_period["k_means_specs"]["factors"]
    elif data.model_config["optimization"]["typicaldays"]["method"]["value"] == 2:
        data_period["hour_factors"] = [1] * len(
            data_period["topology"]["time_index"]["full"]
        )

    # Nr timesteps averaged
    if data.model_config["optimization"]["timestaging"]["value"] != 0:
        data_period["nr_timesteps_averaged"] = data.model_config["optimization"][
            "timestaging"
        ]["value"]
    else:
        data_period["nr_timesteps_averaged"] = 1

    return data_period


def determine_flow_existing_compressors(self, compressor, b_period, node):
    """
    Determines the flow capacity of an existing compressor connection by returning
    the minimum available capacity between the output and input components

    :param compressor: tuple with carrier, component1, component 2
    :param b_period: pyomo block data for period
    :param node: pyomo block data for node
    :return float: minimum capacity between input and output component
    """
    component_output_bound = float("inf")
    component_input_bound = float("inf")
    period_name = b_period.name.split("[")[-1].rstrip("]")
    type_component = [compressor.output_type, compressor.input_type]

    if type_component[0] == "Technology":
        var_output = (
            b_period.node_blocks[node]
            .tech_blocks_active[compressor.output_component]
            .var_output
        )
        component_output_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[0] == "Network":
        component_output_bound = next(
            iter(
                b_period.network_block[
                    compressor.output_component
                ].para_size_initial.values()
            )
        )
    elif type_component[0] == "Import":
        component_output_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Import limit"]
        )

    elif type_component[0] == "Generic production":
        component_output_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Generic production"]
        )

    if type_component[1] == "Technology":
        var_output = (
            b_period.node_blocks[node]
            .tech_blocks_active[compressor.input_component]
            .var_output
        )
        component_input_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[1] == "Network":
        component_input_bound = next(
            iter(
                b_period.network_block[
                    compressor.input_component
                ].para_size_initial.values()
            )
        )
    elif type_component[1] == "Demand":
        component_input_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Demand"]
        )
    elif type_component[1] == "Export":
        component_input_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Export limit"]
        )

    size = min(component_output_bound, component_input_bound)

    return size
