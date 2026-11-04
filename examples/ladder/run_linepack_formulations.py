"""Run the original ladder runner with an optional pipeline formulation.

Copy this file and linepack_formulations.py into examples/ladder, next to run.py.
All original options/defaults, solving and result writing are retained. For a
cold comparison, explicitly include --no-warmstart: run.py otherwise starts the
linepack model from its reference run by default. A reference-derived start is
partial and does not guarantee a feasible pressure/linepack trajectory.

Examples, from examples/ladder (run.py's --timelim is HOURS):

    python run_linepack_formulations.py --formulation combined_cuts \
        --rung 3L3S --case linepack --hours 168 --no-warmstart --timelim 1
    python run_linepack_formulations.py --formulation magnitude_cuts \
        --rung 3L3S --case linepack --hours 168 --no-warmstart \
        --no-pressure-coupling --timelim 1

The original run.py and production classes are not edited. This wrapper accepts
only the tested unscaled, hourly, chronological cost configuration for modified
linepack formulations; baseline passes through the original configuration.
"""

import argparse
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace


FORMULATIONS = ("baseline", "combined_cuts", "magnitude_cuts")


def _load_runner():
    """Find run.py beside the copied files, or above this distribution folder."""
    here = Path(__file__).resolve().parent
    ladder_dir = next(
        (
            path
            for path in (here, here.parent)
            if (path / "run.py").is_file() and (path / "rungs.py").is_file()
        ),
        None,
    )
    if ladder_dir is None:
        raise RuntimeError(
            "Copy both package files into the original examples/ladder folder, "
            "beside run.py and rungs.py."
        )
    # Original run.py imports design/layout/rungs by their flat module names;
    # the repository root also makes the local adopt_net0 package importable.
    paths = (here, ladder_dir, ladder_dir.parent.parent)
    for path in reversed(tuple(dict.fromkeys(str(path) for path in paths))):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(
        "_linepack_original_ladder_run", ladder_dir / "run.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def _check_supported(hub):
    config = hub.data.model_config
    requirements = {
        "cost minimization": config["optimization"]["objective"]["value"] == "costs",
        "scaling disabled": config["scaling"]["scaling_on"]["value"] == 0,
        "typical days disabled": config["optimization"]["typicaldays"]["N"]["value"]
        == 0,
        "time staging disabled": config["optimization"]["timestaging"]["value"] == 0,
        "hourly base resolution": math.isclose(
            float(hub.data.topology["resolution_in_h"]["full"]),
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ),
    }
    unsupported = [name for name, satisfied in requirements.items() if not satisfied]
    if unsupported:
        raise ValueError(
            "The supplied formulation wrapper requires "
            + ", ".join(unsupported)
            + ". Use --formulation baseline for the original behavior."
        )


def _canonicalize_idle_starts(model):
    """Adapt only existing idle starts to magnitude's sum(direction)=installed.

    No values are set for a cold model. A built pipe with both original direction
    starts zero is physically idle; choose its forward zero-flow representation.
    This changes a start value only, never a bound or fixed variable.
    """
    changed = 0
    for period in model.periods.values():
        for block in period.network_block.values():
            if not hasattr(block, "magnitude_pipe"):
                continue
            for source, target in block.set_arcs_unique:
                forward = block.arc_block[source, target]
                installed = forward.var_installed.value
                if installed is None or abs(installed - 1) > 1e-6:
                    continue
                reverse = (
                    block.arc_block[target, source]
                    if (target, source) in block.set_arcs
                    else None
                )
                for time, direction in forward.var_direction.items():
                    values = [direction.value]
                    if reverse is not None:
                        values.append(reverse.var_direction[time].value)
                    if all(
                        value is not None and abs(value) <= 1e-6 for value in values
                    ):
                        direction.set_value(1)
                        changed += 1
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--formulation", choices=FORMULATIONS, default="baseline")
    options, original_args = parser.parse_known_args(argv)
    runner = _load_runner()
    from linepack_formulations import apply_formulation, apply_magnitude_formulation
    from adopt_net0.components.networks.specificNetworks.fluidynamic_pipeline import (
        FluidynamicPipeline,
    )

    original_adopt = runner.adopt
    variant = options.formulation

    class FormulationModelHub(original_adopt.ModelHub):
        def _has_linepack(self):
            return any(
                isinstance(network, FluidynamicPipeline)
                for networks in self.data.network_data.values()
                for network in networks.values()
            )

        def construct_model(self):
            if variant != "baseline" and self._has_linepack():
                _check_supported(self)
            return super().construct_model()

        def construct_balances(self):
            result = super().construct_balances()
            if self._has_linepack():
                model = self.model[self.info_solving_algorithms["aggregation_model"]]
                if variant == "magnitude_cuts":
                    details = apply_magnitude_formulation(
                        model, self.data.network_data, inventory_cuts=True
                    )
                else:
                    details = apply_formulation(model, self.data.network_data, variant)
                print(
                    f"Linepack formulation: {variant} ({details['networks']} networks)"
                )
            return result

        def solve(self):
            if variant == "magnitude_cuts" and self._has_linepack():
                model = self.model[self.info_solving_algorithms["aggregation_model"]]
                changed = _canonicalize_idle_starts(model)
                if changed:
                    print(
                        f"Magnitude start: adapted {changed} built-idle direction labels"
                    )
            return super().solve()

    if "--help" in original_args or "-h" in original_args:
        print(
            "Additional option: --formulation {baseline,combined_cuts,magnitude_cuts}"
        )
        print("Default: baseline. Use --no-warmstart for a cold comparison.")
        print("All remaining options below belong to the original run.py.\n")
    else:
        print(
            f"Selected linepack formulation: {variant}; original run.py options retained"
        )
    runner.adopt = SimpleNamespace(ModelHub=FormulationModelHub)
    try:
        return runner.main(original_args)
    finally:
        runner.adopt = original_adopt


if __name__ == "__main__":
    main()
