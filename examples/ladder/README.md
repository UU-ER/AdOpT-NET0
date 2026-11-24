# The ladder

The NL hydrogen case of `examples/linepack_nl` is nine nodes, a closed backbone ring and
twelve candidate corridors offered in three pipeline types each. It is the system the
study is about, and it is a bad place to find out whether a modelling choice works: a
run is an afternoon, and when it does not solve there is no way to tell whether the
model, the data or the size is at fault.

The ladder is the same system seen through a smaller window. Every rung uses the same
demand, the same capacity factors, the same technology costs, the same pipeline types
and the same backbone as the full case, restricted to a handful of nodes. A rung
therefore answers the same question the full case does, and an answer it gives carries
over. Climb it until something breaks, and what broke is a property of the system rather
than of a toy.

| rung   | large clusters                   | small clusters              | what it adds                                     |
|--------|----------------------------------|-----------------------------|--------------------------------------------------|
| `1L1S` | Rotterdam                        | Dordrecht                   | one corridor, no backbone, no routing at all     |
| `1L2S` | Rotterdam                        | Dordrecht, Venlo            | a second cluster, one of them 132 km out         |
| `1L3S` | Rotterdam                        | Dordrecht, Venlo, Arnhem    | every small cluster, so small-small corridors    |
| `2L2S` | Rotterdam, Zeeland               | Dordrecht, Venlo            | a backbone corridor, so two ways in              |
| `3L3S` | Rotterdam, Zeeland, Chemelot     | Dordrecht, Venlo, Arnhem    | a backbone path, twelve candidate corridors      |

The ladder grows in two directions and they are worth keeping apart when a run
misbehaves. Small clusters add **candidate corridors**, i.e. install binaries, i.e. the
size of the design problem. Large clusters add **backbone**, i.e. routing, i.e. the size
of the operational problem.

Each rung lives in its own folder next to the scripts, with its own `input_data`, its own
`userData` for results and its own `results.html`. Nothing is shared between rungs.

```
examples/ladder/
  rungs.py     which nodes each rung has
  build.py     builds a rung out of the study
  run.py       solves a rung, monolithically
  run_decomposed.py   solves a rung by the bilevel decomposition
  run_linepack_formulations.py   run.py with another pipeline formulation
  linepack_formulations.py       the formulations it can choose
  launch.sh    runs a batch file of jobs in parallel
  batches/     every batch solved so far
  layout.py    which corridors exist and which ones can be built
  design.py    which technologies are dimensioned and which ones are given
  viewer.py    rebuilds results.html out of a rung's userData
  sweep.py     runs the ladder across every axis and tabulates it
  sweep_pack.py   packs the runs of a sweep into json the viewer is dropped
  layouts/     worked layout examples, one per rung that has one
  1L1S/ 1L2S/ 1L3S/ 2L2S/ 3L3S/
      input_data/   the case
      userData/     every result folder of that rung
      rung.json     which nodes the rung has
      reference_design.json   the design of the study, i.e. what --design pins
      res_capacity_factors.csv
      results.html
```

## Building

`build.py` needs the study in `Linepack_modelling` importable, since the demand, the
capacity factors and the costs come from it. Nothing else does: once a rung is built it
is solved with ADOPT alone.

```bash
cd examples/ladder
PYTHONPATH=<adopt>;<linepack_modelling> python build.py --all
PYTHONPATH=<adopt>;<linepack_modelling> python build.py --rung 1L1S
```

A rung is built for the whole year at hourly resolution. The horizon of a run is chosen
afterwards, so a rung is built once and solved many times.

## Solving

```bash
python run.py --rung 1L1S --hours 24          # both network models, the full design problem
python run.py --rung 2L2S --hours 168 --case linepack
python run.py --rung 3L3S --hours 24 --timelim 0.5 --threads 8
```

Four things are chosen per run, and they are independent of each other.

**With or without the pipeline fluid dynamics** — `--case`:

| value       | network type            | what it means                                        |
|-------------|-------------------------|------------------------------------------------------|
| `reference` | `fixed_size_pipeline`   | the flow is bounded by the capacity, no fluid stored |
| `linepack`  | `fluidynamic_pipeline`  | the quasi-dynamic pipeline equation, with linepack   |
| `both`      | both, reference first   | the default; the difference is what linepack is worth |

Both offer exactly the same infrastructure — the same corridors, the same types, the same
capacity per arc, the same investment cost — so the difference between the two objectives
is the transport model and nothing else.

**With or without the components being dimensioned** — `--design`, see `design.py`:

| value     | corridors | technologies | the problem it poses                   |
|-----------|-----------|--------------|----------------------------------------|
| `all`     | decided   | decided      | the full investment problem (default)  |
| `network` | decided   | given        | only the corridors are dimensioned     |
| `tecs`    | given     | decided      | only the clusters are dimensioned      |
| `none`    | given     | given        | pure operation, nothing is invested    |

**Where the given part comes from** — `--from`:

| value           | meaning                                                              |
|-----------------|----------------------------------------------------------------------|
| `study`         | the design the study carries, written next to the case by `build.py`; the default, and it needs nothing to have been solved |
| `solved`        | the design the reference run of this same call finds                 |
| a result folder | the design of an earlier run, e.g. `1L1S/userData/20260917101411-1`  |

When the network is given by `study`, one pipeline of `--given-type` (medium by default)
is put on every candidate corridor. A technology the given design does not mention is
pinned to zero: a design that is given has to say what the cluster is, not only what part
of it is worth mentioning.

A given size is fixed by setting `size_min` and `size_max` to the same value, so the
component is still built and still paid for — the investment cost is simply a constant of
the run. That is deliberate: the objective of a given run stays comparable with the
objective of a run that decided the same design.

**With or without a pipeline already on a particular arc** — `--layout`, see `layout.py`.
`--design network|none` gives the *whole* network at once; a layout file says it corridor
by corridor:

```bash
python run.py --rung 1L1S --hours 24 --layout layouts/1L1S_one_corridor_existing.json
python run.py --rung 2L2S --dump-layout mine.json   # start from what the rung carries
```

A type that appears under `existing` is built and paid for already; a type under
`candidates` carries the install binary. A corridor may carry both, which is one pipeline
already there and a second one that may be laid next to it.

**The start of the linepack run** — `--warmstart` (default), `--no-warmstart`,
`--warmstart-directions`. With `--case both`, the linepack run starts from the reference
solution of the same call. By default only the design binaries are handed over. The
hourly directions of the reference come from flows that ignore the pressure, and fixed in
the start gurobi completed them to a solution a hundred times the reference cost;
`--warmstart-directions` hands them over anyway. `--no-warmstart` solves every run from
scratch, which is what a comparison of formulations or seeds needs.

**Other options of a run:**

| option                     | what it does                                                                 |
|----------------------------|------------------------------------------------------------------------------|
| `--seed N`                 | random seed of gurobi; changes the search path, not the model. Solve times of one job spread by a factor 2 to 14 over seeds, so compare runs over several seeds |
| `--import-price X`         | price of imported hydrogen at every node, in EUR/MWh (210 as built)          |
| `--fix-directions FOLDER`  | fixes every corridor of the linepack run to the direction of its net flow in that result folder, for the whole horizon. Corridors that carried nothing are left free, since a fixed direction forces the pipe in |
| `--linepack-types TYPE...` | pipeline types that keep their linepack; the others store nothing. `H2Pipeline_backbone` alone drops the pack of the thin pipes, which holds less than an hour of their own flow |
| `--hybrid`                 | in the linepack run, only the backbone keeps the pipeline equation, its directions and its linepack; the small, medium and large types are solved as the reference (flow up to the capacity, no pressure, no direction binaries). Much lighter than the full model. Whether the backbone exists or is a candidate is still up to `--layout`, e.g. `layouts/3L3S_backbone_candidate.json` |
| `--large-storage FACTOR`   | offers new hydrogen tanks at the large clusters next to the existing ones, priced like the tanks of the small clusters, up to FACTOR times the existing tank. Needs `--copy-case` and a `--design` that decides the technologies |

```bash
python run.py --rung 3L3S --hours 336 --case linepack --no-warmstart --seed 1
python run.py --rung 3L3S --hours 336 --case linepack --linepack-types H2Pipeline_backbone
python run.py --rung 3L3S --hours 168 --case linepack --fix-directions 3L3S/userData/<result folder>
python run.py --rung 3L3S --hours 336 --copy-case --large-storage 2
python run.py --rung 3L3S --hours 336 --case linepack --no-warmstart --hybrid --layout layouts/3L3S_backbone_candidate.json
```

## Solving with another pipeline formulation

`run_linepack_formulations.py` runs `run.py` with one extra option, `--formulation`, and
passes every other option on unchanged. The formulations live in
`linepack_formulations.py`; `run.py` and `adopt_net0` are not edited, the wrapper swaps
the constraints of the fluid-dynamic pipeline after the model is built.

| `--formulation`        | what it is                                                                    |
|------------------------|-------------------------------------------------------------------------------|
| `baseline`             | the model as it is, i.e. plain `run.py` (default)                            |
| `combined_cuts`        | the original formulation with the convex hull of `linepack = installed * k * p_average`, the two directional pressure drops linked to the physical pressure difference once per pipe, and two valid inventory/drop rows per pipe-hour |
| `magnitude_cuts`       | one SOS2 curve per pipe for the flow and drop magnitudes (`q_forward + q_reverse`, `delta_forward + delta_reverse`), the direction binaries summing to the install binary, plus the inventory hull, the signed pressure link and the inventory cuts |
| `magnitude_ratio_cuts` | `magnitude_cuts` plus, for each direction, `r_min*delta <= q <= r_max*delta`, with `r_min`, `r_max` the smallest and largest flow/pressure-drop ratio over the breakpoints with a nonzero drop |

The modified formulations accept only a cost-minimising, unscaled, hourly, chronological
configuration (no typical days, no time staging), which is what the rungs carry; anything
else is refused and needs `baseline`. Unlike `run.py`, the wrapper puts the repository
root on the path itself, so it runs from `examples/ladder` with nothing installed.

```bash
python run_linepack_formulations.py --formulation magnitude_ratio_cuts --rung 3L3S --hours 48 --case linepack --design all --no-warmstart --threads 4 --mipgap 0.01 --timelim 4 --seed 0
```

The operation runs, where the cuts made the difference, need a reference design first:

```bash
python run_linepack_formulations.py --formulation baseline --rung 3L3S --hours 672 --case reference --design all --threads 4 --mipgap 0.01 --timelim 24
python run_linepack_formulations.py --formulation magnitude_ratio_cuts --rung 3L3S --hours 672 --case linepack --design none --from 3L3S/userData/<reference result folder> --no-warmstart --threads 4 --mipgap 0.01 --timelim 8 --seed 1
```

Always pass `--no-warmstart` when comparing formulations, so that no run starts from the
reference, and repeat over seeds.

## Running batches

`launch.sh` runs a batch file of jobs in parallel, each on its own copy of the case
(`--copy-case` and `--no-viewer` are added to every job). A batch file holds one job per
line, `<name> <options of run_linepack_formulations.py>`; blank lines and lines starting
with `#` are skipped. `batches/` holds every batch solved so far and is the record of
which options each result came from.

```bash
bash launch.sh batches/batch5_ratio_cuts.txt 12      # 12 jobs at a time
nohup bash launch.sh batches/batch5_ratio_cuts.txt > /dev/null 2>&1 &
tail -f logs/batch5_ratio_cuts/status.tsv
```

The log of each job goes to `logs/<batch>/<name>.txt`, and `status.tsv` gets one line per
finished job: name, exit code, seconds, result folder. Jobs named `<group>@<anything>`
race each other: the first of a group to close wins, the others are killed. `PYTHON` picks
the interpreter (default `python3`).

## Solving decomposed

Every rung can also be solved by the bilevel decomposition, which is the point of the
ladder for that algorithm: try it on two nodes before trusting it on nine. The design
options mean the same thing, except that `--from solved` has nothing to refer to, since
the decomposition solves no reference run of its own.

```bash
python run_decomposed.py --rung 1L1S --hours 24 --iterations 2
python run_decomposed.py --rung 2L2S --hours 168 --design none --workers 4 --sub-threads 4
```

Its results go under `<rung>/userData/decomposed/`, one folder per subproblem, because
ADOPT names a result folder after the second it started in and appends to a single
summary spreadsheet.

## Looking at the results

`run.py` rebuilds `<rung>/results.html` after the runs, which is a self-contained page
carrying the most recent runs of that rung. `--no-viewer` skips it, and `viewer.py`
rebuilds it on its own:

```bash
python viewer.py --rung 2L2S -n 6
```

A sweep is the other shape. It solves every rung at every horizon on a machine whose
result folders are gigabytes of h5, and baking all of that into one page would be tens
of megabytes of runs that are never looked at. `sweep_pack.py` therefore writes the page
once, empty, and packs each run on its own into a few hundred kilobytes of json:

```bash
python sweep_pack.py <sweep folder>                # every run of it
python sweep_pack.py <sweep folder> --rung 3L3S --hours 336
python sweep_pack.py <sweep folder> --bundle       # all of it in one file
```

A single run needs none of that: **drop its `optimization_results.h5` on the page, or
the result folder holding it**, and the page reads the h5 itself. Which rung it is comes
from the nodes it names, since no two rungs carry the same ones, and the coordinates
come from a table of every rung's nodes written into the page. The run is labelled by
the folder it was saved to, which the h5 records, so the name survives being dragged out
of that folder. The h5 reader is a wasm build of HDF5 fetched from a CDN the first time
an h5 arrives, so that one drop needs the network; everything else works from disk.

Packing a loose folder still works, and is the way to look at many runs at once without
dropping them one by one. Nothing outside a sweep says which rung a result folder was
solved on, so `--rung` has to:

```bash
python sweep_pack.py 2L2S/userData --rung 2L2S              # every run of a rung
python sweep_pack.py 2L2S/userData/20260917182747-1 --rung 2L2S   # one of them
```

A run that carries no job, packed or dropped, is labelled by its result folder, and the
two network types are told apart the way `viewer.py` does it, i.e. by whether a linepack
was written. Inside a sweep the order the two were solved in says so instead, which is
exact: a linepack run that builds nothing writes no linepack either.

The files land in `<sweep folder>/packed/`, next to `index.csv` saying what each one
holds and `viewer.html` holding no run at all. Open that page and drop the json files
onto it; the `case` menu picks the rung, since two rungs have different nodes and their
objectives do not compare, and the `run` menu picks what the job decided. A run packed
this way names the job it came from, so the page compares it against the same job solved
with the other network type rather than against whatever run has the same horizon.

The h5 alone is not enough to draw the map: it names the nodes but not where they are,
so a packed run joins it with the `NodeLocations.csv` of the rung and with the row of
`results.csv` that says which rung, what was decided and how the solve went.

## Notes

- **Imports.** Only the large clusters may import hydrogen, at twice their own peak
  demand and at the backstop price of the study. A small cluster cannot, so it either
  produces locally or connects, which is the decision a rung is about. `--import-cap`
  tightens the ceiling; on a rung with one large cluster do not set it to zero, since
  Rotterdam cannot serve its own demand without imports and the model is then infeasible.
- **No balance may be violated.** Unlike the study, nothing can be bought out of nothing,
  in any carrier at any node.
- **A rung with one large cluster has no backbone**, since the backbone of the study only
  joins large clusters. Its `Networks.json` carries no existing network and every corridor
  of that rung is a candidate.
- **The case is put back after every run.** `--layout`, `--design` and `--from` rewrite
  the topology and the technology files, and they are restored in a `finally`, so a run
  that fails does not leave the next one with a case it did not ask for.
