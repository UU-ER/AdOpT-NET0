# Linepack in AdOpT-NET0 — what was implemented and why

Date: 2026-09-21. Describes `FluidynamicPipeline`, the pipeline network that carries the
quasi-dynamic pipeline equation and the linepack, how it differs from the transport model
AdOpT-NET0 had before it, and which simplification was taken at each point where the
physics offered a choice.

Code: `adopt_net0/components/networks/specificNetworks/fluidynamic_pipeline.py` and its
parent `fixed_size_pipeline.py`. Derivation of the equations:
`PhD_Brain/Projects/Non Linearity ADOPT/Methodology/12. Linepack linearization for adopt.md`.
Performance evidence: note 13 of the same folder.

---

## 1. What the model had before

AdOpT-NET0 networks are a chain of classes, each adding one thing:

| class | file | what it adds |
|---|---|---|
| `Network` | `networks/network.py` | arcs, size, capex, losses, the inflow/outflow link to the node balances |
| `Fluid` | `genericNetworks/fluid.py` | energy consumption at the sending node (compression), leakage emissions |
| `FixedSizePipeline` | `specificNetworks/fixed_size_pipeline.py` | a pipeline geometry: the capacity of an arc follows from diameter, length and pressure band |
| `FluidynamicPipeline` | `specificNetworks/fluidynamic_pipeline.py` | the pipeline equation and the linepack |

The transport model of the first three is one inequality:

```
flow[t] <= size * rated_capacity
```

Nothing else. There is no pressure anywhere in the optimization; the `pressure` entry of a
fluid network json is input data that only sizes the compressor work (`fluid.py:147`,
`cons_model 2`), and the gas is weightless in time — what enters an arc in hour *t* leaves
it in hour *t*. That is the **basic formulation**, and it is exactly what `--case
reference` solves in `examples/ladder/run.py`.

**The finding that motivated the work:** AdOpT has no linepack. A pipeline in the basic
formulation stores nothing, so hydrogen infrastructure is valued purely as transport
capacity, and whatever flexibility a real network gets from packing the line is invisible
to the investment decision.

## 2. The physics being approximated

Steady-state, isothermal, single-segment flow in a pipe:

$$ f\,|f| = R_e \,\bigl(p_i^2 - p_j^2\bigr) $$

with $f$ the average flow in the arc, $p_i,p_j$ the end pressures and $R_e$ carrying
diameter, length, friction factor and gas properties. The stored mass is proportional to
the average pressure,

$$ \mathrm{LP}_{e,t} = K^{\mathrm{LP}}_e \frac{p_{i,t}+p_{j,t}}{2}, $$

and the pack is what makes inlet and outlet flows differ:

$$ \mathrm{LP}_{e,t} = \mathrm{LP}_{e,t-1} + \Delta t \bigl(f^{\mathrm{in}}_{e,t} - f^{\mathrm{out}}_{e,t}\bigr),
\qquad f_{e,t} = \tfrac12\bigl(f^{\mathrm{in}}_{e,t}+f^{\mathrm{out}}_{e,t}\bigr). $$

Two of the three are already linear. **The whole difficulty is the first equation**, and
everything in §4 is about how it enters a MILP.

Both coefficients are derived in `fit_network_performance` (`fixed_size_pipeline.py:78`),
per arc, because both scale with the length:

```
f_D   = (2 log10(3.7 D / eps))^-2                                             Nikuradse, fully turbulent
R_e   = ed^2 * 3.6^2 * pi^2/16 * D^5 * 1e10 / (f_D * 1000 * Z * R_s * T) / L  [MW^2/bar^2]
K_LP  = ed * (pi D^2/4 * 1000) * 100 / (Z * R_s * T) * L                      [MWh/bar]
```

`ed` is the energy density in MWh/t: the mass flow is converted to an energy flow so that
the arc flow, the arc size and the linepack come out in MW and MWh, consistent with the
energy balance and with the compressors. The SI conversion factors are hardcoded, so the
json has to give diameter and roughness in m, pressures in bar, temperature in K, molar
mass in kg/kmol and the topology distance in km.

Worked values for the pipeline types of the ladder case:

| type | D [m] | band [bar] | p_ref | L used | f_D | R_e | K_LP [MWh/bar] | capacity [MW] | usable pack [MWh] |
|---|---|---|---|---|---|---|---|---|---|
| small    | 0.18 | 5–15  | 10   | 30 km  | 0.0144 | 29.8   | 1.95 | 61    | 19.5  |
| medium   | 0.22 | 15–30 | 22.5 | 20 km  | 0.0138 | 127    | 1.94 | 224   | 29.1  |
| large    | 0.35 | 15–30 | 22.5 | 20 km  | 0.0126 | 1419   | 4.91 | 748   | 73.6  |
| backbone | 1.20 | 30–70 | 50   | 100 km | 0.0100 | 1.69e5 | 288  | 20119 | 11535 |

The last column is the ratio that decides whether any of this matters: a large pipe holds
74 MWh of usable pack against 748 MW of throughput, i.e. **six minutes of its own rated
flow**.

## 3. What the implementation adds, in one table

Per arc and timestep, on top of the basic formulation:

| object | where | why it exists |
|---|---|---|
| `var_pressure[t, node]` | `:282`, on the network block | the potential the flows are driven by |
| `var_flow_in`, `var_flow_out` | `:313`, `:318` | inlet and outlet differ by what is packed |
| `var_flow` (inherited), the mean of the two | `:325` | the parents' opex, emissions and size constraints all read it |
| `var_direction[t]`, binary | `:323` | which of the two directed arc blocks is active |
| `var_delta_pressure[t]` | `:591` | the non-negative pressure drop in the active direction |
| `var_lambda[t, z]` + SOS2 | `:596`, `:674` | interpolation on the pressure–flow curve |
| `var_linepack[t, arc]` | `:376`, on the unique arcs | stored energy, one per pipeline and not per direction |
| `pressure_coupled`, a json switch | `:252`, `:283` | whether the arcs meeting at a node share one pressure, see D9b |

plus the network-level constraints: the linepack definition and balance (`:434`–`:482`),
the direction constraints (`:488`) and the two no-flow pressure equalizations (`:545`,
`:554`).

## 4. The decisions

Each one states the basic formulation, what was done instead, and what it costs.

### D1 — One-dimensional PWL around a reference pressure

$p_i^2-p_j^2 = (p_i-p_j)(p_i+p_j)$, so the flow depends on the pressure *difference* and
on the *absolute level*. A faithful approximation is therefore two-dimensional.

**Done instead:** fix the low pressure at `pressure_ref` and build a one-dimensional curve
(`fixed_size_pipeline.py:139`):

$$ F_z = \sqrt{R_e\,\Delta P_z\,\bigl(2p^{\mathrm{ref}} + \Delta P_z\bigr)},
\qquad \Delta P_z \in \mathrm{linspace}\bigl(0,\; p^{\max}-p^{\mathrm{ref}},\; Z\bigr). $$

**Why:** a 2-D SOS2 needs a triangulation of the $(\Delta p,\bar p)$ plane and roughly
squares the interpolation variables, for an effect that is second order inside a 15 bar
band. A 1-D relation was judged sufficient for a first implementation (note 12).

**Cost:** the same $\Delta p$ gives the same flow whether the pipe sits at the bottom or at
the top of its band, which it does not in reality — a pipe running near $p^{\max}$ moves
more gas for the same drop. The error is one-sided and known: the curve is built at the
*reference* level, i.e. the middle of the band.

**Discretization:** the breakpoints are uniform in $\Delta p$, so they are *not* uniform in
flow — the curve is steepest at the origin, and for medium/20 km the first chord has slope
56 MW/bar against 18 MW/bar for the last. Five breakpoints is the default
(`nr_breakpoints`), and the first chord is the crudest piece of the whole approximation.

### D2 — The arc capacity is defined by the same curve

**Basic formulation:** `size_max` comes from the json and the size is a continuous
decision.

**Done instead:** the capacity of an arc is the top breakpoint, i.e. the flow at
$\Delta p = p^{\max}-p^{\mathrm{ref}}$ (`fixed_size_pipeline.py:155`), and an arc is either
not built or built at that capacity. One json file is one pipeline type; choosing a
capacity means choosing a type.

**Why:** it makes `FixedSizePipeline` and `FluidynamicPipeline` **the same investment
problem** — same corridors, same types, same capacity per arc, same capex — so the
difference between the two objectives is the transport model and nothing else. Without it
the comparison is confounded by the design space.

**Consequence worth remembering:** the capacity depends on the arc length, so the same type
carries less over a longer corridor. And, since the reference capacity *is* the flow at the
top breakpoint, **the two models coincide exactly when the pipes run flat out** — see §6.

### D3 — Direction as a per-timestep binary, not a disjunction

**Basic formulation:** for a bidirectional network AdOpT writes each pipeline as two
directed arc blocks, adds the cut $f_{ij}+f_{ji}\le \text{size}$ (`network.py:871`), and
optionally a GDP disjunction forbidding both directions at once
(`bidirectional_network_precise = 1`, `network.py:883`).

**Done instead:** `var_direction[t]` per directed arc block; at most one direction of a
pipeline active (`:519`); flow only in the active direction (`:333`, `:343`); and the
parent's disjunction switched off in `__init__` (`:181`), because it would enforce the same
condition a second time.

**Why:** the direction binary is needed anyway. $\Delta p$ and $|f|$ are both non-negative
in the PWL, so the sign has to live somewhere, and the two arc blocks are exactly the
forward/reverse split of note 12. Keeping the GDP on top would double the integrality for
no extra information.

### D4 — `sum(lambda) = direction`, not `= 1`

**Textbook SOS2:** $\sum_z \lambda_z = 1$.

**Done instead** (`:643`): $\sum_z \lambda_z = \xi_{e,t}$, the direction binary.

**Why:** it makes an idle or unbuilt arc consistent. With $=1$ every arc would have to sit
on the curve at every hour, i.e. carry flow, and the end pressures of an unbuilt corridor
would still be coupled. With $=\xi$ an inactive direction gets $\lambda = 0$,
$\Delta p = 0$, $f = 0$, and that arc imposes nothing on its end pressures.

**Cost:** presolve cannot touch the SOS2 sets, because their sum is tied to a variable
rather than to a constant. That is one of the reasons the model is expensive (§6).

### D5 — Split big-M on the pressure difference

`var_delta_pressure` is tied to $p_i-p_j$ only in the active direction, which needs a
big-M. Written naively the constant is the full band, $p^{\max}-p^{\min}$.

**Done instead** (`:604`–`:638`), two terms:

```
dP_active  = max(breakpoints)               # = p_max - p_ref
dP_unbuilt = (p_max - p_min) - dP_active
delta_p <= p_from - p_to + dP_active * (1 - direction) + dP_unbuilt * (1 - installed)
```

**Why:** on a *built* pipeline the difference can never leave
$[-\Delta P_{\max},+\Delta P_{\max}]$, because the active direction ties it to a bounded
variable and the inactive one is the active direction of the opposite arc, which forces the
other sign. Only an unbuilt arc has its end pressures free over the whole band. Written as
one term the constant would be twice as large everywhere, since the reference pressure sits
in the middle of the band — a looser relaxation for nothing. The same split appears in the
no-flow pressure constraints (`:538`).

### D6 — The linepack lives on the unique arc, and is tied to `installed`

**Done** (`:404`–`:455`): four constraints instead of one equality.

```
lp <= K * p_max * installed                         # an unbuilt pipe holds nothing
lp >= K * p_min * installed                         # a built one holds at least the minimum
lp <= K * (p_from + p_to)/2                         # needs no big-M: lp >= 0 and p > 0
lp >= K * (p_from + p_to)/2 - K * p_max * (1 - installed)
```

**Why the first two:** without them the linepack of an unbuilt arc is a *free constant*.
Its bounds do not force it to zero, the definition constraints are switched off by their
own big-M, and the balance only ties it to itself once the flow is zero — so it floats at
no cost whenever `installed` is fractional, which is most of the LP relaxation. Tying it to
the install binary also hands the built case a genuine lower bound, since a built pipeline
sits at least at $p^{\min}$.

**Why on the unique arc:** the pack belongs to the pipeline, not to a direction of it. The
balance therefore sums the net injection over both arc blocks (`:383`), at most one of
which is non-zero in a given hour.

### D7 — Cyclic balance, and typical days closed per day

**Done** (`:460`): no initial pack is imposed, and hour 1 is coupled to the last hour, as
for storage technologies. Under typical days with `method 1` the cycle is closed **within
each typical day** instead, using `hours_per_day`.

**Why not the storage trick:** storage technologies in AdOpT keep the level at full
resolution and map the flows through the sequence of typical days. That cannot be done
here — the linepack is not a free variable, it is pinned to the pressures, so it cannot be
at a finer resolution than the pressures, and the pressures cannot be finer than the flows
they are tied to. Closing the cycle per typical day means the pack cannot be carried from
one day into the next, which is acceptable for an intra-day store.

**Cost, and it is the expensive one.** Combining the definition with the balance makes the
pressure at hour *t* equal to hour 1's pressure plus the running sum of net injections,
with every partial sum confined to $[K p^{\min}, K p^{\max}]$ and the total over the
horizon equal to zero. That is a path constraint of length *T* per arc, and it is what a
rounding heuristic cannot respect: fix the directions, set the flows greedily hour by hour,
and the accumulated pack walks out of the pressure band somewhere in the middle of the
horizon (§6).

### D8 — Pipeline pressure kept separate from compressor pressure

**Basic formulation:** the compressor work of a fluid network is a constant times the flow,
derived from fixed `inlet`/`outlet` pressures in the json (`fluid.py:147`).

**Done instead:** the compressors keep seeing the fixed reference pressure, while
`var_pressure` moves freely between `pressure_min` and `pressure_max`. The two are not
linked.

**Why:** deliberately staged (note 12, last line). Linking them makes the compressor work
bilinear in flow and pressure ratio, i.e. a second non-linearity on top of the one being
approximated.

**Cost:** compression is priced as if the network always ran at $p^{\mathrm{ref}}$, so the
model does not pay more for pushing a pipe to the top of its band. It therefore slightly
over-values packing the line. Quantifying that bias is an open item.

### D9 — One pressure field per network object

`var_pressure` is declared on the network block (`:282`), indexed by the nodes the arcs of
*that network* touch. A node served by three pipeline types therefore has three independent
hydrogen pressures.

**This is a known defect, not a choice.** `medium` and `large` share the band 15–30 bar and
should share the node pressure; as written the model may hold one at 30 bar and the other
at 15 bar at the same node in the same hour. It is both physically wrong and a relaxation.
The fix is R2 in note 13: move the variable up to the model block, keyed by (pressure band,
node).

### D9b — `pressure_coupled`, a switch in the json

`"pressure_coupled": 0` indexes the pressure by the *pipeline end* rather than by the node
(`_get_pressure_key`, `:252`), so the two directions of a pipeline still share their two end
pressures but the arcs that meet at a node no longer share anything. Every other constraint
is untouched, and `pressure_coupled: 1` (the default, and what a json without the key gets)
reproduces the model of D9 exactly — same variables, same constraints, same root LP,
verified against the pre-change file on `1L3S` at 24 h.

**It is a relaxation, so it gives a lower bound and not a transport model.** With the
coupling off the pressure does not propagate: each arc keeps its own pipeline equation,
direction binary and linepack, at a pressure level of its own, and what is left is close to
`FixedSizePipeline` plus a pack that costs nothing to position. It also invalidates the path
and circulation cuts of `run_decomposed.py`, which rest on
$\sum_{a\in P}\Delta p_a = p_{\mathrm{start}} - p_{\mathrm{end}}$, i.e. on the coupling
itself.

**What it measures** (see §6): the coupling adds no LP strength on these cases — the root
relaxation is identical with and without it — so everything it costs is LP weight per node.

### D10 — `linepack_on`, a switch in the json

`"linepack_on": 0` zeroes $K^{\mathrm{LP}}$ in `construct_netw_model` (`:196`). Everything
else follows on its own: the bounds become $(0,0)$, the definition constraints pin the
variable to zero, and the balance degenerates to $f^{\mathrm{in}}=f^{\mathrm{out}}$ — the
pipeline equation without the storage. It isolates which half of the model costs what, and
it is the experiment §6 reports.

### D11 — Node balances read the endpoint flows

`_define_inflow_constraints` and `_define_outflow_constraints` are overridden (`:681`,
`:700`) so that a node sends `var_flow_in` and receives `var_flow_out`. The inherited
`var_flow`, the average, stays what the parents' opex, emissions and size constraints see.
Without the override the node balance would use the average and the pack would leak into
the energy balance.

## 5. What is not modelled

Against a transient pipeline model (Shchetinin et al., note 1) the implementation gives up,
deliberately:

- **spatial discretization.** One segment per pipe, so no pressure profile along the line
  and no wave propagation; the pack is a function of the two endpoint pressures only.
- **momentum inertia.** The steady-state momentum balance is used at every hour
  ("quasi-dynamic"), valid when the timestep is long against the travel time — at hourly
  resolution it is.
- **temperature and compressibility dynamics.** Both `T` and `Z` are constants per network.
- **the $f \ge F(\Delta p)$ side**, under the planned R1 relaxation only (§7).
- **compressor–pressure coupling** (D8).

## 6. What it costs, and what it is worth so far

Measured on `examples/ladder/1L3S` (four nodes, no backbone, 160 h, Gurobi 13, 4 threads,
`mipgap 1e-2`); full detail in note 13.

| | reference | linepack **off** (D10) | linepack **on** |
|---|---|---|---|
| rows / cols / nonzeros | 88 725 / 75 674 / 216 358 | — | 143 463 / 126 554 / 469 834 |
| binaries after presolve | 2 898 | — | 5 778 |
| SOS2 sets | 0 | 5 760 | 5 760 |
| B&B nodes | **1** | **1** | **1 197** |
| time | **2.5 s** | **91 s** | **4 154 s** |
| objective | 4.131319e7 | 4.128101e7 | 4.126188e7 |
| gap reached | 0.75 % | 0.42 % | 0.25 % |

Three readings, and they decide where the work goes next:

1. **The cyclic pack balance (D7) is what causes the branching.** Switch the pack off and
   the model solves at the root, like the reference: 1 197 nodes become 1, 45× faster.
2. **The pipeline equation (D1, D3, D4) is a constant factor at the root**, 37× — a heavier
   LP and a long cut loop, not an explosion. The SOS2 sets are never branched on here.
3. **The pack *weakens* the root bound** (4.0965e7 against the reference's 4.1002e7): it
   buys the LP degrees of freedom that the MIP then has to branch away.

And the physical result, consistent across the ladder and the full NL case:

- on `1L3S` the linepack is worth **0.046 %**, and the pack swings ~705 MWh against
  ~181 GWh transported, i.e. 0.4 %;
- on the NL case at 24 h with imports capped, the fluidynamic optimum equals the
  capacity-only optimum **to ten digits**, proved at a 0.0000 % gap;
- in both, $\Delta p$ **saturates the top breakpoint** on every active arc — and by D2 that
  is precisely where the two transport models are the same model.

Both differences are smaller than the tolerance they were measured at, so no claim about
what linepack is worth survives without a rerun at `mipgap 1e-4`. What is already clear is
the regime question: 25 GW pipes carrying a few hundred MW are nowhere near where packing a
line does anything. A case that separates the two models needs **narrow corridors near
capacity, long distances, and supply swinging faster than demand**.

### What switching the coupling off is worth

Same case, same horizon, same design problem, same solver options, the only difference being
`pressure_coupled`. `mipgap 1 %` throughout, so the objectives carry that tolerance.

| case | coupling | nodes | time / gap | objective |
|---|---|---|---|---|
| 1L3S, 24 h | on | 27 | 11.1 s, 0.0006 % | 1 356 202.6127287 |
| 1L3S, 24 h | off | 200 | 13.1 s, 0.748 % | 1 356 202.6127287 |
| 1L3S, 96 h | on | 1 | 76.3 s, 0.52 % | 1.9921328e7 |
| 1L3S, 96 h | off | 1 | 42.3 s, 0.45 % | 1.9884891e7 |
| 3L3S, 168 h, `network`, 3 types | on | 5 200 | **18 000 s (cap), 2.915 %** | 1.6926948e7 |
| 3L3S, 168 h, `network`, 3 types | off | 3 014 | **8 186 s, 0.992 % (solved)** | 1.6793028e7 |

Three readings:

1. **It buys nothing on an easy instance and costs nodes.** At 24 h the objective is the
   same to ten digits and decoupling *tripled* the time — the coupling was never binding, so
   removing it only lost the solver structure it was exploiting.
2. **It is worth a lot on a hard one, and the win is in the LP, not in the tree.** On the
   3L3S job the coupled run sat at the root for 1 271 s, had no useful incumbent until
   1 874 s (gap 16.8 %) and ended on its 5 h cap at 2.915 %; the decoupled run had an
   incumbent at 72 s, passed that five-hour gap after ~25 minutes, and *solved* the instance
   to the 1 % target in 8 186 s. Simplex iterations per node fell from ~20 000 to ~1 000.
   Node counts do not explain it: the coupled run explored *more* nodes (5 200 against
   3 014) and still ended worse.
3. **The coupling adds no LP strength on these cases.** The root relaxation is identical with
   and without it — 1.6215e7 on the 3L3S job, matching the stored coupled run to six digits.
   So it is pure weight: it makes every LP bigger without making the bound better.

The sandwich holds throughout, as it must for a relaxation:
coupled bound $\le$ decoupled optimum $\le$ coupled optimum
(1.6434e7 $\le$ 1.6793e7 $\le$ 1.6927e7 on the 3L3S job). The decoupled optimum sits 0.8 %
below the coupled incumbent, which is what the node coupling is worth in cost on this case —
and is inside the 1 % tolerance both were solved at, so it bounds the effect rather than
measuring it.

.. note::
    The designs are not the same. On the 3L3S job the decoupled linepack model builds
    `large` on Chemelot–Venlo (471) and Venlo–Arnhem (396) where the reference builds
    `large` on Rotterdam–Arnhem (338) and `small` on Chemelot–Venlo (47), and the backbone
    carries a real pack — Zeeland–Chemelot 20 404 MWh mean with a 17 041 MWh swing. This is
    the first instance in the ladder where the pack does anything, against 0.4 % on
    `1L3S`. It is a *relaxation's* design, so it says the regime is worth investigating,
    not that the coupled model would choose the same network.

.. note::
    The coupled 3L3S figures are the stored sweep run (`sweep/20260917215637`), which ran 8
    jobs at a time, against a decoupled run that had the machine to itself. The contention
    flatters the decoupled side; the root relaxation and node counts do not depend on it, and
    the 1 271 s root is too large a difference for contention to explain.

## 7. Open items

| id | change | status |
|---|---|---|
| R5 | close the pack cycle daily instead of over the horizon | first in line — the only option aimed at the measured branching driver, and `hours_per_typical_day` already implements the mechanism |
| R1 | replace the SOS2 equality by the $Z-1$ secant inequalities $f \le s_k\Delta p + b_k\chi$ | the curve is concave, so the secants describe the region under it *exactly*; removes every direction binary and SOS2 set, leaving only the install binaries. Gives up the $f \ge F(\Delta p)$ side, i.e. it allows throttling |
| R12 | two-dimensional $(\Delta p, \bar p)$ approximation, removing D1 | unaffordable under SOS2, cheap under R1: $f \le \sqrt{2R_e\Delta p\,\bar p}$ is a geometric mean, hence concave, hence tangent planes and no binaries |
| R2 | share the node pressure across types of the same band (D9) | correctness, not only speed. Note the direction: R2 adds coupling, and the measurement of D9b says coupling is what the hard instances pay for — so R2 makes the model more correct and slower, and the two cannot both be had without R1 |
| R3 / R4 / R10 | at most one type per corridor; tighter flow big-M; tighter pressure bounds | safe tightenings |
| — | rerun the pair at `mipgap 1e-4` | without it neither the 0.12 % nor the 0.046 % means anything |
| — | quantify the D8 compressor decoupling | unquantified bias, direction known |

## 8. Using it

A pipeline type is one json in `network_data/`, with `"network_type":
"fluidynamic_pipeline"` and, under `Performance`: `diameter`, `roughness`, `pressure_min`,
`pressure_max`, `pressure_ref`, `temperature`, `compressibility_factor`, `molar_mass`,
`energy_density`, `nr_breakpoints`, `linepack_on`, `pressure_coupled`. `size_max` in the
json is ignored — the capacity comes from the geometry (D2). `linepack_on` and
`pressure_coupled` both default to 1, so a json written before they existed is unchanged.

```bash
cd examples/ladder
python run.py --rung 1L3S --hours 24 --case both       # reference first, then linepack
python run.py --rung 1L3S --hours 24 --case linepack
python run.py --rung 1L3S --hours 24 --case linepack --no-linepack            # D10
python run.py --rung 1L3S --hours 24 --case linepack --no-pressure-coupling   # D9b
```

`run.py` writes both switches into the four network jsons before a run and puts them back
afterwards, so a run that fails does not leave the case in a diagnostic state.

.. note::
    `adopt_net0` is not installed into the environment, so `run.py` has to find it on the
    path: run it from the repo root as `python examples/ladder/run.py ...`, or with
    `PYTHONPATH` pointing there. `cd examples/ladder && python run.py` fails with
    `ModuleNotFoundError: No module named 'adopt_net0'`, because python puts the *script's*
    folder on the path and not the working directory.

Results written per arc: `flow_in`, `flow_out`, `delta_pressure`, `direction`; per unique
arc `linepack`; per node `pressure` (`:718`). With `pressure_coupled: 0` the pressure is
written per pipeline end instead, under the key `nodeFrom|nodeTo|node`.

## Related

- `12. Linepack linearization for adopt.md` — the derivation
- `13. How to make model faster - formulation.md` — the measurements and R1–R12
- `11. Implementing adopt comparison.md` — the three fidelity levels and the comparison design
- `1. 09-04-2026 Understanding linepack.md` — the transient model this one approximates
- `examples/ladder/README.md` — the case ladder
