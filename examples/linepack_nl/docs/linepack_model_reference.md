# Linepack model — sets, parameters, variables, constraints

Date: 2026-11-10. A reference card of what `FluidynamicPipeline` actually builds, read off
the code, not off the derivation. For the reasoning behind each choice see
`linepack_formulation.md` (decisions D1–D11); for the physics, vault note
*12. Linepack linearization for adopt*.

Code:

- `FP` = `adopt_net0/components/networks/specificNetworks/fluidynamic_pipeline.py`
- `FSP` = `adopt_net0/components/networks/specificNetworks/fixed_size_pipeline.py`
- `NW` = `adopt_net0/components/networks/network.py`
- `FL` = `adopt_net0/components/networks/genericNetworks/fluid.py`

Line numbers refer to commit `9e530a28`.

Class chain: `Network` → `Fluid` → `FixedSizePipeline` → `FluidynamicPipeline`. One json in
`network_data/` is one pipeline type, i.e. one network object, with its own blocks,
binaries and pressures.

---

## 1. Sets

| Symbol | Code | Meaning |
|---|---|---|
| $t \in \mathcal{T}$ | `self.set_t` | timesteps (hours, or averaged steps under time staging) |
| $a=(i,j) \in \mathcal{A}$ | `b_netw.set_arcs`, `b_netw.arc_block[i, j]` | directed arcs. A bidirectional pipeline is **two** arc blocks, $(i,j)$ and $(j,i)$ |
| $\bar a = (j,i)$ | | the reverse arc of $a$ |
| $e \in \mathcal{E}$ | `b_netw.set_arcs_unique` | unique arcs = physical pipelines, one per corridor |
| $n \in \mathcal{N}^p$ | `b_netw.set_pressure_nodes` (`FP:325`) | pressure keys: the node name if `pressure_coupled = 1`; one key per pipeline end (from, to, node joined by a pipe) if `0` (`FP:252`) |
| $z \in \{0..Z-1\}$ | `b_arc.set_breakpoints` (`FP:648`) | breakpoints of the PWL curve, $Z$ = `nr_breakpoints` |

Notation below: $p_{i}$ means `_get_pressure(b_netw, t, arc, i)`, i.e. the pressure of node
$i$ *as seen by arc* $a$. With coupling on, all arcs of the network meeting at $i$ see the
same variable; with coupling off, each pipeline has its own.

---

## 2. Parameters

### 2.1 Read from the json (`Performance`, `FSP:56`)

| Symbol | json key | Unit |
|---|---|---|
| $D$ | `diameter` | m |
| $\varepsilon$ | `roughness` | m |
| $p^{\min}, p^{\max}$ | `pressure_min`, `pressure_max` | bar |
| $p^{\mathrm{ref}}$ | `pressure_ref` | bar |
| $T$ | `temperature` | K |
| $Z_c$ | `compressibility_factor` | – |
| $M$ | `molar_mass` | kg/kmol |
| $ed$ | `energy_density` | MWh/t |
| $Z$ | `nr_breakpoints` | – |
| | `linepack_on` (default 1, `FP:179`) | switch |
| | `pressure_coupled` (default 1, `FP:180`) | switch |

Plus, from the topology: $L_{a}$, the arc distance in km (`self.distance`). Plus the usual
network economics: `gamma1..4`, `opex_variable`, `opex_fixed`, `loss`, the compressor
`energyconsumption` block.

### 2.2 Derived in `fit_network_performance` (`FSP:78`)

| Symbol | Code key | Formula | Unit |
|---|---|---|---|
| $R_s$ | `r_specific` | $8314 / M$ | J/(kg K) |
| $f_D$ | `friction_factor` | $\bigl(2\log_{10}(3.7D/\varepsilon)\bigr)^{-2}$ (Nikuradse) | – |
| $R_a$ | `r_pipeline` | $\dfrac{ed^2\,3.6^2\,\frac{\pi^2}{16}\,D^5\,10^{10}}{f_D\cdot 1000\cdot Z_c R_s T}\cdot\dfrac{1}{L_a}$ | MW²/bar² |
| $K_a$ | `linepack` | $ed\cdot\dfrac{\pi D^2}{4}\,1000\cdot\dfrac{100}{Z_c R_s T}\cdot L_a$ | MWh/bar |
| $B_z$ | `breakpoints_delta_pressure` | $\mathrm{linspace}(0,\;p^{\max}-p^{\mathrm{ref}},\;Z)$ | bar |
| $F_{a,z}$ | `breakpoints_flow[a]` | $\sqrt{R_a\,B_z\,(2p^{\mathrm{ref}}+B_z)}$ | MW |
| $\bar F_a$ | `size_max_arcs` → `b_arc.para_size_max` | $\max_z F_{a,z}$ (json `size_max` ignored) | MW |

If `linepack_on = 0`, $K_a := 0$ in `construct_netw_model` (`FP:202`).

### 2.3 Derived in the constraints

| Symbol | Formula | Where |
|---|---|---|
| $\Delta P^{\mathrm{act}}$ | $\max_z B_z = p^{\max}-p^{\mathrm{ref}}$ | `FP:588`, `FP:672` |
| $\Delta P^{\mathrm{unb}}$ | $(p^{\max}-p^{\min}) - \Delta P^{\mathrm{act}}$ | `FP:589`, `FP:673` |
| $\Delta t$ | `nr_timesteps_averaged` (time staging value, else 1) | `FP:207` |
| $H$ | `hours_per_typical_day`: hours per day if typical days with `method 1`, else 0 | `FP:216` |
| $\rho$ | `rated_capacity` (default 1, so flow and size are both in MW) | `NW:232` |

---

## 3. Variables

### 3.1 Per directed arc $a$ (in `arc_block[a]`)

| Symbol | Code | Domain / bounds | Defined | Meaning |
|---|---|---|---|---|
| $y_a$ | `var_installed` | binary | `FSP:184` | arc built (fixed to 1 if `existing`) |
| $s_a$ | `var_size` | $= y_a \bar F_a$ | `NW:548`, `FSP:189` | size, MW |
| $f_{a,t}$ | `var_flow` | $[0, \bar F_a\rho]$ | `NW:758` | **mean** flow in the pipe; read by losses, opex, compressor energy |
| $f^{\mathrm{in}}_{a,t}$ | `var_flow_in` | $[0, \bar F_a\rho]$ | `FP:362` | flow entering at the sending node |
| $f^{\mathrm{out}}_{a,t}$ | `var_flow_out` | $[0, \bar F_a\rho]$ | `FP:367` | flow leaving at the receiving node |
| $\ell_{a,t}$ | `var_losses` | $[0, \bar F_a\rho]$ | `NW:766` | transport losses |
| $\xi_{a,t}$ | `var_direction` | binary | `FP:372` | direction $a$ is the active one in hour $t$ |
| $\delta_{a,t}$ | `var_delta_pressure` | $[0, \Delta P^{\mathrm{act}}]$ | `FP:652` | pressure drop along $a$, non-negative |
| $\lambda_{a,t,z}$ | `var_lambda` | $[0,1]$, SOS2 over $z$ | `FP:657` | interpolation weights |
| | `var_consumption_send/receive` | $\ge 0$ | `FL:192` | compressor electricity at the two ends |
| | `var_capex`, `var_capex_aux` | | `NW:657` | arc capex |

### 3.2 Per pipeline $e$ (on the network block)

| Symbol | Code | Domain / bounds | Defined | Meaning |
|---|---|---|---|---|
| $L_{e,t}$ | `var_linepack` | $[0, K_e p^{\max}]$ | `FP:425` | stored energy, MWh |

### 3.3 Per pressure key $n$ (on the network block)

| Symbol | Code | Domain / bounds | Defined | Meaning |
|---|---|---|---|---|
| $p_{n,t}$ | `var_pressure` | $[p^{\min}, p^{\max}]$ | `FP:331` | pressure, bar |

### 3.4 Size per pipeline and hour

Two arc blocks, each with $f, f^{\mathrm{in}}, f^{\mathrm{out}}, \ell, \delta$, $Z$ weights
$\lambda$, one binary $\xi$ and one SOS2 set; plus one $L$ and two $p$ (shared with other
pipelines when coupled). With $Z=5$: 2 binaries, 2 SOS2 sets, about 20 continuous
variables per pipeline-hour.

---

## 4. Constraints

All per $t \in \mathcal{T}$ unless stated otherwise.

### 4.1 Investment (per arc, no $t$)

| # | Constraint | Code |
|---|---|---|
| C1 | $s_a = y_a\,\bar F_a$ | `FSP:189` `const_size_installed` |
| C2 | $y_a = 1$ if the arc is existing | `FSP:187` `const_installed` |
| C3 | $\mathrm{capex}_a = \gamma_1 y_a + \gamma_2 s_a + \gamma_3 L_a y_a + \gamma_4 s_a L_a$ (0 if existing) | `FSP:210` `const_capex_aux` |
| C4 | $y_a = y_{\bar a}$ | `FP:552` `const_installed_bidirectional` |
| C5 | $s_a = s_{\bar a}$ | `NW:858` `const_size_bidirectional` (inherited) |

### 4.2 Flow on an arc

| # | Constraint | Code |
|---|---|---|
| C6 | $f_{a,t} = \tfrac12\bigl(f^{\mathrm{in}}_{a,t} + f^{\mathrm{out}}_{a,t}\bigr)$ | `FP:374` `const_flow_average` |
| C7 | $f^{\mathrm{in}}_{a,t} \le \xi_{a,t}\,\bar F_a\rho$ | `FP:382` `const_flow_direction_in` |
| C8 | $f^{\mathrm{out}}_{a,t} \le \xi_{a,t}\,\bar F_a\rho$ | `FP:392` `const_flow_direction_out` |
| C9 | $f_{a,t} \le s_a\rho$ | `NW:785` `const_flow_size_high` (inherited) |
| C10 | $f_{a,t} + f_{\bar a,t} \le s_a\rho$ | `NW:871` `const_cut_bidirectional` (inherited) |
| C11 | $\ell_{a,t} = f_{a,t}\cdot\mathrm{loss}\cdot L_a$ | `NW:776` `const_flowlosses` |

The parent's GDP disjunction (`bidirectional_network_precise`) is switched off in
`__init__` (`FP:185`): C7, C8 and C17 do its job.

### 4.3 Pipeline equation — PWL with SOS2 (per arc)

The target is $f\lvert f\rvert = R_a(p_i^2-p_j^2)$. With the low pressure fixed at
$p^{\mathrm{ref}}$ it becomes the 1-D curve $F(B)$ through the points $(B_z, F_{a,z})$.

| # | Constraint | Code |
|---|---|---|
| C12 | $\sum_z \lambda_{a,t,z} = \xi_{a,t}$ | `FP:706` `const_lambda` |
| C13 | $\delta_{a,t} = \sum_z B_z\,\lambda_{a,t,z}$ | `FP:714` `const_interpolation_delta_pressure` |
| C14 | $f_{a,t} = \sum_z F_{a,z}\,\lambda_{a,t,z}$ | `FP:724` `const_interpolation_flow` |
| C15 | $\mathrm{SOS2}\bigl(\lambda_{a,t,0},\dots,\lambda_{a,t,Z-1}\bigr)$, weights $z+1$ | `FP:735` `const_lambda_sos2` |

With $\xi_{a,t}=0$: $\lambda = 0 \Rightarrow \delta = 0,\ f = 0$, and by C6–C8
$f^{\mathrm{in}} = f^{\mathrm{out}} = 0$.

C14 interpolates the **mean** flow $f$, not $f^{\mathrm{in}}$ or $f^{\mathrm{out}}$.

### 4.4 Drop and node pressures (per arc $a=(i,j)$)

| # | Constraint | Code |
|---|---|---|
| C16a | $\delta_{a,t} \le (p_{i,t}-p_{j,t}) + \Delta P^{\mathrm{act}}(1-\xi_{a,t}) + \Delta P^{\mathrm{unb}}(1-y_a)$ | `FP:675` `const_delta_pressure_high` |
| C16b | $\delta_{a,t} \ge (p_{i,t}-p_{j,t}) - \Delta P^{\mathrm{act}}(1-\xi_{a,t}) - \Delta P^{\mathrm{unb}}(1-y_a)$ | `FP:690` `const_delta_pressure_low` |

Active and built: $\delta_{a,t} = p_i - p_j \ge 0$, so the flow goes downhill. Inactive
($\delta = 0$): $\lvert p_i - p_j\rvert \le \Delta P^{\mathrm{act}}$. Unbuilt: free over the
whole band.

### 4.5 Directions and no-flow pressures (per pipeline $e=(i,j)$)

| # | Constraint | Code |
|---|---|---|
| C17 | $\xi_{a,t} + \xi_{\bar a,t} \le y_a$ | `FP:572` `const_direction_unique` |
| C18a | $p_{i,t}-p_{j,t} \le \Delta P^{\mathrm{act}}\bigl(\xi_{a,t}+\xi_{\bar a,t}+1-y_a\bigr) + \Delta P^{\mathrm{unb}}(1-y_a)$ | `FP:598` `const_no_flow_pressure_high` |
| C18b | $p_{i,t}-p_{j,t} \ge -\bigl[\Delta P^{\mathrm{act}}\bigl(\xi_{a,t}+\xi_{\bar a,t}+1-y_a\bigr) + \Delta P^{\mathrm{unb}}(1-y_a)\bigr]$ | `FP:611` `const_no_flow_pressure_low` |

C17 lets a built pipe be idle (both $\xi = 0$); C18 then forces $p_i = p_j$ on it.

### 4.6 Linepack (per pipeline $e=(i,j)$, with $\bar p_{e,t} = \tfrac12(p_{i,t}+p_{j,t})$)

| # | Constraint | Code |
|---|---|---|
| C19 | $L_{e,t} \le K_e\,p^{\max}\,y_e$ | `FP:453` `const_linepack_installed_high` |
| C20 | $L_{e,t} \ge K_e\,p^{\min}\,y_e$ | `FP:466` `const_linepack_installed_low` |
| C21 | $L_{e,t} \le K_e\,\bar p_{e,t}$ | `FP:483` `const_linepack_high` |
| C22 | $L_{e,t} \ge K_e\,\bar p_{e,t} - K_e\,p^{\max}(1-y_e)$ | `FP:499` `const_linepack_low` |
| C23 | $L_{e,t} = L_{e,\mathrm{prev}(t)} + \Delta t\sum_{d\in\{a,\bar a\}}\bigl(f^{\mathrm{in}}_{d,t}-f^{\mathrm{out}}_{d,t}\bigr)$ | `FP:513` `const_linepack_balance` |

Built: C21 + C22 give $L = K\bar p$. Unbuilt: C19 gives $L=0$.

$\mathrm{prev}(t)$ in C23:

- no typical days: $t-1$, and $\mathrm{prev}(1) = \lvert\mathcal{T}\rvert$ (cyclic over the
  horizon);
- typical days, `method 1`: $t-1$ inside the day, and the first hour of each day points
  to the last hour of the *same* day (cyclic per day, no carry-over between days).

### 4.7 Coupling to the rest of the model (per node $n$, carrier)

| # | Constraint | Code |
|---|---|---|
| C24 | $\mathrm{inflow}_{n,t} = \sum_{i \to n}\bigl(f^{\mathrm{out}}_{(i,n),t} - \ell_{(i,n),t}\bigr)$ | `FP:753` `const_inflow` |
| C25 | $\mathrm{outflow}_{n,t} = \sum_{n \to j} f^{\mathrm{in}}_{(n,j),t}$ | `FP:773` `const_outflow` |
| C26 | $\mathrm{cons}^{\mathrm{send}}_{a,t} = f_{a,t}\,(k^{\mathrm{flow}} + k^{\mathrm{flowDist}} L_a)$, same at the receiving end | `FL:212`, `FL:226` |
| C27 | variable opex $\propto \sum_t f_{a,t}$; fixed opex $\propto$ capex | `NW:812`, `NW:943` |

The node balance sees the **endpoint** flows (C24, C25), so whatever is packed in the line
is not delivered. Losses, compressor energy and variable opex see the **mean** flow $f$.
The compressor coefficients $k$ come from the fixed json pressures (`FL:103`); they do
**not** depend on $p_{n,t}$.

---

## 5. The chain in one picture

```
                    y_a (binary, per arc, C1-C5)
                     |
      xi_a,t <= y    |  (C17)
        |            |
        |  sum lambda = xi  (C12)
        v
   lambda_a,t,z --SOS2-->  delta_a,t = sum B_z lambda   (C13)
                           f_a,t     = sum F_z lambda   (C14)
        |                     |
        |                     | big-M on xi, y   (C16)
        v                     v
   f_in, f_out  <-- C6 -->  p_i,t - p_j,t   <-- C18 (idle: p_i = p_j)
        |                     |
        | C23 balance         | C21/C22 (L = K * p_mean if built)
        v                     v
              L_e,t  (C19/C20: 0 if unbuilt)
        |
   C24/C25 -> node energy balance (f_out in, f_in out)
```

---

## 6. Results written (`FP:784`)

| h5 path | Variable |
|---|---|
| `<network>/<from><to>/flow_in`, `flow_out` | $f^{\mathrm{in}}, f^{\mathrm{out}}$ |
| `<network>/<from><to>/delta_pressure` | $\delta$ |
| `<network>/<from><to>/direction` | $\xi$ |
| `<network>/<from><to>/linepack` (unique arcs only) | $L$ |
| `<network>/<node>/pressure` (uncoupled: one group per pipeline-end key) | $p$ |

Plus everything the parent classes write (`flow`, `losses`, `consumption`, size, capex).

---

## 7. Variants applied on top (`examples/ladder/linepack_formulations.py`)

These are not in `FP`. `run_linepack_formulations.py` builds the model normally and then
rewrites the pipeline blocks before solving. Same integer feasible set, tighter LP.

| Variant | Removes | Adds |
|---|---|---|
| `baseline` | – | – |
| `hull` | C21 | $L \le K\bar p - K p^{\min}(1-y)$ |
| `signed` | C16a/b, C18a/b | $\bigl\lvert(p_i-p_j)-(\delta_a-\delta_{\bar a})\bigr\rvert \le (p^{\max}-p^{\min})(1-y)$ |
| `combined` | hull + signed | hull + signed |
| `combined_cuts` | as `combined` | $K p^{\min} y + \tfrac K2(\delta_a+\delta_{\bar a}) \le L \le K p^{\max} y - \tfrac K2(\delta_a+\delta_{\bar a})$ |
| `magnitude` | `combined`, plus C12–C15 of both arcs | one $\lambda_{e,t,z}$ per pipeline: $\sum_z\lambda = y$; $f_a+f_{\bar a} = \sum_z F_z\lambda$; $\delta_a+\delta_{\bar a} = \sum_z B_z\lambda$; SOS2; C17 as equality $\xi_a+\xi_{\bar a}=y$ with $\xi_{\bar a}$ continuous; $\delta_d \le \Delta P^{\mathrm{act}}\xi_d$ |
| `magnitude_cuts` | as `magnitude` | + the `combined_cuts` inventory/drop rows |
| `magnitude_ratio_cuts` | as `magnitude_cuts` | per direction $d$: $r^{\min}\delta_d \le f_d \le r^{\max}\delta_d$, $r = F_z/B_z$ over $B_z>0$ |

Line references: `_complete_linepack_hull` `:73`, `_link_signed_pressure` `:98`,
`_add_inventory_drop_cuts` `:155`, `apply_formulation` `:195`,
`apply_magnitude_formulation` `:354` (ratio cuts `:483`).

---

## 8. What the code does *not* have (vs note 12)

- **At most one type per corridor** ($\sum_d y_{e,d} \le 1$): not implemented. Each type
  is its own network object with its own $y$ (R3, open).
- **One pressure per node across types:** not implemented. Two types meeting at a node
  have independent $p$ (D9, fix R2).
- **2-D curve in $(\Delta p, \bar p)$:** not implemented. The curve sits at $p^{\mathrm{ref}}$
  (D1, idea R12).
- **Compressor work depending on $p$:** not implemented (D8).
- **An initial linepack:** none. The balance is cyclic (C23).
