# SPEC: EV Charging Scheduling with QUBO + QAOA (Track 4)

Quantum Hack Day Kochi, Qiskit Fall Fest 2026. Solo build. All data is synthetic.

---

## 0. Rules for the coding agent (read first)

1. **This file is the source of truth.** Do not change the instance numbers (§2), the equations (§4), or the penalty-weight rule (§4.5) without asking. If something looks wrong, stop and report it instead of "fixing" it silently.
2. **Gates are hard stops (§10).** Do not start the next stage until the gate for the current one passes. In particular: no QAOA code until `test_qubo.py` passes.
3. **Write the QUBO ourselves.** Build a `QuadraticProgram` with binary variables and an objective only. No `linear_constraint`, no `QuadraticProgramToQubo`, no `MinimumEigenOptimizer` hiding the penalties. Every term in the code must carry a comment naming its equation number from §4 (for example `# E3: assignment penalty`).
4. **Verify Qiskit APIs against the installed version.** Do not rely on memory or old tutorials. Run a smoke test first (§8). If a call fails, read the installed package's docs or source, then fix it.
5. **Determinism.** Fixed seeds everywhere, no network calls at runtime, no timestamps in results. Outputs are cached to `results/` as JSON/PNG so the demo never depends on a live run.
6. **Small, typed, readable functions.** Judges grade code cleanliness. No dead code, no notebooks as the source of truth (notebooks may only call the scripts).
7. **Print term magnitudes** (linear cost, peak term, penalty terms) for the optimum so the penalty weights can be sanity-checked.

---

## 1. Goal and scope

Decide **when and where** each EV charges so that total energy cost, travel cost, waiting time, and grid peak load are minimized, subject to station capacity and each EV's charging requirement (energy needed, arrival, deadline).

Pipeline:

```
instance → prune → QUBO → verify vs brute force → QAOA → evaluate vs baselines → plots → README
```

**Quantum component:** QAOA on the QUBO's Ising Hamiltonian, implemented with Qiskit, simulated exactly with statevector (12 qubits), plus one noisy Aer run for NISQ awareness.

**Not required:** quantum advantage. Honest comparison against classical baselines is the goal.

### Traceability to the problem statement

| Statement item | Where it lives |
|---|---|
| EV locations | `dist(e,s)` in travel cost (E1) |
| Battery levels | SoC → energy needed → duration `d_e` and power `p_e` (§3) |
| Station locations | Station coordinates (§2) |
| Station capacity | Bay overlap penalty (E4); bays = separate resources |
| Electricity cost by time | Price `π_τ` per slot in energy cost (E1) |
| How vehicles are assigned | Binary variable `x(e,s,t)` (§4.1) |
| How schedules are generated | Decoded bitstring → Gantt chart |
| Capacity and charging requirements | Pruning (§3.3) + assignment penalty (E3) + overlap penalty (E4) |
| Relevant performance metrics | §7 |
| Classical comparison | §5 |

---

## 2. Instance (fixed, synthetic)

Time: T = 5 slots, slot index τ ∈ {0,1,2,3,4}, each slot Δ = 1 h.
Electricity price per kWh by slot: π = [0.30, 0.22, 0.15, 0.18, 0.25].

**Stations** (1 bay each, same charger power):

| Station | x | y | Bays | Power P (kW) |
|---|---|---|---|---|
| S0 | 2 | 3 | 1 | 11 |
| S1 | 7 | 1 | 1 | 11 |

**EVs** (0-indexed):

| EV | x | y | Battery B (kWh) | SoC now | SoC target | Arrival a_e (slot) | Deadline f_e (finish by end of slot f_e − 1) |
|---|---|---|---|---|---|---|---|
| EV0 | 1 | 1 | 40 | 0.30 | 0.80 | 0 | 3 |
| EV1 | 6 | 4 | 60 | 0.20 | 0.70 | 0 | 4 |
| EV2 | 4 | 2 | 30 | 0.40 | 0.90 | 2 | 5 |

**Weights:** travel `κ = 0.3` per km (Euclidean distance), waiting `ω = 0.5` per slot of delay, peak `μ = 0.01`.

`instance.py` must emit this instance as `results/instance.json` and also support a seeded random generator (`make_random_instance(seed, n_ev, n_station, T)`) for the scaling note in the README. The fixed instance above is the one used for all reported results.

---

## 3. Derived quantities and pruning

### 3.1 Per-EV requirements
```
E_e = B_e · (SoC_target − SoC_now)           energy needed (kWh)
d_e = ceil( E_e / (P · Δ) )                  charging duration in slots
p_e = E_e / d_e                              average power per active slot (kW)
I(e,t) = { t, t+1, …, t + d_e − 1 }          slots occupied if charging starts at t
```
Expected for the fixed instance: `E = [20, 30, 15]`, `d = [2, 3, 2]`, `p = [10, 10, 7.5]`.

### 3.2 Why start-slot variables
A variable chooses the **start slot**. The charging duration is then fixed by `d_e`. A larger battery therefore occupies a bay for longer, which is what makes the capacity constraint meaningful.

### 3.3 Pruning (classical preprocessing)
Keep `v = (e, s, t)` only if `a_e ≤ t` and `t + d_e ≤ f_e`. Pruning removes variables that can never be feasible, which saves qubits.

Expected for the fixed instance: EV0 starts ∈ {0,1}, EV1 starts ∈ {0,1}, EV2 starts ∈ {2,3}, each with 2 stations. **n = 12 qubits.** Hard limit: n ≤ 14.

Variable index `i` = position in the list sorted lexicographically by `(e, s, t)`. Qubit `i` ↔ variable `i`.

---

## 4. Mathematical formulation

### 4.1 Variables
`x_v ∈ {0,1}` for `v = (e,s,t)` in the pruned set `V`. `x_v = 1` means EV `e` starts charging at station `s` in slot `t`. `V_e` is the set of variables of EV `e`.

### 4.2 Linear cost of a variable
```
(E1)  c_v = p_e · Σ_{τ ∈ I(e,t)} π_τ  +  κ · dist(e,s)  +  ω · (t − a_e)
            energy cost                  travel cost        waiting cost
```

### 4.3 Grid load per slot
```
(E2)  L_τ(x) = Σ_{v : τ ∈ I(v)} p_{e(v)} · x_v
```

### 4.4 Full objective (what QAOA minimizes)
```
H(x) =  Σ_v c_v x_v                                   linear cost
      + μ · Σ_τ L_τ(x)²                               peak-load proxy
      + λ_A · Σ_e ( Σ_{v∈V_e} x_v − 1 )²              (E3) each EV charges exactly once
      + λ_O · Σ_{(u,v)∈C} x_u x_v                     (E4) one EV per bay at a time
```
`C` = pairs `u=(e,s,t)`, `v=(e',s,t')` with `e ≠ e'`, the same station `s`, and `I(u) ∩ I(v) ≠ ∅`.

For a feasible `x` (penalty terms zero), `H(x) = J(x)` where
```
J(x) = Σ_v c_v x_v + μ · Σ_τ L_τ(x)²        "true objective", used to compare all solvers
```
The `μ Σ L²` term is a smooth proxy for peak minimization (it is lowest when load is flat). The metrics table must also report the true peak `max_τ L_τ`.

### 4.5 Penalty weights
`λ_A` and `λ_O` must be large enough that violating a constraint never pays. Dropping EV `e`'s assignment saves at most its linear cost plus its marginal peak term, so define
```
Λ = max_v [ c_v  +  μ · d_e · p_e · ( p_e + 2 · Σ_{e'≠e} p_{e'} ) ]
```
Use `λ_A = λ_O = 1.1 · Λ` as the default. A λ that is too large flattens the QAOA landscape, so also run a **λ sweep** over `{0.5, 1.0, 1.5, 2, 3} × Λ` and plot feasibility probability and optimal-state probability against λ. Document this argument in the README.

### 4.6 Expansion into QUBO coefficients
Use `x² = x`. Store `H(x) = const + Σ_u Q_uu x_u + Σ_{u<v} Q_uv x_u x_v`.

```
E3 expands:  (Σ x − 1)² = − Σ x_v + 2 Σ_{u<v} x_u x_v + 1
             → Q_vv += −λ_A ;  Q_uv += 2 λ_A  (u,v in the same EV) ;  const += λ_A per EV

Peak term:   L_τ² = Σ_{u∈A_τ} p_u² x_u + 2 Σ_{u<v ∈ A_τ} p_u p_v x_u x_v     (A_τ = variables active at τ)
             → Q_uu += μ · p_u² · d_{e(u)}
             → Q_uv += 2 μ p_u p_v · |I(u) ∩ I(v)|   for every pair u<v (include same-EV pairs for exactness)

E4:          Q_uv += λ_O for (u,v) ∈ C

Linear:      Q_vv += c_v
```

### 4.7 Ising mapping
Substitute `x_i = (1 − Z_i)/2` (this is what `QuadraticProgram.to_ising()` does). The result is a diagonal Hamiltonian `H_C` (sum of `Z` and `ZZ` terms) plus a constant offset. Keep the offset and the sign convention explicit in the code and README.

---

## 5. Classical baselines (`classical.py`)

All solutions are scored with the same `evaluate.py` functions (§7).

1. **Brute force (ground truth).** Enumerate all `2^n` bitstrings (4096 for n = 12). Return the minimum of `H` and the best feasible `J`. Report whether the ground state is unique.
2. **FCFS.** Order EVs by arrival then index. Each takes the earliest feasible `(t)` on the lowest-index station with no bay conflict.
3. **Greedy-cost.** Order EVs by deadline slack. Each takes the conflict-free `v` with the smallest `c_v`. This ignores the peak term, which is the point of the comparison.
4. *(Nice to have)* **ILP with PuLP/CBC.** Same variables and hard constraints (E3, E4 as linear constraints); objective = `Σ c_v x_v + μ' · L_max` with `L_max` an auxiliary variable linearizing the maximum load. Score its schedule with `J`. State clearly that it optimizes a slightly different peak proxy.
5. *(Nice to have)* **Simulated annealing on the same QUBO**, to show the QUBO is good independent of QAOA.

---

## 6. QAOA (`qaoa.py`)

**Primary path: our own QAOA loop** (fewer dependency and version problems):

- Ansatz: `QAOAAnsatz(cost_operator=H_C_scaled, reps=p)` with the default X mixer and `|+⟩^n` initial state.
- **Normalize** the cost operator before optimization: divide `H_C` by the largest absolute Pauli coefficient (record the scale factor). Always report objective values in the original, unscaled units.
- Expectation value: exact, via `StatevectorEstimator`.
- Classical optimizer: `scipy.optimize.minimize(method="COBYLA")`, `maxiter ≈ 300`.
- Depth sweep `p ∈ {1, 2, 3}`. For each `p`: 10 random restarts (seeded `numpy.random.default_rng`), plus one warm start from the best `p−1` parameters. Keep the best by expectation value. Log `⟨H⟩` per iteration for the convergence plot.
- **Readout: exact probabilities** from the bound circuit's statevector (`Statevector(...).probabilities_dict()`), not small-shot sampling. With 4096 basis states, a few thousand shots can miss much of the distribution. Also report one shot-based result (8192 shots) for realism.

*Optional cross-check:* `qiskit_algorithms.QAOA`, only if the installed `qiskit-algorithms` supports V2 primitives (check the installed version). Do not make it the primary path.

**Bit ordering warning:** Qiskit bitstrings are little-endian (qubit 0 is the rightmost character). Decoding must map qubit `i` to variable `i`. This has its own test (§10).

**Noisy run (nice to have):** build an Aer `NoiseModel` (depolarizing error about 0.1% on 1-qubit gates and 1% on 2-qubit gates), transpile to basis `{rz, sx, x, cx}` at a chosen optimization level, and report transpiled depth and two-qubit gate count for `p = 1, 2, 3`. Run `p = 1` with 8192 shots using the noiseless-optimal parameters (state this) and report `P(optimal)` and `P(feasible)` under noise.

---

## 7. Metrics (`evaluate.py`)

One function scores any bitstring. Output one table (CSV + markdown) with a row per method (FCFS, greedy-cost, brute force, QAOA p=1/2/3, optional ILP/SA):

| Metric | Definition |
|---|---|
| Feasible | Passes E3 and E4 exactly |
| `J` | True objective (§4.4) |
| Energy cost | `Σ p_e Σ π_τ` over chosen assignments |
| Travel cost | `Σ κ · dist` |
| Total wait / max wait (slots) | `Σ (t − a_e)` and `max (t − a_e)` |
| Peak load | `max_τ L_τ` (kW) |
| Load variance | Variance of `L_τ` across slots |
| Bay utilization | Busy slot fraction per station |
| Gap to optimal | `(J − J*) / J*` |
| Runtime | Wall-clock seconds |

For QAOA also report: `P(optimal bitstring)`, `P(feasible)`, `⟨H⟩` at convergence, `J` of the most probable feasible bitstring, and the best `J` among 1024 sampled bitstrings.

---

## 8. Environment and Qiskit rules

- Python 3.10+ in a virtualenv. Pin exact versions in `requirements.txt` after installing (`pip freeze` for the packages below).
- Packages: `qiskit`, `qiskit-optimization`, `qiskit-aer`, `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest`. Optional: `pulp`, `qiskit-algorithms`, `pylatexenc` (for circuit drawings).
- **Use V2 primitives** (`StatevectorEstimator`, `StatevectorSampler` from `qiskit.primitives`).
- **Do not use:** `qiskit.execute`, `BasicAer`, `qiskit.algorithms` (the old in-tree module), `qiskit.opflow`, `PauliSumOp`, `QuantumInstance`, V1 `Sampler`/`Estimator`. These are removed or deprecated.
- **Smoke test before real work:** build a 2-variable `QuadraticProgram`, call `to_ising()`, build a `QAOAAnsatz`, evaluate with `StatevectorEstimator`, and decode a bitstring. If any step fails, fix the environment first.

---

## 9. File layout

```
ev-qaoa/
  SPEC.md
  README.md
  requirements.txt
  src/
    instance.py     fixed + random instance, JSON I/O
    preprocess.py   derive E_e, d_e, p_e; prune variables; index map
    qubo.py         c_v, Q coefficients, QuadraticProgram, to_ising, evaluate H and J directly
    classical.py    brute force, FCFS, greedy-cost, optional ILP/SA
    qaoa.py         QAOA loop, p sweep, λ sweep, noisy Aer run
    evaluate.py     feasibility check, metrics, comparison table
    viz.py          all plots
    run_all.py      runs everything and writes results/
  tests/
    test_qubo.py
  results/          cached JSON, CSV, PNG (committed)
```

---

## 10. Gates and tests (`tests/test_qubo.py`)

Run `pytest` at every gate. Failures block the next stage.

| Test | Passes when |
|---|---|
| `test_instance` | Derived values match §3.1 (`E, d, p`) and pruned `n = 12` |
| `test_matrix_equals_formula` | `const + Σ Q_uu x_u + Σ Q_uv x_u x_v` equals the direct formula for `H(x)` (§4.4) on 1000 random bitstrings (tolerance 1e-9) |
| `test_ising_equals_qubo` | The diagonal of the `to_ising()` operator plus offset equals `H(x)` for all `2^n` bitstrings |
| `test_bit_order` | A known bitstring decodes to the expected `(e,s,t)` assignments after Qiskit's little-endian reversal |
| `test_brute_force_feasible` | With `λ = 1.1 Λ`, the brute-force minimizer of `H` is feasible and `min H == min J over feasible x` |
| `test_instance_is_interesting` | Greedy-cost's `J` is strictly greater than the optimal `J`. If not, **stop and ask me**; do not change the instance on your own |
| `test_qubit_budget` | `n ≤ 14` |

QAOA-stage gate: `P(optimal)` should not decrease as `p` increases (allowing small optimizer noise). If it does, report it honestly rather than tuning until it looks good.

---

## 11. Deliverables

**Plots** (`viz.py`, saved to `results/`):
1. Gantt chart: stations on the y-axis, slots on the x-axis, colored by EV. Three panels: greedy-cost, QAOA best, brute-force optimum.
2. Grid load per slot, bars for the same three solvers.
3. QAOA convergence (`⟨H⟩` vs. iteration) for each `p`.
4. `P(optimal)` and `P(feasible)` vs. `p`.
5. λ-sweep plot.
6. Top-10 bitstring probability histogram, colored by feasibility.
7. QAOA circuit diagram at `p = 1`, plus transpiled depth table.

**README must contain:** problem and track; pipeline diagram; the full derivation from §4 including the λ argument; circuit diagram; convergence and comparison plots; the metrics table; honest limitations (NISQ depth, qubit scaling, QAOA not beating classical at n = 12); `requirements.txt` and run instructions (`python -m src.run_all`); a statement that the data is synthetic; acknowledgment of any borrowed ideas or third-party code.

---

## 12. Out of scope

No real EV or personal data, no hardware runs, no custom cryptography or unrelated features, no claims of quantum advantage, and no scaling beyond 14 qubits (describe scaling in the README only).

---

## Appendix: questions judges are likely to ask (for me, not the agent)

1. **Why start-slot variables?** They fix the duration from the battery's energy need, so capacity is a real constraint and big batteries occupy bays longer.
2. **Why quadratic penalties?** They keep the problem QUBO (quadratic, binary) so it maps to an Ising Hamiltonian with only `Z` and `ZZ` terms.
3. **How did you choose λ?** The bound `Λ` guarantees a violation never pays; the sweep shows that larger λ flattens the landscape.
4. **Why the sum of squared load for peak?** It is smooth and quadratic, so it fits a QUBO; the true max load is reported separately.
5. **Why is greedy worse?** It optimizes each EV's own cost and cannot see the coupling through grid load and bay conflicts.
6. **Did QAOA beat classical?** At 12 qubits brute force is exact and instant. We report where QAOA matches the optimum and how its probability changes with depth.
7. **What happens on real hardware?** Depth grows with the number of ZZ terms; see the transpiled depth and the noisy-run results.
8. **How does it scale?** Qubits grow as EVs × stations × start slots, minus pruning; beyond roughly 20 variables, decomposition or hybrid clustering is needed.