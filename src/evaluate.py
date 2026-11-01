"""Solution evaluation, schedule scoring, and comparison table generation (SPEC §7)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.instance import ProblemInstance, get_fixed_instance
from src.preprocess import get_ev_power_and_duration_maps, prune_variables
from src.qubo import compute_linear_costs, find_conflicting_pairs


def score_bitstring(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    bitstring: str,
    optimal_j: Optional[float] = None,
    runtime_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Score any bitstring according to SPEC §7 metrics.

    Uses little-endian mapping: bitstring[-(i + 1)] corresponds to qubit i (variable i).
    """
    n = len(variables)
    powers, durations = get_ev_power_and_duration_maps(instance)
    conflicts = find_conflicting_pairs(variables, durations)
    c_v_list = compute_linear_costs(instance, variables)

    # Decode bits: bits[i] is qubit i
    bits = [int(bitstring[-(i + 1)]) for i in range(n)]
    active_indices = [i for i, b in enumerate(bits) if b == 1]
    active_vars = [variables[i] for i in active_indices]

    # 1. Feasibility check (E3 and E4)
    # E3: Exactly one assignment per EV
    e3_pass = True
    ev_counts: Dict[int, int] = {ev.id: 0 for ev in instance.evs}
    for e, _, _ in active_vars:
        ev_counts[e] = ev_counts.get(e, 0) + 1

    for ev in instance.evs:
        if ev_counts.get(ev.id, 0) != 1:
            e3_pass = False
            break

    # E4: Overlap conflicts
    e4_pass = True
    for u in active_indices:
        for v in active_indices:
            if u < v and (u, v) in conflicts:
                e4_pass = False
                break
        if not e4_pass:
            break

    is_feasible = e3_pass and e4_pass

    # 2. Cost breakdown
    energy_cost = 0.0
    travel_cost = 0.0
    wait_cost = 0.0
    total_wait_slots = 0
    max_wait_slots = 0

    for i in active_indices:
        e, s, t = variables[i]
        ev = instance.evs[e]
        station = instance.stations[s]
        d_e = durations[e]
        p_e = powers[e]

        # Energy
        energy_cost += p_e * sum(instance.prices[tau] for tau in range(t, t + d_e))

        # Travel
        dist = math.sqrt((ev.x - station.x) ** 2 + (ev.y - station.y) ** 2)
        travel_cost += instance.kappa * dist

        # Wait
        wait_slots = t - ev.arrival_slot
        total_wait_slots += wait_slots
        max_wait_slots = max(max_wait_slots, wait_slots)
        wait_cost += instance.omega * wait_slots

    total_linear = energy_cost + travel_cost + wait_cost

    # 3. Grid load and peak
    loads = [0.0] * instance.num_slots
    for i in active_indices:
        e, _, t = variables[i]
        d_e = durations[e]
        p_e = powers[e]
        for tau in range(t, t + d_e):
            loads[tau] += p_e

    peak_load = max(loads) if loads else 0.0
    load_variance = float(np.var(loads)) if loads else 0.0
    peak_term = instance.mu * sum(l**2 for l in loads)

    # True objective J
    true_j = total_linear + peak_term

    # 4. Bay utilization per station
    # Busy slots / total slots per station
    station_busy_slots: Dict[int, Set[int]] = {s.id: set() for s in instance.stations}
    for i in active_indices:
        e, s, t = variables[i]
        d_e = durations[e]
        for tau in range(t, t + d_e):
            station_busy_slots[s].add(tau)

    bay_utilization = {
        f"S{s.id}_util": len(station_busy_slots[s.id]) / float(instance.num_slots)
        for s in instance.stations
    }
    avg_bay_util = sum(bay_utilization.values()) / float(len(instance.stations))

    # Gap to optimal
    gap_to_opt = None
    if optimal_j is not None and is_feasible and optimal_j > 0:
        gap_to_opt = (true_j - optimal_j) / optimal_j

    return {
        "bitstring": bitstring,
        "feasible": is_feasible,
        "J": true_j,
        "gap_to_opt": gap_to_opt,
        "energy_cost": energy_cost,
        "travel_cost": travel_cost,
        "wait_cost": wait_cost,
        "total_wait_slots": total_wait_slots,
        "max_wait_slots": max_wait_slots,
        "peak_load_kw": peak_load,
        "load_variance": load_variance,
        "loads": loads,
        "bay_utilization": bay_utilization,
        "avg_bay_utilization": avg_bay_util,
        "runtime_sec": runtime_sec,
        "assignments": [f"EV{e}@S{s}:t={t}" for e, s, t in active_vars],
    }


def generate_comparison_table(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    solver_results: Dict[str, Dict[str, Any]],
    output_dir: Path = Path("results"),
) -> pd.DataFrame:
    """
    Construct standardized comparison table across all methods (SPEC §7).

    Exports to CSV and Markdown.
    """
    opt_j = solver_results["Brute Force"]["J"]

    rows = []
    for method_name, res in solver_results.items():
        bitstring = res["bitstring"]
        runtime = res.get("runtime_sec")
        score = score_bitstring(
            instance, variables, bitstring, optimal_j=opt_j, runtime_sec=runtime
        )

        row = {
            "Method": method_name,
            "Feasible": score["feasible"],
            "J ($)": round(score["J"], 4),
            "Gap to Opt (%)": (
                f"{score['gap_to_opt'] * 100:.2f}%"
                if score["gap_to_opt"] is not None
                else "-"
            ),
            "Energy Cost ($)": round(score["energy_cost"], 4),
            "Travel Cost ($)": round(score["travel_cost"], 4),
            "Wait Cost ($)": round(score["wait_cost"], 4),
            "Total Wait (slots)": score["total_wait_slots"],
            "Max Wait (slots)": score["max_wait_slots"],
            "Peak Load (kW)": score["peak_load_kw"],
            "Load Var": round(score["load_variance"], 2),
            "Avg Bay Util": f"{score['avg_bay_utilization'] * 100:.1f}%",
            "Runtime (s)": (
                round(runtime, 4) if runtime is not None else "-"
            ),
            "Bitstring": bitstring,
        }

        # Extra metrics for QAOA methods
        if "P_optimal" in res:
            row["P(opt) (%)"] = f"{res['P_optimal'] * 100:.4f}%"
        if "P_feasible" in res:
            row["P(feas) (%)"] = f"{res['P_feasible'] * 100:.2f}%"

        rows.append(row)

    df = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_table.csv"
    md_path = output_dir / "metrics_table.md"

    df.to_csv(csv_path, index=False)
    md_content = df.to_markdown(index=False)
    md_path.write_text(md_content + "\n", encoding="utf-8")

    return df

