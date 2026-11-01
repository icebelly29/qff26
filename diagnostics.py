"""Diagnostics script for EV charging problem instance.

Prints:
1. Total number of feasible schedules out of 2^n.
2. J values for each feasible schedule, sorted.
3. FCFS and Greedy-cost baseline results with feasibility, J, and gap to optimal.
"""

from __future__ import annotations

from src.classical import get_all_feasible_schedules, solve_fcfs, solve_greedy_cost
from src.instance import get_fixed_instance
from src.preprocess import prune_variables
from src.qubo import compute_linear_costs


def run_diagnostics() -> None:
    """Execute complete schedule space diagnosis."""
    instance = get_fixed_instance()
    variables = prune_variables(instance)
    n = len(variables)
    total_states = 1 << n

    print("=" * 70)
    print("EV CHARGING SCHEDULING DIAGNOSTICS")
    print(f"Instance: T = {instance.num_slots} slots, {len(instance.evs)} EVs, {len(instance.stations)} stations")
    print(f"Pruned variables (qubits): n = {n} (total states = {total_states})")
    print("=" * 70)

    # 1. Feasible schedules
    feasible_schedules = get_all_feasible_schedules(instance, variables)
    num_feasible = len(feasible_schedules)
    print(f"\n[1] FEASIBLE SCHEDULES COUNT: {num_feasible} / {total_states} ({num_feasible / total_states * 100:.2f}%)")

    # 2. J for each feasible schedule, sorted
    opt_j = feasible_schedules[0]["J"]
    print(f"\n[2] ALL FEASIBLE SCHEDULES (SORTED BY OBJECTIVE J):")
    print("-" * 70)
    print(f"{'Rank':<5} {'J ($)':<12} {'Bit Int':<9} {'Bitstring (q11..q0)':<22} {'Assignments (EV, Station, Start)'}")
    print("-" * 70)
    for rank, item in enumerate(feasible_schedules, start=1):
        assign_str = ", ".join(f"EV{e}@S{s}:t={t}" for e, s, t in item["assigned_vars"])
        print(f"{rank:<5} {item['J']:<12.6f} {item['int_val']:<9} {item['bitstring']:<22} {assign_str}")

    # 3. Baselines: FCFS and Greedy-cost
    c_v_list = compute_linear_costs(instance, variables)
    fcfs_res = solve_fcfs(instance, variables)
    greedy_res = solve_greedy_cost(instance, variables, c_v_list)

    fcfs_gap = (fcfs_res["J"] - opt_j) / opt_j if fcfs_res["feasible"] else float("inf")
    greedy_gap = (greedy_res["J"] - opt_j) / opt_j if greedy_res["feasible"] else float("inf")

    print("\n[3] CLASSICAL BASELINE COMPARISONS:")
    print("-" * 70)
    print(f"{'Method':<14} {'Feasible':<10} {'J ($)':<12} {'Gap to Opt (%)':<16} {'Assignments'}")
    print("-" * 70)

    # Optimal
    opt_assign = ", ".join(f"EV{e}@S{s}:t={t}" for e, s, t in feasible_schedules[0]["assigned_vars"])
    print(f"{'Brute Force*':<14} {'True':<10} {opt_j:<12.6f} {'0.00%':<16} {opt_assign}")

    # FCFS
    fcfs_assign = ", ".join(f"EV{e}@S{s}:t={t}" for e, s, t in fcfs_res["assigned_vars"])
    print(f"{fcfs_res['method']:<14} {str(fcfs_res['feasible']):<10} {fcfs_res['J']:<12.6f} {fcfs_gap * 100:<15.2f}% {fcfs_assign}")

    # Greedy-cost
    greedy_assign = ", ".join(f"EV{e}@S{s}:t={t}" for e, s, t in greedy_res["assigned_vars"])
    print(f"{greedy_res['method']:<14} {str(greedy_res['feasible']):<10} {greedy_res['J']:<12.6f} {greedy_gap * 100:<15.2f}% {greedy_assign}")
    print("-" * 70)


if __name__ == "__main__":
    run_diagnostics()

