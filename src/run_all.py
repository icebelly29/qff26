"""
End-to-end reproducible execution pipeline for EV Charging Scheduling with QUBO & QAOA.
Regenerates all artifacts, metrics, and visualization deliverables per SPEC §9 and §11.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src.instance import get_fixed_instance, save_instance_to_json
from src.preprocess import prune_variables
from src.qubo import compute_linear_costs, compute_penalty_bound_lambda, brute_force_solve
from src.classical import solve_fcfs, solve_greedy_cost
from src.qaoa import (
    run_depth_sweep,
    run_lambda_sweep,
    save_results,
    save_depth_sweep_results,
    save_lambda_sweep_results,
)
from src.evaluate import generate_comparison_table
from src.viz import generate_all_visualizations


def run_pipeline() -> None:
    start_time = time.time()
    print("=" * 70)
    print("STARTING EV CHARGING QUBO & QAOA REPRODUCIBLE PIPELINE")
    print("=" * 70)

    # 1. Instance generation and serialization
    print("\n[Step 1/6] Loading problem instance and deriving requirements...")
    inst = get_fixed_instance()
    save_instance_to_json(inst, Path("results/instance.json"))
    vars_list = prune_variables(inst)
    print(f"Loaded instance: T={inst.num_slots} slots, {len(inst.stations)} stations, {len(inst.evs)} EVs.")
    print(f"Pruned search space: n = {len(vars_list)} binary variables / qubits (Budget <= 14).")

    # 2. Linear costs and penalty bounds
    print("\n[Step 2/6] Computing QUBO coefficients and penalty bound Lambda...")
    cv = compute_linear_costs(inst, vars_list)
    cap_lambda, lam = compute_penalty_bound_lambda(inst, vars_list, cv)
    print(f"Calculated penalty bound: Lambda = {cap_lambda:.6f}")
    print(f"Selected penalty weight: lambda = 1.1 * Lambda = {lam:.6f}")

    # 3. Ground truth brute-force & classical baselines
    print("\n[Step 3/6] Running Classical Baselines (Brute Force, FCFS, Greedy-Cost)...")
    bf_res = brute_force_solve(inst, vars_list, lam, lam)
    fcfs_res = solve_fcfs(inst, vars_list)
    greedy_res = solve_greedy_cost(inst, vars_list)

    print(f"Brute Force Optimal: J* = {bf_res['best_J']:.6f}, Bitstring = {bf_res['best_J_bitstring']}")
    print(f"FCFS Schedule:       J  = {fcfs_res['true_J']:.6f}, Gap = {fcfs_res['gap_to_optimal']*100:.2f}%")
    print(f"Greedy-Cost:         J  = {greedy_res['true_J']:.6f}, Gap = {greedy_res['gap_to_optimal']*100:.2f}%")

    # 4. QAOA Sweeps
    print("\n[Step 4/6] Executing QAOA Optimization (Depths p=1,2,3 & Lambda Sweep)...")
    depth_sweep = run_depth_sweep(inst, vars_list, lam, depths=(1, 2, 3), num_restarts=10, seed=42)
    save_depth_sweep_results(depth_sweep)
    save_results(depth_sweep["p_1"])

    lambda_sweep = run_lambda_sweep(
        inst,
        vars_list,
        lambda_ratios=(0.5, 1.0, 1.5, 2.0, 3.0),
        p=1,
        num_restarts=10,
        seed=42,
    )
    save_lambda_sweep_results(lambda_sweep)

    # 5. Metrics table generation
    print("\n[Step 5/6] Generating comprehensive evaluation metrics table...")
    csv_str, md_str = generate_comparison_table(
        inst,
        vars_list,
        lam,
        depth_sweep,
        out_csv_path=Path("results/metrics_table.csv"),
        out_md_path=Path("results/metrics_table.md"),
    )
    print("Metrics Table (Markdown):\n")
    print(md_str)

    # 6. Visualization deliverables
    print("\n[Step 6/6] Generating all 7 SPEC §11 visual deliverables & transpilation analysis...")
    generate_all_visualizations()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("All results, tables, and plots saved to results/")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()

