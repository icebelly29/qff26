"""
Visualization module for EV Charging Scheduling with QUBO & QAOA.
Generates all 7 required deliverables per SPEC §11.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator

from src.instance import ProblemInstance, get_fixed_instance
from src.preprocess import prune_variables, get_ev_power_and_duration_maps
from src.qubo import (
    compute_linear_costs,
    compute_penalty_bound_lambda,
    build_quadratic_program,
    qubo_to_ising,
    brute_force_solve,
)
from src.classical import solve_greedy_cost
from src.evaluate import score_bitstring


RESULTS_DIR = Path("results")


def decode_active_assignments(
    variables: List[Tuple[int, int, int]],
    bitstring: str,
) -> List[Tuple[int, int, int]]:
    """Decode bitstring (little-endian) into list of active (e, s, t) assignments."""
    rev = bitstring[::-1]
    active = []
    for i, char in enumerate(rev):
        if char == "1" and i < len(variables):
            active.append(variables[i])
    return active


def plot_gantt_chart(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    schedules: Dict[str, str],
    out_path: Path = RESULTS_DIR / "gantt_chart.png",
) -> None:
    """
    Plot Gantt chart comparing schedules across stations and time slots (SPEC §11 Plot 1).
    Three panels: Greedy-cost, QAOA best, Brute-force optimum.
    """
    fig, axes = plt.subplots(len(schedules), 1, figsize=(10, 3.2 * len(schedules)), sharex=True)
    if len(schedules) == 1:
        axes = [axes]

    ev_colors = {
        0: "#1f77b4",  # Blue
        1: "#ff7f0e",  # Orange
        2: "#2ca02c",  # Green
    }

    num_slots = instance.num_slots
    num_stations = len(instance.stations)
    powers, durations = get_ev_power_and_duration_maps(instance)

    for ax, (title, bitstring) in zip(axes, schedules.items()):
        assignments = decode_active_assignments(variables, bitstring)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Station", fontsize=11)
        ax.set_yticks(range(num_stations))
        ax.set_yticklabels([f"Station S{s} (1 bay)" for s in range(num_stations)])
        ax.set_ylim(-0.5, num_stations - 0.5)
        ax.set_xlim(-0.2, num_slots + 0.2)
        ax.set_xticks(range(num_slots))
        ax.set_xticklabels([f"Slot {t}\n({instance.prices[t]:.2f} $/kWh)" for t in range(num_slots)])
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Draw assignments
        for e, s, t in assignments:
            d = durations[e]
            p_e = powers[e]
            color = ev_colors.get(e, "#7f7f7f")
            rect = plt.Rectangle(
                (t, s - 0.3),
                d,
                0.6,
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
                alpha=0.85,
            )
            ax.add_patch(rect)
            ax.text(
                t + d / 2,
                s,
                f"EV{e} ({d}h, {p_e:.1f}kW)",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )

    legend_patches = [
        mpatches.Patch(color=ev_colors[e], label=f"EV{e} ({instance.evs[e].energy_needed_kwh:.0f} kWh)")
        for e in range(len(instance.evs))
    ]
    axes[0].legend(handles=legend_patches, loc="upper right", framealpha=0.9)
    plt.xlabel("Time Slots & Electricity Prices", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_grid_load(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    schedules: Dict[str, str],
    out_path: Path = RESULTS_DIR / "grid_load.png",
) -> None:
    """
    Plot grid load per time slot comparing solvers (SPEC §11 Plot 2).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    num_slots = instance.num_slots
    x = np.arange(num_slots)
    width = 0.25

    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    offsets = [-width, 0, width]
    powers, durations = get_ev_power_and_duration_maps(instance)

    for idx, (title, bitstring) in enumerate(schedules.items()):
        metrics = score_bitstring(instance, variables, bitstring)
        # Compute slot-by-slot load
        load_per_slot = np.zeros(num_slots)
        assignments = decode_active_assignments(variables, bitstring)
        for e, s, t in assignments:
            d = durations[e]
            p_e = powers[e]
            for slot in range(t, t + d):
                if slot < num_slots:
                    load_per_slot[slot] += p_e

        ax.bar(
            x + offsets[idx],
            load_per_slot,
            width,
            label=f"{title} (Peak: {metrics['peak_load_kw']:.1f} kW, Var: {metrics['load_variance']:.1f})",
            color=colors[idx % len(colors)],
            alpha=0.85,
            edgecolor="black",
        )

    ax.set_xlabel("Time Slot", fontsize=11)
    ax.set_ylabel("Total Grid Charging Load (kW)", fontsize=11)
    ax.set_title("Grid Load Distribution Across Slots by Solver", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Slot {t}\n({instance.prices[t]:.2f} $/kWh)" for t in range(num_slots)])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(framealpha=0.9, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_qaoa_convergence(
    depth_sweep_path: Path = RESULTS_DIR / "qaoa_depth_sweep.json",
    out_path: Path = RESULTS_DIR / "qaoa_convergence.png",
) -> None:
    """
    Plot QAOA expectation value <H> vs COBYLA evaluation iteration for p in {1, 2, 3} (SPEC §11 Plot 3).
    """
    if not depth_sweep_path.exists():
        print(f"Warning: {depth_sweep_path} not found. Skipping convergence plot.")
        return

    with open(depth_sweep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"p_1": "#1f77b4", "p_2": "#ff7f0e", "p_3": "#2ca02c"}
    labels = {"p_1": "QAOA p = 1", "p_2": "QAOA p = 2", "p_3": "QAOA p = 3 (Warm-Started)"}

    for key in ["p_1", "p_2", "p_3"]:
        if key in data and "best_convergence_history" in data[key]:
            hist = data[key]["best_convergence_history"]
            final_h = data[key]["expected_H"]
            ax.plot(
                range(1, len(hist) + 1),
                hist,
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=colors[key],
                label=f"{labels[key]} (Final $\\langle H \\rangle = {final_h:.2f}$)",
            )

    ax.set_xlabel("COBYLA Evaluation Iteration", fontsize=11)
    ax.set_ylabel("Expected Cost $\\langle H \\rangle$ (Original Units)", fontsize=11)
    ax.set_title("QAOA Convergence Trajectory Across Circuit Depths", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(framealpha=0.9, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_p_sweep_probabilities(
    depth_sweep_path: Path = RESULTS_DIR / "qaoa_depth_sweep.json",
    out_path: Path = RESULTS_DIR / "p_sweep_probabilities.png",
) -> None:
    """
    Plot P(optimal) and P(feasible) vs circuit depth p in {1, 2, 3} (SPEC §11 Plot 4).
    """
    if not depth_sweep_path.exists():
        print(f"Warning: {depth_sweep_path} not found. Skipping p-sweep plot.")
        return

    with open(depth_sweep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    p_vals = [1, 2, 3]
    p_opt = [data[f"p_{p}"]["P_optimal"] * 100 for p in p_vals]
    p_feas = [data[f"p_{p}"]["P_feasible"] * 100 for p in p_vals]

    uniform_opt = (1 / 4096) * 100
    uniform_feas = (16 / 4096) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # P(optimal)
    ax1.plot(p_vals, p_opt, marker="s", markersize=8, color="#1f77b4", linewidth=2, label="QAOA Statevector")
    ax1.axhline(uniform_opt, color="gray", linestyle="--", label=f"Uniform Baseline ({uniform_opt:.3f}%)")
    ax1.set_xlabel("QAOA Depth $p$", fontsize=11)
    ax1.set_ylabel("P(optimal bitstring) (%)", fontsize=11)
    ax1.set_title("Success Probability $P(\\text{opt})$ vs. Depth", fontsize=12, fontweight="bold")
    ax1.set_xticks(p_vals)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(framealpha=0.9)

    # Annotate gains
    for p, val in zip(p_vals, p_opt):
        gain = val / uniform_opt
        ax1.annotate(f"{val:.3f}%\n({gain:.1f}x)", (p, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    # P(feasible)
    ax2.plot(p_vals, p_feas, marker="o", markersize=8, color="#2ca02c", linewidth=2, label="QAOA Statevector")
    ax2.axhline(uniform_feas, color="gray", linestyle="--", label=f"Uniform Baseline ({uniform_feas:.2f}%)")
    ax2.set_xlabel("QAOA Depth $p$", fontsize=11)
    ax2.set_ylabel("P(feasible schedule) (%)", fontsize=11)
    ax2.set_title("Feasibility Probability $P(\\text{feas})$ vs. Depth", fontsize=12, fontweight="bold")
    ax2.set_xticks(p_vals)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(framealpha=0.9)

    for p, val in zip(p_vals, p_feas):
        gain = val / uniform_feas
        ax2.annotate(f"{val:.2f}%\n({gain:.1f}x)", (p, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_lambda_sweep(
    lambda_sweep_path: Path = RESULTS_DIR / "qaoa_lambda_sweep.json",
    out_path: Path = RESULTS_DIR / "lambda_sweep.png",
) -> None:
    """
    Plot penalty weight sweep curves (SPEC §11 Plot 5).
    """
    if not lambda_sweep_path.exists():
        print(f"Warning: {lambda_sweep_path} not found. Skipping lambda-sweep plot.")
        return

    with open(lambda_sweep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    ratios = [r["ratio"] for r in results]
    exp_h = [r["expected_H"] for r in results]
    p_feas = [r["P_feasible"] * 100 for r in results]
    p_opt = [r["P_optimal"] * 100 for r in results]
    ranks = [r["most_probable_feasible_rank"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Expectation value & Feasibility
    color1 = "#1f77b4"
    ax1.set_xlabel("Penalty Multiplier $\\lambda / \\Lambda$", fontsize=11)
    ax1.set_ylabel("Expected Cost $\\langle H \\rangle$", color=color1, fontsize=11)
    ax1.plot(ratios, exp_h, marker="o", color=color1, linewidth=2, label="$\\langle H \\rangle$")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_twin = ax1.twinx()
    color2 = "#2ca02c"
    ax1_twin.set_ylabel("P(feasible) (%)", color=color2, fontsize=11)
    ax1_twin.plot(ratios, p_feas, marker="s", color=color2, linewidth=2, linestyle="--", label="P(feasible)")
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("Energy & Feasibility vs. Penalty Multiplier", fontsize=12, fontweight="bold")

    # Rank and P(optimal)
    ax2.plot(ratios, p_opt, marker="^", color="#9467bd", linewidth=2, label="P(optimal) (%)")
    ax2.set_xlabel("Penalty Multiplier $\\lambda / \\Lambda$", fontsize=11)
    ax2.set_ylabel("P(optimal) (%)", color="#9467bd", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(ratios, ranks, marker="d", color="#d62728", linewidth=2, linestyle=":", label="Most Probable Feas Rank")
    ax2_twin.set_ylabel("Rank of Most Probable Feasible Schedule", color="#d62728", fontsize=11)
    ax2_twin.set_yticks([1, 4, 8, 12, 16])
    ax2_twin.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_title("Optimality & Feasible Mode Selection", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_top_bitstrings_hist(
    qaoa_p1_path: Path = RESULTS_DIR / "qaoa_p1.json",
    out_path: Path = RESULTS_DIR / "top_bitstrings_hist.png",
) -> None:
    """
    Plot top-10 bitstrings probability histogram colored by feasibility (SPEC §11 Plot 6).
    """
    if not qaoa_p1_path.exists():
        print(f"Warning: {qaoa_p1_path} not found. Skipping top-10 bitstrings plot.")
        return

    with open(qaoa_p1_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_10 = data.get("top_10_bitstrings", [])
    if not top_10:
        return

    bitstrings = [item["bitstring"] for item in top_10]
    probs = [item["probability"] * 100 for item in top_10]
    feasibilities = [item["feasible"] for item in top_10]
    colors = ["#2ca02c" if f else "#d62728" for f in feasibilities]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(bitstrings))
    bars = ax.barh(y_pos, probs, color=colors, edgecolor="black", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bitstrings, fontfamily="monospace", fontsize=10)
    ax.invert_yaxis()  # Top probability at top
    ax.set_xlabel("Exact Statevector Readout Probability (%)", fontsize=11)
    ax.set_title("Top 10 Readout Bitstrings in QAOA ($p = 1$)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Annotate labels
    for bar, item in zip(bars, top_10):
        w = bar.get_width()
        info = f"{w:.3f}%"
        if item["feasible"]:
            info += f" [Feas Rank {item['feas_rank']}, J={item['J']:.2f}]"
        else:
            info += " [Infeasible]"
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, info, va="center", fontsize=9)

    feas_patch = mpatches.Patch(color="#2ca02c", label="Feasible Schedule (Passes E3 & E4)")
    infeas_patch = mpatches.Patch(color="#d62728", label="Infeasible Schedule (Penalty Violated)")
    ax.legend(handles=[feas_patch, infeas_patch], loc="lower right", framealpha=0.9)

    ax.set_xlim(0, max(probs) * 1.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_circuit_and_transpilation_report(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    lambda_val: float,
    out_circuit_path: Path = RESULTS_DIR / "qaoa_circuit.png",
    out_json_path: Path = RESULTS_DIR / "circuit_transpilation.json",
) -> Dict[str, Any]:
    """
    Generate QAOA circuit diagram at p = 1 and transpiled metrics table (SPEC §11 Plot 7).
    Also performs noisy Aer simulation per SPEC §6.
    """
    qp = build_quadratic_program(instance, variables, lambda_a=lambda_val, lambda_o=lambda_val)
    H_C, offset = qubo_to_ising(qp)

    # 1. Render QAOA circuit diagram at p = 1
    ansatz_p1 = QAOAAnsatz(cost_operator=H_C, reps=1)
    decomp_p1 = ansatz_p1.decompose()
    fig = decomp_p1.draw(output="mpl", style="clifford")
    fig.savefig(out_circuit_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_circuit_path}")

    # 2. Transpilation analysis for p in {1, 2, 3}
    basis = ["rz", "sx", "x", "cx"]
    transpiled_stats = {}

    for p in [1, 2, 3]:
        ansatz = QAOAAnsatz(cost_operator=H_C, reps=p)
        decomp = ansatz.decompose().decompose()
        t_qc = transpile(decomp, basis_gates=basis, optimization_level=2)
        ops = dict(t_qc.count_ops())
        transpiled_stats[f"p_{p}"] = {
            "depth": t_qc.depth(),
            "cx_count": ops.get("cx", 0),
            "total_ops": sum(ops.values()),
            "ops": ops,
        }

    # 3. Noisy Aer simulation for p = 1 using noiseless optimal parameters
    p1_json_path = RESULTS_DIR / "qaoa_p1.json"
    noisy_results = {}
    if p1_json_path.exists():
        with open(p1_json_path, "r", encoding="utf-8") as f:
            p1_cached = json.load(f)
        opt_params = p1_cached["optimal_params"]

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["rz", "sx", "x"])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])

        bound_circ = ansatz_p1.assign_parameters(opt_params)
        bound_circ.measure_all()
        t_bound = transpile(bound_circ, basis_gates=basis, optimization_level=2)

        sim = AerSimulator(noise_model=noise_model, seed_simulator=42)
        shots = 8192
        job_result = sim.run(t_bound, shots=shots).result()
        counts = job_result.get_counts()

        opt_bitstring = "001010000001"
        n_opt = counts.get(opt_bitstring, 0)
        n_feas = sum(cnt for bs, cnt in counts.items() if score_bitstring(instance, variables, bs)["feasible"])

        noisy_results = {
            "shots": shots,
            "noise_model": "depolarizing (0.1% 1-qubit, 1.0% 2-qubit)",
            "P_optimal_noisy": n_opt / shots,
            "P_feasible_noisy": n_feas / shots,
            "optimal_counts": n_opt,
            "feasible_counts": n_feas,
        }

    report = {
        "transpiled_depth_and_gates": transpiled_stats,
        "noisy_simulation": noisy_results,
    }

    out_json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    js_path = out_json_path.with_suffix(".js")
    js_path.write_text(f"window.CIRCUIT_TRANSPILATION = {json.dumps(report, indent=2, sort_keys=True)};\n", encoding="utf-8")
    print(f"Saved {out_json_path} and {js_path}")
    return report


def generate_all_visualizations() -> None:
    """Generate all deliverables required in SPEC §11."""
    inst = get_fixed_instance()
    vars_list = prune_variables(inst)
    cv = compute_linear_costs(inst, vars_list)
    _, lam = compute_penalty_bound_lambda(inst, vars_list, cv)

    # 1. Classical solutions
    greedy_sol = solve_greedy_cost(inst, vars_list)
    bf_sol = brute_force_solve(inst, vars_list, lam, lam)

    # QAOA best bitstring from cached p1 or brute force
    p1_path = RESULTS_DIR / "qaoa_p1.json"
    qaoa_best_bs = bf_sol["best_H_bitstring"]
    if p1_path.exists():
        with open(p1_path, "r", encoding="utf-8") as f:
            p1_data = json.load(f)
        qaoa_best_bs = p1_data["most_probable_feasible"]["bitstring"]

    schedules = {
        "Greedy-Cost": greedy_sol["bitstring"],
        "QAOA Best Feasible": qaoa_best_bs,
        "Brute-Force Optimum": bf_sol["best_H_bitstring"],
    }

    print("\n--- Generating Visualization Plots ---")
    plot_gantt_chart(inst, vars_list, schedules)
    plot_grid_load(inst, vars_list, schedules)
    plot_qaoa_convergence()
    plot_p_sweep_probabilities()
    plot_lambda_sweep()
    plot_top_bitstrings_hist()
    generate_circuit_and_transpilation_report(inst, vars_list, lam)
    print("All visualizations generated successfully.")


if __name__ == "__main__":
    generate_all_visualizations()
