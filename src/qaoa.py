"""QAOA optimization loop (SPEC §6) using QAOAAnsatz, StatevectorEstimator, and scipy COBYLA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from src.classical import get_all_feasible_schedules
from src.instance import ProblemInstance, get_fixed_instance
from src.preprocess import prune_variables
from src.qubo import (
    build_quadratic_program,
    compute_linear_costs,
    compute_penalty_bound_lambda,
    qubo_to_ising,
)


def normalize_cost_operator(h_c: SparsePauliOp) -> Tuple[SparsePauliOp, float]:
    """
    Normalize cost operator by its largest absolute Pauli coefficient (SPEC §6).

    Returns (h_c_scaled, scale_factor).
    """
    scale = float(max(abs(c) for c in h_c.coeffs))
    h_c_scaled = h_c / scale
    return h_c_scaled, scale


def build_qaoa_ansatz(cost_operator: SparsePauliOp, reps: int = 1) -> QuantumCircuit:
    """
    Construct QAOAAnsatz and decompose into standard gates (rzz, rz, rx, h).

    Decomposing avoids expensive matrix exponentials inside simulation primitives.
    """
    raw_ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=reps)
    # Decompose PauliEvolution into concrete 2-qubit and 1-qubit gates
    return raw_ansatz.decompose().decompose()


def optimize_qaoa_for_depth(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    lambda_val: float,
    p: int = 1,
    num_restarts: int = 10,
    maxiter: int = 100,
    seed: int = 42,
    warm_start_params: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Execute QAOA optimization for a specific depth p (SPEC §6).

    Uses exact StatevectorEstimator and scipy COBYLA optimizer with 10 random restarts
    plus an optional warm-start restart from depth p - 1.
    """
    qp = build_quadratic_program(instance, variables, lambda_val, lambda_val)
    h_c, offset = qubo_to_ising(qp)

    h_c_scaled, scale = normalize_cost_operator(h_c)
    ansatz = build_qaoa_ansatz(h_c_scaled, reps=p)
    estimator = StatevectorEstimator()

    feasible_schedules = get_all_feasible_schedules(instance, variables)
    feas_map = {item["bitstring"]: item for item in feasible_schedules}
    opt_schedule = feasible_schedules[0]
    opt_bitstring = opt_schedule["bitstring"]

    rng = np.random.default_rng(seed=seed)
    best_ev = float("inf")
    best_params = None
    best_restart_type = ""
    best_history: List[float] = []
    restart_summaries: List[Dict[str, Any]] = []

    # 1. Warm start from p - 1 (if available)
    if warm_start_params is not None and p > 1:
        prev_p = p - 1
        beta_prev = warm_start_params[:prev_p]
        gamma_prev = warm_start_params[prev_p:]

        # Continuation heuristic: replicate the last layer's angles
        beta_warm = np.append(beta_prev, beta_prev[-1])
        gamma_warm = np.append(gamma_prev, gamma_prev[-1])
        warm_init = np.concatenate([beta_warm, gamma_warm])

        eval_history: List[float] = []

        def warm_cost_func(params: np.ndarray) -> float:
            pub = (ansatz, [h_c_scaled], [params])
            job = estimator.run([pub])
            ev_scaled = float(job.result()[0].data.evs[0])
            ev_original = ev_scaled * scale + offset
            eval_history.append(ev_original)
            return ev_scaled

        res_warm = minimize(
            warm_cost_func,
            warm_init,
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        final_h_warm = float(res_warm.fun) * scale + offset
        restart_summaries.append({
            "restart": "warm_start",
            "nfev": int(res_warm.nfev),
            "final_H": final_h_warm,
            "params": res_warm.x.tolist(),
        })

        best_ev = float(res_warm.fun)
        best_params = res_warm.x
        best_restart_type = "warm_start"
        best_history = eval_history

    # 2. Random restarts
    for restart in range(num_restarts):
        beta_rand = rng.uniform(0.0, np.pi, size=p)
        gamma_rand = rng.uniform(0.0, 2.0 * np.pi, size=p)
        init_params = np.concatenate([beta_rand, gamma_rand])

        eval_history = []

        def cost_func(params: np.ndarray) -> float:
            pub = (ansatz, [h_c_scaled], [params])
            job = estimator.run([pub])
            ev_scaled = float(job.result()[0].data.evs[0])
            ev_original = ev_scaled * scale + offset
            eval_history.append(ev_original)
            return ev_scaled

        res = minimize(
            cost_func,
            init_params,
            method="COBYLA",
            options={"maxiter": maxiter},
        )

        final_h = float(res.fun) * scale + offset
        restart_summaries.append({
            "restart": f"random_{restart + 1}",
            "nfev": int(res.nfev),
            "final_H": final_h,
            "params": res.x.tolist(),
        })

        if res.fun < best_ev:
            best_ev = float(res.fun)
            best_params = res.x
            best_restart_type = f"random_{restart + 1}"
            best_history = eval_history

    best_h = best_ev * scale + offset

    # Readout: Exact probabilities from statevector
    assigned_circuit = ansatz.assign_parameters(
        dict(zip(ansatz.parameters, best_params))
    )
    sv = Statevector(assigned_circuit)
    probs_dict = sv.probabilities_dict()

    # Shot-based sampling (8192 shots) for NISQ realism
    shot_counts = sv.sample_counts(shots=8192)
    shots_opt_count = int(shot_counts.get(opt_bitstring, 0))
    shots_feas_count = sum(int(shot_counts.get(bs, 0)) for bs in feas_map)
    p_optimal_shots = shots_opt_count / 8192.0
    p_feasible_shots = shots_feas_count / 8192.0

    # Exact probabilities
    p_optimal = float(probs_dict.get(opt_bitstring, 0.0))
    p_feasible = float(sum(probs_dict.get(bs, 0.0) for bs in feas_map))

    # Top-10 bitstrings by probability
    sorted_probs = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_10_list: List[Dict[str, Any]] = []
    for rank, (bs, prob) in enumerate(sorted_probs, start=1):
        is_feas = bs in feas_map
        if is_feas:
            j_val = float(feas_map[bs]["J"])
            feas_rank = next(
                i for i, item in enumerate(feasible_schedules, start=1)
                if item["bitstring"] == bs
            )
        else:
            j_val = None
            feas_rank = None

        top_10_list.append({
            "rank": rank,
            "bitstring": bs,
            "probability": float(prob),
            "feasible": is_feas,
            "J": j_val,
            "feas_rank": feas_rank,
        })

    # Most probable feasible bitstring
    feas_probs = [(bs, float(probs_dict.get(bs, 0.0))) for bs in feas_map]
    feas_probs.sort(key=lambda kv: kv[1], reverse=True)
    most_probable_feas_bs, mp_prob = feas_probs[0]
    mp_j = float(feas_map[most_probable_feas_bs]["J"])
    mp_rank = next(
        i for i, item in enumerate(feasible_schedules, start=1)
        if item["bitstring"] == most_probable_feas_bs
    )

    # Feasible distribution
    feasible_distribution = [
        {
            "feas_rank": rank,
            "bitstring": item["bitstring"],
            "probability": float(probs_dict.get(item["bitstring"], 0.0)),
            "J": float(item["J"]),
        }
        for rank, item in enumerate(feasible_schedules, start=1)
    ]

    # Uniform random baselines
    uniform_p_feas = len(feasible_schedules) / (1 << len(variables))  # 16 / 4096 = 0.3906%
    uniform_p_opt = 1.0 / (1 << len(variables))  # 1 / 4096 = 0.0244%

    return {
        "p": p,
        "scale_factor": scale,
        "offset": offset,
        "optimal_params": best_params.tolist(),
        "best_restart_type": best_restart_type,
        "expected_H": best_h,
        "P_optimal": p_optimal,
        "P_feasible": p_feasible,
        "shots_8192": {
            "P_optimal_shots": p_optimal_shots,
            "P_feasible_shots": p_feasible_shots,
            "shots_optimal_count": shots_opt_count,
            "shots_feasible_count": shots_feas_count,
        },
        "uniform_P_feasible": uniform_p_feas,
        "uniform_P_optimal": uniform_p_opt,
        "feasibility_enhancement": p_feasible / uniform_p_feas if uniform_p_feas > 0 else 0.0,
        "optimality_enhancement": p_optimal / uniform_p_opt if uniform_p_opt > 0 else 0.0,
        "top_10_bitstrings": top_10_list,
        "most_probable_feasible": {
            "bitstring": most_probable_feas_bs,
            "probability": mp_prob,
            "J": mp_j,
            "feas_rank": mp_rank,
        },
        "feasible_distribution": feasible_distribution,
        "restart_summaries": restart_summaries,
        "best_convergence_history": best_history,
    }


def optimize_qaoa_p1(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    lambda_val: float,
    num_restarts: int = 10,
    maxiter: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for p = 1 optimization."""
    return optimize_qaoa_for_depth(
        instance=instance,
        variables=variables,
        lambda_val=lambda_val,
        p=1,
        num_restarts=num_restarts,
        maxiter=maxiter,
        seed=seed,
        warm_start_params=None,
    )


def run_depth_sweep(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    lambda_val: float,
    depths: Tuple[int, ...] = (1, 2, 3),
    num_restarts: int = 10,
    maxiter: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute depth sweep p in {1, 2, 3} with warm starting (SPEC §6).

    Returns dictionary mapping depth -> depth results and summary metrics.
    """
    sweep_results: Dict[str, Any] = {}
    prev_params: Optional[np.ndarray] = None

    for p in depths:
        print(f"\n--- Running QAOA Depth p = {p} ---", flush=True)
        res_p = optimize_qaoa_for_depth(
            instance=instance,
            variables=variables,
            lambda_val=lambda_val,
            p=p,
            num_restarts=num_restarts,
            maxiter=maxiter,
            seed=seed + p * 10,
            warm_start_params=prev_params,
        )
        sweep_results[f"p_{p}"] = res_p
        prev_params = np.array(res_p["optimal_params"])
        print(
            f"p={p} Complete: <H>={res_p['expected_H']:.4f}, "
            f"P(opt)={res_p['P_optimal']*100:.4f}%, "
            f"P(feas)={res_p['P_feasible']*100:.4f}% "
            f"(Best restart: {res_p['best_restart_type']})",
            flush=True,
        )

    return sweep_results


def run_lambda_sweep(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    lambda_ratios: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
    p: int = 1,
    num_restarts: int = 10,
    maxiter: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute penalty weight sweep lambda in {0.5, 1.0, 1.5, 2.0, 3.0} * Lambda (SPEC §4.5).
    """
    cv = compute_linear_costs(instance, variables)
    cap_lambda, _ = compute_penalty_bound_lambda(instance, variables, cv)

    lambda_results: List[Dict[str, Any]] = []

    for ratio in lambda_ratios:
        lam = ratio * cap_lambda
        print(f"--- Running Lambda Sweep: ratio={ratio}x Lambda (lambda={lam:.4f}) ---", flush=True)
        res = optimize_qaoa_for_depth(
            instance=instance,
            variables=variables,
            lambda_val=lam,
            p=p,
            num_restarts=num_restarts,
            maxiter=maxiter,
            seed=seed,
            warm_start_params=None,
        )
        lambda_results.append({
            "ratio": ratio,
            "lambda": lam,
            "capital_lambda": cap_lambda,
            "expected_H": res["expected_H"],
            "P_optimal": res["P_optimal"],
            "P_feasible": res["P_feasible"],
            "most_probable_feasible_J": res["most_probable_feasible"]["J"],
            "most_probable_feasible_rank": res["most_probable_feasible"]["feas_rank"],
        })

    return {
        "capital_lambda": cap_lambda,
        "results": lambda_results,
    }


def save_results(
    results: Dict[str, Any],
    json_path: Path = Path("results/qaoa_p1.json"),
    js_path: Path = Path("results/qaoa_p1.js"),
) -> None:
    """Cache QAOA results for demo and report generation."""
    payload = json.dumps(results, indent=2, sort_keys=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(payload + "\n", encoding="utf-8")
    js_path.write_text(f"window.QAOA_P1 = {payload};\n", encoding="utf-8")


def save_depth_sweep_results(
    results: Dict[str, Any],
    json_path: Path = Path("results/qaoa_depth_sweep.json"),
    js_path: Path = Path("results/qaoa_depth_sweep.js"),
) -> None:
    """Save depth sweep results to JSON and JS."""
    payload = json.dumps(results, indent=2, sort_keys=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(payload + "\n", encoding="utf-8")
    js_path.write_text(f"window.QAOA_DEPTH_SWEEP = {payload};\n", encoding="utf-8")


def save_lambda_sweep_results(
    results: Dict[str, Any],
    json_path: Path = Path("results/qaoa_lambda_sweep.json"),
) -> None:
    """Save lambda sweep results to JSON."""
    payload = json.dumps(results, indent=2, sort_keys=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    inst = get_fixed_instance()
    vars_list = prune_variables(inst)
    cv = compute_linear_costs(inst, vars_list)
    _, lam = compute_penalty_bound_lambda(inst, vars_list, cv)

    print("=" * 65)
    print("EXECUTING QAOA DEPTH SWEEP (p = 1, 2, 3)")
    print("=" * 65)
    depth_sweep = run_depth_sweep(inst, vars_list, lam, depths=(1, 2, 3), num_restarts=10, seed=42)
    save_depth_sweep_results(depth_sweep)
    save_results(depth_sweep["p_1"])  # Update cached p=1 results

    print("\n" + "=" * 65)
    print("EXECUTING PENALTY WEIGHT SWEEP (0.5x to 3.0x Lambda)")
    print("=" * 65)
    lambda_sweep = run_lambda_sweep(inst, vars_list, lambda_ratios=(0.5, 1.0, 1.5, 2.0, 3.0), p=1, num_restarts=10, seed=42)
    save_lambda_sweep_results(lambda_sweep)

    print("\nAll sweeps completed and cached successfully to results/.")
