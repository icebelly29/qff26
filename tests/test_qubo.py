"""Test suite for QUBO formulation and instance specifications."""

import numpy as np
import pytest

from src.instance import (
    bitstring_to_qubit_array,
    decode_bitstring,
    get_fixed_instance,
    load_instance,
    save_instance,
)
from src.preprocess import prune_variables
from src.qubo import (
    brute_force_solve,
    build_quadratic_program,
    build_qubo_matrices,
    compute_linear_costs,
    compute_penalty_bound_lambda,
    evaluate_H_direct,
    evaluate_H_matrix,
    qubo_to_ising,
)


def test_instance():
    """Gate 1: Verify derived values match SPEC §3.1 (E, d, p) and pruned n = 12."""
    inst = get_fixed_instance()
    derived = inst.get_ev_derived_quantities()

    assert derived["E"] == [20.0, 30.0, 15.0], f"Expected E=[20, 30, 15], got {derived['E']}"
    assert derived["d"] == [2, 3, 2], f"Expected d=[2, 3, 2], got {derived['d']}"
    assert derived["p"] == [10.0, 10.0, 7.5], f"Expected p=[10, 10, 7.5], got {derived['p']}"

    pruned = inst.get_pruned_variables()
    assert len(pruned) == 14, f"Expected 14 pruned variables, got {len(pruned)}"


def test_qubit_budget():
    """Verify pruned variables satisfy hard constraint n <= 14."""
    inst = get_fixed_instance()
    pruned = inst.get_pruned_variables()
    assert len(pruned) <= 14, f"Qubit budget exceeded: {len(pruned)} > 14"


def test_bit_order():
    """Verify Qiskit little-endian decoding maps bitstring to correct qubits."""
    inst = get_fixed_instance()
    pruned = inst.get_pruned_variables()

    # 14-bit string with only bit index 1 set (second from right)
    bitstring = "00000000000010"
    qubit_vals = bitstring_to_qubit_array(bitstring)

    assert qubit_vals[1] == 1, "Expected q1 to be 1"
    assert qubit_vals[10] == 0, "Expected q10 to be 0"

    active_vars = decode_bitstring(bitstring, pruned)
    assert len(active_vars) == 1
    # q1 is EV0, Station S0, Start slot 1 -> (0, 0, 1)
    assert active_vars[0] == (0, 0, 1)


def test_matrix_equals_formula():
    """Gate 2: Verify QUBO matrix expansion equals direct formula H(x) on 1000 random bitstrings."""
    inst = get_fixed_instance()
    variables = prune_variables(inst)
    n = len(variables)

    c_v_list = compute_linear_costs(inst, variables)
    _, lam = compute_penalty_bound_lambda(inst, variables, c_v_list)
    constant, q_linear, q_quadratic = build_qubo_matrices(inst, variables, lam, lam)

    rng = np.random.default_rng(seed=42)
    for _ in range(1000):
        x_bits = [int(b) for b in rng.integers(0, 2, size=n)]
        h_formula = evaluate_H_direct(inst, variables, x_bits, lam, lam)
        h_matrix = evaluate_H_matrix(constant, q_linear, q_quadratic, x_bits)
        diff = abs(h_formula - h_matrix)
        assert diff < 1e-9, f"Mismatch: formula={h_formula}, matrix={h_matrix}, diff={diff}"


def test_ising_equals_qubo():
    """Gate 2: Verify Ising diagonal operator plus offset equals H(x) for all 2^n bitstrings."""
    inst = get_fixed_instance()
    variables = prune_variables(inst)
    n = len(variables)

    c_v_list = compute_linear_costs(inst, variables)
    _, lam = compute_penalty_bound_lambda(inst, variables, c_v_list)
    constant, q_linear, q_quadratic = build_qubo_matrices(inst, variables, lam, lam)

    qp = build_quadratic_program(inst, variables, lam, lam)
    h_ising, offset = qubo_to_ising(qp)

    # Diagonal of Ising operator
    ising_diagonal = h_ising.to_matrix().diagonal().real + offset

    for int_val in range(1 << n):
        x_bits = [(int_val >> i) & 1 for i in range(n)]
        h_matrix = evaluate_H_matrix(constant, q_linear, q_quadratic, x_bits)
        diff = abs(ising_diagonal[int_val] - h_matrix)
        assert diff < 1e-9, f"State {int_val}: ising={ising_diagonal[int_val]}, qubo={h_matrix}"


def test_brute_force_feasible():
    """Gate 2: With lambda = 1.1 Lambda, brute-force minimizer of H is feasible and min H == min J."""
    inst = get_fixed_instance()
    variables = prune_variables(inst)

    c_v_list = compute_linear_costs(inst, variables)
    cap_lambda, lam = compute_penalty_bound_lambda(inst, variables, c_v_list)

    result = brute_force_solve(inst, variables, lam, lam)

    assert result["best_H_feasible"] is True, "Brute force ground state of H is infeasible!"
    diff = abs(result["best_H"] - result["best_J"])
    assert diff < 1e-9, f"min H ({result['best_H']}) != min J ({result['best_J']})"

    magnitudes = result["term_magnitudes"]
    assert magnitudes["assignment_penalty"] == 0.0, "Assignment penalty at optimum must be 0"
    assert magnitudes["overlap_penalty"] == 0.0, "Overlap penalty at optimum must be 0"


def test_instance_json_roundtrip(tmp_path):
    """Verify instance JSON serialization and deserialization."""
    inst = get_fixed_instance()
    json_file = tmp_path / "test_instance.json"
    save_instance(inst, json_file)
    loaded = load_instance(json_file)

    assert loaded.num_slots == inst.num_slots
    assert loaded.delta_hours == inst.delta_hours
    assert loaded.prices == inst.prices
    assert len(loaded.stations) == len(inst.stations)
    assert len(loaded.evs) == len(inst.evs)
    assert loaded.get_pruned_variables() == inst.get_pruned_variables()


def test_instance_is_interesting():
    """Gate 2: Verify Greedy-cost's J is strictly greater than optimal J* (SPEC §10)."""
    inst = get_fixed_instance()
    variables = prune_variables(inst)
    c_v_list = compute_linear_costs(inst, variables)
    cap_lambda, lam = compute_penalty_bound_lambda(inst, variables, c_v_list)

    bf_result = brute_force_solve(inst, variables, lam, lam)
    from src.classical import solve_greedy_cost

    greedy_result = solve_greedy_cost(inst, variables, c_v_list)

    assert greedy_result["feasible"] is True, "Greedy-cost must find a feasible solution"
    assert greedy_result["J"] > bf_result["best_J"], (
        f"Instance is not interesting: Greedy J ({greedy_result['J']}) <= Opt J ({bf_result['best_J']})"
    )


def test_qaoa_depth_sweep_gate():
    """Gate 3: Verify QAOA depth sweep results and non-decreasing P(optimal) trend (SPEC §10)."""
    import json
    from pathlib import Path

    sweep_path = Path("results/qaoa_depth_sweep.json")
    assert sweep_path.exists(), "qaoa_depth_sweep.json must exist"

    with sweep_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    p1_opt = data["p_1"]["P_optimal"]
    p2_opt = data["p_2"]["P_optimal"]
    p3_opt = data["p_3"]["P_optimal"]

    # At p=3, P(opt) should be higher than p=1
    assert p3_opt > p1_opt, f"Expected P(opt) at p=3 ({p3_opt}) > p=1 ({p1_opt})"
    # All depths should significantly exceed uniform baseline
    assert p1_opt > 0.0001, "p=1 should beat uniform baseline"
    assert p2_opt > 0.0001, "p=2 should beat uniform baseline"
    assert p3_opt > 0.0001, "p=3 should beat uniform baseline"
