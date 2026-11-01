"""QUBO construction, coefficient expansion, QuadraticProgram, and Ising mapping."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram

from src.instance import ProblemInstance
from src.preprocess import get_ev_power_and_duration_maps, prune_variables


def compute_linear_costs(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
) -> List[float]:
    """
    Compute linear cost c_v for each pruned variable v = (e, s, t) (§4.2, Eq. E1).

    c_v = p_e * sum_{tau in I(e,t)} pi_tau + kappa * dist(e, s) + omega * (t - a_e)
    """
    powers, durations = get_ev_power_and_duration_maps(instance)
    c_v_list: List[float] = []

    for e, s, t in variables:
        ev = instance.evs[e]
        station = instance.stations[s]
        d_e = durations[e]
        p_e = powers[e]

        # E1: Energy cost
        energy_cost = p_e * sum(instance.prices[tau] for tau in range(t, t + d_e))

        # E1: Travel cost (Euclidean distance)
        dist = math.sqrt((ev.x - station.x) ** 2 + (ev.y - station.y) ** 2)
        travel_cost = instance.kappa * dist

        # E1: Waiting delay cost
        wait_cost = instance.omega * (t - ev.arrival_slot)

        # E1: Total linear cost for variable v
        c_v = energy_cost + travel_cost + wait_cost
        c_v_list.append(c_v)

    return c_v_list


def compute_penalty_bound_lambda(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    c_v_list: List[float],
) -> Tuple[float, float]:
    """
    Compute analytical penalty bound Lambda and default lambda (§4.5).

    Lambda = max_v [ c_v + mu * d_e * p_e * (p_e + 2 * sum_{e' != e} p_{e'}) ]
    Default lambda_A = lambda_O = 1.1 * Lambda
    """
    powers, durations = get_ev_power_and_duration_maps(instance)
    lambda_terms: List[float] = []

    for i, (e, _, _) in enumerate(variables):
        c_v = c_v_list[i]
        d_e = durations[e]
        p_e = powers[e]
        other_powers_sum = sum(powers[other_e] for other_e in powers if other_e != e)

        # Upper bound on savings from dropping EV e
        term = c_v + instance.mu * d_e * p_e * (p_e + 2.0 * other_powers_sum)
        lambda_terms.append(term)

    capital_lambda = max(lambda_terms)
    default_lambda = 1.1 * capital_lambda
    return capital_lambda, default_lambda


def find_conflicting_pairs(
    variables: List[Tuple[int, int, int]],
    durations: Dict[int, int],
) -> Set[Tuple[int, int]]:
    """
    Find conflicting pairs C for overlap penalty E4 (§4.4).

    C = pairs u = (e, s, t), v = (e', s, t') with e != e', same station s, and overlapping slots.
    """
    conflicts: Set[Tuple[int, int]] = set()
    n = len(variables)

    for u in range(n):
        eu, su, tu = variables[u]
        i_u = set(range(tu, tu + durations[eu]))
        for v in range(u + 1, n):
            ev, sv, tv = variables[v]
            if su == sv and eu != ev:
                i_v = set(range(tv, tv + durations[ev]))
                # Check slot intersection
                if len(i_u & i_v) > 0:
                    conflicts.add((u, v))

    return conflicts


def build_qubo_matrices(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    lambda_a: float,
    lambda_o: float,
) -> Tuple[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """
    Expand objective H(x) into QUBO constant, linear, and quadratic coefficients (§4.6).

    H(x) = const + sum_u Q_uu * x_u + sum_{u < v} Q_uv * x_u * x_v
    """
    powers, durations = get_ev_power_and_duration_maps(instance)
    c_v_list = compute_linear_costs(instance, variables)
    n = len(variables)

    constant: float = 0.0
    q_linear: Dict[int, float] = {i: 0.0 for i in range(n)}
    q_quadratic: Dict[Tuple[int, int], float] = {}

    # E1: Linear cost term -> Q_vv += c_v
    for v in range(n):
        q_linear[v] += c_v_list[v]

    # E3: Assignment penalty term: lambda_A * sum_e (sum_{v in V_e} x_v - 1)^2
    # Expands to: const += lambda_A per EV, Q_vv += -lambda_A, Q_uv += 2 * lambda_A for u < v in same EV
    for ev in instance.evs:
        constant += lambda_a
        ev_indices = [i for i, (e, _, _) in enumerate(variables) if e == ev.id]
        for v in ev_indices:
            q_linear[v] -= lambda_a
        for idx_u in range(len(ev_indices)):
            for idx_v in range(idx_u + 1, len(ev_indices)):
                u = ev_indices[idx_u]
                v = ev_indices[idx_v]
                q_quadratic[(u, v)] = q_quadratic.get((u, v), 0.0) + 2.0 * lambda_a

    # E4: Overlap penalty term: lambda_O * sum_{(u,v) in C} x_u * x_v
    # For conflicting pairs (u, v) in C with u < v -> Q_uv += lambda_O
    conflicts = find_conflicting_pairs(variables, durations)
    for u, v in conflicts:
        q_quadratic[(u, v)] = q_quadratic.get((u, v), 0.0) + lambda_o

    # Peak load proxy term (Eq. E2 expanded): mu * sum_tau L_tau(x)^2
    # L_tau^2 = sum_u p_u^2 * x_u + 2 * sum_{u < v} p_u * p_v * x_u * x_v
    # Summing over tau:
    # -> Q_uu += mu * p_u^2 * d_{e(u)}
    # -> Q_uv += 2 * mu * p_u * p_v * |I(u) cap I(v)| for every pair u < v
    for u in range(n):
        eu = variables[u][0]
        p_u = powers[eu]
        d_u = durations[eu]
        q_linear[u] += instance.mu * (p_u ** 2) * d_u

    for u in range(n):
        eu, _, tu = variables[u]
        p_u = powers[eu]
        i_u = set(range(tu, tu + durations[eu]))
        for v in range(u + 1, n):
            ev, _, tv = variables[v]
            p_v = powers[ev]
            i_v = set(range(tv, tv + durations[ev]))
            overlap_slots = len(i_u & i_v)
            if overlap_slots > 0:
                q_quadratic[(u, v)] = (
                    q_quadratic.get((u, v), 0.0)
                    + 2.0 * instance.mu * p_u * p_v * overlap_slots
                )

    return constant, q_linear, q_quadratic


def build_quadratic_program(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    lambda_a: float,
    lambda_o: float,
) -> QuadraticProgram:
    """
    Construct QuadraticProgram with binary variables and objective only (Rule 3).

    No linear constraints, no QuadraticProgramToQubo.
    """
    constant, q_linear, q_quadratic = build_qubo_matrices(
        instance, variables, lambda_a, lambda_o
    )
    n = len(variables)

    qp = QuadraticProgram(name="ev_charging_qubo")
    for i in range(n):
        qp.binary_var(name=f"x_{i}")

    linear_args = {f"x_{i}": val for i, val in q_linear.items()}
    quadratic_args = {
        (f"x_{u}", f"x_{v}"): val for (u, v), val in q_quadratic.items()
    }

    qp.minimize(
        constant=constant,
        linear=linear_args,
        quadratic=quadratic_args,
    )
    return qp


def evaluate_H_direct(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    x_bits: List[int],
    lambda_a: float,
    lambda_o: float,
) -> float:
    """Evaluate H(x) directly from specification formula (§4.4)."""
    powers, durations = get_ev_power_and_duration_maps(instance)
    c_v_list = compute_linear_costs(instance, variables)
    n = len(variables)

    # E1: Linear cost sum
    linear_cost = sum(c_v_list[i] * x_bits[i] for i in range(n))

    # E2: Grid load per slot and peak proxy
    loads = [0.0] * instance.num_slots
    for i in range(n):
        if x_bits[i]:
            e, _, t = variables[i]
            for tau in range(t, t + durations[e]):
                loads[tau] += powers[e]
    peak_term = instance.mu * sum(l**2 for l in loads)

    # E3: Assignment penalty: lambda_A * sum_e (sum_{v in V_e} x_v - 1)^2
    assignment_penalty = 0.0
    for ev in instance.evs:
        ev_sum = sum(x_bits[i] for i, (e, _, _) in enumerate(variables) if e == ev.id)
        assignment_penalty += lambda_a * ((ev_sum - 1.0) ** 2)

    # E4: Overlap penalty: lambda_O * sum_{(u, v) in C} x_u * x_v
    conflicts = find_conflicting_pairs(variables, durations)
    overlap_penalty = lambda_o * sum(x_bits[u] * x_bits[v] for (u, v) in conflicts)

    return linear_cost + peak_term + assignment_penalty + overlap_penalty


def evaluate_H_matrix(
    constant: float,
    q_linear: Dict[int, float],
    q_quadratic: Dict[Tuple[int, int], float],
    x_bits: List[int],
) -> float:
    """Evaluate H(x) from QUBO matrix coefficients (§4.6)."""
    val = constant
    for i, bit in enumerate(x_bits):
        if bit:
            val += q_linear[i]
    for (u, v), coeff in q_quadratic.items():
        if x_bits[u] and x_bits[v]:
            val += coeff
    return val


def evaluate_J(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    x_bits: List[int],
) -> float:
    """Evaluate true objective J(x) = sum c_v x_v + mu * sum L_tau^2 (§4.4)."""
    powers, durations = get_ev_power_and_duration_maps(instance)
    c_v_list = compute_linear_costs(instance, variables)
    n = len(variables)

    linear_cost = sum(c_v_list[i] * x_bits[i] for i in range(n))

    loads = [0.0] * instance.num_slots
    for i in range(n):
        if x_bits[i]:
            e, _, t = variables[i]
            for tau in range(t, t + durations[e]):
                loads[tau] += powers[e]
    peak_term = instance.mu * sum(l**2 for l in loads)

    return linear_cost + peak_term


def evaluate_term_magnitudes(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    x_bits: List[int],
    lambda_a: float,
    lambda_o: float,
) -> Dict[str, float]:
    """Calculate the individual magnitude of each objective component (Rule 7)."""
    powers, durations = get_ev_power_and_duration_maps(instance)
    c_v_list = compute_linear_costs(instance, variables)
    n = len(variables)

    energy_cost = 0.0
    travel_cost = 0.0
    wait_cost = 0.0

    for i in range(n):
        if x_bits[i]:
            e, s, t = variables[i]
            ev = instance.evs[e]
            st = instance.stations[s]
            d_e = durations[e]
            p_e = powers[e]
            energy_cost += p_e * sum(instance.prices[tau] for tau in range(t, t + d_e))
            dist = math.sqrt((ev.x - st.x) ** 2 + (ev.y - st.y) ** 2)
            travel_cost += instance.kappa * dist
            wait_cost += instance.omega * (t - ev.arrival_slot)

    total_linear = energy_cost + travel_cost + wait_cost

    loads = [0.0] * instance.num_slots
    for i in range(n):
        if x_bits[i]:
            e, _, t = variables[i]
            for tau in range(t, t + durations[e]):
                loads[tau] += powers[e]
    peak_term = instance.mu * sum(l**2 for l in loads)

    assignment_penalty = 0.0
    for ev in instance.evs:
        ev_sum = sum(x_bits[i] for i, (e, _, _) in enumerate(variables) if e == ev.id)
        assignment_penalty += lambda_a * ((ev_sum - 1.0) ** 2)

    conflicts = find_conflicting_pairs(variables, durations)
    overlap_penalty = lambda_o * sum(x_bits[u] * x_bits[v] for (u, v) in conflicts)

    total_h = total_linear + peak_term + assignment_penalty + overlap_penalty
    true_j = total_linear + peak_term

    return {
        "energy_cost": energy_cost,
        "travel_cost": travel_cost,
        "wait_cost": wait_cost,
        "total_linear_cost": total_linear,
        "peak_term": peak_term,
        "assignment_penalty": assignment_penalty,
        "overlap_penalty": overlap_penalty,
        "total_H": total_h,
        "true_J": true_j,
    }


def qubo_to_ising(qp: QuadraticProgram) -> Tuple[SparsePauliOp, float]:
    """Convert QuadraticProgram to Ising Hamiltonian and constant offset (§4.7)."""
    return qp.to_ising()


def brute_force_solve(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    lambda_a: float,
    lambda_o: float,
) -> Dict[str, Any]:
    """
    Exhaustively evaluate all 2^n bitstrings (§5, baseline 1).

    Returns the minimum of H and the best feasible J.
    """
    durations = {ev.id: ev.duration_slots(instance.get_station_power(), instance.delta_hours) for ev in instance.evs}
    conflicts = find_conflicting_pairs(variables, durations)
    n = len(variables)

    best_h = float("inf")
    best_h_int = -1
    best_h_feasible = False
    h_min_count = 0

    best_j = float("inf")
    best_j_int = -1
    j_min_count = 0

    for int_val in range(1 << n):
        x_bits = [(int_val >> i) & 1 for i in range(n)]

        # Feasibility check: exactly one start per EV and zero bay overlaps
        is_feasible = True
        for ev in instance.evs:
            ev_sum = sum(x_bits[i] for i, (e, _, _) in enumerate(variables) if e == ev.id)
            if ev_sum != 1:
                is_feasible = False
                break

        if is_feasible:
            for u, v in conflicts:
                if x_bits[u] and x_bits[v]:
                    is_feasible = False
                    break

        h_val = evaluate_H_direct(instance, variables, x_bits, lambda_a, lambda_o)

        if abs(h_val - best_h) < 1e-9:
            h_min_count += 1
        elif h_val < best_h:
            best_h = h_val
            best_h_int = int_val
            best_h_feasible = is_feasible
            h_min_count = 1

        if is_feasible:
            j_val = evaluate_J(instance, variables, x_bits)
            if abs(j_val - best_j) < 1e-9:
                j_min_count += 1
            elif j_val < best_j:
                best_j = j_val
                best_j_int = int_val
                j_min_count = 1

    best_h_bits = [(best_h_int >> i) & 1 for i in range(n)]
    best_j_bits = [(best_j_int >> i) & 1 for i in range(n)]

    # Qiskit little-endian bitstrings (qubit 0 rightmost)
    best_h_bitstring = "".join(str((best_h_int >> i) & 1) for i in reversed(range(n)))
    best_j_bitstring = "".join(str((best_j_int >> i) & 1) for i in reversed(range(n)))

    term_magnitudes = evaluate_term_magnitudes(
        instance, variables, best_h_bits, lambda_a, lambda_o
    )

    return {
        "best_H": best_h,
        "best_H_int": best_h_int,
        "best_H_bitstring": best_h_bitstring,
        "best_H_bits": best_h_bits,
        "best_H_feasible": best_h_feasible,
        "best_H_unique": (h_min_count == 1),
        "best_J": best_j,
        "best_J_int": best_j_int,
        "best_J_bitstring": best_j_bitstring,
        "best_J_bits": best_j_bits,
        "best_J_unique": (j_min_count == 1),
        "term_magnitudes": term_magnitudes,
    }


if __name__ == "__main__":
    from src.instance import get_fixed_instance

    inst = get_fixed_instance()
    vars_list = prune_variables(inst)
    cv_list = compute_linear_costs(inst, vars_list)
    cap_lambda, lam_val = compute_penalty_bound_lambda(inst, vars_list, cv_list)

    print(f"Lambda = {cap_lambda:.6f}")
    print(f"lambda_A = lambda_O = 1.1 * Lambda = {lam_val:.6f}")

    bf_result = brute_force_solve(inst, vars_list, lam_val, lam_val)
    print("\nBrute Force Results:")
    print(f"Optimal bit integer: {bf_result['best_H_int']}")
    print(f"Optimal bitstring (Qiskit order): {bf_result['best_H_bitstring']}")
    print(f"Minimum H: {bf_result['best_H']:.6f}")
    print(f"Minimum feasible J: {bf_result['best_J']:.6f}")
    print(f"Is feasible: {bf_result['best_H_feasible']}")
    print(f"Is unique ground state: {bf_result['best_H_unique']}")
    print("\nTerm Magnitudes at Optimum:")
    for k, v in bf_result["term_magnitudes"].items():
        print(f"  {k}: {v:.6f}")

