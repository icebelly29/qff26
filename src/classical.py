"""Classical baseline solvers (SPEC §5): Brute Force, FCFS, and Greedy-cost."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from src.instance import ProblemInstance
from src.preprocess import get_ev_power_and_duration_maps, prune_variables
from src.qubo import (
    brute_force_solve,
    compute_linear_costs,
    evaluate_H_direct,
    evaluate_J,
    evaluate_term_magnitudes,
    find_conflicting_pairs,
)


def get_all_feasible_schedules(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
) -> List[Dict[str, Any]]:
    """
    Find all feasible schedules among the 2^n configurations.

    A schedule is feasible if each EV has exactly one start slot and station,
    and no two EVs occupy the same station bay at the same time.
    Returns the list sorted by true objective J ascending.
    """
    powers, durations = get_ev_power_and_duration_maps(instance)
    conflicts = find_conflicting_pairs(variables, durations)
    n = len(variables)

    feasible_list: List[Dict[str, Any]] = []

    for int_val in range(1 << n):
        bits = [(int_val >> i) & 1 for i in range(n)]

        # Constraint E3: Exactly one assignment per EV
        is_feasible = True
        for ev in instance.evs:
            ev_count = sum(bits[i] for i, (e, _, _) in enumerate(variables) if e == ev.id)
            if ev_count != 1:
                is_feasible = False
                break

        if not is_feasible:
            continue

        # Constraint E4: No station bay overlap
        for u, v in conflicts:
            if bits[u] and bits[v]:
                is_feasible = False
                break

        if is_feasible:
            j_val = evaluate_J(instance, variables, bits)
            bitstring = "".join(str((int_val >> i) & 1) for i in reversed(range(n)))
            assigned_vars = [variables[i] for i in range(n) if bits[i]]
            feasible_list.append(
                {
                    "int_val": int_val,
                    "bitstring": bitstring,
                    "bits": bits,
                    "assigned_vars": assigned_vars,
                    "J": j_val,
                }
            )

    feasible_list.sort(key=lambda item: item["J"])
    return feasible_list


def solve_fcfs(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
) -> Dict[str, Any]:
    """
    First-Come, First-Served baseline (§5, Baseline 2).

    Order EVs by arrival then index.
    Each takes the earliest feasible (t) on the lowest-index station with no bay conflict.
    """
    powers, durations = get_ev_power_and_duration_maps(instance)
    conflicts = find_conflicting_pairs(variables, durations)
    n = len(variables)

    ev_order = sorted(instance.evs, key=lambda ev: (ev.arrival_slot, ev.id))
    scheduled_indices: List[int] = []
    is_feasible = True

    for ev in ev_order:
        cand_indices = [i for i, (e, _, _) in enumerate(variables) if e == ev.id]
        # Sort by earliest start slot t, then lowest station s
        cand_indices.sort(key=lambda i: (variables[i][2], variables[i][1]))

        chosen = None
        for idx in cand_indices:
            has_conflict = any(
                (min(idx, prev), max(idx, prev)) in conflicts
                for prev in scheduled_indices
            )
            if not has_conflict:
                chosen = idx
                break

        if chosen is not None:
            scheduled_indices.append(chosen)
        else:
            is_feasible = False

    bits = [1 if i in scheduled_indices else 0 for i in range(n)]
    bitstring = "".join(str(bits[i]) for i in reversed(range(n)))
    assigned_vars = [variables[i] for i in scheduled_indices]

    j_val = evaluate_J(instance, variables, bits) if is_feasible else float("inf")

    return {
        "method": "FCFS",
        "feasible": is_feasible,
        "J": j_val,
        "bits": bits,
        "bitstring": bitstring,
        "assigned_vars": assigned_vars,
    }


def solve_greedy_cost(
    instance: ProblemInstance,
    variables: List[Tuple[int, int, int]],
    c_v_list: List[float] | None = None,
) -> Dict[str, Any]:
    """
    Greedy-cost baseline (§5, Baseline 3).

    Order EVs by deadline slack (f_e - a_e - d_e).
    Each takes the conflict-free v with the smallest c_v (ignoring peak term).
    """
    powers, durations = get_ev_power_and_duration_maps(instance)
    conflicts = find_conflicting_pairs(variables, durations)
    if c_v_list is None:
        c_v_list = compute_linear_costs(instance, variables)
    n = len(variables)

    # Slack = f_e - a_e - d_e
    slack_order = sorted(
        instance.evs,
        key=lambda ev: (ev.deadline_slot - ev.arrival_slot - durations[ev.id], ev.id),
    )

    scheduled_indices: List[int] = []
    is_feasible = True

    for ev in slack_order:
        cand_indices = [i for i, (e, _, _) in enumerate(variables) if e == ev.id]
        # Sort by smallest linear cost c_v, then index
        cand_indices.sort(key=lambda i: (c_v_list[i], i))

        chosen = None
        for idx in cand_indices:
            has_conflict = any(
                (min(idx, prev), max(idx, prev)) in conflicts
                for prev in scheduled_indices
            )
            if not has_conflict:
                chosen = idx
                break

        if chosen is not None:
            scheduled_indices.append(chosen)
        else:
            is_feasible = False

    bits = [1 if i in scheduled_indices else 0 for i in range(n)]
    bitstring = "".join(str(bits[i]) for i in reversed(range(n)))
    assigned_vars = [variables[i] for i in scheduled_indices]

    j_val = evaluate_J(instance, variables, bits) if is_feasible else float("inf")

    return {
        "method": "Greedy-cost",
        "feasible": is_feasible,
        "J": j_val,
        "bits": bits,
        "bitstring": bitstring,
        "assigned_vars": assigned_vars,
    }

