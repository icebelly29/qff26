"""Classical preprocessing, requirement derivation, variable pruning, and index mapping."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src.instance import ProblemInstance


def derive_ev_requirements(instance: ProblemInstance) -> Dict[str, List[float]]:
    """
    Derive E_e, d_e, and p_e for each EV (§3.1).

    E_e = B_e * (SoC_target - SoC_now)
    d_e = ceil(E_e / (P * Delta))
    p_e = E_e / d_e
    """
    return instance.get_ev_derived_quantities()


def prune_variables(instance: ProblemInstance) -> List[Tuple[int, int, int]]:
    """
    Filter variables v = (e, s, t) (§3.3).

    Keep variable v only if arrival_e <= t and t + d_e <= deadline_e.
    Returns list of variables sorted lexicographically by (e, s, t).
    """
    return instance.get_pruned_variables()


def build_index_maps(
    variables: List[Tuple[int, int, int]],
) -> Tuple[Dict[Tuple[int, int, int], int], Dict[int, Tuple[int, int, int]]]:
    """
    Build bidirectional index mappings between variable tuples and qubit indices.

    qubit index i <-> variable tuple (e, s, t).
    """
    var_to_idx = {v: i for i, v in enumerate(variables)}
    idx_to_var = {i: v for i, v in enumerate(variables)}
    return var_to_idx, idx_to_var


def get_ev_power_and_duration_maps(
    instance: ProblemInstance,
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Return mappings from EV id to its active power p_e (kW) and duration d_e (slots)."""
    station_power = instance.get_station_power()
    powers: Dict[int, float] = {}
    durations: Dict[int, int] = {}
    for ev in instance.evs:
        powers[ev.id] = ev.power_per_slot_kw(station_power, instance.delta_hours)
        durations[ev.id] = ev.duration_slots(station_power, instance.delta_hours)
    return powers, durations

