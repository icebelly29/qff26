"""Test suite for QUBO formulation and instance specifications."""

import pytest
from src.instance import get_fixed_instance, load_instance, save_instance


def test_instance():
    """Gate 1: Verify derived values match SPEC §3.1 (E, d, p) and pruned n = 12."""
    inst = get_fixed_instance()
    derived = inst.get_ev_derived_quantities()

    assert derived["E"] == [20.0, 30.0, 15.0], f"Expected E=[20, 30, 15], got {derived['E']}"
    assert derived["d"] == [2, 3, 2], f"Expected d=[2, 3, 2], got {derived['d']}"
    assert derived["p"] == [10.0, 10.0, 7.5], f"Expected p=[10, 10, 7.5], got {derived['p']}"

    pruned = inst.get_pruned_variables()
    assert len(pruned) == 12, f"Expected 12 pruned variables, got {len(pruned)}"


def test_qubit_budget():
    """Verify pruned variables satisfy hard constraint n <= 14."""
    inst = get_fixed_instance()
    pruned = inst.get_pruned_variables()
    assert len(pruned) <= 14, f"Qubit budget exceeded: {len(pruned)} > 14"


def test_bit_order():
    """Verify Qiskit little-endian decoding maps bitstring to correct qubits."""
    from src.instance import bitstring_to_qubit_array, decode_bitstring

    inst = get_fixed_instance()
    pruned = inst.get_pruned_variables()

    # 12-bit string with only bit index 1 set (second from right)
    bitstring = "000000000010"
    qubit_vals = bitstring_to_qubit_array(bitstring)

    assert qubit_vals[1] == 1, "Expected q1 to be 1"
    assert qubit_vals[10] == 0, "Expected q10 to be 0"

    active_vars = decode_bitstring(bitstring, pruned)
    assert len(active_vars) == 1
    # q1 is EV0, Station S0, Start slot 1 -> (0, 0, 1)
    assert active_vars[0] == (0, 0, 1)



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

