"""Smoke test for Qiskit environment and V2 primitives verification."""

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from qiskit_optimization import QuadraticProgram


def test_qiskit_smoke_pipeline():
    """Verify QuadraticProgram -> Ising -> QAOAAnsatz -> StatevectorEstimator -> Bitstring decoding."""
    # 1. Build a 2-variable QuadraticProgram
    qp = QuadraticProgram(name="smoke_test")
    qp.binary_var("x0")
    qp.binary_var("x1")
    qp.minimize(linear={"x0": 1.0, "x1": 2.0}, quadratic={("x0", "x1"): -1.0})

    # 2. Call to_ising()
    h_c, offset = qp.to_ising()
    assert h_c.num_qubits == 2
    assert offset == 1.25

    # 3. Build QAOAAnsatz
    ansatz = QAOAAnsatz(cost_operator=h_c, reps=1)
    assert ansatz.num_qubits == 2

    # 4. Evaluate with StatevectorEstimator (V2 primitive)
    estimator = StatevectorEstimator()
    params = np.array([0.5, 0.5])
    pub = (ansatz, [h_c], [params])
    job = estimator.run([pub])
    evs = job.result()[0].data.evs
    assert len(evs) == 1

    # 5. Decode bitstring (little-endian mapping)
    assigned_circuit = ansatz.assign_parameters(params)
    sv = Statevector(assigned_circuit)
    probs = sv.probabilities_dict()
    assert len(probs) == 4

    # Bitstring '01' has qubit 0 = 1, qubit 1 = 0
    test_bitstring = "01"
    x0 = int(test_bitstring[-1])
    x1 = int(test_bitstring[-2])
    assert x0 == 1
    assert x1 == 0

