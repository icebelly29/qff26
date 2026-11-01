window.CIRCUIT_TRANSPILATION = {
  "noisy_simulation": {
    "P_feasible_noisy": 0.0010986328125,
    "P_optimal_noisy": 0.0,
    "feasible_counts": 9,
    "noise_model": "depolarizing (0.1% 1-qubit, 1.0% 2-qubit)",
    "optimal_counts": 0,
    "shots": 8192
  },
  "transpiled_depth_and_gates": {
    "p_1": {
      "cx_count": 150,
      "depth": 84,
      "ops": {
        "cx": 150,
        "rz": 158,
        "sx": 42
      },
      "total_ops": 350
    },
    "p_2": {
      "cx_count": 300,
      "depth": 132,
      "ops": {
        "cx": 300,
        "rz": 288,
        "sx": 70
      },
      "total_ops": 658
    },
    "p_3": {
      "cx_count": 450,
      "depth": 180,
      "ops": {
        "cx": 450,
        "rz": 418,
        "sx": 98
      },
      "total_ops": 966
    }
  }
};
