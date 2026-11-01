"""Problem instance definitions, synthetic data generation, and JSON I/O."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Station:
    """Charging station configuration."""

    id: int
    name: str
    x: float
    y: float
    bays: int = 1
    power_kw: float = 11.0


@dataclass(frozen=True)
class EV:
    """Electric vehicle specification."""

    id: int
    name: str
    x: float
    y: float
    battery_kwh: float
    soc_now: float
    soc_target: float
    arrival_slot: int
    deadline_slot: int

    @property
    def energy_needed_kwh(self) -> float:
        """Energy required to reach target SoC: E_e = B_e * (SoC_target - SoC_now)."""
        return self.battery_kwh * (self.soc_target - self.soc_now)

    def duration_slots(self, power_kw: float, delta_hours: float = 1.0) -> int:
        """Charging duration in integer slots: d_e = ceil(E_e / (P * delta))."""
        return math.ceil(self.energy_needed_kwh / (power_kw * delta_hours))

    def power_per_slot_kw(self, power_kw: float, delta_hours: float = 1.0) -> float:
        """Average power per active slot: p_e = E_e / d_e."""
        d = self.duration_slots(power_kw, delta_hours)
        return self.energy_needed_kwh / d


@dataclass(frozen=True)
class ProblemInstance:
    """Complete problem instance specification."""

    num_slots: int
    delta_hours: float
    prices: List[float]
    stations: List[Station]
    evs: List[EV]
    kappa: float  # Travel cost weight ($/km)
    omega: float  # Waiting delay weight ($/slot)
    mu: float  # Peak-load proxy weight

    def get_station_power(self) -> float:
        """Uniform station charger power."""
        return self.stations[0].power_kw

    def get_ev_derived_quantities(self) -> Dict[str, List[float]]:
        """Return derived quantities E, d, p for all EVs (§3.1)."""
        power_kw = self.get_station_power()
        energies: List[float] = []
        durations: List[int] = []
        powers: List[float] = []

        for ev in self.evs:
            e_val = ev.energy_needed_kwh
            d_val = ev.duration_slots(power_kw, self.delta_hours)
            p_val = ev.power_per_slot_kw(power_kw, self.delta_hours)
            energies.append(round(e_val, 6))
            durations.append(d_val)
            powers.append(round(p_val, 6))

        return {
            "E": energies,
            "d": durations,
            "p": powers,
        }

    def get_pruned_variables(self) -> List[Tuple[int, int, int]]:
        """
        Return pruned binary variables (e, s, t) (§3.3).

        A variable is kept only if arrival <= t and t + d_e <= deadline.
        The list is sorted lexicographically by (e, s, t).
        """
        power_kw = self.get_station_power()
        variables: List[Tuple[int, int, int]] = []

        for ev in self.evs:
            d_e = ev.duration_slots(power_kw, self.delta_hours)
            for s in self.stations:
                for t in range(self.num_slots):
                    if ev.arrival_slot <= t and t + d_e <= ev.deadline_slot:
                        variables.append((ev.id, s.id, t))

        variables.sort(key=lambda item: (item[0], item[1], item[2]))
        return variables

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProblemInstance:
        """Reconstruct ProblemInstance from dictionary."""
        stations = [Station(**s) for s in data["stations"]]
        evs = [EV(**e) for e in data["evs"]]
        return cls(
            num_slots=data["num_slots"],
            delta_hours=data["delta_hours"],
            prices=list(data["prices"]),
            stations=stations,
            evs=evs,
            kappa=data["kappa"],
            omega=data["omega"],
            mu=data["mu"],
        )


def get_fixed_instance() -> ProblemInstance:
    """Return the fixed synthetic instance specified in SPEC.md §2."""
    stations = [
        Station(id=0, name="S0", x=2.0, y=3.0, bays=1, power_kw=11.0),
        Station(id=1, name="S1", x=7.0, y=1.0, bays=1, power_kw=11.0),
    ]

    evs = [
        EV(
            id=0,
            name="EV0",
            x=1.0,
            y=1.0,
            battery_kwh=40.0,
            soc_now=0.30,
            soc_target=0.80,
            arrival_slot=0,
            deadline_slot=4,
        ),
        EV(
            id=1,
            name="EV1",
            x=6.0,
            y=4.0,
            battery_kwh=60.0,
            soc_now=0.20,
            soc_target=0.70,
            arrival_slot=0,
            deadline_slot=4,
        ),
        EV(
            id=2,
            name="EV2",
            x=4.0,
            y=2.0,
            battery_kwh=30.0,
            soc_now=0.40,
            soc_target=0.90,
            arrival_slot=2,
            deadline_slot=5,
        ),
    ]

    return ProblemInstance(
        num_slots=5,
        delta_hours=1.0,
        prices=[0.30, 0.22, 0.15, 0.18, 0.25],
        stations=stations,
        evs=evs,
        kappa=0.3,
        omega=0.5,
        mu=0.01,
    )


def make_random_instance(
    seed: int,
    n_ev: int = 3,
    n_station: int = 2,
    T: int = 4,
) -> ProblemInstance:
    """Generate a reproducible random instance for scaling studies."""
    import numpy as np

    rng = np.random.default_rng(seed)

    stations = [
        Station(
            id=s,
            name=f"S{s}",
            x=float(rng.uniform(0.0, 10.0)),
            y=float(rng.uniform(0.0, 10.0)),
            bays=1,
            power_kw=11.0,
        )
        for s in range(n_station)
    ]

    prices = [round(float(p), 2) for p in rng.uniform(0.15, 0.35, size=T)]

    evs: List[EV] = []
    for e in range(n_ev):
        battery = float(rng.choice([30.0, 40.0, 50.0, 60.0]))
        soc_now = float(rng.choice([0.20, 0.30, 0.40]))
        soc_target = float(rng.choice([0.70, 0.80, 0.90]))
        arrival = int(rng.integers(0, max(1, T - 2)))
        deadline = int(rng.integers(arrival + 2, T + 1))

        evs.append(
            EV(
                id=e,
                name=f"EV{e}",
                x=float(rng.uniform(0.0, 10.0)),
                y=float(rng.uniform(0.0, 10.0)),
                battery_kwh=battery,
                soc_now=soc_now,
                soc_target=soc_target,
                arrival_slot=arrival,
                deadline_slot=deadline,
            )
        )

    return ProblemInstance(
        num_slots=T,
        delta_hours=1.0,
        prices=prices,
        stations=stations,
        evs=evs,
        kappa=0.3,
        omega=0.5,
        mu=0.01,
    )


def save_instance(instance: ProblemInstance, filepath: str | Path) -> None:
    """Write an instance to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(instance.to_dict(), f, indent=2)


def load_instance(filepath: str | Path) -> ProblemInstance:
    """Load an instance from a JSON file."""
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ProblemInstance.from_dict(data)


def decode_bitstring(
    bitstring: str,
    variables: List[Tuple[int, int, int]],
) -> List[Tuple[int, int, int]]:
    """
    Decode a Qiskit little-endian bitstring into active variables.

    In Qiskit, the rightmost bit is qubit 0 (variable 0).
    bitstring[-(i + 1)] corresponds to qubit i.
    """
    reversed_bits = bitstring[::-1]
    return [variables[i] for i, bit in enumerate(reversed_bits) if bit == "1"]


def bitstring_to_qubit_array(bitstring: str) -> List[int]:
    """Convert little-endian bitstring to an array where index i is qubit i."""
    return [int(b) for b in reversed(bitstring)]



if __name__ == "__main__":
    fixed_inst = get_fixed_instance()
    out_path = Path("results/instance.json")
    save_instance(fixed_inst, out_path)
    print(f"Saved fixed instance to {out_path}")
    derived = fixed_inst.get_ev_derived_quantities()
    print("Derived quantities:", derived)
    pruned = fixed_inst.get_pruned_variables()
    print(f"Pruned variables count: {len(pruned)}")
    print("Variables:", pruned)

