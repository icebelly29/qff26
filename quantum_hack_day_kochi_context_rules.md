# Quantum Hack Day Kochi — Context, Requirements & Evaluation Manual

> **Event Context:** Qiskit Fall Fest 2026 · QuantumX Hackathon  
> **Hosts:** QuantumX Foundation × TinkerHub Foundation / TinkerSpace  
> **Date:** Sunday, 1 November 2026 | 9:00 AM – 6:00 PM IST  
> **Venue:** TinkerSpace, Kochi (CUSAT P.O., Kalamassery, Kochi, Kerala)  
> **Primary Framework:** Qiskit (IBM Quantum SDK), Python, Hybrid Classical-Quantum Workflows  

---

## 1. System Prompt Context (For AI Assistant / Hackathon LLM)

```text
You are an expert Quantum Computing Engineer and Hackathon Technical Mentor operating as an AI assistant for a team competing at "Quantum Hack Day Kochi" (part of IBM Qiskit Fall Fest 2026).

Your objectives when building solutions:
1. Always implement a functional, runnable quantum component using the Qiskit SDK (v1.x+ syntax).
2. Keep qubit scales realistic for simulators (4 to 12 qubits) so code executes without timeouts or out-of-memory errors.
3. Formulate the mathematical foundation cleanly (e.g., QUBO matrix, Hamiltonian mapping, or parameterized ansatz circuits).
4. Build hybrid classical-quantum pipelines: classical data preprocessing -> quantum circuit execution/optimization -> classical metric evaluation and visualization.
5. Provide a baseline classical comparison (e.g., Classical Greedy/Dijkstra vs. QAOA, or Classical Scikit-Learn SVM vs. QSVC/VQC) to satisfy judging criteria.
```

---

## 2. Event Schedule & Sprint Windows

| Time Window | Phase | Key Milestone |
|---|---|---|
| **09:00 – 09:30 AM** | Check-in & Setup | Desk setup, Python/Qiskit environment verification |
| **09:30 – 09:45 AM** | Opening & Briefing | Official rules, mentor intro, guidelines |
| **09:45 – 10:00 AM** | Track Finalization | Final selection of problem track |
| **10:00 – 01:30 PM** | **Hackathon Build Sprint I** | Architecture design, data encoding, quantum circuit build |
| **01:30 – 02:00 PM** | Lunch & Networking | Architecture check with mentors |
| **02:00 – 04:00 PM** | **Hackathon Build Sprint II** | Optimization, baseline comparisons, UI/charts, demo prep |
| **04:15 – 04:30 PM** | **Project Submission & Setup** | Push code to GitHub, verify links, set up presentation laptop |
| **04:30 – 05:15 PM** | **Project Showcase & Demos** | Live prototype pitch to judges and participants |
| **05:15 – 05:30 PM** | Jury Deliberation | Scoring and evaluations |
| **05:30 – 06:00 PM** | Awards & Closing | Winners announced, certificates |

---

## 3. General Requirements & Prerequisites

* **Eligibility:** Open to students, beginners, software engineers, AI/ML developers, and researchers.
* **Prerequisites:** Prior quantum computing experience is **not** mandatory. Basic Python proficiency is strongly recommended.
* **Mandatory Quantum Core:**
  * Every qualifying submission **must** include a genuine, working quantum computing component built with Qiskit or quantum algorithm simulation.
  * Pure classical applications or purely theoretical slide decks without circuit code will be disqualified.
  * Quantum Advantage over classical state-of-the-art is **not** required. The priority is a valid mathematical formulation, circuit execution, and hybrid orchestration.
* **Data Security & Privacy:**
  * All datasets used must be public, synthetically generated, or properly licensed.
  * Using any Personally Identifiable Information (PII) or unanonymized patient medical records is strictly forbidden.

---

## 4. Problem Statements & Track Options

Teams must choose **one** of the five official challenge tracks:

### Track 01: Quantum Optimization | Smart Mobility
* **Problem:** Emergency Vehicle Routing under dynamic traffic and multi-node hospital constraints.
* **Expected Methods:** QAOA, QUBO formulation, Hybrid classical clustering + quantum route optimization.
* **Deliverables:** Mathematical QUBO/Hamiltonian, Qiskit optimization solver, route visualizer, and a comparative benchmark against classical Dijkstra/TSP solver.

### Track 02: Quantum Machine Learning | Healthcare
* **Problem:** Complex multi-dimensional clinical feature classification (e.g., heart disease or cancer diagnosis).
* **Expected Methods:** Variational Quantum Classifiers (VQC), Quantum Kernel Methods (QSVC, ZZFeatureMap), Hybrid PyTorch-Qiskit networks.
* **Deliverables:** Dimensionality reduction pipeline (PCA to 4–8 features), quantum feature encoding circuit, parameter optimization, and comparison with classical SVM/Random Forest.

### Track 03: Quantum Optimization | Energy & Sustainability
* **Problem:** Microgrid unit commitment and dynamic power dispatch balancing intermittent renewable sources, battery storage, and fluctuating loads.
* **Expected Methods:** QUBO mapping of Unit Commitment, QAOA, VQE-based constrained optimization.
* **Deliverables:** Cost and penalty formulation, stable dispatch schedule, and comparison with classical SciPy/PuLP linear solvers.

### Track 04: Quantum Optimization | Electric Mobility
* **Problem:** EV charging bay assignment and scheduling to minimize grid peak load and driver wait times.
* **Expected Methods:** Job-shop/Bin-packing reduced to QUBO/Ising models, QAOA.
* **Deliverables:** Feasible bay/time-slot allocation matrix, penalty constraint model, visual schedule Gantt/chart.

### Track 05: Cybersecurity | Post-Quantum Cryptography (PQC) & Quantum Security
* **Problem:** Mitigating quantum vulnerabilities against classical public-key cryptography (RSA/ECC) using quantum-safe algorithms.
* **Expected Methods:** Lattice-based key encapsulation (ML-KEM / Kyber), quantum stateful signatures (ML-DSA / Dilithium), or QRNG integration.
* **Deliverables:** Proof-of-concept showing classical vulnerability vs. PQC defense, execution latency/key-size benchmarking.

---

## 5. Submission Guidelines

* **Submission Deadline:** Must be ready and submitted between **4:15 PM and 4:30 PM**.
* **Repository Deliverables (GitHub):**
  1. **Working Codebase:** Well-structured Python scripts or Jupyter Notebooks using current Qiskit libraries.
  2. **`README.md` Documentation:**
     * Problem statement and chosen track.
     * System architecture diagram / pipeline flow.
     * Mathematical formulation (equations, QUBO matrix, or ansatz details).
     * Circuit diagram snapshots and convergence/evaluation plots.
     * Clear setup and execution instructions (`requirements.txt`).
  3. **Baseline Comparison:** Clear output tables or plots contrasting classical vs. quantum results.
* **Live Showcase & Pitch Format (4:30 PM – 5:15 PM):**
  * Live demo on team laptop (running simulator or recorded deterministic execution if live runs take too long).
  * 3–5 minute presentation followed by 2–3 minutes of judge Q&A.

---

## 6. Judging Criteria & Scoring Weights

| Criteria | Weight | Focus Areas |
|---|:---:|---|
| **Mathematical Formulation & Technical Depth** | **30%** | Rigor in mapping the real-world problem into quantum gates, operators, cost Hamiltonians, or QUBO matrices. |
| **Code Implementation & Qiskit Usage** | **25%** | Correct usage of Qiskit SDK, circuit depth awareness, qubit efficiency, parameter optimization, and code cleanliness. |
| **Comparative Evaluation & Practicality** | **20%** | Fair benchmarking against a classical algorithmic baseline; realistic awareness of NISQ hardware constraints. |
| **Live Demo & Presentation** | **15%** | Stability of prototype execution, clear communication, and ability to answer technical inquiries from the jury. |
| **Creativity & Problem Framing** | **10%** | Innovative problem formulation, hybrid architecture design, or novel visualization. |

---

## 7. Hackathon Rules & Code of Conduct

1. **Team Size:** Solo or teams of up to **4 members**.
2. **Track Scope:** Teams must select and build for **one single challenge track**.
3. **Build Window:** All core development must occur during the designated build sprints on hack day.
4. **Third-Party Code & Libraries:** Standard open-source libraries (Qiskit, NumPy, NetworkX, SciPy, Matplotlib, PyTorch, Scikit-learn) are permitted. Pre-existing domain code must be acknowledged.
5. **Conduct Standards:**
   * Strictly adhere to the IBM Qiskit Code of Conduct.
   * Maintain an inclusive, collaborative environment. Harassment of any kind results in immediate disqualification and removal.
   * Escalations and inquiries: contact the on-site organizing team or email `events@quantumx.community`.