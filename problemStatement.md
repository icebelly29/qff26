# QuantumX Hackathon — Problem Statements

Welcome, participants! Explore the intersection of quantum computing, AI, energy optimization, and cybersecurity by choosing one of the following challenges.

Each problem is designed to encourage experimentation, practical implementation, and innovative thinking. You can use quantum algorithms, hybrid quantum-classical approaches, or quantum-inspired techniques where appropriate.

---

## Problem 1: Optimize Emergency Vehicle Routing

**Track:** Quantum Optimization | Smart Mobility

### Problem Statement
Emergency response teams need to determine efficient routes for ambulances to reach multiple locations while considering distance, traffic, and road availability.

Your task is to develop a **quantum-assisted route optimization solution** that determines an efficient route for an emergency vehicle visiting multiple locations.

### Challenge
Given:
- A set of locations
- Distances between locations
- Optional traffic or travel-cost values
- A starting location

Build a solution that minimizes total travel cost.

### What to Demonstrate
- Your route optimization model and objective function
- How distance, traffic, and road constraints are incorporated
- The quantum or hybrid optimization technique used
- A comparison with a classical baseline
- Results showing route cost and optimization performance

---

## Problem 2: Quantum Machine Learning for Disease Classification

**Track:** Quantum Machine Learning | Healthcare AI

### Problem Statement
Medical datasets often contain multiple features that can help distinguish between healthy and diseased patients.

Build a **hybrid quantum-classical machine learning model** that classifies patients using a provided or publicly available dataset.

The goal is to explore how quantum circuits can be incorporated into a machine-learning pipeline, not necessarily to prove that quantum machine learning outperforms classical methods.

### Challenge
Develop a classification pipeline that:
- Processes and preprocesses medical data
- Selects relevant features for quantum encoding
- Integrates a parameterized quantum circuit into a learning model
- Evaluates predictions using appropriate metrics

### What to Demonstrate
- Dataset selection and preprocessing
- Quantum feature encoding and circuit architecture
- Training and evaluation methodology
- Performance metrics such as accuracy, precision, recall, F1-score, and ROC-AUC, where appropriate
- Comparison with a classical machine-learning baseline

*Note: This is a research prototype, not a clinically validated diagnostic system. Use public or synthetic data and avoid making clinical claims.*

---

## Problem 3: Quantum Simulation for Energy Systems

**Track:** Quantum Optimization | Sustainable Energy

### Problem Statement
Energy systems must balance generation and consumption while managing operational costs and limited generation capacity.

Develop a simplified quantum-assisted optimization model that determines an efficient allocation of energy resources while satisfying demand.

### Challenge
Given:
- Multiple energy sources
- Total energy demand
- Generation costs
- Capacity constraints for each source

Find an allocation that minimizes total generation cost while meeting demand and respecting capacity constraints.

### What to Demonstrate
- Mathematical formulation of the allocation problem
- Representation of costs and constraints
- Quantum or hybrid optimization approach
- Feasibility of the resulting allocation
- Comparison with a classical optimization baseline

Consider using a small-scale Quadratic Unconstrained Binary Optimization (QUBO) formulation, QAOA, or another suitable optimization method.

---

## Problem 4: Optimize EV Charging Stations

**Track:** Quantum Optimization | Smart Mobility

### Problem Statement
As electric vehicle adoption increases, charging stations can experience periods of high demand, leading to waiting times, uneven station utilization, and higher charging costs.

Develop a quantum-assisted optimization model that determines **when and where electric vehicles should be charged** while minimizing waiting time and energy costs.

### Challenge
Given:
- EV locations and battery levels
- Charging station locations
- Charging station capacity
- Electricity prices across time slots
- Charging demand and time constraints

Determine an optimized charging schedule that allocates vehicles to suitable stations and time slots.

### What to Demonstrate
- A mathematical model for charging assignment and scheduling
- How station capacity and charging constraints are enforced
- An objective that considers energy cost and waiting time
- A quantum or hybrid optimization implementation
- Comparison with a classical scheduling baseline

You may use simulated EV demand and charging infrastructure if real-world data is unavailable.

---

## Problem 5: Post-Quantum Cryptography for a Quantum-Safe Future

**Track:** Cybersecurity | Post-Quantum Cryptography (PQC)

### Problem Statement
Many widely used cryptographic systems could become vulnerable to attacks from sufficiently powerful quantum computers. Organizations need to prepare by exploring cryptographic algorithms designed to protect sensitive data and secure digital communications in a post-quantum world.

Develop a prototype demonstrating how an existing application or secure communication workflow can be upgraded using post-quantum cryptography.

Explore how quantum-resistant algorithms can protect data while considering practical factors such as performance, key sizes, and compatibility.

### Challenge
Build a prototype using one or more post-quantum cryptographic techniques to secure a sample application, communication workflow, or data-protection process.

### Inputs You Can Work With
- A sample application or secure communication workflow
- Conventional cryptographic algorithms for comparison
- Post-quantum algorithms such as ML-KEM or ML-DSA
- Sample messages, files, or data to secure
- Performance metrics such as execution time, key size, ciphertext size, and signature size

### Possible Approaches
- Post-quantum key establishment using ML-KEM
- Digital signatures using ML-DSA
- Hybrid classical and post-quantum cryptography
- Quantum-safe secure messaging or file protection
- PQC performance benchmarking and migration tools

### What to Demonstrate
- How the selected algorithm fits into your application
- How keys are established, messages are encrypted, or signatures are generated and verified
- Comparison with a conventional cryptographic baseline
- Performance measurements and practical trade-offs
- A migration strategy explaining how the solution could help prepare an application for the post-quantum era

*Use established cryptographic libraries and standardized algorithms. Do not design your own cryptographic primitives.*

---

## General Expectations for All Teams

Regardless of the problem you choose, aim to demonstrate:

1. **Problem understanding:** Clearly define the problem, assumptions, and objectives.
2. **Working prototype:** Build a functional implementation rather than only a conceptual presentation.
3. **Quantum component:** Explain the quantum algorithm, circuit, or quantum-assisted technique used.
4. **Evaluation:** Provide meaningful experiments, metrics, and a classical baseline wherever applicable.
5. **Reproducibility:** Document your setup, dependencies, data sources, and instructions to run your solution.
6. **Presentation:** Explain your approach, results, limitations, and potential real-world impact.

**Important:** Quantum advantage is not required. A well-designed hybrid prototype with honest evaluation and clear technical reasoning is valuable.

Choose a challenge, form your team, experiment boldly, and build something meaningful.

**Happy building — Team QuantumX!**