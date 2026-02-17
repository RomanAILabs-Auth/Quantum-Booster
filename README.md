# QuantumBooster 🚀

**Copyright Daniel Harding - RomanAILabs**

---

QuantumBooster is an AI-driven, quantum-inspired task and resource optimizer. It monitors CPU and memory, predicts future resource usage, schedules tasks using simulated quantum annealing, and generates insightful static and interactive visual reports. Operates in single-task, batch, or continuous modes.

---

## Features

- **AI Resource Prediction:** Uses historical execution data and polynomial regression to forecast CPU and memory usage.
- **Quantum Annealing Scheduler:** Optimizes task scheduling across available CPU cores.
- **Parallel Task Execution:** Fully leverages multi-core systems with multiprocessing.
- **Continuous Monitoring Mode:** Live CPU/memory monitoring with safe thresholds to prevent overload.
- **Visual Reports:** Generates static (matplotlib) and interactive (Plotly) 3D charts for performance analysis.
- **JIT Optimizations:** Critical computations accelerated using Numba for faster execution.
- **Configurable:** Adjustable task size, iterations, CPU thresholds, and update intervals via `config.json

- How to run
- python Quantum-Booster-V2.py --continuous
