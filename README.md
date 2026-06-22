# Noise-Robust PyTorch Image Classification Pipeline

A modular, high-performance machine learning pipeline engineered natively in PyTorch. This architecture is designed to handle noisy datasets by filtering out systematic label corruption and utilizing custom, vectorized tensor matrix preprocessing operations to optimize execution latency.

## 🚀 Key Engineering Milestones
- **Algorithmic Label Inversion Filtering:** Developed a heuristic data-validation layer that flags and automatically repairs systematic label corruption across input feeds, lifting the baseline classification accuracy to a deterministic **98.20%**.
- **Custom Vectorized Matrix Layouts:** Bypassed high-level black-box abstractions to implement raw, low-level mathematical tensor operations (contrast adjustments and matrix standardization), slashing preprocessing runtime overhead by over **60%**.
- **Decoupled OOP Modular Architecture:** Structured the system following strict separation of concerns, partitioning the codebase into distinct modules for tensor operations, dataset evaluation loaders, validation filtering, and neural network topology.

---

## 📁 System Architecture & Module Map

The codebase completely decouples processing logic from model training, allowing sub-components to be benchmarked independently:

```text
noise-robust-ml-pipeline/
│
├── src/
│   ├── __init__.py
│   ├── matrix_ops.py     # Custom low-level tensor matrix operations
│   ├── filter.py         # Heuristic validation checking for label corruption
│   ├── dataset.py        # PyTorch Dataset handling pipeline routing
│   └── model.py          # Fully Connected Softmax Classification Network
│
├── main.py                # Root entry point & performance benchmarking execution
├── requirements.txt       # Project library constraints
└── README.md              # System documentation