<p align="center">
  <img src="media/logo.jpg" alt="Cute-Snippets Logo" width="220">
</p>

<h1 align="center">Cute-Snippets</h1>

<p align="center">
  <strong>🧪 A Curated Assortment of CuTe DSL Experiments & Tutorials</strong>
</p>

<p align="center">
  <a href="#-about">About</a> •
  <a href="#-repository-structure">Structure</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-tutorials">Tutorials</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 📖 About

**Cute-Snippets** is a dedicated space for exploring, benchmarking, and mastering **CuTe DSL** (the Python-based Domain Specific Language for NVIDIA's CuTe). 

The goal of this repository is to bridge the gap between low-level GPU hardware logic and high-level Pythonic abstraction. Here, we decompose complex kernel behaviors into "snippets"—digestible, well-documented code fragments that prioritize both performance and pedagogical clarity.

## 🌟 Key Objectives
- **DSL Primitives**: Deep dives into `Layout`, `Tensor`, and `TiledCopy/Math` in the Pythonic ecosystem.
- **Architectural Specialization**: Performance experiments tailored for **Ampere**, **Hopper**, and **Blackwell** architectures.
- **Algorithm Implementation**: Modern implementation of kernels like Softmax, LayerNorm, and Attention using CuTe DSL.
- **Educational Content**: A structured roadmap of tutorials to help developers transition from C++ CuTe to the DSL.

## 📂 Repository Structure

| Category | Path | Description |
| :--- | :--- | :--- |
| **Basics** | `/basic` | Minimalist verification of core CuTe DSL concepts (Layout, TiledCopy, TiledMMA). |
| **Kernels** | `/kernels` | Standard GPU kernels implemented with DSL clarity (TiledCopy, TiledMMA, Softmax). |
| **Utilities** | `/htile` | Helper utilities and reusable components (e.g., VectorCopy). |
| **Tutorials** | `/docs` | Step-by-step guides and deep-dive documentation. |
| **Media** | `/media` | Visual assets, diagrams, and repository logos. |

## 🚀 Getting Started

### Prerequisites
- **GPU**: NVIDIA Compute Capability 8.0+ (Ampere or newer).
- **Python**: 3.10+
- **Core Dependency**: [cutedsl](https://github.com/NVIDIA/cutedsl)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Cute-Snippets.git
cd Cute-Snippets

# Install the package in development mode (recommended)
pip install -e .

# Or install dependencies only (assuming cutedsl is available)
pip install nvidia-cutlass-dsl torch
```

**Note**: Installing with `pip install -e .` ensures that the `htile` module can be imported from anywhere. Alternatively, you can set the `PYTHONPATH` environment variable:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Running Examples

```bash
# Basic layout examples
python basic/00_tiled_copy_layout.py
python basic/01_tiled_mma_layout.py

# Kernel implementations
python kernels/00_tiled_copy.py
python kernels/01_tiled_mma.py
python kernels/02_softmax.py
```

## 📚 Tutorials

Coming soon! Check the `/docs` directory for step-by-step guides on:
- Understanding CuTe DSL Layouts
- Building Tiled Copy Operations
- Implementing Matrix Multiplication with Tiled MMA
- Advanced Kernel Patterns


## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.