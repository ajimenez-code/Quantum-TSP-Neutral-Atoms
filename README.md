# Quantum-TSP-Neutral-Atoms

### Introduction
Solving the Travelling Salesperson Problem on neutral-atom quantum processors, using VQAA and QAOA.

#### The Travelling Salesperson problem
We aim to solve the optimization version of the Travelling Salesperson Problem. Given a set of n cities, all connected to each other, the problem asks: what is the shortest route that must be followed so that each city is visited exactly once before returning to the starting point? The problem is known to be NP-hard.

#### Neutral-atom quantum computers
The two quantum algorithms considered, VQAA (Variational Quantum Adiabatic Algorithm) and QAOA (Quantum Approximate Optimization Algorithm), will be implemented and executed on neutral-atom quantum computers, using the Python libraries Pulser and Qadence, both developed by Pasqal.

#### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/ajimenez-code/Quantum-TSP-Neutral-Atoms.git
cd Quantum-TSP-Neutral-Atoms
pip install -r requirements.txt
```
#### Workflow
The TSP instances are defined in `[TSP_QUBO_Formulation.py]`. These are then mapped onto a neutral-atom system in `[Embedding.py]`. The two algorithms, VQAA and QAOA, are executed in `[VQAA.py]` and `[QAOA.py]`, respectively. The most relevant visual results are stored in the `media` directory, including:
* **Problem instances & embedding layouts** mapped onto the hardware.
* **STRESS evaluation** metrics.
* **Best and worst-case scenarios** for both algorithms and instances.

#### Author
This repository contains the complementary material for my Bachelor's Thesis in Physics, pursued at Universidad Complutense de Madrid. I, Alberto Jiménez Llop, am the sole author of this work.


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pulser-Framework-6C3483?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Quantum-Neutral%20Atoms-00BFFF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>
