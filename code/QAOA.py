#!/usr/bin/env python
# coding: utf-8

import qadence
import torch
import pulser
import optuna
import numpy as np
from Utilities import plot_distribution
from qadence import RydbergDevice, kron, chain, AnalogInteraction, Register, QuantumCircuit, QuantumModel, H, RX, RZ, AnalogRX, AnalogRZ
from qadence import VariationalParameter, BackendName
from Utilities import cost_exp, deviation

def qaoa (Q: np.ndarray, coords: np.ndarray, layers: int, trials:int)->dict:
    detuning = np.diag(Q)
    def rz_layer(idx, detuning):
        ops = []
        s = VariationalParameter(f"s{idx}")
        for i, _ in enumerate(detuning):
            ops.append(RZ(i, s + detuning[i]))
        return kron(*ops)

    def new_rx_layer(idx):
      t = VariationalParameter(f"t{idx}")
      return AnalogRX(t) 

    device = RydbergDevice(rydberg_level = 70)
    reg = Register.from_coordinates(coords, device_specs = device)

    block = chain(*[new_rx_layer(j) * rz_layer(j, detuning) for j in range(layers)])
    circuit = QuantumCircuit(reg, block)
    model = QuantumModel(circuit)
    initial_counts = model.sample({}, n_shots = 1000)[0]
    to_arr_fn = lambda bitstring: np.array(list(bitstring), dtype = int)
    cost_fn = lambda arr: arr.T @ Q @ arr
    param_names = list(model.vparams.keys())
    def set_model_params (param_values: dict):
        for name, param in zip(param_names, model.parameters()):
            param.fill_(param_values[name])

    def evaluate_cost(param_values: dict) -> float:
        samples = model.sample(param_values, n_shots=1000)[0]
        cost = sum(samples[key] * cost_fn(to_arr_fn(key)) for key in samples)
        return cost / sum(samples.values())

    study = optuna.create_study(direction = 'minimize', 
                                pruner = optuna.pruners.MedianPruner (n_warmup_steps = 2),
                                sampler = optuna.samplers.TPESampler(seed = 0)
                               )
    def objective(trial):
        param_values = {}
        for j in range(layers):
            param_values[f"s{j}"] = trial.suggest_float(f"s{j}", 0, np.pi)
            param_values[f"t{j}"] = trial.suggest_float(f"t{j}", -2*np.pi, 0)
        result = evaluate_cost(param_values)
        return result
    study.optimize(objective, n_trials = trials)
    set_model_params(study.best_params)
    optimal_counts = model.sample(study.best_params, n_shots=1000)[0]
    return optimal_counts
