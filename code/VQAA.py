#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pulser
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import pulser_simulation
from Utilities import cost_exp

def pusho (cuts: int, T: float, coords: np.ndarray , Q: np.ndarray, J: np.ndarray, trials: int, warmup_trials: int)-> dict:
    """
    Runs the PUSHO algorithm.
    Parameters
    ----------
    cuts: int
        The number of cuts to perform to the pulse, which will be divided in cuts + 1 intervals. Sets the number of parametrs to optimize: 2*cuts.
    T: float
        The duration of the pulse, in ns.
    coords: np.ndarray
        The positions of the qubits used.
    Q: np.ndarray
        The problem's QUBO matrix.
    J: np.ndarray
        The embedding matrix.
    trials: int
        The number of trials to perferm in the Bayesian Optimization.
    warmup_trials: int
        The number of warmup trials to be performed.

    Returns
    --------
    count_dict: dict
        The list of the bitstrings sampled 1000 times and their frequencies.
    """

    def min_parameters(Omega, Delta):
        sequence = pulser.Sequence(reg, device)
        sequence.declare_channel("rydberg_global", "rydberg_global")
        sequence.config_detuning_map(det_map, "dmm_0")
        pulse = pulser.Pulse(pulser.InterpolatedWaveform(T, Omega), pulser.InterpolatedWaveform(T, Delta), 0,)
        sequence.add(pulse, "rydberg_global")
        sequence.add_dmm_detuning(pulser.ConstantWaveform(T, -maxDelta), "dmm_0")
        simul = pulser_simulation.QutipBackendV2(sequence)
        results = simul.run()
        count_dict = results.final_bitstrings
        return cost_exp(count_dict, Q)

    def objective(trial):
        midOmega = np.array([trial.suggest_float(f'a1_{i}', 0, maxOmega) for i in range(1, cuts+1)])
        Omega = np.concatenate((np.array([0]), midOmega, np.array([0])), axis = 0)
        midDelta = np.array([trial.suggest_float(f'a2_{i}', -maxDelta, maxDelta) for i in range(1, cuts+1)])
        Delta = np.concatenate((np.array([-maxDelta]), midDelta, np.array([maxDelta])), axis = 0)
        return min_parameters(Omega, Delta)

    device = pulser.MockDevice
    qubits = {f"q{i}": coord for (i, coord) in enumerate(coords)}
    reg = pulser.Register(qubits)
    tol = 1e-6
    maxOmega = 3*np.median(J[J> 0].flatten())
    maxDelta = 2*maxOmega
    weights = np.abs(np.diag(Q))
    norm_weights = weights / np.max(weights)
    map_weights = 1 - norm_weights
    det_map = reg.define_detuning_map({f"q{i}": map_weights[i] for i in range(len(map_weights))})
    seed_value = 0
    study = optuna.create_study(direction = 'minimize', pruner = optuna.pruners.MedianPruner(n_warmup_steps = warmup_trials), sampler = optuna.samplers.TPESampler(seed = seed_value))
    study.optimize(objective, n_trials = trials)
    best_params = study.best_params
    best_Omega = np.array([best_params[f'a1_{i}'] for i in range(1, cuts+1)])
    best_Omega = np.concatenate((np.array([0]), best_Omega, np.array([0])), axis = 0)
    best_Delta = np.array([best_params[f'a2_{i}'] for i in range(1, cuts+1)])
    best_Delta = np.concatenate((np.array([-maxDelta]), best_Delta, np.array([maxDelta])), axis = 0)
    sequence = pulser.Sequence(reg, device)
    sequence.declare_channel("rydberg_global", "rydberg_global")
    sequence.config_detuning_map(det_map, "dmm_0")
    pulse = pulser.Pulse(pulser.InterpolatedWaveform(T, best_Omega),pulser.InterpolatedWaveform(T, best_Delta), 0,)
    sequence.add(pulse, "rydberg_global")
    sequence.add_dmm_detuning(pulser.ConstantWaveform(T, -maxDelta), "dmm_0")
    simul = pulser_simulation.QutipBackendV2(sequence)
    results = simul.run()
    count_dict = results.final_bitstrings
    return count_dict



def time_getter(Tmax: int, coords: np.ndarray, Q: np.ndarray, J: np.ndarray, name: str)-> None:
    """
    Runs an easy PUSHO algorithm for a set of times and plots Q's and J's expected cost.
    Parameters
    ----------
    Tmax: int
        The maximun time of the pulse.
    coords: np.ndarray
        The positions of the qubits used.
    Q: np.ndarray
        The problem's QUBO matrix.
    J: np.ndarray
        The embedding matrix.
    name: str
        Name of the file saved.
    """
    times = np.arange(1000, Tmax+1000, 1000)
    np.fill_diagonal(J, np.diag(Q))
    expected_Q = []
    expected_J = []
    for t in times:
        counts = pusho(1, t, coords, Q, J, 10, 2)
        expected_Q.append(cost_exp(counts, Q))
        expected_J.append(cost_exp(counts, J))
    plt.plot(times/1000, expected_Q, "--o", label = r'$\langle Q \rangle$')
    plt.plot(times/1000, expected_J, "--o", color = 'red', label = r'$\langle J \rangle$')
    plt.xlabel(r"Time ($\mu$s)", fontsize = 17)
    plt.ylabel('Expected cost', fontsize = 17)
    plt.legend(fontsize = 17)
    plt.savefig(f"{name}", dpi=300, bbox_inches='tight')
    return


