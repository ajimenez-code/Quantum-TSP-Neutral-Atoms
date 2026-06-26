#!/usr/bin/env python
# coding: utf-8

# In[142]:


"""
embedding.py
--------------------------
Utilities for embedding Q matrixes onto Pulser device layouts.

Functions
---------
    clean_Q   
    stress   
    is_metric     
    draw_register       
    brute_force_finding 
    embedding           
    plot_convergence     
    QUBO_plot            


Dependencies
------------
    TSP_QUBO_Formulation
    numpy
    pulser
    scipy
    matplotlib
    scipy
    itertools

"""

from TSP_QUBO_Formulation import create_TSP, QUBO
import numpy as np
import matplotlib.pyplot as plt
import pulser
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from itertools import product 

def clean_Q (Q: np.ndarray, eps= 1e-2) -> np.ndarray:
    """
    Cleans the zero non-diagonal and the diagonal terms of Q, so that a distance matrix can be defined.

    Parameters
    ----------
    Q: np.ndarray
        The QUBO matrix to be cleaned.
    eps: float
        Min interaction value, defaults to 1e-2.

    Returns
    -------
    Q_off: np.ndarray
        The clean QUBO matrix.
    """
    Q_off = np.copy(Q)
    Q_off[Q_off<eps] = eps
    np.fill_diagonal(Q_off,0)
    return Q_off

def stress (Q: np.ndarray, J: np.ndarray) -> float:
    """
    Finds the Kruskal's STRESS-1 between Q (QUBO matrix) and J (embedding matrix).

    Parameters
    ----------
    Q : np.ndarray
        The QUBO matrix without the diagonal terms.

    J : np.ndarray
        The Van der Waals's embedding matrix.

    Returns
    ---------
    The Normalized stress between Q and J.

    """
    R = Q.ravel()
    d = J.ravel()
    stress_raw = ((R - d)**2).sum() 
    sum_sq = (d**2).sum()
    stress_1 = np.sqrt(stress_raw / sum_sq)
    return stress_1

def is_metric(Q: np.ndarray, eps: float = 1e-4) -> bool:
    """
    Checks if the distance matrix associated with the QUBO (D_ij=(C6/R_ij)**(1/6)) is metric.

    Parameters
    ----------
    Q : np.ndarray
        TSP associated QUBO matrix.

    eps : float
        Min interaction value.

    Returns
    ---------
    bool
        True if the matrix is metric,
        False if not.

    """
    Q_off = clean_Q(Q)
    d = Q_off ** (-1/6)
    np.fill_diagonal(d, 0)
    violations = (d[:, :, None] + d[None, :, :]) < d[:, None, :]
    return  not np.any(violations)

def draw_register (coords: np.ndarray) -> None:
    """
    Draws the qubit register for a set of atoms with given coordinates.

    Parameters
    ----------
    coords : np.ndarray
        An (2*n) array with the qubits's positions.

    """
    device = pulser.MockDevice
    qubits = {f"q{i}": coord for (i, coord) in enumerate(coords)}
    reg = pulser.Register(qubits)
    reg.draw (blockade_radius=device.rydberg_blockade_radius(1.0),
      draw_graph=True,
      draw_half_radius=True,)
    return

def brute_force_finding(Q: np.ndarray)-> dict:
    """
    Finds and sorts (from lower to higher) exhaustively the QUBO solution of a given matrix, Q.

    Parameters
    ----------
    Q: np.ndarray
        The target matrix.

    Returns
    -------
    sorted_energies: dict
        Dictionary with the ordered energies.
    """
    energies = {}
    for state in product([0, 1], repeat=len(Q)):
      z = np.array(state)
      energy = z.T @ Q @ z
      energies[state] = energy
    sorted_energies = sorted(energies.items(), key=lambda x: x[1])
    return sorted_energies

def embedding(Q: np.ndarray, device: pulser.devices.Device, seed: int, method: str, threshold: float = 1e-16)-> tuple[float, np.ndarray, np.ndarray, list]:
    """
    Embeds the QUBO matrix onto a qubit register, minimizing the Kruskal's STRESS-1.

    Parameters
    ----------
    Q: np.ndarray
        QUBO matrix to be embedded.

    device: pulser.devices.Device
        The device used, which will provide the Van der Waals coefficient C6.

    seed: int
        The seed used for the random inicialization.

    method: str
        The minimizing method to use. The ones supported are those in scipy.minimize.

    threshold: optional, default 1e-16
        The tolerance used.

    Returns
    -------
    error: float
        The final Kruskal's STRESSS-1 value.

    J: np.ndarray
        The embedded Van der Waals matrix.

    opt_coords: np.ndarray
        The optimal coordinates of the qubits.

    eval_history: list
        A list with the STRESS-1 values per evaluation.

    """
    Q_off = clean_Q(Q)
    np.random.seed(seed)
    eval_history = []

    def evaluate_mapping(new_coords: np.ndarray, Q_off: np.ndarray, device: pulser.devices.Device):
        new_coords = np.reshape(new_coords, (len(Q), 2))
        J = squareform(device.interaction_coeff / pdist(new_coords) ** 6)
        cost = stress(Q_off, J)
        eval_history.append(cost)
        return cost

    def evaluate_mapping_raw(new_coords, Q_off, device): 
        new_coords = np.reshape(new_coords, (len(Q), 2))
        J = squareform(device.interaction_coeff / pdist(new_coords) ** 6)
        return stress(Q_off, J)

    x0 = 20 * np.random.rand(len(Q), 2).flatten()

    res = minimize(
        evaluate_mapping, x0,
        args=(Q_off, device),
        method=method,
        tol=threshold,
        options={"maxfev": 60000, "maxiter": 20000},
    )

    opt_coords = np.reshape(res.x, (len(Q), 2))   
    J = squareform(device.interaction_coeff / pdist(opt_coords) ** 6)
    error = stress(Q_off, J)
    return error, J, opt_coords, eval_history

def plot_convergence(histories: dict, threshold: float, xlimit: float) -> plt.Figure:
    """
    Plots the STRESS-1 per evaluation for the different methods.

    Parameters
    ----------
    histories: dict
        A list with the STRESS-1 values per evaluation.
    threshold: float

    xlimit: float
        The limit of the x axis to plot.

    """
    max_evals = max(len(h) for h in histories.values())

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10.colors

    for (method, hist), color in zip(histories.items(), colors):
        hist = np.array(hist)

        conv_idx = None
        if threshold is not None:
            crossed = np.where(hist <= threshold)[0]
            if len(crossed):
                conv_idx = crossed[0]

        hist_truncated = hist[:conv_idx + 1] if conv_idx is not None else hist

        pad       = max_evals - len(hist_truncated)
        hist_plot = np.concatenate([hist_truncated, np.full(pad, hist_truncated[-1])])

        evals_plot = np.arange(1, max_evals + 1)
        ax.plot(evals_plot, hist_plot, label=method, color=color, lw=1.6)

        if conv_idx is not None:
            ax.plot(conv_idx + 1, hist[conv_idx], "o", ms=7, color=color, zorder=5)

    if threshold is not None:
        ax.axhline(threshold, color="k", ls="--", lw=1.2)

    ax.set_ylim(0.4, 1.1)
    ax.set_xlim(0, xlimit)
    ax.set_xlabel("Evals", fontsize=21)
    ax.set_ylabel("STRESS", fontsize=19)
    ax.legend(framealpha=0.9, fontsize=17)
    fig.tight_layout()
    return fig

def QUBO_plot(Q: np.ndarray, J: np.ndarray, name: str) -> None:
    """
    Plots the Q and J (with the Q diagonal) QUBO costs. The x axis represents the ordered bitstrings in regards to the Q costs.

    Parameters
    ----------
    Q: np.ndarray
        The target QUBO matrix.
    J: np.ndarray
        The obtained embedding matrix.
    name: str
        Name of the file to be saved.
    """
    np.fill_diagonal(J,np.diag(Q))
    sols = brute_force_finding(Q)
    new_cost=[]
    cost=[]
    x=[]
    for i in range(len(sols)):
      z=np.array(sols[i][0])
      new_cost.append(z.T @ J @ z)
      cost.append(sols[i][1])
      x.append(i)
    colors = ['red', 'green']
    plt.plot(x,new_cost, label= 'J')
    plt.plot(x,cost,label='Q',lw=2)
    labels = ['Js 1st minimum', 'Js 2nd minimum']
    min_indices = np.argsort(new_cost)[:2]
    for rank, idx in enumerate(min_indices):
        plt.scatter(x[idx], new_cost[idx], color=colors[rank], s=40, zorder=5,
                    label=f"{labels[rank]} = {idx:.0f}")
    plt.xlabel('Ordered bitstrings',fontsize=14)
    plt.ylabel('QUBO costs',fontsize=14)
    plt.legend(fontsize=11)
    plt.savefig(f"{name}", dpi=300, bbox_inches='tight')
    plt.show()
    return

