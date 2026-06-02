#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
TSP.py
---------------------
Creates the matrix Q whose QUBO represents the TSP.

Functions
---------------------


Dependencies
---------------------
numpy
matplotlib
networkx
docplex
qiskit_optimization

"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import random
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from docplex.mp.model import Model

def tsp_maker (n: int, seed: int, SHOW_PLOTS=False) -> np.ndarray:
    """
    Creates and represents the complete graph of size n of the TSP considered.

    Parameters
    ----------
    n : int
        size of the problem
        
    seed : int
        seed used
        
    SHOW_PLOTS : bool, optional
        If True, plots the cities and their labels. Defaults to False.
        
    Returns
    -------
    D : np.ndarray, shape (n,n)
        Distance matrix between cities
    """
    np.random.seed(seed)
    pos = {i: 10*np.round(np.random.rand(2),2) for i in range(n)}
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = round(np.linalg.norm(np.array(pos[i]) - np.array(pos[j])), 2)
            G.add_edge(i, j, weight = dist)
            
    D = np.zeros((n, n))
    for i, j, data in G.edges(data=True):
        D[i, j] = data["weight"]
        D[j, i] = data["weight"]
        
    if SHOW_PLOTS:
        fig, ax = plt.subplots(figsize=(7, 5))
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='#E63946',
                node_size=1000, edge_color='black', edgecolors='black',
                linewidths=1, font_weight='bold',font_color='white')
        ax.set_axis_on()
        edge_labels = nx.get_edge_attributes(G, 'weight')
        ax.set_axis_off()
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        plt.show()
    return D

def QUBO (D: np.ndarray, lam: float)-> np.ndarray:
    """
    Creates the QUBO matrix associated with the TSP, assuming the start and end city is 0.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix associated to the problem.
    lam: float
        Penalty used in the restrictions.

    Returns
    -------
    Q : np.ndarray, shape ((n-1)^2,(n-1)^2)
        Matrix associated with the TSP.
    """
    n = len(D)
    mdl = Model(name="TSP_Reduced")
    
    nodes = [i for i in range(1,n)]
    steps = [i for i in range(1,n)]
    
    x = mdl.binary_var_matrix(nodes, steps, name='x')
    
    # First step
    obj = mdl.sum(D[0, i] * x[i, 1]*x[i,1] for i in nodes)
    
    # Intermediate steps
    for p in range(len(steps)-1):
        curr_p = steps[p]
        next_p = steps[p+1]
        obj + = mdl.sum(D[i, j] * x[i, curr_p] * x[j, next_p] for i in nodes for j in nodes if i != j)
    
    # Return step
    obj + = mdl.sum(D[i, 0] * x[i, steps[-1]]*x[i, steps[-1]] for i in nodes)
    mdl.minimize(obj)
    
    # Restrictions
    
    # Each step is one city
    for p in steps:
        mdl.add_constraint(mdl.sum(x[i, p] for i in nodes) == 1)
    
    # Each city can only be visitted once
    for i in nodes:
        mdl.add_constraint(mdl.sum(x[i, p] for p in steps) == 1)
    
    # QUBO converting
    qp = from_docplex_mp(mdl)
    converter = QuadraticProgramToQubo(penalty=lam)
    qubo = converter.convert(qp)
    lineal = qubo.objective.linear.to_array()
    quadratic = qubo.objective.quadratic.to_array()
    qubo = quadratic + np.diag(lineal)
    diag = np.diag(np.diag(qubo))
    tri_upper = qubo - diag
    Q = diag + (tri_upper + tri_upper.T) / 2
    return Q

