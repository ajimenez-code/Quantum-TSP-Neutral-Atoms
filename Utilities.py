#!/usr/bin/env python
# coding: utf-8

# In[4]:


from Embedding import brute_force_finding
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches

def possible_routes (Q: np.ndarray)-> list:
    """
    Finds exhaustively all the possibe routs ina given TSP.

    Parameters
    ----------
    Q: np.ndarray
        The QUBO matrix in which the target TSP is enconded.

    Returns
    -------
    routs: list
        List with all the possible (n-1)! routs, ordered by cost.
    """
    n = int(np.sqrt(len(Q))+1)
    possible = math.factorial(n-1)
    sols = brute_force_finding(Q)
    routs = []
    for i in range (possible):
        routs.append(sols[i][0])
    return routs


def cost_fun(bitstring: str, Q: np.ndarray)-> float:
    """
    Returns the Q's QUBO cost for an especified bitstring.
    Parameters
    ----------
    bitstring: str
        Binary string containing the target state.
    Q: np.ndarray
        QUBO's matrix.

    Returns
    -------
    cost: float
        The coast associated with the bitstring.
    """
    z = np.array(list(bitstring), dtype=int)
    cost = z.T @ Q @ z
    return cost



def cost_exp(counter: dict, Q: np.ndarray)-> int:
    """
    Returns the expected QUBO cost for a set of bitstrings and their frequency.
    Parameters
    -----------
    counter: dict
        The list of bitstrings andtheir frequency.
    Q: np.ndarray
        QUBO's matrix.

    Returns 
    -------
    exp: float
        The expected cost.
    """
    cost = sum(counter[key] * cost_fun(key, Q) for key in counter)
    exp = cost / sum(counter.values()) 
    return exp



def plot_distribution(C: dict, Q: np.ndarray, J: np.ndarray, name: str)-> None:
    """
    Plots the distribution for the bitstrings sampled using pusho, qaa or qaoa.
    Parameters
    ----------
    C: dict
        The list of the bitstrings sampled 1000 times and their frequencies.
    Q: np.ndarray
        The problem's QUBO matrix.
    J: np.ndarray
        The Van der Waals embedding matrix.
    name: str
        Name of the file saved.
    """
    solsQ = possible_routes(Q)
    np.fill_diagonal(J, np.diag(Q))
    solsJ = brute_force_finding(J)
    indexes = []
    other_routes = []
    for i in range (len(solsQ)):
        if i<2:
            indexes.append(''.join(map(str, solsQ[i])))
        else:
            other_routes.append(''.join(map(str, solsQ[i]))) 
    z = []
    for i in range (len(Q)):
        z.append(0)
    no_travel = [''.join(map(str, z))]
    new_indexes = [''.join(map(str, solsJ[0][0]))]
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))

    color_dict = {key: "r" if key in indexes else "b" if key in new_indexes
                  else 'c' if key in other_routes else 'k' if key in no_travel
                  else "g" for key in C}

    plt.figure(figsize=(8, 5))
    plt.xlabel("Bitstrings",fontsize=17)
    plt.ylabel("Counts",fontsize=17)
    plt.xticks([]) 

    plt.bar(C.keys(), C.values(), color=color_dict.values())

    red_patch = mpatches.Patch(color = 'r', label = 'Q solutions')
    blue_patch = mpatches.Patch(color = 'b', label = 'J solution')
    cian_patch = mpatches.Patch(color = 'c',label = 'Other routes')
    black_patch = mpatches.Patch(color = 'k', label = 'No travel')
    
    plt.legend(handles = [red_patch, blue_patch, cian_patch, black_patch], loc = 'upper right', fontsize = 15)
    plt.savefig(f"{name}", dpi=300, bbox_inches='tight')
    print(f"Total bitstrings: {len(C)}")


def deviation(count_dict: dict, Q: np.ndarray)-> float:
    """
    Computes the standard deviation for a dict with the counts for each bitstring.
    Parameters
    -----------
    count_dict: dict
        Dictionary with the bitstrings and their counts.
    Q: np.ndarray
        The problem's QUBO matrix
    Returns
    -------
    dev: float
        Standard deviation

    """
    exp_cost = cost_exp(count_dict, Q)
    var = 0
    for state, freq in count_dict.items():
        z = np.array([int(bit) for bit in state])
        cost = z.T @ Q @ z
        prob = freq / sum(count_dict.values())
        var += prob * ((exp_cost - cost)**2)
    dev = np.sqrt(var)
    return dev


# In[ ]:




