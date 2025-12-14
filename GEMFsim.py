# Modified by Zeyu Xia, zeyu.xia@virginia.edu

##############################################################################
# Copyright (c) 2015, Network Science and Engineering Group (NetSE group)) at Kansas State University.
# http://ece.k-state.edu/sunflower_wiki/index.php/Main_Page
#
# Written by:
# Heman Shakeri:heman@ksu.edu
# All rights reserved.
#
# For details, see https://github.com/scalability-llnl/AutomaDeD
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU General Public License for more details.
##############################################################################
from matplotlib import animation
import numpy as np
import networkx as nx
from matplotlib.patches import Circle

# Initialization


def MyNet(G, weight=None):
    """
    MyNet(G, weight='weight')
    """
    G_adj = nx.to_scipy_sparse_array(G, weight=weight)  # type: ignore
    cx = G_adj.tocoo()
    L2 = cx.row
    L1 = cx.col
    W = cx.data
    N = G.number_of_nodes()
    NeighVec, I1, I2, d, NeighWeight = NeighborhoodData(N, L1, L2, W)
    if nx.is_directed(G):
        NNeighVec, NI1, NI2, Nd, NNeighWeight = NNeighborhoodData(
            N, L1, L2, W)  # ver2

        Net = [NeighVec, I1, I2, d, NeighWeight,
               NNeighVec, NI1, NI2, NNeighWeight]  # ver2
    else:
        Net = [NeighVec, I1, I2, d, NeighWeight]  # ver2
    return Net


def NetCmbn(NetSet):
    """
    Combine different network layers data. This function is used for directed networks.
    """
    is_directed = len(NetSet[0]) > 5

    if is_directed:
        # Extract elements for directed networks
        elements = list(zip(*NetSet))
        Net = [list(elem) for elem in elements]
    else:
        # Extract elements for undirected networks
        elements = list(zip(*NetSet))
        Net = [list(elem) for elem in elements]

    return Net


def NeighborhoodData(N, L1, L2, W):
    """
    A DB that gives the adjacent nodes (not necessary neighbors). For directed graphs we need NNeighborhoodData too.
    (Heman)
    """
    sorted_indices = np.argsort(L1)
    sorted_L1 = L1[sorted_indices]
    NeighVec = L2[sorted_indices]
    NeighWeight = W[sorted_indices]

    # Vectorized computation of node degrees and indices
    unique_nodes, first_indices, counts = np.unique(
        sorted_L1, return_index=True, return_counts=True)

    # Initialize arrays
    I1 = -np.ones(N, dtype=int)
    d = np.zeros(N, dtype=int)

    # Vectorized assignment using fancy indexing
    I1[unique_nodes] = first_indices
    d[unique_nodes] = counts - 1  # degree is count - 1 for the indexing scheme

    I2 = I1 + d
    return NeighVec, I1, I2, d, NeighWeight


def NNeighborhoodData(N, L1, L2, W):
    """
    A DB that gives the adjacent nodes (not necessary neighbors). Useful only for directed graphs.
    (Heman)
    """
    sorted_indices = np.argsort(L2)
    sorted_L2 = L2[sorted_indices]
    NNeighVec = L1[sorted_indices]
    NNeighWeight = W[sorted_indices]

    # Vectorized computation of node degrees and indices
    unique_nodes, first_indices, counts = np.unique(
        sorted_L2, return_index=True, return_counts=True)

    # Initialize arrays
    NI1 = -np.ones(N, dtype=int)
    Nd = np.zeros(N, dtype=int)

    # Vectorized assignment using fancy indexing
    NI1[unique_nodes] = first_indices
    Nd[unique_nodes] = counts - 1  # degree is count - 1 for the indexing scheme

    NI2 = NI1 + Nd
    return NNeighVec, NI1, NI2, Nd, NNeighWeight


# Parameters

def Para_SIS(delta, beta):
    # delta: recovery rate=1
    # beta: infection rate=0.23
    M = 2
    q = np.array([1])
    L = len(q)
    A_d = np.zeros((M, M))
    A_d[1, 0] = delta
    A_b = [np.zeros((M, M)) for _ in range(L)]
    A_b[0][0, 1] = beta

    return [M, q, L, A_d, A_b]


def Para_SIR(delta, beta):
    # delta: recovery rate=1
    # beta: infection rate=0.05
    M = 3
    q = np.array([1])
    L = len(q)
    A_d = np.zeros((M, M))
    A_d[1, 2] = delta

    A_b = [np.zeros((M, M)) for _ in range(L)]
    A_b[0][0, 1] = beta

    return [M, q, L, A_d, A_b]


def Para_SEIR(delta, beta, Lambda):
    # delta: recovery rate=0.2
    # beta: infection rate=0.3
    # Lambda: rate of exposed to infected=0.2
    M = 4
    q = np.array([2])
    L = len(q)
    A_d = np.zeros((M, M))
    A_d[1, 2] = Lambda
    A_d[2, 3] = delta  # Lambda
    A_b = [np.zeros((M, M)) for _ in range(L)]
    A_b[0][0, 1] = beta  # [l][M][M]

    return [M, q, L, A_d, A_b]


def Para_SAIS_Single(delta, beta, beta_a, kappa):
    # delta: recovery rate=1
    # beta: infection rate=2
    # beta_a: infection rate of alert individuals=0.4
    # kappa: rate of susceptible to alert=0.2
    M = 3
    q = np.array([1])
    L = len(q)
    A_d = np.zeros((M, M))
    A_d[1, 0] = delta
    A_b = [np.zeros((M, M)) for _ in range(L)]
    A_b[0][0, 1] = beta  # [l][M][M]
    A_b[0][0, 2] = kappa
    A_b[0][2, 1] = beta_a

    return [M, q, L, A_d, A_b]


def Para_SAIS(delta, beta, beta_a, kappa, mu):
    M = 3
    q = np.array([1, 1])
    L = len(q)
    A_d = np.zeros((M, M))
    A_d[1, 0] = delta
    A_b = [np.zeros((M, M)) for _ in range(L)]
    A_b[0][0, 1] = beta  # [l][M][M]
    A_b[0][0, 2] = kappa
    A_b[1][2, 1] = beta_a
    A_b[1][0, 2] = mu

    return [M, q, L, A_d, A_b]


def Para_SI1I2S(delta1, delta2, beta1, beta2):
    M = 3
    q = np.array([1, 2])
    L = len(q)
    A_d = np.zeros((M, M))
    A_d[1, 0] = delta1
    A_d[2, 0] = delta2
    A_b = [np.zeros((M, M)) for _ in range(L)]
    A_b[0][0, 1] = beta1  # [l][M][M]
    A_b[1][0, 2] = beta2  # [l][M][M]

    return [M, q, L, A_d, A_b]


def Initial_Cond_Gen(N, J, NJ, x0):
    """
    J = initial state for NJ number of whole population N
    Example :   x0 = np.zeros(N, dtype = int)
                Initial_Cond_Gen(10, Para[1][0], 2, x0)
    """
    total_infected = np.sum(NJ)
    if total_infected > N:
        return 'Oops! Initial infection is more than the total population'

    infected_indices = np.random.choice(N, size=total_infected, replace=False)
    x0[infected_indices] = J
    return x0

# Simulations


def GEMF_SIM(Para, Net, x0, StopCond, N):
    """
    An event-driven approach to simulate the stochastic process.

    """
    M = Para[0]
    q = Para[1]
    L = Para[2]
    A_d = Para[3]
    A_b = Para[4]
    Neigh = Net[0]
    I1 = Net[1]
    I2 = Net[2]
    NeighW = Net[4]

    n_index = []
    j_index = []
    i_index = []
    # Pre-compute arrays efficiently using vectorized operations
    # bil[i,l] = sum over j of A_b[l][i,j] (row sums for each layer)
    bil = np.stack([A_b[l].sum(axis=1) for l in range(L)], axis=1)

    # bi[i,j,l] = A_b[l][i,j] (reorganized for efficient access)
    bi = np.stack(A_b, axis=2)

    # The rate that we leave compartment i, due to nodal transitions
    di = A_d.sum(axis=1)
    # ------------------------------
    X = x0.astype(int)

    # ------------------------------
    Nq = np.zeros((L, N))
    # Vectorized computation of neighbor influences
    for l in range(L):
        for n in range(N):
            if I1[l][n] >= 0:  # Only process nodes with neighbors
                start_idx, end_idx = I1[l][n], I2[l][n] + 1
                neighbor_indices = Neigh[l][start_idx:end_idx]
                neighbor_weights = NeighW[l][start_idx:end_idx]
                # Vectorized comparison and weighted sum
                Nq[l, n] = np.dot(
                    (X[neighbor_indices] == q[l]), neighbor_weights)
    # ------------------------------ver2
    Rn = di[X] + np.sum(bil[X, :] * Nq.T, axis=1)
    R = np.sum(Rn)
    # ------------------------------
    RunTime = StopCond[1]
    ts = []
#     #------------------------------
    s = -1
    Tf = 0
    # if len(Net) > 5:
    #     NNeigh = Net[5]
    #     NI1 = Net[6]
    #     NI2 = Net[7]
    #     NNeighW = Net[8]
    #     while Tf < RunTime:
    #         s += 1
    #         ts.append(-np.log(np.random.rand())/R)
    #         # ------------------------------ver 2
    #         ns = rnd_draw(Rn)
    #         iss = X[ns]

    #         # Vectorized transition probability calculation
    #         transition_probs = A_d[iss, :] + np.dot(bi[iss], Nq[:, ns])
    #         js = rnd_draw(transition_probs.ravel())

    #         # Store transition data
    #         n_index.append(ns)
    #         j_index.append(js)
    #         i_index.append(iss)

    #         # Update state and rates
    #         X[ns] = js
    #         old_rate = Rn[ns]
    #         new_rate = di[js] + np.dot(bil[js, :], Nq[:, ns])
    #         Rn[ns] = new_rate
    #         R += new_rate - old_rate

    #         # Vectorized influence layer processing
    #         influenced_layers = np.where(q == js)[0]
    #         for l in influenced_layers:
    #             if NI1[l][ns] >= 0:  # Check if node has neighbors in this layer
    #                 start_idx, end_idx = NI1[l][ns], NI2[l][ns] + 1
    #                 neighbor_nodes = NNeigh[l][start_idx:end_idx]
    #                 edge_weights = NNeighW[l][start_idx:end_idx]

    #                 # Vectorized updates
    #                 Nq[l, neighbor_nodes] += edge_weights
    #                 rate_changes = bil[X[neighbor_nodes], l] * edge_weights
    #                 Rn[neighbor_nodes] += rate_changes
    #                 R += np.sum(rate_changes)

    #         # Process nodes losing influence
    #         losing_influence_layers = np.where(q == iss)[0]
    #         for l in losing_influence_layers:
    #             if NI1[l][ns] >= 0:  # Check if node has neighbors in this layer
    #                 start_idx, end_idx = NI1[l][ns], NI2[l][ns] + 1
    #                 neighbor_nodes = NNeigh[l][start_idx:end_idx]
    #                 edge_weights = NNeighW[l][start_idx:end_idx]

    #                 # Vectorized updates
    #                 Nq[l, neighbor_nodes] -= edge_weights
    #                 rate_changes = bil[X[neighbor_nodes], l] * edge_weights
    #                 Rn[neighbor_nodes] -= rate_changes
    #                 R -= np.sum(rate_changes)
    #         if R < 1e-6:
    #             break
    #         Tf += ts[s]
    # else:
    while Tf < RunTime:
        s += 1
        ts.append(-np.log(np.random.rand())/R)
        # ------------------------------ver 2
        ns = rnd_draw(Rn)
        iss = X[ns]

        # Vectorized transition probability calculation
        transition_probs = A_d[iss, :] + np.dot(bi[iss], Nq[:, ns])
        js = rnd_draw(transition_probs.ravel())

        # Store transition data
        n_index.append(ns)
        j_index.append(js)
        i_index.append(iss)

        # Update state and rates
        X[ns] = js
        old_rate = Rn[ns]
        new_rate = di[js] + np.dot(bil[js, :], Nq[:, ns])
        Rn[ns] = new_rate
        R += new_rate - old_rate

        # Vectorized influence layer processing
        influenced_layers = np.where(q == js)[0]
        for l in influenced_layers:
            if I1[l][ns] >= 0:  # Check if node has neighbors in this layer
                start_idx, end_idx = I1[l][ns], I2[l][ns] + 1
                neighbor_nodes = Neigh[l][start_idx:end_idx]
                edge_weights = NeighW[l][start_idx:end_idx]

                # Vectorized updates
                Nq[l, neighbor_nodes] += edge_weights
                rate_changes = bil[X[neighbor_nodes], l] * edge_weights
                Rn[neighbor_nodes] += rate_changes
                R += np.sum(rate_changes)

        # Process nodes losing influence
        losing_influence_layers = np.where(q == iss)[0]
        for l in losing_influence_layers:
            if I1[l][ns] >= 0:  # Check if node has neighbors in this layer
                start_idx, end_idx = I1[l][ns], I2[l][ns] + 1
                neighbor_nodes = Neigh[l][start_idx:end_idx]
                edge_weights = NeighW[l][start_idx:end_idx]

                # Vectorized updates
                Nq[l, neighbor_nodes] -= edge_weights
                rate_changes = bil[X[neighbor_nodes], l] * edge_weights
                Rn[neighbor_nodes] -= rate_changes
                R -= np.sum(rate_changes)
        if R < 1e-6:
            break
        Tf += ts[s]

    return ts, n_index, i_index, j_index


def Post_Population(x0, M, N, ts, i_index, j_index):
    # Initialize state matrix more efficiently
    X0 = np.zeros((M, N))
    X0[x0.astype(int), np.arange(N)] = 1

    # More efficient time array construction
    T = [0.0] + list(np.cumsum(ts))

    # Initialize state count matrix
    StateCount = np.zeros((M, len(ts) + 1))
    StateCount[:, 0] = X0.sum(axis=1)

    # Vectorized state transitions
    if len(i_index) > 0:
        # Create transition matrices more efficiently
        transitions = np.zeros((M, len(ts)))
        transitions[i_index, np.arange(len(i_index))] = -1
        transitions[j_index, np.arange(len(j_index))] = 1

        # Cumulative sum for state evolution
        StateCount[:, 1:] = StateCount[:, 0:1] + np.cumsum(transitions, axis=1)

    return T, StateCount


def MonteCarlo(Net, Para, StopCond, Init_inf, M, step, nsim, N):
    # M: number of compartments
    # N: the entire population size
    # step: the chosen time step

    tsize = int(StopCond[1] / step)
    t_interval = np.linspace(0, StopCond[1], num=tsize)
    f = np.zeros((M, tsize))

    # Pre-allocate arrays for better memory efficiency
    simulation_results = np.zeros((nsim, M, tsize))

    for n in range(nsim):
        x0 = Initial_Cond_Gen(
            N, Para[1][0], Init_inf, x0=np.zeros(N, dtype=int))

        # finding the events
        ts, n_index, i_index, j_index = GEMF_SIM(Para, Net, x0, StopCond, N)

        # the history for population of each compartment
        T, StateCount = Post_Population(x0, M, N, ts, i_index, j_index)

        # More efficient time interpolation using vectorized operations
        T_array = np.array(T + [1000.0])

        # Vectorized interpolation for all time points at once
        indices = np.searchsorted(T_array, t_interval, side='right') - 1
        indices = np.clip(indices, 0, StateCount.shape[1] - 1)

        # Vectorized normalization
        simulation_results[n] = StateCount[:, indices] / N

    # Mean across all simulations
    # f: the averaged population of each compartment in the time step
    f = np.mean(simulation_results, axis=0)

    return t_interval, f


def rnd_draw(p):
    """
    Optimized random sampling using cumulative distribution.
    """
    p = np.atleast_1d(p)  # Ensure p is at least 1D
    p_sum = np.sum(p)

    if p_sum <= 1e-15:  # More robust zero check
        return 0

    # More efficient normalization and sampling
    return np.searchsorted(np.cumsum(p), np.random.rand() * p_sum)

# Output


def animate_discrete_property_over_graph(g, model, steps, fig, n_index, i_index, j_index, comp, property=None,
                                         color_mapping=None, pos=None, Node_radius=None, **kwords):
    """Draw a graph and animate the progress of a property over it. The
    property values are converted to colours that are then used to colour
    the nodes.
    """
    x0 = model[0]
    n_index = model[1]
    i_index = model[2]
    j_index = model[3]

    # manipulate the axes, since this isn't a data plot
    ax = fig.gca()

    # pos
    if pos is None:
        pos = nx.spring_layout(g)
    ax.grid(False)                # no grid
    ax.get_xaxis().set_ticks([])  # no ticks on the axes
    ax.get_yaxis().set_ticks([])
    nx.draw_networkx_edges(g, pos)

    if Node_radius is None:
        Node_radius = 0.02

    # draw the graph, keeping hold of the node markers
    nodeMarkers = []
    for v in g.nodes():
        circ = Circle(pos[v], radius=Node_radius, zorder=2)
        ax.add_patch(circ)
        nodeMarkers.append({'node_key': v, 'marker': circ})

    # initialisation colours the markers according to the current
    # state of the property being tracked
    def colour_nodes():
        default_colors = ['red', 'green', 'blue',
                          'yellow', 'orange', 'purple', 'gray', 'black']
        for nm in nodeMarkers:
            v = nm['node_key']
            state = g.nodes[v][property]
            if color_mapping is not None:
                c = color_mapping[state]
            else:
                # fallback: use a default color map
                c = default_colors[state % len(default_colors)]
            nm['marker'].set_color(c)

    def init_state():
        """Initialise all nodes in the graph to their initial states."""
        # Vectorized state initialization
        node_keys = list(g.nodes.keys())
        initial_states = x0[node_keys].astype(int)

        for node, state in zip(node_keys, initial_states):
            g.nodes[node]['state'] = comp[state]

        colour_nodes()
        return [nm['marker'] for nm in nodeMarkers]

    # per-frame animation just iterates the model and then colours it
    # to reflect the changed property status of each node

    def frame(i):
        changing_node = n_index[i]
        new_comp = j_index[i]
        g.nodes[changing_node]['state'] = comp[new_comp]
        colour_nodes()
        # Return a list of all node marker patches
        return [nm['marker'] for nm in nodeMarkers]

    # return the animation with the functions etc set up
    return animation.FuncAnimation(fig, frame, init_func=init_state, frames=steps, **kwords)
