import argparse
import os
import sys
import random
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx

import GEMFsim
from models import SIS, HeapTauLeaping_v1


class NXNetwork:
    def __init__(self, edge_index: torch.Tensor, num_node: int):
        self.edge_index = edge_index
        self._num_node = num_node

    @property
    def num_node(self) -> int:
        return self._num_node


def build_nx_graph(kind: str, n: int, seed: int):
    if kind == "BarabasiAlbert":
        return nx.barabasi_albert_graph(n, 4, seed=seed)
    if kind == "WattsStrogatz":
        return nx.watts_strogatz_graph(n, k=8, p=0.1, seed=seed)
    raise ValueError(f"Unknown network kind: {kind}")


def resample_history(times: np.ndarray, counts: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.zeros((counts.shape[0], len(grid)))
    indices = np.searchsorted(times, grid, side="right") - 1
    indices = np.clip(indices, 0, counts.shape[1] - 1)
    return counts[:, indices]


def run_gemfsim(nx_graph, n, tmax, initial_states):
    para = GEMFsim.Para_SIS(1.0, 0.23)
    net = GEMFsim.NetCmbn([GEMFsim.MyNet(nx_graph)])
    stopcond = ["RunTime", tmax]
    ts, n_index, i_index, j_index = GEMFsim.GEMF_SIM(
        para, net, initial_states.copy(), stopcond, n
    )
    times, state_count = GEMFsim.Post_Population(
        initial_states.copy(), para[0], n, ts, i_index, j_index
    )
    return np.array(times), state_count


def run_heap_tau(network, initial_states, tmax, max_steps, theta, kmax, device):
    model = SIS()
    sim = HeapTauLeaping_v1(
        initial_state=initial_states.clone(),
        spreading_model=model,
        network=network,
        theta=theta,
        K_MAX=kmax,
    )
    times = [0.0]
    counts = [sim.count_by_state]
    for _ in range(max_steps):
        tau, _ = sim.step()
        if tau != tau or not np.isfinite(tau):
            break
        if sim.current_time >= tmax:
            break
        times.append(sim.current_time)
        counts.append(sim.count_by_state)
    return np.array(times), np.stack(counts).T


def run_fastgemf(nx_graph, initial_states, tmax, seed):
    fastgemf_src = os.path.join(os.path.dirname(__file__), "fastgemf", "src")
    if fastgemf_src not in sys.path:
        sys.path.insert(0, fastgemf_src)
    import fastgemf as fg
    from fastgemf.post_population import post_population

    np.random.seed(seed)
    random.seed(seed)

    sis_model = (
        fg.ModelSchema("SIS")
        .define_compartment(['S', 'I'])
        .add_network_layer('contact_network')
        .add_node_transition(
            name='recovery',
            from_state='I',
            to_state='S',
            rate='delta'
        )
        .add_edge_interaction(
            name='infection',
            from_state='S',
            to_state='I',
            inducer='I',
            network_layer='contact_network',
            rate='beta'
        )
    )

    contact_network_csr = nx.to_scipy_sparse_array(nx_graph)
    sis_instance = (
        fg.ModelConfiguration(sis_model)
        .add_parameter(beta=0.23, delta=1.0)
        .get_networks(contact_network=contact_network_csr)
    )

    sim = fg.Simulation(
        sis_instance,
        initial_condition={'exact': initial_states},
        stop_condition={'time': tmax},
        nsim=1
    )
    sim.run()
    times, state_count, *_ = post_population(
        sim.setup.X0, sim.setup.model_matrices, sim.setup.event_data, sim.setup.networks.nodes
    )
    return np.array(times), state_count


def main():
    parser = argparse.ArgumentParser(description="Compare trajectories across simulators.")
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--tmax", type=float, default=5.0)
    parser.add_argument("--grid", type=int, default=200)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--theta", type=float, default=5.0)
    parser.add_argument("--kmax", type=int, default=200)
    parser.add_argument("--initial_infected", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--network", type=str, default="BarabasiAlbert",
                        choices=["BarabasiAlbert", "WattsStrogatz"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output", type=str, default="trajectory_compare.csv")
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    nx_graph = build_nx_graph(args.network, args.N, args.seed)
    edge_index = from_networkx(nx_graph).edge_index.to(device)
    network = NXNetwork(edge_index=edge_index, num_node=args.N)

    initial_states = np.zeros(args.N, dtype=int)
    infected = np.random.permutation(args.N)[:args.initial_infected]
    initial_states[infected] = 1

    grid = np.linspace(0, args.tmax, args.grid)

    gemf_times, gemf_counts = run_gemfsim(nx_graph, args.N, args.tmax, initial_states)
    gemf_resampled = resample_history(gemf_times, gemf_counts, grid)

    heap_times, heap_counts = run_heap_tau(
        network,
        torch.tensor(initial_states, dtype=torch.int64, device=device),
        args.tmax,
        args.steps,
        args.theta,
        args.kmax,
        device,
    )
    heap_resampled = resample_history(heap_times, heap_counts, grid)

    fast_times, fast_counts = run_fastgemf(nx_graph, initial_states, args.tmax, args.seed)
    fast_resampled = resample_history(fast_times, fast_counts, grid)

    l1_heap = np.abs(heap_resampled - gemf_resampled).sum(axis=0)
    l1_fast = np.abs(fast_resampled - gemf_resampled).sum(axis=0)

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(grid, l1_heap, label="HeapTau vs GEMFsim", color="purple")
    plt.plot(grid, l1_fast, label="FastGEMF vs GEMFsim", color="orange")
    plt.xlabel("Time")
    plt.ylabel("L1 Difference (state counts)")
    plt.title(f"L1 Error vs Time ({args.network}, N={args.N})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    fig_path = f"figures/trajectory_compare_{args.network}_N{args.N}.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=200)
    print(f"Saved plot: {fig_path}")

    with open(args.output, "w") as f:
        f.write("time,l1_heap,l1_fast\n")
        for t, lh, lf in zip(grid, l1_heap, l1_fast):
            f.write(f"{t},{lh},{lf}\n")
    print(f"Saved CSV: {args.output}")


if __name__ == "__main__":
    main()

