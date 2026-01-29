import argparse
import os
import sys
import time
import random
import csv
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import from_networkx

from models import SIS, Gillespie_v1, SystemWiseTauLeaping_v1, HeapTauLeaping_v1

"Takeaways: Heap tau-leaping is faster than system-wise in this run and closer to Gillespie (lower L1), especially on WattsStrogatz. To firm up, increase runs and steps, maybe sweep theta/K_MAX."

# Benchmark summary (N=200, runs=2, steps=100):
#
# Network: BarabasiAlbert
#   Gillespie_v1             time=0.0056s ±0.0004s L1 vs Gillespie=0.00 ±0.00
#   SystemWiseTauLeaping_v1  time=0.0066s ±0.0000s L1 vs Gillespie=78.00 ±16.00
#   HeapTauLeaping_v1        time=0.0039s ±0.0020s L1 vs Gillespie=69.00 ±5.00
#
# Network: WattsStrogatz
#   Gillespie_v1             time=0.0049s ±0.0000s L1 vs Gillespie=0.00 ±0.00
#   SystemWiseTauLeaping_v1  time=0.0066s ±0.0001s L1 vs Gillespie=49.00 ±39.00
#   HeapTauLeaping_v1        time=0.0044s ±0.0017s L1 vs Gillespie=25.00 ±17.00

class NXNetwork:
    def __init__(self, edge_index: torch.Tensor, num_node: int):
        self.edge_index = edge_index
        self._num_node = num_node

    @property
    def num_node(self) -> int:
        return self._num_node


def build_nx_graph(kind: str, N: int, seed: int):
    if kind == "BarabasiAlbert":
        return nx.barabasi_albert_graph(N, 4, seed=seed)
    if kind == "WattsStrogatz":
        return nx.watts_strogatz_graph(N, k=8, p=0.1, seed=seed)
    raise ValueError(f"Unknown network kind: {kind}")


def run_single(sim_cls, spreading_model, network, initial_states, max_steps, tmax, theta, k_max):
    if sim_cls is Gillespie_v1:
        simulator = sim_cls(initial_state=initial_states.clone(),
                            spreading_model=spreading_model,
                            network=network)
    elif sim_cls is SystemWiseTauLeaping_v1:
        simulator = sim_cls(initial_state=initial_states.clone(),
                            spreading_model=spreading_model,
                            network=network,
                            theta=theta,
                            K_MAX=k_max)
    elif sim_cls is HeapTauLeaping_v1:
        simulator = sim_cls(initial_state=initial_states.clone(),
                            spreading_model=spreading_model,
                            network=network,
                            theta=theta,
                            K_MAX=k_max)
    else:
        raise ValueError(f"Unsupported simulator {sim_cls}")

    start = time.perf_counter()
    simulator.reset()
    tau = 0.0
    current_time = 0.0
    for _ in range(max_steps):
        tau, _ = simulator.step()
        if tau != tau or not np.isfinite(tau):  # NaN check
            break
        current_time = simulator.current_time
        if current_time >= tmax:
            break
    elapsed = time.perf_counter() - start
    final_counts = simulator.count_by_state
    return elapsed, final_counts


def run_fastgemf(nx_graph, initial_states, beta, gamma, tmax, seed):
    fastgemf_src = os.path.join(os.path.dirname(__file__), "fastgemf", "src")
    if fastgemf_src not in sys.path:
        sys.path.insert(0, fastgemf_src)
    try:
        import fastgemf as fg
        from fastgemf.post_population import post_population
    except Exception as exc:
        raise RuntimeError("FastGEMF import failed. Install its dependencies or check path.") from exc

    np.random.seed(seed)
    random.seed(seed)

    sir_model = (
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
    sir_instance = (
        fg.ModelConfiguration(sir_model)
        .add_parameter(beta=beta, delta=gamma)
        .get_networks(contact_network=contact_network_csr)
    )

    sim = fg.Simulation(
        sir_instance,
        initial_condition={'exact': initial_states.cpu().numpy()},
        stop_condition={'time': tmax},
        nsim=1
    )
    start = time.perf_counter()
    sim.run()
    elapsed = time.perf_counter() - start

    T, statecount, *_ = post_population(
        sim.setup.X0, sim.setup.model_matrices, sim.setup.event_data, sim.setup.networks.nodes
    )
    final_counts = statecount[:, -1]
    return elapsed, final_counts


def benchmark(args):
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    torch.manual_seed(args.seed)

    sims = [
        ("Gillespie_v1", Gillespie_v1),
        ("SystemWiseTauLeaping_v1", SystemWiseTauLeaping_v1),
        ("HeapTauLeaping_v1", HeapTauLeaping_v1),
        ("FastGEMF", None),
    ]
    nets = ["BarabasiAlbert", "WattsStrogatz"]

    results = {}

    for net_kind in nets:
        nx_graph = build_nx_graph(net_kind, args.N, args.seed)
        edge_index = from_networkx(nx_graph).edge_index.to(device)
        network = NXNetwork(edge_index=edge_index, num_node=args.N)
        results[net_kind] = {}
        for name, sim_cls in sims:
            runtimes = []
            l1_diffs = []
            for run in range(args.runs):
                seed = args.seed + run
                torch.manual_seed(seed)
                initial_states = torch.zeros(args.N, dtype=torch.int64, device=device)
                infected = torch.randperm(args.N, device=device)[:args.initial_infected]
                initial_states[infected] = 1
                model = SIS()  # simple 2-state model for speed/consistency

                # Baseline using Gillespie for accuracy comparison
                torch.manual_seed(seed)
                base_time, base_counts = run_single(Gillespie_v1, model, network,
                                                    initial_states, args.steps, args.tmax,
                                                    args.theta, args.kmax)

                # Target simulator
                if name == "FastGEMF":
                    sim_time, sim_counts = run_fastgemf(
                        nx_graph, initial_states, model._beta, model._gamma, args.tmax, seed
                    )
                else:
                    torch.manual_seed(seed)
                    sim_time, sim_counts = run_single(sim_cls, model, network,
                                                      initial_states, args.steps, args.tmax,
                                                      args.theta, args.kmax)

                runtimes.append(sim_time)
                l1 = np.abs(sim_counts - base_counts).sum()
                l1_diffs.append(l1)

            results[net_kind][name] = {
                "mean_time": float(np.mean(runtimes)),
                "std_time": float(np.std(runtimes)),
                "mean_l1": float(np.mean(l1_diffs)),
                "std_l1": float(np.std(l1_diffs)),
            }

    print("\nBenchmark summary (N={}, runs={}, steps={}, tmax={}):".format(
        args.N, args.runs, args.steps, args.tmax
    ))
    for net_kind, data in results.items():
        print(f"\nNetwork: {net_kind}")
        for name, stats in data.items():
            print(f"  {name:24s} time={stats['mean_time']:.4f}s ±{stats['std_time']:.4f}s "
                  f"L1 vs Gillespie={stats['mean_l1']:.2f} ±{stats['std_l1']:.2f}")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "network", "algorithm", "mean_time_s", "std_time_s",
                "mean_l1", "std_l1", "N", "runs", "steps", "tmax"
            ])
            for net_kind, data in results.items():
                for name, stats in data.items():
                    writer.writerow([
                        net_kind, name, stats["mean_time"], stats["std_time"],
                        stats["mean_l1"], stats["std_l1"],
                        args.N, args.runs, args.steps, args.tmax
                    ])
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tau-leaping variants vs Gillespie.")
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--theta", type=float, default=5.0)
    parser.add_argument("--tmax", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    parser.add_argument("--kmax", type=int, default=50)
    parser.add_argument("--initial_infected", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    benchmark(args)

