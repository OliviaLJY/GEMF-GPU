import argparse
import time
import numpy as np
import torch

from models import SIS, Gillespie_v1, SystemWiseTauLeaping_v1, HeapTauLeaping_v1
from network import BarabasiAlbert, WattsStrogatz


def build_network(kind: str, N: int, device: str):
    if kind == "BarabasiAlbert":
        return BarabasiAlbert(num_node=N, num_edge=4)
    if kind == "WattsStrogatz":
        return WattsStrogatz(num_node=N, k=8, p=0.1)
    raise ValueError(f"Unknown network kind: {kind}")


def run_single(sim_cls, spreading_model, network, initial_states, max_steps, theta, k_max, device):
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
    for _ in range(max_steps):
        tau, _ = simulator.step()
        if tau != tau or not np.isfinite(tau):  # NaN check
            break
    elapsed = time.perf_counter() - start
    final_counts = simulator.count_by_state
    return elapsed, final_counts


def benchmark(args):
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    torch.manual_seed(args.seed)

    sims = [
        ("Gillespie_v1", Gillespie_v1),
        ("SystemWiseTauLeaping_v1", SystemWiseTauLeaping_v1),
        ("HeapTauLeaping_v1", HeapTauLeaping_v1),
    ]
    nets = ["BarabasiAlbert", "WattsStrogatz"]

    results = {}

    for net_kind in nets:
        network = build_network(net_kind, args.N, device)
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
                                                    initial_states, args.steps,
                                                    args.theta, args.kmax, device)

                # Target simulator
                torch.manual_seed(seed)
                sim_time, sim_counts = run_single(sim_cls, model, network,
                                                  initial_states, args.steps,
                                                  args.theta, args.kmax, device)

                runtimes.append(sim_time)
                l1 = np.abs(sim_counts - base_counts).sum()
                l1_diffs.append(l1)

            results[net_kind][name] = {
                "mean_time": float(np.mean(runtimes)),
                "std_time": float(np.std(runtimes)),
                "mean_l1": float(np.mean(l1_diffs)),
                "std_l1": float(np.std(l1_diffs)),
            }

    print("\nBenchmark summary (N={}, runs={}, steps={}):".format(args.N, args.runs, args.steps))
    for net_kind, data in results.items():
        print(f"\nNetwork: {net_kind}")
        for name, stats in data.items():
            print(f"  {name:24s} time={stats['mean_time']:.4f}s ±{stats['std_time']:.4f}s "
                  f"L1 vs Gillespie={stats['mean_l1']:.2f} ±{stats['std_l1']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tau-leaping variants vs Gillespie.")
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--theta", type=float, default=5.0)
    parser.add_argument("--kmax", type=int, default=50)
    parser.add_argument("--initial_infected", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    benchmark(args)

