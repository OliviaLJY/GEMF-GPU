import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
import math
import time
import argparse
import json
import datetime
import statistics
import os
import sys

import GEMFX
import GEMFsim

def log_message(message, log_file="benchmark_log.json"):
    """Appends a JSON object to the log file and prints to console."""
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'log': message
    }
    
    # Print to console
    print(f"[{timestamp}] {json.dumps(message, indent=4)}")

    # Append to log file
    try:
        with open('logs/' + log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    except IOError as e:
        print(f"Error writing to log file {'logs/' + log_file}: {e}", file=sys.stderr)


def setup_simulation(N, device):
    """Generates the graph, initial states, and rate matrices."""
    M = 2  # Number of states (0: Susceptible, 1: Infected)
    p = 15/(N-1) # ER graph probability

    G = nx.erdos_renyi_graph(N, p)
    edge_index = from_networkx(G).edge_index.to(device)

    initial_infected_count = int(N * 0.1)
    initial_states = torch.zeros(N, dtype=torch.int64).to(device)
    infected_nodes = torch.randperm(N)[:initial_infected_count]
    initial_states[infected_nodes] = 1

    NODE_TRANSITION_RATES = torch.tensor([
        # to 0,   1
        [0.00, 0.80],  # from state 0 (S)
        [1.00, 0.00],  # from state 1 (I)
    ], dtype=torch.float, device=device)

    EDGE_TRANSITION_RATES = torch.tensor([
        # to 0,    1
        [0.00, 0.80],  # from state 0 (S), with infected neighbor
        [0.00, 0.00],  # from state 1 (I)
    ], dtype=torch.float, device=device)

    return G, edge_index, initial_states, M, NODE_TRANSITION_RATES, EDGE_TRANSITION_RATES


def run_simulation(sim_function, sim_params, T_MAX, initial_states, M, edge_index, node_rates, edge_rates, device, track=False):
    """
    Runs a single simulation from t=0 to T_MAX and returns the execution time.
    If tracking is enabled, also returns the history of state counts.
    """
    current_states = initial_states.clone()
    current_time = 0.0
    
    history = []
    if track:
        state_counts = torch.bincount(current_states, minlength=M)
        history.append({'time': current_time, 'counts': state_counts.cpu().numpy()})

    while current_time < T_MAX:
        neighbor_counts = GEMFX.get_neighbor_state_counts(edge_index, current_states, M)
        transition_rates = GEMFX.compute_total_rates(node_rates, edge_rates, current_states, neighbor_counts).clone()
        
        new_states, tau = sim_function(states=current_states, transition_rates=transition_rates, **sim_params)

        if not torch.isfinite(tau) or tau <= 0:
            break

        current_time += tau.item()
        current_states = new_states.clone()
        
        if track:
            state_counts = torch.bincount(current_states, minlength=M)
            history.append({'time': current_time, 'counts': state_counts.cpu().numpy()})

    if track:
        return history
    return None # Only return history if tracking


def benchmark_gemfpy(N, G, T_MAX):
    """Wrapper for the GEMFPy baseline benchmark."""
    Para = GEMFsim.Para_SIS(0.8, 1.0) # Matched to other models
    x0 = np.zeros(N)
    x0 = GEMFsim.Initial_Cond_Gen(N, Para[1][0], [int(N * 0.1)], x0)
    Net = GEMFsim.NetCmbn([GEMFsim.MyNet(G)])
    StopCond = ['RunTime', T_MAX]
    
    # This function is timed directly
    ts, n_index, i_index, j_index = GEMFsim.GEMF_SIM(Para, Net, x0, StopCond, N)
    return ts, n_index, i_index, j_index

def resample_history(history, common_time_points, M):
    """Interpolates a trajectory with irregular time steps onto a common grid."""
    if not history:
        return np.zeros((len(common_time_points), M))

    raw_times = np.array([h['time'] for h in history])
    raw_counts = np.array([h['counts'] for h in history])
    
    # Use 'previous' value interpolation, suitable for step functions
    resampled_counts = np.zeros((len(common_time_points), M))
    for i, t in enumerate(common_time_points):
        # Find the last event that occurred at or before time t
        idx = np.searchsorted(raw_times, t, side='right') - 1
        if idx >= 0:
            resampled_counts[i, :] = raw_counts[idx, :]
        else:
            # For times before the first event, use initial state (already zeros if nothing happened)
             resampled_counts[i, :] = raw_counts[0, :] if raw_counts.shape[0] > 0 else np.zeros(M)
    return resampled_counts

def plot_and_save_results(average_trajectory, common_time_points, M, N, algorithm_name):
    """Plots the averaged state counts over time and saves the figure and data."""
    os.makedirs("figures", exist_ok=True)
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    states_label = ['Susceptible', 'Infected']
    states_colors = ['#0077b6', '#d90429']

    for i in range(M):
        plt.plot(
            common_time_points,
            average_trajectory[:, i],
            label=states_label[i] if i < len(states_label) else f'State {i}',
            color=states_colors[i] if i < len(states_colors) else None
        )

    plt.xlabel('Simulation Time')
    plt.ylabel('Average Number of Nodes')
    plt.title(f'Average Node State Counts Over Time ({algorithm_name})')
    plt.legend()
    plt.xlim(left=0, right=common_time_points[-1])
    plt.ylim(bottom=0, top=N)
    
    figure_path = f"figures/{algorithm_name}_average_tracking.png"
    plt.savefig(figure_path, bbox_inches='tight', dpi=300)
    print(f"\nAverage tracking plot saved to: {figure_path}")

    data_path = f"figures/{algorithm_name}_average_tracking_data.npz"
    np.savez(data_path, average_trajectory=average_trajectory, time_points=common_time_points)
    print(f"Average tracking data saved to: {data_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark script for network simulations.")
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['gillespie', 'discrete_time', 'tau_node', 'tau_system', 'gemfpy'],
                        help="The simulation algorithm to benchmark.")
    parser.add_argument('--N', type=int, default=500, help="Number of nodes in the graph.")
    parser.add_argument('--T_MAX', type=float, default=5.0, help="Maximum simulation time.")
    parser.add_argument('--theta', type=float, default=10.0, help="Theta value for tau-leaping methods.")
    parser.add_argument('--detail', type=int, default=10, help="Detail level for discrete time simulation.")
    parser.add_argument('--runs', type=int, default=1000, help="Number of benchmark repetitions.")
    parser.add_argument('--specific_tracking', action='store_true', help="Enable detailed state tracking for the last run.")
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                        help="Device to run the simulation on ('auto' detects CUDA).")
    parser.add_argument('--log_file', type=str, default="benchmark_log.json", help="Name of the JSON log file.")
    parser.add_argument('--plot_points', type=int, default=200, help="Number of points for the averaged plot.")
    
    args = parser.parse_args()

    # --- Setup Device ---
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.", file=sys.stderr)
        device = 'cpu'

    # --- Log Initial Parameters ---
    benchmark_params = vars(args)
    benchmark_params['device'] = device
    log_message({"event": "benchmark_start", "parameters": benchmark_params}, args.log_file)

    # --- Setup Simulation Environment ---
    print(f"Setting up simulation for N={args.N} on device={device}...")
    G, edge_index, initial_states, M, node_rates, edge_rates = setup_simulation(args.N, device)
    print(f"Generated ER graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    log_message({"event": "graph_generated", "parameters": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}}, args.log_file)
    
    # --- Select Algorithm and Parameters ---
    sim_function = None
    sim_params = {}
    is_gemfpy = False

    if args.algorithm == 'gillespie':
        sim_function = GEMFX.gillespie_step
    elif args.algorithm == 'discrete_time':
        sim_function = GEMFX.discrete_time_simulation
        sim_params = {'detail': args.detail}
    elif args.algorithm == 'tau_node':
        sim_function = GEMFX.tau_leaping_step_node_wise
        sim_params = {'theta': args.theta}
    elif args.algorithm == 'tau_system':
        sim_function = GEMFX.tau_leaping_step_system_wise
        K_MAX = int(args.theta + 6 * math.sqrt(args.N))
        sim_params = {'theta': args.theta, 'K_MAX': K_MAX}
    elif args.algorithm == 'gemfpy':
        if device == 'cuda':
            print("GEMFPy baseline runs on CPU only. Ignoring device setting.", file=sys.stderr)
        is_gemfpy = True
    
    # --- Run Benchmark ---
    timings = []
    common_time_points = np.linspace(0, args.T_MAX, args.plot_points)
    sum_of_trajectories = np.zeros((args.plot_points, M))
    
    if args.specific_tracking:
        print(f"\nStarting benchmark with tracking enabled for {args.runs} runs...")
    else:
        print(f"\nStarting benchmark for {args.runs} runs (Tracking disabled)...")
    
    warmup_runs = 1
    
    with torch.no_grad():
        for i in range(warmup_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            if is_gemfpy:
                benchmark_gemfpy(args.N, G, args.T_MAX)
            else:
                run_simulation(sim_function, sim_params, args.T_MAX, initial_states, M, edge_index, node_rates, edge_rates, device, args.specific_tracking)

            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            print(f"Warmup Run {i+1}/{warmup_runs} completed.", end='\r')

    with torch.no_grad():
        for i in range(args.runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            if is_gemfpy:
                ts, n_index, i_index, j_index = benchmark_gemfpy(args.N, G, args.T_MAX)
                M = GEMFsim.Para_SIS(0.8, 1.0)[0]
                x0 = initial_states.numpy()
                T, StateCount = GEMFsim.Post_Population(x0, M, args.N, ts, i_index, j_index)
                history = [{'time': T[i], 'counts': StateCount[:, i]} for i in range(StateCount.shape[1])]
            else:
                history = run_simulation(sim_function, sim_params, args.T_MAX, initial_states, M, edge_index, node_rates, edge_rates, device, args.specific_tracking)

            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            timings.append(end_time - start_time)
            if args.specific_tracking:
                resampled = resample_history(history, common_time_points, M)
                sum_of_trajectories += resampled
            print(f"Run {i+1}/{args.runs} completed in {timings[-1]:.6f} seconds.", end='\r')
            log_message({"event": "checkpoint", "parameters": {"run": i+1, "time": timings[-1]}}, args.log_file)
    
    print("\nBenchmark finished.")

    # --- Calculate and Report Results ---
    avg_time = statistics.mean(timings)
    std_dev = statistics.stdev(timings) if len(timings) > 1 else 0.0

    final_results = {
        "event": "benchmark_result",
        "parameters": benchmark_params,
        "results": {
            "mean_time_s": avg_time,
            "std_dev_s": std_dev,
            "total_time_s": sum(timings),
            "runs": args.runs
        }
    }
    log_message(final_results, args.log_file)
    print("\n--- Benchmark Summary ---")
    print(f"Algorithm: {args.algorithm}")
    print(f"Runs: {args.runs}")
    print(f"Average Time: {avg_time:.6f} s")
    print(f"Standard Deviation: {std_dev:.6f} s")
    print("-------------------------")


    # --- Specific Tracking ---
    if args.specific_tracking:
        average_trajectory = sum_of_trajectories / args.runs
        plot_and_save_results(average_trajectory, common_time_points, M, args.N, args.algorithm)



if __name__ == '__main__':
    main()