# %%
import torch

from models import *
from network import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

N = 1000
INITIAL_INFECTED = 100
initial_states = torch.zeros(N, dtype=torch.int).to(device)
initial_states[torch.randint(0, N, (INITIAL_INFECTED,))] = 1

models: dict[str, SpreadingModel] = {
    'SIS': SIS(),
    'SIR': SIR(),
    'SAIS': SAIS(),
    'SEIR': SEIR(),
}

networks: dict[str, Network] = {
    'RandomGeometric': RandomGeometric(num_node=N, radius=0.050463),
    'BarabasiAlbert': BarabasiAlbert(num_node=N, num_edge=4),
    'WattsStrogatz': WattsStrogatz(num_node=N, k=8, p=0.1),
}

spreading_model = 'SEIR'

simulators: dict[str, Simulator] = {
    'GEMFsim': GEMFsim(
        initial_state=initial_states.cpu().numpy(),
        spreading_model=models[spreading_model],
        network=networks['RandomGeometric'],
    ),
    "Gillespie_v1": Gillespie_v1(
        initial_state=initial_states,
        spreading_model=models[spreading_model],
        network=networks['RandomGeometric'],
    ),
    "NodeWiseTauLeaping_v1": NodeWiseTauLeaping_v1(
        initial_state=initial_states,
        spreading_model=models[spreading_model],
        network=networks['RandomGeometric'],
        theta=5,
    ),
    "SystemWiseTauLeaping_v1": SystemWiseTauLeaping_v1(
        initial_state=initial_states,
        spreading_model=models[spreading_model],
        network=networks['RandomGeometric'],
        theta=5,
        K_MAX=50,
    ),
    "HeapTauLeaping_v1": HeapTauLeaping_v1(
        initial_state=initial_states,
        spreading_model=models[spreading_model],
        network=networks['RandomGeometric'],
        theta=5,
        K_MAX=50,
    ),
    "AdaptiveTauLeaping_v1": AdaptiveTauLeaping_v1(
        initial_state=initial_states,
        spreading_model=models[spreading_model],
        network=networks['RandomGeometric'],
        theta=8.0,
        max_tau=2.0,
        max_events_per_node=4.0,
        max_total_events=0.4,
        shrink=0.5,
        max_adjust=8,
    ),
}

# %%

import matplotlib.pyplot as plt
import os

for name, simulator in simulators.items():
    print(f"Simulator: {name}")
    
    time_points = []
    state_trajectories = []

    simulator.reset()

    for _ in range(600):
        times, states = simulator.step()
        if times == float('nan'):
            break
        counts = simulator.count_by_state
        time_points.append(simulator.current_time)
        state_trajectories.append(counts)

    state_counts_over_time = np.stack(state_trajectories)

    plt.figure(figsize=(12, 7))

    # Plot the data for each state
    for i in range(state_counts_over_time.shape[1]):
        plt.plot(
            time_points,
            state_counts_over_time[:, i],  # Y-axis: counts for state i
            label=simulator.spreading_model.states_label[i],
            color=simulator.spreading_model.states_colors[i],
            drawstyle='steps-post'  # Use a step plot for event-based simulations
        )

    plt.xlabel('Simulation Time')
    plt.ylabel('Number of Nodes')
    plt.title(f'Node State Counts Over Time ({name} Simulation)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.xlim(left=0, right=6)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{name}.png', bbox_inches='tight')
    plt.show()

# %%
