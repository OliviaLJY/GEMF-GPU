from abc import ABC, abstractmethod
import torch
import numpy as np

from network import Network

eps = torch.finfo(torch.float).eps


class SpreadingModel(ABC):

    states_label: list[str]
    states_colors: list[str]
    num_state: int

    @property
    @abstractmethod
    def NODE_TRANSITION_RATE(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def EDGE_TRANSITION_RATE(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def Para(self) -> list:
        pass


class SIS(SpreadingModel):
    def __init__(self, beta=0.23, gamma=1):
        # beta: infection rate, gamma: recovery rate
        # default: 5 days infectious period, one day to recover

        self._beta = beta
        self._gamma = gamma
        self.states_label = ['Susceptible', 'Infected']
        self.states_colors = ['red', 'black']
        self.num_state = 2

    @property
    def Para(self):
        from GEMFsim import Para_SIS
        return Para_SIS(self._gamma, self._beta)

    @property
    def NODE_TRANSITION_RATE(self):

        # I->S: gamma, recovery
        return torch.tensor([
            # to 0,   1
            [0.00, 0.00],  # from state 0
            [self._gamma, 0.00],  # from state 1
        ], dtype=torch.float)

    @property
    def EDGE_TRANSITION_RATE(self):

        # S->I: beta, infection
        return torch.tensor([
            # to 0,   1
            [0.00, self._beta],  # from state 0
            [0.00, 0.00],  # from state 1
        ], dtype=torch.float)


class SIR(SpreadingModel):
    def __init__(self, beta=0.05, gamma=1):
        # beta: infection rate, gamma: recovery rate
        # default: 20 days infectious period, one day to recover

        self._beta = beta
        self._gamma = gamma
        self.states_label = ['Susceptible', 'Infected', 'Recovered']
        self.states_colors = ['red', 'black', 'green']
        self.num_state = 3

    @property
    def Para(self):
        from GEMFsim import Para_SIR
        return Para_SIR(self._gamma, self._beta)

    @property
    def NODE_TRANSITION_RATE(self):

        # I->R: gamma, recovery
        return torch.tensor([
            # to 0,   1,   2
            [0.00, 0.00, 0.00],  # from state 0
            [0.00, 0.00, self._gamma],  # from state 1
            [0.00, 0.00, 0.00],  # from state 2
        ], dtype=torch.float)

    @property
    def EDGE_TRANSITION_RATE(self):

        # S->I: beta, infection
        return torch.tensor([
            # to 0,   1,   2
            [0.00, self._beta, 0.00],  # from state 0
            [0.00, 0.00, 0.00],  # from state 1
            [0.00, 0.00, 0.00],  # from state 2
        ], dtype=torch.float)


class SAIS(SpreadingModel):
    def __init__(self, beta=2, gamma=1, kappa=0.2, beta_a=0.4):
        # beta: infection rate, gamma: recovery rate
        # kappa: rate of susceptible to alert, beta_a: infection rate of alert individuals
        # default: 0.5 days infectious period, one day to recover
        # 2 days to become alert, 5 days infectious period for alert individuals

        self._beta = beta
        self._gamma = gamma
        self._kappa = kappa
        self._beta_a = beta_a

        self.states_label = ['Susceptible', 'Alert', 'Infected']
        self.states_colors = ['red', 'blue', 'black']
        self.num_state = 3

    @property
    def Para(self):
        from GEMFsim import Para_SAIS_Single
        return Para_SAIS_Single(self._gamma, self._beta, self._beta_a, self._kappa)

    @property
    def NODE_TRANSITION_RATE(self):

        # I->S: gamma, recovery
        return torch.tensor([
            # to 0,   1,   2
            [0.00, 0.00, 0.00],  # from state 0
            [0.00, 0.00, 0.00],  # from state 1
            [self._gamma, 0.00, 0.00],  # from state 2
        ], dtype=torch.float)

    @property
    def EDGE_TRANSITION_RATE(self):

        # S->I: beta, infection
        # S->A: kappa, alert
        # A->I: beta_a, infection of alert individuals
        return torch.tensor([
            # to 0,   1,   2
            [0.00, self._kappa, self._beta],  # from state 0
            [0.00, 0.00, self._beta_a],  # from state 1
            [0.00, 0.00, 0.00],  # from state 2
        ], dtype=torch.float)


class SEIR(SpreadingModel):
    def __init__(self, beta=0.23, gamma=1, sigma=0.2):
        # beta: infection rate, gamma: recovery rate
        # sigma: rate of exposed to infected
        # default: 5 days infectious period, one day to recover

        self._beta = beta
        self._gamma = gamma
        self._sigma = sigma

        self.states_label = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
        self.states_colors = ['red', 'orange', 'black', 'green']
        self.num_state = 4

    @property
    def Para(self):
        from GEMFsim import Para_SEIR
        return Para_SEIR(self._gamma, self._beta, self._sigma)

    @property
    def NODE_TRANSITION_RATE(self):

        # E->I: sigma, infection
        # I->R: gamma, recovery
        return torch.tensor([
            # to 0,   1,   2,   3
            [0.00, 0.00, 0.00, 0.00],  # from state 0
            [0.00, 0.00, self._sigma, 0.00],  # from state 1
            [0.00, 0.00, 0.00, self._gamma],  # from state 2
            [0.00, 0.00, 0.00, 0.00],  # from state 3
        ], dtype=torch.float)

    @property
    def EDGE_TRANSITION_RATE(self):

        # S->E: beta, exposure
        return torch.tensor([
            # to 0,   1,   2,   3
            [0.00, self._beta, 0.00, 0.00],  # from state 0
            [0.00, 0.00, 0.00, 0.00],  # from state 1
            [0.00, 0.00, 0.00, 0.00],  # from state 2
            [0.00, 0.00, 0.00, 0.00],  # from state 3
        ], dtype=torch.float)


class Simulator(ABC):

    initial_state: torch.Tensor | np.ndarray
    current_state: torch.Tensor | np.ndarray
    current_time: float
    spreading_model: SpreadingModel
    network: Network

    @abstractmethod
    def step(self) -> tuple[float, torch.Tensor | np.ndarray]:
        # will update self.current_state
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @property
    @abstractmethod
    def count_by_state(self) -> np.ndarray:
        pass


class GEMFsim(Simulator):
    def __init__(self,
                 initial_state: np.ndarray,
                 spreading_model: SpreadingModel,
                 network: Network
                 ):

        self.initial_state = initial_state
        self.current_state = initial_state.copy()
        self.current_time = 0.0
        self.spreading_model = spreading_model
        self.network = network
        self._para = spreading_model.Para
        self._net = network.gemfsim_net
        import GEMFsim
        self._gemfsim = GEMFsim.GEMF_SIM

    def step(self):
        StopCond = [None, 1e-9]
        ts, n_index, i_index, j_index = self._gemfsim(self._para,
                                                      self._net,
                                                      self.current_state,
                                                      StopCond,
                                                      self.network.num_node)

        if len(ts) == 0 or ts[0] == np.inf:
            tau = float('nan')
        else:
            tau = ts[0]

        if len(ts) > 0:
            node_changed = n_index[0]
            old_state = i_index[0]
            new_state = j_index[0]

            self.current_state[node_changed] = new_state
            self.current_time += tau

        return tau, self.current_state

    def reset(self):
        self.current_state = self.initial_state.copy()  # type: ignore
        self.current_time = 0.0

    @property
    def count_by_state(self):
        num_states = self.spreading_model.num_state
        counts = np.bincount(self.current_state, minlength=num_states)
        return counts


@torch.no_grad()
# @torch.compile()
def get_neighbor_state_counts_v1(edge_index: torch.Tensor, states: torch.Tensor, num_states: int) -> torch.Tensor:

    num_nodes = states.size(0)

    src_nodes = edge_index[0]
    neighbor_states = states[edge_index[1]]

    flat_indices = src_nodes * num_states + neighbor_states

    counts_flat = torch.bincount(
        flat_indices,
        minlength=num_nodes * num_states
    )

    return counts_flat.view(num_nodes, num_states)


@torch.no_grad()
# @torch.compile()
def compute_total_rates_v1(
    node_transition_rates: torch.Tensor,
    edge_transition_rates: torch.Tensor,
    states: torch.Tensor,
    neighbor_state_counts: torch.Tensor,
) -> torch.Tensor:

    spontaneous_rates = node_transition_rates[states]
    induced_rates = edge_transition_rates[states]

    total_rates = torch.addcmul(
        spontaneous_rates,
        neighbor_state_counts.float(),
        induced_rates
    )

    total_rates[torch.arange(states.size(
        0), device=states.device), states] = 0.0

    return total_rates


class Gillespie_v1(Simulator):

    def __init__(self,
                 initial_state: torch.Tensor,
                 spreading_model: SpreadingModel,
                 network: Network,
                 ):

        self.initial_state = initial_state
        self.current_state = initial_state.clone()
        self.current_time = 0.0
        self.spreading_model = spreading_model
        self.network = network
        self._edge_index = network.edge_index.to(initial_state.device)

    @torch.no_grad()
    # @torch.compile()
    def _step(self):

        neighbor_counts = get_neighbor_state_counts_v1(
            self._edge_index, self.current_state, self.spreading_model.num_state  # type: ignore
        )
        transition_rates = compute_total_rates_v1(
            self.spreading_model.NODE_TRANSITION_RATE,
            self.spreading_model.EDGE_TRANSITION_RATE,
            self.current_state,  # type: ignore
            neighbor_counts,
        )

        _, num_states = transition_rates.shape
        total_rate = transition_rates.sum()

        if total_rate <= eps:
            return float('nan'), self.current_state

        tau = - \
            torch.log(torch.rand(
                1, device=self.current_state.device)) / total_rate

        flat_rates = transition_rates.flatten()

        chosen_flat_index = torch.multinomial(flat_rates, num_samples=1)
        node_to_update = chosen_flat_index // num_states
        new_state = chosen_flat_index % num_states

        self.current_state[node_to_update] = new_state.int()

        return tau, self.current_state

    def step(self):
        tau, states = self._step()
        tau = float(tau)
        if tau != float('nan'):
            self.current_time += tau
        return tau, states

    def reset(self):
        self.current_state = self.initial_state.clone()  # type: ignore
        self.current_time = 0.0

    @property
    @torch.no_grad()
    @torch.compile()
    def count_by_state(self):
        num_states = self.spreading_model.num_state
        counts = torch.bincount(
            self.current_state, minlength=num_states).cpu().numpy() # type: ignore
        return counts


class NodeWiseTauLeaping_v1(Simulator):
    """Node-wise tau-leaping simulator.
    theta: int, simultaneous events to process in each step
    """

    def __init__(self,
                 initial_state: torch.Tensor,
                 spreading_model: SpreadingModel,
                 network: Network,
                 theta: int,
                 ):

        self.initial_state = initial_state
        self.current_state = initial_state.clone()
        self.current_time = 0.0
        self.spreading_model = spreading_model
        self.network = network
        self.theta = theta
        self._edge_index = network.edge_index.to(initial_state.device)

    @torch.no_grad()
    # @torch.compile()
    def _step(self):

        neighbor_counts = get_neighbor_state_counts_v1(
            self._edge_index, self.current_state, self.spreading_model.num_state  # type: ignore
        )
        transition_rates = compute_total_rates_v1(
            self.spreading_model.NODE_TRANSITION_RATE,
            self.spreading_model.EDGE_TRANSITION_RATE,
            self.current_state,  # type: ignore
            neighbor_counts,
        )

        node_total_rates = transition_rates.sum(dim=1)
        transition_rate_total = node_total_rates.sum()

        if transition_rate_total <= eps:
            return float('nan'), self.current_state

        tau = self.theta / transition_rate_total

        expected_events_per_node = node_total_rates * tau
        num_events_per_node = torch.poisson(expected_events_per_node)
        num_events_per_node.clamp_(max=1)

        active_nodes_idx = num_events_per_node.nonzero().squeeze(-1)
        if active_nodes_idx.numel() == 0:
            return float('nan'), self.current_state

        active_node_rates = transition_rates[active_nodes_idx]

        chosen_new_states = torch.multinomial(
            active_node_rates, num_samples=1).squeeze(-1)

        self.current_state[active_nodes_idx] = chosen_new_states.int()

        return tau, self.current_state

    def step(self):
        tau, states = self._step()
        tau = float(tau)
        if tau != float('nan'):
            self.current_time += tau
        return tau, states

    def reset(self):
        self.current_state = self.initial_state.clone()  # type: ignore
        self.current_time = 0.0

    @property
    @torch.no_grad()
    @torch.compile()
    def count_by_state(self):
        num_states = self.spreading_model.num_state
        counts = torch.bincount(
            self.current_state, minlength=num_states).cpu().numpy() # type: ignore
        return counts


class SystemWiseTauLeaping_v1(Simulator):
    """Node-wise tau-leaping simulator.
    theta: int, simultaneous events to process in each step
    K_MAX: int, maximum number of nodes to sample in each step
    """

    def __init__(self,
                 initial_state: torch.Tensor,
                 spreading_model: SpreadingModel,
                 network: Network,
                 theta: int,
                 K_MAX: int,
                 ):

        self.initial_state = initial_state
        self.current_state = initial_state.clone()
        self.current_time = 0.0
        self.spreading_model = spreading_model
        self.network = network
        self.theta = theta
        self.K_MAX = K_MAX
        self._edge_index = network.edge_index.to(initial_state.device)

    @torch.no_grad()
    # @torch.compile()
    def _step(self):

        neighbor_counts = get_neighbor_state_counts_v1(
            self._edge_index, self.current_state, self.spreading_model.num_state  # type: ignore
        )
        transition_rates = compute_total_rates_v1(
            self.spreading_model.NODE_TRANSITION_RATE,
            self.spreading_model.EDGE_TRANSITION_RATE,
            self.current_state,  # type: ignore
            neighbor_counts,
        )

        node_total_rates = transition_rates.sum(dim=1)
        transition_rate_total = node_total_rates.sum()

        if transition_rate_total <= eps:
            return float('nan'), self.current_state

        tau = self.theta / transition_rate_total

        n_active = torch.count_nonzero(node_total_rates)
        k_sampled = torch.poisson(torch.tensor(
            self.theta, device=self.current_state.device).float())

        k = torch.min(k_sampled, n_active)

        nodes_to_update_idx = torch.multinomial(
            node_total_rates, num_samples=self.K_MAX, replacement=True
        )

        weights_for_state_sampling = transition_rates[nodes_to_update_idx]

        chosen_new_states = torch.multinomial(
            weights_for_state_sampling, num_samples=1).squeeze(-1)

        update_mask = torch.arange(
            self.K_MAX, device=self.current_state.device) < k
        nodes_to_update_idx = nodes_to_update_idx[update_mask]
        chosen_new_states = chosen_new_states[update_mask]

        self.current_state[nodes_to_update_idx] = chosen_new_states.int()

        return tau, self.current_state

    def step(self):
        tau, states = self._step()
        tau = float(tau)
        if tau != float('nan'):
            self.current_time += tau
        return tau, states

    def reset(self):
        self.current_state = self.initial_state.clone()  # type: ignore
        self.current_time = 0.0

    @property
    @torch.no_grad()
    @torch.compile()
    def count_by_state(self):
        num_states = self.spreading_model.num_state
        counts = torch.bincount(
            self.current_state, minlength=num_states).cpu().numpy() # type: ignore
        return counts


class HeapTauLeaping_v1(Simulator):
    """System-wise tau-leaping using Poisson counts and top-k selection."""

    def __init__(self,
                 initial_state: torch.Tensor,
                 spreading_model: SpreadingModel,
                 network: Network,
                 theta: int,
                 K_MAX: int,
                 ):

        self.initial_state = initial_state
        self.current_state = initial_state.clone()
        self.current_time = 0.0
        self.spreading_model = spreading_model
        self.network = network
        self.theta = theta
        self.K_MAX = K_MAX
        self._edge_index = network.edge_index.to(initial_state.device)

    @torch.no_grad()
    # @torch.compile()
    def _step(self):
        neighbor_counts = get_neighbor_state_counts_v1(
            self._edge_index, self.current_state, self.spreading_model.num_state  # type: ignore
        )
        transition_rates = compute_total_rates_v1(
            self.spreading_model.NODE_TRANSITION_RATE,
            self.spreading_model.EDGE_TRANSITION_RATE,
            self.current_state,  # type: ignore
            neighbor_counts,
        )

        node_total_rates = transition_rates.sum(dim=1)
        transition_rate_total = node_total_rates.sum()

        if transition_rate_total <= eps:
            return float('nan'), self.current_state

        tau = self.theta / transition_rate_total

        expected_events_per_node = node_total_rates * tau
        poisson_counts = torch.poisson(expected_events_per_node)

        positive_nodes = torch.count_nonzero(poisson_counts)
        if positive_nodes == 0:
            return float('nan'), self.current_state

        k = min(self.K_MAX, int(positive_nodes))
        top_counts, top_indices = torch.topk(poisson_counts, k)
        active_mask = top_counts > 0
        active_nodes_idx = top_indices[active_mask]

        if active_nodes_idx.numel() == 0:
            return float('nan'), self.current_state

        active_node_rates = transition_rates[active_nodes_idx]
        chosen_new_states = torch.multinomial(
            active_node_rates, num_samples=1).squeeze(-1)

        self.current_state[active_nodes_idx] = chosen_new_states.int()

        return tau, self.current_state

    def step(self):
        tau, states = self._step()
        tau = float(tau)
        if tau != float('nan'):
            self.current_time += tau
        return tau, states

    def reset(self):
        self.current_state = self.initial_state.clone()  # type: ignore
        self.current_time = 0.0

    @property
    @torch.no_grad()
    @torch.compile()
    def count_by_state(self):
        num_states = self.spreading_model.num_state
        counts = torch.bincount(
            self.current_state, minlength=num_states).cpu().numpy() # type: ignore
        return counts
