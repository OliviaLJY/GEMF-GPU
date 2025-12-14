from abc import ABC, abstractmethod
import networkx as nx
from torch_geometric.utils import from_networkx
import torch


class Network(ABC):

    @property
    @abstractmethod
    def num_node(self) -> int:
        pass

    @property
    @abstractmethod
    def num_edge(self) -> int:
        pass

    @property
    @abstractmethod
    def gemfsim_net(self) -> list:
        pass

    @property
    @abstractmethod
    def edge_index(self) -> torch.Tensor:
        pass

class RandomGeometric(Network):
    def __init__(self, num_node: int, radius: float):
        """
        characteristic: high clustering coefficient, high average shortest path length
        n : int or iterable
            Number of nodes or iterable of nodes
        radius: float
            Distance threshold value
        """
        self._num_node = num_node
        self._radius = radius
        self._graph = nx.random_geometric_graph(num_node, radius)
        self._data = from_networkx(self._graph)

    @property
    def num_node(self) -> int:
        return self._num_node

    @property
    def num_edge(self) -> int:
        return self._graph.number_of_edges()

    @property
    def gemfsim_net(self) -> list:
        from GEMFsim import NetCmbn, MyNet
        return NetCmbn([MyNet(self._graph)])

    @property
    def edge_index(self) -> torch.Tensor:
        return self._data.edge_index

class BarabasiAlbert(Network):
    def __init__(self, num_node: int, num_edge: int):
        """
        characterstic: scale-free, small-world
        n : int
            Number of nodes
        m : int
            Number of edges to attach from a new node to existing nodes
        """
        self._num_node = num_node
        self._num_edge = num_edge
        self._graph = nx.barabasi_albert_graph(num_node, num_edge)
        self._data = from_networkx(self._graph)

    @property
    def num_node(self) -> int:
        return self._num_node

    @property
    def num_edge(self) -> int:
        return self._graph.number_of_edges()

    @property
    def gemfsim_net(self) -> list:
        from GEMFsim import NetCmbn, MyNet
        return NetCmbn([MyNet(self._graph)])

    @property
    def edge_index(self) -> torch.Tensor:
        return self._data.edge_index

class WattsStrogatz(Network):
    def __init__(self, num_node: int, k: int, p: float):
        """
        characteristic: small-world, high clustering coefficient
        n : int
            Number of nodes
        k : int
            Each node is joined with its `k` nearest neighbors in a ring topology
        p : float
            The probability of rewiring each edge
        """
        self._num_node = num_node
        self._k = k
        self._p = p
        self._graph = nx.watts_strogatz_graph(num_node, k, p)
        self._data = from_networkx(self._graph)

    @property
    def num_node(self) -> int:
        return self._num_node

    @property
    def num_edge(self) -> int:
        return self._graph.number_of_edges()

    @property
    def gemfsim_net(self) -> list:
        from GEMFsim import NetCmbn, MyNet
        return NetCmbn([MyNet(self._graph)])

    @property
    def edge_index(self) -> torch.Tensor:
        return self._data.edge_index
