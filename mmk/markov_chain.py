"""
Markov chain definition module
"""
__docformat__ = "google"

from typing import Dict, List

import networkx as nx
import numpy as np


class MarkovChain:
    """
    A Markov chain. (yup)
    """

    _transitions: Dict[str, Dict[str, float]]
    _states: List[str]

    def __init__(self, transitions: Dict[str, Dict[str, float]]) -> None:
        """
        Construct a new Markov chain from a dictionary of transitions, e.g.

            {
                "state_A": {
                    "state_B": .5,
                    "state_C": .5,
                },
                "state_B": {
                    "state_A": .8,
                    "state_C": .2,
                },
                "state_C": {
                    "state_C": 1.,
                }
            }

        Args:
            transitions (Dict[str, Dict[str, float]]): The transitions of the
                Markov chain.
        """
        MarkovChain.assert_valid(transitions)
        self._transitions = transitions
        self._states = sorted(self._transitions.keys())

    @staticmethod
    def assert_valid(transitions: Dict[str, Dict[str, float]]) -> None:
        """
        Checks if a transition dictionary describes a valid Markov chain. If
        not, this method raises `ValueError` with a message explaining the
        problem.
        """
        if not transitions:
            raise ValueError("A Markov chain cannot be empty.")
        for k, v in transitions.items():
            for t in v.keys():
                if t not in transitions:
                    raise ValueError(
                        f"State '{k}' has an outgoing transition towards "
                        f"unknown state '{t}'."
                    )
            if not sum(v.values()) == 1.0:
                raise ValueError(
                    f"The outgoing transition of state '{k}' have "
                    "probabilities that do not sum up to 1."
                )

    def predecessors(self, states: List[str]) -> List[str]:
        """
        Returns the list of states that have outgoing transitions towards any
        state in the provided list of states.

        The returned list is sorted.
        """
        g = nx.reverse_view(self.to_graph())
        result = set()
        for s in states:
            result.update(nx.algorithms.dag.descendants(g, s))
        return sorted(list(result))

    def probability_matrix(self) -> np.ndarray:
        """
        Returns the probability matrix of the Markov chain. The (i, j)-th entry
        corresponds to the probability of the transition from state i to state
        j, where i and j are indices in the list of states **sorted
        alphabetically**. See `state_index`.

        For example, if the transition dictionary is

            {
                "state_B": {
                    "state_A": .8,
                    "state_C": .2,
                },
                "state_C": {
                    "state_C": 1.,
                }
                "state_A": {
                    "state_C": .5,
                    "state_B": .5,
                },
            }

        then the probability matrix is

            array([[0. , 0.5, 0.5],
                   [0.8, 0. , 0.2],
                   [0. , 0. , 1. ]])

        Note that the first row corresponds to the outgoing transition
        probabilities of "state_A", while the 3rd column are incoming
        transition probabilities of "state_C".
        """
        return np.array(
            [
                [
                    self._transitions[a][b]
                    if b in self._transitions[a]
                    else 0.0
                    for b in self._states
                ]
                for a in self._states
            ]
        )

    def probability_of_path(self, path: List[str]) -> float:
        """
        Returns the probability of a path, represented as a list of states.
        Note that consecutive state need not actually be connected in the
        underlying graph of the Markov chain, in which case the transition
        probability is treated as 0.

        By convention, the probability of a path of length <= 1 is 1.
        """
        if len(path) <= 1:
            return 1.0
        p = 1.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            p *= self._transitions[a].get(b, 0.0)
            if p == 0.0:
                return 0.0
        return p

    def restrict_tensor_to_states(
        self, tensor: np.ndarray, states: List[str]
    ) -> np.ndarray:
        """
        Sets rows and columns of a tensor to 0 at every index not corresponding
        to any states given as argument.

        For example, consider a Markov chain whose states are

            ["A", "B", "C", "D"]

        if the argument `states` is

            states=["A", "D"]

        and tensor if the following matrix:

            array([[ 1,  2,  3,  4],
                   [ 5,  6,  7,  8],
                   [ 9, 10, 11, 12],
                   [13, 14, 15, 16]])

        then the result is

            array([[ 1,  0,  0,  4],
                   [ 0,  0,  0,  0],
                   [ 0,  0,  0,  0],
                   [13,  0,  0, 16]])

        The rows and columns corresponding to states `B` and `C` have been
        zeroed.

        The tensor can also be a vector.
        """
        r = len(tensor.shape)
        if r not in [1, 2]:
            raise NotImplementedError(
                "The rank of the tensor should be 1 or 2."
            )
        states_c = set(self._states) - set(states)
        indices = [self.state_index(s) for s in states_c]
        result = np.array(tensor, copy=True)
        if r == 1:
            result[indices] = 0.0
        elif r == 2:
            result[indices] = 0.0
            result[:, indices] = 0.0
        return result

    def state_index(self, state: str) -> int:
        """
        Returns the index of a state of the Markov chain.
        """
        return self._states.index(state)

    def state_mask(self, states: List[str]) -> np.ndarray:
        """
        Given a list of states, returns a vector whose entries are 1. at
        indices corresponding to one of the given states, and 0. otherwise. For
        example, if the states of the chains are

            ["A", "B", "C", "D", "E"]

        and the given list is

            ["B", "E", "A"]

        then the resulting vector is

            array([1., 1., 0., 0., 1.])

        """
        x = np.zeros((len(self._states),), dtype=float)
        for s in states:
            x[self.state_index(s)] = 1.0
        return x

    def state_weights_to_vector(
        self, distribution: Dict[str, float]
    ) -> np.ndarray:
        """
        Creates a vector given a distribution in the states of the Markov
        chain, represented by a dictionary mapping a state to its probability.

        For example, if the Markov chain is

            {
                "state_A": {
                    "state_B": .5,
                    "state_C": .5,
                },
                "state_B": {
                    "state_A": .8,
                    "state_C": .2,
                },
                "state_C": {
                    "state_C": 1.,
                },
            }

        and the distribution is

            {
                "state_A": .9,
                "state_C": .1,
            }

        then the resulting vector is

            array([0.9, 0. , 0.1])

        See also `vector_to_state_weights`.
        """
        if not sum(distribution.values()) == 1.0:
            raise ValueError("The state probabilities don't sum up to 1.")
        return np.array([distribution.get(s, 0.0) for s in self._states])

    def to_graph(self) -> nx.DiGraph:
        """
        Returns the current Markov chain as a weighted directed graph.
        """
        graph = nx.DiGraph()
        graph.add_weighted_edges_from(
            [
                (a, b, w)
                for a, ta in self._transitions.items()
                for b, w in ta.items()
            ]
        )
        return graph

    def vector_to_state_weights(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Inverse to `vector_to_state_weights`. For example, if the Markov chain
        is

            {
                "state_A": {
                    "state_B": .5,
                    "state_C": .5,
                },
                "state_B": {
                    "state_A": .8,
                    "state_C": .2,
                },
                "state_C": {
                    "state_C": 1.,
                },
            }

        and the vector is

            array([0.9, 0. , 0.1])

        then this methods returns the following dictionary

            {
                "state_A": .9,
                "state_B": 0.,
                "state_C": .1,
            }

        """
        return dict(zip(self._states, vector))
