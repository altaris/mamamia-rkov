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

    _transisions: Dict[str, Dict[str, float]]
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
            transitions (Dict[str, Dict[str, float]]): The transisions of the
                Markov chain.
        """
        MarkovChain.assert_valid(transitions)
        self._transisions = transitions
        self._states = sorted(self._transisions.keys())

    @staticmethod
    def assert_valid(transitions: Dict[str, Dict[str, float]]) -> None:
        """
        Checks if a transision dictionary describes a valid Markov chain. If
        not, this method raises `ValueError` with a message explaining the
        problem.
        """
        if not transitions:
            raise ValueError("A Markov chain cannot be empty.")
        for k, v in transitions.items():
            for t in v.keys():
                if t not in transitions:
                    raise ValueError(
                        f"State '{k}' has an outgoing transision towards "
                        f"unknown state '{t}'."
                    )
            if not sum(v.values()) == 1.0:
                raise ValueError(
                    f"The outgoing transision of state '{k}' have "
                    "probabilities that do not sum up to 1."
                )

    def state_distribution_to_vector(
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

        """
        if not sum(distribution.values()) == 1.0:
            raise ValueError("The state probabilities don't sum up to 1.")
        return np.array([distribution.get(s, 0.0) for s in self._states])

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
                    self._transisions[a][b]
                    if b in self._transisions[a]
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
            p *= self._transisions[a].get(b, 0.0)
            if p == 0.0:
                return 0.0
        return p

    def state_index(self, state: str) -> int:
        """
        Returns the index of a state of the Markov chain.
        """
        return self._states.index(state)

    def to_graph(self) -> nx.DiGraph:
        """
        Returns the current Markov chain as a weighted directed graph.
        """
        graph = nx.DiGraph()
        graph.add_weighted_edges_from(
            [
                (a, b, w)
                for a, ta in self._transisions.items()
                for b, w in ta.items()
            ]
        )
        return graph
