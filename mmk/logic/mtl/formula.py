"""
AST for MTL
"""
__docformat__ = "google"

from dataclasses import dataclass
from typing import List, Optional

from scipy.optimize import fixed_point
import numpy as np

from mmk import MarkovChain

# pylint: disable=missing-class-docstring


@dataclass
class Formula:
    """
    Abstract MTL formula.
    """

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        raise NotImplementedError()


@dataclass
class Predicate(Formula):
    """
    A predicate is just a set of states.
    """

    states: List[str]

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        return chain.state_mask(self.states)


@dataclass
class AlwaysFormula(Formula):
    child: Formula
    # within: Optional[int] = None

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain. Transforms the

            always P

        into

            not eventually not P

        and evaluates the latter, where `P` is the child of this formula.
        """
        formula = NotFormula(
            child=EventuallyFormula(
                child=NotFormula(
                    child=self.child,
                ),
            ),
        )
        return formula.evaluate_on(chain)


@dataclass
class EventuallyFormula(Formula):
    child: Formula
    within: Optional[int] = None

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        n = len(chain._states)
        p = self.child.evaluate_on(chain)
        ps = [a for a, b in zip(chain._states, p) if b]
        s = chain.state_complement(ps, chain.predecessors(ps))
        b = chain.probability_matrix() @ p
        if self.within is None:
            a = chain.restrict_tensor_to_states(
                np.identity(n) - chain.probability_matrix(), s
            )
            if np.linalg.matrix_rank(a) == n:
                x = np.linalg.solve(a, b)
            else:
                x, *_ = np.linalg.lstsq(a, b, rcond=None)
            x += chain.state_mask(ps)
        else:
            a = chain.restrict_tensor_to_states(chain.probability_matrix(), s)
            x = np.zeros(n)
            for _ in range(self.within):
                x = a @ x + b
        return x


@dataclass
class NotFormula(Formula):
    child: Formula

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        return 1.0 - self.child.evaluate_on(chain)


@dataclass
class UntilFormula(Formula):
    left_child: Predicate
    right_child: Formula
    within: Optional[int] = None

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        p_left = self.left_child.evaluate_on(chain)
        p_right = self.right_child.evaluate_on(chain)
        s_left = chain.states_from_mask(p_left > 0)
        s1 = chain.states_from_mask(p_right > 0)
        s0 = chain.state_complement(s_left + s1)
        s = chain.state_complement(s0 + s1)
        a = chain.restrict_tensor_to_states(chain.probability_matrix(), s)
        b = chain.probability_matrix() @ p_right
        x = np.zeros(len(chain._states))
        if self.within is None:
            x = fixed_point(lambda y: a @ y + b, x)
        else:
            for _ in range(self.within):
                x = a @ x + b
        return x
