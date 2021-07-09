"""
AST for PCTL
"""
__docformat__ = "google"


import operator
from dataclasses import dataclass
from typing import List, Optional

from scipy.optimize import fixed_point
import numpy as np

from mmk import MarkovChain, PRECISION

# pylint: disable=missing-function-docstring,missing-class-docstring


@dataclass
class Interval:
    """
    A bounded real interval.
    """

    lower_bound_inclusive: bool
    lower_bound: float
    upper_bound_inclusive: bool
    upper_bound: float

    def contains(self, x: np.ndarray) -> np.ndarray:
        lop = operator.ge if self.lower_bound_inclusive else operator.gt
        uop = operator.le if self.upper_bound_inclusive else operator.lt
        result = lop(x, self.lower_bound) * uop(x, self.upper_bound)
        return result.astype(float)


@dataclass
class Formula:
    """
    Abstract PCTL formula.
    """

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        raise NotImplementedError()


@dataclass
class PathFormula(Formula):
    """
    Abstract PCTL path formula.
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
    within: Optional[int] = None

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
                within=self.within,
            ),
        )
        return formula.evaluate_on(chain)


@dataclass
class AndFormula(Formula):
    left_child: Formula
    right_child: Formula

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        # fmt: off
        return (
            self.left_child.evaluate_on(chain)
            * self.right_child.evaluate_on(chain)
        )
        # fmt: on


@dataclass
class EventuallyFormula(Formula):
    child: Formula
    within: Optional[int] = None

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        formula = UntilFormula(
            left_child=TrueFormula(),
            right_child=self.child,
            within=self.within,
        )
        return formula.evaluate_on(chain)


@dataclass
class NextFormula(PathFormula):
    child: Formula

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        return chain.probability_matrix() @ self.child.evaluate_on(chain)


@dataclass
class NotFormula(PathFormula):
    child: Formula

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        return 1.0 - self.child.evaluate_on(chain)


@dataclass
class ProbabilityFormula(Formula):
    child: PathFormula
    interval: Interval

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        return self.interval.contains(self.child.evaluate_on(chain))


@dataclass
class TrueFormula(Formula):
    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.
        """
        return chain.state_mask(chain._states)


@dataclass
class UntilFormula(PathFormula):
    left_child: Formula
    right_child: Formula
    within: Optional[int] = None

    def evaluate_on(self, chain: MarkovChain) -> np.ndarray:
        """
        Evaluates the formula on a Markov chain.

        Todo: Deduplicate code.
        """
        b_left = self.left_child.evaluate_on(chain).astype(bool)
        b_right = self.right_child.evaluate_on(chain).astype(bool)
        s_left = chain.states_from_mask(b_left)
        s1 = chain.states_from_mask(b_right)
        s0 = chain.state_complement(s_left + s1)
        s = chain.state_complement(s0 + s1)
        a = chain.restrict_tensor_to_states(chain.probability_matrix(), s)
        b = chain.probability_matrix() @ b_right.astype(float)
        x = np.zeros(len(chain._states))
        if self.within is None:
            x = fixed_point(lambda y: a @ y + b, x)
        else:
            for _ in range(self.within):
                x = a @ x + b
        return x.round(PRECISION)
