"""
Parser for PCTL
"""
__docformat__ = "google"


from typing import Union

from parsimonious import Grammar, NodeVisitor
import numpy as np


from mmk import MarkovChain

# pylint: disable=wildcard-import,unused-wildcard-import
from .formula import *


grammar = Grammar(
    r"""
    formula                 = and_formula /
                              not_formula /
                              parenthesis_formula /
                              probability_formula /
                              true /
                              false /
                              state_list
    path_formula            = always_within_formula /
                              always_formula /
                              eventually_within_formula /
                              eventually_formula /
                              next_formula /
                              parenthesis_path_formula /
                              until_within_formula /
                              until_formula

    always_within_formula
                            = always ws within ws number ws formula
    always_formula          = always ws formula
    and_formula             = and ws formula ws formula
    eventually_formula      = eventually ws formula
    eventually_within_formula
                            = eventually ws within ws number ws formula
    next_formula            = next ws formula
    not_formula             = not ws formula
    parenthesis_formula     = "(" ws formula ws ")"
    parenthesis_path_formula
                            = "(" ws path_formula ws ")"
    probability_formula     = probability ws within ws interval ws path_formula
    until_formula           = formula ws until ws formula
    until_within_formula    = formula ws until ws within ws number ws formula

    interval        = bracket ws float ws float ws bracket

    always          = "always" / "henceforth" / "[]" / "□"
    and             = "and" / "/\\" / "∧" / "⋀"
    bracket         = "[" / "]"
    eventually      = "eventually" / "<>" / "◊"
    false           = "false" / "F" / "⊥"
    next            = "next" / "O" / "○"
    not             = "not" / "¬"
    probability     = "probability" / "P" / "ℙ"
    true            = "true" / "T" / "⊤"
    until           = "until" / "U" / "⋃" / "◡"
    within          = "within"

    state           = ~"[A-Z0-9_]+"i
    comma_state     = "," state
    state_list      = state comma_state*

    float           = ~"[0-9]+(\.[0-9]+)?"i
    number          = ~"[0-9]+"i
    ws              = ~"\s+"
    """
)

# pylint: disable=missing-function-docstring,no-self-use,too-many-public-methods,unused-argument
class FormulaVisitor(NodeVisitor):
    """
    Formula visitor.
    """

    def generic_visit(self, node, visited_children):
        return visited_children

    def visit_always_formula(self, node, visited_children):
        return AlwaysFormula(child=visited_children[-1])

    def visit_always_within_formula(self, node, visited_children):
        return AlwaysFormula(
            child=visited_children[-1],
            within=visited_children[4],
        )

    def visit_and_formula(self, node, visited_children):
        return AndFormula(
            left_child=visited_children[2],
            right_child=visited_children[-1],
        )

    def visit_eventually_formula(self, node, visited_children):
        return EventuallyFormula(child=visited_children[-1])

    def visit_eventually_within_formula(self, node, visited_children):
        return EventuallyFormula(
            child=visited_children[-1],
            within=visited_children[4],
        )

    def visit_bracket(self, node, visited_children):
        return node.text

    def visit_comma_state(self, node, visited_children):
        return visited_children[-1]

    def visit_false(self, node, visited_children):
        return NotFormula(child=TrueFormula())

    def visit_float(self, node, visited_children):
        return float(node.text)

    def visit_formula(self, node, visited_children):
        return visited_children[0]

    def visit_interval(self, node, visited_children):
        lbb, _, lb, _, ub, _, ubb = visited_children
        return Interval(
            lower_bound_inclusive=(lbb == "["),
            lower_bound=lb,
            upper_bound_inclusive=(ubb == "]"),
            upper_bound=ub,
        )

    def visit_next_formula(self, node, visited_children):
        return NextFormula(child=visited_children[-1])

    def visit_number(self, node, visited_children):
        return int(node.text)

    def visit_parenthesis_formula(self, node, visited_children):
        return visited_children[2]

    def visit_parenthesis_path_formula(self, node, visited_children):
        return visited_children[2]

    def visit_path_formula(self, node, visited_children):
        return visited_children[0]

    def visit_probability_formula(self, node, visited_children):
        return ProbabilityFormula(
            child=visited_children[-1],
            interval=visited_children[4],
        )

    def visit_state(self, node, visited_children):
        return node.text

    def visit_state_list(self, node, visited_children):
        return Predicate(
            states=sorted([visited_children[0]] + visited_children[1])
        )

    def visit_true(self, node, visited_children):
        return TrueFormula()

    def visit_until_formula(self, node, visited_children):
        return UntilFormula(
            left_child=visited_children[0],
            right_child=visited_children[-1],
        )

    def visit_until_within_formula(self, node, visited_children):
        return UntilFormula(
            left_child=visited_children[0],
            right_child=visited_children[-1],
            within=visited_children[6],
        )


def evaluate(chain: MarkovChain, formula: Union[str, Formula]) -> np.ndarray:
    """
    Evaluates the probability vector of an MTL formula on a given Markov chain.
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)
    return formula.evaluate_on(chain)


def parse_formula(formula: str) -> Formula:
    """
    Parses a formula and returns its AST.
    """
    return FormulaVisitor().visit(grammar.parse(formula))
