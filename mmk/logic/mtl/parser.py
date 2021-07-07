"""
Parser for MTL
"""
__docformat__ = "google"

from typing import Union

from parsimonious import Grammar, NodeVisitor

# pylint: disable=wildcard-import,unused-wildcard-import
from .formula import *

grammar = Grammar(
    r"""
    formula             = always_within_formula /
                          always_formula /
                          eventually_within_formula /
                          eventually_formula /
                          not_formula /
                          until_within_formula /
                          until_formula /
                          state_list

    always_within_formula
                        = always ws within ws number ws formula
    always_formula      = always ws formula
    eventually_formula  = eventually ws formula
    eventually_within_formula
                        = eventually ws within ws number ws formula
    not_formula         = not ws formula
    until_formula       = state_list ws until ws formula
    until_within_formula
                        = state_list ws until ws within ws number ws formula

    always          = "always" / "henceforth" / "[]"
    eventually      = "eventually" / "<>"
    not             = "not" / "¬"
    until           = "until" / "U"
    within          = "within"

    state           = ~"[A-Z0-9_]+"i
    comma_state     = "," state
    state_list      = state comma_state*

    number          = ~"[0-9]+"i
    ws              = ~"\s+"
    """
)


# pylint: disable=missing-function-docstring,no-self-use,unused-argument
class FormulaVisitor(NodeVisitor):
    """
    Formula visitor.
    """

    def visit_always_formula(self, node, visited_children):
        return AlwaysFormula(child=visited_children[-1])

    def visit_always_within_formula(self, node, visited_children):
        return AlwaysFormula(
            child=visited_children[-1],
            within=visited_children[4],
        )

    def visit_comma_state(self, node, visited_children):
        return visited_children[-1]

    def visit_eventually_formula(self, node, visited_children):
        return EventuallyFormula(child=visited_children[-1])

    def visit_eventually_within_formula(self, node, visited_children):
        return EventuallyFormula(
            child=visited_children[-1],
            within=visited_children[4],
        )

    def visit_formula(self, node, visited_children):
        return visited_children[0]

    def visit_not_formula(self, node, visited_children):
        return NotFormula(child=visited_children[-1])

    def visit_number(self, node, visited_children):
        return int(node.text)

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

    def visit_state(self, node, visited_children):
        return node.text

    def visit_state_list(self, node, visited_children):
        return Predicate(
            states=sorted([visited_children[0]] + visited_children[1])
        )

    def generic_visit(self, node, visited_children):
        return visited_children


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
