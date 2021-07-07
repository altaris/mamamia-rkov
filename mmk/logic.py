"""
Definition of the LTL-style logic for Markov chains.
"""
__docformat__ = "google"

from typing import Union

from parsimonious import Grammar, NodeVisitor
import numpy as np

from .markov_chain import MarkovChain

grammar = Grammar(
    r"""
    formula         = diamond_formula /
                      not_formula /
                      state_list

    diamond_formula = diamond ws formula
    not_formula     = not ws formula

    diamond         = "eventually" / "<>"
    not             = "not" / "¬"

    state           = ~"[A-Z0-9_]+"i
    comma_state     = "," state
    state_list      = state comma_state*

    ws              = ~"\s+"
    """
)

# pylint: disable=C0116, R0201, W0613
class FormulaVisitor(NodeVisitor):
    """
    Formula visitor
    """

    def visit_comma_state(self, node, visited_children):
        return visited_children[-1]

    def visit_diamond_formula(self, node, visited_children):
        return {"child": visited_children[-1], "type": "eventually"}

    def visit_formula(self, node, visited_children):
        return visited_children[0]

    def visit_not_formula(self, node, visited_children):
        return {"child": visited_children[-1], "type": "not"}

    def visit_state(self, node, visited_children):
        return node.text

    def visit_state_list(self, node, visited_children):
        states = sorted([visited_children[0]] + visited_children[1])
        return {"type": "states", "states": states}

    def generic_visit(self, node, visited_children):
        return visited_children


def parse_formula(formula: str) -> dict:
    """
    Parses a formula and returns a nested dict representing its structure. For
    example, the formula

        <> state_A,state_B

    returns

        {
            "type": "eventually"
            "child": {"states": ["state_A", "state_B"]},
        }

    """
    tree = grammar.parse(formula)
    visitor = FormulaVisitor()
    return visitor.visit(tree)


def probability_of_formula(
    chain: MarkovChain,
    formula: Union[str, dict],
) -> np.ndarray:
    if isinstance(formula, str):
        formula = parse_formula(formula)

    if formula["type"] == "eventually":
        n = len(chain._states)
        p = probability_of_formula(chain, formula["child"])
        s = [a for i, a in enumerate(chain._states) if p[i] > 0.0]
        y = chain.state_mask(s)
        s = list(set(chain.predecessors(s)) - set(s))
        a = chain.restrict_tensor_to_states(
            np.identity(n) - chain.probability_matrix(), s
        )
        b = chain.probability_matrix() @ p
        if np.linalg.matrix_rank(a) == n:
            x = np.linalg.solve(a, b)
        else:
            x, *_ = np.linalg.lstsq(a, b, rcond=None)
        return x + y

    if formula["type"] == "not":
        return 1.0 - probability_of_formula(chain, formula["child"])

    if formula["type"] == "states":
        return np.array([float(s in formula["states"]) for s in chain._states])

    raise RuntimeError("Unknown formula type '%s'." % formula["type"])
