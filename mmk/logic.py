"""
Definition of the LTL-style logic for Markov chains.
"""
__docformat__ = "google"

from typing import Union

from parsimonious import Grammar, NodeVisitor
import numpy as np
from scipy.optimize import fixed_point

from .markov_chain import MarkovChain

grammar = Grammar(
    r"""
    formula             = always_formula /
                          eventually_within_formula /
                          eventually_formula /
                          not_formula /
                          until_within_formula /
                          until_formula /
                          state_list

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

# pylint: disable=C0116, R0201, W0613
class FormulaVisitor(NodeVisitor):
    """
    Formula visitor
    """

    def visit_always_formula(self, node, visited_children):
        return {"child": visited_children[-1], "type": "always"}

    def visit_comma_state(self, node, visited_children):
        return visited_children[-1]

    def visit_eventually_formula(self, node, visited_children):
        return {"child": visited_children[-1], "type": "eventually"}

    def visit_eventually_within_formula(self, node, visited_children):
        return {
            "child": visited_children[-1],
            "type": "eventually",
            "within": visited_children[4],
        }

    def visit_formula(self, node, visited_children):
        return visited_children[0]

    def visit_not_formula(self, node, visited_children):
        return {"child": visited_children[-1], "type": "not"}

    def visit_number(self, node, visited_children):
        return int(node.text)

    def visit_until_formula(self, node, visited_children):
        return {
            "left_child": visited_children[0],
            "right_child": visited_children[-1],
            "type": "until",
        }

    def visit_until_within_formula(self, node, visited_children):
        return {
            "left_child": visited_children[0],
            "right_child": visited_children[-1],
            "type": "until",
            "within": visited_children[6],
        }

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

    if formula["type"] == "always":
        return probability_of_formula(
            chain,
            {
                "type": "not",
                "child": {
                    "type": "eventually",
                    "child": {
                        "type": "not",
                        "child": formula["child"],
                    },
                },
            },
        )

    if formula["type"] == "eventually":
        n = len(chain._states)
        p = probability_of_formula(chain, formula["child"])
        s = [a for i, a in enumerate(chain._states) if p[i] > 0.0]
        y = chain.state_mask(s)
        s = chain.state_complement(s, chain.predecessors(s))
        b = chain.probability_matrix() @ p
        if "within" in formula:
            a = chain.restrict_tensor_to_states(chain.probability_matrix(), s)
            x = np.zeros(n)
            for _ in range(formula["within"]):
                x = a @ x + b
        else:
            a = chain.restrict_tensor_to_states(
                np.identity(n) - chain.probability_matrix(), s
            )
            if np.linalg.matrix_rank(a) == n:
                x = np.linalg.solve(a, b)
            else:
                x, *_ = np.linalg.lstsq(a, b, rcond=None)
            x += y
        return x

    if formula["type"] == "not":
        return 1.0 - probability_of_formula(chain, formula["child"])

    if formula["type"] == "until":
        p_left = probability_of_formula(chain, formula["left_child"])
        p_right = probability_of_formula(chain, formula["right_child"])
        s_left = chain.states_from_mask(p_left > 0)
        s1 = chain.states_from_mask(p_right > 0)
        s0 = chain.state_complement(s_left + s1)
        s = chain.state_complement(s0 + s1)
        a = chain.restrict_tensor_to_states(chain.probability_matrix(), s)
        b = chain.probability_matrix() @ p_right
        x = np.zeros(len(chain._states))
        if "within" in formula:
            for _ in range(formula["within"]):
                x = a @ x + b
        else:
            x = fixed_point(lambda y: a @ y + b, x)
        return x

    if formula["type"] == "states":
        return np.array([float(s in formula["states"]) for s in chain._states])

    raise RuntimeError("Unknown formula type '%s'." % formula["type"])
