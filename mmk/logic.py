"""
Definition of the LTL-style logic for Markov chains.
"""
__docformat__ = "google"

from parsimonious import Grammar, NodeVisitor

grammar = Grammar(
    r"""
    formula         = diamond_formula / state
    diamond_formula = diamond ws formula

    diamond = "eventually" / "<>"
    state   = ~"[A-Z0-9_]+"i
    ws      = ~"\s+"
    """
)

# pylint: disable=C0116, R0201, W0613
class FormulaVisitor(NodeVisitor):
    """
    Formula visitor
    """

    def visit_formula(self, node, visited_children):
        return visited_children[0]

    def visit_diamond_formula(self, node, visited_children):
        return {
            "child": visited_children[-1],
            "type": "eventually",
        }

    def visit_state(self, node, visited_children):
        return {"state": node.text}

    def generic_visit(self, node, visited_children):
        return None


def parse(formula: str) -> dict:
    """
    Parses a formula and returns a nested dict representing its structure. For
    example, the formula

        <> state_A

    returns

        {
            "type": "eventually"
            "child": {"state": "state_A"},
        }

    """
    tree = grammar.parse(formula)
    visitor = FormulaVisitor()
    return visitor.visit(tree)
