import random
import typing
import unittest

import graph_tool_based_de_bruijn as g


class TestGraph(unittest.TestCase):

    def test_debrujn_simplify(self):
        graph = g.DeBruijnGraph(4)
        graph.add_edge('ATGC', 'TGCT')
        graph.add_edge('GTGC', 'TGCT')
        graph.add_edge('TGCT', 'GCTA')
        graph.add_edge('GCTA', 'CTAG')
        graph.add_edge('CTAG', 'TAGA')
        graph.add_edge('TAGA', 'AGAT')
        graph.add_edge('TAGA', 'AGAC')
        # print(list(graph.vertices()))
        control_graph = g.DeBruijnGraph(4)
        control_graph.add_edge('ATGC', 'TGCTAGA')
        control_graph.add_edge('GTGC', 'TGCTAGA')
        control_graph.add_edge('TGCTAGA', 'AGAC')
        control_graph.add_edge('TGCTAGA', 'AGAT')
        graph.simplify()
        # g.graph_draw(graph, vertex_text = graph.values)
        # g.graph_draw(control_graph, vertex_text = control_graph.values)
        self.assertEqual(list(graph.vertices()), list(control_graph.vertices()))


if __name__ == '__main__':
    unittest.main()