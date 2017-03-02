import unittest
import typing
import graphs as g


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
        control_graph = g.DeBruijnGraph(4)
        control_graph.add_edge('ATGC', 'TGCTAGA')
        control_graph.add_edge('GTGC', 'TGCTAGA')
        control_graph.add_edge('TGCTAGA', 'AGAC')
        control_graph.add_edge('TGCTAGA', 'AGAT')
        graph.simplify()
        self.assertDictEqual(graph.nodes, control_graph.nodes)


if __name__ == '__main__':
    unittest.main()