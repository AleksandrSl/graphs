import random
import typing
import unittest

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

    def test_oriented_node_equality(self):
        node = []
        seqs = ['ACGT', 'AGCT', 'ATTT', 'ACGT', 'GGGG', 'GGGG']
        # for i in range(10):
        #     seq = ''.join(random.choice('ACGT') for _ in range(4))
        for i, seq in enumerate(seqs):
            node.append(g.DirectedNode(seq, i))

        node[0].add_in_edges(node[1], node[5])
        node[3].add_in_edges(node[1], node[4])
        node[2].add_in_edges(node[3:4])
        node[4].add_in_edges(node[1], node[5])
        self.assertNotEqual(node[0], node[4])  # Node values differ
        self.assertEqual(node[0], node[3])  # In_edges equal, values equal
        self.assertNotEqual(node[1], node[2])  # In_edges not equal, values equal

        node[4].add_out_edges(node[1])
        node[5].add_out_edges(node[3])
        node[2].add_out_edges(node[3])

        self.assertEqual(node[5], node[4])  # Out_edges equal, values equal
        self.assertNotEqual(node[0], node[3])  # Out_edges not equal, values equal


if __name__ == '__main__':
    unittest.main()