#!/usr/bin/env python3

import argparse
from tqdm import tqdm


class Edge:
    show_sequences = False

    def __init__(self, v1, v2):
        if not (isinstance(v1, Vertex) and isinstance(v2, Vertex)):
            raise ValueError()
        self.v1 = v1
        self.v2 = v2
        self.coverage = 1
        self.sequence = v1.sequence + v2.sequence[-1]

    def inc_coverage(self):
        self.coverage += 1

    def __len__(self):
        return len(self.sequence)

    def merge(self, following_edge):
        # Вообще здесь неплохо бы знать k, но это не возможно. Поэтому будем считать,
        # что following_edge всегда размером с k
        new = self.__copy__()
        new.sequence += following_edge.sequence[-1]
        new.coverage = (self.coverage * len(self) + following_edge.coverage * len(following_edge)) \
            / (len(self) + len(following_edge))
        return new

    def __str__(self):
        return self.sequence

    def __copy__(self):
        new = Edge(self.v1, self.v2)
        new.__dict__.update(self.__dict__)
        return new


class Vertex:
    show_sequences = False

    def __init__(self, seq):
        self.in_edges = {}
        self.out_edges = {}
        self.sequence = seq

    def add_edge(self, edge):
        if edge.v1 == self:
            if edge.v2 not in self.out_edges:
                self.out_edges[edge.v2] = edge
            else:
                self.out_edges[edge.v2].inc_coverage()
        elif edge.v2 == self:
            if edge.v1 not in self.in_edges:
                self.in_edges[edge.v1] = edge
            else:
                self.in_edges[edge.v1].inc_coverage()

    def __str__(self):
        return self.sequence

    def __hash__(self):
        return hash(self.sequence)

    def compress(self):
        if len(self.in_edges) == 1:
            for in_node, in_edge in self.in_edges.items():
                pass
            # in_node, in_edge = self.in_edges.popitem()
            if len(in_node.out_edges) == 1:
                in_node.out_edges.popitem()  # Remove edge to current node
                while self.out_edges:
                    vertex, edge = self.out_edges.popitem()
                    new_edge = in_edge.merge(edge)
                    in_node.out_edges[vertex] = new_edge
                    vertex.in_edges[in_node] = new_edge
                    vertex.in_edges.pop(self)

                return True
        return False

    def __eq__(self, other):
        return self.sequence == other.sequence


class Graph:
    show_vertex_sequence = False
    show_edge_sequence = False

    def __init__(self, k):
        self.k = k
        self.vertices = {}

    def add_edge(self, seq1, seq2):
        # Increases coverage if the edge already exists
        if seq1 not in self.vertices:
            self.vertices[seq1] = Vertex(seq1)
        if seq2 not in self.vertices:
            self.vertices[seq2] = Vertex(seq2)
        edge = Edge(self.vertices[seq1], self.vertices[seq2])
        self.vertices[seq1].add_edge(edge)
        self.vertices[seq2].add_edge(edge)

    def add_seq(self, seq):
        for i in range(len(seq) - self.k):
            kmer1 = seq[i: i + self.k]
            kmer2 = seq[i + 1: i + 1 + self.k]
            self.add_edge(kmer1, kmer2)

    def compress(self):

        to_delete = []  # List of redundant vertices

        for kmer, vertex in self.vertices.items():
            if vertex.compress():
                to_delete.append(kmer)
        for kmer in to_delete:
            self.vertices.pop(kmer)

    def save_dot(self, out_f):
        out_f.write('graph debruijn {\n')
        for vertex in self.vertices.values():
            for out_vertex, out_edge in vertex.out_edges.items():
                out_f.write('     {} -- {} [label = "{}"];\n'.format(vertex.sequence,
                                                                     out_vertex.sequence, out_edge.sequence))
        out_f.write('}')

    def adjacency_list(self):
        for vertex in self.vertices.values():
            for out_vertex in vertex.out_edges:
                print('{} -> {}'.format(vertex.sequence, out_vertex.sequence))


complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


def reverse_complement(seq):
    return ''.join(complement[nt] for nt in seq[::-1])


def read_fastq(f):
    for line in f:
        name = line.strip()
        seq = next(f).strip()
        next(f)
        next(f)
        yield name, seq


def read_fasta(f):
    name = None
    seq = None
    for line in f:
        if line.startswith('>'):
            if name:
                yield name, seq
            name = line.lstrip('>').strip()
            seq = ''
        else:
            seq += line.strip()
    yield name, seq


def read(f):
    if f.name.endswith('a'):
        return read_fasta(f)
    else:
        return read_fastq(f)


def main():
    parser = argparse.ArgumentParser(description='De Bruijn graph')
    parser.add_argument('-i', '--input', help='Input fastq', metavar='File',
                        type=argparse.FileType(), required=True)
    parser.add_argument('-k', help='k-mer size (default: 55)', metavar='Int',
                        type=int, default=55)
    parser.add_argument('-o', '--output', help='Output dot', metavar='File',
                        type=argparse.FileType('w'), required=True)
    parser.add_argument('-c', '--compress', help='Shrink graph', action='store_true')
    parser.add_argument('--vertex', help='Show vertex sequences', action='store_true')
    parser.add_argument('--edge', help='Show edge sequences', action='store_true')
    args = parser.parse_args()

    # Vertex.show_sequences = args.vertex_seq
    # Edge.show_sequences = args.edge_seq

    graph = Graph(args.k)
    for name, seq in tqdm(read(args.input)):
        graph.add_seq(seq)
        # graph.add_seq(reverse_complement(seq))

    if args.compress:
        graph.compress()

    graph.adjacency_list()
    graph.save_dot(args.output)


if __name__ == '__main__':
    main()
