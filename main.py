#!/usr/bin/python3
import graphs as g
import argparse

parser = argparse.ArgumentParser('Build and simplify DeBruijn Graph')
parser.add_argument('--fastq', type=str, required=True, help='Regular fastq file with reads')
parser.add_argument('-k', type=int, required=True, help='k-mer size')
parser.add_argument('--draw', action='store_true', default=False, help='Draw graph in an interactive window')
parser.add_argument('-l', '--adjacency_list', action='store_true', default=False, help='Print adjacency list')
parser.add_argument('-f', '--draw_full_nodes', action='store_true', default=False, help='Draw full nodes value')


if __name__ == '__main__':

    args = parser.parse_args()
    graph = g.DeBruijnGraph()
    graph.build_from_fastq(args.fastq, args.k)
    graph.simplify()
    if args.adjacency_list:
        for line in graph.adjacency_list():
            print(line)
    if args.draw:
        if args.draw_full_nodes:
            print('Full nodes')
            graph.draw(draw_full_nodes=True)
        else:
            graph.draw()
