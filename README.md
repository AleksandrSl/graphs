## graphs
Simple implementation of graphs. Support Directed, Undirected and DeBruijn graphs.

Work with python3

Install requirements via:
```
pip install -r requirements.txt
```
If you want to draw graphs, install [graph-tool](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions) manually

Simple example:

This will build the graph from file simple_reads.fastq with k = 3 and draw it in interactive window with full nodes values
```
./main.py --fastq simple_reads.fastq -k 3 --draw -f
```

To see all options run:
```
./main.py --help
```