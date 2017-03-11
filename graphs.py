from collections import defaultdict, namedtuple
from itertools import chain
from tqdm import tqdm
from operator import itemgetter
from abc import abstractmethod, abstractproperty, ABCMeta
import typing
from random import choice

import graph_tool
import graph_tool.draw

WeightedEdge = namedtuple('WeightedEdge', ['vertex', 'weight'])


# NodeState = namedtuple('Node', ['entry_time', 'exit_time'])


class INode:
    # __slots__ = ['__value', '__index']
    def __init__(self, value: typing.Any, index: int) -> None:
        self.__value = value
        self.__index = index

    @property
    def index(self) -> int:
        return self.__index

    @index.setter
    def index(self, index: int) -> None:
        self.__index = index

    @property
    def value(self) -> typing.Any:
        return self.__value

    @value.setter
    def value(self, value: typing.Any):
        self.__value = value

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(self.index)  # Не value, так как при упрощении графа в value оказывается список

    def __repr__(self) -> str:
        return 'Node({}, {})'.format(self.value, self.index)

    def __eq__(self, other):
        return self.value == other.value

    # TODO: implement deepcopy method
    # @classmethod или лучше заменить __class__ на classmethod
    # def __deepcopy__(self, memodict={}):
    #     copy = self.__class__(self.value, self.index)
    #     copy.in_edges = self.in_edges[:]
    #     copy.out_edges = self.out_edges[:]
    #     return copy


class UndirectedNode(INode):
    __slots__ = ['__value', '__index', 'edges']

    def __init__(self, value: object, index: int) -> None:
        self.edges = defaultdict(int)
        super().__init__(value, index)

    def add_edge(self, node: 'UndirectedNode') -> None:
        """Add new edge.

        :param node: node which will be connected with this node
        """
        self.edges[node] += 1

    def del_edge(self, node: 'UndirectedNode') -> None:
        """Delete edge.

        :param node: node edge to which will be removed
        """
        if self.edges[node]:
            self.edges[node] -= 1
        else:
            self.edges.pop(node)

    @property
    def degree(self) -> int:
        """Degree of the edge.

        :return: number of connected nodes
        """
        return len(self.edges)

    def neighbours(self) -> typing.Generator['UndirectedNode', None, None]:
        """Get generator over connected nodes.

        :return: generator over nodes connected with this node
        """
        for node in self.edges:
            yield node


class DirectedNode(INode):
    __slots__ = ['__value', '__index', 'out_edges', 'in_edges']

    def __init__(self, value: object, index: int) -> None:
        self.out_edges = defaultdict(int)
        self.in_edges = defaultdict(int)
        super().__init__(value, index)

    def add_in_edge(self, node: 'DirectedNode') -> None:
        """Add in edge.

        :param node: Node will be connected to
        :return: None
        """
        self.in_edges[node] += 1

    def add_out_edge(self, node: 'DirectedNode') -> None:
        """Add out edge.

        :param node: Node will be connected to
        :return: None
        """
        self.out_edges[node] += 1

    def add_in_edges(self, *nodes: typing.Union[typing.Iterable['DirectedNode'], typing.List['DirectedNode']]) -> None:
        """Add multiple in edges.

        :param node: Nodes will be connected to
        :return: None
        """
        if len(nodes) == 1 and isinstance(nodes[0], list):
            nodes = nodes[0]
        for node in nodes:
            self.in_edges[node] += 1

    def add_out_edges(self, *nodes: typing.Union[typing.Iterable['DirectedNode'], typing.List['DirectedNode']]) -> None:
        """Add multiple out edges.

        :param node: Nodes will be connected to
        :return: None
        """
        if len(nodes) == 1 and isinstance(nodes, list):
            nodes = nodes[0]
        for node in nodes:
            self.out_edges[node] += 1

    def del_in_edge(self, node: 'DirectedNode') -> None:
        """Delete in edge.

        :param node: Node to delete
        :return: None
        """
        # Raise error on absence or return special(desired value)?
        if self.out_edges[node]:
            self.out_edges[node] -= 1
        else:
            self.out_edges.pop(node)

    def del_out_edge(self, node: 'DirectedNode') -> None:
        """Delete out edge.

        :param node: Node to delete
        :return: None
        """
        # Raise error on absence or return special(desired value)?
        if self.out_edges[node]:
            self.out_edges[node] -= 1
        else:
            self.out_edges.pop(node)

    def del_out_node(self, node: 'DirectedNode') -> None:
        """Delete out node with all it's corresponding edges.

        :param node: Node to delete
        :return: None
        """
        if node in self.out_edges:
            self.out_edges.pop(node)

    def del_in_node(self, node: 'DirectedNode') -> None:
        """Delete in node with all it's corresponding edges.

        :param node: Node to delete
        :return: None
        """
        if node in self.in_edges:
            self.in_edges.pop(node)

    @property
    def in_degree(self) -> int:
        """
        :return: Number of in edges
        """
        return sum(self.in_edges.values())

    @property
    def out_degree(self) -> int:
        """Number of out edges

        :return: int
        """
        return sum(self.out_edges.values())

    def get_out_node(self, value: str=None) -> 'DirectedNode':
        """Get out node specified by value. If value is not provided return random node.

        :return:  DirectedNode
        """
        # Not strictly random
        for node in self.out_edges:
            if not value:
                return node
            elif node.value == value:
                return node


    def pop_in_node(self) -> typing.Union[typing.Tuple['DirectedNode', int], typing.Tuple[None, None]]:
        """Pop in node with all edges leading to it from this node.

        :return: (DirectedNode, int) or (None, None)
        """
        # TODO delete this node from popped node
        if not self.in_edges:
            return None, None
        return self.in_edges.popitem()

    def pop_out_node(self) -> typing.Union[typing.Tuple['DirectedNode', int], typing.Tuple[None, None]]:
        """Pop out node with all edges leading to it from this node.

        :return: (DirectedNode, int) or (None, None)
        """
        # TODO delete this node from popped node
        if not self.out_edges:
            return None, None
        return self.out_edges.popitem()

    def pop_out_edge(self) -> typing.Optional['DirectedNode']:
        """Delete out edge and return connected node.

        :return: (DirectedNode) or (None)
        """
        if self.out_edges:
            for node in self.out_edges:
                self.del_out_edge(node)
                node.del_in_edge(self)
                return node


    @property
    def out_nodes_count(self) -> int:
        """
        :return: Number of out nodes
        """
        return len(self.out_edges)

    @property
    def in_nodes_count(self) -> int:
        """
        :return: Number of in nodes
        """
        return len(self.in_edges)

    @property
    def neighbour_nodes_count(self) -> int:
        """
        :return: Number of in and out nodes
        """
        return self.in_nodes_count + self.out_nodes_count

    def out_nodes(self) -> typing.Generator['DirectedNode', None, None]:
        """
        :return: Generator over out nodes
        """
        for child in self.out_edges:
            yield child

    def in_nodes(self) -> typing.Generator['DirectedNode', None, None]:
        """
        :return: Generator over out nodes
        """
        for parent in self.in_edges:
            yield parent

    def neighbours(self) -> typing.Generator['DirectedNode', None, None]:
        """
        :return: Generator over in and out nodes
        """
        return chain(self.out_edges, self.in_edges)

    def __eq__(self, other: 'DirectedNode'):
        if self.value != other.value:
            return False
        return True

    def __hash__(self) -> int:
        return hash(self.index)  # Не value, так как при упрощении графа в value оказывается список


class Graph(metaclass=ABCMeta):
    __slots__ = ['nodes', 'size']

    def __init__(self) -> None:
        self.nodes = {}
        self.size = 0

    @abstractmethod
    def add_edge(self, value1, value2) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_node(self, value) -> None:
        """Add new node with specified value. Do nothing if such node is already exists.

        :param value: Value for the node
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def adjacency_list(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_node(self, value) -> None:
        raise NotImplementedError


class DirectedGraph(Graph):
    __slots__ = ['nodes', 'cycles', 'node_state', 'size']

    def __init__(self) -> None:
        self.cycles = []
        self.node_state = None
        super().__init__()

    def find_start(self) -> typing.Optional[DirectedNode]:
        """Get node without incoming edges.

        :return: DirectedNode or None
        """
        for node in self.nodes.values():
            if node.in_degree == 0:
                return node

    def __eq__(self, other: 'DirectedGraph'):
        for node in self.nodes:
            if node not in other.nodes:
                return False
        for value, node in self.nodes.items():
            if node != other.nodes[value]:
                return False
        # TODO: Implement comparison by out_nodes and in_nodes

    def add_edge(self, value1, value2) -> None:
        """Add new edge from value1 to value2. Create nodes if they don't exist

        :param value1: Value for the first node
        :param value2: Value for the second node
        :return: None
        """
        self.add_node(value1)
        self.add_node(value2)
        node1 = self.nodes[value1]
        node2 = self.nodes[value2]
        node1.add_out_edge(node2)
        node2.add_in_edge(node1)

    def add_node(self, value) -> None:
        """Add new node with specified value. Do nothing if such node is already exists.

        :param value: Value for the node
        :return: None
        """
        if value not in self.nodes:
            self.size += 1
            self.nodes[value] = DirectedNode(value, self.size)

    def get_node(self, value=None) -> typing.Optional[DirectedNode]:
        """Get node with specified value. If no value is given get first(random?) node

        :param value: Value of the node
        :return: DirectedNode
        """
        if not value:
            for node in self.nodes:
                return node
        if value in self.nodes:
            return self.nodes[value]

    def get_uneven_node(self) -> 'DirectedNode':
        """Get node with in_degree isn't equal out_degree

        :return: DirectedNode
        """
        for node in self.nodes.values():  # Find start node with uneven number of edges
            if node.out_nodes_count - node.in_nodes_count == 1:
                return node
        return(self.get_node())

    def adjacency_list(self):
        raise NotImplementedError

    def set_exit_time(self, start_value=None):
        """Set entry and exit times for the nodes to find cycles.

        :param start_value:
        :return:
        """
        if start_value is None:
            start_node = self.find_start()
        else:
            start_node = self.get_node(start_value)

        self.node_state = {n: [0] * 2 for n in self.nodes.values()}
        path = []
        time = 1

        def dfs(n):
            nonlocal time
            for n in n.out_nodes:
                path.append(n)
                if not self.node_state[n][0]:
                    self.node_state[n][0] = time
                elif not self.node_state[n][1]:
                    self.cycles.append(path[path.index(n):])
                    return
                dfs(n)
                self.node_state[n][1] = time
                time += 1
                path.pop()

        self.node_state[start_node][0] = 1
        path.append(start_node)
        dfs(start_node)
        self.node_state[start_node][1] = time

    def topology_sort(self) -> None:
        """Set indexes of nodes due to their positions after topology sort

        :return: None
        """
        if self.cycles:
            print('No topology sort is avaliable since graph has cycles')
            return
        for node in self.nodes.values():
            node.index = self.length - self.node_state[node][1]

            # def connected_components(self):
            #     connected_components = 0
            #     for v in self.nodes.values():
            #         if not v.visited:
            #             v.visited = True
            #             connected_components += 1
            #             self.dfs(v)
            #     return connected_components


class UndirectedGraph(Graph):

    def add_edge(self, value1: typing.Any, value2: typing.Any) -> None:
        """Add new edge from value1 to value2. Create nodes if they don't exist

        :param value1: Value for the first node
        :param value2: Value for the second node
        :return: None
        """
        node1 = self.get_node(value1)
        node2 = self.get_node(value2)
        node1.add_edge(node2)
        node2.add_edge(node1)

    def add_node(self, value: typing.Any) -> None:
        """Add new node with specified value. Do nothing if such node is already exists.

        :param value: Value for the node
        :return: None
        """
        if value not in self.nodes:
            self.size += 1
            self.nodes[value] = UndirectedNode(value, self.size)

    def get_node(self, value: typing.Any) -> typing.Optional[UndirectedNode]:
        """Get node with specified value.

        :param value: Value of the node
        :return: DirectedNode or None
        """
        if value in self.nodes:
            return self.nodes[value]

    def adjacency_list(self):
        raise NotImplementedError

        # def dfs(self, node):
        #     for v in node.:
        #         if not v.visited:
        #             v.visited = True
        #             self.dfs(v)
        #
        # def connected_components(self):
        #     connected_components = 0
        #     for v in self.nodes.values():
        #         if not v.visited:
        #             v.visited = True
        #             connected_components += 1
        #             self.dfs(v)
        #     return connected_components


class DeBruijnGraph(DirectedGraph):

    def __init__(self) -> None:
        """
        :param k: k-mer length
        """
        self.eulerian_path = []
        super().__init__()

    def reset_node_value(self, old_value: str, new_value: str) -> None:
        """Set new value for the node.

        :param old_value: Old node value
        :param new_value: New node value
        :return: None
        """
        node = self.nodes.pop(old_value)
        node.value = new_value
        self.nodes[new_value] = node

    def build_from_fastq(self, fastq_file_name: str, k: int) -> None:
        """Build the graph from reads in fastq format

        :param fastq_file_name: Fastq with reads
        :param k: Size of kmer
        :return: None
        """
        self.k = k
        with open(fastq_file_name, 'r') as in_f:
            for line in tqdm(in_f):
                # Skip name line
                read = next(in_f).strip()
                for i in range(len(read) - k):
                    value1 = read[i: i + k]
                    value2 = read[i + 1: i + k + 1]
                    self.add_edge(value1, value2)
                next(in_f)  # Skip +
                next(in_f)  # Skip quality string


    def simplify(self) -> None:
        """Turn all paths that contain nodes with one in edge and one out edge into one node.

        :return: None
        """
        visited = set()
        nodes_to_remove = set()

        def collapse(node: DirectedNode) -> None:
            visited.add(node)
            value_to_append = []
            out_nodes_count = node.out_nodes_count
            while out_nodes_count == 1:
                next_node = node.get_out_node()
                # print('Next node {}'.format(next_node))
                if next_node.in_nodes_count != 1:
                    # print('Break on node {}'.format(next_node))
                    break
                visited.add(next_node)
                value_to_append.append(next_node.value[self.k - 1:])
                out_nodes_count = next_node.out_nodes_count
                nodes_to_remove.add(next_node)
                node.out_edges = next_node.out_edges
                # if len(node.out_edges) > 1:
                #     print('Last value to append {}'.format(node.get_out_node().value[self.k - 1:]))


            if value_to_append:
                new_value = node.value + ''.join(value_to_append)
                self.reset_node_value(node.value, new_value)

        for read in self.nodes.copy():
            node = self.nodes[read]
            if node not in visited:
                collapse(node)
        # print(self.adjacency_list())
        self.del_nodes(nodes_to_remove)

    def del_nodes(self, nodes: typing.Iterable[DirectedNode]) -> None:
        for node in nodes:
            for in_node in node.in_nodes():
                # print('Before {}'.format(in_node.out_edges))
                in_node.del_out_node(in_node)
                # print('After {}'.format(in_node.out_edges))
            for out_node in node.out_nodes():
                out_node.del_in_node(out_node)
            self.nodes.pop(node.value)

    def adjacency_list(self) -> typing.List[str]:
        """Represent graph as the adjacency list.

        :return: list(str)
        """
        adjacency_list = []
        for in_node in self.nodes.values():
            for out_node in in_node.out_nodes():
                adjacency_list.append('{} -> {}'.format(in_node.value, out_node.value))
        return adjacency_list

    def draw(self, output_file_name: str=None) -> None:
        """Draw graph to screen or to file if output_file_name is provided.

        :param output_file_name: File to draw graph to
        :return: None
        """
        graph_to_draw = graph_tool.Graph()
        # print(len(self.nodes))
        visited = {}
        values = graph_to_draw.new_vertex_property('string')
        # print('Nodes to draw {}'.format(self.nodes))
        for node_value in self.nodes:
            node = self.nodes[node_value]
            if node not in visited:
                visited[node] = graph_to_draw.add_vertex()
            node1 = visited[node]
            values[node1] = node.value[:self.k]
            for out_node in node.out_nodes():
                if out_node not in visited:
                    visited[out_node] = graph_to_draw.add_vertex()
                node2 = visited[out_node]
                values[node2] = out_node.value[-self.k:]
                graph_to_draw.add_edge(node1, node2, add_missing=False)

        # print(values)
        if output_file_name:
            graph_tool.draw.graph_draw(graph_to_draw, vertex_font_size=1, output_size=(1000, 10),
                                       vertex_size=1, vertex_color=[1, 1, 1, 0], output=output_file_name)
        else:
            graph_tool.draw.graph_draw(graph_to_draw,nodesfirst=True, vertex_text=values,
                                       vertex_size = 0.1, vertex_font_size=10, edge_pen_width = 5,
                                       edge_color=[255, 255, 0, 1]) #, vertex_font_size=1, output_size=(1000, 1000),
                                       # vertex_size=1, vertex_color=[1, 1, 1, 0])

    def find_eulerian_path(self) -> None:
        """Find eulerian path in graph. It will be set to eulerian_path graph attribute.

        :return: None
        """
        stack = []
        eulerian_path = []
        node = self.get_uneven_node()
        stack.append(node)
        while node.out_edges or stack:
            # print(node.out_edges) TODO: в графе остается куча пустых узлов, надо их удалить
            if not node.out_edges:
                eulerian_path.append(node.value)
                node = stack.pop()
            else:
                stack.append(node)
                node = node.pop_out_edge()
        eulerian_path.reverse()

        k = self.k - 1
        self.eulerian_path.append(eulerian_path[0])
        for step in self.eulerian_path[1:]:
            self.eulerian_path.append(step[k:])
        self.eulerian_path = ''.join(self.eulerian_path)
