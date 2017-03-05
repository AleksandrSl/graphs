from collections import defaultdict, namedtuple
from itertools import chain
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
        if len(nodes) == 1 and issubclass(nodes, list):
            nodes = nodes[0]
        for node in nodes:
            self.in_edges[node] += 1

    def add_out_edges(self, *nodes: typing.Union[typing.Iterable['DirectedNode'], typing.List['DirectedNode']]) -> None:
        """Add multiple out edges.

        :param node: Nodes will be connected to
        :return: None
        """
        if len(nodes) == 1 and issubclass(nodes, list):
            nodes = nodes[0]
        for node in nodes[0]:
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

    def get_out_node(self, value: str) -> 'DirectedNode':
        """Get out node specified by value. If value is not provided return random node.

        :return:  DirectedNode
        """
        # Not strictly random
        for node in self.out_edges:
            return node

    def pop_in_node(self) -> typing.Optional[typing.Tuple['DirectedNode', int]]:
        """Pop in node with all edges leading to it from this node.

        :return: (DirectedNode, int) or (None, None)
        """
        # TODO delete this node from popped node
        if not self.in_edges:
            return None, None
        return self.in_edges.popitem()

    def pop_out_node(self) -> typing.Optional[typing.Tuple['DirectedNode', int]]:
        """Pop out node with all edges leading to it from this node.

        :return: (DirectedNode, int) or (None, None)
        """
        # TODO delete this node from popped node
        if not self.out_edges:
            return None, None
        return self.out_edges.popitem()

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
        if self.in_edges != other.in_edges:
            return False
        if self.out_edges != other.out_edges:
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

    def get_node(self, value) -> typing.Optional[DirectedNode]:
        """Get node with specified value.

        :param value: Value of the node
        :return: DirectedNode or None
        """
        if value in self.nodes:
            return self.nodes[value]

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

    def __init__(self, k) -> None:
        """
        :param k: k-mer length
        """
        self.eulerian_walk = []
        self.k = k
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

    def simplify(self) -> None:
        """Turn all paths that contain nodes with one in edge and one out edge into one node.

        :return: None
        """
        visited = set()

        def collapse(node: DirectedNode) -> None:
            visited.add(node)
            value_to_append = []
            out_nodes_count = node.out_nodes_count
            out_degree = node.out_degree
            while out_nodes_count == 1 and out_degree == 1:
                next_node = node.get_out_node()
                print('Next node {}'.format(next_node))
                if next_node.in_nodes_count != 1:
                    print('Break on node {}'.format(next_node))
                    break
                visited.add(next_node)
                value_to_append.append(next_node.value[self.k - 1:])
                out_nodes_count = next_node.out_nodes_count
                out_degree = next_node.out_degree
                self.nodes.pop(next_node.value)
                node.out_edges = next_node.out_edges
            if value_to_append:
                new_value = node.value + ''.join(value_to_append)
                self.reset_node_value(node.value, new_value)

        for read in self.nodes.copy():
            if read not in self.nodes:
                continue
            node = self.nodes[read]
            if node not in visited:
                collapse(node)

    def adjacency_list(self) -> typing.List[str]:
        """Represent graph as the adjacency list

        :return: list(str)
        """
        adjacency_list = []
        for vertex in self.nodes.values():
            for vertex_ in vertex.out_nodes:
                adjacency_list.append('{} -> {}'.format(vertex.value, vertex_.value))

    def eulerian_path(self) -> None:
        stack = []
        not_visited = self.nodes.copy()  # Посмотреть что именно копируется
        node = None
        for n in self.nodes.values():  # Find start node with uneven number of edges
            if n.out_nodes_count - n.in_nodes_count == 1:
                node = n
        if not node:
            node = next(iter(not_visited.values()))
        stack.append(node)
        while node.out_edges or stack:
            # print(node.out_edges) TODO: в графе остается куча пустых узлов, надо их удалить
            if not node.out_edges:
                self.eulerian_walk.append(node.value)
                node = stack.pop()
            else:
                stack.append(node)
                node, _ = node.pop_out_edge()
        self.eulerian_walk.reverse()

    def reconstruct_from_eul_walk(self) -> str:
        k = self.k - 1
        if not self.eulerian_walk:
            self.eulerian_path()
        res = [self.eulerian_walk[0]]
        for step in self.eulerian_walk[1:]:
            res.append(step[k:])
        return ''.join(res)

    def draw(self, name=None) -> None:
        g = graph_tool.Graph()
        visited = {}
        for vertex in self.nodes:
            node = self.nodes[vertex]
            if node not in visited:
                visited[node] = g.add_vertex()
            v1 = visited[node]
            for node_, edges_n in node.out_edges.items():
                if node_ not in visited:
                    visited[node_] = g.add_vertex()
                v2 = visited[node_]
                g.add_edge(v1, v2, add_missing=False)

        graph_tool.draw.graph_draw(g, vertex_font_size=1, output_size=(500, 500),
                                   vertex_size=10, vertex_color=[1, 1, 1, 0], output=name)  # vertex_text=g.vertex_index
