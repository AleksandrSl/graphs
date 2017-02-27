from collections import defaultdict, namedtuple
from itertools import chain
from operator import itemgetter
from abc import abstractmethod, ABCMeta
import typing
from random import choice

import graph_tool
import graph_tool.draw

WeightedEdge = namedtuple('WeightedEdge', ['vertex', 'weight'])


# NodeState = namedtuple('Node', ['entry_time', 'exit_time'])


class INode:
    # __slots__ = ['__value', '__index']
    def __init__(self, value: object, index: int) -> None:
        self.__value = value
        self.__index = index

    @property
    def index(self) -> int:
        return self.__index

    @index.setter
    def index(self, index: int) -> None:
        self.__index = index

    @property
    def value(self) -> object:
        return self.__value

    @value.setter
    def value(self, value: object):
        self.__value = value

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(self.index)  # Не value, так как при упрощении графа в value оказывается список

    def __repr__(self) -> str:
        return 'Node({}, {})'.format(self.value, self.index)

        # TODO: implement deepcopy method
        # @classmethod или лучше заменить __class__ на classmethod
        # def __deepcopy__(self, memodict={}):
        #     copy = self.__class__(self.value, self.index)
        #     copy.in_edges = self.in_edges[:]
        #     copy.out_edges = self.out_edges[:]
        #     return copy


class UnorientedNode(INode):
    __slots__ = ['__value', '__index', 'edges']

    def __init__(self, value: object, index: int) -> None:
        self.edges = defaultdict(int)
        super().__init__(value, index)

    def add_edge(self, node: 'UnorientedNode') -> None:
        """
        Add new edge
        :param node: node which will be connected with this node
        """
        self.edges[node] += 1

    def del_edge(self, node: 'UnorientedNode') -> None:
        """
        Delete edge
        :param node: node edge to which will be removed
        """
        if self.edges[node]:
            self.edges[node] -= 1
        else:
            self.edges.pop(node)

    @property
    def degree(self) -> int:
        """
        Get degree of the edge
        :return: number of connected nodes
        """
        return len(self.edges)

    def neighbours(self) -> typing.Generator['UnorientedNode', None, None]:
        """
        Generator over connected nodes
        :return: generator over nodes connected with this node
        """
        for node in self.edges:
            yield node


class OrientedNode(INode):
    __slots__ = ['__value', '__index', 'out_edges', 'in_edges']

    def __init__(self, value: object, index: int) -> None:
        self.out_edges = defaultdict(int)
        self.in_edges = defaultdict(int)
        super().__init__(value, index)

    def add_out_edge(self, node: 'OrientedNode') -> None:
        self.out_edges[node] += 1

    def add_in_edge(self, node: 'OrientedNode') -> None:
        self.in_edges[node] += 1

    def del_out_edge(self, node: 'OrientedNode') -> None:
        if self.out_edges[node]:
            self.out_edges[node] -= 1
        else:
            self.out_edges.pop(node)

    def del_in_edge(self, node: 'OrientedNode') -> None:
        if self.out_edges[node]:
            self.out_edges[node] -= 1
        else:
            self.out_edges.pop(node)

    @property
    def in_degree(self) -> int:
        return sum(self.in_edges.values())

    @property
    def out_degree(self) -> int:
        return sum(self.out_edges.values())

    def get_out_node(self) -> 'OrientedNode':
        return choice(self.out_edges)

    def pop_out_node(self) -> typing.Optional[typing.Tuple('OrientedNode', int)]:
        """
        Pop out node with all edges leading to it
        :return: Node and number of edges leading to this node
        """
        # TODO delete this node from popped node
        if not self.out_edges:
            return None, None
        return self.out_edges.popitem()

    def pop_in_node(self) -> typing.Optional[typing.Tuple('OrientedNode', int)]:
        """
        Pop out node with all edges leading to it
        :return: Node and number of edges leading to this node
        """
        # TODO delete this node from popped node
        if not self.in_edges:
            return None, None
        return self.in_edges.popitem()

    @property
    def children_count(self) -> int:
        return len(self.out_edges)

    @property
    def parents_count(self) -> int:
        return len(self.in_edges)

    @property
    def neighbours_count(self) -> int:
        return self.parents_count + self.children_count

    def children(self) -> typing.Generator['OrientedNode', None, None]:
        for child in self.out_edges:
            yield child

    def parents(self) -> typing.Generator['OrientedNode', None, None]:
        for parent in self.in_edges:
            yield parent

    def neighbours(self) -> typing.Generator['OrientedNode', None, None]:
        return chain(self.out_edges, self.in_edges)


class WeightedNode(OrientedNode):

    def __init__(self, value, index):
        super().__init__(value, index)  # Костыль, так как иначе in/out_edges определяются как словари из OrientedNode
        self.out_edges = {}
        self.out_edges_sorted_by_weights = []
        self.out_edges_to_sort = []
        self.in_edges = {}
        self.in_edges_to_sort = []
        self.in_edges_sorted_by_weights = []

    def add_in_edge(self, node: OrientedNode, weight: typing.Any) -> None:
        """
        Add incoming edge
        :param node: 
        :param weight: 
        """
        self.in_edges[node] = weight
        self.in_edges_to_sort.append(WeightedEdge(node, weight))

    def add_out_edge(self, node: OrientedNode, weight: typing.Any) -> None:
        """
        Add outcoming edge
        :param node:
        :param weight:
        """
        self.out_edges[node] = weight
        self.out_edges_to_sort.append(WeightedEdge(node, weight))

    def del_in_edge(self, node, weight):
        pass
        # if weight in self.in_edges[node]:
        #     self.in_edges[node].remove(weight)
        #     if not self.in_edges[node]:
        #         self.in_edges.pop(node)
        # else:
        #     print('Edge doesn\'t exist')

    def del_out_edge(self, node, weight):
        pass
        # if weight in self.out_edges[node]:
        #     self.out_edges[node].remove(weight)
        #     if not self.out_edges[node]:
        #         self.out_edges.pop(node)
        # else:
        #     print('Edge doesn\'t exist')

    def pop_in_edge(self):
        if not self.in_edges:
            return None, None
        node = choice(self.in_edges.keys())
        return self.in_edges.pop(node)

    def pop_out_edge(self):
        if not self.out_edges:
            return None, None
        node = choice(self.out_edges.keys())
        return self.out_edges.pop(node)

    def sort_out_weight(self, reverse):
        self.out_edges_sorted_by_weights = sorted(self.out_edges_to_sort, key=itemgetter(1), reverse=reverse)

    def sort_in_weight(self, reverse):
        self.in_edges_sorted_by_weights = sorted(self.in_edges_to_sort, key=itemgetter(1), reverse=reverse)


class IGraph:
    __slots__ = ['nodes', 'length']

    def __init__(self) -> None:
        self.nodes = {}
        self.length = 0

    def add_edge(self, value1, value2) -> None:
        raise NotImplementedError

    def add_node(self, value) -> None:
        raise NotImplementedError

    def adjacency_list(self) -> None:
        raise NotImplementedError

    def get_node(self, value) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.length


class OrientedGraph(IGraph):
    __slots__ = ['nodes', 'cycles', 'node_state', 'length']

    def __init__(self) -> None:
        self.cycles = []
        self.node_state = None
        super().__init__()

    def find_start(self) -> typing.Optional[OrientedNode]:
        for n in self.nodes.values():
            if n.in_degree == 0:
                return n
        return None

    def add_edge(self, value1, value2) -> None:
        node1 = self.get_node(value1)
        node2 = self.get_node(value2)
        node1.add_out_edge(node2)
        node2.add_in_edge(node1)

    def add_node(self, value) -> None:
        self.nodes[value] = OrientedNode(value, len(self.nodes) + 1)
        self.length += 1

    def get_node(self, value) -> OrientedNode:
        if value not in self.nodes:
            self.add_node(value)
        return self.nodes[value]

    def adjacency_list(self):
        raise NotImplementedError

    def set_exit_time(self, start_value=None):
        if start_value is None:
            start_node = self.find_start()
        else:
            start_node = self.get_node(start_value)

        self.node_state = {n: [0] * 2 for n in self.nodes.values()}
        path = []
        time = 1

        def dfs(n):
            nonlocal time
            for n in n.children():
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


class UnorientedGraph(IGraph):

    def add_edge(self, value1, value2) -> None:
        node1 = self.get_node(value1)
        node2 = self.get_node(value2)
        node1.add_edge(node2)
        node2.add_edge(node1)

    def add_node(self, value) -> None:
        self.nodes[value] = UnorientedNode(value, len(self.nodes) + 1)

    def get_node(self, value) -> UnorientedNode:
        if value not in self.nodes:
            self.add_node(value)
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


class WeightedGraph(OrientedGraph):

    def add_edge(self, value1, value2, weight) -> None:
        node1 = self.get_node(value1)
        node2 = self.get_node(value2)
        node1.add_out_edge(node2, weight)
        node2.add_in_edge(node1, weight)

    def add_node(self, value) -> None:
        self.nodes[value] = WeightedNode(value, len(self.nodes) + 1)

    def sort_out_by_weight(self, reverse=False) -> None:
        for vertex in self.nodes.values():
            vertex.sort_out_weight(reverse)

    def sort_in_by_weight(self, reverse=False) -> None:
        for vertex in self.nodes.values():
            vertex.sort_in_weight(reverse)

    def dfs(self, node) -> None:
        for weighted_edge in node.children():
            if not weighted_edge.vertex.visited:
                weighted_edge.vertex.visited = True
                self.dfs(weighted_edge.vertex)

    def adjacency_list(self) -> None:
        for vertex in self.nodes.values():
            for weighted_edge in vertex.children():
                print('{} -> {} : {}'.format(vertex.index, weighted_edge.vertex.index, weighted_edge.weight))

    def draw(self, name) -> None:
        g = graph_tool.Graph()
        # print(len(list(self.vertices.values())))
        visited = {}
        for vertex in self.nodes:
            node = self.nodes[vertex]
            if node not in visited:
                visited[node] = g.add_vertex()
            v1 = visited[node]
            for node_, _ in node.children():
                if node_ not in visited:
                    visited[node_] = g.add_vertex()
                v2 = visited[node_]
                g.add_edge(v1, v2, add_missing=False)
        graph_tool.draw.graph_draw(g, vertex_font_size=1, output_size=(1000, 1000),
                                   vertex_size=4, vertex_color=[1, 1, 1, 0], output=name)  # vertex_text=g.vertex_index

    def draw_from_start(self, start) -> None:
        g = graph_tool.Graph()
        vertex_stack = [self.nodes[start]]
        while vertex_stack:
            vertex = vertex_stack.pop()
            for weighted_edge in vertex.children():
                g.add_edge(vertex.index, weighted_edge.vertex.index, add_missing=True)
                vertex_stack.append(weighted_edge.vertex)

        graph_tool.draw.graph_draw(g, vertex_text=g.vertex_index, vertex_font_size=1, output_size=(4000, 4000),
                                   vertex_size=4, vertex_color=[1, 1, 1, 0], output="two-nodes.png")

    def simplify(self) -> None:
        visited = set()

        def collapse(vertex):
            visited.add(vertex)
            vertex.value = [vertex.value]  # !!!!!
            while vertex.children_count == 1:
                if len(vertex.out_edges[0].vertex.in_edges) != 1:
                    break
                vertex_, pos = vertex.pop_out_edge()
                visited.add(vertex_)
                if vertex_.value[pos:]:
                    vertex.value.append(vertex_.value[pos:])  # !!!!!
                vertex.out_edges = vertex_.out_edges
                vertex_.out_edges = None
                if type(vertex_.value) == list:  # !!!!!
                    self.nodes.pop(vertex_.value[0])  # !!!!!
                else:  # !!!!!
                    self.nodes.pop(vertex_.value)  # !!!!!  Всю эту жесть под восклицательными знаками пришлось сделать,
                    # так как вершины выбираются не в порядке следования, и может упроститься конец
                    # а потом начаться упрощения и когда все дойдет до уже упрощенной вершины, если
                    # я просто буду обновлять в ней значение, то я не смогу ее удалить,
                    # так как не буду знать по какому ключу она лежит, а так в списке
                    # хранятся все значения, включая изначальное, но надо как то по-другому сделать

        for read in self.nodes.copy():
            if read not in self.nodes:
                continue
            node = self.nodes[read]
            if node not in visited:
                collapse(node)

        def collapse_values():
            for node in self.nodes.values():
                if type(node.value) == list:
                    node.value = ''.join(node.value)

        collapse_values()

    def hamiltonian_path_recursive(self, start) -> str:  # Жрет слишком много памяти
        node = self.nodes[start]

        def ham(start, path):
            path += str(start.index)
            if len(start.out_edges) == 0:
                return path
            return ':'.join(ham(node_, path) for node_, _ in start.children())

        return ham(node, '')

    def connected_components(self) -> None:
        visited = set()
        count = 0
        while len(visited) != len(self.nodes):
            connected_comp_reads = set()
            for node in self.nodes.values():
                if node not in visited:
                    start_node = node
                    break

            visited.add(start_node)
            connected_comp_reads.add(start_node)
            node_stack = [start_node]
            while node_stack:
                node = node_stack.pop()
                for node, pos in node.neighbours():
                    if node not in visited:
                        visited.add(node)
                        connected_comp_reads.add(node)
                        node_stack.append(node)
            with open('reads' + str(count), 'w') as read_file:  # Для записи ридов компоненты в отдельный файл
                for read in connected_comp_reads:
                    read_file.write(read.value + '\n')
            count += 1
        return count


class DeBruijnGraph(OrientedGraph):

    def __init__(self, k) -> None:
        self.eulerian_walk = []
        self.k = k
        super().__init__()

    def simplify(self) -> None:
        """
        Turn all paths that contains nodes with one in edge and one out edge into one node
        """
        visited = set()
        k = self.k - 1

        def collapse(node: OrientedNode):
            visited.add(node)
            value_to_append = []
            out_nodes = node.children_count
            out_degree = node.out_degree
            while out_nodes == 1 and out_degree == 1:
                next_node = node.get_out_node()
                if next_node.parents_count != 1:
                    break
                node.del_out_edge(next_node)
                visited.add(next_node)
                value_to_append.append(next_node.value[k:])
                out_nodes = next_node.out_edges
                self.nodes.pop(next_node.value)
            node.value += ''.join(value_to_append)
            # print('New value', node.value)
            # print('New value length', len(node.value))

        for read in self.nodes.copy():
            if read not in self.nodes:
                continue
            node = self.nodes[read]
            if node not in visited:
                # print('Start collapsing:', node)
                collapse(node)

    def adjacency_list(self) -> None:
        for vertex in self.nodes.values():
            for vertex_ in vertex.children():
                print('{} -> {}'.format(vertex.value, vertex_.value))

    def eulerian_path(self) -> None:
        stack = []
        not_visited = self.nodes.copy()  # Посмотреть что именно копируется
        node = None
        for n in self.nodes.values():  # Find start node with uneven number of edges
            if n.children_count - n.parents_count == 1:
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

    def draw(self, name) -> None:
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
