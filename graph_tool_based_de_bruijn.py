import graph_tool
from graph_tool.draw import graph_draw
import typing

# class DeBruijnNode(graph_tool.Vertex):
#
#     def get_out_node(self):


class DeBruijnGraph(graph_tool.Graph):


    def __init__(self, k) -> None:
        """
        :param k: k-mer length
        """
        super().__init__()
        self.eulerian_walk = []
        self.k = k
        self.values = self.new_vertex_property('string')
        self.indexes = {}

    def add_edge(self, value1, value2, add_missing=True):
        # ToDo check arguments
        """Add new edge from value1 to value2. Create nodes if they don't exist

        :param value1: Value for the first node
        :param value2: Value for the second node
        :return: None
        """
        node1 = self.add_node(value1, return_node=True)
        node2 = self.add_node(value2, return_node=True)
        super(DeBruijnGraph, self).add_edge(node1, node2, False)

    def add_edge_list(self, edge_list, hashed=False, string_vals=False, eprops=None):
        super(DeBruijnGraph, self).add_edge_list((self.vertex_index[node1], self.vertex_index[node2])
                                                  for node1, node2 in edge_list)

    def add_node(self, value: str, return_node: bool=False) -> None:
        """Add a vertex to the graph, and return it. If ``n != 1``, ``n``
        vertices are inserted and an iterator over the new vertices is returned.
        This operation is :math:`O(n)`.
        """
        if value in self.values:
            node = self.vertex(self.indexes[value])
        else:
            node = self.add_vertex()
            self.values[node] = value
            self.indexes[value] = self.vertex_index[node]
        if return_node:
            return node

    def reset_node_value(self, node: graph_tool.Vertex, new_value: str) -> None:
        """Set new value for the node.

        :param old_value: Old node value
        :param new_value: New node value
        :return: None
        """
        self.values[node] = new_value

    def simplify(self) -> None:
        """Turn all paths that contain nodes with one in edge and one out edge into one node.

        :return: None
        """
        visited = set()

        def collapse(node: graph_tool.Vertex) -> None:
            visited.add(node)
            value_to_append = []
            out_degree = node.out_degree()
            out_neighbours = node.out_neighbours()
            while out_degree == 1:
                for next_node in out_neighbours:
                    break
                print('Next node {}'.format(next_node))
                if next_node.in_degree() != 1:
                    print('Break on node {}'.format(next_node))
                    break
                visited.add(next_node)
                value_to_append.append(self.values[next_node][self.k - 1:])
                out_degree = next_node.out_degree()
                out_neighbours = list(next_node.out_neighbours())
                # self.remove_vertex(next_node)
            if value_to_append:
                new_value = self.values[node] + ''.join(value_to_append)
                self.reset_node_value(node, new_value)

                edge_list = [(node, node1) for node1 in out_neighbours]
                print(edge_list)
                self.add_edge_list(edge_list)

        for read in self.vertices():
            if read not in self.vertices():
                continue
            node = self.vertex(read)
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

