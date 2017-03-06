import graph_tool

class DeBruijn():
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
