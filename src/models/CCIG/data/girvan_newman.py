# -*- coding: utf-8 -*-
"""Functions for computing communities based on centrality notions."""
import random
import math
from collections import Counter
from graph_tool.all import *

__all__ = ['girvan_newman']


def selfloop_edges(g):
    es = []
    for e in g.edges():
        if e.source() == e.target():
            es.append(e)
    return es


def get_connected_components(g):
    """
    Given a graph, extract connected components as
    a list of sub-graphs.
    :param g: input graph
    :return: a list of connected component sub-graphs
    """
    c = label_components(g)[0]
    components = []
    c_set = set(c)
    for c_label in c_set:
        u = GraphView(g, vfilt=(c.a == c_label))
        u = Graph(u, prune=True)
        components.append(u)
    return components


def num_connected_components(g):
    """
    Given a graph, count connected components.
    :param g: input graph
    :return: number of connected component sub-graphs
    """
    return len(set(label_components(g)[0]))


def most_valuable_edge(g, eprop_weight=None):
    """
    Given a graph, find the most valuable edge to remove.
    :param g: input graph
    :param eprop_weight:
    :return: the most valuable edge
    """
    bv, be = betweenness(g)
    max_be = be.a.max()
    es = find_edge(g, be, max_be)
    if len(es) == 1:
        return es[0]
    else:
        if eprop_weight is None:
            return random.choice(es)
        else:
            min_ew = min([g.edge_properties[eprop_weight][e] for e in es])
            candidates = [e for e in es if (
                min_ew == g.edge_properties[eprop_weight][e])]
            return random.choice(candidates)


def duplicate_edge_condition(g, e, eprop_weight=None):
    source_cp = 0
    target_cp = 0
    min_cp_to_duplicate_edge = 1 / 1.7
    if eprop_weight is None:
        source_out_edges = g.get_out_edges(e.source())[:, 2]
        source_cp = 1.0 / len(source_out_edges)
        target_out_edges = g.get_out_edges(e.target())[:, 2]
        target_cp = 1.0 / len(target_out_edges)
    else:
        s_from_idx = g.get_out_edges(e.source())[:, 0]
        s_to_idx = g.get_out_edges(e.source())[:, 1]
        num_edges = len(s_from_idx)
        source_out_edges_weight = [g.edge_properties[eprop_weight][
            g.edge(s_from_idx[i], s_to_idx[i])] for i in range(num_edges)]
        total_weight = sum(source_out_edges_weight)
        source_cp = g.edge_properties[eprop_weight][e] / float(total_weight)
        t_from_idx = g.get_out_edges(e.target())[:, 0]
        t_to_idx = g.get_out_edges(e.target())[:, 1]
        num_edges = len(t_from_idx)
        target_out_edges_weight = [g.edge_properties[eprop_weight][
            g.edge(t_from_idx[i], t_to_idx[i])] for i in range(num_edges)]
        total_weight = sum(target_out_edges_weight)
        target_cp = g.edge_properties[eprop_weight][e] / float(total_weight)
    return [source_cp > min_cp_to_duplicate_edge,
            target_cp > min_cp_to_duplicate_edge]


def duplicate_edge(g, e, cp_status, vprop_name, eprop_weight=None):
    copy_target_vertex = cp_status[0]
    copy_source_vertex = cp_status[1]
    if (not copy_target_vertex) and (not copy_source_vertex):
        g.remove_edge(e)
    else:
        v_source_idx = e.source()
        v_target_idx = e.target()
        v_source_name = g.vertex_properties[vprop_name][g.vertex(v_source_idx)]
        v_target_name = g.vertex_properties[vprop_name][g.vertex(v_target_idx)]
        e_weight = 0
        if eprop_weight is not None:
            e_weight = g.edge_properties[eprop_weight][e]
        g.remove_edge(e)

        if copy_target_vertex and (not copy_source_vertex):
            v_dup = g.add_vertex()
            v_dup_idx = g.vertex_index[v_dup]
            g.vertex_properties[vprop_name][v_dup] = v_target_name
            e_dup = g.add_edge(v_source_idx, v_dup_idx)
            if eprop_weight is not None:
                g.edge_properties[eprop_weight][e_dup] = e_weight
        elif (not copy_target_vertex) and copy_source_vertex:
            v_dup = g.add_vertex()
            v_dup_idx = g.vertex_index[v_dup]
            g.vertex_properties[vprop_name][v_dup] = v_source_name
            e_dup = g.add_edge(v_dup_idx, v_target_idx)
            if eprop_weight is not None:
                g.edge_properties[eprop_weight][e_dup] = e_weight
        else:
            v_dup = g.add_vertex()
            v_dup_idx = g.vertex_index[v_dup]
            g.vertex_properties[vprop_name][v_dup] = v_target_name
            e_dup = g.add_edge(v_source_idx, v_dup_idx)
            if eprop_weight is not None:
                g.edge_properties[eprop_weight][e_dup] = e_weight
            v_dup = g.add_vertex()
            v_dup_idx = g.vertex_index[v_dup]
            g.vertex_properties[vprop_name][v_dup] = v_source_name
            e_dup = g.add_edge(v_dup_idx, v_target_idx)
            if eprop_weight is not None:
                g.edge_properties[eprop_weight][e_dup] = e_weight
    return g


def stop_condition(g, betweenness_threshold_coef=1.0, max_c_size=10, min_c_size=3):
    """
    Given a graph, decide whether stop community detection or not.
    """
    graph_size = g.num_vertices()
    if graph_size <= min_c_size:
        return True
    possible_path = min(graph_size * (graph_size - 1) / 2,
                        max_c_size * (max_c_size - 1) / 2)
    # NOTICE: if betweenness_threshold_coef smaller, the community can be smaller.
    threshold = betweenness_threshold_coef * math.log(possible_path) / math.log(2) + 1

    bv, be = betweenness(g)
    max_be = be.a.max()
    max_betweenness = max_be * graph_size * (graph_size - 1) / 2.0

    # print max_betweenness
    # print threshold

    if (graph_size > min_c_size and max_betweenness > threshold) or (graph_size > max_c_size):
        return False
    else:
        return True


def is_finalized(u):
    for v in u.vertices():
        if u.vertex_properties["finalized"][v] == "N":
            return False
    return True


def set_finalized(u):
    for v in u.vertices():
        u.vertex_properties["finalized"][v] = "Y"
    return u


def set_not_finalized(u):
    for v in u.vertices():
        u.vertex_properties["finalized"][v] = "N"
    return u


def girvan_newman(G, vprop_name, eprop_weight=None, betweenness_threshold_coef=1.0, max_c_size=10, min_c_size=3):
    """
    Finds communities in a graph using the Girvan–Newman method.
    The Girvan–Newman algorithm detects communities by progressively
    removing edges from the original graph. The algorithm removes the
    "most valuable" edge, traditionally the edge with the highest
    betweenness centrality, at each step.
    """
    # If the graph is already empty,
    # simply return its connected components.
    if G.num_edges() == 0:
        # yield get_connected_components(g)
        return G

    # The copy of G here must include the edge weight data.
    g = Graph(G)
    vprop_finalized = g.new_vertex_property("string")
    g.vertex_properties["finalized"] = vprop_finalized
    g = set_not_finalized(g)

    # Self-loops must be removed because their removal has no effect on
    # the connected components of the graph.
    loop_es = selfloop_edges(g)
    for e in loop_es:
        g.remove_edge(e)

    g_changed = True
    while g.num_edges() > 0 and g_changed:
        c = label_components(g)[0]
        c_set = set(c)
        cnt = Counter(c)
        max_size = cnt.most_common(1)[0][1]
        if len(c_set) < max(3, g.graph_properties["numsent"] / 6.0) or (max_size > max_c_size):  # !!!!!!
            g_changed = False
            for c_label in c_set:
                # process each connected component
                u = GraphView(g, vfilt=(c.a == c_label))
                if not stop_condition(u, betweenness_threshold_coef, max_c_size, min_c_size) and not is_finalized(u):
                    g_changed = True
                    u = _without_most_central_edges(u, vprop_name, eprop_weight)
        else:
            break
    return g


def _without_most_central_edges(g, vprop_name, eprop_weight=None):
    """Returns the connected components of the graph that results from
    repeatedly removing the most "valuable" edge in the graph.
    `G` must be a non-empty graph. This function modifies the graph `G`
    in-place; that is, it removes edges on the graph `G`.
    `most_valuable_edge` is a function that takes the graph `G` as input
    (or a subgraph with one or more edges of `G` removed) and returns an
    edge. That edge will be removed and this process will be repeated
    until the number of connected components in the graph increases.
    """
    original_num_vertices = g.num_vertices()
    num_components = 1
    while num_components <= 1:
        e = most_valuable_edge(g, eprop_weight)
        cp_status = duplicate_edge_condition(g, e, eprop_weight)
        g = duplicate_edge(g, e, cp_status, vprop_name, eprop_weight)
        num_components = num_connected_components(g)

        if num_components > 1:
            c = label_components(g)[0]
            c_set = set(c)
            for c_label in c_set:
                u = GraphView(g, vfilt=(c.a == c_label))
                if u.num_vertices() == original_num_vertices:
                    u = set_finalized(u)
    return g


if __name__ == "__main__":
    print(0)
    # # construct graph
    # f = open("network.txt", "r")
    # lines = [[int(n) for n in x.split()] for x in f.readlines()]
    # f.close()
    # g = Graph(directed=False)
    # g.add_vertex(1 + max([l[0] for l in lines] + [l[1] for l in lines]))
    # vprop_name = g.new_vertex_property("string")
    # for v in g.vertices():
    #     vprop_name[v] = str(g.vertex_index[v])
    # g.vertex_properties["name"] = vprop_name

    # eprop_weight = g.new_edge_property("int")
    # for line in lines:
    #     g.add_edge(g.vertex(line[0]), g.vertex(line[1]))
    #     eprop_weight[g.edge(g.vertex(line[0]), g.vertex(line[1]))] = 1
    # eprop_weight[g.edge(2, 4)] = 10  # for test duplicate edge, set 10.
    # g.edge_properties["weight"] = eprop_weight

    # # community detection
    # graph_draw(g, output="before.pdf", vertex_text=g.vertex_properties["name"])
    # print g.num_vertices()
    # g = girvan_newman(g, "name", "weight")
    # print g.num_edges()
    # print g.edge(1, 2)
    # if g.edge(1, 5) is None:
    #     print "It is None."
    # graph_draw(g, output="after.pdf", vertex_text=g.vertex_properties["name"])
    # c = label_components(g)[0]
    # print c.a
