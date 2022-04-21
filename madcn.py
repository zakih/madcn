'''
Description: Solves 3 problems (min-energy, min-uptime, max-data) for a data network consisting of multiple data sources and compression capability.
Date: January, 2022
Author: Zaki Hasnain
Author: Joshua Vander Hook

Copyright (c) 2022 California Institute of Technology (“Caltech”). U.S. Government sponsorship acknowledged.

'''

import networkx as nx
import glpk
import yaml
import os
import json
import numpy as np
import argparse
import json


COLOR_RELAY = "#dceb50"
COLOR_SOURCE = "#eb5850"
COLOR_BASE = "#6050eb"
COLOR_DATAPATH = "#1e99e6"
COLOR_SUBGRAPH = "lightgrey"


def readConfig(path_to_config=None):
    ''' 
    Read config file

    Parameters
    ----------
    path_to_config : str
        Valid path to YAML config file.
        Default None which reads 'config.yaml' in home dir.

    Returns
    -------
    config : dict
        Network and simulation options

    '''

    if path_to_config is None:
        default_config = os.path.join(os.path.dirname(__file__), 'config.yaml')
    else:
        if not os.path.isfile(path_to_config):
            raise IOError("Path to config file is not valid")
        default_config = path_to_config
    try:
        with open(default_config) as f:
            config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    except:
        raise IOError("Config file should be a YAML file")

    return config


def generateScenario(config):
    '''
    Generate a scenario with given number of
    data relay and  data source nodes.

    Parameters
    ----------
    config : dict
        Network and simulation options

    Returns
    -------
    nodes : list
        list of dictionaries descrbing node 'name' and 'data' volume
    edges : list
        list of dictionaries descrbing edges by 'to' and 'from' node names

    '''

    nodes = []
    edges = []

    num_relay_nodes = int(config['num_relay_nodes'])
    num_data_source_nodes = int(config['num_data_source_nodes'])
    prob_connection_r_ds = config['p_r_ds']
    prob_connection_ds_base = config['p_ds_base']
    capacity = config['edge_capacities']
    min_data_at_source = config['min_data_at_source']
    max_data_at_source = config['max_data_at_source']

    for i in range(num_relay_nodes):
        node = {'name': "r-"+str(i+1), 'data': 0}
        nodes.append(node)

    for i in range(num_data_source_nodes):
        data = min_data_at_source + (max_data_at_source-min_data_at_source)*np.random.rand()
        data = int(np.round(data, 0))
        node = {'name': "ds-"+str(i+1), 'data': data}
        nodes.append(node)

    nodes.append({'name': "base", 'data': 0})

    # Probabilistically create communication edges
    for i in range(len(nodes)):
        node_i = nodes[i]['name']
        for j in range(len(nodes)):
            if j != i:
                node_j = nodes[j]['name']
                if node_i != "base" and "r" in node_i:
                    if "r" in node_j:
                        # Connect all relay nodes
                        edges.append({'from': node_i, 'to': node_j, 'capacity': capacity})
                        edges.append({'from': node_j, 'to': node_i, 'capacity': capacity})
                    elif "ds" in node_j:
                        # Probabilisticly allow connection from data source nodes to relay nodes
                        p = np.random.rand()
                        if p <= prob_connection_r_ds:
                            edges.append({'from': node_j, 'to': node_i, 'capacity': capacity})
                    else:
                        # Connect all relay to base
                        edges.append({'from': node_i, 'to': node_j, 'capacity': capacity})
                elif "ds" in node_i and node_j == "base":
                    # Probabilisticly allow connection from data source nodes to base nodes
                    p = np.random.rand()
                    if p <= prob_connection_ds_base:
                        edges.append({'from': node_i, 'to': node_j, 'capacity': capacity})

    graph = {'nodes': nodes, 'edges': edges}

    if not os.path.isdir(config['output_dir']):
        os.makedirs(config['output_dir'])
    file_name = os.path.join(config['output_dir'], "graph_data_generated.json")
    json.dump(graph, open(file_name, "w"), indent=4)

    return nodes, edges


def makeGraph(config, verbose=False):
    '''
    Read graph data (nodes and edges) or generate a
    random scenario and create a graph object

    Parameters
    ----------
    config : dict
        Network and simulation options
    verbose : bool
        If true, print details to terminal

    Returns
    -------
    G : networkx.classes.digraph.DiGraph
        data network graph
    num_relay_nodes : int
        Number of relay nodes
    num_data_source_nodes : int
        Number of data source nodes
    total_data : float
        Total volume of data at all
        data source nodes

    '''

    if config['generate_random_graph']:
        print("generating scenario for random graph")
        nodes, edges = generateScenario(config)
    else:
        data = json.load(open(config['graph_data_file'], 'r'))
        nodes = data['nodes']
        edges = data['edges']
        legal_edges = []
        for edge in edges:
            if edge['from'] != 'base':
                legal_edges.append(edge)
            else:
                print("removing edge from base to", edge['to'], ", outgoing edges from base not used.")
        edges = legal_edges


    # Add vitual 'compression' nodes and associated edges
    nodes_to_add = []
    edges_to_add = []
    num_data_source_nodes = 0
    num_relay_nodes = 0
    total_data = 0
    for node in nodes:
        total_data += node['data']
        if node["name"] != "base":
            # Add virtual ndoe
            nodes_to_add.append(
                {'name': node["name"]+"-x", 'data': 0})
            # Add edge to perform compression
            edges_to_add.append(
                {'from': node["name"], 'to': node["name"]+"-x", "capacity": 10000000.0})

            if "r" in node["name"]:
                num_relay_nodes += 1
            elif "ds" in node["name"]:
                num_data_source_nodes += 1

    for edge in edges:
        from_node = edge['from']+"-x"
        to_node = edge['to']
        if to_node != "base":
            to_node = to_node+"-x"
        # Add edges from virtual compression nodes
        edges_to_add.append({"from": from_node, "to": to_node, "capacity":edge['capacity']})
    nodes.extend(nodes_to_add)
    edges.extend(edges_to_add)

    # Create graph with given nodes, edges, and edge capacities
    edge_energy_rates = config['cost_rates']['energy']
    edge_time_rates = config['cost_rates']['time']

    e_compress = edge_energy_rates['e_compress']  # energy rate for uncompressed data
    e_transfer_compressed = edge_energy_rates['e_transfer_compressed']  # energy rate for data compression
    e_transfer_uncompressed = edge_energy_rates['e_transfer_uncompressed']  # energy rate for compressed data

    t_compress = edge_time_rates['t_compress']  # time for uncompressed data
    t_transfer_compressed = edge_time_rates['t_transfer_compressed']  # time rate for data compression
    t_transfer_uncompressed = edge_time_rates['t_transfer_uncompressed']  # time rate for compressed data

    G = nx.DiGraph()
    for node in nodes:
        if "r" in node["name"]:
            color = COLOR_RELAY
        elif "ds" in node['name']:
            color = COLOR_SOURCE
        else:
            color = COLOR_BASE

        if "-x" in node["name"]:
            shape = "circle"
        elif "base" in node["name"]:
            shape = "triangle"
        else:
            shape = "square"

        G.add_node(node["name"], data=node["data"],
                   color=color, shape=shape, style='filled', fontcolor="black")

    for edge in edges:
        from_node = edge['from']
        to_node = edge['to']
        capacity = edge['capacity']

        energy_rate = e_transfer_uncompressed
        if "-x" in from_node:
            energy_rate = e_transfer_compressed
        elif "-x" in to_node:
            energy_rate = e_compress

        time_rate = t_transfer_uncompressed
        if "-x" in from_node:
            time_rate = t_transfer_compressed
        elif "-x" in to_node:
            time_rate = t_compress

        if "-x" in from_node:
            style = "dashed"
        else:
            style = "solid"

        if not "-x" in from_node and "-x" in to_node:
            edge_capacity = 10000000.0
            if config['viz_edge_props']:
                edge_label="e=" + str(energy_rate) + "," + "t=" +str(time_rate) + "\n k=inf" 
            else:
                edge_label="    "
        else:
            edge_capacity = capacity
            if config['viz_edge_props']:
                edge_label="e=" + str(energy_rate) + "," + "t=" +str(time_rate) + "\n" + "k=" +str(edge_capacity)
            else:
                edge_label="    "
        G.add_edge(from_node, to_node, capacity=edge_capacity, energy_rate=energy_rate, time_rate=time_rate, label=edge_label, style=style)

    if verbose:
        print("number of nodes: ", len(G.nodes))
        print("number of edges: ", len(G.edges))

        print("---- Nodes ----")
        for node in G.nodes:
            print(node)
        print("----")
        print("---- Edges ----")
        for edge in G.edges:
            print(edge)
        print("----")


    return G, num_relay_nodes, num_data_source_nodes, total_data


def drawGraph(G, config, file_name):
    '''
    Draw data network as a graph

    Parameters
    ----------
    G : networkx.classes.digraph.DiGraph
        data network graph
    config : dict
        network and simulation options
    file_name : str
        Name of output graph

    '''
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(dpi=200)
    
    circle = config['viz_circle_layout']
    if len(G.nodes)>6:
        circle =True
    if circle:
        A.layout(prog="circo")
    else:
        
        nodes = G.nodes
        if len(nodes) <=10: # clustering only seems to work with a few nodes
            nodes = sorted(nodes)
            cluster_index=0
            for node in nodes:
                if node != 'base' and '-x' not in node:
                    A.add_subgraph([node,node+'-x'], name = 'cluster'+str(cluster_index), color=COLOR_SUBGRAPH, style='filled', rank="same;"+node+";"+node+'-x;')
                    cluster_index += 1

        A.layout(prog="dot")

    if not os.path.isdir(config['output_dir']):
        os.makedirs(config['output_dir'])
    file_name = os.path.join(config['output_dir'], file_name + "_" + config['sim_name'] + ".png")
    A.draw(file_name, format="png")


def problemMinCost(G, problem, verbose=False):
    '''
    Solve problem 1 (minimize energy) and problem 2 (minimize uptime)
    to transfer all data at source nodes to base node.

    Parameters
    ----------
    G : networkx.classes.digraph.DiGraph
        data network graph
    verbose : bool
        If true, print details to terminal

    Returns
    -------
    G : networkx.classes.digraph.DiGraph
        data network graph with
        edges comprising solution highlighted
    total_energy : float
        Total energy required for data flow
    total_uptime : float
        Total uptime required for data flow
    total_edge_weights : float
        net edge costs accrued in data flow,
        this is for troubleshooting only
    data_out_of_nodes : dict
        Volume of data transmitted from each
        node, keys are node names
    data_transfered_to_base : float
        Total volume of data transmitted to base
        station
    makespan : float
        Total time required to complete data flow
    all_paths_to_base : list
        Nested list containing data flow from
        data source nodes to base station. The
        data, energy, and time of each communication
        along the path is provided
    compression_nodes : list
        Names of nodes which perform data compression,
        may be empty if no compression occurs in data
        flow

    '''

    ####### Adapt for LP #######
    lp = glpk.LPX()
    glpk.env.term_on = False

    node_size = len(G.nodes)
    edge_size = len(G.edges)

    # X is the volume of data sent over each link
    lp.cols.add(edge_size)

    # Find edge pairs which represent compression decisions
    compression_edge_pairs = []
    compression_edge_pair_capacities = []
    # Make a edge/id lookup dictionaries
    edge_idx = {e: i for i, e in enumerate(G.edges)}
    idx_edge = {i: e for i, e in enumerate(G.edges)}
    for edge_id, edge in enumerate(G.edges):
        node_from, node_to = edge
        # Check if edge is carrying uncommpressed data
        if not "-x" in node_from and not "-x" in node_to:
            # Find id of compressed version of edge
            node_from += "-x"
            if node_to != "base":
                node_to += "-x"
            matching_edge = (node_from, node_to)
            matching_edge_id = edge_idx[matching_edge]
            compression_edge_pairs.append((edge_id, matching_edge_id))
            edge_data = G.get_edge_data(node_from, node_to)
            compression_edge_pair_capacities.append(edge_data['capacity'])

    # There will be a constraint per node (flow constraints)
    # There will be one constraint equations per each pair of compression/non-compression edges from a node with compression capability
    num_compression_decisions = len(compression_edge_pairs)
    num_row_constraints = node_size + num_compression_decisions
    lp.rows.add(num_row_constraints)

    # Make a node/id lookup dictionaries
    node_idx = {n: i for i, n in enumerate(G.nodes)}
    idx_node = {i: n for i, n in enumerate(G.nodes)}

    # Make a sparse matrix for constraints
    constraint_matrix = []
    total_data = 0
    for node in G.nodes:
        total_data += G.nodes[node]["data"]

    # Each of the first n=node_size rows corresponds to a node, and we want 1s in that nodes' edges, and the flow (+/-) should sum to 0
    for r, row in enumerate(lp.rows):
        if r < len(G.nodes):
            # data source-sink constraints for 3 types of nodes
            node = idx_node[r]
            # print("r: ", r, ", node: ", node)
            if G.nodes[node]["data"] > 0:
                data = G.nodes[node]["data"]
                # row.bounds = -data, -data
                row.bounds = -1.0*data
            elif "base" in node:
                # row.bounds = total_data, total_data
                row.bounds = total_data
            else:
                row.bounds = 0
        else:
            # capacity constraints for compression edge pairs
            row.bounds = 0.0, compression_edge_pair_capacities[r-node_size]

    for edge in enumerate(G.edges):
        # What stat variable is for this edge?
        edge_id = edge[0]
        # And what nodes are connected by this edge?
        (node_from, node_to) = edge[1]
        # and what is the row indecies of those nodes?
        node_from_id = node_idx[node_from]
        node_to_id = node_idx[node_to]
        # and finally what metadata corresponds to this edge?
        edge_data = G.get_edge_data(node_from, node_to)
        # First set edge volume capacity constraints
        lp.cols[edge_id].bounds = 0, edge_data['capacity']
        # Set objective function coefficients
        if problem == 'min_energy':
            lp.obj[edge_id] = G.edges[edge[1]]["energy_rate"]
        else: # min_uptime
            lp.obj[edge_id] = G.edges[edge[1]]["time_rate"]

        # Now set flow in/out constraints
        # Need to set +1 if it flows into a node
        # And -1 if it flows out of a node
        # To just enforce that limit, do this:
        constraint_matrix.append((node_from_id, edge_id, -1.0))
        constraint_matrix.append((node_to_id, edge_id, 1.0))

    # Add constraint matrix entries for compression decision constraints
    constraint_index = node_size
    for edge_pair in compression_edge_pairs:
        constraint_matrix.append((constraint_index, edge_pair[0], 1.0))
        constraint_matrix.append((constraint_index, edge_pair[1], 1.0))
        constraint_index += 1

    lp.obj.maximize = False
    lp.matrix = constraint_matrix
    out = lp.simplex()

    # Get data, energy, and time for each edge
    total_energy = 0
    total_uptime = 0
    total_edge_weights = 0
    data_transfered_to_base = 0
    data_out_of_nodes = {}
    data_on_edge = {}
    energy_on_edge = {}
    time_on_edge = {}
    compression_nodes = []
    for (col, edge) in zip(lp.cols, G.edges):
        # Energy = (bits transmitted on edge i)(joules/bits on edge i)
        total_energy += col.value*G.edges[edge]["energy_rate"]
        # Time = (bits transmitted on edge i)(seconds/bits on edge i)
        total_uptime += col.value*G.edges[edge]["time_rate"]
        # Total data transfered to base
        if "base" in edge[1]:
            data_transfered_to_base += col.value
        # Total energy rates of edges used in final data flow (for troubleshooting)
        if col.value > 10**-5:
            if problem == 'min_energy':
                total_edge_weights += G.edges[edge]["energy_rate"]
            else:  # min_uptime
                total_edge_weights += G.edges[edge]["time_rate"]
        # Data leaving each node, used to update source data if needed
        from_node, to_node = edge
        if from_node in data_out_of_nodes.keys():
            data_out_of_nodes[from_node] += col.value
        else:
            data_out_of_nodes[from_node] = col.value
        #  Map of data, energy, time by node
        if from_node in data_on_edge.keys():
            data_on_edge[from_node].update({to_node:col.value})
            energy_on_edge[from_node].update({to_node:col.value*G.edges[edge]["energy_rate"]})
            time_on_edge[from_node].update({to_node:col.value*G.edges[edge]["time_rate"]})
        else:
            data_on_edge[from_node] = {to_node:col.value}
            energy_on_edge[from_node] = {to_node:col.value*G.edges[edge]["energy_rate"]}
            time_on_edge[from_node] = {to_node:col.value*G.edges[edge]["time_rate"]}
        # Check for compression decision
        if "-x" not in from_node and "-x" in to_node and col.value > 1**-5:
            compression_nodes.append(from_node)
        # Make labels and such for the graph visualization
        if col.value > 10**-5:
            G.edges[edge]["color"] = COLOR_DATAPATH
        else:
            G.edges[edge]["color"] = "grey"
        edge_capacity = G.edges[edge]["capacity"]
        if not "-x" in from_node and "-x" in to_node:
            edge_capacity = "inf" 
        if problem == 'min_energy':
            edge_cost = G.edges[edge]["energy_rate"]
        else:  # min_uptime
            edge_cost = G.edges[edge]["time_rate"]
        data_transmitted = np.round(col.value, 2)
        edge_label = str(edge_cost) + "\n" + str(data_transmitted) + "/" + str(edge_capacity)
        G.edges[edge]['label'] = edge_label

    # Get makespan and plan (paths from each data source node to base)
    all_paths_to_base = []
    makespan = 0.0
    for i, node in enumerate(G.nodes):
        if 'ds' in node and '-x' not in node:
            # Inspect all paths from source to base
            paths_to_base = list(nx.all_simple_paths(G, source=node, target='base'))
            for path in paths_to_base:
                data_along_path = []
                energy_along_path = []
                time_along_path = []
                path_and_data = []
                path_used = True
                for i, node in enumerate(path[:-1]):
                    data = data_on_edge[node][path[i+1]]
                    energy = energy_on_edge[node][path[i+1]]
                    time = time_on_edge[node][path[i+1]]
                    data_along_path.append(data)
                    energy_along_path.append(energy)
                    time_along_path.append(time)
                    path_and_data.append([node, path[i+1], data, energy, time])
                    if data == 0.0:
                        # path_used = True if np.prod(data_along_path) >0 else False
                        path_used = False

                if path_used:
                    all_paths_to_base.append(path_and_data)
                    time_to_base = np.sum(time_along_path)
                    if time_to_base > makespan:
                        makespan = time_to_base

    if data_transfered_to_base < total_data:
        print("All source data not transmitted to base node in min-cost solution!")

    if verbose:
        print("total source data:", total_data)
        print("nodes: ")
        for i, node in enumerate(G.nodes):
            print(i,node)
        print("edges: ")
        for edge in enumerate(G.edges):
            (node_from, node_to) = edge[1]
            edge_data = G.get_edge_data(node_from, node_to)
            print(node_from, node_to, edge_data['capacity'], G.edges[edge[1]]["weight"])
        A = np.zeros((num_row_constraints, edge_size))
        for const in constraint_matrix:
            A[const[0],const[1]] = const[2]
        print("Constraint matrix: ")
        print(A)
        for idx_edge in enumerate(G.edges):
            # What stat variable is for this edge?
            edge_id = idx_edge[0]
            edge = idx_edge[1]
            print(
                "Edge # {}, Cap {}/{}, ([{}] {} --> [{}] {}) ".format(
                    edge_id,
                    lp.cols[edge_id].value, G.get_edge_data(
                        edge[0], edge[1])['capacity'],
                    node_idx[edge[0]], edge[0],
                    node_idx[edge[1]], edge[1]
                ))
        print("data transferred out of nodes:")
        for key, val in data_out_of_nodes.items():
            print(key, val)
        print("Data on edges: ")
        for key, val in data_on_edge.items():
            print(key, val)
        print("makespan: ", makespan)
        print("all paths to base: ")
        for path in all_paths_to_base:
            print(path)

    return G, total_energy, total_uptime, total_edge_weights, data_out_of_nodes, data_transfered_to_base, makespan, all_paths_to_base, compression_nodes


def problemMaxData(config, G, verbose=False):
    '''

    Solve problem of finding maximum transferable
    data given energy and makespan limit.
 
    Parameters
    ----------
    config : dict
        network and simulation options
    G : networkx.classes.digraph.DiGraph
        data network graph
    verbose : bool
        If true, print details to terminal

    Returns
    -------
    G : networkx.classes.digraph.DiGraph
        data network graph with
        edges comprising solution highlighted
    total_energy : float
        Total energy required for data flow
    total_uptime : float
        Total uptime required for data flow
    total_edge_weights : float
        net edge costs accrued in data flow,
        this is for troubleshooting only
    data_out_of_nodes : dict
        Volume of data transmitted from each
        node, keys are node names
    data_transfered_to_base : float
        Total volume of data transmitted to base
        station
    makespan : float
        Total time required to complete data flow
    all_paths_to_base : list
        Nested list containing data flow from
        data source nodes to base station. The
        data, energy, and time of each communication
        along the path is provided
    compression_nodes : list
        Names of nodes which perform data compression,
        may be empty if no compression occurs in data
        flow
    '''

    ####### Adapt for LP #######
    lp = glpk.LPX()
    glpk.env.term_on = False

    node_size = len(G.nodes)
    edge_size = len(G.edges)

    # X is the amount of data sent over each link
    lp.cols.add(edge_size)

    # Find edge pairs which represent compression decisions
    compression_edge_pairs = []
    compression_edge_pair_capacities = []
    # Make a edge/id lookup dictionaries
    edge_idx = {e: i for i, e in enumerate(G.edges)}
    idx_edge = {i: e for i, e in enumerate(G.edges)}
    for edge_id, edge in enumerate(G.edges):
        node_from, node_to = edge
        # Check if edge is carrying uncommpressed data
        if not "-x" in node_from and not "-x" in node_to:
            # Find id of compressed version of edge
            node_from += "-x"
            if node_to != "base":
                node_to += "-x"
            matching_edge = (node_from, node_to)
            matching_edge_id = edge_idx[matching_edge]
            compression_edge_pairs.append((edge_id, matching_edge_id))
            edge_data = G.get_edge_data(node_from, node_to)
            compression_edge_pair_capacities.append(edge_data['capacity'])

    # Count number of simple paths from source to base
    num_simple_paths = 0
    simple_paths_to_base = []
    for i, node in enumerate(G.nodes):
        if 'ds' in node and '-x' not in node:
            paths_to_base = list(nx.all_simple_paths(G, source=node, target='base'))
            num_simple_paths += len(paths_to_base)
            for path in paths_to_base:
                # Get all edge ID's in this path and make constraint equation by summing time over all edges in path
                edges_in_path = []  # for each edge in path add (edge_id, edge_time_rate)
                for i, node_from in enumerate(path[:-1]):
                    node_to = path[i+1]
                    # Get edge ID and time rate for edge between (node_from, node_to)
                    edge = (node_from, node_to)
                    edge_id = edge_idx[edge]
                    edge_time_rate = G.edges[edge]['time_rate']
                    edges_in_path.append((edge_id, edge_time_rate))
                simple_paths_to_base.append(edges_in_path)


    # There will be a constraint per node (flow constraints)
    # There will be one constraint equations per each pair of compression/non-compression edges from a node with compression capability
    # There will be one constraint equation for max energy 
    # There will be one constraint equation for each simple path from source to base for max makespan
    num_compression_decisions = len(compression_edge_pairs)    
    num_row_constraints = node_size + 1 + num_compression_decisions + num_simple_paths
    lp.rows.add(num_row_constraints)

    # Make a node/id lookup dictionaries
    node_idx = {n: i for i, n in enumerate(G.nodes)}
    idx_node = {i: n for i, n in enumerate(G.nodes)}

    # Make a sparse matrix for constraints
    constraint_matrix = []
    max_energy = config['max_energy']
    max_makespan = config['max_makespan']
    for r, row in enumerate(lp.rows):
        if r < len(G.nodes): # flow constraints
            # Each row corresponds to a node, and we want 1s in that nodes' edges, and the flow (+/-) should sum to 0
            row.bounds = 0.0, 0.0
        elif r == len(G.nodes):  # max energy constraint
            # Total cost of data transfer over all edges
            row.bounds = 0.0, max_energy
        elif r < node_size + 1 + num_compression_decisions:  # compression constraints
            row.bounds = 0.0, compression_edge_pair_capacities[r-node_size-1]
        else:  # makespan constraints
            row.bounds = 0.0, max_makespan


    for edge_id, edge in enumerate(G.edges):
        # And what nodes are connected by this edge?
        (node_from, node_to) = edge
        # and what is the row indecies of those nodes?
        node_from_id = node_idx[node_from]
        node_to_id = node_idx[node_to]
        # and finally what metadata corresponds to this edge?
        edge_data = G.get_edge_data(node_from, node_to)
        # First set the capacity limits
        lp.cols[edge_id].bounds = 0, edge_data['capacity']
        # Set objective function coefficient = 1 if incoming edge to base
        if "base" in node_to:
            lp.obj[edge_id] = 1.0
        else:
            lp.obj[edge_id] = 0

        # Now set flow in/out constraints
        # Need to set +1 if it flows into a node
        # And -1 if it flows out of a node
        # To just enforce that limit, do this:
        # But we don't want to model flow "into" data source nodes balancing what we're extracting
        if not G.nodes[node_from]["data"]>0:
            # Except we don't require
            constraint_matrix.append((node_from_id, edge_id, -1.0))
        # And we don't want to model flow "out of base" balancing the downlink
        if node_to != "base":
            constraint_matrix.append((node_to_id, edge_id, 1.0))

        # add max-energy constraint to row=len(G.nodes) of constraint matrix
        energy_rate = G.edges[edge]["energy_rate"]
        constraint_matrix.append((len(G.nodes), edge_id, energy_rate))

    # Add constraint matrix entries for compression decision constraints
    constraint_index = node_size+1
    for edge_pair in compression_edge_pairs:
        constraint_matrix.append((constraint_index, edge_pair[0], 1.0))
        constraint_matrix.append((constraint_index, edge_pair[1], 1.0))
        constraint_index += 1

    # Add constraint matrix entries for makespan constraints
    for path in simple_paths_to_base:
        for edge in path:
            edge_id, edge_time_rate = edge
            constraint_matrix.append((constraint_index, edge_id, edge_time_rate))
        constraint_index += 1

    lp.obj.maximize = True
    lp.matrix = constraint_matrix
    out = lp.simplex()

    # Get data, energy, and time for each edge
    total_energy = 0
    total_uptime = 0
    total_edge_weights = 0
    data_transfered_to_base = 0
    data_out_of_nodes = {}
    data_on_edge = {}
    energy_on_edge = {}
    time_on_edge = {}
    compression_nodes = []
    for (col, edge) in zip(lp.cols, G.edges):
        # Energy = (bits transmitted on edge i)(joules/bits on edge i)
        total_energy += col.value*G.edges[edge]["energy_rate"]
        # Time = (bits transmitted on edge i)(seconds/bits on edge i)
        total_uptime += col.value*G.edges[edge]["time_rate"]
        # Total data transfered to base
        if "base" in edge[1]:
            data_transfered_to_base += col.value
        # Total energy rates of edges used in final data flow (for troubleshooting)
        if col.value > 10**-5:
            total_edge_weights += G.edges[edge]["energy_rate"]
        # Data leaving each node, used to update source data if needed
        from_node, to_node = edge
        if from_node in data_out_of_nodes.keys():
            data_out_of_nodes[from_node] += col.value
        else:
            data_out_of_nodes[from_node] = col.value
        #  Map of data, energy, time by node
        if from_node in data_on_edge.keys():
            data_on_edge[from_node].update({to_node:col.value})
            energy_on_edge[from_node].update({to_node:col.value*G.edges[edge]["energy_rate"]})
            time_on_edge[from_node].update({to_node:col.value*G.edges[edge]["time_rate"]})
        else:
            data_on_edge[from_node] = {to_node:col.value}
            energy_on_edge[from_node] = {to_node:col.value*G.edges[edge]["energy_rate"]}
            time_on_edge[from_node] = {to_node:col.value*G.edges[edge]["time_rate"]}
        # Check for compression decision
        if "-x" not in from_node and "-x" in to_node and col.value > 1**-5:
            compression_nodes.append(from_node)
        # Make labels and such for the graph visualization
        if col.value > 10**-5:
            G.edges[edge]["color"] = COLOR_DATAPATH
        else:
            G.edges[edge]["color"] = "grey"
        edge_capacity = G.edges[edge]["capacity"]
        if not "-x" in from_node and "-x" in to_node:
            edge_capacity = "inf" 
        edge_cost = G.edges[edge]["energy_rate"]
        data_transmitted = np.round(col.value, 2)
        edge_label = str(edge_cost) + "\n" + str(data_transmitted) + "/" + str(edge_capacity)
        G.edges[edge]['label'] = edge_label


    # Get makespan and plan (paths from each data source node to base)
    all_paths_to_base = []
    makespan = 0.0
    for i, node in enumerate(G.nodes):
        if 'ds' in node and '-x' not in node:
            # Inspect all paths from source to base
            paths_to_base = list(nx.all_simple_paths(G, source=node, target='base'))
            for path in paths_to_base:
                data_along_path = []
                energy_along_path = []
                time_along_path = []
                path_and_data = []
                path_used = True
                for i, node in enumerate(path[:-1]):
                    data = data_on_edge[node][path[i+1]]
                    energy = energy_on_edge[node][path[i+1]]
                    time = time_on_edge[node][path[i+1]]
                    data_along_path.append(data)
                    energy_along_path.append(energy)
                    time_along_path.append(time)
                    path_and_data.append([node, path[i+1], data, energy, time])
                    if data == 0.0:
                        # path_used = True if np.prod(data_along_path) >0 else False
                        path_used = False

                if path_used:
                    all_paths_to_base.append(path_and_data)
                    time_to_base = np.sum(time_along_path)
                    if time_to_base > makespan:
                        makespan = time_to_base

    if verbose:
        print("nodes: ")
        for i, node in enumerate(G.nodes):
            print(i,node)
        print("edges: ")
        for edge in enumerate(G.edges):
            (node_from, node_to) = edge[1]
            edge_data = G.get_edge_data(node_from, node_to)
            print(node_from, node_to, edge_data['capacity'], G.edges[edge[1]]["weight"])
        A = np.zeros((num_row_constraints, edge_size))
        for const in constraint_matrix:
            A[const[0],const[1]] = const[2]
        print("Constraint matrix: ")
        print(A)
        for idx_edge in enumerate(G.edges):
            # What stat variable is for this edge?
            edge_id = idx_edge[0]
            edge = idx_edge[1]
            print(
                "Edge # {}, Cap {}/{}, ([{}] {} --> [{}] {}) ".format(
                    edge_id,
                    lp.cols[edge_id].value, G.get_edge_data(
                        edge[0], edge[1])['capacity'],
                    node_idx[edge[0]], edge[0],
                    node_idx[edge[1]], edge[1]
                ))
        print("data transferred out of nodes:")
        for key, val in data_out_of_nodes.items():
            print(key, val)
        print("Data on edges: ")
        for key, val in data_on_edge.items():
            print(key, val)
        print("makespan: ", makespan)
        print("all paths to base: ")
        for path in all_paths_to_base:
            print(path)
    return G, total_energy, total_uptime, total_edge_weights, data_out_of_nodes, data_transfered_to_base, makespan, all_paths_to_base, compression_nodes


def writeOutputs(config, outputs, num_relay_nodes, num_data_source_nodes, avg_out_degree, avg_ds_out_degree, sim_name):
    '''
    Write solutions for problem 1 (min energy), problem 2 (min uptime), and problem 3 (max data) 

    Parameters
    ----------
    config : dict
        network and simulation options
    outputs : dict
        Problem 1 or 2 outputs
    num_relay_nodes : int
        Number of relay nodes
    num_data_source_nodes : int
        Number of data source nodes
    avg_out_degree : float
        Average graph out degree
    avg_ds_out_degree : float
        Average graph out degree for data source nodes
    sim_name : str
        Text tag for simulation case

    '''

    header = [
        'simulation_name',
        'network',
        'num_relay_nodes',
        'num_data_source_nodes',
        'p_r_ds',
        'p_ds_base',
        'avg_out_degree',
        'avg_ds_out_degree',
        'edge_capacities',
        'e_compress',
        'e_transfer_compressed',
        'e_transfer_uncompressed',
        't_compress',
        't_transfer_compressed',
        't_transfer_uncompressed',
        'compression_nodes',
        'any_compression',
        'makespan'
    ]

    if config['generate_random_graph']:
        network = 'random'
        p_r_ds = config['p_r_ds']
        p_ds_base = config['p_ds_base']
        edge_capacities = config['edge_capacities']
    else:
        network = config['graph_data_file']
        p_r_ds = None
        p_ds_base = None
        edge_capacities = 'user specified'

    if outputs['problem'] == 'min_energy':
        file_name = 'result_min_energy_problem.csv'
        file_name = os.path.join(config['output_dir'], file_name)
        if not os.path.isfile(file_name):
            header.extend([
                'minimum_energy_to_transfer_data_to_base',
                'uptime_to_transfer_data_to_base',
                'data_transfered_to_base',
                'max_data_available'])
            header = (",").join(header)
            header = header + '\n'
            # write header
            file = open(file_name, 'w')
            file.write(header)
            file.close()
    elif outputs['problem'] == 'min_uptime':
        file_name = 'result_min_uptime_problem.csv'
        file_name = os.path.join(config['output_dir'], file_name)
        if not os.path.isfile(file_name):
            header.extend([
                'energy_to_transfer_data_to_base',
                'minimum_uptime_to_transfer_data_to_base',
                'data_transfered_to_base',
                'max_data_available'])
            header = (",").join(header)
            header = header + '\n'
            # write header
            file = open(file_name, 'w')
            file.write(header)
            file.close()
    else:
        file_name = 'result_max_data_problem.csv'
        file_name = os.path.join(config['output_dir'], file_name)
        if not os.path.isfile(file_name):
            header.extend([
                'energy_to_transfer_data_to_base',
                'uptime_to_transfer_data_to_base',
                'data_transfered_to_base',
                'max_energy_available',
                'max_time_available'])
            header = (",").join(header)
            header = header + '\n'
            # write header
            file = open(file_name, 'w')
            file.write(header)
            file.close()

    any_compression = 0
    if len(outputs['compression_nodes'])>0:
        any_compression = 1
    line = [
        config['sim_name'],
        network,
        num_relay_nodes,
        num_data_source_nodes,
        p_r_ds,
        p_ds_base,
        avg_out_degree,
        avg_ds_out_degree,
        edge_capacities,
        config['cost_rates']['energy']['e_compress'],
        config['cost_rates']['energy']['e_transfer_compressed'],
        config['cost_rates']['energy']['e_transfer_uncompressed'],
        config['cost_rates']['time']['t_compress'],
        config['cost_rates']['time']['t_transfer_compressed'],
        config['cost_rates']['time']['t_transfer_uncompressed'],
        (" ").join(outputs['compression_nodes']),
        any_compression,
        outputs['makespan'],
        outputs['energy'],
        outputs['uptime'],
        outputs['data'],
    ]
        
    if outputs['problem'] == 'min_energy' or outputs['problem'] == 'min_uptime':
        line.append(outputs['total_data_available'])
    else:
        line.append(outputs['total_energy_available'])
        line.append(outputs['total_time_available'])
        
    line = [str(l) for l in line]
    line = ", ".join(line)
    line = line + '\n'
    file = open(file_name, 'a')
    file.write(line)
    file.close()

    # Save data paths to base (paths_to_base)
    plan = {
        'data': outputs['data'],
        'energy': outputs['energy'],
        'uptime': outputs['uptime'],
        'makespan': outputs['makespan']
    }
    path_num = 0
    for path in outputs['paths_to_base']:
        path_from_source = []
        for edge in path:
            a_path = {}
            a_path['from'] = edge[0]
            a_path['to'] = edge[1]
            a_path['data'] = edge[2]
            a_path['energy'] = edge[3]
            a_path['time'] = edge[4]
            path_from_source.append(a_path)
        plan['data_flow_' + str(path_num)]= path_from_source
        path_num += 1

    if outputs['problem'] == 'min_energy':
        outfile = 'data_flow_min_energy_' + str(sim_name) + '.json'
    elif outputs['problem'] == 'min_uptime':
        outfile = 'data_flow_min_uptime_' + str(sim_name) + '.json'
    else:
        outfile = 'data_flow_max_data_' + str(sim_name) + '.json'
    outfile = os.path.join(config['output_dir'], outfile)
    json.dump(plan, open(outfile, 'w'), indent=4)


def updateSourceDataProduction(G, data_out_of_nodes):
    '''
    Adjust data produced at data source nodes to amount in
    data_out_of_nodes

    Parameters
    ----------
    G : networkx.classes.digraph.DiGraph
        data network graph
    data_out_of_nodes : dict
        data produced at source nodes

    Returns
    -------
    G : networkx.classes.digraph.DiGraph
        data network graph with
        'data' property of nodes in
        data_out_of_nodes updated

    '''

    for node in G.nodes:
        if "ds" in node and '-x' not in node:
            G.nodes[node]["data"] = data_out_of_nodes[node]

    return G


def main(config):
    '''

    Calls min-cost and max-data problem solvers

    Parameters
    ----------
    config : dict
        network and simulation options

    '''

    # 1. Make graph
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
    G, num_relay_nodes, num_data_source_nodes, total_data = makeGraph(config, verbose=config['verbose'])

    # print(">>>>>>> 1. right after make")
    # for edge in enumerate(G.edges):
    #     (node_from, node_to) = edge[1]
    #     edge_data = G.get_edge_data(node_from, node_to)
    #     print(node_from, node_to, edge_data['capacity'], G.edges[edge[1]]["energy_rate"], G.edges[edge[1]]["time_rate"])

    # 2. Check minimum data requirement feasibility for problem 1 (min energy) and problem 2 (min uptime)
    if config['solve_min_energy'] or config['solve_min_uptime']:
        # Check if specified data requirment is feasible
        config_check = config.copy()
        config_check['visualize_graphs']=False
        config_check['verbose']=False
        config_check['max_energy'] = 100000000
        config_check['max_makespan'] = 10000000
        _, _, _, _, data_out_of_nodes, data_transfered_to_base, _, _, _= problemMaxData(config_check, G.copy(), verbose=config_check['verbose'])
        print("max data check: ", data_transfered_to_base)
        feasible = data_transfered_to_base >= total_data
        if not feasible:
            print("Specified total data (", total_data, ") not transferable to base station. Limiting to max feasible data (", data_transfered_to_base, ") to solve min-cost")
            G = updateSourceDataProduction(G, data_out_of_nodes)

    # print(">>>>>>> 2. after feasability check")
    # for edge in enumerate(G.edges):
    #     (node_from, node_to) = edge[1]
    #     edge_data = G.get_edge_data(node_from, node_to)
    #     print(node_from, node_to, edge_data['capacity'], G.edges[edge[1]]["weight"])

    # 3. Visualize graph (optional)
    if config['visualize_graphs']:
        drawGraph(G, config, "full_network")

    # 4. Check graph properties (optional) 
    G_bare = G.copy()
    nodes_to_remove = []
    for node in G_bare.nodes():
        if "-x" in node:
            nodes_to_remove.append(node)
    for node in nodes_to_remove:
        G_bare.remove_node(node)
    out_degree = nx.algorithms.centrality.out_degree_centrality(G_bare)
    ds_out_degree = {k: out_degree[k]
                      for k in out_degree.keys() if "ds-" in k}
    out_degree = np.array(list(out_degree.values()))
    ds_out_degree = np.array(list(ds_out_degree.values()))
    avg_out_degree = np.mean(out_degree)
    avg_ds_out_degree = np.mean(ds_out_degree)

    # 5. Solve problem 1 (min-energy)
    if config['solve_min_energy']:
        print("=== PROBELM 1: min energy ===")
        problem='min_energy'
        G_minc, total_energy, total_uptime, total_edge_weights, data_out_of_nodes, data_transfered_to_base, makespan, all_paths_to_base, compression_nodes = problemMinCost(G, problem, verbose=config['verbose'])
        print("Minimum energy of transmitting all data:", total_energy)
        print("Uptime of transmitting all data:", total_uptime)
        print("Data transmitted to base station: ", data_transfered_to_base)
        print("Makespan: ", makespan)
        print("Total weight of traversed edges: ", total_edge_weights)
        print("All paths to base: ")
        for path in all_paths_to_base:
            print(path)
        if config['visualize_graphs']:
            drawGraph(G_minc, config, problem)
        outputs = {
            'problem': problem,
            'data': data_transfered_to_base,
            'energy': total_energy,
            'uptime': total_uptime,
            'makespan': makespan,
            'paths_to_base': all_paths_to_base,
            'total_data_available': total_data,
            'compression_nodes': compression_nodes
        }
        writeOutputs(config, outputs, num_relay_nodes, num_data_source_nodes,
                     avg_out_degree, avg_ds_out_degree, config['sim_name'])

    # print(">>>>>>> 5. after min energy")
    # for edge in enumerate(G.edges):
    #     (node_from, node_to) = edge[1]
    #     edge_data = G.get_edge_data(node_from, node_to)
    #     print(node_from, node_to, edge_data['capacity'], G.edges[edge[1]]["weight"])

    # 6. Solve problem 2 (min-uptime)
    if config['solve_min_uptime']:
        print("=== PROBELM 2: min uptime ===")
        problem='min_uptime'
        G_minc, total_energy, total_uptime, total_edge_weights, data_out_of_nodes, data_transfered_to_base, makespan, all_paths_to_base, compression_nodes = problemMinCost(G, problem, verbose=config['verbose'])
        print("Energy of transmitting all data:", total_energy)
        print("Minimum uptime of transmitting all data:", total_uptime)
        print("Data transmitted to base station: ", data_transfered_to_base)
        print("Makespan: ", makespan)
        print("Total weight of traversed edges: ", total_edge_weights)
        print("All paths to base: ")
        for path in all_paths_to_base:
            print(path)
        if config['visualize_graphs']:
            drawGraph(G_minc, config, problem)
        outputs = {
            'problem': problem,
            'data': data_transfered_to_base,
            'energy': total_energy,
            'uptime': total_uptime,
            'makespan': makespan,
            'paths_to_base': all_paths_to_base,
            'total_data_available': total_data,
            'compression_nodes': compression_nodes
        }
        writeOutputs(config, outputs, num_relay_nodes, num_data_source_nodes,
                     avg_out_degree, avg_ds_out_degree, config['sim_name'])
    # print(">>>>>>> 6. after min uptime")
    # for edge in enumerate(G.edges):
    #     (node_from, node_to) = edge[1]
    #     edge_data = G.get_edge_data(node_from, node_to)
    #     print(node_from, node_to, edge_data['capacity'], G.edges[edge[1]]["weight"])

    # 7. Solve problem 3 (max-data)
    if config['solve_max_data']:
        print("=== PROBELM 3: max data ===")
        problem = "max_data"
        G_maxd, total_energy, total_uptime, total_edge_weights, data_out_of_nodes, data_transfered_to_base, makespan, all_paths_to_base, compression_nodes = problemMaxData(config, G, verbose=config['verbose'])
        print("Energy of transmitting max data: ", total_energy)
        print("Uptime of transmitting max data: ", total_uptime)
        print("Max data transmitted to base station: ", data_transfered_to_base)
        print("Makespan: ", makespan)
        print("Total weight of traversed edges: ", total_edge_weights)
        print("All paths to base: ")
        for path in all_paths_to_base:
            print(path)
        if config['visualize_graphs']:
            drawGraph(G_maxd, config, problem)
        outputs = {
            'problem': 'max_data',
            'data': data_transfered_to_base,
            'energy': total_energy,
            'uptime': total_uptime,
            'makespan': makespan,
            'paths_to_base': all_paths_to_base,
            'total_energy_available': config['max_energy'],
            'total_time_available': config['max_makespan'],
            'compression_nodes': compression_nodes
        }
        writeOutputs(config, outputs, num_relay_nodes, num_data_source_nodes,
                     avg_out_degree, avg_ds_out_degree, config['sim_name'])


def minEnergy(config, graph_data_file=None, generate_random_graph=True, visualize_graphs=True, verbose=False, sim_name="test"):
    '''
    Entry point for solving minimum energy problem

    Parameters
    ----------
    config : dict
        network and simulation options
    graph_data_file : str
        Path to JSON file with graph node and edge data,
        can be None if generate_random_graph=True
    generate_random_graph : bool
        If true, generate network randomly
    visualize_graphs : bool
        If true, output plots of network
    verbose : bool
        To see intermediate computations
    sim_name : str
        Text tag for simulation case
    
    '''
    config['solve_min_energy'] = True
    config['solve_min_uptime'] = False
    config['solve_max_data'] = False

    config['graph_data_file'] = graph_data_file
    config['generate_random_graph'] = generate_random_graph
    config['visualize_graphs'] = visualize_graphs
    config['verbose'] = verbose
    config['sim_name'] = sim_name

    main(config)


def minUptime(config, graph_data_file=None, generate_random_graph=True, visualize_graphs=True, verbose=False, sim_name="test"):
    '''
    Entry point for solving minimum uptime problem

    Parameters
    ----------
    config : dict
        network and simulation options
    graph_data_file : str
        Path to JSON file with graph node and edge data,
        can be None if generate_random_graph=True
    generate_random_graph : bool
        If true, generate network randomly
    visualize_graphs : bool
        If true, output plots of network
    verbose : bool
        To see intermediate computations
    sim_name : str
        Text tag for simulation case
    
    '''
    
    config['solve_min_energy'] = False
    config['solve_min_uptime'] = True
    config['solve_max_data'] = False

    config['graph_data_file'] = graph_data_file
    config['generate_random_graph'] = generate_random_graph
    config['visualize_graphs'] = visualize_graphs
    config['verbose'] = verbose
    config['sim_name'] = sim_name
    main(config)


def maxData(config, graph_data_file=None, generate_random_graph=True, visualize_graphs=True, verbose=False, sim_name="test"):
    '''
    Entry point for solving maximum data problem

    Parameters
    ----------
    config : dict
        network and simulation options
    graph_data_file : str
        Path to JSON file with graph node and edge data,
        can be None if generate_random_graph=True
    generate_random_graph : bool
        If true, generate network randomly
    visualize_graphs : bool
        If true, output plots of network
    verbose : bool
        To see intermediate computations
    sim_name : str
        Text tag for simulation case
    
    '''

    config['solve_min_energy'] = False
    config['solve_min_uptime'] = False
    config['solve_max_data'] = True

    config['graph_data_file'] = graph_data_file
    config['generate_random_graph'] = generate_random_graph
    config['visualize_graphs'] = visualize_graphs
    config['verbose'] = verbose
    config['sim_name'] = sim_name

    main(config)


if __name__ == "__main__":
    # os.system("clear")
    # Update settings passed via CLI
    help_message = 'Max-data and min-cost solvers for a multi source network with data compression'
    repo_readme = 'https://github.jpl.nasa.gov/hook/jpl.cuberover-dev/blob/master/README.md'
    parser = argparse.ArgumentParser(description=help_message,
                                     usage=repo_readme,
                                     allow_abbrev=False
                                     )

    # 1. Which problem to solve
    parser.add_argument('--minenergy', action='store_true',
                        help='Solve minimum energy problem', dest='solve_min_energy')
    parser.add_argument('--minuptime', action='store_true',
                        help='Solve minimum uptime problem', dest='solve_min_uptime')
    parser.add_argument('--maxdata', action='store_true',
                        help='Solve max data problem', dest='solve_max_data')

    # 2. File to read graph data from
    parser.add_argument('--graph', help='Path to graph data JSON file',
                        type=str, dest='graph_data_file')

    # 3. Generate graph randomly (ignores graph data file)
    parser.add_argument('--random', action='store_true',
                        help='Generate random network', dest='generate_random_graph')

    # 4. Simulation i/o
    parser.add_argument('--viz', action='store_true',
                        help='Draw network and resulting flows', dest='visualize_graphs')
    parser.add_argument('--v', action='store_true',
                        help='verbose output', dest='verbose')
    parser.add_argument('--sim', type=str,
                        help='simulation name', dest='sim_name')

    # 5. Energy and uptime limits for max data (problem 3) respectively
    parser.add_argument(
        '--E', help='Maximum compression and transmission energy (problem 3)', type=float, dest='max_energy')
    parser.add_argument(
        '--T', help='Maximum compression and transmission time (problem 3)', type=float, dest='max_makespan')
    parser.add_argument('--k', help='Edge capacity/max transmission data between nodes',
                        type=float, dest='edge_capacities')

    # Execute the parse_args() method
    args = parser.parse_args()
    cli_args = vars(args)

    # Load default config
    if os.path.exists(os.path.join(os.path.dirname(__file__), 'config.yaml')):
        config = readConfig()
    else:
        config = {}

    # Update defaults with passed args
    config.update((k, cli_args[k])
                  for k in cli_args.keys() if cli_args[k] is not None)
    config.update((k, cli_args[k])
                  for k in cli_args.keys() if k not in config.keys())

    # Solve problems
    if config['solve_min_energy'] or config['solve_min_uptime'] or config['solve_max_data']:
        main(config)
    else:
        print("pass at least one: [--minenergy, --minuptime, --maxdata]")
