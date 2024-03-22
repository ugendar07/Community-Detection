import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import time
import pandas as pd
from sortedcontainers import SortedSet
from collections import defaultdict


def import_facebook_data(path):
    print("In import Function : ")
    file = open(path, 'r')
    lines = file.readlines()
    edges = []
    for line in lines:
        line=line.strip().split(" ")
        line[0]=int(line[0])
        line[1]=int(line[1])
        if line not in edges:
            edges.append(line)
    return np.array(edges)

def import_bitcoin_data(path):
    df = pd.read_csv(path, header=None)
    nodes_connectivity_list_btc = np.array(df.iloc[:,0:2])-1
    return nodes_connectivity_list_btc

    # data_bit=pd.read_csv(path,names=["a","b","c","d"])
    # aa=data_bit[["a","b"]]
    # return aa.values-1


def adjacency_matrix(edges):
    print("In adjacency Matrix :")
    nodes=list(SortedSet(np.array(edges).flatten()))
    sh=(len(nodes),len(nodes))
    mat = np.zeros(sh)
    for edge in edges:
        if edge[0]!='' and edge[1]!='':
            i=nodes.index(edge[0])
            j=nodes.index(edge[1])
            mat[i][j] = 1
            mat[j][i] = 1
    return np.array(nodes), mat

def createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb):
    _ , adjacency = adjacency_matrix(nodes_connectivity_list_fb)
    temp = np.argsort(graph_partition_fb[:,1])
    return adjacency[temp].T[temp]

##################### Spectral Decomposition ########################

def spectralDecomp_OneIter(nodes_connectivity_list_fb):
    print("In Spectral Decomposition One iteration :")
    nodes, adj_mat_fb = adjacency_matrix(nodes_connectivity_list_fb)
    num_nodes = nodes.shape[0]
    eig_values , eig_vecs = np.linalg.eig(np.diagflat(adj_mat_fb.sum(axis=1)) - adj_mat_fb)
    temp = np.argsort(eig_values)
    fielder_vec_fb = eig_vecs[:,temp[1]]
    
    i=0
    positive=None
    negative=None
    while  (positive==None or negative==None) and i<fielder_vec_fb.shape[0]:
        if positive==None and fielder_vec_fb[i] >= 0:
            positive = nodes[i]
        if negative==None and fielder_vec_fb[i] < 0:
            negative = nodes[i]
        i=i+1
    graph_partition_fb = np.zeros((num_nodes,2),dtype=int)
    for i in range(fielder_vec_fb.shape[0]):
        if fielder_vec_fb[i]<0:
            graph_partition_fb[i][0],graph_partition_fb[i][1] = int(nodes[i]),int(negative)
        else:
            graph_partition_fb[i][0],graph_partition_fb[i][1] = int(nodes[i]),int(positive)
    
    return fielder_vec_fb, adj_mat_fb, graph_partition_fb 
 

def spectralDecomposition(nodes_connectivity_list):
    print("In Spectral DEcomposition :")
    clusters = list([nodes_connectivity_list])
    nodes, _ = adjacency_matrix(nodes_connectivity_list)
    graph_partition_fb=[]
    for i in nodes:
        graph_partition_fb.append([i,0])
    
    graph_partition_fb=np.array(graph_partition_fb)

    while len(clusters)!=0:
        cluster = clusters.pop(0)
        _, adj_mat_fb, graph_partition =spectralDecomp_OneIter(cluster)

        for n in range(graph_partition.shape[0]):
            temp = graph_partition[n]
            graph_partition_fb[np.where(graph_partition_fb[:,0] == temp[0])[0][0]][1] = temp[1]
            
        try:
            temp1 = np.unique(graph_partition[:,0])
            clus = defaultdict(list)
            for n in range(graph_partition.shape[0]):
                clus[graph_partition[n][1]].append(graph_partition[n][0])
            edges = []
            for c in clus:
                temp2 = []
                for i in range(len(clus[c])):
                    temp3 = adj_mat_fb[np.where(temp1==clus[c][i])[0][0]]
                    for j in range(clus[c][i],temp3.shape[0]):
                        if temp3[j]==1:
                            temp2.append([clus[c][i],temp1[j]])
                edges.append(np.array(temp2))
            nodes_count = []
            for c in clus:
                nodes_count.append(len(clus[c]))

            a,b = edges[0],edges[1]
            if a.shape[0] >500 and a.shape != cluster.shape and nodes_count[0]>10:
                clusters.append(a)
            if b.shape[0] >500 and b.shape != cluster.shape and nodes_count[1]>10:   
                clusters.append(b)
        except Exception as e:
            pass
    
    return graph_partition_fb



###########################  The Louvian Algo  ################################

def calc_node_wts(edge_wts):
    # Calculates node wts (sum of edge wts of a node)

    node_wts = defaultdict(float)
    for node in edge_wts.keys():
        node_wts[node] = sum([w for w in edge_wts[node].values()])

    return node_wts

def get_neighbor_nodes(node, edge_wts):
    # Returns neighbors of a node along with edge wts

    if node not in edge_wts:
        return 0

    return edge_wts[node].items()

def get_node_wt_in_cluster(node, node2com, edge_wts):
    # Calculates node wts (sum of edge wts of a node) within a cluster/community

    node_com = node2com[node]
    nei_nodes = get_neighbor_nodes(node, edge_wts)
    wts = 0.
    for nei_node in nei_nodes:
        if node_com == node2com[nei_node[0]]:
            wts += nei_node[1]

    return wts

def get_tot_wt(node, node2com, edge_wts):
    # Calculates total weight of nodes in a community 

    nodes = [n for n, cid in node2com.items() if cid == node2com[node] and node != n]
    wt = 0.
    for n in nodes:
        wt += sum(list(edge_wts[n].values()))

    return wt

def get_cluster_deg(nodes, edge_wts):
    # Calculates the cluster degree

    return sum([sum(list(edge_wts[n].values())) for n in nodes])

def splitting_phase(node2com, edge_wts):

    node_wts = calc_node_wts(edge_wts)
    all_edge_wts = sum([wt for start in edge_wts.keys() for end, wt in edge_wts[start].items()]) / 2
    flag = True
    while flag:
        flags = []
        for node in node2com.keys():
            flags = []
            nei_nodes = [edge[0] for edge in get_neighbor_nodes(node, edge_wts)]
            max_delta = 0.
            cid = node2com[node]
            mcid = cid
            communities = {}
            for nei_node in nei_nodes:
                node2com_copy = node2com.copy()
                if node2com_copy[nei_node] in communities:
                    continue
                communities[node2com_copy[nei_node]] = 1
                node2com_copy[node] = node2com_copy[nei_node]
                delta_q = 2 * get_node_wt_in_cluster(node, node2com_copy, edge_wts) - (get_tot_wt(node, node2com_copy, edge_wts) * node_wts[node] / all_edge_wts)
                if delta_q > max_delta:
                    mcid = node2com_copy[nei_node]
                    max_delta = delta_q           
            flags.append(cid != mcid)
            node2com[node] = mcid
        if sum(flags) == 0:
            break

    return node2com, node_wts

def merging_phase(node2com, edge_wts):

    new_edge_wts = defaultdict(lambda : defaultdict(float))
    node2com_new = {}
    com2node = defaultdict(list)
    for node, cid in node2com.items():
        com2node[cid].append(node)
        if cid not in node2com_new:
            node2com_new[cid] = cid
    nodes = list(node2com.keys())
    node_pairs = list(permutations(nodes, 2)) + [(node, node) for node in nodes]
    for edge in node_pairs:
        new_edge_wts[node2com_new[node2com[edge[0]]]][node2com_new[node2com[edge[1]]]] += edge_wts[edge[0]][edge[1]]

    return node2com_new, new_edge_wts

def partition_update(node2com_new, partition):
    # Updates the community id of the nodes

    reverse_partition = defaultdict(list)
    for node, cid in partition.items():
        reverse_partition[cid].append(node)
    for old_cid, new_cid in node2com_new.items():
        for old_com in reverse_partition[old_cid]:
            partition[old_com] = new_cid

    return partition

def louvain_one_iter(edgelist):

    G = nx.Graph()
    G.add_edges_from(edgelist)
    node2com = {}
    edge_wts = defaultdict(lambda : defaultdict(float))
    for idx, node in enumerate(G.nodes()):
        node2com[node] = idx
        for edge in G[node].items():
            edge_wts[node][edge[0]] = 1.0
    node2com, node_wts = splitting_phase(node2com, edge_wts)
    partition = node2com.copy()
    node2com_new, new_edge_wts = merging_phase(node2com, edge_wts)  
    partition = partition_update(node2com_new, partition)
  
    return np.array(list(partition.items()))

 


if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)

    sorted_vector_real = sorted(fielder_vec_fb)
    # imaginary_part = np.imag(fielder_vec_fb)

    plt.scatter(range(len(sorted_vector_real)),sorted_vector_real)
    # plt.plot(imaginary_part, label='Imaginary Part')
    plt.xlabel('Node Index')
    plt.ylabel('Fielder Vector Value')
    plt.title('Fielder Vector')
    # plt.legend()
    plt.show()

    adj_mat_magnitude = np.abs(adj_mat_fb)
    plt.imshow(adj_mat_magnitude, cmap='Purples_r', interpolation='none')
    plt.title('Adjacency Matrix')
    # plt.colorbar()
    plt.show()


    print('Spectral Decomposition One Iteration')
    print(f"No of communities detected: {np.unique(graph_partition_fb[:,1],return_counts=True)[1].shape[0]}")
    part = {row[0]:row[1] for row in graph_partition_fb}
    # print(part)
    G = nx.Graph(adj_mat_fb, nodetype=int)
    plt.figure(figsize=(5,5))
    values = [part.get(node) for node in G.nodes()]
    # clust=[i*30 for i in nx.clustering(G, weight='weight').values()]
    nx.draw(G,font_size=8, node_size=30, node_color=values, 
        width=np.power([ d['weight'] for (u,v,d) in G.edges(data=True)],2), 
        with_labels=False, font_color='black', edge_color='grey', cmap=plt.cm.Spectral, alpha=0.7)
    plt.show()
    print('visualization of communities in the network')



    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.

    print('Spectral Decomposition Technique')
    start = time.time()
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)
    end = time.time()
    print(f'Time Elapsed: {end-start:0.2f}secs')
    print(f"No of communities detected: {np.unique(graph_partition_fb[:,1],return_counts=True)[1].shape[0]}")
    part = {row[0]:row[1] for row in graph_partition_fb}
    # print(part)
    G = nx.Graph(adj_mat_fb, nodetype=int)
    plt.figure(figsize=(5,5))
    values = [part.get(node) for node in G.nodes()]
    # clust=[i*30 for i in nx.clustering(G, weight='weight').values()]
    nx.draw(G,font_size=8, node_size=30, node_color=values, 
        width=np.power([ d['weight'] for (u,v,d) in G.edges(data=True)],2), 
        with_labels=False, font_color='black', edge_color='grey', cmap=plt.cm.Spectral, alpha=0.7)
    plt.show()
    print('visualization of communities in the network')

    

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.

    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    plt.figure(figsize=(5,5))
    plt.imshow(clustered_adj_mat_fb, cmap="Purples_r", interpolation="none")
    plt.show()
    print('Sorted Adjacency Matrix')


    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.

    print('Louvain algorithm')
    start = time.time()
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    end = time.time()
    print(f'Time Elapsed: {end-start:0.2f}secs')
    print(f"No of communities detected: {np.unique(graph_partition_louvain_fb[:,1],return_counts=True)[1].shape[0]}")
    part = {row[0]:row[1] for row in graph_partition_louvain_fb}
    G = nx.Graph(adj_mat_fb, nodetype=int)
    plt.figure(figsize=(5,5))
    values = [part.get(node) for node in G.nodes()]
    # clust=[i*30 for i in nx.clustering(G, weight='weight').values()]
    nx.draw(G,font_size=8, node_size=30, node_color=values, width=np.power([ d['weight'] for (u,v,d) in G.edges(data=True)],2), 
        with_labels=False, font_color='black', edge_color='grey', cmap=plt.cm.Spectral, alpha=0.7)
    plt.show()
    print('visualization of communities in the network')

    print('*'*30)



    # ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    vecb = sorted(fielder_vec_btc)
    plt.scatter(range(len(vecb)),vecb)
    # plt.scatter(range(len(fielder_vec_btcve)),fielder_vec_btc)
    plt.xlabel('Node Index')
    plt.ylabel('Fielder Vector Value')
    plt.title('Fielder Vector')
    # plt.legend()
    plt.show()

    adj_mat_magnitude = np.abs(adj_mat_btc)
    plt.imshow(adj_mat_magnitude, cmap='Purples_r', interpolation='none')
    plt.title('Adjacency Matrix')
    # plt.colorbar()
    plt.show()

    print('Spectral Decomposition One Iteration')
    print(f"No of communities detected: {np.unique(graph_partition_btc[:,1],return_counts=True)[1].shape[0]}")
    part = {row[0]:row[1] for row in graph_partition_btc}
    # print(part)
    G = nx.Graph(adj_mat_btc, nodetype=int)
    plt.figure(figsize=(5,5))
    values = [part.get(node) for node in G.nodes()]
    # clust=[i*30 for i in nx.clustering(G, weight='weight').values()]
    nx.draw(G,font_size=8, node_size=30, node_color=values, 
        width=np.power([ d['weight'] for (u,v,d) in G.edges(data=True)],2), 
        with_labels=False, font_color='black', edge_color='grey', cmap=plt.cm.Spectral, alpha=0.7)
    plt.show()
    print('visualization of communities in the network')



    # # Question 2
    start = time.time()
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)
    end = time.time()
    # print(graph_partition_btc[10])
    print(f'Time Elapsed: {end-start:0.2f}secs')
    print(f"No of communities detected:{np.unique(graph_partition_btc[:,1],return_counts=True)[1].shape[0]}")
    G = nx.Graph(adj_mat_btc, nodetype=int)
    # print(" G  Values :",G[10])
   
    part = {row[0]:row[1] for row in graph_partition_btc}
    # part = {node: data['class'] for node, data in G.nodes(data=True)}

    # print(len(G))
    all_nodes = set(G.nodes())
    nodes_in = set(part.keys())
    nodes_out = all_nodes - nodes_in
    # G.remove_nodes_from(nodes_without_community)
    # print(len(G))
    # print(len(part))
    # print("nodes without :",nodes_without_community)
    # print("nodes with :",nodes_with_community)
    default_value = 0   
    for node in nodes_out:
        # print(node)
        part[node] = default_value
    # print("the Part Values :",part)

    plt.figure(figsize=(5,5))
     
    values = [part.get(node) for node in G.nodes()]
    # clust=[i*30 for i in nx.clustering(G, weight='weight').values()]
    try:
        
        nx.draw(G,font_size=8, node_size=30, node_color=values, width=np.power([ d['weight'] for (u,v,d) in G.edges(data=True)],2), 
            with_labels=False, font_color='black', edge_color='grey', cmap=plt.cm.Spectral, alpha=0.7)
        plt.show()
        print('visualization of communities in the network')
    except Exception as e:
        print("error value is :",str(e))

    # # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)
    plt.figure(figsize=(5,5))
    plt.imshow(clustered_adj_mat_btc, cmap="Purples_r", interpolation="none")
    plt.show()
    print('Sorted Adjacency Matrix')



    # # Question 4

    print('Louvain algorithm')
    start = time.time()
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    end = time.time()
    print(f'Time Elapsed: {end-start:0.2f}secs')
    print(f"No of communities detected:{np.unique(graph_partition_louvain_btc[:,1],return_counts=True)[1].shape[0]}")
    part = {row[0]:row[1] for row in graph_partition_louvain_btc}
    G = nx.Graph(adj_mat_btc, nodetype=int)
    all_nodes_l = set(G.nodes())
    nodes_in_l = set(part.keys())
    nodes_out_l = all_nodes_l - nodes_in_l
    default_value = 0   
    for node in nodes_out_l:
        part[node] = default_value
    plt.figure(figsize=(5,5))
    values = [part.get(node) for node in G.nodes()]
    clust=[i*30 for i in nx.clustering(G, weight='weight').values()]
    try:
        nx.draw(G,font_size=8, node_size=30, node_color=values, 
        width=np.power([ d['weight'] for (u,v,d) in G.edges(data=True)],2), 
        with_labels=False, font_color='black', edge_color='grey', cmap=plt.cm.Spectral, alpha=0.7)
        plt.show()
        print('visualization of communities in the network')
    except Exception as e:
        print("error value is :",str(e))

