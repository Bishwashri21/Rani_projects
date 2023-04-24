import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import numpy as np
import pandas
# import PraSar
import cdlib
from cdlib import algorithms, NodeClustering
from cdlib import evaluation
from networkx.classes.function import common_neighbors
# Load the Karate Club network
#G = pandas.read_csv("dolphins.csv")
#G = nx.karate_club_graph()
def str_to_int(x):
    return [[int(v) for v in line.split()] for line in x]
def get_neighbors(node):
    return list(G.neighbors(node))

def get_common_neighbors(a, b):
    return list(common_neighbors(G, a, b))
path='./datasets/Real-world/15rw_t07.csv'
pathh='./datasets/Real-world/'
G = nx.Graph()
edge_file=open(path,'r')
edge_list = edge_file.readlines()
for edge in edge_list:
    edge = edge.split()
    comm_file=open(pathh+'15rw_comm_t07.txt','r')
    comm_list=comm_file.readlines()
    comm_list=str_to_int(comm_list)
    G.add_node(int(edge[0]))
    G.add_node(int(edge[1]))
    G.add_edge(int(edge[0]), int(edge[1]))
G = G.to_undirected()
adj_list = {}
for node in G.nodes():
    neighbors = get_neighbors(node)
    adj_list[int(node)] = neighbors
#print("Adjacency list", adj_list)
adjacency_list1 = adj_list
#print(G.edges())
# Create an empty dictionary to store nodes grouped by degree
'''each grouped_node key will have values in order = [degree, labels, diffusion_flag
 ,highest_degree_node if degree=1]'''

#Functions
def visualize(grouped_nodes):
    color=[]
    community=1
    for key in grouped_nodes:
        if grouped_nodes[key][1][0]==1:
            color.append('blue')
        elif grouped_nodes[key][1][0]==2:
            color.append('green')
        elif grouped_nodes[key][1][0]==3:
            color.append('brown')
        elif grouped_nodes[key][1][0]==4:
            color.append('white')
        elif grouped_nodes[key][1][0]==5:
            color.append('yellow')
        elif grouped_nodes[key][1][0]==6:
            color.append('orange')
        elif grouped_nodes[key][1][0]==7:
            color.append('cyan')
        elif grouped_nodes[key][1][0]==8:
            color.append('magenta')
        elif grouped_nodes[key][1][0]==9:
            color.append('purple')
        elif grouped_nodes[key][1][0]==10:
            color.append('violet')
        elif grouped_nodes[key][1][0]==11:
            color.append('indigo')
        elif grouped_nodes[key][1][0]==12:
            color.append('grey')
        elif grouped_nodes[key][1][0]==13:
            color.append('pink')
        elif grouped_nodes[key][1][0]==14:
            color.append('olive')
        elif grouped_nodes[key][1][0]==15:
            color.append('skyblue')
        elif grouped_nodes[key][1][0]==16:
            color.append('indigo')
        else:
            color.append('red')
    community=community+1
    plt.figure(figsize=(50,50))
    nx.draw_networkx(G,with_labels= True,node_color=color,node_size=300)
    plt.show()

def allHaveSameLabel(node,grouped_nodes):
    neighbors=list(G.neighbors(node))
    a=grouped_nodes[neighbors[0]][1]
    for nodex in neighbors:
        if grouped_nodes[nodex][1]!=a:
            return False
        return True
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def allHaveLabel(dict):
    for key in dict:
        if 0 in dict[key][1]:
            return False
    return True
def ECCix(node):
    dictEccix= {}
    neighbouri=list(G.neighbors(node))
    for neighbors_of_node in neighbouri:
        neighbourx=list(G.neighbors(neighbors_of_node))
        eccix=len(intersection(neighbouri,neighbourx))/(len(neighbourx)*len(neighbouri))
        dictEccix[neighbors_of_node]=eccix
    return dictEccix

def candidate_node_score(node):
    value = G.degree(node)
    neighbors_of_node=G.neighbors(node)
    for key in neighbors_of_node:
        value+=influence(node,key)
    return value

def ECCi(node):
    templist=ECCix(node)
    sumofeccix=0
    for i in templist:
        sumofeccix+=templist[i]
    return sumofeccix/len(templist)

def SECCi(neighbour_nodes):
    dictSECCi={}
    tempdict={}
    for node in neighbour_nodes:
        tempdict=ECCix(node)
        dictSECCi[node]=sum(tempdict.values())
    return dictSECCi

def Si(degree_n_nodelist):
    dictSi={}
    for node in degree_n_nodelist:
        sum=0
        neighbourlist=list(G.neighbors(node))
        for neighbour in neighbourlist:
            sum+=G.degree(neighbour)
        dictSi[node]=sum
    return dictSi

def Connectedness(node,grouped_nodes):
    neighbors=list(G.neighbors(node)) #[2,3,4]
    dictLabel=defaultdict(list)
    commLabel={}
    for n in neighbors:
        for label in grouped_nodes[n][1]:
            dictLabel[label].append(n)   #{1:[2],2:[3,4]}
    for key in dictLabel:  #2
        sum=0
        for comm_node in dictLabel[key]:#3
            sum+=G.degree(comm_node)
        inter = []            
        if len(dictLabel[key])>1:
            inter=list(G.neighbors(dictLabel[key][0]))
            for i in range(1,len(dictLabel[key])):
                inter=intersection(inter,G.neighbors(dictLabel[key][i]))
        commLabel[key]=sum+len(inter)
    return commLabel



def influence(n1,n2):
    neighbor1=list(G.neighbors(n1))
    neighbor2=list(G.neighbors(n2))
    return (((len(intersection(neighbor1,neighbor2))+2)*len(neighbor1))/((len(neighbor2))+1))

def commoneNeighbors(n1,n2):
    l1=list(G.neighbors(n1))
    l2=list(G.neighbors(n2))
    return intersection(l1,l2)

def lsmd(G):
    grouped_nodes = {}
    label_counter=1
    nodes=nx.nodes(G)
    n_size=[]
    # Group nodes by degree
    for node in nodes:
        n_size.append(150)
        degree = G.degree(node)
        if node in grouped_nodes:
            grouped_nodes[node].append(degree)
        else:
            grouped_nodes[node] = [degree]
        grouped_nodes[node].append([0])
        grouped_nodes[node].append(False)
 
    #Calculating degree of neighbors of nodes with degree 1 and finding nodes of deg=1
    deg_1_nodes=[]
    for node in nodes: 
        if grouped_nodes[node][0]!=1:
            grouped_nodes[node].append(-1)
        else:
            grouped_nodes[node].append(G.degree(list(G.neighbors(node))[0]))
            deg_1_nodes.append(node)
    #print("Algorithm 1")
    #Algorithm 1
    while len(deg_1_nodes)>0:
        change_label= True
        #Find the target node for deg 1(non-labelled)
        target_node=-1
        max_neighbor_deg=-1
        for node in deg_1_nodes:
            if grouped_nodes[node][3]>max_neighbor_deg and grouped_nodes[node][2]==False:
                max_neighbor_deg=grouped_nodes[node][3]
                target_node=node
        #print("Target Node = ",target_node)
        #for target node
        if grouped_nodes[target_node][1]==[0]:
            if grouped_nodes[list(G.neighbors(target_node))[0]][1]==[0]:
                #print("Target = ",target_node,"neighbor= ",list(G.neighbors(target_node))[0],grouped_nodes[list(G.neighbors(target_node))[0]])
                if label_counter not in grouped_nodes[target_node][1]:
                    grouped_nodes[target_node][1].append(label_counter)
                if 0 in grouped_nodes[target_node][1]:
                    grouped_nodes[target_node][1].remove(0)
                if grouped_nodes[target_node][1][0] not in grouped_nodes[list(G.neighbors(target_node))[0]][1]:
                    grouped_nodes[list(G.neighbors(target_node))[0]][1].append(grouped_nodes[target_node][1][0])
                if 0 in grouped_nodes[list(G.neighbors(target_node))[0]][1]:
                    grouped_nodes[list(G.neighbors(target_node))[0]][1].remove(0)
            else:
                grouped_nodes[target_node][1]=grouped_nodes[list(G.neighbors(target_node))[0]][1]
                if 0 in grouped_nodes[target_node][1]:
                    grouped_nodes[target_node][1].remove(0)
                change_label=False
            grouped_nodes[target_node][2]=True
            #print("label= ", grouped_nodes[target_node][1])

            #Calculating ECCix and ECCi
            edge_clustering_coeff_i_x=ECCix(list(G.neighbors(target_node))[0])
            edge_clustering_coeff_i=ECCi(list(G.neighbors(target_node))[0])
            second_level_neighbours=[]
            for key in edge_clustering_coeff_i_x:
                if(edge_clustering_coeff_i_x[key]>=edge_clustering_coeff_i):
                    if grouped_nodes[target_node][1][0] not in grouped_nodes[key][1]:
                        grouped_nodes[key][1].append(grouped_nodes[target_node][1][0])
                    if 0 in grouped_nodes[key][1]:
                        grouped_nodes[key][1].remove(0)
                    second_level_neighbours.append(key)
                grouped_nodes[list(G.neighbors(target_node))[0]][2]=True
  
            #Finding the next best neighbour
            sum_of_edge_clustering_coeff_in_second_order_neighbours=SECCi(second_level_neighbours)
            bestNeighbour=-1
            bestVal=-1
            for key in sum_of_edge_clustering_coeff_in_second_order_neighbours:
                if sum_of_edge_clustering_coeff_in_second_order_neighbours[key]>bestVal:
                    bestVal=sum_of_edge_clustering_coeff_in_second_order_neighbours[key]
                    bestNeighbour=key

            #diffusing to the 3rd level neighbours
            if grouped_nodes[bestNeighbour][2]==False:
                edge_clustering_coeff_i_x=ECCix(bestNeighbour)
                edge_clustering_coeff_i=ECCi(bestNeighbour)
                third_level_neighbours=[]
                for key in edge_clustering_coeff_i_x:
                    if(edge_clustering_coeff_i_x[key]>=edge_clustering_coeff_i):
                        if grouped_nodes[target_node][1][0] not in grouped_nodes[key][1]:
                            grouped_nodes[key][1].append(grouped_nodes[target_node][1][0])
                        if 0 in grouped_nodes[key][1]:
                            grouped_nodes[key][1].remove(0)
                        #grouped_nodes[key][2]=True
                        third_level_neighbours.append(key)
                grouped_nodes[bestNeighbour][2]=True
    
            if change_label==True:
                label_counter+=1
        deg_1_nodes.remove(target_node)

    '''Degree 1 nodes are diffused
        Now we start with degree 2 Nodes'''
    #print("after deg 1 transformation:")
    n=2
    max_degree=2
    for key in grouped_nodes:
        if grouped_nodes[key][0]>max_degree:
            max_degree=grouped_nodes[key][0]
    first_key=True
    #calculating nodes of degree n
    while n<=max_degree:
        degree_n_nodes=[]
    
        for node in grouped_nodes:
            if grouped_nodes[node][0]==n : 
                degree_n_nodes.append(node)

        while len(degree_n_nodes)>0:

        #finding target node  (Si = sum of degree of neighbors)
            max_si=-1
            target_node=-1
            si=Si(degree_n_nodes)
            for key in si:
                #print(key,si[key],max_si)
                if si[key]>max_si:
                    #temp_neighbors=list(G.neighbors(key))
                    if first_key==True :
                        if allHaveSameLabel(key,grouped_nodes)==False:
                            continue
                    first_key=False
                    target_node=key
                    max_si=si[key]

                
            neighbors_of_target_node=list(G.neighbors(target_node))


            #calculating connectedness of target node
            connectedness_of_node= Connectedness(target_node,grouped_nodes)
            next_label=-1
            max_conn=-1
            can_diffuse=False
            #print("connectedness of node = ", connectedness_of_node)
            #choosing the appropriate label for target node


            for key in connectedness_of_node:
                if connectedness_of_node[key]>max_conn:
                    if ECCi(target_node)==0:
                        if key==0:
                            continue
                    max_conn=connectedness_of_node[key]
                    next_label=key
                if ECCi(target_node)==0:
                    if next_label not in grouped_nodes[target_node][1]:
                        grouped_nodes[target_node][1].append(next_label)
                    if 0 in grouped_nodes[target_node][1]:
                        grouped_nodes[target_node][1].remove(0)
                    break
                if key==0:
                    can_diffuse=True
            if(ECCi(target_node)==0):
                break
            if next_label==0:
                next_label=label_counter
                label_counter=label_counter+1
            if next_label not in grouped_nodes[target_node][1]:
                grouped_nodes[target_node][1].append(next_label)
            if 0 in grouped_nodes[target_node][1]:
                grouped_nodes[target_node][1].remove(0)

            #calculating the influecne of first level neighbors
            avg_influence=0
            dictinlufluence={}
            if can_diffuse==True:
                grouped_nodes[target_node][2]=True
                for node in neighbors_of_target_node:
                    dictinlufluence[node]=influence(target_node,node)
                    avg_influence+=dictinlufluence[node]
                avg_influence/=len(neighbors_of_target_node)

                #diffusing to first level neighbors
                for key in dictinlufluence:
                    if key not in dictinlufluence.keys():
                        continue
                    if dictinlufluence[key]>avg_influence:
                        if next_label not in grouped_nodes[key][1]:
                            grouped_nodes[key][1].append(next_label)
                        if 0 in grouped_nodes[key][1]:
                            grouped_nodes[key][1].remove(0)
                        common_neighbors=list(nx.common_neighbors(G,target_node,key))
                 
                    #common nodes of target node and diffused first level neighbor will also get same label
                        for node in common_neighbors:
                            if next_label not in grouped_nodes[node][1]:
                                grouped_nodes[node][1].append(next_label)
                            if 0 in grouped_nodes[node][1]:
                                grouped_nodes[node][1].remove(0)
                        second_level_neighbours=[]
                        #commmon nodes of first level neighbors will also have same label
                        for i in range(len(neighbors_of_target_node)-1):
                            for j in range(i+1,len(neighbors_of_target_node)):
                                common_neighbors=list(nx.common_neighbors(G,neighbors_of_target_node[i],neighbors_of_target_node[j]))
                        for node in common_neighbors:
                            if node not in second_level_neighbours:
                                second_level_neighbours.append(node)
                            if next_label in grouped_nodes[node][1]:
                                continue
                            if next_label not in grouped_nodes[node][1]:
                                grouped_nodes[node][1].append(next_label)
                            if 0 in grouped_nodes[node][1]:
                                grouped_nodes[node][1].remove(0)
                        #for 3rd level neighbors
                        target_node2=-1
                        max_ecci=-1
                        for node in second_level_neighbours:
                            if ECCi(node)>max_ecci:
                                max_ecci=ECCi(node)
                                target_node2=node
                    
                        #condition 1 for diffusion
                        avg_influence=0
                        dictinlufluence={}
                        for node in list(G.neighbors(target_node2)):
                            dictinlufluence[node]=influence(target_node2,node)
                            avg_influence+=dictinlufluence[node]
                        avg_influence/=len(list(G.neighbors(target_node2)))
                        for node in list(G.neighbors(target_node2)):
                            if dictinlufluence[node]>=avg_influence:
                                if next_label not in grouped_nodes[node][1]:
                                    grouped_nodes[node][1].append(next_label)
                                if 0 in grouped_nodes[node][1]:
                                    grouped_nodes[node][1].remove(0)
                        #condition 2 for diffustion
                        dicteccix=ECCix(target_node2)
                        eccinode=ECCi(target_node2)
                        for key in dicteccix:
                            if dicteccix[key]>eccinode:
                                if next_label not in grouped_nodes[key][1]:
                                    grouped_nodes[key][1].append(next_label)
                                if 0 in grouped_nodes[key][1]:
                                    grouped_nodes[key][1].remove(0)
                        grouped_nodes[target_node2][2]=True
            degree_n_nodes.remove(target_node)
        n+=1
    #choosing most appropriate labels for overlapping labels
    nodes=nx.nodes(G)
    for node in nodes:
        connectedness_of_node=Connectedness(node,grouped_nodes)
        appropriate_label=-1
        max_connectedness=-1
        for key in connectedness_of_node:
            if connectedness_of_node[key]>max_connectedness:
                max_connectedness=connectedness_of_node[key]
                appropriate_label=key
        grouped_nodes[node][1]=[appropriate_label]
    #print(grouped_nodes)
    #calculating no of nodes of each community 
    dictLabel=defaultdict(list)
    for n in nodes:
        dictLabel[grouped_nodes[n][1][0]].append(n) 
    #ignoring the largest community
    dictLabelSize=defaultdict(list)
    for key in dictLabel:
        dictLabelSize[key]=len(dictLabel[key])
    max_label_size=-1
    largest_comm=-1
    avg_comm_size=0
    for key in dictLabelSize:
        if dictLabelSize[key]>max_label_size:
            max_label_size=dictLabelSize[key]
            largest_comm=key
            avg_comm_size+=dictLabelSize[key]
    avg_comm_size-=dictLabelSize[largest_comm]
    del dictLabelSize[largest_comm]
    del dictLabel[largest_comm]

    #computing avg community size
    avg_comm_size/=len(dictLabelSize)


    for key in dictLabelSize:
        if dictLabelSize[key]<avg_comm_size:
            max_score=-1
            rep_node=-1
            for node in dictLabel[key]:
                if candidate_node_score(node) > max_score:
                    max_score=candidate_node_score(node)
                    rep_node=node 
            initial_label=key
            max_score_neighbour=-1
            highest_neighbour=-1
            for node in list(G.neighbors(rep_node)):
                if candidate_node_score(node)>max_score_neighbour:
                    max_score_neighbour=candidate_node_score(node)
                    highest_neighbour=node
            if max_score_neighbour>max_score and G.degree(highest_neighbour) > G.degree(rep_node) and grouped_nodes[highest_neighbour][1]!= grouped_nodes[rep_node][1]:
                for node in dictLabel[initial_label]:
                    grouped_nodes[node][1]=grouped_nodes[highest_neighbour][1]

    #print("after deg n transformation")
    #print(grouped_nodes)

    #making list of communities
    i=1

    communities=[]
    while i<label_counter:
        temp=[]
        for node in grouped_nodes:
            if grouped_nodes [node][1][0]==i:
                temp.append(node)
        i+=1
        if len(temp)>0:
            communities.append(temp)
    max_com=0
    #print(communities)
    for i in communities:
        if len(i)!=0:
            max_com+=1
    
    return communities
    visualize(communities)
print(lsmd(G))

