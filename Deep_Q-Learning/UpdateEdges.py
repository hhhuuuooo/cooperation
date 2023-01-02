import numpy as np
import random
import math
from collections import defaultdict

''' Functions to handle edges in our network. '''

''' 
Randomly deletes some number of edges
between min_edge_removal and max_edge_removal 
'''
def Delete(dyNetwork, min_edge_removal, max_edge_removal):
    edges = dyNetwork._network.edges()
    deletion_number = random.randint(min_edge_removal, min(max_edge_removal, len(edges) - 1))
    print("deletion_number:", deletion_number )
    strip = random.sample(edges, k=deletion_number)
    temp = []
    for s_edge, e_edge in strip:
        temp.append((s_edge,e_edge,dyNetwork._network[s_edge][e_edge]))
    strip = temp
    print("strip:", strip)
    dyNetwork._network.remove_edges_from(strip)
    dyNetwork._stripped_list.extend(strip)

''' 
Randomly restores some
edges we have deleted 
'''
def Restore(dyNetwork):
    restore_number = random.randint(0, len(dyNetwork._stripped_list))
    restore = random.choices(dyNetwork._stripped_list, k=restore_number)
    dyNetwork._network.add_edges_from(restore)


def Add(dyNetwork, move_number):
    renew_nodes = []
    for i in range(move_number):
        not_full_nodes = list(range(len(dyNetwork._network.nodes)))
        nodeIdx = random.choice(not_full_nodes)
        while len(list(dyNetwork._network.neighbors(nodeIdx))) < 2:
            not_full_nodes.remove(nodeIdx)
            try:
                nodeIdx = random.choice(not_full_nodes)
            except:
                print("Error: All Nodes do not have more than 2 neighbor")
                return
        renew_nodes.append(nodeIdx)
        print("选取的移动node是:", nodeIdx)
        print("选取的移动的node的邻居节点的个数：", len(list(dyNetwork._network.neighbors(nodeIdx))))
        node1 = dyNetwork._network.nodes[nodeIdx]
        position1 = node1['position']
        neighbor_full_nodes = []
        for n in dyNetwork._network.nodes:
            node11 = dyNetwork._network.nodes[n]
            position11 = node11['position']
            distance11 = getDist_P2P(position1, position11)
            if distance11 < 1.8:
                if dyNetwork._network.has_edge(nodeIdx, n):
                   neighbor_full_nodes.append(n)
        for n in neighbor_full_nodes:
            if dyNetwork._network.has_edge(nodeIdx, n):
               dyNetwork._network.remove_edge(nodeIdx, n)
        node_list = []
        node = dyNetwork._network.nodes[nodeIdx]
        position = node['position']
        print("改变前的node['position']：", node['position'])
        for i in range(2):
            if node['position'][i] > 2 :
                node['position'][i] = node['position'][i] - random.uniform(1.8, 2)
            else:
                node['position'][i] = node['position'][i] + random.uniform(1.8, 2)
        print("改变后的node['position']：", node['position'])
        position_new = node['position']
        for idx in dyNetwork._network.nodes:
            node_1 = dyNetwork._network.nodes[idx]
            position_1 = node_1['position']
            distance = getDist_P2P(position_new, position_1)
            if distance < 1.8:
                node_list.append(idx)
        # node_list = list(np.unique(node_list))
        # print("node_list:", node_list)
        # new_neighbor_nodes = random.sample(node_list, len(neighbor_full_nodes))
        # print("new_neighbor_nodes:", new_neighbor_nodes)
        for index in node_list:
            dyNetwork._network.add_edge(nodeIdx, index)
            dyNetwork._network[nodeIdx][index]['edge_delay'] = random.randint(int(distance*10), int(distance*10)+5)
            dyNetwork._network[nodeIdx][index]['new'] = 1
        new_nodes = random.choices(node_list, k = 2)
        # print("new_nodes:", new_nodes)
        for item in new_nodes:
            renew_nodes.append(item)
    renew_nodes = list(np.unique(renew_nodes))
    print("renew_nodes:", renew_nodes)
    return renew_nodes
       # print("与新的邻居节点重连后的graph的edge总数是：", len(dyNetwork._network.edges()))
       # print("与新的邻居节点重连后的graph是：", dyNetwork._network.edges())


''' Randomly change edge weights '''

def Random_Walk(dyNetwork):
    for s_edge, e_edge in dyNetwork._network.edges():
        try:
            changed = random.randint(-2, 2) + dyNetwork._network[s_edge][e_edge]['edge_delay']
            dyNetwork._network[s_edge][e_edge]['edge_delay'] = max(changed, 1)
        except:
            print(s_edge, e_edge)
            
''' 
Change edge weights so that the edge weight 
changes will be roughly sinusoidal across the simulation
'''
def Sinusoidal(dyNetwork):
    for s_edge, e_edge in dyNetwork._network.edges():
        dyNetwork._network[s_edge][e_edge]['edge_delay'] = max(1, int(dyNetwork._network[s_edge][e_edge]['initial_weight']* (1 + 0.5 * math.sin(dyNetwork._network[s_edge][e_edge]['sine_state']))))
        dyNetwork._network[s_edge][e_edge]['sine_state'] += math.pi/6

''' 
Not in use. If it were used the edge weight would be the
average of the number of packets in each
queue of the endpoints of the edge. 
'''
def Average(dyNetwork):
    for node1, node2 in dyNetwork._network.edges(data = False):
        tot_queue1 = dyNetwork._network.nodes[node1]['sending_queue']
        tot_queue2 = dyNetwork._network.nodes[node2]['sending_queue']
        avg = np.avg([tot_queue1, tot_queue2])
        dyNetwork._network[node1][node2]['edge_delay'] = avg
        del tot_queue1, tot_queue2

def getDist_P2P(Point0,PointA):
    distance = math.pow((Point0[0]-PointA[0]),2) + math.pow((Point0[1]-PointA[1]),2)
    distance = math.sqrt(distance)
    return distance

def Add1(dyNetwork, move_number):
    renew_nodes = []
    for i in range(move_number):
        not_full_nodes = list(range(len(dyNetwork._network.nodes)))
        nodeIdx = random.choice(not_full_nodes)
        while len(list(dyNetwork._network.neighbors(nodeIdx))) < 4:
            not_full_nodes.remove(nodeIdx)
            try:
                nodeIdx = random.choice(not_full_nodes)
            except:
                print("Error: All Nodes do not have more than 3 neighbor")
                return
        renew_nodes.append(nodeIdx)
        print("选取的移动node是:", nodeIdx)
        # print("选取的移动的node的邻居节点的个数：", len(list(dyNetwork._network.neighbors(nodeIdx))))
        node1 = dyNetwork._network.nodes[nodeIdx]
        position1 = node1['position']
        print("选取的移动node的position:", node1['position'])
        neighbor_full_nodes = []
        for n in dyNetwork._network.nodes:
            node11 = dyNetwork._network.nodes[n]
            position11 = node11['position']
            distance11 = getDist_P2P(position1, position11)
            if distance11 == 2.5:
                # print("n:", n)
                if dyNetwork._network.has_edge(nodeIdx, n):
                    dyNetwork._network.remove_edge(nodeIdx, n)
                    neighbor_full_nodes.append(n)
                    renew_nodes.append(n)
        node_exchange_idx = random.choice(neighbor_full_nodes)
        print("node_exchange_idx:", node_exchange_idx)
        node_exchange = dyNetwork._network.nodes[node_exchange_idx]
        position2 = node_exchange['position']
        print("node_exchange['position']:", node_exchange['position'])
        for n in dyNetwork._network.nodes:
            node3 = dyNetwork._network.nodes[n]
            position3 = node3['position']
            distance23 = getDist_P2P(position2, position3)
            if distance23 == 2.5:
                if dyNetwork._network.has_edge(node_exchange_idx, n):
                    dyNetwork._network.remove_edge(node_exchange_idx, n)
                    # exchange_neighbor_full_nodes.append(n)
                    renew_nodes.append(n)
        print("交换两个节点的位置")
        node1['position'] = position2
        node_exchange['position'] = position1
        print("选取的移动node的position:", dyNetwork._network.nodes[nodeIdx]['position'])
        print("node_exchange['position']:", dyNetwork._network.nodes[node_exchange_idx]['position'])
        renew_nodes.append(node_exchange_idx)
        for n in dyNetwork._network.nodes:
            node11 = dyNetwork._network.nodes[n]
            position11 = node11['position']
            distance11 = getDist_P2P(node1['position'], position11)
            if distance11 == 2.5:
                dyNetwork._network.add_edge(nodeIdx, n)
                dyNetwork._network[nodeIdx][n]['edge_delay'] = random.randint(int(distance11 * 10),int(distance11 * 10) + 5)
                dyNetwork._network[nodeIdx][n]['new'] = 1
            distance22 = getDist_P2P(node_exchange['position'], position11)
            if distance22 == 2.5:
                dyNetwork._network.add_edge(node_exchange_idx, n)
                dyNetwork._network[node_exchange_idx][n]['edge_delay'] = random.randint(int(distance22 * 10),int(distance22 * 10) + 5)
                dyNetwork._network[node_exchange_idx][n]['new'] = 1

        for idx in dyNetwork._network.nodes:
            node = dyNetwork._network.nodes[idx]
            distance_1 = getDist_P2P(node1['position'], node['position'])
            if distance_1 == math.sqrt(12.5):
                renew_nodes.append(idx)
            distance_2 = getDist_P2P(node_exchange['position'], node['position'])
            if distance_2 == math.sqrt(12.5):
                renew_nodes.append(idx)
    renew_nodes = list(np.unique(renew_nodes))
    print("renew_nodes:", renew_nodes)
    return renew_nodes