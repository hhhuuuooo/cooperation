import os.path
import numpy as np
import random
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
import math
import json

main_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = main_dir + '/'
with open(main_dir + 'Setting.json') as f:
    setting = json.load(f)
nnodes =setting['NETWORK']['number nodes']
def new_graph(n):
    border= n*2
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if n == 1:
        return G

    # for i in G.nodes:
    #     G.nodes[i]['x'] = random.randint(0, border)
    #     G.nodes[i]['y'] = random.randint(0, border)
    # positions = nx.spring_layout(G)
    positions = {}
    for i in range(n):
        positions[i] = [ random.uniform(0, math.sqrt(n)),random.uniform(0, math.sqrt(n))]
    for u in G.nodes:
        for v in G.nodes:
            if u==v or G.has_edge(u, v):
                continue
            else:
                distance = getDist_P2P(positions[u],positions[v])
                # print(distance)
                if distance <1.8:
                   G.add_edge(u, v)
                   G[u][v]['edge_delay'] = random.randint(int(distance*10), int(distance*10)+5)
                   # print("G's delay:", G[u][v]['edge_delay'])
    nx.draw(G, pos=positions, node_size=200, font_size=8, font_weight='bold', edge_color='k')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'network_image/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.axis('off')
    plt.savefig("network_image/dynet1"  + ".png")
    # plt.show()
    plt.clf()
    return G, positions

def new_graph1(n,annodes):
    G = nx.Graph()
    areas = []
    G.add_nodes_from(range(n))
    if n == 1:
        return G
    number = 0
    for i in range(3):
        areas.append([])
        for j in range(3):
            # cur_area = area(i,j,annodes)
            for k in range(annodes):


                G.nodes[number]['x'] = random.randint(0, annodes) + i*annodes
                G.nodes[number]['y'] = random.randint(0, annodes) + j*annodes
                G.nodes[number]['id'] = number
                G.nodes[number]['areax'] = i
                G.nodes[number]['areay'] = j
                G.nodes[number]['areaid'] = k
                # cur_area.addnode(G.nodes[number])
                number = number + 1

            # areas[i].append(cur_area)
    for u in G.nodes:
        for v in G.nodes:
            if u == v or G.has_edge(u, v):
                continue
            else:
                G.add_edge(u, v)
    temp = []
    for s_edge in G.nodes:
        for e_edge in G.nodes:
            x = [G.nodes[s_edge]['x'], G.nodes[e_edge]['x']]
            y = [G.nodes[s_edge]['y'], G.nodes[e_edge]['y']]
            sum = (x[0] - x[1]) * (x[0] - x[1]) + (y[0] - y[1]) * (y[0] - y[1])
            if s_edge == e_edge:
                continue
            elif sum >annodes*9:
                temp.append((s_edge, e_edge, G[s_edge][e_edge]))
            else:

                G[s_edge][e_edge]['edge_delay'] = int(sum / annodes) + 1

def new_graph2(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if n == 1:
        return G
    border = 15
    positions = {}
    j = 0
    for i in range(0,49):
        if (i!=0) and (i % 7 == 0) :
            j += 1
        positions[i] = [(border/6)*(i-7*j), 0+j*(border/6)]
    # print("positions:", positions)
    for u in G.nodes:
        for v in G.nodes:
            if u==v or G.has_edge(u, v):
                continue
            else:
                distance = getDist_P2P(positions[u],positions[v])
                # print(distance)
                if distance == (border/6):
                   G.add_edge(u, v)
                   G[u][v]['edge_delay'] = random.randint(int(distance*10), int(distance*10)+5)

    nx.draw(G, pos=positions, node_size=200, font_size=8, font_weight='bold', edge_color='k')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'network_image/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.axis('off')
    plt.savefig("network_image/dynet2" + ".png")
    # plt.show()
    plt.clf()
    return G, positions

def getDist_P2P(Point0,PointA):
    distance = math.pow((Point0[0]-PointA[0]),2) + math.pow((Point0[1]-PointA[1]),2)
    distance = math.sqrt(distance)
    return distance



# new_graph2(49)