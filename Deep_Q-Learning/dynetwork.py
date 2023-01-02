import time
import random
import numpy as np
import Packet
import copy
import networkx as nx
import json
import os
''' 
    Class created to store network and network attributes as well as generate packets 
    File contains functoins:
        randomGeneratePackets: initialize packets to network in the begining 
        GeneratePacket: generate additional packets as previous packets are delivered to keep network
                        working in specified load
'''
with open('Setting.json') as f:
    setting = json.load(f)

class DynamicNetwork(object):
    ''' 
    Initialize an instance of a dynamic network capable of routing packets
    network: the framework network that packets will be routed on
    max_initializations: the maximum number of packets that can be injected throughout a simulation 
    packets: a list containing all packets currently existing within the network
    '''

    def __init__(self, network, max_initializations=0, packets=None, rejections=0, deliveries=0,):

        self._network = copy.deepcopy(network)
        self._num_nodes = None
        self.adjacency_matrix = nx.to_numpy_matrix(
            self._network, nodelist=sorted(list(self._network.nodes())))
        self._max_initializations = max_initializations
        self._packets = packets
        self._rejections = rejections
        self._deliveries = deliveries
        self.delayed_queue = []
        self._stripped_list = []
        self._delivery_times = []
        self._avg_q_len_arr = []
        self._purgatory = []
        self._num_empty_node = []
        self._num_capacity_node = []
        self._num_working_node = []
        self._num_congestions = 0
        self._num_retransmission = 0
        self._importance = 0
        self._important_nodes = []
        self._congestions = []
        self._retransmission = []
        self._initializations = 0
        self._max_queue_length = 0

        '''shortest path attributes'''
        self.sp_packets = packets
        self.sp_rejections = rejections
        self.sp_deliveries = deliveries
        self.sp_delayed_queue = []
        self.sp_stripped_list = []
        self.sp_delivery_times = []
        self.sp_initializations = 0
        self.sp_max_queue_length = 0
        self.sp_purgatory = []
        self.sp_avg_q_len_arr = []
        self.sp_num_capacity_node = []
        self.sp_num_working_node = []
        self.sp_num_empty_node = []
        self.sp_num_congestions = 0
        self.sp_congestions = []

        self.max_hold = setting['NETWORK']['holding capacity']
        self.max_send = setting['NETWORK']['sending capacity']
    ''' 
    Function used to generate packets handle both first initialization
    or later additional injections 
    num_packets_to_generate: the number of packets that the network will initially load in 

    '''

    def randomGeneratePackets(self, num_packets_to_generate, SP):
        nodeList = {}
        self.num_nodes = len(list(self._network.nodes()))
        not_full_nodes = list(range(self.num_nodes))
        for index in range(num_packets_to_generate):
            curPack, not_full_nodes = self.GeneratePacket(index=index, SP=False, wait=0, midSim=False, not_full_nodes=copy.deepcopy(not_full_nodes))
            # print("curPack是：", curPack.get_index())
            # print("curPack_start:", curPack.get_startPos())
            # print("curPack_end:", curPack.get_endPos())
            #  put curPack into the chosen starting node's (startNode) queue
            if SP:
                self._network.nodes[curPack.get_startPos()]['sp_sending_queue'].append(curPack.get_index())
            else:
                # if len(self._network.nodes[curPack.get_startPos()]['sending_queue']) < 2:
                self._network.nodes[curPack.get_startPos()]['sending_queue'].append(curPack.get_index())
                if curPack.get_startPos() == curPack.get_endPos():
                    print("startPos():",curPack.get_startPos())
                    print("curPos():", curPack.get_curPos())
                    print("endPos():", curPack.get_endPos())
            nodeList[index] = curPack
        ''' create Packets Object, contains references to all packets being routed on the network    '''
        packetsObj = Packet.Packets(nodeList)

        ''' Assign Packets Object to the network '''
        if SP:
            self.sp_packets = copy.deepcopy(packetsObj)
        else:
            self._packets = copy.deepcopy(packetsObj)
        del packetsObj
        del nodeList

    """ 
    called by randomGeneratePackets when generating additional packets
    and after new packets are being generated
    index: the packet ID, int
    wait: how many time steps the packet will have to wait before it is assigned to a new node, int
    midSim: boolean value that tells us if we are generating packets at the inception of the network or not
    not_full_nodes: a list containing the identities of nodes that do not have a full queue 
    """

    def GeneratePacket(self, index, SP, wait=0, midSim=True, not_full_nodes=None):
        """checks to see if we have exceed the maximum number
           of packets alloted in the simulation"""

        initializations = self._initializations
        sending_queue = 'sending_queue'
        receiving_queue = 'receiving_queue'
        packets = self._packets
        purgatory = self._purgatory

        if initializations >= self._max_initializations:
            pass
        elif wait <= 0:
            """ creates a list of not full nodes to check during new packet creation """
            if midSim:
                not_full_nodes = list(range(self.num_nodes))
            startNode = random.choice(not_full_nodes)
            endNode = random.randint(0, self._network.number_of_nodes() - 1)
            while (self.max_send + len(self._network.nodes[startNode][receiving_queue])>= self._network.nodes[startNode]['max_receive_capacity']):
            #while (len(self._network.nodes[startNode][sending_queue]) + len(self._network.nodes[startNode][receiving_queue]) >= self._network.nodes[startNode][
                          # 'max_receive_capacity']):

                    not_full_nodes.remove(startNode)
                    try:
                        startNode = random.choice(not_full_nodes)
                    except:
                        print("Error: All Nodes are Full")
                        return
            while (startNode == endNode):
                    endNode = random.randint(0, self.num_nodes - 1)

            curPack = Packet.Packet(startNode, endNode, startNode, index, 0, flag=0)
            if midSim:
               packets.packetList[index] = curPack
               if SP:
                    self.sp_initializations += 1
               else:
                    self._initializations += 1
               self._network.nodes[curPack.get_startPos()][receiving_queue].append((curPack.get_index(), 0))
               try:
                   purgatory.remove((index, wait))
               except:
                    pass
               return
            return curPack, not_full_nodes
        else:
            purgatory.append((index, wait - 1))
