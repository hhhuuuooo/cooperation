import json
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import copy
import dynetwork
import gym
from gym import error
from gym.utils import closer
import numpy as np
import networkx as nx
import math
import os
from our_agent import QAgent
import Packet
import random
import UpdateEdges as UE
from neural_network import NeuralNetwork
import matplotlib

'''Open file Setting.json which contains learning and network parameters. '''
main_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = main_dir + '/'
with open(main_dir + 'Setting.json') as f:
    setting = json.load(f)

""" This class contains our gym environment which contains all of the necessary components for agents to take actions and receive rewards. file contains functions: 

    change_network: edge deletion/re-establish, edge weight change
    purgatory: queue to generate additional queues as previous packets are delivered
    step: obtain rewards for updating Q-table after an action
    is_capacity: check if next node is full and unable to receive packets
    send_packet: attempt to send packet to next node
    reset: reset environment after each episode
    resetForTest: reset environment for each trial (test for different networkloads)
    get_state: obtain packet's position info
    update_queues: update each nodes packet holding queue
    update_time: update packet delivery time
    calc_avg_delivery: helper function to calculate delivery time
    router: used to route all packets in ONE time stamp
    updateWhole: helper funciton update network environment and packets status
"""


class dynetworkEnv(gym.Env):
    '''Initialization of the network'''

    def __init__(self):
        self.nnodes = setting['NETWORK']['number nodes']
        self.nedges = setting['NETWORK']['edge degree']
        self.max_queue = setting['NETWORK']['holding capacity']
        self.max_transmit = setting['NETWORK']['sending capacity']
        self.max_initializations = setting['NETWORK']['max_additional_packets']
        self.npackets = setting['NETWORK']['initial num packets']
        self.max_edge_weight = setting['NETWORK']['max_edge_weight']
        self.min_edge_removal = setting['NETWORK']['min_edge_removal']
        self.max_edge_removal = setting['NETWORK']['max_edge_removal']
        self.move_number = setting['NETWORK']['node_move_number']
        self.edge_change_type = setting['NETWORK']['edge_change_type']
        self.network_type = setting['NETWORK']['network_type']
        self.initial_dynetwork = None
        self.dynetwork = None
        self.router_type = 'dijkstra'
        self.packet = -1
        self.curr_queue = []
        self.remaining = []
        self.nodes_traversed = 0
        self.print_edge_weights = True
        """ below we create a dynetwork type object """
        self.input_q_size = setting['DQN']['take_queue_size_as_input']
        self.input_buffer_size = setting['DQN']['take_buffer_size_as_input']
        self.input_max_neighbour_buffer_size = setting['DQN']['take_max_neighbour_buffer_size_as_input']
        '''For Shortest Path'''
        self.sp_packet = -1
        self.sp_curr_queue = []
        self.sp_remaining = []
        self.sp_nodes_traversed = 0
        self.preds = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = self.init_dqns()
        self.renew_nodes = []
        self.batch_size = setting['DQN']['memory_batch_size']
        self.gamma = setting['AGENT']['gamma_for_next_q_val']
        if self.network_type == 'barabasi-albert':
            network = nx.barabasi_albert_graph(self.nnodes, self.nedges)
        else:
            network = nx.gnm_random_graph(self.nnodes, self.nedges)

        '''Shortest Path specific'''
        sp_receiving_queue_dict, sp_sending_queue_dict = {}, {}
        for i in range(self.nnodes):
            temp = {'sp_receiving_queue': []}
            temp2 = {'sp_sending_queue': []}
            sp_receiving_queue_dict.update({i: temp})
            sp_sending_queue_dict.update({i: temp2})
        del temp, temp2
        nx.set_node_attributes(network, sp_receiving_queue_dict)
        nx.set_node_attributes(network, sp_sending_queue_dict)
        nx.set_node_attributes(network, 0, 'sp_max_queue_len')
        nx.set_node_attributes(network, 0, 'sp_avg_q_len_array')

        receiving_queue_dict, sending_queue_dict = {}, {}
        for i in range(self.nnodes):
            temp = {'receiving_queue': []}
            temp2 = {'sending_queue': []}
            receiving_queue_dict.update({i: temp})
            sending_queue_dict.update({i: temp2})
        del temp, temp2
        '''Attribute added'''
        """ node attributes """
        nx.set_node_attributes(network, copy.deepcopy(
            self.max_transmit), 'max_send_capacity')
        nx.set_node_attributes(network, copy.deepcopy(
            self.max_queue), 'max_receive_capacity')
        nx.set_node_attributes(network, copy.deepcopy(
            self.max_queue), 'congestion_measure')
        nx.set_node_attributes(network, receiving_queue_dict)
        nx.set_node_attributes(network, sending_queue_dict)
        """ edge attributes """
        nx.set_edge_attributes(network, 0, 'num_traversed')
        nx.set_edge_attributes(network, 0, 'edge_delay')
        nx.set_edge_attributes(network, 0, 'sine_state')
        nx.set_edge_attributes(network, 0, 'new')
        """ CongestM Attribute added """
        nx.set_node_attributes(network, 0, 'max_queue_len')
        nx.set_node_attributes(network, 0, 'avg_q_len_array')
        nx.set_node_attributes(network, 0, 'growth')
        """ max_weight for edges """
        for s_edge, e_edge in network.edges:
            network[s_edge][e_edge]['edge_delay'] = random.randint(
                0, self.max_edge_weight)
            # 在setting中将max_edge_weight初始化为10
            network[s_edge][e_edge]['sine_state'] = random.uniform(0, math.pi)
            network[s_edge][e_edge]['initial_weight'] = network[s_edge][e_edge]['edge_delay']
            network[s_edge][e_edge]['new'] = 0
        """ make a copy so that we can preserve the initial state of the network """

        """ here, we save the initial network as a .gexf file  """
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'q-learning/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        nx.write_gpickle(network, results_dir + "graph2.gpickle")
        # network = nx.read_gpickle(results_dir + "graph.gpickle")
        self.initial_dynetwork = dynetwork.DynamicNetwork(
            copy.deepcopy(network), self.max_initializations)
        self.dynetwork = copy.deepcopy(self.initial_dynetwork)
        """ use dynetwork class method randomGeneratePackets to populate the network with packets """
        self.dynetwork.randomGeneratePackets(copy.deepcopy(self.npackets), False)
        self._positions = nx.spring_layout(self.dynetwork._network)

    ''' Function to handle routing all the packets in one time step. 
    Set will_learn to True if we are training and wish to update the Q-table; 
    else if we are testing set will_learn = False. '''

    def router(self, agent,t, will_learn=True, SP = False):
        """ router attempts to route as many packets as the network will allow """
        node_queue_lengths = [0]
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0
        for nodeIdx in self.dynetwork._network.nodes:
            """ the self.nodes_traversed tracks the number of nodes we have looped over, 
            guaranteeing that each packet will have the same epsilon at each time step"""
            self.nodes_traversed += 1
            if self.nodes_traversed == self.nnodes:
                agent.config['update_epsilon'] = True
                self.nodes_traversed = 0
            node = self.dynetwork._network.nodes[nodeIdx]
            """ provides pointer for queue of current node """
            self.curr_queue = node['sending_queue']
            # print("nodeIdx:", end = '')
            # print(nodeIdx)
            # print(node['sending_queue'])
            sending_capacity = node['max_send_capacity']
            holding_capacity = node['max_receive_capacity']
            queue_size = len(self.curr_queue)

            """ Congestion Measure #1: maximum queue lengths """
            if (queue_size > self.dynetwork._max_queue_length):
                self.dynetwork._max_queue_length = queue_size

            """ Congestion Measure #2: avg queue len pt1 """
            if (queue_size > 0):
                node_queue_lengths.append(queue_size)
                num_nonEmpty_nodes += 1
                """ Congestion Measure #3: avg percent at capacity """
                if (queue_size > sending_capacity):
                    """ increment number of nodes that are at capacity """
                    num_nodes_at_capacity += 1
            """ stores packets which currently have no destination path """
            self.remaining = []
            sendctr = 0
            for i in range(queue_size):
                """ when node cannot send anymore packets break and move to next node """
                if sendctr == sending_capacity:
                    self.dynetwork._rejections += (1 *(len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue[0]
                pkt_state = self.get_state(self.packet)
                # get_state函数：返回packet的位置和目的地，pkt_state[0]指的是该packet当前的位置
                nlist = sorted(list(self.dynetwork._network.neighbors(pkt_state[0])))
                # nlist由该packet所在的位置的邻居节点组成
                cur_state = F.one_hot(torch.tensor([pkt_state[1]]), self.nnodes)
                # print("输出最初生成的cur_state的shape:")
                # print(cur_state.size())

                """ whether or not we input nodes' queue_size to the network """
                if (self.input_q_size):
                    cur_size = torch.tensor([len(self.curr_queue)]).unsqueeze(0)
                    # print("生成的cur_size这一张量的shape：")
                    # print(cur_size.size())
                    cur_state = torch.cat((cur_state, cur_size), dim=1)
                    # print("输出加入queue_size后生成的新的cur_state这一张量的shape：")
                    # print(cur_state.size())
                    # dim =1 按照行堆积起来
                if (self.input_buffer_size):
                    receiving_queue_size = len(self.dynetwork._network.nodes[pkt_state[1]]['receiving_queue'])
                    buffer_size = torch.tensor([holding_capacity - sending_capacity - receiving_queue_size]).unsqueeze(0)
                    cur_state = torch.cat((cur_state, buffer_size), dim=1)
                    # print("最终的cur_state的shape:")
                    # print(cur_state.size())
                if (self.input_max_neighbour_buffer_size):
                    buffer_sizes = []
                    for j in nlist:
                        receiving_queue_size = len(self.dynetwork._network.nodes[j]['receiving_queue'])
                        buffer_size = holding_capacity - sending_capacity - receiving_queue_size
                        buffer_sizes.append(buffer_size)
                    max_buffer_size = max(buffer_sizes)
                    max_neighbor_buffer = torch.tensor([max_buffer_size]).unsqueeze(0)
                    # print("输出生成的max_neighbor_buffer这一张量的shape:")
                    # print(max_neighbor_buffer.size())
                    cur_state = torch.cat((cur_state, max_neighbor_buffer), dim=1)
                # torch的squeeze()函数的作用是压缩一个tensor的维数为1的维度，使该tensor降维变成最紧凑的形式
                # unsqueeze()函数的功能是在tensor的某个维度上添加一个维数为1的维度 具体例子见csdn收藏夹
                # torch.cat()是为了把多个tensor进行拼接而存在的

                if SP:
                    action = self.get_next_step(pkt_state[0], pkt_state[1], self.router_type)
                else:
                    action = agent.act(self.dqn[pkt_state[0]], cur_state, nlist)

                reward,  self.remaining, self.curr_queue, action = self.step(action, pkt_state[0])
                if reward != None:
                    sendctr += 1
                if will_learn:
                    # 在training时，更新
                    if action != None:
                        next_state = F.one_hot(
                            torch.tensor([pkt_state[1]]), self.nnodes)
                        """ whether or not we input nodes' queue_size to the network """
                        if (self.input_q_size):
                            next_size = len(self.dynetwork._network.nodes[action]['sending_queue'])
                            next_size_tensor = torch.tensor([next_size]).unsqueeze(0)
                            next_state = torch.cat((next_state, next_size_tensor), dim=1).float()
                        if (self.input_buffer_size):
                            receiving_queue_size = len(self.dynetwork._network.nodes[action]['receiving_queue'])
                            next_buffer_size = holding_capacity - sending_capacity - receiving_queue_size
                            next_buffer_size_tensor = torch.tensor([next_buffer_size]).unsqueeze(0)
                            next_state = torch.cat((next_state, next_buffer_size_tensor), dim=1).float()
                        if (self.input_max_neighbour_buffer_size):
                            next_buffer_sizes = []
                            next_nlist = sorted(list(self.dynetwork._network.neighbors(action)))
                            for j in next_nlist:
                                receiving_queue_size = len(self.dynetwork._network.nodes[j]['receiving_queue'])
                                next_buffer_size = holding_capacity - sending_capacity - receiving_queue_size
                                next_buffer_sizes.append(next_buffer_size)
                            max_buffer_size = max(next_buffer_sizes)
                            max_neighbor_buffer = torch.tensor([max_buffer_size]).unsqueeze(0)
                            # print("输出生成的max_neighbor_buffer这一张量的shape:")
                            # print(max_neighbor_buffer.size())
                            next_state = torch.cat((next_state, max_neighbor_buffer), dim=1)
                        agent.learn(self.dqn[pkt_state[0]], self.dqn, cur_state, action, reward, next_state)



            node['sending_queue'] = self.remaining + node['sending_queue']
        """ Congestion Measure #2: avg queue len pt2 """
        if len(node_queue_lengths) > 1:
            self.dynetwork._avg_q_len_arr.append(
                np.average(node_queue_lengths[1:]))
        """ Congestion Measure #3: percent node at capacity """
        self.dynetwork._num_capacity_node.append(num_nodes_at_capacity)

        self.dynetwork._num_working_node.append(num_nonEmpty_nodes)

        """ Congestion Mesure #4: percent empty nodes """
        self.dynetwork._num_empty_node.append(
            self.dynetwork.num_nodes - num_nonEmpty_nodes)
        # print("num_congestions是：",self.dynetwork._num_congestions)
        self.dynetwork._congestions.append(self.dynetwork._num_congestions)
    '''helper function to update learning enviornment in each time stamp'''

    def updateWhole(self, agent,t, learn=True,  SP = False,savesteps=False):

        self.purgatory(False)
        self.update_queues(False)
        self.update_time(False)
        self.router(agent,t, learn, SP)
    '''Use to update edges in network'''
    def change_network(self):
        # print("change network!")
        # UE.Delete(self.dynetwork, self.min_edge_removal, self.max_edge_removal)
        # UE.Restore(self.dynetwork)
        renew_nodes = UE.Add(self.dynetwork, self.move_number)
        if self.edge_change_type == 'none':
            pass
        elif self.edge_change_type == 'sinusoidal':
            UE.Sinusoidal(self.dynetwork)
        else:
            UE.Random_Walk(self.dynetwork)
        self.changed_dynetwork = dynetwork.DynamicNetwork(copy.deepcopy(self.dynetwork._network), self.max_initializations)
        self.renew_nodes = renew_nodes
        print("renew_nodes:", self.renew_nodes)

    def reset(self, curLoad=None, Change = False, SP = False):
        if Change:
            self.dynetwork = copy.deepcopy(self.changed_dynetwork)
        else:
            self.dynetwork = copy.deepcopy(self.initial_dynetwork)
        if curLoad != None:
            self.npackets = curLoad
        self.dynetwork.randomGeneratePackets(self.npackets, SP)
        print('Environment reset')
    ''' return packet's position and destinition'''

    def purgatory(self, SP = False):
        # purgatory中放的已经被delivered的packet,purgatory中的每个元素为（index，weight）
        # index，weight）：（packet的idx，还差几个时间步会被放入新的节点）
        if SP:
            temp_purgatory = copy.deepcopy(self.dynetwork.sp_purgatory)
            self.dynetwork.sp_purgatory = []
        else:
            temp_purgatory = copy.deepcopy(self.dynetwork._purgatory)
            self.dynetwork._purgatory = []
        for (index, weight) in temp_purgatory:
            self.dynetwork.GeneratePacket(index, SP, weight)
        # Packet的weight对应wait，
        # wait是指how many time steps the packet will have to wait before it is assigned to a new node
        # print("purgatory is:")
        # for x in self.dynetwork._purgatory:
            # print(x, end=' ')
    ''' Takes packets which are now ready to be sent and puts them in the sending queue of the node '''

    def update_queues(self, SP = False):
        if SP:
            sending_queue = 'sp_sending_queue'
            receiving_queue = 'sp_receiving_queue'
        else:
            sending_queue = 'sending_queue'
            receiving_queue = 'receiving_queue'
        for nodeIdx in self.dynetwork._network.nodes:
            node = self.dynetwork._network.nodes[nodeIdx]
            if not SP:
               node['growth'] = len(node[receiving_queue])
            queue = copy.deepcopy(node[receiving_queue])
            for elt in queue:
                ''' increment packet delivery time stamp '''
                pkt = elt[0]
                if elt[1] == 0:
                    node[sending_queue].append(pkt)
                    node[receiving_queue].remove(elt)
                else:
                    idx = node[receiving_queue].index(elt)
                    node[receiving_queue][idx] = (pkt, elt[1] - 1)

    ''' Update time spent in queues for each packets '''

    def update_time(self, SP = False):
        if SP:
            sending_queue = 'sp_sending_queue'
            receiving_queue = 'sp_receiving_queue'
            packets = self.dynetwork.sp_packets
        else:
            sending_queue = 'sending_queue'
            receiving_queue = 'receiving_queue'
            packets = self.dynetwork._packets

        for nodeIdx in self.dynetwork._network.nodes:
            for elt in self.dynetwork._network.nodes[nodeIdx][receiving_queue]:
                ''' increment packet delivery time stamp '''
                pkt = elt[0]
                curr_time = packets.packetList[pkt].get_time()
                packets.packetList[pkt].set_time(curr_time + 1)
            for c_pkt in self.dynetwork._network.nodes[nodeIdx][sending_queue]:
                curr_time = packets.packetList[c_pkt].get_time()
                packets.packetList[c_pkt].set_time(curr_time + 1)

    ''' given an neighboring node (action), will check if node has a available space in that queue. if it does not, the packet stays at current queue. else, packet is sent to action node's queue. '''

    def step(self, action, curNode=None):
        reward = None
        ''' checks if action is None, in which case current node has no neighbors and also checks to see if target node has space in queue '''
        if (action == None) :
            self.curr_queue.remove(self.packet)
            self.remaining.append(self.packet)
            self.dynetwork._rejections += 1
        else:
            reward,  self.curr_queue = self.send_packet(action)
        return reward, self.remaining, self.curr_queue, action

    """ checks to see if there is space in target_nodes queue """

    def is_capacity(self, target_node, SP = False):
        if SP:
            sending_queue = 'sp_sending_queue'
            receiving_queue = 'sp_receiving_queue'
        else:
            sending_queue = 'sending_queue'
            receiving_queue = 'receiving_queue'
        total_queue_len = len(self.dynetwork._network.nodes[target_node][sending_queue]) + \
                          len(self.dynetwork._network.nodes[target_node][receiving_queue])
        return total_queue_len >= self.dynetwork._network.nodes[target_node]['max_receive_capacity']

    ''' Given next_step, send packet to next_step. Check if the node is full/other considerations beforehand. '''

    def send_packet(self, next_step):

        pkt = self.dynetwork._packets.packetList[self.packet]
        curr_node = pkt.get_curPos()
        dest_node = pkt.get_endPos()
        weight = self.dynetwork._network[curr_node][next_step]['edge_delay']
        pkt.set_curPos(next_step)
        self.dynetwork._packets.packetList[self.packet].set_time(pkt.get_time() + weight)
        receiving_capacity = self.max_queue - self.max_transmit
        if len(self.dynetwork._network.nodes[pkt.get_curPos()]['receiving_queue']) >= receiving_capacity:
            # print("发送失败的packet为：", self.packet)
            # print("发生拥塞，节点为：", next_step)
            self.dynetwork._num_congestions += 1
            self.dynetwork._packets.packetList[self.packet]._flag = -1
            if self.dynetwork._initializations < self.dynetwork._max_initializations:
                self.dynetwork.GeneratePacket(self.packet, False, 0, True)
            self.curr_queue.remove(self.packet)
            reward = -50
            return reward, self.curr_queue
        else:
            if pkt.get_curPos() == dest_node:
                """ if packet has reached destination, a new packet is created with the same 'ID' (packet index) but a new destination, which is then redistributed to another node """
                self.dynetwork._delivery_times.append(self.dynetwork._packets.packetList[self.packet].get_time())
                # print("成功到达的packet是：",self.packet )
                self.dynetwork._deliveries += 1
                self.dynetwork._packets.packetList[self.packet]._flag = 1
                if self.dynetwork._initializations < self.dynetwork._max_initializations:
                   self.dynetwork.GeneratePacket(self.packet, False, 0, True)
                self.curr_queue.remove(self.packet)
                reward = 50
                # reward = 20 * self.nnodes
            else:
                self.curr_queue.remove(self.packet)
                try:
                     """ we reward the packet for being sent to a node according to our current reward function """
                     # q = len(self.dynetwork._network.nodes[next_step]['sending_queue']) + len(self.dynetwork._network.nodes[next_step]['receiving_queue'])
                     # q_eq = 0.8 * self.max_queue
                     # w = 5
                     # growth = self.dynetwork._network.nodes[next_step]['growth']
                     # reward = (-(q - q_eq + w * growth))
                     reward = -1
                except nx.NetworkXNoPath:
                    """ if the node the packet was just sent to has no available path to dest_node, we assign a reward of -50 """
                    reward = -50

                self.dynetwork._network.nodes[next_step]['receiving_queue'].append((self.packet, weight))

            return reward,  self.curr_queue

    def get_state(self, pktIdx):
        pkt = self.dynetwork._packets.packetList[self.packet]
        return (pkt.get_curPos(), pkt.get_endPos())

    '''helper function to calculate delivery times'''

    def calc_avg_delivery(self, SP = False):

        if SP:
            delivery_times = self.dynetwork.sp_delivery_times
            avg = sum(delivery_times) / len(delivery_times)
        else:
            ''' sum of avg deliveries / num packets delivered  '''
            try:
                avg = sum(self.dynetwork._delivery_times) / len(self.dynetwork._delivery_times)
            except:
                avg = None
              # print("avg deliv time:", avg)
        return avg

    '''Initialize all neural networks with one neural network initialized for each node in the network.'''

    def init_dqns(self):
        temp_dqns = []
        for i in range(self.nnodes):
            if(self.input_q_size):
              temp_dqn = NeuralNetwork(i, self.nnodes, self.input_q_size)
              temp_dqns.append(temp_dqn)

            if(self.input_buffer_size):
              temp_dqn = NeuralNetwork(i, self.nnodes, self.input_buffer_size)
              temp_dqns.append(temp_dqn)

            if (self.input_max_neighbour_buffer_size):
                temp_dqn = NeuralNetwork(i, self.nnodes, self.input_max_neighbour_buffer_size)
                temp_dqns.append(temp_dqn)
        return temp_dqns

    '''Update the target neural network to match the policy neural network'''

    # 更新target_network参数的函数：在DeepQsimualtion.py中被使用，每隔一定步数，会进行更新
    # 所隔步数在setting.json中定义："num_time_step_to_update_target_network": 10,
    def update_target_weights(self):
        for nn in self.dqn:
            nn.target_net.load_state_dict(nn.policy_net.state_dict())

    def save(self, opt):
        if opt == 1:
            print("训练完毕后save model parameters")
            path = './net_params2.pth'
            states = {}
            for nn in self.dqn:
                index_model = 'model' + str(nn.ID) + '_dict'
                index_optimizer = 'optimizer' + str(nn.ID) + '_dict'
                states[index_model] = self.dqn[nn.ID].policy_net.state_dict()
                states[index_optimizer] = self.dqn[nn.ID].optimizer.state_dict()
            torch.save(states, path)

    def load(self):
        path = './net_params2.pth'
        if os.path.exists("net_params2.pth"):
            self.dqn = self.init_dqns()
            checkpoint = torch.load(path)
            for nn in self.dqn:
                index_model = 'model' + str(nn.ID) + '_dict'
                index_optimizer = 'optimizer' + str(nn.ID) + '_dict'
                nn.policy_net.load_state_dict(checkpoint[index_model])
                nn.target_net.load_state_dict(checkpoint[index_model])
                nn.optimizer.load_state_dict(checkpoint[index_optimizer])

    def helper_calc_reward(self):
        state = F.one_hot(torch.tensor([0]), self.nnodes)

        if (self.input_q_size):
            print("Take_Queue_Size_As_Additional_Input")
            size = torch.tensor([5]).unsqueeze(0)
            state = torch.cat((state, size), dim=1)
            past_reward = self.dqn[1].policy_net(state.float())
        # 初始化时 self.dqn = self.init_dqns()
        if (self.input_buffer_size):
            print("buffer_size_As_Additional_Input")
            size = torch.tensor([5]).unsqueeze(0)
            state = torch.cat((state, size), dim=1)
            past_reward = self.dqn[1].policy_net(state.float())
        if (self.input_max_neighbour_buffer_size):
            print("max_neighbour_buffer_size_As_Additional_Input")
            size = torch.tensor([5]).unsqueeze(0)
            state = torch.cat((state, size), dim=1)
            past_reward = self.dqn[1].policy_net(state.float())
        return past_reward

    ''' Save an image of the current state of the network '''

    def render(self, i):
        node_labels = {}
        for node in self.dynetwork._network.nodes:
            # node_labels[node] = len(self.dynetwork._network.nodes[node]['sending_queue']) + len(self.dynetwork._network.nodes[node]['receiving_queue'])
            node_labels[node] = node
        nx.draw(self.dynetwork._network, pos=self._positions,
                labels=node_labels, node_size=200, font_size=8, font_weight='bold', edge_color='k')
        if self.print_edge_weights:
            edge_labels = nx.get_edge_attributes(
                self.dynetwork._network, 'edge_delay')
            nx.draw_networkx_edge_labels(
                self.dynetwork._network, pos=self._positions, edge_labels=edge_labels)

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'network_images/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.axis('off')
        plt.figtext(0.1, 0.1, "total injections: " +
                    str(self.npackets + self.dynetwork._initializations))
        plt.savefig("network_images/dynet" + str(i) + ".png")
        # plt.show()
        plt.clf()

    def draw(self, i, currTrial):
        node_labels = {}
        for node in self.dynetwork._network.nodes:
            node_labels[node] = node
        nx.draw_networkx_nodes(self.dynetwork._network.nodes, pos=self._positions,node_size=100)
        edge_list = []
        edge_list1 = []
        for s_edge, e_edge in self.dynetwork._network.edges:
            if self.dynetwork._network[s_edge][e_edge]['new'] == 1:
               edge_list.append([s_edge,e_edge])
            else:
               edge_list1.append([s_edge,e_edge])
        nx.draw_networkx_edges(self.dynetwork._network, pos=self._positions, edgelist=edge_list,edge_color='r')
        nx.draw_networkx_edges(self.dynetwork._network,pos=self._positions,edgelist=edge_list1, edge_color='k')
        nx.draw_networkx_labels(self.dynetwork._network.nodes, pos=self._positions,
                                labels=node_labels,font_size=8,font_color='k')
        if self.print_edge_weights:
            edge_labels = nx.get_edge_attributes(
                self.dynetwork._network, 'edge_delay')
            nx.draw_networkx_edge_labels(
                self.dynetwork._network, pos=self._positions, edge_labels=edge_labels)

        script_dir_1 = os.path.dirname(__file__)
        results_dir_1 = os.path.join(script_dir_1, 'images/')
        if not os.path.isdir(results_dir_1):
           os.makedirs(results_dir_1)
        plt.axis('off')
        plt.figtext(0.1, 0.1, "total injections: " +
                str(self.npackets + self.dynetwork._initializations))
        plt.savefig("images/dynet" + str(i) + str("_")+ str(currTrial+1)+ ".png")
        plt.clf()

    def get_next_step(self, currPos, destPos, router_type):
        if len(nx.dijkstra_path(self.dynetwork._network, currPos, destPos, weight='edge_delay')) == 1:
            print("没有下一跳")
            print("currPos:", currPos)
            print("destPos:", destPos)
            print("邻接矩阵：", self.dynetwork.adjacency_matrix)
            return None
        else:
            return nx.dijkstra_path(self.dynetwork._network, currPos, destPos, weight='edge_delay')[1]

    def router_test(self,agent, will_learn = True):
        """ router attempts to route as many packets as the network will allow """
        node_queue_lengths = [0]
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0
        for nodeIdx in self.dynetwork._network.nodes:
            """ the self.nodes_traversed tracks the number of nodes we have looped over, 
            guaranteeing that each packet will have the same epsilon at each time step"""
            self.nodes_traversed += 1
            if self.nodes_traversed == len(self.dynetwork._network.nodes):
                agent.config['update_epsilon'] = True
                self.nodes_traversed = 0
            node = self.dynetwork._network.nodes[nodeIdx]
            """ provides pointer for queue of current node """
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            holding_capacity = node['max_receive_capacity']
            queue_size = len(self.curr_queue)

            """ Congestion Measure #1: maximum queue lengths """
            if (queue_size > self.dynetwork._max_queue_length):
                self.dynetwork._max_queue_length = queue_size

            """ Congestion Measure #2: avg queue len pt1 """
            if (queue_size > 0):
                node_queue_lengths.append(queue_size)
                num_nonEmpty_nodes += 1
                """ Congestion Measure #3: avg percent at capacity """
                if (queue_size > sending_capacity):
                    """ increment number of nodes that are at capacity """
                    num_nodes_at_capacity += 1
            """ stores packets which currently have no destination path """
            self.remaining = []
            sendctr = 0
            for i in range(queue_size):
                """ when node cannot send anymore packets break and move to next node """
                if sendctr == sending_capacity:
                    self.dynetwork._rejections += (1 *(len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue[0]
                pkt_state = self.get_state(self.packet)
                nlist = sorted(list(self.dynetwork._network.neighbors(pkt_state[0])))
                # nlist由该packet所在的位置的邻居节点组成
                cur_state = F.one_hot(torch.tensor([pkt_state[1]]), self.nnodes)
                # print("输出最初生成的cur_state的shape:")
                # print(cur_state.size())

                """ whether or not we input nodes' queue_size to the network """
                if (self.input_q_size):
                    cur_size = torch.tensor([len(self.curr_queue)]).unsqueeze(0)
                    cur_state = torch.cat((cur_state, cur_size), dim=1)

                action = agent.act(self.dqn[pkt_state[0]], cur_state, nlist)
                reward,  self.remaining, self.curr_queue, action = self.step(action, pkt_state[0])
                if reward != None:
                    sendctr += 1
                if will_learn:
                    # 在training时，更新
                    if action != None:
                        next_state = F.one_hot(torch.tensor([pkt_state[1]]), self.nnodes)
                        if (self.input_q_size):
                            next_size = len(self.dynetwork._network.nodes[action]['sending_queue'])
                            next_size_tensor = torch.tensor([next_size]).unsqueeze(0)
                            next_state = torch.cat((next_state, next_size_tensor), dim=1).float()
                        for idx in self.renew_nodes:
                            if idx == pkt_state[0]:
                                agent.learn(self.dqn[pkt_state[0]], self.dqn, cur_state, action, reward, next_state)

            node['sending_queue'] = self.remaining + node['sending_queue']
        if len(node_queue_lengths) > 1:
            self.dynetwork._avg_q_len_arr.append(np.average(node_queue_lengths[1:]))
        self.dynetwork._num_capacity_node.append(num_nodes_at_capacity)
        self.dynetwork._num_working_node.append(num_nonEmpty_nodes)
        self.dynetwork._num_empty_node.append(self.dynetwork.num_nodes - num_nonEmpty_nodes)
        self.dynetwork._congestions.append(self.dynetwork._num_congestions)

