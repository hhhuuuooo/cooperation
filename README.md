# Code of multi-agent collaborate RL-Route algorithm<br>
The code of multi-agent distributed cooperative routing algorithm based on reinforcement-learning.
We have tune the hyperparameters.<br>
## Project description
We conduct the packet routing simulation on a large-scale dynamic network in order to cope with the issues caused by movement of nodes.And we carry out experiments for both a fixed large topology having 49 nodes and a topology where a small amount of nodes move respectively.<br>
***For the fixed large topology***, a certain amount of packets having random source and destination pairs, refer to network load, are generated and entered into nodes' sending-queue during network initialization.Each node has specific length sending-queue and receiving-buffer respectively.When the next-hop node's receiving-queue is full,the packet will be **retransmitted**.And if the packet has been retransmitted exceeding the limit of times we set,then the packet will be dropped, which is regraded as **congestion**. We compare the performance of the traditional shortest-path via Dijkstra's algorithm and our proposed Collabarate DQN Route algorithm in terms of **average-delivery-ratio, average-delivery-time, packet-loss-number and retransmission-ratio.**<br>
***For the dynamic large topology***, a small amount of nodes are randomly selected to change its locations and reestablish new links on the basis of existing topoloy. And we evaluate the above mentioned performance of four approaches when the topology changes dynamically including shortest-path(SP), Collaborate DQN Route algorithm without retraining, Collabarate DQN Route algorithm with global training, Collabarate DQN Route algorithm with local training.<br>
The system model and **MARL learning framework** is as follows:<br>
![](https://img-blog.csdnimg.cn/img_convert/25db9b3c957ec25276a0be7e8f05d71f.png)<br>
The **packet routing process** and the **forward path change of packet p** due to the movement of nodes are as follows:<br>
![alt-text-1](https://img-blog.csdnimg.cn/img_convert/a955c93a2fd1f66060e32e2d103633bc.png)![alt-text-2](https://img-blog.csdnimg.cn/img_convert/9978baec61297b0dad697a85414716b2.png)
## Code structure
* our_env3.py
  * Create the environment for both fixed network and dynamic network scenarios. And the functions it possessed are as follows:
    * Generate and save the topology, initialize the network and all parameters.
    * Implement the route decision and the delivery of packets
    * Save and load the training neural networks' model
* our_agent.py
  * Create DQN agent instance and realize multi-agent collaboration. And the functions it possessed are as follows:
    * Select action with epsilon-greedy policy.
    * Extract Experiences and teach the agent to make more accurate route policy through achieving multi-agent collaboration.
* dqnSimualtion.py
  * Train and save the neural networks of each agent under a series of network load for a certain number of episodes.
  * Test the performance of different methods in fixed topology and dynamic topology respectively.
    * For fixed topology, utilizing the saved model to test the SP and Collaborate RL-Route algorithm.
    * For dynamic topology, carrying out global retraining and local retraining based on the existing neural network model and evaluating their performance.
* dynetwork.py
  * Define network and realize the packets' generation of the network.And the functions it possessed are as follows:
    * Generate packets having random source and destination pairs during intialization.
    * Generate new packets when the congestion occurs or a packet has successfully arrive at its destination.
* get_graph.py
  * Generate the regular 7*7 topology using in simulations.
* UpdateEdges.py
  * Perform dynamically changing topology through randomly selecting nodes to change their coordinates.
* neural_network.py
  * Define the struture of DQN agent's Neural Network.
* replay_memory.py
  * Accomplish saving and sampling the generated experiences during routing process.
* Packet.py
  * Define packet object.
* net_params.pth
  * The trained neural networks' model
* experiences.txt
  * The generated data set.
* draw_plots.py
  * Plot the experimental results.
* Setting.json
  * Define all parameters required for the simulation experiments.
## Requirements
* NerworkX
* Python 3.6
* Pytorch
* Matplotlib
## Usage
You can run dqnSimualtion.py to test Collaborate DQN Agent in environment created by our_env3.py.And the parameters can be modified in Setting.json.
## Please cite the following paper
Multi Agent Distributed Cooperative Routing for Maritime Emergency Communication.
The authors of this paper:<br>
**Tingting Yang** Dalian Maritime University yangtingting@dlmu.edu.cn<br>
**Yujia Huo** Dalian Maritime University hyj09@dlmu.edu.cn<br>
**Chengzhuo Han** Southeast University 230208965@seu.edu.cn<br>
**Xin Sun** Dalian Maritime University sunny_xin1996@163.com


