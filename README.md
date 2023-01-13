# code of multi-agent collaborate RL-Route algorithm<br>
The code of multi-agent distributed cooperative routing algorithm based on reinforcement-learning.
We have tune the hyperparameters.<br>
## Project description
We conduct the packet routing simulation on a large-scale dynamic network in order to cope with the issues caused by movement of nodes.And we carry out experiments for both a fixed large topology having 49 nodes and a topology where a small amount of nodes move respectively.<br>
***For the fixed large topology***, a certain amount of packets having random source and destination pairs, refer to network load, are generated and entered into nodes' sending-queue during network initialization.Each node has specific length sending-queue and receiving-buffer respectively.When the next-hop node's receiving-queue is full,the packet will be **retransmitted**.And if the packet has been retransmitted exceeding the limit of times we set,then the packet will be dropped, which is regraded as **congestion**. We compare the performance of the traditional shortest-path via Dijkstra's algorithm and our proposed Collabarate DQN Route algorithm in terms of **average-delivery-ratio, average-delivery-time, packet-loss-number and retransmission-ratio.**<br>
***For the dynamic large-topology***, a small amount of nodes are randomly selected to change its locations and reestablish new links on the basis of existing topoloy.And we evaluate the above mentioned performance of four approaches when the topology changes dynamically including shortest-path(SP), Collabarate DQN Route algorithm without retraining, Collabarate DQN Route algorithm with global training, Collabarate DQN Route algorithm with local training.<br>
## code structure

