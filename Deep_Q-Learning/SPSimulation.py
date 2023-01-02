from our_env import *
import sys
time_steps = setting["Simulation"]["max_allowed_time_step_per_episode"]
learning_plot = True
comparison_plots = True
trials = setting["Simulation"]["test_trials_per_load"]
show_example_comparison = False
'''Mark true to perform shortest path simultaneously during testing for comparison to Q-learning'''
SP = True
env = dynetworkEnv()
agent = QAgent(env.dynetwork)
starting_size = setting["Simulation"]["test_network_load_min"]
ending_size = setting["Simulation"]["test_network_load_max"] + \
    setting["Simulation"]["test_network_load_step_size"]
step_size = setting["Simulation"]["test_network_load_step_size"]
network_load = np.arange(starting_size,ending_size, step_size)
for i in network_load:
    if i <= 0:
        print("Error: Network load must be positive.")
        sys.exit
    if i >= env.nnodes*(env.max_queue - env.max_transmit):
        print("Error: Network load cannot exceed nodes times max queue size.")

'''--------------------------TESTING PROCESS--------------------------'''
'''Performance Measures for Shortest Path'''
sp_avg_deliv = []
sp_avg_deliv_ratio = []
sp_congestions_number = []
for i in range(len(network_load)):
    curLoad = network_load[i]
    print("---------- Testing Load of ", curLoad, " ----------")
    for currTrial in range(trials):
        print("trial",currTrial)
        env.reset(curLoad, True)
        sp_step = []
        sp_deliveries = []
        sp_not_deliveries = 0
        sp_congestions = 0
        sp_remove = 0
        '''iterate each time step try to finish routing within time_steps'''
        for t in range(time_steps):
            if (t + 1) % 200 == 0:
                print("Time step", t + 1)
            total = env.npackets + env.dynetwork.sp_initializations
            env.updateWhole(agent, learn=False, SP=True, savesteps=False)
            if env.dynetwork.sp_deliveries >= total:
               print("Finished trial ", currTrial)
               break
        for index in env.dynetwork.sp_packets.packetList:
            if env.dynetwork.sp_packets.packetList[index].get_flag() == 0:
                sp_not_deliveries += 1
            if env.dynetwork.sp_packets.packetList[index].get_flag() == -2:
                sp_remove += 1
        print("pkts delivered:", env.dynetwork.sp_deliveries)
        print("pkts not delivered:", sp_not_deliveries)
        print("pkts removed:", sp_remove)
        print("pkts in purgatory:", len(env.dynetwork.sp_purgatory))
        print("congestion happened,the number of dropped packets is:", env.dynetwork.sp_congestions[-1])
        print("total pkts:", env.npackets + env.dynetwork.sp_initializations)
        print("delivery ratio:", env.dynetwork.sp_deliveries / (env.dynetwork.sp_deliveries + env.dynetwork.sp_congestions[-1]))
        sp_avg_deliv.append(env.calc_avg_delivery(True))
        sp_congestions_number.append(env.dynetwork.sp_congestions[-1])
        sp_avg_deliv_ratio.append(env.dynetwork.sp_deliveries / (env.npackets + env.dynetwork.sp_initializations))
print("Shortest Path-average_delivery_time: ", np.around(np.array(sp_avg_deliv),3))
print("Shortest Path-average_delivery_ratio:",sp_avg_deliv_ratio )
print("Shortest Path_congestions:", sp_congestions_number)