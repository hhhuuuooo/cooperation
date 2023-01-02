import sys
from datetime import datetime
import torch.nn.functional as F
import time
from our_env3 import *
import matplotlib.pyplot as plt
import matplotlib
import json
import numpy
import os
import draw_plots

'''Open file Setting.json which contains learning parameters. '''
with open('Setting.json') as f:
    setting = json.load(f)

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print("Current Time =", start_time)

''' aka number of rounds of routing '''
numEpisode = setting["Simulation"]["training_episodes"]
Episode = setting["Simulation"]["testing_episodes"]
Episode1 = setting["Simulation"]["testing_episodes_local_training"]
''' number of steps give the network to sent packet '''
time_steps = setting["Simulation"]["max_allowed_time_step_per_episode"]
''' mark True if want to generate graphs for stat measures while learning '''
learning_plot = setting["Simulation"]["learning_plot"]
''' mark True if want to generate graphs for stat measures for testing among various network loads '''
comparison_plots = setting["Simulation"]["test_diff_network_load_plot"]
plot_opt = setting["Simulation"]["plot_routing_network"]
''' target_update is the number of steps we wish to wait until we decide to update our network '''
TARGET_UPDATE = setting["Simulation"]["num_time_step_to_update_target_network"]

starting_size = setting["Simulation"]["test_network_load_min"]
ending_size = setting["Simulation"]["test_network_load_max"] + setting["Simulation"]["test_network_load_step_size"]
step_size = setting["Simulation"]["test_network_load_step_size"]
network_load = np.arange(starting_size,ending_size, step_size)  # np.arange(500, 5500, 500)
opt = setting["Simulation"]["whether_save"]
for i in network_load:
    if i <= 0:
        print("Error: Network load must be positive.")
        sys.exit()

env = dynetworkEnv()
env.reset(max(network_load))
print("max(network_load)",max(network_load))
agent = QAgent(env.dynetwork)
if agent.config['update_less'] == False:
    agent.config["update_models"][:, :] = True

''' check valid input configuration '''
if agent.config["sample_memory"]+agent.config["recent_memory"]+agent.config["priority_memory"] != 1:
    print("Error: Check memory type!")
    sys.exit()

'''stats measures'''
avg_deliv_learning = []
deliv_ratio_learning =[]
congestions_number_learning = []
retransmission_ratio_learning = []

past_reward = env.helper_calc_reward()
dqn0to1_reward_diff = []


f = open("experiences", "a")
model_path = setting["DQN"]["model_path"]
train_times = setting["DQN"]["train_times"]
if train_times != 0:
    print("使用之前训练过的模型")
    print("model_path:", model_path)
    print("重新训练前，清空replay_memory")
    agent.config['epsilon'] = 0.5481219996180631
    env.clean_replay_memories()
    env.load(model_path)
''' we simulate one instance of finite packet routing numEpisode times '''
for i_episode in range(numEpisode):
    print("---------- Episode:", i_episode+1, " ----------")
    step = []
    deliveries = []
    not_deliveries = 0
    false_generate = 0
    start = time.time()
    f.writelines(["Episode " + str(i_episode) + ":\n"])
    ''' iterate each time step try to finish routing within time_steps '''
    for t in range(time_steps):
        if (t+1) % 200 == 0:
            print("Time step", t + 1)
        env.updateWhole(agent,  t, learn=True, SP = False)
        if agent.config['update_less']:
            agent.config["update_models"][:, :] = True
            for destination_node in range(len(agent.config["update_models"][0, :])):
                agent.learn(env.dqn[destination_node], env.dqn,None, 0, 0, destination_node)
            agent.config["update_models"][:, :] = False
        ''' store attributes for stats measure'''
        step.append(t)
        deliveries.append(copy.deepcopy(env.dynetwork._deliveries))
        if (t+1) % TARGET_UPDATE == 0:
            env.update_target_weights()
        if (env.dynetwork._deliveries >= (env.npackets + env.dynetwork._max_initializations)):
            print("done! Finished in " + str(t + 1) + " time steps")
            break
    # if plot_opt:
    #     env.render(i_episode)
    for index in env.dynetwork._packets.packetList:
        if env.dynetwork._packets.packetList[index].get_flag() == 0:
            not_deliveries += 1
    end = time.time()
    print("Epsilon", agent.config['epsilon'])
    print("pkts delivered:", env.dynetwork._deliveries)
    print("pkts not_delivered:", not_deliveries)
    print("pkts in purgatory:", len(env.dynetwork._purgatory))
    print("congestion happened,the number of dropped packets is:", env.dynetwork._congestions[-1])
    print("the number of retransmission is",env.dynetwork._retransmission[-1] )
    print("the ratio of retransmission is:", env.dynetwork._retransmission[-1]/(env.dynetwork._deliveries+ env.dynetwork._retransmission[-1]))
    print("初始化的packets：",env.npackets)
    print("total packets:",env.npackets + env.dynetwork._initializations)
    print("delivery_ratio:", env.dynetwork._deliveries / (env.dynetwork._deliveries + env.dynetwork._congestions[-1]))
    print("avg_delivery_time:", env.calc_avg_delivery())
    # f.writelines(["delivery_ratio: " + str(env.dynetwork._deliveries / (env.dynetwork._deliveries + env.dynetwork._congestions[-1])) + "\n"])
    # f.writelines(["avg_delivery_time: " + str(env.calc_avg_delivery()) + "\n"])
    avg_deliv_learning.append(env.calc_avg_delivery())  # 计算的是所有packets的平均交付时间
    deliv_ratio_learning.append(env.dynetwork._deliveries / (env.dynetwork._deliveries + env.dynetwork._congestions[-1]))
    congestions_number_learning.append(env.dynetwork._congestions[-1])
    retransmission_ratio_learning.append(env.dynetwork._retransmission[-1]/(env.dynetwork._deliveries+ env.dynetwork._retransmission[-1]))

    reward = env.helper_calc_reward() #相当于得到的是Q估计
    diff = torch.norm(past_reward - reward) / torch.norm(past_reward)
    past_reward = reward
    dqn0to1_reward_diff.append(diff.tolist())

    env.reset(max(network_load))
    reward_slice = np.array(dqn0to1_reward_diff[-5:])
    if ((reward_slice < 0.05).sum() == 5):
        numEpisode = i_episode + 1
        break
# 画图
if learning_plot == 1:
    draw_plots.draw_learning(avg_deliv_learning,deliv_ratio_learning,congestions_number_learning,retransmission_ratio_learning)
# 保存训练好的模型参数
env.save(opt, model_path)

test_opt = setting["Simulation"]["whether_test"]
network_opt = setting["Simulation"]["network_opt_test"]
if test_opt == 1:
    print("进入测试部分")
    # 读取存的地图
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'q-learning/')
    if network_opt == 1:
        print("当前使用的network是graph1--测试SP时修改节点数目时使用")
        network = nx.read_gpickle(results_dir + "graph1.gpickle")
        env.initial_dynetwork = dynetwork.DynamicNetwork(copy.deepcopy(network), env.max_initializations)
        env.dynetwork = copy.deepcopy(env.initial_dynetwork)
    else:
        print("当前测试使用的network是graph3")
        network = nx.read_gpickle(results_dir + "graph3.gpickle")
        env.initial_dynetwork = dynetwork.DynamicNetwork(copy.deepcopy(network), env.max_initializations)
        env.dynetwork = copy.deepcopy(env.initial_dynetwork)

    # 测试部分
    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")
    print("Current_Train_Time =", start_time)
    agent.config['epsilon'] = 0.01
    agent.config['decay_rate'] = 1


    def Test(currTrial, curLoad, SP=False):
        step = []
        deliveries = []
        not_deliveries_testing = 0
        ''' iterate each time step try to finish routing within time_steps '''
        for t in range(time_steps):
            if SP:
                env.updateWhole(agent, t, learn=False, SP=True)
            else:
                env.updateWhole(agent, t, learn=False, SP=False)
            step.append(t)
            deliveries.append(copy.deepcopy(env.dynetwork._deliveries))
            if (env.dynetwork._deliveries >= (env.dynetwork._initializations + curLoad)):
                print("Finished trial,the delivery-ratio is 100%", currTrial)
                print("pkts delivered:", env.dynetwork._deliveries)
                print("total pkts:", env.npackets + env.dynetwork._initializations)
                break
        for index in env.dynetwork._packets.packetList:
            if env.dynetwork._packets.packetList[index].get_flag() == 0:
                not_deliveries_testing += 1
        print("pkts delivered:", env.dynetwork._deliveries)
        print("pkt not_deliveried:", not_deliveries_testing)
        # print("pkts in purgatory:", len(env.dynetwork._purgatory))
        print("congestion happened,the number of dropped packets is:", env.dynetwork._congestions[-1])
        # print("the number of retransmission is:", env.dynetwork._retransmission[-1])
        print("the ratio of retransmission is:", env.dynetwork._retransmission[-1]/(env.dynetwork._deliveries + env.dynetwork._retransmission[-1]))
        print("total pkts:", curLoad + env.dynetwork._initializations)
        print("delivery ratio:",
              env.dynetwork._deliveries / (env.dynetwork._deliveries + env.dynetwork._congestions[-1]))
        avg = env.calc_avg_delivery()
        print("avg_delivery_time:", avg)
        avg_deliv = env.calc_avg_delivery()
        avg_deliv_ratio=env.dynetwork._deliveries / (env.dynetwork._deliveries + env.dynetwork._congestions[-1])
        congestions_number= env.dynetwork._congestions[-1]
        retransmission_ratio= env.dynetwork._retransmission[-1]/(env.dynetwork._deliveries + env.dynetwork._retransmission[-1])
        return avg_deliv, avg_deliv_ratio, congestions_number, retransmission_ratio


    trials = setting["Simulation"]["test_trials_per_load"]
    SP_test_opt = setting["Simulation"]["SP_test_opt"]
    DQN_test_opt = setting["Simulation"]["DQN_test_opt"]
    SP_test_opt_change = setting["Simulation"]["SP_test_opt_change"]
    DQN_test_opt_change = setting["Simulation"]["DQN_test_opt_change"]
    # 平均交付时间
    all_dqn_avg_delivs = []
    all_sp_avg_delivs = []
    dqn_avg_delivs = []
    sp_avg_delivs = []
    # 交付率
    all_dqn_avg_deliv_ratios = []
    all_sp_avg_deliv_ratios = []
    dqn_avg_deliv_ratios = []
    sp_avg_deliv_ratios = []
    # 重传率
    all_dqn_retransmission_ratios = []
    all_sp_retransmission_ratios = []
    dqn_retransmission_ratios = []
    sp_retransmission_ratios = []
    # 拥塞次数
    all_dqn_congestions_numbers =[]
    all_sp_congestions_numbers = []
    dqn_congestions_numbers = []
    sp_congestions_numbers = []
    for i in range(len(network_load)):
        curLoad = network_load[i]
        # 平均交付时间
        dqn_avg_delivs.append([])
        sp_avg_delivs.append([])
        all_dqn_avg_delivs.append([])
        all_sp_avg_delivs.append([])
        # 交付率
        dqn_avg_deliv_ratios.append([])
        sp_avg_deliv_ratios.append([])
        all_dqn_avg_deliv_ratios.append([])
        all_sp_avg_deliv_ratios.append([])
        # 重传率
        dqn_retransmission_ratios.append([])
        sp_retransmission_ratios.append([])
        all_dqn_retransmission_ratios.append([])
        all_sp_retransmission_ratios.append([])
        # 拥塞次数
        dqn_congestions_numbers.append([])
        sp_congestions_numbers.append([])
        all_dqn_congestions_numbers.append([])
        all_sp_congestions_numbers.append([])
        print("---------- Testing:", curLoad, " ----------")
        for currTrial in range(trials):
            print("-----currTrial:", currTrial + 1, "-----")
            env.render(curLoad)
            # 测试节点不变的结果
            if DQN_test_opt == 1:
                env.reset(curLoad, False, False)
                # 读取保存的模型
                env.load(model_path)
                print("测试节点不变的dqn的结果")
                dqn_avg_deliv,dqn_avg_deliv_ratio, dqn_congestions_number, dqn_retransmission_ratio = Test(currTrial, curLoad, SP=False)
                dqn_avg_delivs[i].append(dqn_avg_deliv)
                dqn_avg_deliv_ratios[i].append(dqn_avg_deliv_ratio)
                dqn_retransmission_ratios[i].append(dqn_retransmission_ratio)
                dqn_congestions_numbers[i].append(dqn_congestions_number)
            if SP_test_opt == 1:
                env.reset(curLoad, False, False)
                print("测试节点不变的sp的结果")
                sp_avg_deliv,sp_avg_deliv_ratio, sp_congestions_number, sp_retransmission_ratio =Test(currTrial, curLoad, SP=True)
                sp_avg_delivs[i].append(sp_avg_deliv)
                sp_avg_deliv_ratios[i].append(sp_avg_deliv_ratio)
                sp_retransmission_ratios[i].append(sp_retransmission_ratio)
                sp_congestions_numbers[i].append(sp_congestions_number)
        # 求当前network_load的trails的平均值
        dqn_avg_deliv_time = sum(dqn_avg_delivs[i]) / len(dqn_avg_delivs[i])
        sp_avg_deliv_time = sum(sp_avg_delivs[i]) / len(sp_avg_delivs[i])
        dqn_avg_delivery_ratio = sum(dqn_avg_deliv_ratios[i]) / len(dqn_avg_deliv_ratios[i])
        sp_avg_delivery_ratio = sum(sp_avg_deliv_ratios[i]) / len(sp_avg_deliv_ratios[i])
        dqn_retrans_ratio = sum(dqn_retransmission_ratios[i]) / len(dqn_retransmission_ratios[i])
        sp_retrans_ratio = sum(sp_retransmission_ratios[i]) / len(sp_retransmission_ratios[i])
        dqn_congest_number = sum(dqn_congestions_numbers[i]) /len(dqn_congestions_numbers[i])
        sp_congest_number = sum(sp_congestions_numbers[i]) / len(sp_congestions_numbers[i])
        # 记录每个network_load对应的平均值
        all_dqn_avg_delivs[i].append(dqn_avg_deliv_time)
        all_sp_avg_delivs[i].append(sp_avg_deliv_time)
        all_dqn_avg_deliv_ratios[i].append(dqn_avg_delivery_ratio)
        all_sp_avg_deliv_ratios[i].append(sp_avg_delivery_ratio)
        all_dqn_retransmission_ratios[i].append(dqn_retrans_ratio)
        all_sp_retransmission_ratios[i].append(sp_retrans_ratio)
        all_dqn_congestions_numbers[i].append(dqn_congest_number)
        all_sp_congestions_numbers[i].append(sp_congest_number)
    # 画图
    draw_plots.draw_testing(all_dqn_avg_delivs, all_sp_avg_delivs,all_dqn_avg_deliv_ratios,all_sp_avg_deliv_ratios,all_dqn_retransmission_ratios,all_sp_retransmission_ratios,all_dqn_congestions_numbers, all_sp_congestions_numbers)


    whether_retrain_opt = setting["Simulation"]["whether_retrain"]
    # 平均交付时间
    all_dqn_avg_delivs = []
    all_sp_avg_delivs = []
    all_global_training_dqn_avg_delivs = []
    all_local_training_dqn_avg_delivs = []
    dqn_avg_delivs = []
    sp_avg_delivs = []
    global_training_dqn_avg_delivs = []
    local_training_dqn_avg_delivs = []
    # 交付率
    all_dqn_avg_deliv_ratios = []
    all_sp_avg_deliv_ratios = []
    all_global_training_dqn_avg_deliv_ratios = []
    all_local_training_dqn_avg_deliv_ratios = []
    dqn_avg_deliv_ratios = []
    sp_avg_deliv_ratios = []
    global_training_dqn_avg_deliv_ratios = []
    local_training_dqn_avg_deliv_ratios = []
    # 重传率
    all_dqn_retransmission_ratios = []
    all_sp_retransmission_ratios = []
    all_global_training_dqn_retransmission_ratios = []
    all_local_training_dqn_retransmission_ratios = []
    dqn_retransmission_ratios = []
    sp_retransmission_ratios = []
    global_training_dqn_retransmission_ratios = []
    local_training_dqn_retransmission_ratios = []
    # 拥塞次数
    all_dqn_congestions_numbers = []
    all_sp_congestions_numbers = []
    all_global_training_dqn_congestions_numbers = []
    all_local_training_dqn_congestions_numbers = []
    dqn_congestions_numbers = []
    sp_congestions_numbers = []
    global_training_dqn_congestions_numbers = []
    local_training_dqn_congestions_numbers = []
    # 训练时间
    time_global_training = []
    time_local_training = []
    avg_time_global_training = []
    avg_time_local_training = []
    for i in range(len(network_load)):
        curLoad = network_load[i]
        # 平均交付时间
        dqn_avg_delivs.append([])
        sp_avg_delivs.append([])
        global_training_dqn_avg_delivs.append([])
        local_training_dqn_avg_delivs.append([])
        all_dqn_avg_delivs.append([])
        all_sp_avg_delivs.append([])
        all_global_training_dqn_avg_delivs.append([])
        all_local_training_dqn_avg_delivs.append([])
        # 交付率
        dqn_avg_deliv_ratios.append([])
        sp_avg_deliv_ratios.append([])
        global_training_dqn_avg_deliv_ratios.append([])
        local_training_dqn_avg_deliv_ratios.append([])
        all_dqn_avg_deliv_ratios.append([])
        all_sp_avg_deliv_ratios.append([])
        all_global_training_dqn_avg_deliv_ratios.append([])
        all_local_training_dqn_avg_deliv_ratios.append([])
        # 重传率
        dqn_retransmission_ratios.append([])
        sp_retransmission_ratios.append([])
        global_training_dqn_retransmission_ratios.append([])
        local_training_dqn_retransmission_ratios.append([])
        all_dqn_retransmission_ratios.append([])
        all_sp_retransmission_ratios.append([])
        all_global_training_dqn_retransmission_ratios.append([])
        all_local_training_dqn_retransmission_ratios.append([])
        # 拥塞次数
        dqn_congestions_numbers.append([])
        sp_congestions_numbers.append([])
        global_training_dqn_congestions_numbers.append([])
        local_training_dqn_congestions_numbers.append([])
        all_dqn_congestions_numbers.append([])
        all_sp_congestions_numbers.append([])
        all_global_training_dqn_congestions_numbers.append([])
        all_local_training_dqn_congestions_numbers.append([])
        # 训练时间
        time_global_training.append([])
        time_local_training.append([])
        avg_time_global_training.append([])
        avg_time_local_training.append([])
        print("---------- Testing:", curLoad, " ----------")
        for currTrial in range(trials):
            print("-----currTrial:", currTrial + 1, "-----")
            # 节点发生变化
            env.change_network()
            env.draw(curLoad,0)
            # 保存已经发生变化的网络
            # 测试节点发生变化后，未重新训练的dqn结果
            if DQN_test_opt_change == 1:
                print("测试发生变化后，未重新训练的结果")
                env.reset(curLoad, True, False)
                env.load(model_path)
                dqn_avg_deliv,dqn_avg_deliv_ratio, dqn_congestions_number, dqn_retransmission_ratio = Test(currTrial, curLoad, SP=False)
                dqn_avg_delivs[i].append(dqn_avg_deliv)
                dqn_avg_deliv_ratios[i].append(dqn_avg_deliv_ratio)
                dqn_retransmission_ratios[i].append(dqn_retransmission_ratio)
                dqn_congestions_numbers[i].append(dqn_congestions_number)
            if SP_test_opt_change == 1:
                print("测试发生变化后，最短路径的结果")
                env.reset(curLoad, True, False)
                sp_avg_deliv,sp_avg_deliv_ratio, sp_congestions_number, sp_retransmission_ratio = Test(currTrial, curLoad, SP=True)
                sp_avg_delivs[i].append(sp_avg_deliv)
                sp_avg_deliv_ratios[i].append(sp_avg_deliv_ratio)
                sp_retransmission_ratios[i].append(sp_retransmission_ratio)
                sp_congestions_numbers[i].append(sp_congestions_number)
            # 节点发生变化后，重新训练并测试
            if whether_retrain_opt == 1:
                print("全部重新训练")
                now = datetime.now()
                start_time = now.strftime("%H:%M:%S")
                print("Current Time =", start_time)
                env.load(model_path)
                env.reset(curLoad, True, False)
                for num in range(Episode):
                    for t in range(time_steps):
                        if (t + 1) % 200 == 0:
                            print("Time step", t + 1)
                        env.updateWhole(agent, t, learn=True, SP=False)
                        if agent.config['update_less']:
                            agent.config["update_models"][:, :] = True
                        for destination_node in range(len(agent.config["update_models"][0, :])):
                            agent.learn(env.dqn[destination_node], env.dqn, None, 0, 0, destination_node)
                        agent.config["update_models"][:, :] = False
                        if (t + 1) % TARGET_UPDATE == 0:
                            env.update_target_weights()
                    env.reset(curLoad, True, False)
                end = datetime.now()
                end_time = end.strftime("%H:%M:%S")
                print("End Time =", end_time)
                print("全局训练结束")
                time_global = (end - now).seconds
                time_global_training[i].append(time_global)
                print("本次全局训练的时间为：", time_global)
                env.reset(curLoad, True, False)
                print("重新测试变化后的DQN--1")
                # if DQN_test_opt_change == 1:
                dqn_avg_deliv,dqn_avg_deliv_ratio, dqn_congestions_number, dqn_retransmission_ratio =Test(currTrial, curLoad, SP=False)
                global_training_dqn_avg_delivs[i].append(dqn_avg_deliv)
                global_training_dqn_avg_deliv_ratios[i].append(dqn_avg_deliv_ratio)
                global_training_dqn_retransmission_ratios[i].append(dqn_retransmission_ratio)
                global_training_dqn_congestions_numbers[i].append(dqn_congestions_number)
            if whether_retrain_opt == 1:
                print("局部重新训练")
                now = datetime.now()
                start_time = now.strftime("%H:%M:%S")
                print("Current Time =", start_time)
                env.load(model_path)
                env.reset(curLoad, True, False)
                for num in range(Episode1):
                    for t in range(time_steps):
                        if (t + 1) % 200 == 0:
                            print("Time step", t + 1)
                        env.purgatory(False)
                        env.update_queues(False)
                        env.update_time(False)
                        env.router_test(agent, True)
                        if agent.config['update_less']:
                            agent.config["update_models"][:, :] = True
                        for destination_node in range(len(agent.config["update_models"][0, :])):
                            agent.learn(env.dqn[destination_node], env.dqn, None, 0, 0, destination_node)
                        agent.config["update_models"][:, :] = False
                        if (t + 1) % TARGET_UPDATE == 0:
                            env.update_target_weights()
                    env.reset(curLoad, True, False)
                end = datetime.now()
                end_time = end.strftime("%H:%M:%S")
                print("End Time =", end_time)
                print("局部训练结束，开始重新测试")
                time_local = (end - now).seconds
                time_local_training[i].append(time_local)
                print("本次局部训练的时间为：", time_local)
                env.reset(curLoad, True, False)
                dqn_avg_deliv,dqn_avg_deliv_ratio, dqn_congestions_number, dqn_retransmission_ratio =Test(currTrial, curLoad, SP=False)
                local_training_dqn_avg_delivs[i].append(dqn_avg_deliv)
                local_training_dqn_avg_deliv_ratios[i].append(dqn_avg_deliv_ratio)
                local_training_dqn_retransmission_ratios[i].append(dqn_retransmission_ratio)
                local_training_dqn_congestions_numbers[i].append(dqn_congestions_number)
        # 求当前network_load的trails的平均值
        dqn_avg_deliv_time = sum(dqn_avg_delivs[i]) / len(dqn_avg_delivs[i])
        sp_avg_deliv_time = sum(sp_avg_delivs[i]) / len(sp_avg_delivs[i])
        global_avg_deliv_time = sum(global_training_dqn_avg_delivs[i])/len(global_training_dqn_avg_delivs[i])
        local_avg_deliv_time = sum(local_training_dqn_avg_delivs[i]) / len(local_training_dqn_avg_delivs[i])

        dqn_avg_delivery_ratio = sum(dqn_avg_deliv_ratios[i]) / len(dqn_avg_deliv_ratios[i])
        sp_avg_delivery_ratio = sum(sp_avg_deliv_ratios[i]) / len(sp_avg_deliv_ratios[i])
        global_avg_delivery_ratio = sum(global_training_dqn_avg_deliv_ratios[i]) / len(global_training_dqn_avg_deliv_ratios[i])
        local_avg_delivery_ratio = sum(local_training_dqn_avg_deliv_ratios[i]) / len(local_training_dqn_avg_deliv_ratios[i])

        dqn_retrans_ratio = sum(dqn_retransmission_ratios[i]) / len(dqn_retransmission_ratios[i])
        sp_retrans_ratio = sum(sp_retransmission_ratios[i]) / len(sp_retransmission_ratios[i])
        global_retrans_ratio = sum(global_training_dqn_retransmission_ratios[i]) / len(global_training_dqn_retransmission_ratios[i])
        local_retrans_ratio = sum(local_training_dqn_retransmission_ratios[i]) / len(local_training_dqn_retransmission_ratios[i])

        dqn_congest_number = sum(dqn_congestions_numbers[i]) / len(dqn_congestions_numbers[i])
        sp_congest_number = sum(sp_congestions_numbers[i]) / len(sp_congestions_numbers[i])
        global_congest_num = sum(global_training_dqn_congestions_numbers[i])/len(global_training_dqn_congestions_numbers[i])
        local_congest_num = sum(local_training_dqn_congestions_numbers[i]) / len(local_training_dqn_congestions_numbers[i])

        avg_global_time = sum(time_global_training[i]) / len(time_global_training[i])
        avg_local_time = sum(time_local_training[i]) / len(time_local_training[i])
        # 记录每个network_load对应的平均值
        all_dqn_avg_delivs[i].append(dqn_avg_deliv_time)
        all_sp_avg_delivs[i].append(sp_avg_deliv_time)
        all_global_training_dqn_avg_delivs[i].append(global_avg_deliv_time)
        all_local_training_dqn_avg_delivs[i].append(local_avg_deliv_time)

        all_dqn_avg_deliv_ratios[i].append(dqn_avg_delivery_ratio)
        all_sp_avg_deliv_ratios[i].append(sp_avg_delivery_ratio)
        all_global_training_dqn_avg_deliv_ratios[i].append(global_avg_delivery_ratio)
        all_local_training_dqn_avg_deliv_ratios[i].append(local_avg_delivery_ratio)

        all_dqn_retransmission_ratios[i].append(dqn_retrans_ratio)
        all_sp_retransmission_ratios[i].append(sp_retrans_ratio)
        all_global_training_dqn_retransmission_ratios[i].append(global_retrans_ratio)
        all_local_training_dqn_retransmission_ratios[i].append(local_retrans_ratio)

        all_dqn_congestions_numbers[i].append(dqn_congest_number)
        all_sp_congestions_numbers[i].append(sp_congest_number)
        all_global_training_dqn_congestions_numbers[i].append(global_congest_num)
        all_local_training_dqn_congestions_numbers[i].append(local_congest_num)

        avg_time_global_training[i].append(avg_global_time)
        avg_time_local_training[i].append(avg_local_time)
    draw_plots.testing_changed_network_avg_deliv_time(all_dqn_avg_delivs, all_sp_avg_delivs, all_global_training_dqn_avg_delivs,all_local_training_dqn_avg_delivs)
    draw_plots.testing_changed_network_avg_deliv_ratio(all_dqn_avg_deliv_ratios,all_sp_avg_deliv_ratios,all_global_training_dqn_avg_deliv_ratios,all_local_training_dqn_avg_deliv_ratios)
    draw_plots.testing_changed_network_retransmission_ratios(all_dqn_retransmission_ratios,all_sp_retransmission_ratios,all_global_training_dqn_retransmission_ratios,all_local_training_dqn_retransmission_ratios)
    draw_plots.testing_changed_network_congestions(all_dqn_congestions_numbers,all_sp_congestions_numbers,all_global_training_dqn_congestions_numbers,all_local_training_dqn_congestions_numbers)
    draw_plots.draw_time(avg_time_global_training, avg_time_local_training)
print("start Time =", start_time)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Whole End Time =", current_time)

# main_dir = os.path.dirname(os.path.realpath(__file__))
# np.save(os.path.join(main_dir, "dqn_avg_deliv"), avg_deliv)
# np.save(os.path.join(main_dir, "dqn_avg_congestion"), congestions_number)
