import datetime
import random
import time
from collections import namedtuple
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

import environment
from dye_agents import SD_Agent, A1_Agent
from environment import seconds_to_hour
from gantt import Gantt
from parameters import Parameters
from pg_re import discount
from ppo_lstm import PPO_LSTM
import copy

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'c_state', 'h_state'])
zero_time = datetime.timedelta(hours=0)


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def plot_lr_curve(output_file_prefix, mean_rew_lr_curve, max_rew_lr_curve, min_rew_lr_curve, batch_mean_rew_lr_curve,
                  schedule_mean_rew_lr_curve,
                  mean_dly_lr_curve, max_dly_lr_curve, min_dly_lr_curve, episode_time_curve,
                  unfinished_job_curve, decision_count_curve, ref_eptime, ref_dly, ref_unfinished_job_num,
                  ref_decision_count, c_loss, critic_value):
    colors = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Greys']
    color_dict = {}

    i = 0
    for RL_type in mean_rew_lr_curve.keys():
        cm = plt.get_cmap(colors[i])
        color_dict[RL_type] = [cm(40), cm(130)]
        i += 1
    for test_type in ref_dly.keys():
        cm = plt.get_cmap(colors[i])
        color_dict[test_type] = cm(130)
        i += 1

    fig = plt.figure(figsize=(25, 22))

    ax = fig.add_subplot(331)
    for RL_type in min_rew_lr_curve.keys():
        plt.fill_between(range(len(min_rew_lr_curve[RL_type])), min_rew_lr_curve[RL_type], max_rew_lr_curve[RL_type],
                         color=color_dict[RL_type][1], alpha=0.25)
        ax.plot(mean_rew_lr_curve[RL_type], linewidth=2, label=RL_type, color=color_dict[RL_type][1])
        plt.legend(loc=7)
        plt.xlabel("Iteration", fontsize=20)
        plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(332)
    for RL_type in min_rew_lr_curve.keys():
        ax.plot(batch_mean_rew_lr_curve[RL_type], linewidth=2, label=RL_type, color=color_dict[RL_type][1])
        plt.legend(loc=7)
        plt.xlabel("Iteration", fontsize=20)
        plt.ylabel("Discounted Total Reward of Batch Agent", fontsize=20)

    ax = fig.add_subplot(333)
    for RL_type in min_rew_lr_curve.keys():
        ax.plot(schedule_mean_rew_lr_curve[RL_type], linewidth=2, label=RL_type, color=color_dict[RL_type][1])
        plt.legend(loc=7)
        plt.xlabel("Iteration", fontsize=20)
        plt.ylabel("Discounted Total Reward of Schedule Agent", fontsize=20)

    ax = fig.add_subplot(334)
    for RL_type in min_rew_lr_curve.keys():
        plt.fill_between(range(len(max_dly_lr_curve[RL_type])), min_dly_lr_curve[RL_type], max_dly_lr_curve[RL_type],
                         color=color_dict[RL_type][1], alpha=0.25)
        ax.plot(mean_dly_lr_curve[RL_type], linewidth=2, label=RL_type, color=color_dict[RL_type][1])

    for k in ref_dly:
        ax.plot(np.tile(np.sum(ref_dly[k]), len(mean_dly_lr_curve[RL_type])), linewidth=1, label=k,
                color=color_dict[k])
    plt.legend(loc=7)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Sum of Delay", fontsize=20)

    ax = fig.add_subplot(335)
    for RL_type in min_rew_lr_curve.keys():
        ax.plot(episode_time_curve[RL_type], linewidth=2, label=RL_type, color=color_dict[RL_type][1])

    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Episode Days", fontsize=20)

    ax = fig.add_subplot(336)
    for RL_type in min_rew_lr_curve.keys():
        ax.plot(unfinished_job_curve[RL_type], linewidth=2, label=RL_type, color=color_dict[RL_type][1])

    for k in ref_unfinished_job_num:
        ax.plot(np.tile(np.sum(ref_unfinished_job_num[k]), len(unfinished_job_curve[RL_type])), linewidth=2, label=k,
                color=color_dict[k])
    plt.legend()
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Total Num Of Unfinished Job ", fontsize=20)

    ax = fig.add_subplot(337)
    for RL_type in min_rew_lr_curve.keys():
        ax.plot(decision_count_curve[RL_type], linewidth=2, label=RL_type, color=color_dict[RL_type][1])

    plt.legend()
    plt.xticks()
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Total Decision Time", fontsize=20)

    ax = fig.add_subplot(338)
    RL_type = 'MA-RPPO'
    ax.plot(movingaverage(c_loss[RL_type], 500), linewidth=2, label=RL_type, color=color_dict[RL_type][1])
    xticks = np.linspace(0, len(c_loss[RL_type]), len(decision_count_curve[RL_type]))
    plt.xticks(xticks, range(len(decision_count_curve[RL_type])))
    plt.legend()
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Critic Loss", fontsize=20)

    ax = fig.add_subplot(339)
    RL_type = 'MA-RPPO'
    ax.plot(movingaverage(critic_value[RL_type], 500), linewidth=2, label=RL_type, color=color_dict[RL_type][1])
    xticks = np.linspace(0, len(critic_value[RL_type]), len(decision_count_curve[RL_type]))
    plt.xticks(xticks, range(len(decision_count_curve[RL_type])))
    plt.legend()
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Critic Value", fontsize=20)

    filename = 'dye_data/lr_curve/' + output_file_prefix + "_lr_curve" + ".pdf"
    try:
        plt.savefig(filename)
    except:
        print('写入文件异常，可能被占用！')
        pass


def get_traj(batch_agent, schedule_agent, env, pa, agent):
    """
    Run agent-environment lAttributeError: posoop for one whole episode (trajectory)
    Return dictionary of results
    """
    obs, next_obs, acts, rews, info, action_probs, h_states, c_states, next_h_states, next_c_states = [], [], [], [], [], [], [], [], [], []
    agent_record = []
    ep_r = 0
    h_state, c_state = torch.zeros(1, 1, pa.lstm_hidden_size), torch.zeros(1, 1, pa.lstm_hidden_size)
    state = env.reset()
    old_state = state = np.append(state, 1)
    action = pa.num_slot
    job_num = 0

    env.decision_time_record = []

    while True:
        env.decision_time_record.append(env.curr_time)

        env.refresh_job()
        print('当前待组批任务：',len(env.waiting_job), '当前批占用情况：',[int(batch.dyeKG) for batch in env.batches],end=' || ')
        remain_time = [int(machine.job_remain_time.total_seconds() / 3600) for machine in env.machine]
        stes = ['空闲' if machine.is_idle else '忙碌' for machine in env.machine]
        print('设备剩余加工时间：',['#'+str(i)+'('+str(env.machine[i].max_capacity)+'KG): '+ stes[i] + '剩余' + str(remain_time[i]) +'h' for i in range(env.pa.num_machine)])

        for job in env.waiting_job:

            s_a = np.append(old_state, action)
            s_a = torch.from_numpy(np.reshape(s_a, [1, 1, pa.lstm_input_dim])).to(torch.float32)

            action, action_prob, next_h_state, next_c_state = batch_agent.select_action(state, s_a, h_state, c_state)
            next_h_state, next_c_state = next_h_state.data, next_c_state.data
            if env.curr_time > env.time_horizon:
                old_state = state
            # print(action)
            next_state, reward, done, batch_info = env.batch_step(action, job)
            obs.append(state)
            next_obs.append(next_state)
            acts.append(action)
            rews.append(reward)
            action_probs.append(action_prob)
            h_states.append(h_state)
            c_states.append(c_state)
            next_h_states.append(next_h_state)
            next_c_states.append(next_c_state)
            agent_record.append(0)
            state, h_state, c_state = next_state, next_h_state, next_c_state
            # print(state)
            env.decision_count += 1

        while True:
            s_a = np.append(old_state, action)
            s_a = torch.from_numpy(np.reshape(s_a, [1, 1, pa.lstm_input_dim])).to(torch.float32)

            action, action_prob, next_h_state, next_c_state = schedule_agent.select_action(state, s_a, h_state, c_state)
            next_h_state, next_c_state = next_h_state.data, next_c_state.data
            if env.curr_time > env.time_horizon:
                old_state = state
            next_state, reward, done, schedule_info, allocated = env.schedule_step(action)
            # print(action)
            obs.append(state)
            next_obs.append(next_state)
            acts.append(action)
            rews.append(reward)
            action_probs.append(action_prob)
            h_states.append(h_state)
            c_states.append(c_state)
            next_h_states.append(next_h_state)
            next_c_states.append(next_c_state)
            agent_record.append(1)
            state, h_state, c_state = next_state, next_h_state, next_c_state
            # print(state)
            env.decision_count += 1
            if not allocated:
                break

        if done:
            num_time_step = env.curr_time - env.time_horizon
            episode_days = (env.curr_time - env.time_horizon).days
            job_record = env.job_record
            batch_record = env.batch_record
            decision_count = env.decision_count
            decision_time_record = env.decision_time_record
            env.reset()
            break
        # 时间更新
        env.refresh_time()

        # 刷新设备时间，释放完工的设备
        for machine in env.machine:
            finished_batch = machine.time_proceed(env.curr_time)

            if finished_batch is not None:
                for DyeJob in finished_batch.job_batch:
                    if DyeJob.finish_time > DyeJob.RequestTime:
                        delay_start = max(env.curr_time - env.curr_time_window, DyeJob.RequestTime)
                        delay_end = env.curr_time
                        delta_dly = seconds_to_hour((delay_end - delay_start).total_seconds())
                        current_dly = seconds_to_hour((env.curr_time - DyeJob.RequestTime).total_seconds())
                        assert abs(delta_dly - (current_dly - DyeJob.dly)) < 0.000001
                        env.s_reward += -delta_dly
                        DyeJob.dly = current_dly
                    max_delay_days = env.pa.episode_days.days
                    env.s_reward += 24 * max_delay_days

    traj = {'rews': np.asarray(rews),
            'obs': np.asarray(obs),
            'acts': np.asarray(acts),
            'action_probs': np.asarray(action_probs),
            'next_obs': np.asarray(obs),
            'h_states': h_states,
            'c_states': c_states,
            'time': num_time_step,
            'agent_record': np.asarray(agent_record)
            }

    info = {'time': episode_days,
            'job_info': job_record,
            'batch_info': batch_record,
            'decision_time_record': decision_time_record,
            'decision_count': decision_count
            }

    return traj, info


def process_all_info(infos):
    enter_time = []
    finish_time = []
    start_time = []
    total_dly = []
    decision_count = []

    for info in infos:
        enter_time.append(
            np.array([info['job_info'].job_record[i].BillDate for i in range(len(info['job_info'].job_record))]))

        start_time.append(
            np.array([info['job_info'].job_record[i].start_time for i in range(len(info['job_info'].job_record))]))

        finish_time.append(
            np.array([info['job_info'].job_record[i].finish_time for i in range(len(info['job_info'].job_record))]))

        total_dly.append(
            np.sum([info['job_info'].job_record[i].dly for i in range(len(info['job_info'].job_record))]))

        decision_count.append(info['decision_count'])

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    start_time = np.concatenate(start_time)
    decision_count = np.asarray(decision_count)

    return enter_time, finish_time, start_time, total_dly, decision_count


def get_traj_worker(batch_agent, schedule_agent, env, pa, agent):

    trajs, infos = [], []

    for i in range(pa.num_seq_per_batch):
        traj, info = get_traj(batch_agent, schedule_agent, env, pa, agent)
        trajs.append(traj)
        infos.append(info)

    if batch_agent.name == 'PPO' and schedule_agent.name == 'PPO':
        # 两个agent分别折扣reward
        batch_rets = [discount(traj["rews"][np.where(traj["agent_record"] == 0)], pa.discount) for traj in trajs]
        schedule_rets = [discount(traj["rews"][np.where(traj["agent_record"] == 1)], pa.discount) for traj in trajs]

        tmp = [[] for i in range(len(trajs))]
        for i in range(len(trajs)):
            traj = trajs[i]
            agent_record = traj["agent_record"]
            bi, si = 0, 0
            for j in range(len(agent_record)):
                if agent_record[j] == 0:
                    tmp[i].append(batch_rets[i][bi])
                    bi+=1
                if agent_record[j] == 1:
                    tmp[i].append(schedule_rets[i][si])
                    si+=1
        rets = tmp

    else:
        # 两个agent整体折扣reward
        rets = [discount(traj["rews"], pa.discount) for traj in trajs]

    # Compute time-dependent baseline
    maxlen = max(len(ret) for ret in rets)
    padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]
    baseline = np.mean(padded_rets, axis=0)

    # Compute advantage function
    advs = [ret - baseline[:len(ret)] for ret in rets]  # 用于放到神经网络进行计算的reward

    # 提取有用信息

    all_eprews = np.array(
        [discount(traj["rews"], pa.discount)[0] for traj in trajs])  # episode total rewards
    all_batch_eprews = np.array(
        [np.mean(discount(traj["rews"], pa.discount)[traj['agent_record'] == 0]) for traj in trajs])
    all_schedule_eprews = np.array(
        [np.mean(discount(traj["rews"], pa.discount)[traj['agent_record'] == 1]) for traj in trajs])
    all_eptime = np.array([info["time"] for info in infos])  # episode lengths
    enter_time, finish_time, start_time, all_total_delay, all_decision_count = process_all_info(infos)
    finished_idx = (finish_time != -1)
    unfinished_idx = (1 - finished_idx).astype(np.bool)
    unfinished_job_num = np.sum(unfinished_idx)

    # print('检查奖励值与目标值')
    # print('总拖期', all_total_delay)
    # sum_rewards = [sum(traj["rews"]) for traj in trajs]
    # print('总奖励值', sum_rewards)

    # 存储经验
    agents = [batch_agent, schedule_agent]
    # print('获取经验数量： ', sum([len(traj["obs"]) for traj in trajs]))
    for i, traj in enumerate(trajs):
        for j in range(len(traj["obs"])):
            trans = Transition(traj["obs"][j], traj["acts"][j], traj["action_probs"][j], advs[i][j],
                               traj["next_obs"][j], traj["c_states"][j], traj["h_states"][j])
            agents[traj["agent_record"][j]].store_transition(trans)
    best_index = all_total_delay.index(min(all_total_delay))

    return all_eprews, all_batch_eprews, all_schedule_eprews, all_eptime, unfinished_job_num, all_total_delay, all_decision_count, \
           infos[best_index]


def launch(pa, pg_resume=None, render=False, repre='parameters', end='no_new_job'):
    job_dicts, machine_dict = init_job_machine(pa)
    envs = []
    for ex in range(pa.num_ex):
        env = environment.DyeEnv(pa, ex, job_dicts[ex], machine_dict)
        env.seq_no = ex
        envs.append(env)

    assert env.time_horizon + pa.episode_days > max([DyeJob.BillDate for DyeJob in env.job])

    # 获取对比参数
    test_types = ['SD', 'A1']

    ref_eptime, ref_dly, ref_unfinished_job_num, ref_decision_count, ref_infos = get_ref_trajs(test_types, envs, pa)
    plot_gantt(pa, ref_infos, envs, ref_dly)
    for test_type in test_types:
        print('---------%s----------' % test_type)
        print("%s用时：%s天" % (test_type, ref_eptime[test_type]))
        print("%s拖期：%s小时" % (test_type, ref_dly[test_type]))
        print("%s决策次数：%s" % (test_type, ref_decision_count[test_type]))

    # 建立智能体
    schedule_agent = PPO_LSTM.PPO_LSTM(pa)
    schedule_agent.name = 'schedule'

    # copy一个pa给batch agent
    batch_agent = PPO_LSTM.PPO_LSTM(pa)
    batch_agent.name = 'batch'
    batch_agent.critic_net = schedule_agent.critic_net
    batch_agent.lstm_net = schedule_agent.lstm_net
    batch_agent.critic_net.Lstm = batch_agent.lstm_net
    batch_agent.actor_net.Lstm = batch_agent.lstm_net
    batch_agent.actor_optimizer.lr = pa.actor_lr_rate
    batch_agent.critic_net_optimizer.lr = pa.critic_lr_rate

    pa.lstm_output_dim = 1
    pa.ppo_network_input_dim = pa.network_input_dim + pa.lstm_output_dim
    single_batch_agent = PPO_LSTM.PPO_LSTM(pa)
    single_batch_agent.name = 'PPO'
    single_schedule_agent = PPO_LSTM.PPO_LSTM(pa)
    single_schedule_agent.name = 'PPO'

    timer_start = time.time()
    agent = True

    RL_types = ['MA-RPPO','PPO']
    RL_batch_agents = {'MA-RPPO': batch_agent, 'PPO': single_batch_agent}
    RL_schedule_agents = {'MA-RPPO': schedule_agent, 'PPO': single_schedule_agent}

    unfinished_job_num_curve = {}
    episode_time_curve = {}
    mean_rew_lr_curve = {}
    max_rew_lr_curve = {}
    min_rew_lr_curve = {}
    mean_batch_rew_lr_curve = {}
    mean_schedule_rew_lr_curve = {}
    decision_count_curve = {}
    mean_dly_lr_curve = {}
    max_dly_lr_curve = {}
    min_dly_lr_curve = {}
    critic_loss_curve = {}
    critic_value_curve = {}

    for RL_type in RL_types:
        unfinished_job_num_curve[RL_type] = []
        episode_time_curve[RL_type] = []
        mean_rew_lr_curve[RL_type] = []
        max_rew_lr_curve[RL_type] = []
        min_rew_lr_curve[RL_type] = []
        mean_batch_rew_lr_curve[RL_type] = []
        mean_schedule_rew_lr_curve[RL_type] = []
        decision_count_curve[RL_type] = []
        mean_dly_lr_curve[RL_type] = []
        max_dly_lr_curve[RL_type] = []
        min_dly_lr_curve[RL_type] = []
        critic_loss_curve[RL_type] = []
        critic_value_curve[RL_type] = []

    for iteration in count():
        for RL_type in RL_types:
            schedule_agent = RL_batch_agents[RL_type]
            batch_agent = RL_schedule_agents[RL_type]

            max_rew_lr_list = []
            min_rew_lr_list = []
            mean_rew_lr_list = []
            mean_batch_rew_lr_list = []
            unfinished_job_num_list = []
            episode_time_list = []
            mean_schedule_rew_lr_list = []
            mean_dly_lr_list = []
            min_dly_lr_list = []
            max_dly_lr_list = []
            decision_count_list = []
            infos = []

            for ex in range(pa.num_ex):
                all_eprews, all_batch_eprews, all_schedule_eprews, all_eptime, unfinished_job_num, all_total_dly, all_decision_count, best_info = get_traj_worker(
                    batch_agent, schedule_agent, envs[ex], pa, agent)

                min_rew_lr_list.append(np.min(all_eprews))
                mean_rew_lr_list.append(np.mean(all_eprews))
                max_rew_lr_list.append(np.max(all_eprews))
                mean_batch_rew_lr_list.append(np.mean(all_batch_eprews))
                unfinished_job_num_list.append(np.sum(unfinished_job_num))
                episode_time_list.append(np.mean(all_eptime))
                mean_schedule_rew_lr_list.append(np.mean(all_schedule_eprews))
                max_dly_lr_list.append(np.max(all_total_dly))
                mean_dly_lr_list.append(np.mean(all_total_dly))
                min_dly_lr_list.append(np.min(all_total_dly))
                decision_count_list.append(np.mean(all_decision_count))
                infos.append(best_info)

            schedule_agent.update(iteration)
            batch_agent.update(iteration)

            # ppo_lstm.writer.add_scalar('liveTime/livestep', t, global_step=iteration)

            timer_end = time.time()

            print("-----------------")
            print("Iteration: \t %i" % iteration)
            print("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
            print("Delay: \t %s Days" % (np.sum(min_dly_lr_list)/24))
            print("Batch rewards: \t %s" % (np.sum(mean_batch_rew_lr_list)))
            print("Schedule rewards: \t %s" % (np.sum(mean_schedule_rew_lr_list)))
            print("Sum of episode days: \t %s" % (np.sum(episode_time_list)), 'days')
            print("Mean decision times: \t %s" % (np.mean(decision_count_list)), 'times')
            print("Unfinish Job Num\t %s" % np.sum(unfinished_job_num_list))
            print("-----------------")

            timer_start = time.time()

            unfinished_job_num_curve[RL_type].append(np.sum(unfinished_job_num_list))
            episode_time_curve[RL_type].append(np.sum(episode_time_list))
            mean_rew_lr_curve[RL_type].append(np.sum(mean_rew_lr_list))
            max_rew_lr_curve[RL_type].append(np.sum(max_rew_lr_list))
            min_rew_lr_curve[RL_type].append(np.sum(min_rew_lr_list))
            mean_batch_rew_lr_curve[RL_type].append(np.sum(mean_batch_rew_lr_list))
            mean_schedule_rew_lr_curve[RL_type].append(np.sum(mean_schedule_rew_lr_list))
            mean_dly_lr_curve[RL_type].append(np.sum(mean_dly_lr_list))
            max_dly_lr_curve[RL_type].append(np.sum(max_dly_lr_list))
            min_dly_lr_curve[RL_type].append(np.sum(min_dly_lr_list))
            decision_count_curve[RL_type].append(np.sum(decision_count_list))
            critic_loss_curve[RL_type] = batch_agent.critic_net.loss
            critic_value_curve[RL_type] = batch_agent.critic_net.critic_value

            for ex in range(pa.num_ex):
                if all(ref_dly[test_type][ex] > min_dly_lr_list[ex] for test_type in test_types) and unfinished_job_num_list == 0:
                    info = {ex: infos[ex]}
                    plot_gantt(pa, {'MA-RPPO': info}, envs, {'MA-RPPO': min_dly_lr_list})

        if iteration > 0 and iteration % pa.output_freq == 0:
            plot_lr_curve(pa.output_filename, mean_rew_lr_curve, max_rew_lr_curve, min_rew_lr_curve,
                          mean_batch_rew_lr_curve, mean_schedule_rew_lr_curve, mean_dly_lr_curve, max_dly_lr_curve,
                          min_dly_lr_curve, episode_time_curve,
                          unfinished_job_num_curve, decision_count_curve, ref_eptime, ref_dly, ref_unfinished_job_num,
                          ref_decision_count, critic_loss_curve, critic_value_curve)

        plt.close('all')


def to_dict(test_type, max_length, dict, figure_num, time_horizon, score):
    p = []
    xticks = []
    temp = {}
    sampledict = {}

    for i in range(len(dict)):
        job = dict[i]
        if job.finish_time != -1:
            temp['label'] = job.process_mach_num
            temp["part_num"] = job.id
            temp["start"] = (job.start_time - time_horizon).total_seconds() / 3600.0 / 24
            temp["end"] = (job.finish_time - time_horizon).total_seconds() / 3600.0 / 24
            # temp["milestones"] = [job.enter_time]
            temp["delay"] = 0
            # xticks.append(int(self.start_point[i] / 10))
            #  xticks.append(int(self.end_point[i] / 10))
            p.append(temp.copy())

    # xticks = [x*100 for x in range(0,end_point // 100)]
    sampledict["packages"] = p
    sampledict['max_length'] = max_length
    sampledict["title"] = "Gantt Chart: " + str(test_type) + '_' + str(int(score))
    sampledict["xlabel"] = "time"
    sampledict["ylabel"] = "Machine No"
    sampledict["figure_num"] = figure_num
    sampledict["xticks"] = delta = list(np.arange(0, max_length + 5, 5))
    sampledict["xticks_label"] = [(time_horizon + datetime.timedelta(days=int(i))).strftime("%Y-%m-%d") for i in delta]
    #  self.sampledict["xticks"] = xticks
    # out_js = json.dumps(self.sampledict)
    return sampledict


def plot_gantt(pa, all_infos, envs, scores):
    Gant = Gantt()
    for test_type, infos in all_infos.items():
        for ex, info in infos.items():
            dict = to_dict(test_type, pa.episode_days.days, info['batch_info'].batch_record, 0, envs[ex].time_horizon,
                           scores[test_type][ex])
            Gant.initialize(test_type, ex, dict=dict)
            Gant.render()
            # Gant.show()
            Gant.save(pa.output_filename)

            save_date = False
            if save_date:
                df = pd.DataFrame(
                    columns=('id', 'process_time', 'weight', 'enter_time', 'start_time', 'finish_time'))
                for i in info.job_record:
                    job = info.job_record[i]
                    df = df.append(
                        pd.DataFrame({'id': [job.id], 'process_time': [job.len], 'weight': [job.weight],
                                      'enter_time': [job.enter_time], 'start_time': [job.start_time],
                                      'finish_time': [job.finish_time]}), ignore_index=True)
                    df.to_csv('gantt_info_ex' + str(ex) + '.csv', index=False)


def str_to_num(series_):
    exist_str = []
    for idx, value in series_.items():
        if value in exist_str:
            series_[idx] = exist_str.index(series_[idx])
        else:
            series_[idx] = len(exist_str)
            exist_str.append(value)
    return series_


def init_job_machine(pa):
    np.random.seed(314159)
    random.seed(314159)
    path = 'D:\OneDrive - mail.dhu.edu.cn\Coding\PycharmProjects\DyeScheduleMARPPO\DyeDataProcess\染缸调度数据'
    job_dic_file = pd.read_json(path + '\job_file.json')
    machine_dic = pd.read_json(path + '\machine_file.json')

    # job_dic_file = job_dic_file.drop_duplicates(keep='first', inplace=False)  # 去除重复
    job_dic_file = job_dic_file.dropna(axis=0,
                                       subset=['dyeKG', 'ColorClass', 'ColorID', 'ColorCode', 'BillDate', 'redye',
                                               'dyeCnt', 'dyelot', 'standardtime', 'RequestTime', 'batch'])
    for index, row in job_dic_file.iterrows():
        cls = job_dic_file.loc[index, 'ColorCode']
        idx = -2
        if cls[-1] == ')':
            for i in range(len(cls)):
                if cls[-i-1] == '(':
                    idx = -i
        if idx<-2:
            job_dic_file.loc[index, 'ColorClass'] = cls[idx:idx+2]
        else:
            job_dic_file.loc[index, 'ColorClass'] = cls[idx:]

    machine_dic = machine_dic.drop_duplicates(keep='first', inplace=False)
    machine_dic = machine_dic.dropna(axis=0, how='any')

    print('--------读取文件--------')
    print('读取文件任务数：', job_dic_file.shape[0])
    print('读取文件机器数：', machine_dic.shape[0])

    for index, row in job_dic_file.iterrows():
        job_dic_file.loc[index, 'BillDate'] = pd.to_datetime(row['BillDate']) + pd.Timedelta(
            hours=random.randint(-12, 12))

    job_dic_file = job_dic_file.sort_values(by=['BillDate', 'RequestTime'], ascending=[True, True])

    job_dic_file['RequestTime'], job_dic_file['BillDate'] = pd.to_datetime(job_dic_file['RequestTime']), pd.to_datetime(
        job_dic_file['BillDate'])
    job_dic_file['dueTime'] = job_dic_file['RequestTime'] - job_dic_file['BillDate']

    print('任务时长增加')
    job_dic_file['standardtime'] = pa.multiple_processtime * job_dic_file['standardtime']

    # 调节这个参数可以调整任务的密度
    row_list = []

    for en in range(pa.num_ex):
        j = 0
        for i in range(job_dic_file.shape[0]):
            if random.random() < pa.new_job_rate:
                row_list.append(i)
                j += 1
            if j == pa.num_job:
                break

    job_dic_file = job_dic_file.iloc[row_list].reset_index(drop=True)
    machine_dic = machine_dic.head(pa.num_machine)
    job_dic_file.to_excel('out1.xlsx')

    # 任务族编号
    family_features_names = ['ColorName','ColorCode','ColorClass', 'ColorID', 'RequestCnt', 'batch']
    family_list = []
    family_features = []
    standard_times = []

    for index, row in job_dic_file.iterrows():

        if row['redye']:
            job_dic_file.loc[index, 'RequestTime'] = job_dic_file.loc[index, 'BillDate'] + pd.Timedelta(
                hours=random.randint(-72, 12))
        elif row['dueTime'] > pd.Timedelta(days=3):
            job_dic_file.loc[index, 'RequestTime'] = job_dic_file.loc[index, 'BillDate'] + pd.Timedelta(
                hours=random.randint(72, 240))

        features = [row[f] for f in family_features_names]
        if features in family_features:
            # 若已有该族，则获取编号
            f = family_features.index(features)
            family_list.append(f)
            job_dic_file.loc[index, 'standardtime'] = standard_times[f]
        else:
            # 随机生成和过去一样的族
            if len(family_features) and random.random() < 0.7:
                f = random.randint(max(0, len(family_features) - 3), len(family_features) - 1)
                job_dic_file.loc[index, family_features_names] = family_features[f]
                job_dic_file.loc[index, 'standardtime'] = standard_times[f]
            else:
                f = len(family_features)
                family_features.append(features)
                standard_times.append(row['standardtime'])

            family_list.append(f)

    job_dic_file['family'] = family_list
    print('加工族数:', max(family_list))
    # 字符串处理成数字编号, 这些列需进一步归一化
    str_col_name = ['batch', 'ColorClass', 'ColorID']

    job_dicts = []
    for ex in range(pa.num_ex):
        job_dict = job_dic_file.iloc[ex * pa.num_job:(ex + 1) * pa.num_job]

        for col in str_col_name:
            job_dict[col] = str_to_num(job_dict[col])

        # 归一化
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_col = ['ColorID', 'redye', 'ColorClass', 'batch', 'RequestCnt']
        for col in min_max_col:
            job_dict[col] = min_max_scaler.fit_transform(np.reshape(job_dict[col].values, [-1, 1]).tolist())

        job_dicts.append(job_dict)
        if ex == 0:
            job_dict.to_excel('out2.xlsx')


    print('--------读取设备--------')

    for index, row in machine_dic.iterrows():
        machine_dic.loc[index, 'MachineKG'] = pa.machine_capacity_list[index]
        print('机器ID:%s, 最大容量:%s' % (row['ID'], row['MachineKG']))

    print('##根据设备数据确定最大容量：%s KG' % pa.batch_capacity)
    return job_dicts, machine_dic


def get_ref_trajs(test_types, envs, pa):
    ref_eptime = {}
    ref_dly = {}
    ref_unfinished_job_num = {}
    ref_decision_count = {}
    all_infos = {}

    for test_type in test_types:
        eptime_list = []
        diy_list = []
        decision_count_list = []
        unfinished_job_num_list = []
        infos = []
        for ex in range(pa.num_ex):
            traj, info = get_ref_traj(test_type, envs[ex], pa)
            infos.append(info)

            # 提取关键信息
            all_eptime = np.array(info["time"])  # episode lengths
            enter_time, finish_time, start_time, dly, all_decision_count = process_all_info([info])
            finished_idx = (finish_time != -1)
            unfinished_idx = (1 - finished_idx).astype(np.bool)
            unfinished_job_num = np.sum(unfinished_idx)

            # 计算拖期
            all_dly = np.sum(dly)

            eptime_list.append(all_eptime)
            diy_list.append(all_dly)
            decision_count_list.append(all_decision_count)
            unfinished_job_num_list.append(unfinished_job_num)

            print('----------%s ex%s-----------' % (test_type, ex))
            print('Delay:%s Days' % int(all_dly / 24))
            print("Episode days: \t %s" % all_eptime, 'days')
            print('----------------------------')

        ref_decision_count[test_type] = np.sum(decision_count_list)
        ref_eptime[test_type] = np.sum(eptime_list)
        ref_dly[test_type] = diy_list
        ref_unfinished_job_num[test_type] = np.sum(unfinished_job_num_list)
        all_infos[test_type] = dict(zip(range(pa.num_ex), infos))

    return ref_eptime, ref_dly, ref_unfinished_job_num, ref_decision_count, all_infos


def get_ref_traj(test_type, env, pa):
    if test_type == 'SD':
        agent = SD_Agent(pa)
    elif test_type == 'A1':
        agent = A1_Agent(pa)
    else:
        print('test type error!')
        exit(1)

    obs, next_obs, acts, rews, info, action_probs, h_states, c_states, next_h_states, next_c_states = [], [], [], [], [], [], [], [], [], []
    agent_record = []
    env.reset()
    env.decision_time_record = []

    while True:
        while True:
            env.decision_time_record.append(env.curr_time)
            env.refresh_job()
            allocated, done = agent.select_action_step(env)
            if not allocated:
                break
            else:
                env.decision_count += 1

        if done:
            num_time_step = env.curr_time - env.time_horizon
            episode_days = (env.curr_time - env.time_horizon).days
            job_record = env.job_record
            batch_record = env.batch_record
            decision_count = env.decision_count
            decision_time_record = env.decision_time_record
            env.reset()
            break

        # 时间更新
        env.refresh_time()
        for machine in env.machine:
            machine.time_proceed(env.curr_time)

    traj = {'rews': np.asarray(rews),
            'obs': np.asarray(obs),
            'acts': np.asarray(acts),
            'action_probs': np.asarray(action_probs),
            'next_obs': np.asarray(obs),
            'h_states': h_states,
            'c_states': c_states,
            'time': num_time_step,
            'agent_record': np.asarray(agent_record)
            }

    info = {'time': episode_days,
            'job_info': job_record,
            'batch_info': batch_record,
            'decision_time_record': decision_time_record,
            'decision_count': decision_count
            }

    return traj, info


def main():
    pa = Parameters()
    pa.time_horizon = datetime.timedelta(days=1)
    pa.episode_days = datetime.timedelta(days=50)
    pa.time_window = datetime.timedelta(hours=4)

    # 重要参数
    pa.max_waiting_num = 10
    pa.num_batch = 15
    pa.num_ex = 1
    pa.num_job = 50
    pa.max_decision_time = pa.num_job*5*3
    pa.multiple_processtime = 6
    pa.new_job_rate = 0.2
    pa.machine_capacity_list = [2000, 2000, 1000, 1000, 500, 500]
    pa.num_machine = len(pa.machine_capacity_list)
    pa.batch_capacity = max(pa.machine_capacity_list)
    pa.discount = 0.99
    pa.num_seq_per_batch = 4

    pa.output_freq = 10

    pa.network_input_dim = pa.num_machine * 7 + pa.max_waiting_num * 8 + pa.num_batch * 9 + 1
    pa.lstm_output_dim = 8
    pa.lstm_input_dim = pa.network_input_dim + 1
    pa.ppo_network_input_dim = pa.network_input_dim + pa.lstm_output_dim
    pa.ppo_network_output_dim = pa.num_batch + 1

    pa.actor_lr_rate = 1e-5
    pa.critic_lr_rate = 1e-5

    pg_resume = None
    render = False

    Timestamp = '{0:%Y-%m-%d-%H-%M}'.format(datetime.datetime.now())
    pa.output_filename = pa.output_filename + str(Timestamp) + '_mach_' + str(pa.num_machine) + '_job_' + str(
        pa.num_job) + 'msglen' + str(pa.lstm_output_dim)

    launch(pa, pg_resume, render, repre=pa.repre, end='all_done')


if __name__ == '__main__':
    main()
