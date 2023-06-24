import os
import time
from collections import namedtuple
from itertools import count
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json



Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


def init_accums(pg_learner):  # in rmsprop
    accums = []
    params = pg_learner.get_params()
    for param in params:
        accum = np.zeros(param.shape, dtype=param.dtype)
        accums.append(accum)
    return accums


def rmsprop_updates_outside(grads, params, accums, stepsize, rho=0.9, epsilon=1e-9):
    assert len(grads) == len(params)
    assert len(grads) == len(accums)
    for dim in range(len(grads)):
        accums[dim] = rho * accums[dim] + (1 - rho) * grads[dim] ** 2
        params[dim] += (stepsize * grads[dim] / np.sqrt(accums[dim] + epsilon))


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def get_dye_sch_traj(ppo, env, pa, agent):
    """
    Run dye-agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    obs, next_obs, acts, rews, info, action_probs = [], [], [], [], [], []
    ep_r = 0

    state = env.reset()

    for t in count():
        action, action_prob = ppo.select_action(state)
        next_state, reward, done, info = env.step(action, repeat=True)
        action_probs.append(action_prob)
        obs.append(state)
        next_obs.append(next_state)
        acts.append(action)
        rews.append(reward)

        # if render: env.render()
        state = next_state

        if done:
            num_time_step = env.history_time
            break

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'action_probs': np.array(action_probs),
            'next_ob': np.array(next_obs),
            'info': info,
            'time': num_time_step
            }


def get_traj(ppo, env, pa, agent):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    obs, next_obs, acts, rews, info, action_probs = [], [], [], [], [], []
    ep_r = 0

    state = env.reset()

    for t in count():
        action, action_prob = ppo.select_action(state)
        next_state, reward, done, info = env.step(action, repeat=True)
        action_probs.append(action_prob)
        obs.append(state)
        next_obs.append(next_state)
        acts.append(action)
        rews.append(reward)

        # if render: env.render()
        state = next_state

        if done:
            num_time_step = env.history_time
            break

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'action_probs': np.array(action_probs),
            'next_ob': np.array(next_obs),
            'info': info,
            'time': num_time_step
            }


def concatenate_all_ob(trajs, pa):
    timesteps_total = 0
    for i in range(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])

    all_ob = np.zeros(
        (timesteps_total, pa.network_input_dim),
        dtype=np.float64)

    timesteps = 0
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['reward'])):
            all_ob[timesteps, :] = trajs[i]['ob'][j]
            timesteps += 1

    return all_ob


def process_all_info(trajs, pa):
    enter_time = []
    finish_time = []
    job_len = []
    weight = []
    delay_time = []

    for traj in trajs:
        enter_time.append(
            np.array([traj['info'].job_record[i].enter_time for i in range(len(traj['info'].job_record))]))
        finish_time.append(
            np.array([traj['info'].job_record[i].finish_time for i in range(len(traj['info'].job_record))]))
        job_len.append(np.array([traj['info'].job_record[i].len for i in range(len(traj['info'].job_record))]))
        weight.append(np.array([traj['info'].job_record[i].weight for i in range(len(traj['info'].job_record))]))
        delay_time.append(
            np.array([traj['info'].job_record[i].delay_time for i in range(len(traj['info'].job_record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)
    weight = np.concatenate(weight)
    delay_time = np.concatenate(delay_time)

    return enter_time, finish_time, job_len, weight, delay_time


def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve, episode_time_curve,
                  unfinished_job_curve,
                  ref_discount_rews, ref_slow_down, ref_episode_time):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(15, 12))

    ax = fig.add_subplot(221)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
    ax.plot(mean_rew_lr_curve, linewidth=1, label='RL mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=1, label=k)
    ax.plot(max_rew_lr_curve, linewidth=1, label='RL max')
    plt.legend(loc=7)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(222)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
    ax.plot(slow_down_lr_curve, linewidth=1, label='RL mean')
    for k in ref_discount_rews:
        #  concatenate 把每一个ex连接起来，所有值取平均
        ax.plot(np.tile(np.average(ref_slow_down[k]), len(slow_down_lr_curve)), linewidth=1, label=k)
    plt.legend(loc=7)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Total Weighted Completion Time", fontsize=20)

    ax = fig.add_subplot(223)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(episode_time_curve, linewidth=1, label='RL mean')
    for k in ref_discount_rews:
        #  concatenate 把每一个ex连接起来，所有值取平均
        ax.plot(np.tile(np.average(ref_episode_time[k]), len(slow_down_lr_curve)), linewidth=1, label=k)
    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Episode Time Steps", fontsize=20)

    ax = fig.add_subplot(224)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
    ax.plot(unfinished_job_curve, linewidth=1, label='RL mean')
    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Total Num Of Unfinished Job ", fontsize=20)

    plt.savefig('data/lr_curve/' + output_file_prefix + "_lr_curve" + ".pdf")


def get_traj_worker(ppo, env, pa, agent):
    trajs = []

    for i in range(pa.num_seq_per_batch):
        traj = get_traj(ppo, env, pa, agent)
        trajs.append(traj)

    # Compute discounted sums of rewards
    rets = [discount(traj["reward"], pa.discount) for traj in trajs]
    maxlen = max(len(ret) for ret in rets)
    padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

    # Compute time-dependent baseline
    baseline = np.mean(padded_rets, axis=0)

    # Compute advantage function
    advs = [ret - baseline[:len(ret)] for ret in rets]  # Baseline, 得到用于放到神经网络进行计算的reward

    all_eprews = np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards
    all_eplens = np.array([traj["time"] for traj in trajs])  # episode lengths

    # All Job Stat
    enter_time, finish_time, job_len, weight, delay_time = process_all_info(trajs, pa)
    finished_idx = (finish_time >= 0)
    all_weicomptime = np.sum(finish_time[finished_idx] * weight[finished_idx]) / pa.num_seq_per_batch
    unfinished_idx = (1 - finished_idx).astype(np.bool)
    unfinished_weicomptime = [pa.episode_max_length for i in range(sum(unfinished_idx))] * weight[unfinished_idx]

    unfinished_job_num = np.sum(unfinished_idx)

    # 存储经验
    for i, traj in enumerate(trajs):
        for j in range(len(traj["ob"])):
            trans = Transition(traj["ob"][j], traj["action"][j], traj["action_probs"][j], advs[i][j],
                               traj["next_obs"][j])
            ppo.store_transition(trans)

    return all_eprews, all_eplens, all_weicomptime, unfinished_job_num, {'job_len': job_len, 'weight': weight,
                                                                         'delay_time': delay_time}


def dict_to_json(file_name, dict):
    # 转换成json
    j = json.dumps(dict)
    # 写到json文件
    fileObject = open(file_name, 'w')
    fileObject.write(j)
    fileObject.close()

