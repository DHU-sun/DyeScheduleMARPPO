# -*-coding:utf-8-*-
import gc
import os
from collections import namedtuple
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torch.distributions import Categorical
from torch.nn import Module
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from parameters import Parameters

TIMESTEP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10

EP_MAX = 10000
EP_LEN = 200
GAMMA = 0.9

BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]  # choose the method for optimization

torch.manual_seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'c_state', 'h_state'])


def drawing(x, name, file_name):
    plt.cla()
    gc.collect()
    print(plt.get_fignums())
    plt.figure(num=name)
    plt.plot(x, color='g', linewidth=0.8)
    plt.title(name + ' loss')
    plt.grid()
    plt.legend(['loss'])
    # plt.pause(0.0001)
    filename = 'dye_data/loss_curve/' + file_name + "_" + name + "_loss_curve.pdf"
    try:
        plt.savefig(filename)
    except:
        print('写入文件异常，可能被占用！')
        pass


class Actor(Module):
    def __init__(self, pa, Lstm):
        super(Actor, self).__init__()
        self.loss = []
        self.pa = pa
        self.Lstm = Lstm
        self.fc1 = nn.Linear(self.pa.ppo_network_input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, pa.ppo_network_output_dim)
        for param in self.parameters():  # 权重可训练
            param.requires_grad = True

    def forward(self, state, old_s_a, h_state, c_state):
        message, h_, c_ = self.Lstm(old_s_a, h_state, c_state)
        message = torch.squeeze(message, 2)
        m_s = torch.cat([message, state], 1)
        m_s = F.relu(self.fc1(m_s))
        m_s = F.relu(self.fc2(m_s))
        action_prob = F.softmax(self.action_head(m_s), dim=1)
        return action_prob, h_, c_


class Critic(Module):
    def __init__(self, pa, Lstm):
        super(Critic, self).__init__()
        self.loss = []
        self.critic_value = []
        self.pa = pa
        self.Lstm = Lstm
        self.fc1 = nn.Linear(self.pa.ppo_network_input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, state, old_s_a, h_state, c_state):
        message, h_, c_ = self.Lstm(old_s_a, h_state, c_state)
        m_s = torch.cat([torch.squeeze(message, 2), state], 1)
        m_s = F.relu(self.fc1(m_s))
        m_s = F.relu(self.fc2(m_s))
        value = self.state_value(m_s)
        return value, h_, c_


class Lstm(Module):
    def __init__(self, pa):
        super(Lstm, self).__init__()
        self.pa = pa
        self.lstm = nn.LSTM(
            input_size=self.pa.lstm_input_dim,
            hidden_size=self.pa.lstm_hidden_size,
            num_layers=self.pa.lstm_num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.pa.lstm_hidden_size, self.pa.lstm_output_dim)

    def forward(self, inputs, h_state, c_state):
        # inputs (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        lstm_out, (h_state, c_state) = self.lstm(inputs, (h_state, c_state))
        lstm_out = lstm_out.view(-1, self.pa.lstm_hidden_size)
        message = self.fc(lstm_out)
        return message.view(inputs.size(0), -1, self.pa.lstm_output_dim), h_state, c_state


class PPO_LSTM():

    def __init__(self, pa):
        self.name = ''
        self.pa = pa
        self.lstm_net = Lstm(self.pa)
        self.actor_net = Actor(self.pa, self.lstm_net)
        self.critic_net = Critic(self.pa, self.lstm_net)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('dye_data/param/exp/' + self.pa.output_filename)
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_update_time = 10
        # = pa.num_ex * pa.num_seq_per_batch * pa.episode_max_length * 3
        self.batch_size = 4096
        self.buffer_capacity = self.batch_size * 100
        self.actor_train_parameters = [param for param in self.actor_net.parameters() if param.requires_grad == True]
        self.critic_train_parameters = [param for param in self.actor_net.parameters() if param.requires_grad == True]
        self.actor_optimizer = optim.Adam(self.actor_train_parameters, pa.actor_lr_rate)
        self.critic_net_optimizer = optim.Adam(self.critic_train_parameters, pa.critic_lr_rate)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')
            os.makedirs('param/exp')

    def select_action(self, state, old_s_a, h_state, c_state):
        state = torch.from_numpy(state.astype(np.float32)).to(torch.float32)
        state = torch.reshape(state, [-1, self.pa.network_input_dim])
        with torch.no_grad():
            action_prob, h_state, c_state = self.actor_net(state, old_s_a, h_state, c_state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item(), h_state, c_state

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self, file_name):
        actor_state = {'model': self.actor_net.state_dict(), 'optimizer': self.actor_optimizer.state_dict(),
                       'epoch': self.training_step}
        torch.save(actor_state, 'param/net_param/actor_net_' + file_name + '.pkl')
        critic_state = {'model': self.critic_net.state_dict(), 'optimizer': self.critic_net_optimizer.state_dict(),
                        'epoch': self.training_step}
        torch.save(critic_state, 'param/net_param/critic_net_' + file_name + '.pkl')

    def load_param(self, param_resume):
        # load actor
        path = 'param/net_param/actor_net_' + str(param_resume) + '.pkl'
        checkpoint = torch.load(path)
        self.actor_net.load_state_dict(checkpoint['model'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['epoch']

        # load critic
        path = 'param/net_param/critic_net_' + str(param_resume) + '.pkl'
        checkpoint = torch.load(path)
        self.critic_net.load_state_dict(checkpoint['model'])
        self.critic_net_optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['epoch']

    def store_transition(self, transition):
        if len(self.buffer) >= self.buffer_capacity:
            del self.buffer[0]
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        h_state = torch.cat([t.h_state for t in self.buffer], 1)
        c_state = torch.cat([t.c_state for t in self.buffer], 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        Gt = torch.tensor(reward, dtype=torch.float)

        # print("The agent is updateing....")

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer) - 1)), self.batch_size, False):
                plus1_index = [i + 1 for i in index]

                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                s_a = torch.cat((state[index], action[index].to(torch.float)), 1)
                s_a = torch.reshape(s_a, [len(index), -1, self.pa.lstm_input_dim])
                V, _, _ = self.critic_net(state[plus1_index], s_a, h_state[:, plus1_index, :],
                                          c_state[:, plus1_index, :])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob, _, _ = self.actor_net(state[plus1_index], s_a, h_state[:, plus1_index, :],
                                                   c_state[:, plus1_index, :])  # new policy
                action_prob = action_prob.gather(1, action[plus1_index])
                ratio = (action_prob / old_action_log_prob[plus1_index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.actor_net.loss.append(action_loss.item())
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, delta)
                self.critic_net.critic_value.append(torch.mean(V).item())
                self.critic_net.loss.append(value_loss.item())
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience


def main():
    restore = False
    pa = Parameters()
    env = gym.make('Pendulum-v0').unwrapped
    ppo = PPO_LSTM(pa)
    all_ep_r = []
    action_set = np.linspace(-2, 2, 100)
    action = 0
    h_state, c_state, a = torch.zeros(1, 1, pa.lstm_hidden_size), torch.zeros(1, 1,
                                                                              pa.lstm_hidden_size), 0
    render = False
    if restore:
        ppo.load_param('aaa')
    for ep in range(EP_MAX):
        state = old_state = env.reset()
        ep_r = 0
        for t in range(EP_LEN):  # in one episode
            if render:
                env.render()
            s_a = np.append(old_state, action)
            s_a = torch.from_numpy(np.reshape(s_a, [1, 1, 4])).to(torch.float32)

            action, action_prob, next_h_state, next_c_state = ppo.select_action(state, s_a, h_state, c_state)
            next_h_state, next_c_state = next_h_state.data, next_c_state.data
            if t > 0:
                old_state = state

            a = action_set[action]
            next_state, reward, done, _ = env.step(np.array([a]))
            next_state = np.reshape(next_state, [3])
            trans = Transition(state, action, action_prob, reward, next_state, c_state=c_state, h_state=h_state)
            if render: env.render()
            ppo.store_transition(trans)
            state, h_state, c_state = next_state, next_h_state, next_c_state
        ep_r = np.mean([t.reward for t in ppo.buffer])
        all_ep_r.append(ep_r)
        ppo.update(ep)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )
        ppo.writer.add_scalar('reward/action_loss' + TIMESTEP, ep_r, global_step=ppo.training_step)
        if ep % 100 == 0:
            if ep == 900:
                pass
                # render = True
            if render:
                ppo.save_param('aaa')
                plt.plot(np.arange(len(all_ep_r)), all_ep_r);
                plt.xlabel('Episode');
                plt.ylabel('Moving averaged episode reward');
                plt.show()


if __name__ == '__main__':
    main()
    print("end")
