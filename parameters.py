import numpy as np
import math
import datetime



class Parameters:
    def __init__(self):
        self.output_filename = '_debug_tmp'
        self.pre_training = False
        self.num_pre_training_time = 100

        self.multi_objective = False

        self.lstm_input_dim = 4
        self.lstm_output_dim = 8
        self.time_step = 1
        self.lstm_hidden_size = 32
        self.lstm_num_layers = 1
        self.ppo_network_input_dim = 5
        self.ppo_network_output_dim = 100

        self.num_epochs = 500  # number of training epochs
        self.simu_len = 500  # length of the busy cycle that repeats itself
        self.num_ex = 10  # number of sequences
        self.num_res = 1
        self.res = 1

        self.multiple_processtime = 2
        self.machine_capacity_list = [200, 500, 1000]
        self.max_decision_time = 10000
        self.batch_capacity = 1000

        self.time_window = datetime.timedelta(hours=6)

        self.slice_rate = 0.08
        self.slice_len = 6

        self.output_freq = 50  # interval for output and store parameters

        self.num_seq_per_batch = 10  # number of sequences to compute baseline
        self.episode_max_length = 800  # enforcing an artificial terminal

        #self.num_res = 2  # number of resources in the system
        #self.num_nw = 5  # maximum allowed number of work in the queue

        self.num_machine = 4
        self.mach_repre_num = 2
        self.job_repre_num = 3
        self.num_batch = 10

        self.time_horizon = 10  # number of time steps in the graph
        self.max_job_len = 10  # maximum duration of new jobs
        self.num_slot = 10  # maximum number of available resource slots
        self.max_job_size = 1  # maximum resource request of new work

        self.backlog_size = self.time_horizon * 3  # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 40  # maximum number of distinct colors in current work graph

        self.new_job_rate = 0.7  # lambda in new job arrival Poisson Process

        self.discount = 0.99  # discount factor

        self.max_waiting_num = 5

        self.episode_days = 365

        self.repre = 'parameters'

        # distribution for new job arrival

        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_length = self.time_horizon

        # for extra info, 1) time since last new job

        # compact representation
        self.network_input_dim = self.num_machine * 3 + self.num_slot * 3 + 1

        self.network_output_dim = self.num_slot + 1  # + 1 for void action

        self.delay_penalty = -1  # penalty for delaying things in the current work screen
        self.hold_penalty = -1  # penalty for holding things in the new work screen
        self.dismiss_penalty = -1  # penalty for missing a job because the queue is full
        self.tradness_penalty = -1  # 拖期惩罚因子
        self.transform_penalty = -1  # 染色切换惩罚因子

        self.num_frames = 1  # number of frames to combine and process
        self.lr_rate = 0.001  # learning rate
        self.actor_lr_rate = 0.0001
        self.critic_lr_rate = 0.0002
        self.max_job_res = 1

        self.rms_rho = 0.9  # for rms prop
        self.rms_eps = 1e-9  # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 1024
        self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = self.backlog_size / self.time_horizon

        self.network_input_dim = self.num_machine * 3 + self.num_slot * 3 + 1
        self.network_output_dim = self.num_slot + 1  # + 1 for void action

        self.lstm_input_dim = self.network_input_dim + 1

        self.ppo_network_input_dim = self.network_input_dim + self.lstm_output_dim
        self.ppo_network_output_dim = self.network_output_dim

