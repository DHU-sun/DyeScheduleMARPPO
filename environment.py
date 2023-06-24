import copy
import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parameters

zero_time = datetime.timedelta(hours=0)


def seconds_to_hour(seconds):
    return seconds / 60 / 60


class Env:
    def __init__(self, pa, nw_len_seqs=None, nw_size_seqs=None, nw_weight_seqs=None,
                 seed=42, render=False, repre='image', end='all_done'):

        self.pa = pa
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.pa.build_dist()
        self.nw_dist = self.pa.dist.bi_model_dist

        self.curr_time = 0
        self.done = False
        self.finished = False

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        if nw_len_seqs is None or nw_size_seqs is None:
            # generate new work
            self.nw_len_seqs, self.nw_size_seqs, self.nw_weight_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)

            self.workload = np.zeros(pa.num_res)

            self.workload = np.sum(self.nw_len_seqs) / float(self.pa.simu_len) / self.pa.num_machine / self.pa.num_ex
            print("Load on machine dimension is " + str(self.workload * 100) + "%")

            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                          [self.pa.num_ex, self.pa.simu_len])
            self.nw_weight_seqs = np.reshape(self.nw_weight_seqs,
                                             [self.pa.num_ex, self.pa.simu_len])
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])


        else:
            self.nw_len_seqs = nw_len_seqs
            self.nw_size_seqs = nw_size_seqs
            self.nw_weight_seqs = nw_weight_seqs

        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence

        # initialize system
        self.machine = [Machine(self.pa, i) for i in range(self.pa.num_machine)]
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

    def generate_sequence_work(self, simu_len):

        nw_len_seq = np.zeros(simu_len, dtype=int)
        nw_size_seq = np.zeros(simu_len, dtype=int)
        nw_weight_seq = np.zeros(simu_len, dtype=int)

        for i in range(simu_len):

            if np.random.rand() < self.pa.new_job_rate:  # a new job comes

                nw_len_seq[i], nw_size_seq[i], nw_weight_seq[i] = self.nw_dist()

                # 插入长长长短短短的片段

            if np.random.rand() < self.pa.slice_rate:
                # print('Add slice!!!!!!!!!!')
                j = 0
                while True:
                    i += 1
                    if i == simu_len:
                        break

                    elif np.random.rand() < self.pa.new_job_rate:  # a new job comes
                        if j < 3:
                            nw_len_seq[i], nw_weight_seq[i] = self.pa.dist.get_long_job()
                            j += 1

                        elif j >= 3:
                            nw_len_seq[i], nw_weight_seq[i] = self.pa.dist.get_short_job()
                            j += 1
                    if j > self.pa.slice_len:
                        break

        return nw_len_seq, nw_size_seq, nw_weight_seq

    def get_new_job_from_seq(self, seq_no, seq_idx):
        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_weight=self.nw_weight_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.job_record),
                      enter_time=self.curr_time)
        return new_job

    def observe(self):
        if self.repre == 'image':
            exit(1)

        if self.repre == 'parameters':
            para_repre = []  # shape:[pa.num_res*pa.mach_repre_num + pa.num_nw*pa.num]

            mach_repre = []
            for machine in self.machine:
                # 当前加工任务的剩余加工时间
                job_remain_time = machine.job_remain_time / self.pa.time_horizon
                # 空闲时间
                idle_time = machine.idle_time / self.pa.time_horizon
                # 是否空闲
                is_idle = machine.is_idle
                mach_repre.extend([job_remain_time, idle_time, is_idle])
            para_repre.extend(mach_repre)

            # Slot状态参数：为Slot里面的Job的参数
            job_repre = []
            for job in self.job_slot.slot:
                if job is not None:
                    weight = job.weight / self.pa.time_horizon
                    job_len = job.len / self.pa.time_horizon
                    waiting_time = (self.curr_time - job.enter_time) / self.pa.time_horizon

                    job_repre.extend([weight, job_len, waiting_time])
                else:
                    # 否则填充0
                    job_repre.extend([0, 0, 0])
            para_repre.extend(job_repre)

            # Backlog状态参数：占用率
            job_backlog_repre = self.job_backlog.curr_size / self.pa.backlog_size
            para_repre.append(job_backlog_repre)

            para_repre = np.reshape(para_repre, [-1, 1])

            return para_repre.ravel()

    def plot_state(self):
        plt.figure("screen", figsize=(20, 5))

        skip_row = 0

        for i in range(self.pa.num_res):

            plt.subplot(self.pa.num_res,
                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                        i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

            plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)

            for j in range(self.pa.num_nw):

                job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))
                if self.job_slot.slot[j] is not None:  # fill in a block of work
                    job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res_vec[i]] = 1

                plt.subplot(self.pa.num_res,
                            1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                            1 + i * (
                                    self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

                plt.imshow(job_slot, interpolation='nearest', vmax=1)

                if j == self.pa.num_nw - 1:
                    skip_row += 1

        skip_row -= 1
        backlog_width = np.int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
        backlog = np.zeros((self.pa.time_horizon, backlog_width), dtype=int)

        backlog[: int(self.job_backlog.curr_size / backlog_width), : backlog_width] = 1
        backlog[int(self.job_backlog.curr_size / backlog_width), : self.job_backlog.curr_size % backlog_width] = 1

        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_nw + 1 + 1)

        plt.imshow(backlog, interpolation='nearest', vmax=1)

        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_res * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

        extra_info = np.ones((self.pa.time_horizon, 1)) * \
                     self.extra_info.time_since_last_new_job / \
                     float(self.extra_info.max_tracking_time_since_last_job)

        plt.imshow(extra_info, interpolation='nearest', vmax=1)

        # plt.show()     # manual
        plt.pause(0.01)  # automatic

    def get_reward(self):

        reward = 0

        count = 0

        for j in self.job_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty * j.weight
                count += 1

        for j in self.job_backlog.backlog:
            if j is not None:
                reward += self.pa.hold_penalty * j.weight
                count += 1
        # for m in self.machine:
        #    reward += self.pa.hold_penalty * m.running_job.weight * time_count
        #    count += 1
        # print('now the unfinished job:', count)

        if self.done:
            if self.finished:
                assert count == 0
                reward += 100
            else:
                reward += -count * 50

        return reward

    def check_done(self):
        if self.end == "no_new_job":  # end of new job sequence
            if self.seq_idx >= self.pa.simu_len:
                self.done = True
                self.finished = True

            elif self.curr_time >= self.pa.episode_max_length:  # run too long, force termination
                self.done = True
                self.finished = False
            else:
                self.done = False
                self.finished = False

        elif self.end == "all_done":  # everything has to be finished
            if self.seq_idx >= self.pa.simu_len and \
                    all(self.machine[i].running_job is None for i in range(self.pa.num_res)) and \
                    all(s is None for s in self.job_slot.slot) and \
                    all(s is None for s in self.job_backlog.backlog):
                self.done = True
                self.finished = True

            elif self.curr_time >= self.pa.episode_max_length:  # run too long, force termination
                self.done = True
                self.finished = False
            else:
                self.done = False
                self.finished = False

        else:
            print('Error：没有这个结束的类型!')
            exit(1)

    def step(self, a, repeat=False):

        # print('action:   ', a)

        self.done = False
        reward = 0
        info = None
        busy = True

        for m in self.machine:
            if m.running_job == None:
                busy = False
                break

        allocated = False
        if a != self.pa.num_slot:
            job = self.job_slot.slot[a]

            if job is not None:
                if not busy:
                    allocated = m.allocate_job(job, self.curr_time)

                if allocated:
                    self.job_slot.slot[a] = None
                    if self.job_backlog.curr_size > 0:
                        self.job_slot.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
                        self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                        self.job_backlog.backlog[-1] = None
                        self.job_backlog.curr_size -= 1

        if (not busy) and (not allocated):
            for job in self.job_slot.slot:
                if job is not None:
                    job.delay_time += 1

        if not allocated:
            self.curr_time += 1
            for machine in self.machine:
                finished_job = machine.time_proceed(self.curr_time)
                if finished_job is not None:
                    self.job_record.job_record[finished_job.id] = finished_job
            self.check_done()
            if self.done:
                pass
            else:
                if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
                    new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)

                    if new_job.len > 0:  # a new job comes

                        to_backlog = True

                        empty_slot_idx = []
                        for i in range(self.pa.num_slot):
                            if self.job_slot.slot[i] is None:  # put in new visible job slots
                                empty_slot_idx.append(i)
                        if len(empty_slot_idx) > 0:
                            slot_idx = np.random.choice(empty_slot_idx)

                            self.job_slot.slot[slot_idx] = new_job
                            self.job_record.job_record[new_job.id] = new_job
                            to_backlog = False

                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.job_record[new_job.id] = new_job
                            else:  # abort, backlog full
                                self.job_record.job_record[new_job.id] = new_job
                                print("Backlog is full.")
                                # exit(1)

                        process_time = False
                        self.extra_info.new_job_comes()

                    # add new jobs
                    self.seq_idx += 1

        self.check_done()
        ob = self.observe()
        info = self.job_record
        if allocated:
            reward = 0
        else:
            reward = self.get_reward()
        done = self.done

        if self.done:
            self.seq_idx = 0
            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex
            self.history_time = self.curr_time
            # print('episode time:', self.history_time)
            self.reset()
        return ob, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0
        self.finished = False
        self.done = False

        # initialize system
        self.machine = [Machine(self.pa, i) for i in range(self.pa.num_machine)]
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)
        return self.observe()


class Job:
    def __init__(self, res_vec, job_len, job_weight, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.weight = job_weight
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1
        self.p_mach = -1
        self.process_mach_num = -1
        self.delay_time = 0


class DyeJob:
    def __init__(self, pa):
        self.time_cost = 0
        self.water_cost = 0
        self.RequestTime = zero_time
        self.BillDate = zero_time
        self.ColorID = -1
        self.ColorName = -1
        self.ColorClass = -1
        self.family = -1
        self.dyeKG = -1
        self.dyeCnt = -1
        self.FixTime = -1
        self.ProduceNo = -1
        self.dyelot = -1
        self.standardtime = zero_time
        self.redye = -1
        self.start_time = -1
        self.finish_time = -1
        self.dly = 0.0
        self.adv_rew = False
        self.fabric_batch = -1



class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_slot


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.job_record = {}


class BatchRecord:
    def __init__(self):
        self.batch_record = {}


class DyeEnv:
    def __init__(self, pa, seq_no, job_dict, machine_dict):
        self.b_reward = 0
        self.s_reward = 0
        self.seq_no = seq_no
        self.decision_count = 0
        self.decision_time_record = []
        self.batch_discount = 0.8
        self.curr_time_window = pa.time_window
        self.pa = pa
        self.waiting_job = []
        self.batch_job_temp = []
        self.batches = [Batch(self.pa) for i in range(self.pa.num_batch)]
        self.machine_dict, self.job_dict = machine_dict, job_dict
        self.machine, self.job = [], []
        self.job_record = JobRecord()
        self.batch_record = BatchRecord()
        self.job_slot = JobSlot(pa)
        self.initialize_job_machine()
        print('--------初始化任务和机器--------')
        print('机器数：%s, 任务数%s' % (len(self.machine), len(self.job)))
        print('程序开始时间:', self.time_horizon)
        print('最晚到达的任务时间：', self.job[-1].BillDate)
        print('建议在最晚的任务交货期之前停止加工：', max([job.RequestTime for job in self.job]))
        print('任务到达时间跨度：', (self.job[-1].BillDate - self.time_horizon).days)
        self.curr_time = self.time_horizon
        self.refresh_job()
        np.random.seed(314159)

    def initialize_job_machine(self):
        self.job = []
        self.machine = []
        # initialize job
        # 任务按时间排序
        self.job_dict = self.job_dict.sort_values(by="BillDate", ascending=True)

        datatime_col = ['BillDate', 'FixTime', 'dyeTime', 'RequestTime']
        for col in datatime_col:
            self.job_dict[col] = pd.to_datetime(self.job_dict[col], format='%Y-%m-%d %H:%M:%S.%f')
        id = 0
        for index, row in self.job_dict.iterrows():
            new_job = DyeJob(self.pa)
            new_job.dyeKG = row['dyeKG']
            new_job.ColorClass = row['ColorClass']
            new_job.ColorID = row['ColorID']
            new_job.BillDate = row['BillDate']
            new_job.redye = row['redye']
            new_job.dyeCnt = row['RequestCnt']
            new_job.dyelot = row['dyelot']
            new_job.standardtime = datetime.timedelta(days=row['standardtime'])
            new_job.RequestTime = row['RequestTime']
            new_job.fabric_batch = row['batch']
            new_job.family = row['family']
            new_job.id = id
            self.job.append(new_job)
            self.job_record.job_record[new_job.id] = new_job
            id += 1
        self.time_horizon = self.job_dict['BillDate'].min()

        # initialize machine
        for index, row in self.machine_dict.iterrows():
            new_machine = DyeMachine(self.pa, index, self.time_horizon)
            new_machine.max_capacity = row['MachineKG']
            new_machine.KindID = row['KindID']
            self.machine.append(new_machine)

    def observe(self):
        para_repre = []  # shape:[pa.num_res*pa.mach_repre_num + pa.num_nw*pa.num]

        mach_repre = []
        for machine in self.machine:
            # 当前加工任务的剩余加工时间
            job_remain_time = machine.job_remain_time / self.pa.time_horizon
            # 正在加工的颜色
            process_color = machine.ColorID
            # 正在加工的色系
            colorClass = machine.ColorClass
            # 设备容量
            capacity = machine.max_capacity / self.pa.batch_capacity  # 最大的染缸容量是2000以实现归一化
            # 正在加工的任务kg数
            if machine.running_job is not None:
                job_num_kg = machine.running_job.dyeKG / self.pa.batch_capacity
            else:
                job_num_kg = 0
            # 空闲时间
            idle_time = machine.idle_time / self.pa.time_horizon
            # 是否空闲
            is_idle = machine.is_idle
            mach_repre.extend([job_remain_time, process_color, colorClass, capacity, job_num_kg, idle_time, is_idle])
        para_repre.extend(mach_repre)

        job_repre = []
        for idx, job in enumerate(self.waiting_job):
            if job is not None:
                job_len = job.standardtime / self.pa.time_horizon
                waiting_time = (self.curr_time - job.BillDate) / self.pa.time_horizon / 10
                remain_time = (job.RequestTime - job.standardtime - self.curr_time) / self.pa.time_horizon / 30
                num_kg = job.dyeKG / self.pa.batch_capacity
                ColorID = job.ColorID
                colorClass = job.ColorClass
                fabric_batch = job.fabric_batch
                redye = job.redye
                job_repre.extend([job_len, waiting_time, remain_time, num_kg, ColorID, colorClass, fabric_batch, redye])
            if idx + 1 == self.pa.max_waiting_num:
                break
        job_repre = list(job_repre + [0] * (self.pa.max_waiting_num * 8 - len(job_repre)))

        para_repre.extend(job_repre)

        # Batch状态参数：为BatchSlot里面的Batch的参数
        batch_repre = []
        for batch in self.batches:
            if len(batch.job_batch) > 0:
                job_len = batch.standardtime / self.pa.time_horizon
                waiting_time = sum(
                    [(self.curr_time - job.BillDate) / self.pa.time_horizon / 10 for job in batch.job_batch])
                mean_remain_time = np.mean(
                    [(job.RequestTime - job.standardtime - self.curr_time) / self.pa.time_horizon / 30 for job in
                     batch.job_batch])
                min_remain_time = min(
                    [(job.RequestTime - job.standardtime - self.curr_time) / self.pa.time_horizon / 30 for job in
                     batch.job_batch])
                occupancy = batch.dyeKG / batch.max_capacity
                ColorID = batch.ColorID
                colorClass = batch.ColorClass
                fabric_batch = batch.fabric_batch
                redye = batch.redye
                batch_repre.extend(
                    [job_len, waiting_time, mean_remain_time, min_remain_time, occupancy, ColorID, colorClass,
                     fabric_batch, redye])
            else:
                # 否则填充0
                batch_repre.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
        para_repre.extend(batch_repre)

        para_repre = np.reshape(para_repre, [-1, 1])

        return para_repre.ravel()

    def batch_step(self, a, job):
        # 搞一个容器装待分配的job
        batched = False
        if a < self.pa.num_batch:
            batched = self.batches[a].in_batch(job)
        if batched:
            # 记录组批成功的任务
            self.batch_job_temp.append(job)

        # 直到最后一个工件才获取该时间内的reward，同一时刻只计算一次reward
        if job == self.waiting_job[-1]:
            # 为了正常循环，须一起删除这些组批成功的任务
            for job in self.batch_job_temp:
                self.waiting_job.remove(job)
            self.batch_job_temp = []
            reward = self.batch_reward()
        else:
            reward = self.b_reward
            self.b_reward = 0
        ob = self.observe()
        ob = np.append(ob, 0)
        done = self.check_done()

        info = self.job_record

        # 最终未完成负向惩罚与任务完成即给正向奖励，后者更合理，因此注释此段
        # if done:
        #     finish_time = np.array([info.job_record[i].finish_time for i in range(len(info.job_record))])
        #     finished_idx = (finish_time != -1)
        #     unfinished_idx = (1 - finished_idx).astype(np.bool)
        #     unfinished_job_num = np.sum(unfinished_idx)
        #     max_delay_days = self.pa.episode_days.days
        #     reward += -unfinished_job_num * 24 * max_delay_days
        return ob, reward, done, info

    def schedule_step(self, a):
        allocated = False
        if a < self.pa.num_batch:
            batch = self.batches[a]
            if batch.dyeKG > 0:
                time_cost_list,water_cost_list = [],[]
                for machine in self.machine:
                    if machine.max_capacity > batch.dyeKG:
                        if machine.max_capacity == min(
                                self.pa.machine_capacity_list) or batch.dyeKG > machine.max_capacity * 0.3:
                            time_cost, water_cost = machine.color_trans_cost(batch)
                            time_cost_list.append(time_cost)
                            water_cost_list.append(time_cost)
                        else:
                            time_cost_list.append(datetime.timedelta(hours=1e10))
                            water_cost_list.append(1e10)
                    else:
                        time_cost_list.append(datetime.timedelta(hours=1e10))
                        water_cost_list.append(1e10)
                # 要找空闲机器里面最小的
                machine_idx = time_cost_list.index(min(time_cost_list))
                machine = self.machine[machine_idx]
                if machine.running_job == None and time_cost_list[machine_idx] < datetime.timedelta(
                        hours=1e10) and machine.max_capacity >= batch.dyeKG:
                    allocated = machine.allocate_job(batch, self.curr_time)
                if allocated:
                    if self.pa.multi_objective:
                        self.s_reward += self.pa.transform_penalty * water_cost_list[machine_idx]  # 染色切换成本奖励值，算给排缸agent
                    print('排缸成功：安排批次  ', a, '  至染缸  #', machine_idx, '||染缸容量:', batch.dyeKG, '/', machine.max_capacity,
                          '||切换时间:', time_cost_list[machine_idx],
                          '||批次颜色：', batch.ColorID, '||批次色系：', batch.ColorClass, '||批次族号：',
                          batch.family)
                    batch.id = len(self.batch_record.batch_record)
                    self.batch_record.batch_record[batch.id] = copy.deepcopy(batch)
                    batch.reset()

        if not allocated:
            reward = self.schedule_reward()
        else:
            reward = self.s_reward
            self.s_reward = 0
        ob = self.observe()
        ob = np.append(ob, 1)
        done = self.check_done()
        info = self.batch_record
        return ob, reward, done, info, allocated

    def batch_reward(self):

        for DyeJob in self.waiting_job:
            current_dly = seconds_to_hour((self.curr_time - DyeJob.RequestTime).total_seconds())
            if DyeJob.dly < current_dly:
                if DyeJob.BillDate == self.curr_time and DyeJob.RequestTime < self.curr_time:
                    delta_dly = seconds_to_hour((self.curr_time - DyeJob.RequestTime).total_seconds())
                    self.b_reward += self.pa.tradness_penalty*delta_dly
                    DyeJob.dly = delta_dly

                elif DyeJob.RequestTime < self.curr_time:
                    delay_start = max(self.curr_time - self.curr_time_window, DyeJob.RequestTime)
                    delay_end = self.curr_time
                    delta_dly = seconds_to_hour((delay_end - delay_start).total_seconds())
                    current_dly = seconds_to_hour((self.curr_time - DyeJob.RequestTime).total_seconds())
                    assert abs(delta_dly - (current_dly - DyeJob.dly)) < 0.000001
                    self.b_reward += self.pa.tradness_penalty*delta_dly
                    DyeJob.dly = seconds_to_hour((self.curr_time - DyeJob.RequestTime).total_seconds())

        b_reward = self.b_reward
        self.b_reward = 0
        return b_reward

    def schedule_reward(self):

        for batch in self.batches:
            for DyeJob in batch.job_batch:
                current_dly = seconds_to_hour((self.curr_time - DyeJob.RequestTime).total_seconds())
                if DyeJob.dly < current_dly:
                    if DyeJob.BillDate == self.curr_time and DyeJob.RequestTime < self.curr_time:
                        delta_dly = seconds_to_hour((self.curr_time - DyeJob.RequestTime).total_seconds())
                        self.s_reward += self.pa.tradness_penalty * delta_dly

                    elif DyeJob.RequestTime < self.curr_time:
                        delay_start = max(self.curr_time - self.curr_time_window, DyeJob.RequestTime)
                        delay_end = self.curr_time
                        delta_dly = seconds_to_hour((delay_end - delay_start).total_seconds())
                        assert abs(delta_dly - (current_dly - DyeJob.dly)) < 0.000001
                        self.s_reward += self.pa.tradness_penalty*delta_dly

                    DyeJob.dly = current_dly

        for machine in self.machine:
            batch = machine.running_job
            if batch is not None:
                for DyeJob in batch.job_batch:
                    current_dly = seconds_to_hour((self.curr_time - DyeJob.RequestTime).total_seconds())
                    if DyeJob.dly < current_dly:
                        if DyeJob.BillDate == self.curr_time and DyeJob.RequestTime < self.curr_time:
                            delta_dly = seconds_to_hour((self.curr_time - DyeJob.RequestTime).total_seconds())
                            self.s_reward += self.pa.tradness_penalty * delta_dly
                        elif DyeJob.RequestTime < self.curr_time:
                            delay_start = max(self.curr_time - self.curr_time_window, DyeJob.RequestTime)
                            delay_end = self.curr_time
                            delta_dly = seconds_to_hour((delay_end - delay_start).total_seconds())
                            assert abs(delta_dly - (current_dly - DyeJob.dly)) < 0.000001
                            self.s_reward += self.pa.tradness_penalty*delta_dly
                        DyeJob.dly = current_dly

        reward = self.s_reward
        self.s_reward = 0
        return reward

    def check_done(self):
        done = False
        if len(self.job) == 0 and all(machine.running_job is None for machine in self.machine) and all(
                len(batch.job_batch) == 0 for batch in self.batches) and len(self.waiting_job) == 0:
            done = True
        elif self.curr_time > self.time_horizon + self.pa.episode_days:
            done = True
            print('生产天数超时：', self.curr_time - self.time_horizon, '决策次数：', self.decision_count, end=' ')
            print('剩余未完成任务数：', sum([job.finish_time == -1 for _, job in self.job_record.job_record.items()]))
        elif self.decision_count > self.pa.max_decision_time:
            done = True
            print('决策次数超时：', self.decision_count,'天数：', self.curr_time - self.time_horizon, end=' ')
            print('剩余未完成任务数：', sum([job.finish_time == -1 for _, job in self.job_record.job_record.items()]))

        return done

    def refresh_job(self):
        new_jobs = []

        for job in self.job:
            if job.BillDate == self.curr_time:
                new_jobs.append(job)
            else:
                break

        for job in new_jobs:
            self.waiting_job.append(job)
            self.job.remove(job)

    def refresh_time(self):
        self.curr_time += self.pa.time_window
        self.curr_time_window = self.pa.time_window

        if len(self.job) > 0:
            if self.job[0].BillDate < self.curr_time:
                self.curr_time_window -= self.curr_time - self.job[0].BillDate
                self.curr_time = self.job[0].BillDate

        for machine in self.machine:
            if machine.running_job is not None:
                if machine.running_job.finish_time < self.curr_time:
                    self.curr_time_window -= self.curr_time - machine.running_job.finish_time
                    self.curr_time = machine.running_job.finish_time

    def reset(self):
        self.job_record = JobRecord()
        self.batch_record = BatchRecord()
        self.initialize_job_machine()
        self.curr_time = self.time_horizon
        self.job_slot = JobSlot(self.pa)
        self.waiting_job = []
        self.batches = [Batch(self.pa) for i in range(self.pa.num_batch)]
        self.curr_time_window = self.pa.time_window
        self.decision_count = 0
        self.decision_time_record = []
        self.refresh_job()

        return self.observe()


class Batch:
    def __init__(self, pa):
        self.pa = pa
        self.max_capacity = self.pa.batch_capacity
        self.job_batch = []
        self.dyeKG = 0
        self.ColorID = -1
        self.ColorClass = -1
        self.fabric_batch = -1
        self.start_time = zero_time
        self.finish_time = zero_time
        self.mean_wait_time = zero_time
        self.min_due_time = zero_time
        self.job_remain_time = zero_time
        self.redye = 0
        self.standardtime = zero_time

    def in_batch(self, DyeJob):
        batch_success = False
        if DyeJob.dyeKG + self.dyeKG > self.max_capacity:
            return batch_success

        if self.dyeKG == 0:
            self.job_batch.append(DyeJob)
            batch_success = True
            # 加入一个工件后初始化批特征
            self.ColorID = DyeJob.ColorID
            self.ColorClass = DyeJob.ColorClass
            self.fabric_batch = DyeJob.fabric_batch
            self.redye = DyeJob.redye
            self.dyeKG += DyeJob.dyeKG
            self.standardtime = DyeJob.standardtime
            self.family = DyeJob.family

        elif self.dyeKG + DyeJob.dyeKG < self.max_capacity:

            if self.family != DyeJob.family:
                return batch_success
            else:
                self.dyeKG += DyeJob.dyeKG
                self.job_batch.append(DyeJob)
                batch_success = True
                if DyeJob.redye > self.redye:
                    self.redye = DyeJob.redye

        else:
            batch_success = False

        return batch_success

    def reset(self):
        self.job_batch = []
        self.dyeKG = 0
        self.ColorID = -1
        self.ColorClass = -1
        self.fabric_batch = -1
        self.start_time = zero_time
        self.finish_time = zero_time
        self.mean_wait_time = zero_time
        self.min_due_time = zero_time
        self.job_remain_time = zero_time
        self.redye = 0
        self.standardtime = zero_time


class Machine:
    def __init__(self, pa, machine_id):
        self.machine_id = machine_id
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.running_job = None
        self.idle_time = zero_time
        self.is_idle = 1
        self.job_remain_time = zero_time  # 需要除以时间基线以归一化
        self.pa = pa
        # colormap for graphical representation
        self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        np.random.shuffle(self.colormap)

        # graphical representation
        # self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))

    def allocate_job(self, batch, curr_time):
        allocated = False
        assert self.is_idle == 1
        assert self.running_job == None
        # print('allocate job %s' % (job.id), 'on machine %s' % self.machine_id)
        batch.start_time = curr_time
        batch.finish_time = curr_time + batch.len
        batch.process_mach_num = self.machine_id

        allocated = True

        assert batch.start_time != -1
        assert batch.finish_time != -1
        assert batch.process_mach_num != -1
        assert batch.finish_time > batch.start_time

        self.running_job = batch
        self.is_idle = 0
        self.job_remain_time = self.running_job.len
        self.idle_time = 0

        return allocated

    def color_trans_cost(self, job):
        if self.running_job is not None or self.max_capacity < job.dyeKG:
            time_cost, water_cost = 1e3, 1e3

        elif self.ColorClass == job.ColorClass:
            if self.ColorID == job.ColorID:
                time_cost, water_cost = 0, 0
            elif self.ColorID > job.ColorID:
                time_cost, water_cost = 1, 0
            else:
                time_cost, water_cost = 2, 0
        else:
            time_cost, water_cost = 2, 2

        return datetime.timedelta(hours=time_cost), water_cost


class DyeMachine(Machine):
    def __init__(self, pa, id, time_horizon):
        super().__init__(pa, id)
        self.max_capacity = -1
        self.ColorID = -1
        self.ColorClass = -1
        self.min_due_time = zero_time
        self.occupancy_ratio = 0
        self.KindID = -1
        self.release_time = time_horizon
        self.idle_time = zero_time

    def allocate_job(self, job_batch, curr_time):
        allocated = False

        assert self.is_idle == True
        assert self.running_job == None
        # print('allocate job %s' % (job.id), 'on machine %s' % self.machine_id)

        self.is_idle = False
        self.job_remain_time = job_batch.standardtime
        self.idle_time = zero_time

        time_cost, water_cost = self.color_trans_cost(job_batch)

        self.ColorID = job_batch.ColorID
        self.ColorClass = job_batch.ColorClass
        self.occupancy_ratio = job_batch.dyeKG / self.max_capacity

        job_batch.time_cost, job_batch.water_cost = time_cost, water_cost

        job_batch.start_time = curr_time + time_cost
        job_batch.finish_time = curr_time + time_cost + job_batch.standardtime
        job_batch.process_mach_num = self.machine_id

        allocated = True

        for job in job_batch.job_batch:
            job.finish_time = job_batch.finish_time
            job.start_time = job_batch.start_time
            job.water_cost = job_batch.water_cost
            job.time_cost = job_batch.time_cost

        assert job_batch.start_time != zero_time
        assert job_batch.finish_time != zero_time
        assert job_batch.process_mach_num != zero_time
        assert job_batch.finish_time > job_batch.start_time

        self.running_job = copy.deepcopy(job_batch)

        return allocated

    def time_proceed(self, curr_time):

        finished_job = None

        if self.running_job is not None:
            self.job_remain_time = self.running_job.finish_time - curr_time

            # 如果当前任务恰好加工完毕
            if self.job_remain_time == zero_time:
                finished_job = self.running_job
                self.release_time = curr_time
                self.is_idle = True
                self.running_job = None

        else:
            self.job_remain_time = zero_time
            self.idle_time = curr_time - self.release_time

        return finished_job

    def color_trans_cost(self, job):
        if self.running_job is not None or self.max_capacity < job.dyeKG:
            time_cost, water_cost = 1e10, 1e10

        elif self.ColorClass == job.ColorClass:
            if self.ColorID == job.ColorID:
                time_cost, water_cost = 0, 0
            elif self.ColorID > job.ColorID:
                time_cost, water_cost = 2, 20
            else:
                time_cost, water_cost = 0.5, 5
        elif self.ColorClass != job.ColorClass:
            time_cost, water_cost = 2, 20

        return datetime.timedelta(hours=time_cost), water_cost


class ExtraInfo:

    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters.Parameters()
    pa.num_slot = 5
    pa.simu_len = 50
    pa.num_machine = 3
    pa.num_ex = 10
    pa.new_job_rate = 1
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='parameters')
    ob = env.observe()
    print(ob)

    for j in range(100):
        for i in range(5):
            env.step(i)
            ob = env.observe()
            print(ob)

    env.step(5)
    assert env.job_backlog.backlog[0] is not None
    assert env.job_backlog.backlog[1] is None
    print("New job is backlogged.")

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slot.slot[3] == job

    print("- Backlog test passed -")


def test_compact_speed():
    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='compact')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


def test_image_speed():
    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


if __name__ == '__main__':
    test_backlog()
    test_compact_speed()
    test_image_speed()
