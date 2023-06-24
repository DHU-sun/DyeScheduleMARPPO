# -*- coding: utf-8 -*-
from environment import seconds_to_hour
import numpy as np
import copy

class SD_Agent:
    '''
    REFREENCE:Mathirajan M , Ramasubramanian M . Efficient Heuristic Solution Methodologies for Scheduling Batch Processor with Incompatible Job-Families, Non-identical Job-Sizes and Non-identical Job-Dimensions[M]// Advances in Production Management Systems. Towards Smart Production Management Systems. 2019.
    Method: Sort by due date, 按照截止时间由近至远排序
    '''
    def __init__(self, pa):
        self.pa = pa

    def select_action_step(self, env):
        key = 'RequestTime'
        current_jobs = {}
        allocated = False

        # 选择该批加工
        idle_machines = []
        for DyeMachine in env.machine:
            if DyeMachine.is_idle:
                idle_machines.append(DyeMachine)
        # 有设备
        if len(idle_machines) > 0:
            # 找到当前待加工任务并根据加工族划分
            for id, dyeJob in env.job_record.job_record.items():
                # 筛选已到达且未加工的工件
                if dyeJob.BillDate <= env.curr_time and dyeJob.finish_time == -1:
                    if dyeJob.family in current_jobs.keys():
                        current_jobs[dyeJob.family].append(dyeJob)
                    else:
                        current_jobs[dyeJob.family] = [dyeJob]
            # 有任务
            if current_jobs:
                # 将族内任务排序
                for family, family_jobs in current_jobs.items():
                    family_jobs.sort(key=lambda x: getattr(x, key), reverse=False)  # 升序排列，截止时间越靠前任务优先级越高

                # 选出一个某个指标最大的族
                temp_key = env.time_horizon + self.pa.episode_days
                temp_fly = []
                for family, family_jobs in current_jobs.items():
                    if len(family_jobs) > 0:
                        key_value = getattr(family_jobs[0], key)
                        if key_value <= temp_key:
                            temp_key = key_value
                            temp_fly = family_jobs

                # 选择设备并安排一个最大的批，然后刷新
                idle_machines.sort(key=lambda x: x.max_capacity, reverse=False)
                sumKG = sum([dyeJob.dyeKG for dyeJob in temp_fly])

                max_machine = idle_machines[-1]
                if sumKG < max_machine.max_capacity:
                    # 若能将这个族全部组成一批加工，则用刚好大一点的设备进行加工
                    batch_job = temp_fly
                    for DyeMachine in idle_machines:
                        if DyeMachine.max_capacity > sumKG:
                            break
                else:
                    # 否则用最大的设备，按照族内任务的排序，加工尽可能多的任务然后退出
                    DyeMachine = idle_machines[-1]
                    batch_job = []
                    batch_kg = 0

                    for idx, dyeJob in enumerate(temp_fly):
                        if idx == 0 and dyeJob.dyeKG > DyeMachine.max_capacity:
                            break
                        elif batch_kg + dyeJob.dyeKG > DyeMachine.max_capacity:
                            break
                        else:
                            batch_kg += dyeJob.dyeKG
                            batch_job.append(dyeJob)
                            assert batch_kg <= DyeMachine.max_capacity

                env.batches[0].reset()
                if len(batch_job)>0:
                    for dyeJob in batch_job:
                        in_batch = env.batches[0].in_batch(dyeJob)
                        if not in_batch:
                            print('组批失败')
                            exit(1)

                    allocated = DyeMachine.allocate_job(env.batches[0], env.curr_time)
                    if allocated:
                        env.batches[0].id = len(env.batch_record.batch_record)
                        env.batch_record.batch_record[env.batches[0].id] = copy.deepcopy(env.batches[0])
                    else:
                        print('排缸失败')
                        exit(1)

        done = True
        for id, job in env.job_record.job_record.items():
            if job.finish_time == -1:
                done = False
                break

        if done:
            for id, job in env.job_record.job_record.items():
                if job.finish_time != -1:
                    if job.finish_time > job.RequestTime:
                        job.dly = seconds_to_hour((job.finish_time - job.RequestTime).total_seconds())
                elif env.curr_time > job.RequestTime:
                    job.dly = seconds_to_hour((env.curr_time - job.RequestTime).total_seconds())

        return allocated, done

class A1_Agent:
    '''
    MATHIRAJAN M, SIVAKUMAR A I. Minimizing total weighted tardiness on heterogeneous batch processing machines with incompatible job families[J]. International Journal of Advanced Manufacturing Technology, 2006, 28(9): 1038–1047., 按照截止时间由近至远排序
    '''
    def __init__(self, pa):
        self.pa = pa

    def select_action_step(self, env):
        key = 'RequestTime'
        current_jobs = {}
        allocated = False

        # 选择该批加工
        idle_machines = []
        for DyeMachine in env.machine:
            if DyeMachine.is_idle == True:
                idle_machines.append(DyeMachine)
        # 有设备
        if len(idle_machines) > 0:
            # 找到当前待加工任务并根据加工族划分
            for id, dyeJob in env.job_record.job_record.items():
                # 筛选已到达且未加工的工件
                if dyeJob.BillDate <= env.curr_time and dyeJob.finish_time == -1:
                    if dyeJob.family in current_jobs.keys():
                        current_jobs[dyeJob.family].append(dyeJob)
                    else:
                        current_jobs[dyeJob.family] = [dyeJob]
            # 有任务
            if current_jobs:
                maxIdx = float('-inf')
                # 计算族指标并选出最大得PT/CDD

                for family, family_jobs in current_jobs.items():
                    PT = family_jobs[0].standardtime.total_seconds()
                    CDD = sum([(DyeJob.RequestTime - env.curr_time).total_seconds() + 1.000001 for DyeJob in family_jobs])
                    index = PT / CDD
                    if index > maxIdx:
                        temp_fly = family_jobs
                        maxIdx = index

                temp_fly.sort(key=lambda x: getattr(x, key), reverse=False)  # 升序排列，到期时间越靠前任务优先级越高

                # 选择设备并安排一个最大的批，然后刷新
                idle_machines.sort(key=lambda x: x.max_capacity, reverse=False)
                sumKG = sum([dyeJob.dyeKG for dyeJob in temp_fly])

                max_machine = idle_machines[-1]
                if sumKG < max_machine.max_capacity:
                    # 若能将这个族全部组成一批加工，则用刚好大一点的设备进行加工
                    batch_job = temp_fly
                    for DyeMachine in idle_machines:
                        if DyeMachine.max_capacity > sumKG:
                            break
                else:
                    # 否则用最大的设备，按照族内任务的排序，加工尽可能多的任务然后退出
                    DyeMachine = idle_machines[-1]
                    batch_job = []
                    batch_kg = 0

                    for idx, dyeJob in enumerate(temp_fly):
                        if idx == 0 and dyeJob.dyeKG > DyeMachine.max_capacity:
                            break
                        elif batch_kg + dyeJob.dyeKG > DyeMachine.max_capacity:
                            break
                        else:
                            batch_kg += dyeJob.dyeKG
                            batch_job.append(dyeJob)
                            assert batch_kg <= DyeMachine.max_capacity

                env.batches[0].reset()
                if len(batch_job)>0:
                    for dyeJob in batch_job:
                        in_batch = env.batches[0].in_batch(dyeJob)
                        if not in_batch:
                            print('组批失败')
                            exit(1)

                    allocated = DyeMachine.allocate_job(env.batches[0], env.curr_time)
                    if allocated:
                        env.batches[0].id = len(env.batch_record.batch_record)
                        env.batch_record.batch_record[env.batches[0].id] = copy.deepcopy(env.batches[0])
                    else:
                        print('排缸失败')
                        exit(1)

        done = True
        for id, job in env.job_record.job_record.items():
            if job.finish_time == -1:
                done = False
                break

        if done:
            for id, job in env.job_record.job_record.items():
                if job.finish_time != -1:
                    if job.finish_time > job.RequestTime:
                        job.dly = seconds_to_hour((job.finish_time - job.RequestTime).total_seconds())
                elif env.curr_time > job.RequestTime:
                    job.dly = seconds_to_hour((env.curr_time - job.RequestTime).total_seconds())

        return allocated, done