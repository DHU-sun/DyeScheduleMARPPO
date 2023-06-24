
# -*- coding: utf-8 -*-
from pg_re import plot_lr_curve
import json
file = open('data/test_jsonFile.json', 'r')
cruve_dic = json.load(file)

max_rew_lr_curve = cruve_dic['max_rew_lr_curve']
mean_rew_lr_curve = cruve_dic['mean_rew_lr_curve']
slow_down_lr_curve = cruve_dic['slow_down_lr_curve']
episode_time_curve = cruve_dic['episode_time_curve']
unfinished_job_num_curve = cruve_dic['unfinished_job_num_curve']
ref_discount_rews = cruve_dic['ref_discount_rews']
ref_slow_down = cruve_dic['ref_slow_down']
ref_episode_time = cruve_dic['ref_episode_time']

plot_lr_curve('comtrust_test', max_rew_lr_curve, mean_rew_lr_curve,
              slow_down_lr_curve, episode_time_curve, unfinished_job_num_curve,
              ref_discount_rews,ref_slow_down,ref_episode_time)









