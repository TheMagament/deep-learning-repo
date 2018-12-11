# -*- coding: utf-8 -*-
"""
Converts the statistics to a CSV row
"""

import numpy as np
import pickle
LOG_EVERY_N_STEPS = 10000
LEARNING_STARTS = 50000


file_r = open('statistics.pkl','rb')
file_w = open('statistics.csv','w')

Statistic = pickle.load(file_r)
mean_rewards = Statistic['mean_episode_rewards']
running_times = Statistic['running_times']
interesting_idxes=range(LEARNING_STARTS,len(mean_rewards),LOG_EVERY_N_STEPS)

for idx in interesting_idxes:
    file_w.write(str(idx)+','+str(running_times[idx])+','+str(mean_rewards[idx])+'\n');
    
    
file_r.close();
file_w.close();

