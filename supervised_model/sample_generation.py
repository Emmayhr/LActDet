# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   sample_generation.py
# @Time     :   2021/12/27

'''
样本生成：
根据alert的漏报、误报率生成样本
'''

from operator import index
import os
import pandas as pd
from tqdm import *
import numpy as np
from scipy.special import comb

# 漏报样本构造
def event_combination_delete(alert_path, sample_path):
    '''Construct data set'''

    df = pd.read_excel(alert_path, engine='openpyxl')

    event_num = len(df)
    percent = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    delete_list = []

    count = 0
    for i in tqdm(percent):
        dnum = int(event_num * i)   # delete number
        C = comb(event_num, dnum)

        if C < 300:
            for j in np.arange(300):
                random_list = np.random.choice(event_num, size = dnum, replace = False) # choose the deleted index
                index_list = random_list.tolist()

                if index_list not in delete_list:
                    delete_list.append(index_list)

                    tmp = df.drop(index_list)   # delete events

                    count = count + 1

                    sample_file = sample_path + "_" + str(count) + "_" + str(i) + '.csv'     # txt file path: dir/filename_count_percent.txt
                    tmp.to_csv(sample_file, na_rep = '')

                else:
                    continue
        
        else:
            for j in range(300):
                random_list = np.random.choice(event_num, size = dnum, replace = False)
                index_list = random_list.tolist()

                if index_list not in delete_list:
                    delete_list.append(index_list)

                    tmp = df.drop(index_list)

                    count = count + 1

                    sample_file = sample_path + "_" + str(count) + "_" + str(i) + '.csv'     # txt file path: dir/filename_count_percent.txt
                    tmp.to_csv(sample_file, na_rep = '')

                else:
                    continue


# 误报样本构造
def event_combination_replace(event_path, txt_path):
    '''Construct data set'''

    df = pd.read_excel(event_path, engine='openpyxl')
    replace = pd.read_excel('/home/kk/attack_activity_detect/data/test.xlsx', engine='openpyxl')

    event_num = len(df)
    percent = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    replace_list = []

    count = 0
    for i in tqdm(percent):
        dnum = int(event_num * i)   # delete number
        C = comb(event_num, dnum)

        if C < 300:
            for j in np.arange(300):
                random_list = np.random.choice(event_num, size = dnum, replace = False) # choose the deleted index
                index_list = random_list.tolist()

                # if len(index_list) == 0:
                #     continue

                if index_list not in replace_list:
                    replace_list.append(index_list)

                    tmp = df.copy(deep = True)
                    for k in index_list:
                        # row = k % 2
                        row = 0
                        tmp.iloc[k,:] = replace.iloc[row,:]  # replace events

                    count = count + 1

                    sample_file = txt_path + "_" + str(count) + "_" + str(i) + '.csv'     # txt file path: dir/filename_count_percent.txt
                    tmp.to_csv(sample_file, index=False, header=True)

                else:
                    continue
        
        else:
            for j in range(300):
                random_list = np.random.choice(event_num, size = dnum, replace = False)
                index_list = random_list.tolist()

                # if len(index_list) == 0:
                #     continue

                if index_list not in replace_list:
                    replace_list.append(index_list)

                    tmp = df.copy(deep = True)
                    for k in index_list:
                        # row = k % 2
                        row = 0
                        tmp.iloc[k,:] = replace.iloc[row,:]  # replace events

                    count = count + 1

                    sample_file = txt_path + "_" + str(count) + "_" + str(i) + '.csv'     # txt file path: dir/filename_count_percent.txt
                    tmp.to_csv(sample_file, na_rep = '')

                else:
                    continue


if __name__ == "__main__":
    classes = ['ddos1', 'ddos2', 'infiltration', 'httpdos', 'brute_force', 'Andariel', 'APT-29', 'AZORult', 'IcedID', 'Raccoon', 'Heartbleed', 'Wannacry']

    excel_dir = "/home/kk/attack_activity_detect/data/activity/"
    miss_sample_dir = "/home/kk/attack_activity_detect/test/miss_sample/"

    for excel in os.listdir(excel_dir):
        index = int(excel.split("_")[0]) - 1
        excel_path = excel_dir + excel

        print("Processing miss file: ", excel)
        miss_sample_path = miss_sample_dir + classes[index] + "/" + os.path.splitext(excel)[0]  # txt_dir + 去掉后缀的filename
        event_combination_delete(excel_path, miss_sample_path)