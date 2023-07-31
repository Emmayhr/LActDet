# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   event_extraction.py
# @Time     :   2022/03/28

'''
1. alert去重得到event

2. 构造document:
每条记录拼接为一个word
每个attack_phase为一个sentence
一个attack_activity为一个document
'''

from operator import index
import pandas as pd
import os
from tqdm import tqdm


attack_phase = ['Reconnaissance', 'Resource_Development', 'Initial_Access', 'Execution', 'Persistence', 'Privilege_Escalation', 
                'Defense_Evasion', 'Credential_Access', 'Discovery', 'Lateral_Movement', 'Collection', 'Command_and_Control', 'Exfiltration', 'Impact']

# phase_dict = dict([(k, []) for k in attack_phase])

def dict_to_txt(phase_dict, txt_file):
    '''transform dictionary to .txt file'''

    with open (txt_file, 'a+', encoding = 'utf-8') as f:    # 'a+': append and write; if file not exist, then create
        for phase in phase_dict.values():
            if not phase:
                pass
            else:
                line = "\t".join(phase)     # events in an attack phase are seperated by '\t'
                f.write(line)
            
            f.write("\n")   # attack phases in an activity are seperated by '\n'

def event_to_phaseDict(df):
    '''aggregate event from excel into attack phases'''

    phase_dict = dict([(k, []) for k in attack_phase])

    for indexs in df.index:
        alert = df.loc[indexs].values[:-2]     # attack_phase field is discarded

        line = ",".join([str(al) for al in alert])  # fields in an event are seperated by ','

        key = df.loc[indexs].values[-1]
        if pd.isnull(key):
            continue
        else:
            phase_dict[key].append(line)

    return phase_dict


def event_extraction(sample_path, csv_path, txt_path):
    '''alert去重. 获得attack event'''

    df = pd.read_csv(sample_path, low_memory=False)
    # df = pd.read_excel(sample_path, engine='openpyxl')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.hour   # 只保留小时

    newdf = df.drop_duplicates(subset = ['timestamp', 'sig_id', 'ip_src', 'src_port', 'ip_dst', 'dst_port'])    # 去重
    tmp = newdf.drop(columns = df.columns[[0, 1, 2, 3, 7, 8, 9, 10, 11, 14]])
    
    phase_dict = event_to_phaseDict(tmp)

    basename = os.path.basename(sample_path)    # filename.xlsx
    (filename, suffix) = os.path.splitext(basename)

    tmp.to_csv(csv_path + filename + ".csv", index=False, header=True)
    dict_to_txt(phase_dict, txt_path + filename + '.txt')
    

if __name__ == "__main__":

    classes = ['ddos1', 'ddos2', 'infiltration', 'httpdos', 'brute_force', 'Andariel', 'APT-29', 'AZORult', 'IcedID', 'Raccoon', 'Heartbleed', 'Wannacry']
    
    miss_sample_dir = "test/miss_sample/"
    miss_event_dir = "test/miss_event_csv/"
    miss_txt_dir = "test/miss_txt/"

    for dir in os.listdir(miss_sample_dir):
        print("Processing: {}".format(dir))
        sample_dir = miss_sample_dir + dir + "/"   # miss_excel/Andariel/
        csv_path = miss_event_dir + dir + "/"   # miss_event_csv/Andariel/
        txt_path = miss_txt_dir + dir + "/"  # miss_event_csv/Andariel/
        
        for sample in tqdm(os.listdir(sample_dir)):
            sample_path = sample_dir + sample   # miss_excel/Andariel/6_Andariel-2019-1_1_0.05.csv
            event_extraction(sample_path, csv_path, txt_path)


    # '''原始样本'''
    # sample_dir = "/home/kk/attack_activity_detect/data/activity/"
    # csv_path = "/home/kk/attack_activity_detect/test/"
    # txt_path = "/home/kk/attack_activity_detect/test/"
    # for sample in tqdm(os.listdir(sample_dir)):
    #     sample_path = sample_dir + sample   # miss_excel/Andariel/6_Andariel-2019-1_1_0.05.csv
    #     event_extraction(sample_path, csv_path, txt_path)

    # '''write to corpus.txt'''
    # corpus_path = "/home/kk/attack_activity_detect/data/corpus.txt"
    # excel_dir = "/home/kk/attack_activity_detect/test/"
    # for filename in os.listdir(excel_dir):
    #     file_path = excel_dir + filename
    #     if os.path.isdir(file_path):
    #         pass
    #     elif os.path.splitext(filename)[-1] == ".csv":
    #         print("Processing file: ", filename)
    #         df = pd.read_csv(file_path)
    #         event_to_phaseDict(df)
    #     else:
    #         pass

    # dict_to_txt(phase_dict, corpus_path)
