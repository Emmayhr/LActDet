# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   deduplicate.py
# @Time     :   2022/10/15

'''
根据向量距离对攻击事件进行去重
输出: np.save() np.npy
'''

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Doc2Vec
# import torch
# import torch.nn as nn

import sys
sys.path.append("json2mysql/")
from log import ApiLog

attack_phase = ['Reconnaissance', 'Resource_Development', 'Initial_Access', 'Execution', 'Persistence', 'Privilege_Escalation', 
                'Defense_Evasion', 'Credential_Access', 'Discovery', 'Lateral_Movement', 'Collection', 'Command_and_Control', 'Exfiltration', 'Impact']

vector_size = 64
threshold = 0.83

def read_line(events):
    if len(events) == 0:
        return
    for event in events:
        if event.isspace():
            continue
        else:
            event = event.replace("\tnan", "")
            event = event.replace("\t0", "")
            event = event.replace("\tfailed", "")
            
            tokens = event.strip().split("\t")[:-1]
            yield tokens


def obtain_event_vector(event_set, filename, dv_model, log):
    '''
    deduplicate and encoding
    filename: 3_infiltration_1_0.05.csv
    np_dir: test/miss_doc2vec/infiltration/
    attack activity vector: 14*256
    '''
    log.logger.debug('Embedding file: {}'.format(filename))

    event_vector_set = np.empty((vector_size))
    event_vector_set = event_vector_set.astype('float32')
    event_zero = np.zeros((vector_size))
    event_zero = event_zero.astype('float')
    yield_event = list(read_line(event_set))
    j = 0   # attack event
    while j < 50:
        if j < len(event_set): 
            nowRow = yield_event[j]
            event_vector = dv_model.infer_vector(nowRow)
            event_vector_set = np.vstack([event_vector_set, event_vector])
        else:
            event_vector_set = np.vstack([event_vector_set, event_zero])
        j += 1
    return event_vector_set[1:]

 
def obtain_activity_vector(phase_dict, filename, dv_model, log):
    '''
    deduplicate and encoding
    filename: 3_infiltration_1_0.05.csv
    np_dir: test/miss_doc2vec/infiltration/
    attack activity vector: 14*256
    '''
    log.logger.debug('Embedding file: {}'.format(filename))

    activity_vector = np.zeros((14, vector_size))
    activity_vector = activity_vector.astype('float32')
    
    i = 0   # attack phase
    for phase in phase_dict.values():
        if not phase:
            pass
        elif len(phase)==0:
            pass
        else:
            yield_event = list(read_line(phase))
            preRow = []
            j = 0   # attack event
            while j < len(phase): 
                nowRow = yield_event[j]
                activity_vector[i] += dv_model.infer_vector(nowRow)
                j += 1
        if len(phase)>0:
            activity_vector[i] = activity_vector[i] / len(phase)
        i += 1
    return activity_vector
    
    '''
    percent = filename.split("_")[-1]
    percent_path = np_dir + percent

    if not os.path.exists(percent_path):
        os.mkdir(percent_path)
    else:
        pass

    vector_path = percent_path + '/' + filename
    
    np.save(vector_path, activity_vector)   # .npy
    '''

def event_to_phaseDict(df):
    '''aggregate event from excel into attack phases'''

    phase_dict = dict([(k, []) for k in attack_phase])
    event_vector_set = []

    for indexs in df.index:
        # alert = df.loc[indexs].values[:-1]     # attack_phase field is discarded
        alert = df.loc[indexs]

        line = "\t".join([str(al) for al in alert])  # fields in an event are seperated by '\t'
        
        event_vector_set.append(line)

        key = df.loc[indexs].values[-1]
        if pd.isnull(key):
            continue
        else:
            phase_dict[key].append(line)
    return phase_dict, event_vector_set





def event_extraction(sample_path, dv_model, np_path, log):
    '''
    alert去重, 获得attack event
    输出: 按攻击阶段排序, 每一行表示一个event
    '''

    df = pd.read_csv(sample_path, low_memory=False)
    #df = pd.read_excel(sample_path, engine='openpyxl')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.hour   # 只保留小时

    newdf = df.drop_duplicates(subset = ['timestamp', 'sig_id', 'ip_src', 'src_port', 'ip_dst', 'dst_port'])    # 去重
    #tmp = newdf.drop(columns = df.columns[[0, 1, 2, 3, 8, 9, 10, 11, 14]])
    tmp = newdf[newdf.columns[4:5]]# .drop(columns = df.columns[[0, 1, 2, 3, 8, 9, 10, 11, 14]])

    basename = os.path.basename(sample_path)    # filename.xlsx
    (filename, suffix) = os.path.splitext(basename)
    phase_dict, event_vector_set = event_to_phaseDict(df)
    phase_vector_sequence = obtain_activity_vector(phase_dict, filename, dv_model, log)
    event_vector_sequence = obtain_event_vector(event_vector_set, filename, dv_model, log)
    return phase_vector_sequence, event_vector_sequence
    

if __name__ == "__main__":
    dv_model = Doc2Vec.load('checkout/doc2vec_msg64.model')

    miss_sample_dir = "test/miss_sample/test/" #"test/miss_sample_Andariel/"
    miss_s2s_dir = "test/miss_s2s_msg64/"

    log_name = "log/embedding.log"
    log = ApiLog(log_name)
    log.logger.debug("------- Attack Activity Embedding -------")


    
    for dir in os.listdir(miss_sample_dir):
        phases = []
        events = []
        labels = []
        print("Processing: {}".format(dir))
        sample_dir = miss_sample_dir + dir + "/"   # miss_sample/infiltration/
        vector_dir = miss_s2s_dir + dir + "/"
        if not os.path.exists(vector_dir):
            os.mkdir(vector_dir)
        ratio_dict = {'0.05':0, '0.1':0, '0.15':0, '0.2':0, '0.25':0, '0.3':0, '0.35':0, '0.4':0, '0.45':0, '0.5':0, '1':0}
        i = 0        
        # for sample in tqdm(os.listdir(sample_dir)):
        for sample in os.listdir(sample_dir):
            print(sample)
            label = int(sample.split("_")[0]) - 1
            ratio = sample.split("_")[-1][:-4]
            ratio_dict[ratio] += 1
            if ratio_dict[ratio] > 200:
                continue
            sample_path = sample_dir + sample   # miss_sample/infiltration/3_infiltration_1_0.05.csv
            phase_vector_sequence, event_vector_sequence = event_extraction(sample_path, dv_model, miss_s2s_dir, log)
            phases.append(phase_vector_sequence)# = np.vstack([phases, [phase_vector_sequence]])
            events.append(event_vector_sequence)# = np.vstack([events, [event_vector_sequence]])
            labels.append(label)

        rphases = np.array(phases)
        revents = np.array(events)
        rlabels = np.array(labels)
        # restore phases
        print(rphases.shape)
        phase_vector_path = vector_dir + '/' + "phase_vector_sequence"
        np.save(phase_vector_path, rphases)   # .npy
        # restore events
        event_vector_path = vector_dir + '/' + "event_vector_sequence"
        print(revents.shape)
        np.save(event_vector_path, revents)
        # restor labels
        labels_vector_path = vector_dir + '/' + "label_vector_sequence"
        print(rlabels.shape)
        np.save(labels_vector_path, rlabels)


