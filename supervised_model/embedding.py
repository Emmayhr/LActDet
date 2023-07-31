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

def check_redundancy(event1, event2, simDict):
    '''
    input of this function are two attack event, output is whether they are redundancy event
    True means they are redundancy event
    the criteria has been writen into the csv file: similarity.csv
    '''
    event1 = event1.strip()
    event2 = event2.strip()
    if (event1 > event2):
        event1, event2 = event2, event1
    key = (event1, event2)
    #print(key)
    if key in simDict:
        similarity = simDict[key]
    else:
        a = event1.split()
        b = event2.split()
        similarity = dv_model.similarity_unseen_docs(a, b)
        log.logger.debug('\npreRow: {}\nnowRow: {}\nthere is none similarity record in database\n\n'.format(event1,event2))
        log.logger.debug('similarity is {}'.format(similarity))
    if similarity > threshold:
        return True
    else:
        return False


def obtain_activity_vector(phase_dict, filename, dv_model, np_dir, log, simDict):
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
        else:
            yield_event = list(read_line(phase))

            preRow = []
            j = 0   # attack event
            while j < len(phase):
                nowRow = yield_event[j]

                if (j == 0):
                    activity_vector[i] = dv_model.infer_vector(nowRow)
                    j += 1
                    preRow = nowRow
                else:
                    a = preRow[0].strip().split()
                    b = nowRow[0].strip().split()
                    similarity = check_redundancy(preRow[0], nowRow[0], simDict)
                    #log.logger.debug('\npreRow: {}\nnowRow: {}\nsimilarity: {}\n\n'.format(preRow[0], nowRow[0], similarity))
                    
                    if (similarity):   # deduplicate, keep the previous one
                        j += 1
                    else:
                        '''encoding: Multiple attack events within an attack phase are encoded as a vector'''
                        activity_vector[i] += dv_model.infer_vector(nowRow) # 同一个攻击阶段内的event vector相加
                        j += 1
                        preRow = nowRow
        i += 1

    # save numpy array
    percent = filename.split("_")[-1]
    percent_path = np_dir + percent

    if not os.path.exists(percent_path):
        os.mkdir(percent_path)
    else:
        pass

    vector_path = percent_path + '/' + filename
    
    np.save(vector_path, activity_vector)   # .npy


def event_to_phaseDict(df):
    '''aggregate event from excel into attack phases'''

    phase_dict = dict([(k, []) for k in attack_phase])

    for indexs in df.index:
        # alert = df.loc[indexs].values[:-1]     # attack_phase field is discarded
        alert = df.loc[indexs]

        line = "\t".join([str(al) for al in alert])  # fields in an event are seperated by '\t'

        key = df.loc[indexs].values[-1]
        if pd.isnull(key):
            continue
        else:
            phase_dict[key].append(line)

    return phase_dict


def read_similarity(sim_file):
    similarity_dict = {}
    df = pd.read_csv(sim_file)
    for i in range(len(df['preRow'])):
        preRow = df['preRow'][i]
        nowRow = df['nowRow'][i]
        similarity_value = df['similarity'][i]
        key = (preRow , nowRow)
        similarity_dict[key] = similarity_value
    return similarity_dict


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
    tmp = newdf.drop(columns = df.columns[[0, 1, 2, 3, 8, 9, 10, 11, 14]])

    # WCMC
    simDict = read_similarity('data/similarity.csv')

    phase_dict = event_to_phaseDict(tmp)

    basename = os.path.basename(sample_path)    # filename.xlsx
    (filename, suffix) = os.path.splitext(basename)

    obtain_activity_vector(phase_dict, filename, dv_model, np_path, log, simDict)


if __name__ == "__main__":
    dv_model = Doc2Vec.load('checkout/doc2vec_msg.model')

    miss_sample_dir = "test/miss_sample/"
    miss_doc2vec_dir = "test/miss_doc2vec/"

    log_name = "log/embedding.log"
    log = ApiLog(log_name)
    log.logger.debug("------- Attack Activity Embedding -------")

    similarity_path = "data/similarity.csv"
    similarity_dict = read_similarity(similarity_path)
    

    
    for dir in os.listdir(miss_sample_dir):
        print("Processing: {}".format(dir))
        sample_dir = miss_sample_dir + dir + "/"   # miss_sample/infiltration/
        np_path = miss_doc2vec_dir + dir + "/"   # miss_doc2vec/infiltration/
        
        # for sample in tqdm(os.listdir(sample_dir)):
        for sample in os.listdir(sample_dir):
            sample_path = sample_dir + sample   # miss_sample/infiltration/3_infiltration_1_0.05.csv
            event_extraction(sample_path, dv_model, np_path, log)
    
    
