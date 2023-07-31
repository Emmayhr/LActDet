# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   run_fasttext.py
# @Time     :   2022/04/11

'''
word to vector
'''

import os
import pandas as pd
from tqdm import tqdm

def get_corpus(dir_path):
    corpus = []
    files = os.listdir(dir_path)
    for filename in files:
        print("process file: {}".format(filename))
        file_path = dir_path + filename
        df = pd.read_csv(file_path)
        l = (len(df))
        for i in range(len(df)):
            corpus.append(df.iloc[i].at['msg'])
            corpus.append(df.iloc[i].at['category'])
            if len(str(df.iloc[i].at['app_proto'])) != 0:
                corpus.append(df.iloc[i].at['app_proto'])
            if len(str(df.iloc[i].at['smtp_helo'])) != 0:
                corpus.append(df.iloc[i].at['smtp_helo'])
            if len(str(df.iloc[i].at['http_host_name'])) != 0:
                corpus.append(df.iloc[i].at['http_host_name'])
            if len(str(df.iloc[i].at['http_url'])) != 0:
                corpus.append(df.iloc[i].at['http_url'])
            if len(str(df.iloc[i].at['http_user_agent'])) != 0:
                corpus.append(df.iloc[i].at['http_user_agent'])
            if len(str(df.iloc[i].at['http_content_type'])) != 0:
                corpus.append(df.iloc[i].at['http_content_type'])
            if len(str(df.iloc[i].at['http_refer'])) != 0:
                corpus.append(df.iloc[i].at['http_refer'])
            if len(str(df.iloc[i].at['http_method'])) != 0:
                corpus.append(df.iloc[i].at['http_method'])
            if len(str(df.iloc[i].at['http_proto'])) != 0:
                corpus.append(df.iloc[i].at['http_proto'])
            if len(str(df.iloc[i].at['dns_type'])) != 0:
                corpus.append(df.iloc[i].at['dns_type'])
            if len(str(df.iloc[i].at['dns_rrname'])) != 0:
                corpus.append(df.iloc[i].at['dns_rrname'])
            if len(str(df.iloc[i].at['dns_rrtype'])) != 0:
                corpus.append(df.iloc[i].at['dns_rrtype'])
            if len(str(df.iloc[i].at['ftp_data_filename'])) != 0:
                corpus.append(df.iloc[i].at['ftp_data_filename'])
            if len(str(df.iloc[i].at['ftp_data_command'])) != 0:
                corpus.append(df.iloc[i].at['ftp_data_command'])
            if len(str(df.iloc[i].at['smb_dialect'])) != 0:
                corpus.append(df.iloc[i].at['smb_dialect'])
            if len(str(df.iloc[i].at['smb_command'])) != 0:
                corpus.append(df.iloc[i].at['smb_command'])
            if len(str(df.iloc[i].at['smb_status'])) != 0:
                corpus.append(df.iloc[i].at['smb_status'])
            if len(str(df.iloc[i].at['smb_status_code'])) != 0:
                corpus.append(df.iloc[i].at['smb_status_code'])
            if len(str(df.iloc[i].at['smb_client_dialects'])) != 0:
                corpus.append(df.iloc[i].at['smb_client_dialects'])
            if len(str(df.iloc[i].at['smb_server_guid'])) != 0:
                corpus.append(df.iloc[i].at['smb_server_guid'])
            if len(str(df.iloc[i].at['tls_subject'])) != 0:
                corpus.append(df.iloc[i].at['tls_subject'])
            if len(str(df.iloc[i].at['tls_issuer'])) != 0:
                corpus.append(df.iloc[i].at['tls_issuer'])
            if len(str(df.iloc[i].at['tls_serial'])) != 0:
                corpus.append(df.iloc[i].at['tls_serial'])
            if len(str(df.iloc[i].at['tls_fingerprint'])) != 0:
                corpus.append(df.iloc[i].at['tls_fingerprint'])
            if len(str(df.iloc[i].at['tls_ja3_hash'])) != 0:
                corpus.append(df.iloc[i].at['tls_ja3_hash'])
            if len(str(df.iloc[i].at['tls_ja3_string'])) != 0:
                corpus.append(df.iloc[i].at['tls_ja3_string'])
            if len(str(df.iloc[i].at['tls_ja3s_hash'])) != 0:
                corpus.append(df.iloc[i].at['tls_ja3s_hash'])
            if len(str(df.iloc[i].at['tls_ja3s_string'])) != 0:
                corpus.append(df.iloc[i].at['tls_ja3s_string'])

    with open("word_corpus.txt", "a+", encoding = 'utf-8') as f:
        line = " ".join([str(al) for al in corpus])
        f.write(line)


from gensim.models.fasttext import FastText
from gensim.test.utils import datapath

def fasttext_model(corpus):
    '''train doc2vec model'''

    model = FastText(vector_size = 80) # n > 8logN, where N is the size of the words in the corpus
 
    # Set file names for train and test data
    corpus_file = datapath(corpus)
  
    # build the vocabulary
    model.build_vocab(corpus_file = corpus_file)
 
    # train the model
    model.train(
        corpus_file = corpus_file, epochs = model.epochs,
        total_examples = model.corpus_count, total_words = model.corpus_total_words
    )

    model.save("fasttext.model")


import numpy as np
def row2vec(df_iloc, modelVec):
    rowVec = np.zeros(0)

    for j in range(len(df_iloc) - 1):   # attack phase不需要embed
        if j in [2,3,4,5,6,7,8,18,19,21,26,31,32]:
            if pd.isnull(df_iloc[j]):
                rowVec = np.concatenate((rowVec, np.zeros(1)))
            else:
                rowVec = np.concatenate((rowVec, [int(df_iloc[j])])) # 数字, 拼接
        else:
            if pd.isnull(df_iloc[j]):
                rowVec = np.concatenate((rowVec, np.zeros(80)))
            else:
                words = df_iloc[j].split()
                vec = np.zeros(80)
                for word in words:  # 字符串
                    vec = vec + modelVec[word]  # 一个字符串中所有单词向量相加
                rowVec = np.concatenate((rowVec, vec))

    return rowVec


if __name__ == "__main__":
    # dir_path = "/home/kk/attack_activity_detect/test/full_sample/"
    # get_corpus(dir_path)

    # fasttext_model("/home/kk/attack_activity_detect/data/word_corpus.txt")
    load_model = FastText.load("fasttext.model")
    wv = load_model.wv
    # print(wv['CN=dradis'])

    csv_dir = "/home/kk/attack_activity_detect/test/miss_event_csv/"   # txt
    np_dir = "/home/kk/attack_activity_detect/test/miss_word/"   # fasttext result saved path
    print("-- word embedding ......")

    attack_phase = ['Reconnaissance', 'Resource_Development', 'Initial_Access', 'Execution', 'Persistence', 'Privilege_Escalation', 
                'Defense_Evasion', 'Credential_Access', 'Discovery', 'Lateral_Movement', 'Collection', 'Command_and_Control', 'Exfiltration', 'Impact']
    phase_dict = dict()
    for i in range(14):
        phase_dict[attack_phase[i]] = i

    for dirname in os.listdir(csv_dir):
        print("-- obtain attack activity {} vector metrix".format(dirname))

        activity_path = csv_dir + dirname   # false_event_csv/Andariel
        np_path = np_dir + dirname  # # false_word2vec/Andariel

        files = os.listdir(activity_path)   # read folder

        for filename in tqdm(files):

            label = int(filename.split("_")[0]) - 1

            file_path = activity_path + "/" + filename  # false_event_csv/Andariel/6_Andariel-2019-1_1_0.05.csv

            '''分攻击阶段'''
            # activity_vector = np.zeros((14, 2413))   # activity metrix
            # activity_vector = activity_vector.astype('float32')

            # df = pd.read_csv(file_path)
            # for i in range(len(df)):
            #     phase = df.loc[i].values[-1]
            #     vec = row2vec(df.iloc[i], wv)
            #     activity_vector[phase_dict[phase]] += vec

            '''不分攻击阶段'''
            activity_vector = np.zeros(2413)   # activity metrix
            activity_vector = activity_vector.astype('float32')

            df = pd.read_csv(file_path)
            for i in range(len(df)):
                # phase = df.loc[i].values[-1]
                vec = row2vec(df.iloc[i], wv)
                activity_vector += vec
            
            
            # save numpy array
            filename = os.path.splitext(filename)[0]
            percent = filename.split("_")[-1]
            percent_path = np_path + "/" + percent

            if not os.path.exists(percent_path):
                os.mkdir(percent_path)
            else:
                pass

            vector_path = percent_path + "/" + filename
            
            np.save(vector_path, activity_vector)