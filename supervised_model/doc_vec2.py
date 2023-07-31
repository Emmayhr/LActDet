# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   doc2vec.py
# @Time     :   2021/12/27


'''
利用doc2vec将处理后的结果转换为向量
doc2vec训练好, 能够更好地提取语义信息

输出: 攻击事件向量
'''

import os
from tqdm import *
import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


attack_phase = ['Reconnaissance', 'Resource_Development', 'Initial_Access', 'Execution', 'Persistence', 'Privilege_Escalation', 
                'Defense_Evasion', 'Credential_Access', 'Discovery', 'Lateral_Movement', 'Collection', 'Command_and_Control', 'Exfiltration', 'Impact']

vector_size = 512

def iter_count(file_name):
    '''Count the number of file lines'''

    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def read_corpus(fname, tokens_only = False):
    '''read document'''

    with open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.isspace():  # the ith attack phase has no events
                continue
            else:
                # line = line.replace("\n", "")
                line = line.replace("\tnan", "")
                line = line.replace("\t0", "")
                line = line.replace("\tfailed", "")
                
                tokens = line.strip().split("\t")
                tokens = tokens[0].strip().split()
                #print(tokens)
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield TaggedDocument(words = tokens, tags = [str(i)])


def doc2vec_model(train_file):
    '''train doc2vec model'''

    train_corpus = list(read_corpus(train_file))
    # print(train_corpus)
    model = Doc2Vec(vector_size = vector_size, epochs = 5, min_count = 1)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples = model.corpus_count, epochs = model.epochs)
    model.save("checkout/doc2vec.model")


if __name__ == "__main__":
    print("-- train doc2vec model ......")

    train_file = 'data/WCMC_corpus_msg.txt'
    doc2vec_model(train_file)

    dv_model = Doc2Vec.load('checkout/doc2vec.model')

    txt_dir = "test/miss_txt/"   # txt

    np_dir = "test/miss_doc2vec/"   # doc2vec result saved path

    print("-- sequence embedding ......")
    for dirname in os.listdir(txt_dir):
        print("-- obtain attack activity {} vector metrix".format(dirname))

        activity_path = txt_dir + dirname
        np_path = np_dir + dirname

        files = os.listdir(activity_path)   # read folder

        for filename in tqdm(files):
            label = int(filename.split("_")[0]) - 1

            file_path = activity_path + "/" + filename
            lines = iter_count(file_path)
            test_corpus = list(read_corpus(file_path, tokens_only=True))

            # activity_vector = np.zeros((14, 512))   # activity metrix, 14 * 512 (attack_phase * attack_phase_vec_dimension)
            activity_vector = np.zeros((lines, vector_size))
            activity_vector = activity_vector.astype('float32')

            with open (file_path, 'r') as f:
                j = 0
                for i, line in enumerate(f):
                    if line.isspace():
                        continue
                    else:
                        vec = dv_model.infer_vector(test_corpus[j])
                        j = j + 1
                        activity_vector[i] = vec
            
            # save numpy array
            filename = os.path.splitext(filename)[0]
            percent = filename.split("_")[-1]
            percent_path = np_path + "/" + percent

            if not os.path.exists(percent_path):
                os.mkdir(percent_path)
            else:
                pass

            vector_path = percent_path + "/" + filename
            
            np.save(vector_path, activity_vector)   # .npy


    '''single sample test'''
    # file_path = "test/txt/3_infiltration.txt"
    # np_path = "test/np/3_infiltration"
    # label = 3

    # lines = iter_count(file_path)
    # test_corpus = list(read_corpus(file_path, tokens_only=True))
    # activity_vector = np.zeros((lines, 256))
    # activity_vector = activity_vector.astype('float32')

    # with open (file_path, 'r') as f:
    #     j = 0
    #     for i, line in enumerate(f):
    #         if line.isspace():
    #             continue
    #         else:
    #             vec = dv_model.infer_vector(test_corpus[j])
    #             j = j + 1
    #             activity_vector[i] = vec
    
    # # save numpy array    
    # np.save(np_path, activity_vector)   # .npy
