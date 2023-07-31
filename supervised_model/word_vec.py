# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   word_vec.py
# @Time     :   2022/03/11


'''
利用word2vec将处理后的结果转换为向量

'''

import os
from tqdm import *
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


attack_phase = ['Reconnaissance', 'Resource_Development', 'Initial_Access', 'Execution', 'Persistence', 'Privilege_Escalation', 
                'Defense_Evasion', 'Credential_Access', 'Discovery', 'Lateral_Movement', 'Collection', 'Command_and_Control', 'Exfiltration', 'Impact']


def read_corpus(fname, tokens_only = False):
    '''read document'''

    with open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.isspace():  # the ith attack phase has no events
                continue
            else:
                line = line.replace("\n", "")
                line = line.replace(",nan", "")
                line = line.replace(",0", "")
                line = line.replace(",failed", "")
                
                tokens = line.strip().split("\t")
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield TaggedDocument(words = tokens, tags = [attack_phase[i]])


def doc2vec_model(train_file):
    '''train doc2vec model'''

    train_corpus = list(read_corpus(train_file))
    # print(train_corpus)
    model = Doc2Vec(vector_size = 512, epochs = 10, min_count = 1)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples = model.corpus_count, epochs = model.epochs)
    model.save("doc2vec.model")


if __name__ == "__main__":
    print("-- train doc2vec model ......")

    train_file = '/home/kk/attack_activity_detect/data/corpus.txt'
    doc2vec_model(train_file)

    dv_model = Doc2Vec.load('doc2vec.model')

    txt_dir = "/home/kk/attack_activity_detect/data/txt/"   # txt

    np_dir = "/home/kk/attack_activity_detect/data/np512/"   # doc2vec result saved path

    print("-- sequence embedding ......")
    for dirname in os.listdir(txt_dir):
        print("-- obtain attack activity {} vector metrix".format(dirname))

        activity_path = txt_dir + dirname
        np_path = np_dir + dirname

        files = os.listdir(activity_path)   # read folder

        for filename in tqdm(files):
            label = int(filename.split("_")[0]) - 1

            file_path = activity_path + "/" + filename
            test_corpus = list(read_corpus(file_path, tokens_only=True))

            activity_vector = np.zeros((14, 512))   # activity metrix, 14 * 20 (attack_phase * attack_phase_vec_dimension)
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
            
            np.save(vector_path, activity_vector)