#!/usr/bin/python3
# -*- coding:utf-8 -*-
import logging

class ApiLog(object):
    def __init__(self, file_name='/home/kk/code_log/code.log'):
        self.logger = logging.getLogger('api_log')
        self.logger.setLevel(logging.DEBUG)
        self.file = logging.FileHandler(file_name, mode='a', encoding=None, delay=False)
        formatter = logging.Formatter('%(asctime)s %(name)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        self.file.setFormatter(formatter)
        self.logger.addHandler(self.file)
    def close_log_file(self):
        self.logger.removeHandler(self.file)
        self.file.close()
        
