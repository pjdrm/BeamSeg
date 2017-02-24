'''
Created on Feb 23, 2017

@author: root
'''
import logging

logger_id = 0

def log_init(sampler_log_file):
    global logger_id
    my_logger = logging.getLogger("Logger"+str(logger_id))
    logger_id += 1
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    fileHandler = logging.FileHandler(sampler_log_file, mode='w')
    fileHandler.setFormatter(formatter)
    my_logger.setLevel(logging.INFO)
    my_logger.addHandler(fileHandler)
    return my_logger