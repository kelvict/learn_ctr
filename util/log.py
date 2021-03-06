#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import os
import sys
import time
import json
import logging

log_format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

def config_log(log_path_prefix):
    time_str = time.strftime("%Y%m%d_%H%M%S")
    print "Log to %s"%("%s.%s.log"%(log_path_prefix, time_str))

    logging.basicConfig(level=logging.DEBUG,
                format= log_format,
                datefmt=date_format,
                filename="%s.%s.log"%(log_path_prefix, time_str),
                filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(log_format)
    logger = logging.getLogger(log_path_prefix)
    logger.addHandler(console)
    log("pid: %s"%str(os.getpid()))
    return time_str

def log(msg):
    logging.info(str(msg))

def log_and_print(msg):
    logging.info(str(msg))
    time_str = time.strftime("%Y%m%d_%H%M%S")
    print "[%s] %s"%(time_str, msg)
    sys.stdout.flush()

def print_with_time(msg):
    time_str = time.strftime("%Y%m%d_%H%M%S")
    print "[%s] %s"%(time_str, msg)

def now_str():
    return time.strftime("%Y%m%d_%H%M%S")

def pretty_print_json_obj(obj):
    logging.info(json.dumps(obj, indent=4))