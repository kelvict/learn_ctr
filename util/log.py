#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import os
import sys
import time

def log(msg):
    print str(os.getpid())+'-[' + time.asctime( time.localtime(time.time()) ) + "] " + msg
    sys.stdout.flush()