# -*- coding: utf-8 -*-
# __author__:bin_ze
# 9/29/22 4:01 PM
import time

def timer(fun):

    def wapper(*args, **kwargs):
        start_time = time.time()
        fun(*args, **kwargs)
        print(f'run time {time.time() - start_time} s')

    return wapper