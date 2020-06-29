#!/usr/bin/env python3

from time import time


def timing(f):
    '''Funcion decorador utilizada para medir el tiempo que lleva ejecutar una funcion                                       
    '''
    def wrap(*args):
        time1 = time()
        ret = f(*args)
        time2 = time()
        print('Tiempo de computo: %0.5f s' % ((time2-time1)*1))
        return ret
    return wrap
