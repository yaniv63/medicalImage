# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:16:52 2017

@author: yaniv
"""

class Mycontainer(object):
    def __init__(self,store_type):
        self.container = {}
        self.store_type = str(store_type)

    def get(self,key):
        if key in self.container:
            return self.container[key]
        else:
            raise  Exception("no such {} for voxel".format(self.store_type))

    def get_all(self):
        return self.container
    
    def add(self,key,data):
        if key in self.container:
             raise Exception("already have {} for voxel".format(self.store_type))
        else:
            self.container[key] = data
            
