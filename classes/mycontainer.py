# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:16:52 2017

@author: yaniv
"""

class Mycontainer(dict):
    def __init__(self,store_type):
        self.container = {}
        self.store_type = store_type
        self.type = get_type(self)
        
    def get(self,key):
        if key in self.container:
            return self.container[key]
        else:
            raise  Exception("no such {} for voxel".format(str(self.store_type)))

    def get_all(self):
        return self.container.items()
    
    def add(self,data,key = None):
        if key == None:
            key = get_type(data)
        elif isinstance(data,self.store_type):
            self.container[key] = data
        else:
            raise Exception("expected type {}".format(str(self.store_type)))
    
    def __setitem__(self,index,val):
        self.add(val,index)       
    def __getitem__(self,index):
        return self.get(index)
         
    def __delitem__(self,index):
        return self.container.__delitem__(index)
    
    def __hash__(self):
        return hash(repr(self))
        
def get_type(obj):
        return  type(obj).__name__
        