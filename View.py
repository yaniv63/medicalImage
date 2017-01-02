# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:20:19 2016

@author: yaniv
"""

class TimeStampError(Exception):
    pass
        


class View(object):
    
    def __init__(self):
        self.patches = {}
    
    def get(self,time_stamp=1):
        if time_stamp in self.patches:
            return self.patches[time_stamp]
        else:
            raise TimeStampError("no such time stamp for voxel")
    
    def get_all(self):
        return self.patches
    
    def add(self,time_stamp=1,patch):
        if time_stamp in self.patches:
             raise TimeStampError("already have time stamp for voxel")
        else:
            self.patches[time_stamp] = patch

class Axial(View):
    def __init__(self):
        super(Axial,self).__init__()
        
class Coronal(View):
    def __init__(self):
        super(Axial,self).__init__()
        
class Sagittal(View):
    def __init__(self):
        super(Axial,self).__init__()
