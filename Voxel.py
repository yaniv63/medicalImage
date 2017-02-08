# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:51:18 2017

@author: yaniv
"""    
from mycontainer import Mycontainer
from timestamp import TimeStamp

class Voxel(Mycontainer):
   
    def __init__(self,person):
        self.person = person
        super(Voxel,self).__init__(type(TimeStamp()))
        self.type = self.type + "_" + str(person)
            
    def add(self,time,modality,view,patch):
        super(Voxel,self).add(key=time,data=TimeStamp(time))
        self[time].add(modality,view,patch)
        
    
