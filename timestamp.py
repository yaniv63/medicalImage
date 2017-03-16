# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:01:06 2017

@author: yaniv
"""

from mycontainer import Mycontainer
from ImageModality import ImageModality

class TimeStamp(Mycontainer):
    def __init__(self,index=None,labels = None):
        super(TimeStamp,self).__init__(type(ImageModality()))
        if index != None:
            self.type = self.type + str(index)
        self.labels = labels
    
    def add(self,modality,view,patch):
        super(TimeStamp,self).add(key=modality,data=ImageModality())
        self[modality].add_view(view,patch)

    
    
