# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 17:03:19 2016

@author: yaniv
"""
from mycontainer import Mycontainer
from numpy import ndarray

class ImageModality(Mycontainer):
    def __init__(self):
        super(ImageModality,self).__init__(type(ndarray(1)))
    
    def add_view(self,view,patch):
        self[view] = patch
    
class T2(ImageModality):
    def __init__(self):
        super(T2,self).__init__()

class FLAIR(ImageModality):
    def __init__(self):
        super(FLAIR,self).__init__()

class T1(ImageModality):
    def __init__(self):
        super(T1,self).__init__()

