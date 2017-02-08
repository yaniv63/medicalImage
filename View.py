# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:20:19 2016

@author: yaniv
"""

from mycontainer import Mycontainer
from numpy import ndarray

class View(Mycontainer):
    def __init__(self):
        super(View,self).__init__(type(ndarray(1)))
        
        
class Axial(View):
    def __init__(self):
        super(Axial,self).__init__()
        
class Coronal(View):
    def __init__(self):
        super(Axial,self).__init__()
        
class Sagittal(View):
    def __init__(self):
        super(Axial,self).__init__()
