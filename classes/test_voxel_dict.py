# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:17:34 2017

@author: yaniv
"""

from Voxel import Voxel
import numpy as np
v = Voxel("person2")
patch = np.random.rand(2)
time = "01"
view = "axial"
mod = "t2"
v.add(time,mod,view,patch)

patch = np.random.rand(2)
time = "02"
view = "coronal"
mod = "t2"
v.add(time,mod,view,patch)
v.get_all()