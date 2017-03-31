# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Sun Mar 26 12:50:14 2017

@author: priyanjit
"""

import cv2
import numpy as np

vidcap = cv2.VideoCapture('sample.mp4')

success,image = vidcap.read()

count = 0
success = True

while success:
  success,image = vidcap.read()
  
  print 'Read a new frame: ', success
  cv2.imwrite("frame%d.jpg" % count, image)     
  
  count += 1