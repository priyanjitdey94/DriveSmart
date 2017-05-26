#Class responsible recording the video and storing it in a specified location.

import cv2;
import numpy as np;
from videocamera import CameraClass;

'''
# setDriveLocation: set the path where video will be stored.
# getDriveLocation: get the path where video will be stored.
# setpreferedFormat: format in which video is to recorded.
# setpreferedFormat: format in which video is recorded.
# starRecording: initiates the camera and starts recording.
'''

class RecordVideo(CameraClass):




	
#Input path and video type
print "Enter the location where you want to store the video:";
path=raw_input('---->');
print "Enter the format of video:";
vFormat=raw_input('---->');

#Create an object.
RV=RecordVideo(path,vFormat);

RV.startRecording();
