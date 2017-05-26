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

	def __init__(self,url,vFormat):
		#super(RecordVideo, self).__init__(0,"mp4");
		self.driveLocation=url;
		self.preferedFormat=vFormat;

	def setDriveLocation(self,url):
		self.driveLocation=url;

	def getDriveLocation(self):
		return self.driveLocation;

	def setPreferedFormat(self,vFormat):
		self.preferedFormat=vFormat;

	def getPreferedFormat(self):
		return self.preferedFormat;

	def startRecording(self):
		CameraClass.startCamera(0);


	
#Input path and video type
print "Enter the location where you want to store the video:";
path=raw_input('---->');
print "Enter the format of video:";
vFormat=raw_input('---->');

#Create an object.
RV=RecordVideo(path,vFormat);

RV.startRecording();
